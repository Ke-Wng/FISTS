import logging
import pickle
import threading
import time

import numpy as np
import pandas as pd

from spn.structure.Base import bfs, get_nodes_by_type, Leaf
from spn.structure.StatisticalTypes import MetaType

from tsspn.utils.utils import parse_query
from tsspn.algorithms.ranges import NominalRange, NumericRange
from tsspn.learning.tsspn_learning import create_custom_leaf
from tsspn.structure.base import Product, TSProduct
from tsspn.structure.discrete_multi_histogram import DiscreteMultiHistogram, create_discrete_multi_histogram_leaf

logger = logging.getLogger(__name__)


class CardinalityEstimator:
    def __init__(self, name, schema, overwrite=False):
        self.name = name
        self.schema = schema
        self.overwrite = overwrite

    def estimate_points(self, query):
        raise NotImplementedError

    def estimate_series(self, query):
        metric_start_idx = query.find(" AND " + self.schema.metric)
        query = query[:metric_start_idx] + ";"
        pred_points, infer_time = self.estimate_points(query)
        return self.schema.n_series * pred_points / self.schema.n_points, infer_time

    def get_model_size():
        raise NotImplementedError

class TSSPNEstimator(CardinalityEstimator):
    def __init__(self, schema, model_path, overwrite=False):
        super().__init__("tsspn", schema, overwrite)
        if not overwrite:
            return
        self.tsspn = pickle.load(open(model_path, "rb"))
        self._lock = threading.Lock()
        self._updating = False
        self.updating_cnt = 0
        self.updating_time = 0
        self.miss_cnt = 0
        self.last_query_miss = False
        self.hot_tags_scope = []

        dataset = pd.read_parquet(schema.parquet_file_path)
        dataset = dataset[schema.tags]
        series_start_indices = np.linspace(0, dataset.shape[0]-1, schema.n_series, dtype=np.int64)
        dataset  = dataset.iloc[series_start_indices]
        for tag in schema.tags:
            dataset[tag] = dataset[tag].map(self.tsspn.table_meta_data[schema.table_name]["categorical_columns_dict"][schema.table_name + "." + tag])
        self.tag_data = dataset.values

    def estimate_cardinality(self, sql):
        sql = sql.replace("*", "COUNT(*)")
        query = parse_query(sql, self.tsspn.schema_graph)
        ranges = self._parse_conditions(query.conditions)

        self.last_query_miss = False
        for i, r in enumerate(ranges[0][:len(self.schema.tags)]):
            if r is not None and i not in self.hot_tags_scope:
                self.miss_cnt += 1
                self.last_query_miss = True

        start_t = time.perf_counter()
        with self._lock:
            points_prob, series_prob = self.tsspn._probability(ranges)
        points_card = max(self.schema.n_points * points_prob, 1)
        series_card = max(self.schema.n_series * series_prob, 1)
        end_t = time.perf_counter()
        return points_card, series_card, end_t - start_t

    def update_hot_tags(self, hot_tags, sync=False, min_hot_tags_num=2, max_hot_tags_num=5):
        if self._updating:
            logging.debug("the last update hasn't completed, ignore hot tags {}".format(hot_tags))
            return

        def background_update():
            start_t = time.perf_counter()
            try:
                # collect all ts-product nodes
                ts_products = get_nodes_by_type(self.tsspn.tsspn, TSProduct)
                for tsnode in ts_products:
                    tag_node = tsnode.get_tag_node()
                    # collect all independent components
                    independent_components = []
                    def collect_independent_components(node):
                        if isinstance(node, Product):
                            for child in node.children:
                                if not isinstance(child, Leaf):
                                    return
                            independent_components.append(node) 
                    bfs(tag_node, collect_independent_components)
                    # re-train hot tags for each independent components
                    for independent_component in independent_components:
                        independent_scope = independent_component.scope
                        if len(independent_scope) <= min_hot_tags_num:
                            # assert isinstance(independent_component, DiscreteMultiHistogram)
                            continue

                        local_hot_scope = [s for s in hot_tags_scope if s in independent_scope]
                        if len(local_hot_scope) <= 1:
                            continue
                        need_rebuild = True
                        for node in independent_component.children:
                            if set(local_hot_scope).issubset(node.scope) and len(node.scope) <= max_hot_tags_num:
                                need_rebuild = False
                                break
                        if need_rebuild:
                            new_children = []
                            local_data = self.tag_data[tsnode.series_ids, :]
                            if len(local_hot_scope) > max_hot_tags_num:
                                local_hot_scope = local_hot_scope[:max_hot_tags_num]
                            start_t = time.perf_counter()
                            node = create_discrete_multi_histogram_leaf(local_data[:, np.array(local_hot_scope)], local_hot_scope)
                            end_t = time.perf_counter()
                            logging.debug(
                                "\t\tcreated multi histogram leaf with scope {} (in {:.5f} secs)".format(local_hot_scope, end_t - start_t)
                            )

                            start_t = time.perf_counter()
                            new_children.append(node)
                            local_cold_scope = [scope for scope in independent_scope if scope not in local_hot_scope]
                            for scope in local_cold_scope:
                                node = create_custom_leaf(local_data[:, scope].reshape(-1, 1), self.tsspn.ds_context, [scope])
                                new_children.append(node)
                            end_t = time.perf_counter()
                            logging.debug(
                                "\t\tcreated histogram leaves for scope {} (in {:.5f} secs)".format(local_cold_scope, end_t - start_t)
                            )

                            # update tsspn
                            with self._lock:
                                independent_component.children = new_children
            finally:
                self._updating = False
            end_t = time.perf_counter()
            self.updating_cnt += 1
            self.updating_time += end_t - start_t
            self.hot_tags_scope = list(hot_tags_scope)
            logging.debug("finish hot tags updating (in {} secs)".format(end_t - start_t))


        self._updating = True
        hot_tags_scope = {self.schema.tags.index(tag) for tag in hot_tags}
        if sync:
            logging.debug("start updating hot tags {}, scope {} ...".format(hot_tags, hot_tags_scope))
            background_update()
        else:
            logging.debug("start updating hot tags {}, scope {} in background ...".format(hot_tags, hot_tags_scope))
            threading.Thread(target=background_update, daemon=True).start()

    def is_updating(self):
        return self._updating

    def get_tags(self):
        return self.schema.tags

    def save(self, model_path):
        return self.tsspn.save(model_path)

    def _parse_conditions(self, conditions, group_by_columns=None, group_by_tuples=None):
        """
        Translates string conditions to NumericRange and NominalRanges the SPN understands.
        """
        assert self.tsspn.column_names is not None, "For probability evaluation column names have to be provided."
        group_by_columns_merged = None
        if group_by_columns is None or group_by_columns == []:
            ranges = np.array([None] * len(self.tsspn.column_names)).reshape(1, len(self.tsspn.column_names))
        else:
            ranges = np.array([[None] * len(self.tsspn.column_names)] * len(group_by_tuples))
            group_by_columns_merged = [table + '.' + attribute for table, attribute in group_by_columns]

        for (table, condition) in conditions:

            table_obj = self.tsspn.schema_graph.table_dictionary[table]

            # is an nn attribute condition
            if table_obj.table_nn_attribute in condition:
                full_nn_attribute_name = table + '.' + table_obj.table_nn_attribute
                # unnecessary because column is never NULL
                if full_nn_attribute_name not in self.tsspn.column_names:
                    continue
                # column can become NULL
                elif condition == table_obj.table_nn_attribute + ' IS NOT NULL':
                    attribute_index = self.tsspn.column_names.index(full_nn_attribute_name)
                    ranges[:, attribute_index] = NominalRange([1])
                    continue
                elif condition == table_obj.table_nn_attribute + ' IS NULL':
                    attribute_index = self.tsspn.column_names.index(full_nn_attribute_name)
                    ranges[:, attribute_index] = NominalRange([0])
                    continue
                else:
                    raise NotImplementedError

            # for other attributes parse. Find matching attr.
            # matching_fd_cols = [column for column in list(self.tsspn.table_meta_data[table]['fd_dict'].keys())
            #                     if column + '<' in table + '.' + condition or column + '=' in table + '.' + condition
            #                     or column + '>' in table + '.' + condition or column + ' ' in table + '.' + condition]
            matching_fd_cols = []
            matching_cols = [column for column in self.tsspn.column_names if column + '<' in table + '.' + condition or
                             column + '=' in table + '.' + condition or column + '>' in table + '.' + condition
                             or column + ' ' in table + '.' + condition]
            assert len(matching_cols) == 1 or len(matching_fd_cols) == 1, "Found multiple or no matching columns"
            if len(matching_cols) == 1:
                matching_column = matching_cols[0]

            elif len(matching_fd_cols) == 1:
                matching_fd_column = matching_fd_cols[0]

                def find_recursive_values(column, dest_values):
                    source_attribute, dictionary = list(self.tsspn.table_meta_data[table]['fd_dict'][column].items())[0]
                    if len(self.tsspn.table_meta_data[table]['fd_dict'][column].keys()) > 1:
                        logging.warning(f"Current functional dependency handling is not designed for attributes with "
                                       f"more than one ancestor such as {column}. This can lead to error in further "
                                       f"processing.")
                    source_values = []
                    for dest_value in dest_values:
                        if not isinstance(list(dictionary.keys())[0], str):
                            dest_value = float(dest_value)
                        source_values += dictionary[dest_value]

                    if source_attribute in self.tsspn.column_names:
                        return source_attribute, source_values
                    return find_recursive_values(source_attribute, source_values)

                if '=' in condition:
                    _, literal = condition.split('=', 1)
                    literal_list = [literal.strip(' "\'')]
                elif 'NOT IN' in condition:
                    literal_list = _literal_list(condition)
                elif 'IN' in condition:
                    literal_list = _literal_list(condition)

                matching_column, values = find_recursive_values(matching_fd_column, literal_list)
                attribute_index = self.tsspn.column_names.index(matching_column)

                if self.tsspn.meta_types[attribute_index] == MetaType.DISCRETE:
                    condition = matching_column + 'IN ('
                    for i, value in enumerate(values):
                        condition += '"' + value + '"'
                        if i < len(values) - 1:
                            condition += ','
                    condition += ')'
                else:
                    min_value = min(values)
                    max_value = max(values)
                    if values == list(range(min_value, max_value + 1)):
                        ranges = _adapt_ranges(attribute_index, max_value, ranges, inclusive=True, lower_than=True)
                        ranges = _adapt_ranges(attribute_index, min_value, ranges, inclusive=True, lower_than=False)
                        continue
                    else:
                        raise NotImplementedError

            attribute_index = self.tsspn.column_names.index(matching_column)

            if self.tsspn.meta_types[attribute_index] == MetaType.DISCRETE:

                val_dict = self.tsspn.table_meta_data[table]['categorical_columns_dict'][matching_column]

                if '=' in condition:
                    column, literal = condition.split('=', 1)
                    literal = literal.strip(' "\'')

                    if group_by_columns_merged is None or matching_column not in group_by_columns_merged:
                        if literal not in val_dict:
                            val = -1 # invalid value
                        else:
                            val = val_dict[literal]
                        ranges[:, attribute_index] = NominalRange([val])
                    else:
                        matching_group_by_idx = group_by_columns_merged.index(matching_column)
                        # due to functional dependencies this check does not make sense any more
                        # assert val_dict[literal] == group_by_tuples[0][matching_group_by_idx]
                        for idx in range(len(ranges)):
                            literal = group_by_tuples[idx][matching_group_by_idx]
                            ranges[idx, attribute_index] = NominalRange([literal])

                elif 'NOT IN' in condition:
                    literal_list = _literal_list(condition)
                    single_range = NominalRange(
                        [val_dict[literal] for literal in val_dict.keys() if not literal in literal_list])
                    if self.tsspn.null_values[attribute_index] in single_range.possible_values:
                        single_range.possible_values.remove(self.tsspn.null_values[attribute_index])
                    if all([single_range is None for single_range in ranges[:, attribute_index]]):
                        ranges[:, attribute_index] = single_range
                    else:
                        for i, nominal_range in enumerate(ranges[:, attribute_index]):
                            ranges[i, attribute_index] = NominalRange(
                                list(set(nominal_range.possible_values).intersection(single_range.possible_values)))

                elif 'IN' in condition:
                    literal_list = _literal_list(condition)
                    single_range = NominalRange([val_dict[literal] for literal in literal_list])
                    if all([single_range is None for single_range in ranges[:, attribute_index]]):
                        ranges[:, attribute_index] = single_range
                    else:
                        for i, nominal_range in enumerate(ranges[:, attribute_index]):
                            ranges[i, attribute_index] = NominalRange(list(
                                set(nominal_range.possible_values).intersection(single_range.possible_values)))

            elif self.tsspn.meta_types[attribute_index] == MetaType.REAL:
                if '<=' in condition:
                    col, literal = condition.split('<=', 1)
                    if col == "_timestamp":
                        import pandas as pd
                        literal = float(pd.Timestamp(int(literal)).value)
                    else:
                        literal = float(literal.strip())
                    ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=True, lower_than=True)

                elif '>=' in condition:
                    col, literal = condition.split('>=', 1)
                    if col == "_timestamp":
                        import pandas as pd
                        literal = float(pd.Timestamp(int(literal)).value)
                    else:
                        literal = float(literal.strip())
                    ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=True, lower_than=False)
                elif '=' in condition:
                    col, literal = condition.split('=', 1)
                    if col == "_timestamp":
                        import pandas as pd
                        literal = float(pd.Timestamp(literal).value)
                    else:
                        literal = float(literal.strip())

                    def non_conflicting(single_numeric_range):
                        assert single_numeric_range[attribute_index] is None or \
                               (single_numeric_range[attribute_index][0][0] > literal or
                                single_numeric_range[attribute_index][0][1] < literal), "Value range does not " \
                                                                                        "contain any values"

                    map(non_conflicting, ranges)
                    if group_by_columns_merged is None or matching_column not in group_by_columns_merged:
                        ranges[:, attribute_index] = NumericRange([[literal, literal]])
                    else:
                        matching_group_by_idx = group_by_columns_merged.index(matching_column)
                        assert literal == group_by_tuples[0][matching_group_by_idx]
                        for idx in range(len(ranges)):
                            literal = group_by_tuples[idx][matching_group_by_idx]
                            ranges[idx, attribute_index] = NumericRange([[literal, literal]])

                elif '<' in condition:
                    col, literal = condition.split('<', 1)
                    if col == "_timestamp":
                        import pandas as pd
                        literal = float(pd.Timestamp(literal).value)
                    else:
                        literal = float(literal.strip())
                    ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=False, lower_than=True)
                elif '>' in condition:
                    col, literal = condition.split('>', 1)
                    if col == "_timestamp":
                        import pandas as pd
                        literal = float(pd.Timestamp(literal).value)
                    else:
                        literal = float(literal.strip())
                    ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=False, lower_than=False)
                else:
                    raise ValueError("Unknown operator")

                def is_invalid_interval(single_numeric_range):
                    assert single_numeric_range[attribute_index].ranges[0][1] >= \
                           single_numeric_range[attribute_index].ranges[0][0], \
                        "Value range does not contain any values"

                map(is_invalid_interval, ranges)

            else:
                raise ValueError("Unknown Metatype")

        if group_by_columns_merged is not None:
            for matching_group_by_idx, column in enumerate(group_by_columns_merged):
                if column not in self.tsspn.column_names:
                    continue
                attribute_index = self.tsspn.column_names.index(column)
                if self.tsspn.meta_types[attribute_index] == MetaType.DISCRETE:
                    for idx in range(len(ranges)):
                        literal = group_by_tuples[idx][matching_group_by_idx]
                        if not isinstance(literal, list):
                            literal = [literal]

                        if ranges[idx, attribute_index] is None:
                            ranges[idx, attribute_index] = NominalRange(literal)
                        else:
                            updated_possible_values = set(ranges[idx, attribute_index].possible_values).intersection(
                                literal)
                            ranges[idx, attribute_index] = NominalRange(list(updated_possible_values))

                elif self.tsspn.meta_types[attribute_index] == MetaType.REAL:
                    for idx in range(len(ranges)):
                        literal = group_by_tuples[idx][matching_group_by_idx]
                        assert not isinstance(literal, list)
                        ranges[idx, attribute_index] = NumericRange([[literal, literal]])
                else:
                    raise ValueError("Unknown Metatype")

        return ranges

def _literal_list(condition):
    _, literals = condition.split('(', 1)
    return [value.strip(' "\'') for value in literals[:-1].split(',')]

def _adapt_ranges(attribute_index, literal, ranges, inclusive=True, lower_than=True):
    matching_none_intervals = [idx for idx, single_range in enumerate(ranges[:, attribute_index]) if
                               single_range is None]
    if lower_than:
        for idx, single_range in enumerate(ranges):
            if single_range[attribute_index] is None or single_range[attribute_index].ranges[0][1] <= literal:
                continue
            ranges[idx, attribute_index].ranges[0][1] = literal
            ranges[idx, attribute_index].inclusive_intervals[0][1] = inclusive

        ranges[matching_none_intervals, attribute_index] = NumericRange([[-np.inf, literal]],
                                                                        inclusive_intervals=[[False, inclusive]])

    else:
        for idx, single_range in enumerate(ranges):
            if single_range[attribute_index] is None or single_range[attribute_index].ranges[0][0] >= literal:
                continue
            ranges[idx, attribute_index].ranges[0][0] = literal
            ranges[idx, attribute_index].inclusive_intervals[0][0] = inclusive
        ranges[matching_none_intervals, attribute_index] = NumericRange([[literal, np.inf]],
                                                                        inclusive_intervals=[[inclusive, False]])

    return ranges
