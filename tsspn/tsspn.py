import logging
import os
import pickle
import threading
import time
from functools import partial

import numpy as np
from spn.structure.StatisticalTypes import MetaType
from spn.structure.Base import Context, get_nodes_by_type, Leaf

from evaluation.utils import parse_query
from tsspn.algorithms.expectations import expectation
from tsspn.algorithms.ranges import NominalRange, NumericRange
from tsspn.algorithms.validity.validity import is_valid
from tsspn.learning.tsspn_learning import create_custom_leaf, learn_mspn
from tsspn.structure.base import Product, TSProduct
from tsspn.structure.hypercube import HyperCube, hypercube_likelihood_range
from tsspn.structure.leaves import IdentityNumericLeaf, TimeSeriesLeaf, identity_expectation, Categorical, categorical_likelihood_range, \
    identity_likelihood_range, time_series_likelihood_range
from tsspn.structure.discrete_multi_histogram import DiscreteMultiHistogram, create_discrete_multi_histogram_leaf, multi_histogram_likelihood_range

logger = logging.getLogger(__name__)


def build_ds_context(column_names, meta_types, null_values, table_meta_data, no_compression_scopes, train_data, n_series,
                     group_by_threshold=1200):
    """
    Builds context according to training data.
    :param column_names:
    :param meta_types:
    :param null_values:
    :param table_meta_data:
    :param train_data:
    :return:
    """
    ds_context = Context(meta_types=meta_types)
    ds_context.null_values = null_values
    assert ds_context.null_values is not None, "Null-Values have to be specified"

    ds_context.field_idx = len(column_names) - 2
    ds_context.timestamp_idx = len(column_names) - 1

    ds_context.n_series = n_series
    ds_context.n_timestamps = train_data.shape[0] // n_series

    domain = []
    no_unique_values = []
    # If metadata is given use this to build domains for categorical values
    unified_column_dictionary = None
    if table_meta_data is not None:
        unified_column_dictionary = {k: v for table, table_md in table_meta_data.items() if
                                     table != 'inverted_columns_dict' and table != 'inverted_fd_dict'
                                     for k, v in table_md['categorical_columns_dict'].items()}

    # domain values
    group_by_attributes = []
    for col in range(train_data.shape[1]):

        feature_meta_type = meta_types[col]
        min_val = np.nanmin(train_data[:, col])
        max_val = np.nanmax(train_data[:, col])

        unique_vals = len(np.unique(train_data[:, col]))
        no_unique_values.append(unique_vals)
        if column_names is not None:
            if unique_vals <= group_by_threshold and 'mul_' not in column_names[col] and '_nn' not in column_names[col]:
                group_by_attributes.append(col)

        domain_values = [min_val, max_val]

        if feature_meta_type == MetaType.REAL:
            min_val = np.nanmin(train_data[:, col])
            max_val = np.nanmax(train_data[:, col])
            domain.append([min_val, max_val])
        elif feature_meta_type == MetaType.DISCRETE:
            # if no metadata is given, infer domains from data
            if column_names is not None \
                    and unified_column_dictionary.get(column_names[col]) is not None:
                no_diff_values = len(unified_column_dictionary[column_names[col]].keys())
                domain.append(np.arange(0, no_diff_values + 1, 1))
            else:
                domain.append(np.arange(domain_values[0], domain_values[1] + 1, 1))
        else:
            raise Exception("Unkown MetaType " + str(meta_types[col]))

    ds_context.domains = np.asanyarray(domain)
    ds_context.no_unique_values = np.asanyarray(no_unique_values)
    ds_context.group_by_attributes = group_by_attributes

    if no_compression_scopes is None:
        no_compression_scopes = []
    ds_context.no_compression_scopes = no_compression_scopes

    return ds_context


class TSSPN:

    def __init__(self, meta_types, null_values, full_sample_size, schema_graph, n_series, column_names=None, table_meta_data=None):

        self.meta_types = meta_types
        self.null_values = null_values
        self.full_sample_size = full_sample_size
        self.schema_graph = schema_graph
        self.n_series = n_series
        self.column_names = column_names
        self.table_meta_data = table_meta_data
        self.tsspn = None
        self.ds_context = None

        self.use_generated_code = False

        # training stats
        self.learn_time = None
        self.rdc_threshold = None
        self.min_instances_slice = None

    def learn(self, train_data, rdc_threshold=0.3, min_instances_slice=1, min_series_slice=1000, max_sampling_threshold_cols=10000,
              max_sampling_threshold_rows=100000, no_compression_scopes=None, build_mh=False, build_hist=False, max_clustering_variance=0.05):

        # build domains (including the dependence analysis)
        domain_start_t = time.perf_counter()
        ds_context = build_ds_context(self.column_names, self.meta_types, self.null_values, self.table_meta_data,
                                      no_compression_scopes, train_data, self.n_series)
        self.ds_context = ds_context
        domain_end_t = time.perf_counter()
        logging.debug(f"Built domains in {domain_end_t - domain_start_t} sec")

        # learn mspn
        learn_start_t = time.perf_counter()

        self.tsspn = learn_mspn(train_data, ds_context,
                               min_instances_slice=min_instances_slice, min_series_slice=min_series_slice, threshold=rdc_threshold,
                               max_sampling_threshold_cols=max_sampling_threshold_cols,
                               max_sampling_threshold_rows=max_sampling_threshold_rows,
                               build_mh=build_mh, build_hist=build_hist, max_clustering_variance=max_clustering_variance)
        # assert is_valid(self.mspn, check_ids=True)
        learn_end_t = time.perf_counter()
        self.learn_time = learn_end_t - learn_start_t
        logging.debug(f"Built SPN in {learn_end_t - learn_start_t} sec")

        # statistics
        self.rdc_threshold = rdc_threshold
        self.min_instances_slice = min_instances_slice

    def _add_null_values_to_ranges(self, range_conditions):
        for col in range(range_conditions.shape[1]):
            if range_conditions[0][col] is None:
                continue
            for idx in range(range_conditions.shape[0]):
                range_conditions[idx, col].null_value = self.null_values[col]

        return range_conditions

    def _probability(self, range_conditions, series=False):
        """
        Compute probability of range conditions.

        e.g. np.array([NominalRange([0]), NumericRange([[0,0.3]]), None])
        """
        return self._indicator_expectation([], range_conditions=range_conditions, series=series)

    def _indicator_expectation(self, feature_scope, identity_leaf_expectation=None, inverted_features=None,
                               range_conditions=None, force_no_generated=False, gen_code_stats=None, series=False):
        """
        Compute E[1_{conditions} * X_feature_scope]. Can also compute products (specify multiple feature scopes).
        For inverted features 1/val is used.

        Is basis for both unnormalized and normalized expectation computation.

        Uses safe evaluation for products, i.e. compute extra expectation for every multiplier. If results deviate too
        largely (>max_deviation), replace by mean. We also experimented with splitting the multipliers into 10 random
        groups. However, this works equally well and is faster.
        """

        if inverted_features is None:
            inverted_features = [False] * len(feature_scope)

        if range_conditions is None:
            range_conditions = np.array([None] * len(self.tsspn.scope)).reshape(1, len(self.tsspn.scope))
        else:
            range_conditions = self._add_null_values_to_ranges(range_conditions)

        if identity_leaf_expectation is None:
            _node_expectation = {IdentityNumericLeaf: identity_expectation}
        else:
            _node_expectation = {IdentityNumericLeaf: identity_leaf_expectation}

        _node_likelihoods_range = {IdentityNumericLeaf: identity_likelihood_range,
                                   Categorical: categorical_likelihood_range,
                                   TimeSeriesLeaf: time_series_likelihood_range,
                                   DiscreteMultiHistogram: multi_histogram_likelihood_range,
                                   HyperCube: hypercube_likelihood_range}

        if hasattr(self, 'use_generated_code') and self.use_generated_code and not force_no_generated:
            full_result = expectation(self.tsspn, feature_scope, inverted_features, range_conditions,
                                      node_expectation=_node_expectation, node_likelihoods=_node_likelihoods_range,
                                      use_generated_code=True, spn_id=self.id, meta_types=self.meta_types,
                                      gen_code_stats=gen_code_stats, series=series)
        else:
            full_result = expectation(self.tsspn, feature_scope, inverted_features, range_conditions,
                                      node_expectation=_node_expectation, node_likelihoods=_node_likelihoods_range,
                                      series=series)

        return full_result

    def _augment_not_null_conditions(self, feature_scope, range_conditions):
        if range_conditions is None:
            range_conditions = np.array([None] * len(self.tsspn.scope)).reshape(1, len(self.tsspn.scope))

        # for second computation make sure that features that are not normalized are not NULL
        for not_null_scope in feature_scope:
            if self.null_values[not_null_scope] is None:
                continue

            if range_conditions[0, not_null_scope] is None:
                if self.meta_types[not_null_scope] == MetaType.REAL:
                    range_conditions[:, not_null_scope] = NumericRange([[-np.inf, np.inf]], is_not_null_condition=True)
                elif self.meta_types[not_null_scope] == MetaType.DISCRETE:
                    NumericRange([[-np.inf, np.inf]], is_not_null_condition=True)
                    categorical_feature_name = self.column_names[not_null_scope]

                    for table in self.table_meta_data.keys():
                        categorical_values = self.table_meta_data[table]['categorical_columns_dict'] \
                            .get(categorical_feature_name)
                        if categorical_values is not None:
                            possible_values = list(categorical_values.values())
                            possible_values.remove(self.null_values[not_null_scope])
                            range_conditions[:, not_null_scope] = NominalRange(possible_values,
                                                                               is_not_null_condition=True)
                            break

        return range_conditions

    def _indicator_expectation_with_std(self, feature_scope, inverted_features=None,
                                        range_conditions=None):
        """
        Computes standard deviation of the estimator for 1_{conditions}*X_feature_scope. Uses the identity
        V(X)=E(X^2)-E(X)^2.
        :return:
        """
        e_x = self._indicator_expectation(feature_scope, identity_leaf_expectation=identity_expectation,
                                          inverted_features=inverted_features,
                                          range_conditions=range_conditions)

        not_null_conditions = self._augment_not_null_conditions(feature_scope, None)
        n = self.probability(not_null_conditions) * self.full_sample_size

        # shortcut: use binomial std if it is just a probability
        if len(feature_scope) == 0:
            std = np.sqrt(e_x * (1 - e_x) * 1 / n)
            return std, e_x

        e_x_sq = self._indicator_expectation(feature_scope,
                                             identity_leaf_expectation=partial(identity_expectation, power=2),
                                             inverted_features=inverted_features,
                                             range_conditions=range_conditions,
                                             force_no_generated=True)

        v_x = e_x_sq - e_x * e_x

        # Indeed divide by sample size of SPN not only qualifying tuples. Because this is not a conditional expectation.
        std = np.sqrt(v_x / n)

        return std, e_x

    def _unnormalized_conditional_expectation(self, feature_scope, inverted_features=None, range_conditions=None,
                                              impute_p=False, gen_code_stats=None):
        """
        Compute conditional expectation. Can also compute products (specify multiple feature scopes).
        For inverted features 1/val is used. Normalization is not possible here.
        """

        range_conditions = self._augment_not_null_conditions(feature_scope, range_conditions)
        unnormalized_exp = self._indicator_expectation(feature_scope, inverted_features=inverted_features,
                                                       range_conditions=range_conditions, gen_code_stats=gen_code_stats)

        p = self.probability(range_conditions)
        if any(p == 0):
            if impute_p:
                impute_val = np.mean(
                    unnormalized_exp[np.where(p != 0)[0]] / p[np.where(p != 0)[0]])
                result = unnormalized_exp / p
                result[np.where(p == 0)[0]] = impute_val
                return result

            return self._indicator_expectation(feature_scope, inverted_features=inverted_features,
                                               gen_code_stats=gen_code_stats)

        return unnormalized_exp / p

    def _unnormalized_conditional_expectation_with_std(self, feature_scope, inverted_features=None,
                                                       range_conditions=None, gen_code_stats=None):
        """
        Compute conditional expectation. Can also compute products (specify multiple feature scopes).
        For inverted features 1/val is used. Normalization is not possible here.
        """
        range_conditions = self._augment_not_null_conditions(feature_scope, range_conditions)
        p = self.probability(range_conditions)

        e_x_sq = self._indicator_expectation(feature_scope,
                                             identity_leaf_expectation=partial(identity_expectation, power=2),
                                             inverted_features=inverted_features,
                                             range_conditions=range_conditions,
                                             force_no_generated=True,
                                             gen_code_stats=gen_code_stats) / p

        e_x = self._indicator_expectation(feature_scope, inverted_features=inverted_features,
                                          range_conditions=range_conditions,
                                          gen_code_stats=gen_code_stats) / p

        v_x = e_x_sq - e_x * e_x

        n = p * self.full_sample_size
        std = np.sqrt(v_x / n)

        return std, e_x

    def _normalized_conditional_expectation(self, feature_scope, inverted_features=None, normalizing_scope=None,
                                            range_conditions=None, standard_deviations=False, impute_p=False,
                                            gen_code_stats=None):
        """
        Computes unbiased estimate for conditional expectation E(feature_scope| range_conditions).
        To this end, normalization might be required (will always be certain multipliers.)
        E[1_{conditions} * X_feature_scope] / E[1_{conditions} * X_normalizing_scope]

        :param feature_scope:
        :param inverted_features:
        :param normalizing_scope:
        :param range_conditions:
        :return:
        """
        if range_conditions is None:
            range_conditions = np.array([None] * len(self.tsspn.scope)).reshape(1, len(self.tsspn.scope))

        # If normalization is not required, simply return unnormalized conditional expectation
        if normalizing_scope is None or len(normalizing_scope) == 0:
            if standard_deviations:
                return self._unnormalized_conditional_expectation_with_std(feature_scope,
                                                                           inverted_features=inverted_features,
                                                                           range_conditions=range_conditions,
                                                                           gen_code_stats=gen_code_stats)
            else:
                return None, self._unnormalized_conditional_expectation(feature_scope,
                                                                        inverted_features=inverted_features,
                                                                        range_conditions=range_conditions,
                                                                        impute_p=impute_p,
                                                                        gen_code_stats=gen_code_stats)

        assert set(normalizing_scope).issubset(feature_scope), "Normalizing scope must be subset of feature scope"

        # for computation make sure that features that are not normalized are not NULL
        range_conditions = self._augment_not_null_conditions(set(feature_scope).difference(normalizing_scope),
                                                             range_conditions)

        # E[1_{conditions} * X_feature_scope]
        std = None
        if standard_deviations:
            std, _ = self._unnormalized_conditional_expectation_with_std(feature_scope,
                                                                         inverted_features=inverted_features,
                                                                         range_conditions=range_conditions,
                                                                         gen_code_stats=gen_code_stats)

        nominator = self._indicator_expectation(feature_scope,
                                                inverted_features=inverted_features,
                                                range_conditions=range_conditions,
                                                gen_code_stats=gen_code_stats)

        # E[1_{conditions} * X_normalizing_scope]
        inverted_features_of_norm = \
            [inverted_features[feature_scope.index(variable_scope)] for variable_scope in normalizing_scope]
        assert all(inverted_features_of_norm), "Normalizing factors should be inverted"

        denominator = self._indicator_expectation(normalizing_scope,
                                                  inverted_features=inverted_features_of_norm,
                                                  range_conditions=range_conditions)
        return std, nominator / denominator

#     def _parse_conditions(self, conditions, group_by_columns=None, group_by_tuples=None):
#         """
#         Translates string conditions to NumericRange and NominalRanges the SPN understands.
#         """
#         assert self.column_names is not None, "For probability evaluation column names have to be provided."
#         group_by_columns_merged = None
#         if group_by_columns is None or group_by_columns == []:
#             ranges = np.array([None] * len(self.column_names)).reshape(1, len(self.column_names))
#         else:
#             ranges = np.array([[None] * len(self.column_names)] * len(group_by_tuples))
#             group_by_columns_merged = [table + '.' + attribute for table, attribute in group_by_columns]

#         for (table, condition) in conditions:

#             table_obj = self.schema_graph.table_dictionary[table]

#             # is an nn attribute condition
#             if table_obj.table_nn_attribute in condition:
#                 full_nn_attribute_name = table + '.' + table_obj.table_nn_attribute
#                 # unnecessary because column is never NULL
#                 if full_nn_attribute_name not in self.column_names:
#                     continue
#                 # column can become NULL
#                 elif condition == table_obj.table_nn_attribute + ' IS NOT NULL':
#                     attribute_index = self.column_names.index(full_nn_attribute_name)
#                     ranges[:, attribute_index] = NominalRange([1])
#                     continue
#                 elif condition == table_obj.table_nn_attribute + ' IS NULL':
#                     attribute_index = self.column_names.index(full_nn_attribute_name)
#                     ranges[:, attribute_index] = NominalRange([0])
#                     continue
#                 else:
#                     raise NotImplementedError

#             # for other attributes parse. Find matching attr.
#             # matching_fd_cols = [column for column in list(self.table_meta_data[table]['fd_dict'].keys())
#             #                     if column + '<' in table + '.' + condition or column + '=' in table + '.' + condition
#             #                     or column + '>' in table + '.' + condition or column + ' ' in table + '.' + condition]
#             matching_fd_cols = []
#             matching_cols = [column for column in self.column_names if column + '<' in table + '.' + condition or
#                              column + '=' in table + '.' + condition or column + '>' in table + '.' + condition
#                              or column + ' ' in table + '.' + condition]
#             assert len(matching_cols) == 1 or len(matching_fd_cols) == 1, "Found multiple or no matching columns"
#             if len(matching_cols) == 1:
#                 matching_column = matching_cols[0]

#             elif len(matching_fd_cols) == 1:
#                 matching_fd_column = matching_fd_cols[0]

#                 def find_recursive_values(column, dest_values):
#                     source_attribute, dictionary = list(self.table_meta_data[table]['fd_dict'][column].items())[0]
#                     if len(self.table_meta_data[table]['fd_dict'][column].keys()) > 1:
#                         logger.warning(f"Current functional dependency handling is not designed for attributes with "
#                                        f"more than one ancestor such as {column}. This can lead to error in further "
#                                        f"processing.")
#                     source_values = []
#                     for dest_value in dest_values:
#                         if not isinstance(list(dictionary.keys())[0], str):
#                             dest_value = float(dest_value)
#                         source_values += dictionary[dest_value]

#                     if source_attribute in self.column_names:
#                         return source_attribute, source_values
#                     return find_recursive_values(source_attribute, source_values)

#                 if '=' in condition:
#                     _, literal = condition.split('=', 1)
#                     literal_list = [literal.strip(' "\'')]
#                 elif 'NOT IN' in condition:
#                     literal_list = _literal_list(condition)
#                 elif 'IN' in condition:
#                     literal_list = _literal_list(condition)

#                 matching_column, values = find_recursive_values(matching_fd_column, literal_list)
#                 attribute_index = self.column_names.index(matching_column)

#                 if self.meta_types[attribute_index] == MetaType.DISCRETE:
#                     condition = matching_column + 'IN ('
#                     for i, value in enumerate(values):
#                         condition += '"' + value + '"'
#                         if i < len(values) - 1:
#                             condition += ','
#                     condition += ')'
#                 else:
#                     min_value = min(values)
#                     max_value = max(values)
#                     if values == list(range(min_value, max_value + 1)):
#                         ranges = _adapt_ranges(attribute_index, max_value, ranges, inclusive=True, lower_than=True)
#                         ranges = _adapt_ranges(attribute_index, min_value, ranges, inclusive=True, lower_than=False)
#                         continue
#                     else:
#                         raise NotImplementedError

#             attribute_index = self.column_names.index(matching_column)

#             if self.meta_types[attribute_index] == MetaType.DISCRETE:

#                 val_dict = self.table_meta_data[table]['categorical_columns_dict'][matching_column]

#                 if '=' in condition:
#                     column, literal = condition.split('=', 1)
#                     literal = literal.strip(' "\'')

#                     if group_by_columns_merged is None or matching_column not in group_by_columns_merged:
#                         if literal not in val_dict:
#                             val = -1 # invalid value
#                         else:
#                             val = val_dict[literal]
#                         ranges[:, attribute_index] = NominalRange([val])
#                     else:
#                         matching_group_by_idx = group_by_columns_merged.index(matching_column)
#                         # due to functional dependencies this check does not make sense any more
#                         # assert val_dict[literal] == group_by_tuples[0][matching_group_by_idx]
#                         for idx in range(len(ranges)):
#                             literal = group_by_tuples[idx][matching_group_by_idx]
#                             ranges[idx, attribute_index] = NominalRange([literal])

#                 elif 'NOT IN' in condition:
#                     literal_list = _literal_list(condition)
#                     single_range = NominalRange(
#                         [val_dict[literal] for literal in val_dict.keys() if not literal in literal_list])
#                     if self.null_values[attribute_index] in single_range.possible_values:
#                         single_range.possible_values.remove(self.null_values[attribute_index])
#                     if all([single_range is None for single_range in ranges[:, attribute_index]]):
#                         ranges[:, attribute_index] = single_range
#                     else:
#                         for i, nominal_range in enumerate(ranges[:, attribute_index]):
#                             ranges[i, attribute_index] = NominalRange(
#                                 list(set(nominal_range.possible_values).intersection(single_range.possible_values)))

#                 elif 'IN' in condition:
#                     literal_list = _literal_list(condition)
#                     single_range = NominalRange([val_dict[literal] for literal in literal_list])
#                     if all([single_range is None for single_range in ranges[:, attribute_index]]):
#                         ranges[:, attribute_index] = single_range
#                     else:
#                         for i, nominal_range in enumerate(ranges[:, attribute_index]):
#                             ranges[i, attribute_index] = NominalRange(list(
#                                 set(nominal_range.possible_values).intersection(single_range.possible_values)))

#             elif self.meta_types[attribute_index] == MetaType.REAL:
#                 if '<=' in condition:
#                     col, literal = condition.split('<=', 1)
#                     if col == "_timestamp":
#                         import pandas as pd
#                         literal = float(pd.Timestamp(int(literal)).value)
#                     else:
#                         literal = float(literal.strip())
#                     ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=True, lower_than=True)

#                 elif '>=' in condition:
#                     col, literal = condition.split('>=', 1)
#                     if col == "_timestamp":
#                         import pandas as pd
#                         literal = float(pd.Timestamp(int(literal)).value)
#                     else:
#                         literal = float(literal.strip())
#                     ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=True, lower_than=False)
#                 elif '=' in condition:
#                     col, literal = condition.split('=', 1)
#                     if col == "_timestamp":
#                         import pandas as pd
#                         literal = float(pd.Timestamp(literal).value)
#                     else:
#                         literal = float(literal.strip())

#                     def non_conflicting(single_numeric_range):
#                         assert single_numeric_range[attribute_index] is None or \
#                                (single_numeric_range[attribute_index][0][0] > literal or
#                                 single_numeric_range[attribute_index][0][1] < literal), "Value range does not " \
#                                                                                         "contain any values"

#                     map(non_conflicting, ranges)
#                     if group_by_columns_merged is None or matching_column not in group_by_columns_merged:
#                         ranges[:, attribute_index] = NumericRange([[literal, literal]])
#                     else:
#                         matching_group_by_idx = group_by_columns_merged.index(matching_column)
#                         assert literal == group_by_tuples[0][matching_group_by_idx]
#                         for idx in range(len(ranges)):
#                             literal = group_by_tuples[idx][matching_group_by_idx]
#                             ranges[idx, attribute_index] = NumericRange([[literal, literal]])

#                 elif '<' in condition:
#                     col, literal = condition.split('<', 1)
#                     if col == "_timestamp":
#                         import pandas as pd
#                         literal = float(pd.Timestamp(literal).value)
#                     else:
#                         literal = float(literal.strip())
#                     ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=False, lower_than=True)
#                 elif '>' in condition:
#                     col, literal = condition.split('>', 1)
#                     if col == "_timestamp":
#                         import pandas as pd
#                         literal = float(pd.Timestamp(literal).value)
#                     else:
#                         literal = float(literal.strip())
#                     ranges = _adapt_ranges(attribute_index, literal, ranges, inclusive=False, lower_than=False)
#                 else:
#                     raise ValueError("Unknown operator")

#                 def is_invalid_interval(single_numeric_range):
#                     assert single_numeric_range[attribute_index].ranges[0][1] >= \
#                            single_numeric_range[attribute_index].ranges[0][0], \
#                         "Value range does not contain any values"

#                 map(is_invalid_interval, ranges)

#             else:
#                 raise ValueError("Unknown Metatype")

#         if group_by_columns_merged is not None:
#             for matching_group_by_idx, column in enumerate(group_by_columns_merged):
#                 if column not in self.column_names:
#                     continue
#                 attribute_index = self.column_names.index(column)
#                 if self.meta_types[attribute_index] == MetaType.DISCRETE:
#                     for idx in range(len(ranges)):
#                         literal = group_by_tuples[idx][matching_group_by_idx]
#                         if not isinstance(literal, list):
#                             literal = [literal]

#                         if ranges[idx, attribute_index] is None:
#                             ranges[idx, attribute_index] = NominalRange(literal)
#                         else:
#                             updated_possible_values = set(ranges[idx, attribute_index].possible_values).intersection(
#                                 literal)
#                             ranges[idx, attribute_index] = NominalRange(list(updated_possible_values))

#                 elif self.meta_types[attribute_index] == MetaType.REAL:
#                     for idx in range(len(ranges)):
#                         literal = group_by_tuples[idx][matching_group_by_idx]
#                         assert not isinstance(literal, list)
#                         ranges[idx, attribute_index] = NumericRange([[literal, literal]])
#                 else:
#                     raise ValueError("Unknown Metatype")

#         return ranges

#     def estimate_cardinality(self, sql):
#         query = parse_query(sql, self.schema_graph)
#         ranges = self._parse_conditions(query.conditions)
#         start_t = time.perf_counter()
#         points_prob, series_prob = self._probability(ranges)
#         end_t = time.perf_counter()
#         return points_prob, series_prob, end_t - start_t

#     def update_hot_tags(self, hot_tags, tag_data, ds_context, sync=False, min_hot_tags_num=2, max_hot_tags_num=5):
#         if self._updating:
#             logging.debug("the last update hasn't completed, ignore hot tags {}".format(hot_tags))
#             return

#         def background_update():
#             try:
#                 # collect all ts-product nodes
#                 ts_products = get_nodes_by_type(self.tsspn, TSProduct)
#                 for tsnode in ts_products:
#                     tag_node = tsnode.get_tag_node()
#                     # re-train hot tags for each independent components
#                     for independent_component in tag_node.children:
#                         assert isinstance(independent_component, Product) or isinstance(independent_component, Leaf)
#                         independent_scope = independent_component.scope
#                         if len(independent_scope) <= min_hot_tags_num:
#                             assert isinstance(independent_component, DiscreteMultiHistogram)
#                             continue

#                         local_hot_tags = hot_tags.intersection(independent_scope)
#                         if len(local_hot_tags) <= 1:
#                             continue
#                         need_rebuild = True
#                         for node in independent_component.children:
#                             if local_hot_tags.issubset(node.scope):
#                                 need_rebuild = False
#                                 break
#                         if need_rebuild:
#                             new_children = []
#                             local_data = tag_data[tsnode.series_ids, :]

#                             if len(local_hot_tags) > max_hot_tags_num:
#                                 local_hot_tags = local_hot_tags[:max_hot_tags_num]
#                             start_t = time.perf_counter()
#                             node = create_discrete_multi_histogram_leaf(local_data[:, local_hot_tags], independent_scope)
#                             end_t = time.perf_counter
#                             logging.debug(
#                                 "\t\tcreated multi histogram leaf with scope {} (in {:.5f} secs)".format(local_hot_tags, end_t - start_t)
#                             )

#                             start_t = time.perf_counter()
#                             new_children.append(node)
#                             local_cold_tags = independent_scope - local_hot_tags
#                             for scope in local_cold_tags:
#                                 node = create_custom_leaf(local_data[:, scope], ds_context, [scope])
#                                 new_children.append(node)
#                             end_t = time.perf_counter()
#                             logging.debug(
#                                 "\t\tcreated histogram leaves for scope {} (in {:.5f} secs)".format(local_cold_tags, end_t - start_t)
#                             )

#                             # update tsspn
#                             with self._lock:
#                                 independent_component.children = new_children
#             finally:
#                 self._updating = False

#         self._updating = True
#         if sync:
#             logging.debug("start updating hot tags {} ...".format(hot_tags))
#             background_update()
#         else:
#             logging.debug("start updating hot tags {} in background ...".format(hot_tags))
#             threading.Thread(target=background_update, daemon=True).start()

#     def is_updating(self):
#         return self._updating

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return os.path.getsize(path)

# def _literal_list(condition):
#     _, literals = condition.split('(', 1)
#     return [value.strip(' "\'') for value in literals[:-1].split(',')]

# def _adapt_ranges(attribute_index, literal, ranges, inclusive=True, lower_than=True):
#     matching_none_intervals = [idx for idx, single_range in enumerate(ranges[:, attribute_index]) if
#                                single_range is None]
#     if lower_than:
#         for idx, single_range in enumerate(ranges):
#             if single_range[attribute_index] is None or single_range[attribute_index].ranges[0][1] <= literal:
#                 continue
#             ranges[idx, attribute_index].ranges[0][1] = literal
#             ranges[idx, attribute_index].inclusive_intervals[0][1] = inclusive

#         ranges[matching_none_intervals, attribute_index] = NumericRange([[-np.inf, literal]],
#                                                                         inclusive_intervals=[[False, inclusive]])

#     else:
#         for idx, single_range in enumerate(ranges):
#             if single_range[attribute_index] is None or single_range[attribute_index].ranges[0][0] >= literal:
#                 continue
#             ranges[idx, attribute_index].ranges[0][0] = literal
#             ranges[idx, attribute_index].inclusive_intervals[0][0] = inclusive
#         ranges[matching_none_intervals, attribute_index] = NumericRange([[literal, np.inf]],
#                                                                         inclusive_intervals=[[inclusive, False]])

#     return ranges
