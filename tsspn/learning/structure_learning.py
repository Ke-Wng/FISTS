import logging
import multiprocessing
import os
from collections import deque
from enum import Enum

import numpy as np
from spn.structure.Base import get_nodes_by_type, Leaf, Node, get_number_of_edges, get_depth, assign_ids
from spn.algorithms.StructureLearning import default_slicer

from tsspn.algorithms.transform_structure import Prune
from tsspn.algorithms.validity.validity import is_valid
from tsspn.learning.rspn_learning import build_rspn
from tsspn.learning.tsspn_learning import find_optimal_n_segments, transform_data_to_time_series
from tsspn.structure.base import Sum, Product, TSProduct, TSSum
from tsspn.structure.Multi_Histograms import create_multi_histogram_leaf

from FSPN.Structure.nodes import Context
from FSPN.Structure.leaves.parametric.Parametric import Categorical, Gaussian
from tsspn.structure.hypercube import create_hypercube_leaf

logger = logging.getLogger(__name__)

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

parallel = True

if parallel:
    cpus = max(1, os.cpu_count() - 2)  # - int(os.getloadavg()[2])
else:
    cpus = 1
pool = multiprocessing.Pool(processes=cpus)


class Operation(Enum):
    SPLIT_SERIES = 1
    SPLIT_TIME_RANGE = 2
    CREATE_TS_PRODUCT = 3


class Task:
    def __init__(self, dataset, parent, children_pos, n_timestamps, no_series, no_time_range):
        self.dataset = dataset
        self.parent = parent
        self.children_pos = children_pos
        self.n_timestamps = n_timestamps
        self.no_series = no_series
        self.no_time_range = no_time_range

    def flatten(self):
        return (self.dataset, self.parent, self.children_pos, self.n_timestamps, self.no_series, self.no_time_range)


def get_next_operation(min_series_slices=100, min_timestamps_slices=30):
    def next_operation(
            data,
            n_timestamps,
            no_series=False,
            no_time_range=False,
            is_first=False,
    ):

        # minimalTimeRange = n_timestamps <= min_timestamps_slices
        is_minimum_series = data.shape[0] // n_timestamps <= min_series_slices

        # Util reach the minimun series, always split series
        if is_minimum_series or no_series:
            return Operation.CREATE_TS_PRODUCT, None
        return Operation.SPLIT_SERIES, None

        if no_series and no_time_range:
            return Operation.CREATE_TS_PRODUCT, None

        if is_first:
            return Operation.SPLIT_SERIES, None

        # if no_series:
        #     if minimalTimeRange or no_time_range:
        #         return Operation.CREATE_TS_PRODUCT, None
        #     else:
        #         return Operation.SPLIT_TIME_RANGE, None

        if is_minimum_series:
            return Operation.CREATE_TS_PRODUCT, None
        return Operation.SPLIT_SERIES, None

    return next_operation


def learn_structure(
        dataset,
        ds_context,
        split_rows,
        split_cols,
        split_series,
        split_time_range,
        create_leaf,
        min_instances_slice=100,
        min_series_slice=1000,
        initial_scope=None,
        data_slicer=default_slicer,
        build_mh=False,
        build_hist=False,
):
    assert dataset is not None
    assert ds_context is not None
    assert split_rows is not None
    assert split_cols is not None
    assert split_series is not None
    assert split_time_range is not None
    assert create_leaf is not None

    next_operation = get_next_operation(min_series_slices=min_series_slice)

    if initial_scope is None:
        initial_scope = list(range(dataset.shape[1]))
        num_conditional_cols = None
    elif len(initial_scope) < dataset.shape[1]:
        num_conditional_cols = dataset.shape[1] - len(initial_scope)
    else:
        num_conditional_cols = None
        assert len(initial_scope) > dataset.shape[1], "check initial scope: %s" % initial_scope

    root = Product()
    root.children.append(None)
    root.cardinality = dataset.shape[0]

    # assign series ids
    series_ids = np.repeat(np.arange(ds_context.n_series), dataset.shape[0] // ds_context.n_series).reshape(-1, 1)
    dataset = np.hstack((dataset, series_ids))

    tasks = deque()
    tasks.append(Task(dataset, root, 0, dataset.shape[0] // ds_context.n_series, False, False))

    while tasks:

        local_data, parent, children_pos, n_timestamps, no_series, no_time_range = tasks.popleft().flatten()

        operation, op_params = next_operation(
            local_data,
            n_timestamps,
            no_series=no_series,
            no_time_range=no_time_range,
            is_first=(parent is root),
        )

        logging.debug("OP: {} on {} series {} timestamps (remaining tasks {})".format(operation, local_data.shape[0] // n_timestamps, n_timestamps, len(tasks)))

        if operation == Operation.SPLIT_SERIES:
            split_start_t = perf_counter()
            data_slices, _ = split_series(local_data, ds_context, initial_scope, n_timestamps)
            split_end_t = perf_counter()
            logging.debug(
                "\t\tfound {} series clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
            )

            if len(data_slices) == 1:
                tasks.append(Task(data_slices[0][0], parent, children_pos, n_timestamps, True, True))
                continue

            create_sum_node(children_pos, data_slices, parent, initial_scope, tasks)

            continue

        elif operation == Operation.SPLIT_TIME_RANGE:

            split_start_t = perf_counter()
            data_slices = split_time_range(local_data, ds_context, n_timestamps)
            split_end_t = perf_counter()
            logging.debug(
                "\t\tfound {} time range clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
            )

            create_sum_node(children_pos, data_slices, parent, initial_scope, tasks)

            continue

        elif operation == Operation.CREATE_TS_PRODUCT:
            assert local_data.shape[1] == len(initial_scope) + 1
            local_data, series_ids = local_data[:,:-1], local_data[:,-1]
            series_start_indices = np.arange(0, local_data.shape[0], n_timestamps)
            series_ids = series_ids[series_start_indices].astype(np.int64)
            tag_data = local_data[series_start_indices, 0:-2]

            ts_product_node = TSProduct(series_ids=series_ids)
            ts_product_node.cardinality = local_data.shape[0]
            ts_product_node.scope.extend(initial_scope)

            tag_node = build_rspn(tag_data, ds_context, split_rows, split_cols, create_leaf, min_instances_slice=min_instances_slice, build_mh=build_mh, build_hist=build_hist)

            parametric_types = [Categorical] * tag_data.shape[1] + [Gaussian, Categorical]
            ds_context_tmp = Context(parametric_types=parametric_types).add_domains(local_data)

            # fspn = FSPN()
            # fspn.learn_from_data(tag_data, ds_context_tmp, min_instances_slice=ds_context.n_series * 0.01, max_sampling_threshold_cols=10000)
            # tag_node = fspn.model

            ts_product_node.add_tag_node(tag_node)
            
            metric_data = local_data[:, -2:]
            if build_hist:
                metric_leaf = create_leaf(metric_data[:,0].reshape(-1, 1), ds_context, initial_scope[-2:])
                ts_leaf = create_leaf(metric_data[:,1].reshape(-1, 1), ds_context, initial_scope[-1:])
                ts_product_node.children.append(metric_leaf)
                ts_product_node.children.append(ts_leaf)
            else:
                metric_ts_leaf = create_multi_histogram_leaf(metric_data, ds_context_tmp, initial_scope[-2:], [], discretize=True)
                ts_product_node.add_metric_ts_node(metric_ts_leaf, len(metric_data))

            parent.children[children_pos] = ts_product_node

            continue

        else:
            raise Exception("Invalid operation: " + operation)

    node = root.children[0]

    print(get_structure_stats(node))

    assign_ids(node)
    valid, err = is_valid(node)
    assert valid, "invalid spn: " + err
    # node = Prune(node)
    # valid, err = is_valid(node)
    # assert valid, "invalid spn: " + err

    return node


def create_sum_node(children_pos, data_slices, parent, scope, tasks, cluster_centers=None, no_series=False):
    node = TSSum()
    node.scope.extend(scope)

    parent.children[children_pos] = node
    # assert parent.scope == node.scope
    cardinality = 0
    for i, (data_slice, n_timestamps, proportion) in enumerate(data_slices):
        cardinality += len(data_slice)
        node.children.append(None)
        node.weights.append(proportion)
        tasks.append(Task(data_slice, node, len(node.children) - 1, n_timestamps, no_series, False))
    node.cardinality = cardinality


def split_series_into_segments(data, n_segments, n_series):
    if n_segments == 1:
        return [data]
    n_points = data.shape[0] // n_series
    all_segments = []
    n_points_per_segment = n_points // n_segments
    for i in range(n_segments):
        segments = []
        segment_start_idx = i * n_points_per_segment
        segment_end_idx = (i + 1) * n_points_per_segment
        for j in range(n_series):
            series_start_idx = j * n_points
            segment = data[series_start_idx + segment_start_idx:series_start_idx + segment_end_idx, :]
            segments.append(segment)
        all_segments.append(np.concatenate(segments, axis=0))
    return all_segments


def get_structure_stats(node):
    num_nodes = len(get_nodes_by_type(node, Node))
    sum_nodes = get_nodes_by_type(node, Sum)
    n_sum_nodes = len(sum_nodes)
    n_prod_nodes = len(get_nodes_by_type(node, Product))
    n_ts_prod_nodes = len(get_nodes_by_type(node, TSProduct))
    n_leaf_nodes = len(get_nodes_by_type(node, Leaf))
    edges = get_number_of_edges(node)
    layers = get_depth(node)
    params = 0
    for n in sum_nodes:
        params += len(n.children)
    #for l in leaf_nodes:
     #   params += len(l.parameters)


    return """---Structure Statistics---
# nodes               %s
    # sum nodes       %s
    # prod nodes      %s
      # ts-prod nodes %s
    # leaf nodes      %s
# params              %s
# edges               %s
# layers              %s""" % (
        num_nodes,
        n_sum_nodes,
        n_prod_nodes,
        n_ts_prod_nodes,
        n_leaf_nodes,
        params,
        edges,
        layers,
    )