import logging
from time import perf_counter

import numpy as np
from spn.algorithms.Inference import likelihood

from FSPN.Structure.model import FSPN
from tsspn.code_generation.convert_conditions import convert_range

from tsspn.structure.base import Sum, Product, TSProduct, TSSum
from spn.structure.Base import Leaf

logger = logging.getLogger(__name__)


def expectation(spn, feature_scope, inverted_features, ranges, node_expectation=None, node_likelihoods=None,
                use_generated_code=False, spn_id=None, meta_types=None, gen_code_stats=None, series=False):
    """Compute the Expectation:
        E[1_{conditions} * X_feature_scope]
        First factor is one if condition is fulfilled. For the second factor the variables in feature scope are
        multiplied. If inverted_features[i] is True, variable is taken to denominator.
        The conditional expectation would be E[1_{conditions} * X_feature_scope]/P(conditions)
    """

    # evidence_scope = set([i for i, r in enumerate(ranges) if not np.isnan(r)])
    evidence_scope = set([i for i, r in enumerate(ranges[0]) if r is not None])
    evidence = ranges

    assert not (len(evidence_scope) > 0 and evidence is None)

    relevant_scope = set()
    relevant_scope.update(evidence_scope)
    relevant_scope.update(feature_scope)
    if len(relevant_scope) == 0:
        return np.ones((ranges.shape[0], 1))

    if ranges.shape[0] == 1:

        applicable = True
        if use_generated_code:
            boolean_relevant_scope = [i in relevant_scope for i in range(len(meta_types))]
            boolean_feature_scope = [i in feature_scope for i in range(len(meta_types))]
            applicable, parameters = convert_range(boolean_relevant_scope, boolean_feature_scope, meta_types, ranges[0],
                                                   inverted_features)

        # generated C++ code
        if use_generated_code and applicable:
            time_start = perf_counter()
            import optimized_inference

            spn_func = getattr(optimized_inference, f'spn{spn_id}')
            result = np.array([[spn_func(*parameters)]])

            time_end = perf_counter()

            if gen_code_stats is not None:
                gen_code_stats.calls += 1
                gen_code_stats.total_time += (time_end - time_start)

            # logger.debug(f"\t\tGenerated Code Latency: {(time_end - time_start) * 1000:.3f}ms")
            return result

        # lightweight non-batch version
        else:
            return expectation_recursive(spn, feature_scope, inverted_features, relevant_scope, evidence,
                                        node_expectation, node_likelihoods, series)
    # full batch version
    return expectation_recursive_batch(spn, feature_scope, inverted_features, relevant_scope, evidence,
                                       node_expectation, node_likelihoods, series)


def expectation_recursive_batch(node, feature_scope, inverted_features, relevant_scope, evidence, node_expectation,
                                node_likelihoods, series=False):
    if isinstance(node, Product):

        llchildren = np.concatenate(
            [expectation_recursive_batch(child, feature_scope, inverted_features, relevant_scope, evidence,
                                         node_expectation, node_likelihoods)
             for child in node.children if
             len(relevant_scope.intersection(child.scope)) > 0], axis=1)
        return np.nanprod(llchildren, axis=1).reshape(-1, 1)

    elif isinstance(node, Sum):
        if len(relevant_scope.intersection(node.scope)) == 0:
            return np.full((evidence.shape[0], 1), np.nan)

        llchildren = np.concatenate(
            [expectation_recursive_batch(child, feature_scope, inverted_features, relevant_scope, evidence,
                                         node_expectation, node_likelihoods)
             for child in node.children], axis=1)

        relevant_children_idx = np.where(np.isnan(llchildren[0]) == False)[0]
        if len(relevant_children_idx) == 0:
            return np.array([np.nan])

        weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
        b = np.array(node.weights)[relevant_children_idx] / weights_normalizer

        return np.dot(llchildren[:, relevant_children_idx], b).reshape(-1, 1)

    else:
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:
                exps = np.zeros((evidence.shape[0], 1))

                feature_idx = feature_scope.index(node.scope[0])
                inverted = inverted_features[feature_idx]

                exps[:] = node_expectation[t_node](node, evidence, inverted=inverted)
                return exps
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        return likelihood(node, evidence, node_likelihood=node_likelihoods, series=series)


def nanproduct(product, factor):
    if np.isnan(product):
        if not np.isnan(factor):
            return factor
        else:
            return np.nan
    else:
        if np.isnan(factor):
            return product
        else:
            return product * factor


def expectation_recursive(node, feature_scope, inverted_features, relevant_scope, evidence, node_expectation,
                          node_likelihoods, series=False):
    # if not hasattr(node, "last_condition"):
    #     node.last_condition = [None] * len(node.scope)
    #     node.cached_prob = None

    # if not series:
    #     condition_matched = True
    #     for i, scope in enumerate(node.scope):
    #         scope_cond = evidence[0][scope]
    #         if scope_cond is None and node.last_condition[i] is None:
    #             continue
    #         if scope_cond is not None:
    #             if node.last_condition[i] != scope_cond.get_ranges()[0]:
    #                 condition_matched = False
    #                 node.last_condition[i] = scope_cond.get_ranges()[0]
    #         else:
    #             condition_matched = False
    #             node.last_condition[i] = None
    #     if condition_matched:
    #         return node.cached_prob

    if isinstance(node, TSProduct):
        assert len(node.children) >= 2
        assert evidence.shape[0] == 1
        ranges = evidence[0]
        l_bound = np.full((1, len(ranges)), -np.inf)
        r_bound = np.full((1, len(ranges)), np.inf)
        for i, r in enumerate(ranges):
            if r is not None:
                if isinstance(r.get_ranges(), list):
                    l_bound[0][i] = r.get_ranges()[0][0]
                    r_bound[0][i] = r.get_ranges()[0][1]
                else:
                    l_bound[0][i] = r.get_ranges()[0]
                    r_bound[0][i] = l_bound[0][i]

        tag_node = node.children[0]
        start_t = perf_counter()
        if isinstance(tag_node, Sum) or isinstance(tag_node, Product) or isinstance(tag_node, Leaf):
            tag_prob = expectation_recursive(tag_node, feature_scope, inverted_features, relevant_scope, evidence, node_expectation, node_likelihoods, series)
        else:
            fspn = FSPN()
            fspn.model = tag_node
            fspn.store_factorize_as_dict()
            tag_prob = fspn.probability((l_bound[:, :-2], r_bound[:, :-2]))[0]
        end_t = perf_counter()
        tag_node.execution_time = end_t - start_t

        metric_prob = 0
        for metric_node, n_points in node.get_metric_ts_nodes():
            start_t = perf_counter()
            if len(relevant_scope.intersection(metric_node.scope)) > 0:
                metric_prob += n_points * metric_node.query((l_bound[:, -2:], r_bound[:, -2:]), metric_node.scope)[0]
            end_t = perf_counter()
            metric_node.execution_time = end_t - start_t
        metric_prob /= node.total_points_num()

        return nanproduct(tag_prob, metric_prob), nanproduct(tag_prob, 1 if metric_prob > 0.01 else 0)

    elif isinstance(node, TSSum):
        if len(relevant_scope.intersection(node.scope)) == 0:
            return np.nan, np.nan

        points_llchildren = []
        series_llchildren = []
        for child in node.children:
            points_prob, series_prob = expectation_recursive(child, feature_scope, inverted_features, relevant_scope, evidence,
                                            node_expectation, node_likelihoods, series)
            points_llchildren.append(points_prob)
            series_llchildren.append(series_prob)

        points_prob, series_prob = np.nan, np.nan
        relevant_children_idx = np.where(np.isnan(points_llchildren) == False)[0]
        if len(relevant_children_idx) > 0:
            weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
            weighted_sum = sum(node.weights[j] * points_llchildren[j] for j in relevant_children_idx)
            points_prob = weighted_sum / weights_normalizer
        relevant_children_idx = np.where(np.isnan(series_llchildren) == False)[0]
        if len(relevant_children_idx) > 0:
            weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
            weighted_sum = sum(node.weights[j] * series_llchildren[j] for j in relevant_children_idx)
            series_prob = weighted_sum / weights_normalizer

        return points_prob, series_prob

    elif isinstance(node, Product):
        product = np.nan
        for child in node.children:
            if len(relevant_scope.intersection(child.scope)) > 0:
                factor = expectation_recursive(child, feature_scope, inverted_features, relevant_scope, evidence,
                                            node_expectation, node_likelihoods, series)
                product = nanproduct(product, factor)
        return product

    elif isinstance(node, Sum):
        if len(relevant_scope.intersection(node.scope)) == 0:
            return np.nan
        if hasattr(node, 'min_ts') and hasattr(node, 'max_ts'):
            ranges = evidence[0][-1].get_ranges()
            possible = False
            for rg in ranges:
                if node.min_ts <= rg[1] and node.max_ts >= rg[0]:
                    possible = True
                    break
            if not possible:
                return 0

        llchildren = [expectation_recursive(child, feature_scope, inverted_features, relevant_scope, evidence,
                                            node_expectation, node_likelihoods, series)
                      for child in node.children]

        relevant_children_idx = np.where(np.isnan(llchildren) == False)[0]

        if len(relevant_children_idx) == 0:
            return np.nan

        weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
        weighted_sum = sum(node.weights[j] * llchildren[j] for j in relevant_children_idx)

        return weighted_sum / weights_normalizer
    
    else:
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:

                feature_idx = feature_scope.index(node.scope[0])
                inverted = inverted_features[feature_idx]

                return node_expectation[t_node](node, evidence, inverted=inverted).item()
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        return node_likelihoods[type(node)](node, evidence, series=series).item()
