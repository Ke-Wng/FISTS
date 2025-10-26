import logging
from spn.structure.Base import Node
from spn.structure.Base import Leaf
from FSPN.Structure.leaves.parametric.Parametric import Categorical, Gaussian
from FSPN.Structure.nodes import Context
from structure.Multi_Histograms import Multi_histogram, create_multi_histogram_leaf
from pybloom_live import BloomFilter
import time

class Sum(Node):
    def __init__(self, weights=None, children=None, cluster_centers=None, cardinality=None):
        Node.__init__(self)

        if weights is None:
            weights = []
        self.weights = weights

        if children is None:
            children = []
        self.children = children

        if cluster_centers is None:
            cluster_centers = []
        self.cluster_centers = cluster_centers

        if cardinality is None:
            cardinality = 0
        self.cardinality = cardinality

    @property
    def parameters(self):
        sorted_children = sorted(self.children, key=lambda c: c.id)
        params = [(n.id, self.weights[i]) for i, n in enumerate(sorted_children)]
        return tuple(params)


class Product(Node):
    def __init__(self, children=None):
        Node.__init__(self)
        if children is None:
            children = []
        self.children = children
        self.cached_product = {}

    @property
    def parameters(self):
        return tuple(map(lambda n: n.id, sorted(self.children, key=lambda c: c.id)))


class TSProduct(Product):
    def __init__(self, series_ids=None):
        Product.__init__(self, None)
        self.children = []
        # TODO: introduce a sum node
        self.n_points = [] # used for calculating weights for metric nodes
        self.series_ids_bloom = BloomFilter(capacity=len(series_ids), error_rate=0.01)
        start_t = time.perf_counter()
        for sid in series_ids:
            self.series_ids_bloom.add(sid)
        end_t = time.perf_counter()
        logging.debug(f"Spend {(end_t - start_t):.2f} seconds to build bloom filter in TSProduct, {len(series_ids)} series, size {len(self.series_ids_bloom) / 1024:.2f} KB")
        self.series_ids = series_ids

    def add_tag_node(self, node):
        assert len(self.children) == 0
        assert isinstance(node, Product)
        self.children.append(node)

    def add_metric_ts_node(self, node, n_points):
        self.children.append(node)
        self.n_points.append(n_points)

    def get_tag_node(self):
        return self.children[0]

    def get_metric_ts_nodes(self):
        return zip(self.children[1:], self.n_points)

    def contains_series_id(self, sid):
        return sid in self.series_ids_bloom

    def total_points_num(self):
        return sum(self.n_points)


    def insert_data(self, data):
        assert data.shape[1] == len(self.scope)
        # TODO: check metric ditribution diverge here
        start_t = time.perf_counter()
        parametric_types = [Categorical] * (len(self.scope) - 2) + [Gaussian, Categorical]
        ds_context = Context(parametric_types=parametric_types).add_domains(data)
        metric_data = data[:, -2:].astype(int)
        metric_ts_leaf = create_multi_histogram_leaf(metric_data, ds_context, self.children[-1].scope, [], discretize=True)
        self.add_metric_ts_node(metric_ts_leaf, len(data))
        end_t = time.perf_counter()
        logging.debug(f"TSProduct inserted {len(data)} points, costed {end_t - start_t} seconds")



class TSSum(Sum):
    def __init__(self, weights=None, children=None, cluster_centers=None, cardinality=None):
        Sum.__init__(self, weights, children, cluster_centers, cardinality)