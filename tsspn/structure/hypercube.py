import numpy as np
from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import Type, MetaType

class HyperCube(Leaf):
    def __init__(self, data, scope):
        super().__init__(scope)
        self.data = data
        self.scope = scope

    def query(self, query):
        low = query[0]
        high = query[1]
        mask = np.all((self.data >= low) & (self.data <= high), axis=1)
        return np.sum(mask) / len(self.data)

def hypercube_likelihood_range(node, data, dtype=np.float64, **kwargs):
    assert data.shape[0] == 1
    ranges = data[0]
    l_bound = np.full((1, len(node.scope)), -np.inf)
    r_bound = np.full((1, len(node.scope)), np.inf)
    for i, scope in enumerate(node.scope):
        r = ranges[scope]
        if r is not None:
            if isinstance(r.get_ranges(), list):
                l_bound[0][i] = r.get_ranges()[0][0]
                r_bound[0][i] = r.get_ranges()[0][1]
            else:
                l_bound[0][i] = r.get_ranges()[0]
                r_bound[0][i] = l_bound[0][i]
    return node.query((l_bound, r_bound))


def create_hypercube_leaf(data, ds_context, scope):
    return HyperCube(data, scope)

    
