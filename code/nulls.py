"""Null wirings. Every family keeps the neuron set and the number of edges; each draw has its own fixed seed.

N1 weight shuffle   permute the signed synapse counts across existing edges: topology kept, which edge has which
                    strength (and sign) removed. Our reading of Shiu et al.'s control, "connectivity weights were
                    shuffled randomly (while maintaining the global connectivity weight distribution)".
N2 degree-preserving  permute postsynaptic ends across all edges; each edge keeps its presynaptic neuron, synapse count
                    and sign. Keeps every neuron's out-degree, out-strength and sign, and in-degree (directed
                    configuration null, as in Lin et al., Nature 2024). Removes partners and cell-type structure.
N3 cell-type block  as N2, but postsynaptic ends are permuted only among edges with the same (presynaptic cell type,
                    postsynaptic cell type). Also keeps edge and synapse counts between every pair of cell types.
                    v783 only (needs cell types for every neuron).
N4 boundary         edges out of sensory neurons and edges into motor, descending or visceral/circulatory neurons stay
                    fixed; the remaining (interior) edges get N2's permutation among themselves (after Park 2026).
                    v783 only (needs super classes).
Self-loops and repeated pairs created by permutation are kept; their counts are reported by check().
"""
from pathlib import Path

import numpy as np

from lif import Network
import wiring

FAMILY_CODE = {'N1': 1, 'N2': 2, 'N3': 3, 'N4': 4}
VERSION_CODE = {'v630': 630, 'v783': 783}
SENSORY = {'sensory', 'sensory_ascending', 'sensory_descending'}
OUTPUT = {'motor', 'descending', 'visceral_circulatory'}


def labels_v783():
    z = np.load(wiring.V783)
    cell_type = z['cell_type'].astype(np.int64)
    super_class = np.array([str(x) for x in z['lookup_super_class']])[z['super_class']]
    types = np.array([str(x) for x in z['lookup_cell_type']])[z['cell_type']]
    return cell_type, super_class, types


def mn9_other_v783(net):
    """Index of the second MN9 on v783: the MN9-labelled neuron that is not 720575940660219265."""
    import inputs as I
    _, _, types = labels_v783()
    idx = [i for i in np.flatnonzero(types == 'MN9') if net.ids[i] != I.MN9]
    return int(idx[0]) if len(idx) == 1 else None


def _permute_within(groups, values, rng):
    """Permute values among positions with the same group key."""
    base = np.argsort(groups, kind='stable')
    shuffled = np.lexsort((rng.random(groups.size), groups))
    out = values.copy()
    out[base] = values[shuffled]
    return out


def make(version, measured, family, draw):
    if family in ('N3', 'N4') and version != 'v783':
        raise ValueError(f'{family} needs v783 labels')
    rng = np.random.default_rng([FAMILY_CODE[family], VERSION_CODE[version], int(draw)])
    pre, post, signed = measured.pre, measured.post, measured.signed
    if family == 'N1':
        new_pre, new_post, new_signed = pre, post, signed[rng.permutation(signed.size)]
    elif family == 'N2':
        new_pre, new_post, new_signed = pre, post[rng.permutation(post.size)], signed
    elif family == 'N3':
        cell_type, _, _ = labels_v783()
        t = int(cell_type.max()) + 1
        groups = cell_type[pre] * t + cell_type[post]
        new_pre, new_post, new_signed = pre, _permute_within(groups, post, rng), signed
    else:
        _, super_class, _ = labels_v783()
        fixed = np.isin(super_class[pre], list(SENSORY)) | np.isin(super_class[post], list(OUTPUT))
        groups = fixed.astype(np.int64)  # interior edges form one group; fixed edges are restored below
        new_post = _permute_within(groups, post, rng)
        new_post[fixed] = post[fixed]
        new_pre, new_signed = pre, signed
    net = Network(measured.n, new_pre, new_post, new_signed)
    net.ids, net.index = measured.ids, measured.index
    return net


def check(version, measured, null, family):
    """Statistics a family should keep (True) and how much wiring it changed."""
    n = measured.n
    def deg(a): return np.bincount(a, minlength=n)
    def strength(idx, w): return np.bincount(idx, weights=np.abs(w), minlength=n)
    key_m = measured.pre * n + measured.post
    key_n = null.pre * n + null.post
    res = {
        'out_degree_same': bool(np.array_equal(deg(measured.pre), deg(null.pre))),
        'in_degree_same': bool(np.array_equal(deg(measured.post), deg(null.post))),
        'out_strength_same': bool(np.allclose(strength(measured.pre, measured.signed), strength(null.pre, null.signed))),
        'in_strength_same': bool(np.allclose(strength(measured.post, measured.signed), strength(null.post, null.signed))),
        'weight_multiset_same': bool(np.array_equal(np.sort(measured.signed), np.sort(null.signed))),
        'pairs_retained': float(np.isin(key_n, key_m).mean()),
        'self_loops': int((null.pre == null.post).sum()),
        'repeated_pairs': int(key_n.size - np.unique(key_n).size),
    }
    if version == 'v783':
        cell_type, super_class, _ = labels_v783()
        t = int(cell_type.max()) + 1
        def blocks(net):
            return np.unique(cell_type[net.pre] * t + cell_type[net.post], return_counts=True)
        bm, bn = blocks(measured), blocks(null)
        res['type_block_counts_same'] = bool(np.array_equal(bm[0], bn[0]) and np.array_equal(bm[1], bn[1]))
        def boundary(net):
            f = np.isin(super_class[net.pre], list(SENSORY)) | np.isin(super_class[net.post], list(OUTPUT))
            return np.sort((net.pre * n + net.post)[f])
        res['boundary_edges_same'] = bool(np.array_equal(boundary(measured), boundary(null)))
    return res
