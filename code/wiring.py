"""Load measured wirings as lif.Network objects.

v630: Shiu et al.'s own model files (the earlier release, used for replication), data/shiu2024-repo/2023_03_23_completeness_630_final.csv (neuron order)
and 2023_03_23_connectivity_630_final.parquet (Presynaptic_Index, Postsynaptic_Index, 'Excitatory x Connectivity').
v783: FlyWire v783, the paper's primary wiring, data/v783/fafb783-connectome.npz, built by scripts/build_v783.py from pinned public
sources (Shiu et al.'s v783 connectivity table and the Bates/Lee lab compiled FlyWire metadata).
"""
from pathlib import Path
import numpy as np
import pandas as pd

from lif import Network

P1 = Path(__file__).resolve().parents[1]
REPO = P1 / 'data/shiu2024-repo'
V783 = P1 / 'data/v783/fafb783-connectome.npz'


def _attach_ids(net, ids):
    net.ids = np.asarray(ids, dtype=np.int64)
    net.index = {int(x): i for i, x in enumerate(net.ids)}
    return net


def load_v630_edges():
    comp = pd.read_csv(REPO / '2023_03_23_completeness_630_final.csv', index_col=0)
    con = pd.read_parquet(REPO / '2023_03_23_connectivity_630_final.parquet')
    ids = comp.index.values.astype(np.int64)
    pre = con['Presynaptic_Index'].values.astype(np.int64)
    post = con['Postsynaptic_Index'].values.astype(np.int64)
    signed = con['Excitatory x Connectivity'].values.astype(np.int64)
    return ids, pre, post, signed


def load_v630():
    ids, pre, post, signed = load_v630_edges()
    return _attach_ids(Network(ids.size, pre, post, signed), ids)


def load_v783():
    z = np.load(V783)
    ids = z['neuron_id'].astype(np.int64)
    signed = z['synapses'].astype(np.int64) * z['shiu_sign'].astype(np.int64)
    return _attach_ids(Network(ids.size, z['pre'], z['post'], signed), ids)
