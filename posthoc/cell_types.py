"""Post hoc description of the cell-type labels that the N3 family permutes within (written 18 October 2026, after
the pre-set analysis). It describes the v783 input table only; no simulation, score or null draw is involved.

Usage, from the repository root: python posthoc/cell_types.py      (writes posthoc/cell_types.json)
Reads the v783 wiring table read-only, from the path that code/wiring.py uses, and reports: the number of distinct
cell-type labels; the neurons whose label is one of the harmonised table's coarse "(untyped)" fallback labels, which N3
treats as types like any other; the cell types with a single neuron; and the number of (presynaptic type, postsynaptic
type) blocks that hold at least one connection.
"""
import hashlib, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import wiring  # noqa: E402  (only for the table's path)


def main():
    path = Path(wiring.V783)
    z = np.load(path)
    types = np.array([str(x) for x in z['lookup_cell_type']])
    ct = z['cell_type'].astype(np.int64)
    names = types[ct]
    labels, counts = np.unique(names, return_counts=True)
    untyped = np.char.find(names.astype(str), '(untyped)') >= 0
    singleton_labels = labels[counts == 1]
    t = int(ct.max()) + 1
    blocks = np.unique(ct[z['pre']] * t + ct[z['post']])
    out = {
        'note': 'Post hoc description of the v783 cell-type labels used by N3; not part of the frozen protocol.',
        'table_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'neurons': int(names.size),
        'connections': int(z['pre'].size),
        'cell_type_labels': int(labels.size),
        'untyped_neurons': int(untyped.sum()),
        'untyped_labels': int(np.unique(names[untyped]).size),
        'untyped_by_label': {str(k): int(v) for k, v in zip(*np.unique(names[untyped], return_counts=True))},
        'singleton_types': int(singleton_labels.size),
        'type_blocks_with_connections': int(blocks.size),
    }
    dest = ROOT / 'posthoc' / 'cell_types.json'
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'untyped_by_label'}, indent=1))


if __name__ == '__main__':
    main()
