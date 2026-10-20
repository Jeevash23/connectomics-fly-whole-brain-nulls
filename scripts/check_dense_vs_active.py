"""Check that the simulator's two implementations give identical spike counts on the measured v783 wiring.

lif.py has a reference implementation that updates every neuron at every step (dense) and the faster one used for all
runs, which skips neurons at exact rest. This runs three experiments, shortened to 200 ms and 3 trials so that the
dense version finishes in minutes, and compares every neuron's spike count.

Usage, from the repository root: python scripts/check_dense_vs_active.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import experiments as E
import nulls
import wiring


def main():
    net = wiring.load_v783()
    exps, _ = E.build(E.Mapping(net, 'v783', mn9_other_index=nulls.mn9_other_v783(net)))
    failures = 0
    for name in ('sugarR_100', 'waterR_160_sil_Zorro', 'JON_160'):
        drive, silence = exps[name]
        kw = dict(silence=silence, n_trials=3, duration_ms=200.0, seed=E.seed_for(name))
        active = net.run(drive, **kw)
        dense = net.run(drive, dense=True, **kw)
        same = np.array_equal(active, dense)
        failures += not same
        print(f'{name:22s} spikes {int(active.sum()):8d}  neurons firing {int((active.sum(0) > 0).sum()):6d}  '
              f'{"identical" if same else "DIFFERENT"}')
    print('both implementations agree' if not failures else f'{failures} experiment(s) differ')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
