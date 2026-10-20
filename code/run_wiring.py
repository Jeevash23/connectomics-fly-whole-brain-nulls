"""Run every experiment on one wiring and save readout firing rates.

Usage (from the code folder):
  python run_wiring.py --version v630 --wiring M
  python run_wiring.py --version v783 --wiring N3 --draw 4
Output: results/<version>/<wiring>[_<draw>]/rates.json (mean and per-trial counts of readout neurons per experiment)
and run.json (timing, wiring hash, edge count). An existing rates.json is never overwritten; experiments already saved
in partial.json are skipped, so an interrupted run resumes.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments as E
import nulls
import wiring

P1 = Path(__file__).resolve().parents[1]


def wiring_hash(net):
    h = hashlib.sha256()
    for a in (net.pre, net.post, net.signed):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', choices=['v630', 'v783'], required=True)
    ap.add_argument('--wiring', choices=['M', 'N1', 'N2', 'N3', 'N4'], required=True)
    ap.add_argument('--draw', type=int, default=0)
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--only', default='', help='comma-separated experiment names (run only these)')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    tag = a.wiring if a.wiring == 'M' else f'{a.wiring}_{a.draw:02d}'
    out = Path(a.out) if a.out else P1 / 'results' / a.version / tag
    if (out / 'rates.json').exists():
        print('exists, skipping:', out); return
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    measured = wiring.load_v630() if a.version == 'v630' else wiring.load_v783()
    net = measured if a.wiring == 'M' else nulls.make(a.version, measured, a.wiring, a.draw)
    mapping = E.Mapping(net, a.version, mn9_other_index=nulls.mn9_other_v783(measured) if a.version == 'v783' else None)
    exps, readout = E.build(mapping)
    if a.only:
        exps = {k: v for k, v in exps.items() if k in a.only.split(',')}
    labels = sorted(readout)
    cols = np.array([readout[k] for k in labels])

    part_path = out / 'partial.json'
    done = json.loads(part_path.read_text()) if part_path.exists() else {}
    timing = {}
    for name, (drive, silence) in exps.items():
        if name in done:
            continue
        t = time.time()
        counts = net.run(drive, silence=silence, n_trials=a.trials, seed=E.seed_for(name))
        sel = counts[:, cols]
        done[name] = {'mean': {k: float(sel[:, j].mean()) for j, k in enumerate(labels)},
                      'trials': {k: sel[:, j].astype(int).tolist() for j, k in enumerate(labels)},
                      'n_driven': len(drive), 'n_silenced': len(silence),
                      'total_spikes': int(counts.sum()), 'neurons_active': int((counts.sum(0) > 0).sum())}
        timing[name] = round(time.time() - t, 2)
        part_path.write_text(json.dumps(done))
        print(f'{name:32s} {timing[name]:6.1f}s  MN9 {done[name]["mean"].get("MN9", float("nan")):7.2f}', flush=True)

    (out / 'rates.json').write_text(json.dumps(done))
    part_path.unlink(missing_ok=True)
    run = {'version': a.version, 'wiring': a.wiring, 'draw': a.draw, 'trials': a.trials, 'experiments': len(done),
           'neurons': net.n, 'edges': int(net.post.size), 'wiring_sha256': wiring_hash(net),
           'seconds_total': round(time.time() - t0, 1), 'seconds_per_experiment': timing,
           'threads': os.environ.get('NUMBA_NUM_THREADS'), 'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (out / 'run.json').write_text(json.dumps(run, indent=1))
    print('done', out, run['seconds_total'], 's')


if __name__ == '__main__':
    main()
