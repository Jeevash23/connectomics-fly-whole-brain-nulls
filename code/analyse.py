"""Compare the measured wiring with each null family, as fixed in the frozen protocol (docs/DESIGN.md).

Usage: python analyse.py v783      (writes results/<version>/analysis.json and prints a summary)
For each null family: balanced accuracy of every draw, the family mean, and the contrast measured minus family mean,
for all items, the screen and the other items. Intervals: 10,000 item bootstraps, resampling items within each
category, with the null draws fixed; each bootstrap recomputes balanced accuracy for the measured wiring and every draw.
Also: for each item, the share of draws in each family that predict it correctly.
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments as E
import score

P1 = Path(__file__).resolve().parents[1]
B = 10_000


def ba(pred, truth, mask):
    """Balanced accuracy over items in mask; pred, truth arrays of 0/1 (pred may be -1 for not measurable)."""
    ok = mask & (pred >= 0)
    pos, neg = ok & (truth == 1), ok & (truth == 0)
    if not pos.any() or not neg.any():
        return np.nan
    return 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean())


def ba_weighted(pred, truth, w):
    """Balanced accuracy with item weights (bootstrap multiplicities); pred may be -1 for not measurable."""
    ok = (pred >= 0)
    pos, neg = ok & (truth == 1), ok & (truth == 0)
    wp, wn = (w * pos).sum(axis=-1), (w * neg).sum(axis=-1)
    hit = (w * (pos & (pred == 1))).sum(axis=-1) / wp
    cr = (w * (neg & (pred == 0))).sum(axis=-1) / wn
    return 0.5 * (hit + cr)


def main(version):
    base = P1 / 'results' / version
    targets = E.load_targets()
    items = list(targets)
    truth = np.array([int(targets[i]['truth']) for i in items])
    cats = np.array([targets[i]['category'] for i in items])
    sets = {'all': np.ones(len(items), bool), 'screen': cats == '2A', 'non_screen': cats != '2A'}

    def preds_of(run):
        p = score.score_rates(json.loads((run / 'rates.json').read_text()))['predictions']
        return np.array([-1 if p.get(i) is None else p[i] for i in items])

    measured = preds_of(base / 'M')
    families = {}
    for run in sorted(base.iterdir()):
        if run.is_dir() and run.name != 'M' and (run / 'rates.json').exists():
            families.setdefault(run.name.split('_')[0], []).append((run.name, preds_of(run)))

    rng = np.random.default_rng(20261016)
    # stratified item bootstrap: multiplicity weights per item
    W = np.zeros((B, len(items)))
    for c in np.unique(cats):
        idx = np.flatnonzero(cats == c)
        draws = rng.integers(0, idx.size, size=(B, idx.size))
        for j in range(idx.size):
            np.add.at(W, (np.arange(B), idx[draws[:, j]]), 1)

    out = {'version': version, 'items': len(items), 'measured': {}, 'families': {}, 'item_survival': {}}
    for sname, mask in sets.items():
        out['measured'][sname] = float(ba(measured, truth, mask))
    for fam, runs in sorted(families.items()):
        fo = {'draws': [r for r, _ in runs], 'by_set': {}}
        P = np.array([p for _, p in runs])
        for sname, mask in sets.items():
            per_draw = [float(ba(p, truth, mask)) for p in P]
            wmask = W * mask
            bm = ba_weighted(measured[None, :], truth, wmask)
            bn = np.mean([ba_weighted(p[None, :], truth, wmask) for p in P], axis=0)
            diff = bm - bn
            fo['by_set'][sname] = {
                'per_draw': per_draw, 'family_mean': float(np.mean(per_draw)),
                'family_min': float(np.min(per_draw)), 'family_max': float(np.max(per_draw)),
                'measured_minus_mean': float(out['measured'][sname] - np.mean(per_draw)),
                'ci95': [float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))],
                'draws_at_or_above_measured': int(sum(d >= out['measured'][sname] for d in per_draw))}
        surv = {}
        for k, i in enumerate(items):
            ok = [(p[k] == truth[k]) for p in P if p[k] >= 0]
            surv[i] = None if not ok else float(np.mean(ok))
        out['item_survival'][fam] = surv
        out['families'][fam] = fo
    (base / 'analysis.json').write_text(json.dumps(out, indent=1))
    print(f"{version}: measured balanced accuracy  all {out['measured']['all']:.3f}  screen {out['measured']['screen']:.3f}  "
          f"non-screen {out['measured']['non_screen']:.3f}")
    for fam, fo in out['families'].items():
        s = fo['by_set']
        print(f"  {fam} ({len(fo['draws'])} draws): " + '  '.join(
            f"{k} mean {s[k]['family_mean']:.3f} diff {s[k]['measured_minus_mean']:+.3f} [{s[k]['ci95'][0]:+.3f}, {s[k]['ci95'][1]:+.3f}]"
            for k in ('all', 'screen', 'non_screen')))


if __name__ == '__main__':
    main(sys.argv[1])
