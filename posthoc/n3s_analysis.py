"""Analysis of the N3s runs, fixed before they ran (19 October 2026; see docs/DESIGN.md).

Reads posthoc/n3s/v783/N3s_<dd>/rates.json, the published measured and N3 runs (results/v783/), and writes
posthoc/n3s.json. Usage, from the repository root: python posthoc/n3s_analysis.py
"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(HERE))
import analyse
import analyses as A
import score

ITEMS, TRUTH, CAT = A.ITEMS, A.TRUTH, A.CAT
WATER = ['4C:Bract', '4C:Clavicle', '4C:Rattle', '4C:Roundup', '4C:Zorro']


def preds(rates):
    p = score.score_rates(rates)['predictions']
    return np.array([-1 if p.get(i) is None else p[i] for i in ITEMS])


def item_bootstrap():
    """The pre-set bootstrap weights, generated exactly as in the frozen analyse.py."""
    rng = np.random.default_rng(20261016)
    W = np.zeros((analyse.B, len(ITEMS)))
    for c in np.unique(CAT):
        idx = np.flatnonzero(CAT == c)
        draws = rng.integers(0, idx.size, size=(analyse.B, idx.size))
        for j in range(idx.size):
            np.add.at(W, (np.arange(analyse.B), idx[draws[:, j]]), 1)
    return W


def cluster_bootstrap():
    """The clustered bootstrap weights, generated exactly as in analyses.py."""
    rng = np.random.default_rng(20261019)
    B = 10_000
    clusters = np.array([A.cluster_of(it) for it in ITEMS])
    strata = {'screen': sorted({c for c, k in zip(clusters, CAT) if k == '2A'}),
              'other': sorted({c for c, k in zip(clusters, CAT) if k != '2A'})}
    W = np.zeros((B, len(ITEMS)))
    for names in strata.values():
        names = np.array(names)
        draws = rng.integers(0, len(names), size=(B, len(names)))
        for j in range(len(names)):
            picked = names[draws[:, j]]
            for c in np.unique(picked):
                rows = np.flatnonzero(picked == c)
                W[np.ix_(rows, np.flatnonzero(clusters == c))] += 1
    return W


def main():
    base = ROOT / 'results' / 'v783'
    runs = sorted((HERE / 'n3s' / 'v783').glob('N3s_*'))
    runs = [r for r in runs if (r / 'rates.json').exists()]
    draws = [int(r.name.split('_')[1]) for r in runs]
    R_s = [json.loads((r / 'rates.json').read_text()) for r in runs]
    R_n3 = [json.loads((base / f'N3_{d:02d}' / 'rates.json').read_text()) for d in draws]
    m = preds(json.loads((base / 'M' / 'rates.json').read_text()))
    S = np.array([preds(x) for x in R_s])
    N3 = np.array([preds(x) for x in R_n3])
    sets = {'all': np.ones(len(ITEMS), bool), 'screen': CAT == '2A', 'non_screen': CAT != '2A'}
    out = {'note': 'Post hoc N3s check, analysis fixed before the runs (see docs/DESIGN.md).', 'draws': len(runs),
           'measured': {k: float(A.ba(m, mask)[0]) for k, mask in sets.items()}, 'by_set': {}}
    Wi, Wc = item_bootstrap(), cluster_bootstrap()
    for k, mask in sets.items():
        per_s = A.ba(S, mask.astype(float))
        per_3 = A.ba(N3, mask.astype(float))
        bm_i = analyse.ba_weighted(m[None, :], TRUTH, Wi * mask)
        bs_i = np.mean([analyse.ba_weighted(p[None, :], TRUTH, Wi * mask) for p in S], axis=0)
        d_i = bm_i - bs_i
        res = {'n3s_per_draw': per_s.tolist(), 'n3s_mean': float(per_s.mean()), 'n3s_min': float(per_s.min()),
               'n3s_max': float(per_s.max()), 'n3_mean_same_draws': float(per_3.mean()),
               'measured_minus_n3s': float(A.ba(m, mask)[0] - per_s.mean()),
               'ci95_item_bootstrap': [float(np.nanpercentile(d_i, 2.5)), float(np.nanpercentile(d_i, 97.5))],
               'draws_at_or_above_measured': int((per_s >= A.ba(m, mask)[0]).sum()),
               'paired_n3s_minus_n3_mean': float((per_s - per_3).mean()),
               'paired_higher': int((per_s > per_3).sum()), 'paired_equal': int((per_s == per_3).sum()),
               'paired_lower': int((per_s < per_3).sum())}
        if k == 'all':
            bm_c = A.ba(m, Wc)
            bs_c = np.mean([A.ba(p, Wc) for p in S], axis=0)
            d_c = bm_c - bs_c
            res['ci95_cluster_bootstrap'] = [float(np.nanpercentile(d_c, 2.5)), float(np.nanpercentile(d_c, 97.5))]
        out['by_set'][k] = res
    out['water_control_silent'] = {'n3s': int(sum(x['waterR_160']['mean']['MN9'] == 0 for x in R_s)),
                                   'n3_same_draws': int(sum(x['waterR_160']['mean']['MN9'] == 0 for x in R_n3))}
    out['mean_yes_predictions'] = {'n3s': float((S == 1).sum(1).mean()), 'n3_same_draws': float((N3 == 1).sum(1).mean())}
    out['water_items_correct'] = {it: {'n3s': int((S[:, ITEMS.index(it)] == 1).sum()),
                                       'n3_same_draws': int((N3[:, ITEMS.index(it)] == 1).sum())} for it in WATER}
    out['repairs'] = [json.loads((r / 'run.json').read_text()).get('repair') for r in runs]
    (HERE / 'n3s.json').write_text(json.dumps(out, indent=1) + '\n')
    a = out['by_set']['all']
    print(f"N3s: {len(runs)} draws; mean {a['n3s_mean']:.3f} (N3 same draws {a['n3_mean_same_draws']:.3f}); measured minus "
          f"N3s {a['measured_minus_n3s']:+.3f} [{a['ci95_item_bootstrap'][0]:+.3f}, {a['ci95_item_bootstrap'][1]:+.3f}]; "
          f"paired mean {a['paired_n3s_minus_n3_mean']:+.3f}; water control silent {out['water_control_silent']}")


if __name__ == '__main__':
    main()
