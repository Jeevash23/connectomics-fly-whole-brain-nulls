"""Post hoc analyses of the saved outputs (19 October 2026), added after the pre-set analysis was complete.

None of these was in the frozen protocol; all are labelled post hoc wherever they are reported. They read the saved
rates.json of every published run (results/<version>/<run>/, from the archive or a full rerun) and code/; nothing is
simulated or changed.

  1. How often each wiring predicts "yes", and how many neurons respond to 50 Hz sugar input.
  2. The N3 contrast split item by item and by category.
  3. Water and sugar requirement items, split by whether MN9 fires in the experiment's control run.
  4. A cluster bootstrap: items that share a stimulus or control run are resampled together.
  5. The 0.8 "required" threshold moved to 0.7 and 0.9.
  6. Descriptive facts about the v783 wiring and items (input populations present, coarse labels, item survival).

Usage, from the repository root: python posthoc/analyses.py     writes posthoc/posthoc.json
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import experiments as E
import score

OUT = Path(__file__).resolve().parent / 'posthoc.json'
TARGETS = E.load_targets()
ITEMS = list(TARGETS)
TRUTH = np.array([int(TARGETS[i]['truth']) for i in ITEMS])
CAT = np.array([TARGETS[i]['category'] for i in ITEMS])
REQUIRED = {'1F': ('sugarR_50', 'sugarR_50_sil_{}', 'MN9'), '4C': ('waterR_160', 'waterR_160_sil_{}', 'MN9'),
            '5D': ('JON_160', 'JON_160_sil_{}', 'aDN1')}


def runs(version, family):
    base = ROOT / 'results' / version
    return [base / 'M'] if family == 'M' else sorted(base.glob(f'{family}_*'))


def load(run):
    return json.loads((run / 'rates.json').read_text())


def preds(rates, required_threshold=0.8):
    p = score.score_rates(rates)['predictions']
    if required_threshold != 0.8:
        for item in ITEMS:
            cat, name = item.split(':', 1)
            if cat in REQUIRED and p.get(item) is not None:
                ctrl, sil, target = REQUIRED[cat]
                c = rates[ctrl]['mean'][target]
                s = rates[sil.format(name)]['mean'][target]
                p[item] = int(c > 0 and s <= required_threshold * c)
    return np.array([-1 if p.get(i) is None else p[i] for i in ITEMS])


def ba(P, w=None):
    """Balanced accuracy of prediction rows P (draws x items; -1 = not measurable), optionally with item weights."""
    P = np.atleast_2d(P)
    w = np.ones(len(ITEMS)) if w is None else w
    ok = P >= 0
    pos, neg = ok & (TRUTH == 1), ok & (TRUTH == 0)
    hit = (w * (pos & (P == 1))).sum(-1) / (w * pos).sum(-1)
    cr = (w * (neg & (P == 0))).sum(-1) / (w * neg).sum(-1)
    return 0.5 * (hit + cr)


def cluster_of(item):
    cat, name = item.split(':', 1)
    if cat == '2A' or cat == '5I':
        return item                                   # each screen type and each 5I item has its own run
    if cat == '1C' or (cat == '3BC' and 'weak' in name):
        return 'sugar_100'                            # 1C and the weak-sugar 3BC items share the 100 Hz sugar run
    if cat == '1D' or cat == '3BC':
        return 'sugar_200'                            # 1D and the strong-sugar 3BC items share the 200 Hz sugar run
    return cat                                        # 1F, 4A, 4C, 5B, 5D: one shared stimulus or control run each


def main():
    out = {'note': 'Post hoc analyses (19 October 2026); not part of the frozen protocol.'}
    P, R = {}, {}
    for version, fams in (('v783', ('M', 'N1', 'N2', 'N3', 'N4')), ('v630', ('M', 'N1', 'N2'))):
        for f in fams:
            rr = [load(r) for r in runs(version, f)]
            R[version, f] = rr
            P[version, f] = np.array([preds(x) for x in rr])
    measurable = P['v783', 'M'][0] >= 0

    # 1. "yes" predictions and responding neurons
    out['yes_predictions'] = {}
    for (version, f), M in P.items():
        yes = (M == 1).sum(1)
        active = [x['sugarR_50']['neurons_active'] for x in R[version, f]]
        out['yes_predictions'][f'{version}/{f}'] = {
            'items_measurable': int((M[0] >= 0).sum()), 'mean_yes': float(yes.mean()), 'min_yes': int(yes.min()),
            'max_yes': int(yes.max()), 'mean_neurons_active_sugar50': float(np.mean(active)),
            'mn9_fires_sugar50_draws': int(sum(x['sugarR_50']['mean']['MN9'] > 0 for x in R[version, f])),
            'draws': len(M)}
    out['truth_yes_measurable_v783'] = int(((TRUTH == 1) & measurable).sum())

    # 2. N3 contrast by item and category (v783, all measurable items)
    m, n3 = P['v783', 'M'][0], P['v783', 'N3']
    n_yes, n_no = int(((TRUTH == 1) & measurable).sum()), int(((TRUTH == 0) & measurable).sum())
    wt = np.where(TRUTH == 1, 1 / (2 * n_yes), 1 / (2 * n_no)) * measurable
    contrib = wt * ((m == TRUTH).astype(float) - (n3 == TRUTH).mean(0))
    water = [i for i, it in enumerate(ITEMS) if it in ('4C:Bract', '4C:Clavicle', '4C:Rattle', '4C:Roundup', '4C:Zorro')]
    by_cat = {c: float(contrib[CAT == c].sum()) for c in sorted(set(CAT))}
    out['n3_contrast_split'] = {
        'total': float(contrib.sum()), 'five_water_items': float(contrib[water].sum()),
        'all_other_items': float(contrib.sum() - contrib[water].sum()), 'by_category': by_cat,
        'one_yes_item_weight': 1 / (2 * n_yes), 'one_no_item_weight': 1 / (2 * n_no)}

    # 3. requirement items split by whether MN9 (aDN1 for 5D) fires in the control run
    out['control_split'] = {}
    for f in ('N3', 'N4'):
        for cat in ('4C', '1F'):
            ctrl, _, target = REQUIRED[cat]
            fires = np.array([x[ctrl]['mean'][target] > 0 for x in R['v783', f]])
            Pm = P['v783', f]
            items = [i for i, it in enumerate(ITEMS) if CAT[i] == cat and TRUTH[i] == 1 and m[i] == 1]
            out['control_split'][f'{f}/{cat}'] = {
                'draws': len(fires), 'control_silent': int((~fires).sum()),
                'measured_control_rate': float(R['v783', 'M'][0][ctrl]['mean'][target]),
                'items': {ITEMS[i]: {'correct_all_draws': int((Pm[:, i] == 1).sum()),
                                     'correct_when_control_fires': int((Pm[fires, i] == 1).sum()),
                                     'draws_control_fires': int(fires.sum())} for i in items}}

    # 4. cluster bootstrap of measured minus family mean, strata: screen items and the other items' clusters
    rng = np.random.default_rng(20261019)
    B = 10_000
    clusters = np.array([cluster_of(it) for it in ITEMS])
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
    out['cluster_bootstrap'] = {'clusters_screen': len(strata['screen']), 'clusters_other': len(strata['other']),
                                'other_clusters': strata['other'], 'resamples': B, 'families': {}}
    bm = ba(m, W)
    for f in ('N1', 'N2', 'N3', 'N4'):
        bn = np.mean([ba(p, W) for p in P['v783', f]], axis=0)
        d = bm - bn
        out['cluster_bootstrap']['families'][f] = {
            'measured_minus_mean': float(ba(m)[0] - ba(P['v783', f]).mean()),
            'ci95': [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}

    # 5. "required" threshold at 0.7 and 0.9
    out['threshold_sensitivity'] = {}
    for thr in (0.7, 0.8, 0.9):
        mt = preds(R['v783', 'M'][0], thr)
        res = {'measured': float(ba(mt)[0])}
        for f in ('N3', 'N4'):
            ft = np.array([preds(x, thr) for x in R['v783', f]])
            res[f'{f}_mean'] = float(ba(ft).mean())
            res[f'measured_minus_{f}'] = float(ba(mt)[0] - ba(ft).mean())
        out['threshold_sensitivity'][str(thr)] = res

    # 6. descriptive facts
    a = json.loads((ROOT / 'results/v783/analysis.json').read_text())
    surv = a['item_survival']['N3']
    errors = [it for i, it in enumerate(ITEMS) if measurable[i] and m[i] != TRUTH[i]]
    out['measured_errors'] = {'count': len(errors),
                              'n3_majority_correct': sorted(e for e in errors if surv[e] is not None and surv[e] > 0.5),
                              'n3_some_correct': sorted(e for e in errors if surv[e])}
    out['n3_1C_right_sugar_survival'] = surv['1C:sugarR_contra_gt_ipsi']
    out['n3_best_draw'] = a['families']['N3']['by_set']['all']['family_max']
    import inputs as I
    import wiring
    net = wiring.load_v783()
    present = lambda ids: sum(int(x) in net.index for x in ids)
    out['inputs_present_v783'] = {k: [present(ids), len(ids)] for k, ids in
                                  (('sugar_right', I.SUGAR_R), ('sugar_left', I.SUGAR_L), ('water', I.WATER),
                                   ('johnstons_organ', I.JON_ALL))}
    z = np.load(wiring.V783)
    labels = np.array([str(x) for x in z['lookup_cell_type']])[z['cell_type']]
    untyped = np.char.endswith(labels.astype(str), '(untyped)')
    touch = untyped[z['pre']] | untyped[z['post']]
    out['coarse_labels_v783'] = {'labels': int(len({l for l in labels[untyped]})), 'cells': int(untyped.sum()),
                                 'connections_touching': int(touch.sum()), 'connections_total': int(len(touch)),
                                 'cells_with_any_connection': int(len(np.unique(np.concatenate(
                                     [z['pre'][untyped[z['pre']]], z['post'][untyped[z['post']]]]))))}
    # 7. added after a second review: MN9 at 100 Hz sugar (the rate of Shiu et al.'s shuffle control), screen types
    #    partly present on v783, the size of FlyWire's cell types and blocks, what draw 0 of each family keeps and merges,
    #    contributions of items chosen with and without the model, and the standard error of each family mean
    for (version, f), rr in R.items():
        out['yes_predictions'][f'{version}/{f}']['mn9_fires_sugar100_draws'] = int(
            sum(x['sugarR_100']['mean']['MN9'] > 0 for x in rr))
    present_counts = [sum(int(x) in net.index for x in ids) for ids in I.screen_types().values()]
    out['screen_types_v783'] = {'types': len(present_counts),
                                'absent': sum(c == 0 for c in present_counts),
                                'partly_present': sum(0 < c < len(ids) for c, ids in
                                                      zip(present_counts, I.screen_types().values()))}
    cells_per_label = np.bincount(z['cell_type'])
    block = z['cell_type'][z['pre']].astype(np.int64) * len(cells_per_label) + z['cell_type'][z['post']]
    _, inverse, block_size = np.unique(block, return_inverse=True, return_counts=True)
    out['cell_types_v783'] = {'labels': int(len(cells_per_label)), 'median_cells_per_label': float(np.median(cells_per_label)),
                              'labels_with_two_cells': int((cells_per_label == 2).sum()),
                              'connections_alone_in_block': float((block_size[inverse] == 1).mean())}
    import nulls
    key_m = net.pre.astype(np.int64) * net.n + net.post
    w_m = np.abs(net.signed).astype(float)
    out['draw0_structure_v783'] = {}
    for f in ('N2', 'N3', 'N4'):
        null = nulls.make('v783', net, f, 0)
        key = null.pre.astype(np.int64) * null.n + null.post
        w = np.abs(null.signed).astype(float)
        kept = np.isin(key, key_m)
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        merged = cnt[inv] > 1
        loops = null.pre == null.post
        out['draw0_structure_v783'][f] = {
            'connections_on_measured_pairs': float(kept.mean()), 'synapses_on_measured_pairs': float(w[kept].sum() / w.sum()),
            'connections_in_repeated_pairs': float(merged.mean()), 'synapses_in_repeated_pairs': float(w[merged].sum() / w.sum()),
            'connections_self_loops': float(loops.mean()), 'synapses_self_loops': float(w[loops].sum() / w.sum())}
    chosen = [i for i, it in enumerate(ITEMS) if CAT[i] == '4C' or it in ('4A:Fudog', '4A:Zorro')]
    out['n3_contrast_split']['items_chosen_with_model'] = float(contrib[chosen].sum())
    out['n3_contrast_split']['items_chosen_without_model'] = float(contrib.sum() - contrib[chosen].sum())
    out['family_mean_standard_error_v783'] = {
        f: float(np.std(a['families'][f]['by_set']['all']['per_draw'], ddof=1)
                 / np.sqrt(len(a['families'][f]['by_set']['all']['per_draw']))) for f in ('N1', 'N2', 'N3', 'N4')}
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
