"""Post hoc (19 October 2026): summarise the seed runs made with seeds.py.

Reads posthoc/seeds/v783/<run>_seed<k>/rates.json (seed sets k >= 1) and the published runs (seed set 0, in
results/v783/), scores them, and writes posthoc/seeds.json:
  - the measured v783 wiring's balanced accuracy under seed sets 0-9;
  - N3 draws 0-9 under seed sets 0 and 1, and the measured-minus-N3 contrast over those ten draws for each seed set;
  - for those N3 draws, whether MN9 fires in the 160 Hz water control under each seed set.
Usage, from the repository root: python posthoc/seeds_summary.py
"""
import json, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'code'))
import score


def rates(version, tag, k):
    path = (ROOT / 'results' / version / tag) if k == 0 else (HERE / 'seeds' / version / f'{tag}_seed{k}')
    return json.loads((path / 'rates.json').read_text())


def bas(r):
    s = score.score_rates(r)
    return {k: s[k]['balanced_accuracy'] for k in ('all', 'screen', 'non_screen')}


def main():
    out = {'note': 'Post hoc seed runs (19 October 2026). Seed set 0 is the published run; seed sets 1-9 shift every '
                   "experiment's Poisson seed (seeds.py). Same code, wiring, 30 trials and rules."}
    m = {k: bas(rates('v783', 'M', k)) for k in range(10)}
    allv = [m[k]['all'] for k in range(10)]
    out['measured_v783'] = {'per_seed_set': m, 'mean': statistics.mean(allv), 'sd': statistics.stdev(allv),
                            'min': min(allv), 'max': max(allv)}
    n3 = {}
    for d in range(10):
        tag = f'N3_{d:02d}'
        r0, r1 = rates('v783', tag, 0), rates('v783', tag, 1)
        n3[tag] = {'seed0': bas(r0)['all'], 'seed1': bas(r1)['all'],
                   'water_control_fires_seed0': r0['waterR_160']['mean']['MN9'] > 0,
                   'water_control_fires_seed1': r1['waterR_160']['mean']['MN9'] > 0}
    diffs = [v['seed1'] - v['seed0'] for v in n3.values()]
    out['n3_draws_0_9'] = {
        'per_draw': n3, 'mean_seed0': statistics.mean(v['seed0'] for v in n3.values()),
        'mean_seed1': statistics.mean(v['seed1'] for v in n3.values()),
        'sd_of_seed_differences': statistics.stdev(diffs), 'max_abs_seed_difference': max(abs(x) for x in diffs),
        'water_control_status_changed': sum(v['water_control_fires_seed0'] != v['water_control_fires_seed1']
                                            for v in n3.values())}
    out['contrast_measured_minus_n3_draws_0_9'] = {
        'seed_set_0': m[0]['all'] - out['n3_draws_0_9']['mean_seed0'],
        'seed_set_1': m[1]['all'] - out['n3_draws_0_9']['mean_seed1']}
    (HERE / 'seeds.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'note'}, indent=1)[:3000])


if __name__ == '__main__':
    main()
