"""Post hoc (19 October 2026, after an independent review of the manuscript): rerun a wiring with a different set of
Poisson input seeds, to measure how much a score moves with input noise alone.

The frozen code gives every experiment one seed, derived from its name (experiments.seed_for); trial t uses that seed
plus t. Seed set k shifts every experiment's seed by k x 1,000,003 (modulo the same bound), so k = 0 is the published
run and k = 1, 2, ... are fresh, non-overlapping input sequences. Nothing else changes: same code, wirings, 30 trials,
prediction rules and scoring.

Usage, from the code folder:
  python ../posthoc/seeds.py --version v783 --wiring M --seed-set 3 --out ../posthoc/seeds/v783/M_seed3
Writes <dir>/rates.json and run.json exactly as run_wiring.py does, plus "seed_set" in run.json.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import experiments as E
import run_wiring

BOUND = 2 ** 31 - 64
STEP = 1_000_003


def main():
    args = sys.argv[1:]
    i = args.index('--seed-set')
    k = int(args[i + 1])
    del args[i:i + 2]
    base = E.seed_for
    E.seed_for = lambda name: (base(name) + STEP * k) % BOUND
    sys.argv = [sys.argv[0]] + args
    run_wiring.main()
    out = Path(args[args.index('--out') + 1])
    run = json.loads((out / 'run.json').read_text())
    run['seed_set'] = k
    (out / 'run.json').write_text(json.dumps(run, indent=1))


if __name__ == '__main__':
    main()
