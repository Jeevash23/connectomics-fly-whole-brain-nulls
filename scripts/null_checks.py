"""Save nulls.check() for draw 0 of every null family (kept statistics, self-loops and repeated pairs).

Usage, from the repository root: python scripts/null_checks.py [--out FILE]
  writes results/null_checks_draw0.json, or FILE; compare FILE with the shipped results/null_checks_draw0.json
Uses the frozen code unchanged; it builds wirings only and runs no simulation.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import nulls
import run_wiring
import wiring

out = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else ROOT / 'results/null_checks_draw0.json'
if out.exists():
    sys.exit(f'{out} exists; not overwriting')
res = {}
for version, load, families in (('v783', wiring.load_v783, ('N1', 'N2', 'N3', 'N4')),
                                 ('v630', wiring.load_v630, ('N1', 'N2'))):
    measured = load()
    for fam in families:
        null = nulls.make(version, measured, fam, 0)
        r = nulls.check(version, measured, null, fam)
        r['edges'] = int(null.post.size)
        r['wiring_sha256'] = run_wiring.wiring_hash(null)
        res[f'{version}/{fam}_00'] = r
        print(version, fam, r, flush=True)
out.write_text(json.dumps(res, indent=1))
