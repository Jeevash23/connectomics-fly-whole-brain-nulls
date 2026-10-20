"""Check a reproduction against the published runs in results/runs.csv.

Usage, from the repository root:
  python scripts/verify.py wirings --version v783 --runs M N1_00 N2_00 N3_00 N4_00
      rebuild these wirings (no simulation; seconds to minutes each) and compare their SHA-256 with the published runs
  python scripts/verify.py rates
      compare the rates.json of every run present under results/<version>/ with the published SHA-256
Run names are as in results/: M for the measured wiring, <family>_<draw> for a null draw (N1_07 is draw 7 of N1).
"""
import argparse, csv, hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def published():
    with open(ROOT / 'results/runs.csv') as f:
        rows = list(csv.DictReader(f))
    return {(r['version'], r['wiring'] if r['wiring'] == 'M' else f"{r['wiring']}_{int(r['draw']):02d}"): r for r in rows}


def check_wirings(version, runs, table):
    import nulls, run_wiring, wiring
    measured = wiring.load_v630() if version == 'v630' else wiring.load_v783()
    failures = 0
    for name in runs:
        row = table.get((version, name))
        if row is None:
            print(f'{version} {name}: not a published run'); failures += 1; continue
        if name == 'M':
            net = measured
        else:
            family, draw = name.split('_')
            net = nulls.make(version, measured, family, int(draw))
        got = run_wiring.wiring_hash(net)
        ok = got == row['wiring_sha256']
        failures += not ok
        print(f'{version} {name:6s} wiring {got[:16]}…  {"matches" if ok else "DIFFERS from " + row["wiring_sha256"][:16] + "…"}')
    return failures


def check_rates(table):
    failures = checked = 0
    for version in ('v783', 'v630'):
        base = ROOT / 'results' / version
        if not base.is_dir():
            continue
        for run in sorted(p for p in base.iterdir() if (p / 'rates.json').is_file()):
            row = table.get((version, run.name))
            got = hashlib.sha256((run / 'rates.json').read_bytes()).hexdigest()
            ok = row is not None and got == row['rates_sha256']
            checked += 1
            failures += not ok
            print(f'{version} {run.name:6s} rates  {got[:16]}…  {"matches" if ok else "DIFFERS or not published"}')
    if not checked:
        print('no rates.json found under results/<version>/')
    return failures


def main():
    ap = argparse.ArgumentParser(description='Compare a reproduction with the published runs.')
    sub = ap.add_subparsers(dest='command', required=True)
    w = sub.add_parser('wirings', help='rebuild wirings and compare their hashes')
    w.add_argument('--version', choices=['v630', 'v783'], required=True)
    w.add_argument('--runs', nargs='+', required=True)
    sub.add_parser('rates', help='compare rates.json files with the published hashes')
    a = ap.parse_args()
    table = published()
    failures = check_wirings(a.version, a.runs, table) if a.command == 'wirings' else check_rates(table)
    print('all match' if failures == 0 else f'{failures} mismatch(es)')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
