"""Download the pinned input files listed in data/sources.json and check each one's SHA-256.

Usage, from the repository root:
  python scripts/fetch_data.py              the files the simulations need (about 90 MB)
  python scripts/fetch_data.py --optional   also Shiu et al.'s supplementary spreadsheet (2.5 MB), needed only to
                                            rebuild data/derived/targets.csv or recount their published accuracy,
                                            and their original model.py and figures.ipynb, for reference
A file that is already present with the recorded hash is not downloaded again; a file with a different hash stops the
script, so a changed upstream file is never used silently. The v783 wiring has its own script: scripts/build_v783.py.
"""
import argparse, hashlib, json, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(entry):
    path = ROOT / entry['file']
    if path.exists():
        if sha256(path) == entry['sha256']:
            print(f'present  {entry["file"]}')
            return
        sys.exit(f'{entry["file"]} exists but its SHA-256 differs from data/sources.json. Delete it to download again.')
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'download {entry["file"]} ({entry["bytes"] / 1e6:.1f} MB)', flush=True)
    part = path.with_name(path.name + '.part')
    request = urllib.request.Request(entry['url'], headers={'User-Agent': 'connectome-null-wiring-benchmark'})
    with urllib.request.urlopen(request, timeout=300) as response, open(part, 'wb') as out:
        for block in iter(lambda: response.read(1 << 20), b''):
            out.write(block)
    if sha256(part) != entry['sha256']:
        part.unlink()
        sys.exit(f'{entry["file"]}: downloaded file does not match the recorded SHA-256; nothing was kept.')
    part.rename(path)


def main():
    ap = argparse.ArgumentParser(description='Download and check the pinned input files.')
    ap.add_argument('--optional', action='store_true', help="also fetch Shiu et al.'s supplementary spreadsheet")
    a = ap.parse_args()
    sources = json.loads((ROOT / 'data/sources.json').read_text())
    for entry in sources['files']:
        if entry.get('optional') and not a.optional:
            continue
        fetch(entry)
    print('all files present and checked')


if __name__ == '__main__':
    main()
