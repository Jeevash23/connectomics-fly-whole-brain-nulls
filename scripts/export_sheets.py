"""Export every sheet of Shiu et al.'s supplementary spreadsheet to CSV, using the standard library only.

Checks the spreadsheet against data/sources.json, then writes one CSV per sheet to
data/derived/shiu2024-sheets/ with a manifest of file names and SHA-256 hashes. Cells keep their stored text.
Usage, from the repository root: python3 -I scripts/export_sheets.py
"""
from pathlib import Path
import csv, hashlib, json, re, zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data/shiu2024-supplement'
XLSX = SRC / '41586_2024_7763_MOESM2_ESM.xlsx'
OUT = ROOT / 'data/derived/shiu2024-sheets'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}

expected = next(f for f in json.loads((ROOT / 'data/sources.json').read_text())['files'] if f['file'].endswith(XLSX.name))['sha256']
assert hashlib.sha256(XLSX.read_bytes()).hexdigest() == expected, 'spreadsheet hash differs from data/sources.json'

book = zipfile.ZipFile(XLSX)
shared = [''.join(t.text or '' for t in si.iter('{%s}t' % NS['m']))
          for si in ET.fromstring(book.read('xl/sharedStrings.xml')).findall('m:si', NS)]
targets = {r.get('Id'): r.get('Target') for r in ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))}


def column(ref):
    n = 0
    for ch in re.match(r'[A-Z]+', ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


OUT.mkdir(parents=True, exist_ok=True)
manifest = []
for i, sheet in enumerate(ET.fromstring(book.read('xl/workbook.xml')).find('m:sheets', NS)):
    name = sheet.get('name')
    path = 'xl/' + targets[sheet.get('{%s}id' % NS['r'])].removeprefix('/xl/').removeprefix('xl/')
    rows = []
    for row in ET.fromstring(book.read(path)).iter('{%s}row' % NS['m']):
        values = {}
        for cell in row.findall('m:c', NS):
            kind, v = cell.get('t'), cell.find('m:v', NS)
            if kind == 'inlineStr':
                values[column(cell.get('r'))] = ''.join(x.text or '' for x in cell.iter('{%s}t' % NS['m']))
            elif v is not None:
                values[column(cell.get('r'))] = shared[int(v.text)] if kind == 's' else v.text
        if values:
            rows.append([values.get(c, '') for c in range(max(values) + 1)])
    width = max((len(r) for r in rows), default=0)
    file = OUT / f"{i:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}.csv"
    with file.open('w', newline='') as f:
        csv.writer(f).writerows(r + [''] * (width - len(r)) for r in rows)
    manifest.append({'sheet': name, 'file': file.name, 'rows': len(rows),
                     'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
(OUT / 'manifest.json').write_text(json.dumps({'source_sha256': expected, 'sheets': manifest}, indent=2) + '\n')
print(f'{len(manifest)} sheets written to {OUT.relative_to(ROOT)}')
