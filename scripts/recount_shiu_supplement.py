"""Recount Shiu et al. (Nature 2024) headline accuracy from their supplementary spreadsheet.

Reads data/shiu2024-supplement/41586_2024_7763_MOESM2_ESM.xlsx with the standard library only.
Prints Supplementary Table 10 (the 164 tested predictions by category) with its totals, and the
yes/no table of the optogenetic screen (Supplementary Table 3, 106 cell types) at several model cut-offs.

Usage, from the repository root: python3 -I scripts/recount_shiu_supplement.py
"""
from pathlib import Path
import hashlib, json, re, zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/shiu2024-supplement'
XLSX = DATA / '41586_2024_7763_MOESM2_ESM.xlsx'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}

record = next(f for f in json.loads((ROOT / 'data/sources.json').read_text())['files'] if f['file'].endswith(XLSX.name))
assert hashlib.sha256(XLSX.read_bytes()).hexdigest() == record['sha256'], 'spreadsheet hash differs from data/sources.json'

book = zipfile.ZipFile(XLSX)
shared = [''.join(t.text or '' for t in si.iter('{%s}t' % NS['m']))
          for si in ET.fromstring(book.read('xl/sharedStrings.xml')).findall('m:si', NS)]
targets = {r.get('Id'): r.get('Target') for r in ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))}
sheets = {s.get('name'): 'xl/' + targets[s.get('{%s}id' % NS['r'])].removeprefix('/xl/').removeprefix('xl/')
          for s in ET.fromstring(book.read('xl/workbook.xml')).find('m:sheets', NS)}


def column(ref):
    n = 0
    for ch in re.match(r'[A-Z]+', ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def rows(prefix):
    path = next(p for name, p in sheets.items() if name.startswith(prefix))
    for row in ET.fromstring(book.read(path)).iter('{%s}row' % NS['m']):
        values = {}
        for cell in row.findall('m:c', NS):
            v = cell.find('m:v', NS)
            if v is not None:
                values[column(cell.get('r'))] = shared[int(v.text)] if cell.get('t') == 's' else v.text
        yield values


print('Supplementary Table 10: tested predictions by category')
correct = total = screen_correct = screen_total = 0
for r in list(rows('Supp Table 10'))[1:]:
    name, figure = r.get(0, ''), r.get(1, '')
    if not name or name.startswith('Total') or not r.get(3, '').isdigit():
        continue
    c, t = int(r[2]), int(r[3])
    correct += c; total += t
    if figure.strip() == '2A':
        screen_correct, screen_total = c, t
    print(f'  {c:3d}/{t:3d}  {figure:22s} {name[:80]}')
print(f'  total {correct}/{total} = {correct / total:.4f}')
print(f'  without the screen (figure 2A) {correct - screen_correct}/{total - screen_total} = '
      f'{(correct - screen_correct) / (total - screen_total):.4f}')

table = list(rows('Sup Table 3'))
head = table[0]
first = {head[k]: k for k in sorted(head, reverse=True)}  # first occurrence: means, not standard deviations
left, right = first['50 Hz MN9_Left'], first['50 Hz MN9_Right']
screen = [(r[0], float(r.get(1) or 0), max(float(r.get(left) or 0), float(r.get(right) or 0)))
          for r in table[1:] if r.get(0)]
print(f'\nOptogenetic screen (Supplementary Table 3): {len(screen)} cell types; experiment "yes" = activation rate > 0')
for cut in (0.0, 0.05, 0.1, 0.3, 0.4, 1.0):
    tp = sum(e > 0 and m > cut for _, e, m in screen); fn = sum(e > 0 and m <= cut for _, e, m in screen)
    fp = sum(e == 0 and m > cut for _, e, m in screen); tn = sum(e == 0 and m <= cut for _, e, m in screen)
    print(f'  model "yes" if MN9 > {cut:<4} Hz at 50 Hz: hits {tp}, misses {fn}, false alarms {fp}, correct no {tn}; '
          f'accuracy {(tp + tn) / len(screen):.3f}, balanced {(tp / (tp + fn) + tn / (tn + fp)) / 2:.3f}')
negatives = sum(e == 0 for _, e, _ in screen)
print(f'  always "no": accuracy {negatives / len(screen):.3f}, balanced 0.500')
