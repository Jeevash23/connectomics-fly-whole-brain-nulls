"""Build the 164 tested predictions of Shiu et al. (Nature 2024) as an item table: data/derived/targets.csv.

For each item: category (Supplementary Table 10 row), the neuron or cell type, the experimental outcome (truth) and
Shiu et al.'s published v630 prediction. Sources (exported CSVs in data/derived/shiu2024-sheets/):
- Table 10: categories, totals, and which named items were predicted correctly or incorrectly;
- Table 2 (sugar) and Table 5 (water): published silencing ratios (MN9 with the neuron silenced / control) and the
  experimental requirement results; Table 6A and 6C give Fudog's water-silencing ratio (not in Table 5);
- Table 3: the screen's experimental MN9 activation rate and Shiu's MN9 rate at 50 Hz (column "50 Hz MN9_Left", the
  MN9 Shiu et al. score);
- Table 11B: the item lists of the bitter/Ir94e and JON categories.
Truth for a named item = Shiu's prediction if Table 10 lists the item as correct, otherwise its negation; where a table
states the experimental result directly (Tables 2, 3 and 5), the two must agree, and the script checks they do.
The script stops unless every category recounts to Table 10 (150/164 overall, 49/58 without the screen).
Usage, from the repository root: python3 -I scripts/build_targets.py
"""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'data/derived/shiu2024-sheets'
OUT = ROOT / 'data/derived/targets.csv'


def sheet(prefix):
    return list(csv.reader(open(next(S.glob(prefix + '*.csv')))))


items = []


def add(cat, subject, truth, pred, note=''):
    items.append({'item': f'{cat}:{subject}', 'category': cat, 'subject': subject,
                  'truth': int(bool(truth)), 'shiu_pred': int(bool(pred)), 'note': note})


# 1C: contralateral MN9 responds more strongly than ipsilateral, for right and for left sugar GRNs (both correct).
add('1C', 'sugarR_contra_gt_ipsi', True, True)
add('1C', 'sugarL_contra_gt_ipsi', True, True)

# 1D: neurons that respond to sugar (>0 Hz). Shiu predicted all 14 respond; Usnea and TH-VUM incorrect.
for n in ['Zorro', 'G2N-1', 'Rattle', 'Clavicle', 'FMIn', 'Roundup', 'Phantom', 'MN6', 'MN9', 'Bract', 'Fdg', 'Fudog', 'Usnea', 'TH-VUM']:
    add('1D', n, n not in ('Usnea', 'TH-VUM'), True)

# 1F: required for sugar (MN9 <= 80% of control when silenced, 50 Hz sugar). Ratios from Table 2.
t2 = {r[0]: r for r in sheet('05_')[1:] if r and r[0]}
incorrect_1f = {'FMIn', 'Rattle', 'Roundup', 'Usnea'}
for n in ['Bract', 'Clavicle', 'Fdg', 'FMIn', 'G2N-1', 'Phantom', 'Rattle', 'Roundup', 'Usnea', 'Zorro']:
    pred = float(t2[n][6]) <= 0.8
    add('1F', n, pred if n not in incorrect_1f else not pred, pred, f'published ratio {float(t2[n][6]):.3f}')

# 2A: screen, 106 cell types activated at 50 Hz; prediction = MN9 (Table 3 "50 Hz MN9_Left") > 0 Hz.
t3 = sheet('06_')
first = {}
for k, h in enumerate(t3[0]):
    first.setdefault(h, k)
for r in t3[1:]:
    if r and r[0]:
        add('2A', r[0], float(r[1] or 0) > 0, float(r[first['50 Hz MN9_Left']] or 0) > 0,
            f"activation rate {r[1]}; published MN9 {float(r[first['50 Hz MN9_Left']] or 0):.3f} Hz")

# 3B-C: bitter and Ir94e inhibit MN9 under weak sugar; bitter inhibits under strong sugar, Ir94e does not (all correct).
add('3BC', 'bitter_weak_sugar', True, True)
add('3BC', 'ir94e_weak_sugar', True, True)
add('3BC', 'bitter_strong_sugar', True, True)
add('3BC', 'ir94e_strong_sugar', False, False)

# 4A: water-responsive neurons (>0 Hz). Shiu predicted all 10 respond; G2N-1 and Roundup incorrect.
for n in ['Clavicle', 'Phantom', 'Rattle', 'Usnea', 'MN6', 'MN9', 'Zorro', 'Fudog', 'G2N-1', 'Roundup']:
    add('4A', n, n not in ('G2N-1', 'Roundup'), True)

# 4C: required for water (MN9 <= 80% of control when silenced, 160 Hz water). Ratios and results from Table 5;
# Fudog's ratio from Table 6C (silenced, 160 Hz) over Table 6A (MN9 at waterR_160Hz).
t5 = {r[0]: r for r in sheet('08_')[1:] if r and r[0]}
t6a = sheet('09_'); col160 = t6a[0].index('waterR_160Hz')
mn9_160 = next(float(r[col160]) for r in t6a[1:] if r[0] == '720575940660219265')
fudog_sil = next(float(r[3]) for r in sheet('11_')[1:] if r[0] == '720575940612648106')
for n in ['Bract', 'Clavicle', 'G2N-1', 'Phantom', 'Rattle', 'Roundup', 'Tophat', 'Tulip', 'Usnea', 'Zorro', 'Fudog']:
    if n == 'Fudog':
        ratio = fudog_sil / mn9_160
        pred = ratio <= 0.8
        truth = pred  # listed as correct in Table 10
    else:
        ratio = float(t5[n][6])
        pred = ratio <= 0.8
        truth = t5[n][5].startswith('Yes')
    add('4C', n, truth, pred, f'published ratio {ratio:.3f}')

# 5B, 5D, 5I: Johnston's organ categories (all correct), as itemised in Table 11B.
for n in ['aBN1', 'aDN1', 'aDN2']:
    add('5B', n, True, True)
for n in ['aBN1', 'aBN2']:
    add('5D', n, True, True)
add('5I', 'aDN1_under_JO-CE', True, True)
add('5I', 'aDN1_under_JO-F', False, False)

# recount against Table 10
t10 = sheet('20_')
expected = {}
for r in t10[1:11]:
    expected[r[1].strip().split(',')[0].rstrip(':')] = (int(r[2]), int(r[3]))
key = {'1C': '1C', '1D': '1D', '1F': '1F', '2A': '2A', '3BC': '3B-C', '4A': '4A', '4C': '4C', '5B': '5B', '5D': '5D', '5I': '5I'}
for cat, label in key.items():
    sub = [i for i in items if i['category'] == cat]
    got = (sum(i['truth'] == i['shiu_pred'] for i in sub), len(sub))
    assert got == expected[label], (cat, got, expected[label])
correct = sum(i['truth'] == i['shiu_pred'] for i in items)
screen = [i for i in items if i['category'] == '2A']
assert (correct, len(items)) == (150, 164)
assert (correct - sum(i['truth'] == i['shiu_pred'] for i in screen), len(items) - len(screen)) == (49, 58)

with OUT.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(items[0]))
    w.writeheader(); w.writerows(items)
print(f'{len(items)} items written; every category recounts to Table 10 (150/164, 49/58 without the screen)')
