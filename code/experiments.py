"""Experiments behind the 164 tested predictions, and the rules that turn their firing rates into predictions.

Operational choices follow Shiu et al.'s published scoring where it is stated (Methods; figures.ipynb; Supplementary
Tables 2, 3, 5 and 11) and were fixed in the frozen protocol before any null wiring was run (docs/DESIGN.md):
  1C  sugar R and sugar L at 100 Hz: contralateral MN9 rate > ipsilateral MN9 rate
  1D  sugar R at 200 Hz: neuron rate > 0 Hz
  1F  sugar R at 50 Hz, each neuron silenced: required if MN9 <= 0.8 x control (control 0 Hz: not required)
  2A  each screen cell type at 50 Hz: MN9 rate > 0 Hz
  3BC sugar R at 100 Hz (weak) or 200 Hz (strong) with bitter or Ir94e GRNs at 200 Hz: inhibits if MN9 <= 0.1 x MN9 with
      the same sugar alone (sugar alone 0 Hz: no inhibition can be shown, predicted no)
  4A  water at 260 Hz: neuron rate > 0 Hz
  4C  water at 160 Hz, each neuron silenced: required if MN9 <= 0.8 x control
  5B  all JONs at 220 Hz: neuron rate > 0 Hz
  5D  all JONs at 160 Hz, aBN1 or aBN2 silenced: required if aDN1 <= 0.8 x control
  5I  JO-CE or JO-F at 150 Hz: aDN1 rate > 0 Hz
"""
import csv
import zlib
from pathlib import Path

import numpy as np

import inputs as I

P1 = Path(__file__).resolve().parents[1]
SILENCE_1F = ['Bract', 'Clavicle', 'Fdg', 'FMIn', 'G2N-1', 'Phantom', 'Rattle', 'Roundup', 'Usnea', 'Zorro']
SILENCE_4C = ['Bract', 'Clavicle', 'G2N-1', 'Phantom', 'Rattle', 'Roundup', 'Tophat', 'Tulip', 'Usnea', 'Zorro', 'Fudog']
RESPOND_1D = ['Zorro', 'G2N-1', 'Rattle', 'Clavicle', 'FMIn', 'Roundup', 'Phantom', 'MN6', 'MN9', 'Bract', 'Fdg', 'Fudog', 'Usnea', 'TH-VUM']
RESPOND_4A = ['Clavicle', 'Phantom', 'Rattle', 'Usnea', 'MN6', 'MN9', 'Zorro', 'Fudog', 'G2N-1', 'Roundup']


def seed_for(name):
    """Same Poisson seed for an experiment on every wiring (common random numbers)."""
    return zlib.crc32(name.encode()) % (2 ** 31 - 64)


class Mapping:
    """Resolves Shiu et al.'s v630 IDs on a wiring. On v630 every ID resolves. On v783 an ID resolves only if the root ID
    is unchanged; the second MN9 is found by its v783 cell-type label."""

    def __init__(self, net, version, mn9_other_index=None):
        self.net, self.version = net, version
        self.mn9 = net.index.get(I.MN9)
        self.mn9_other = net.index.get(I.MN9_OTHER) if mn9_other_index is None else mn9_other_index
        self.named = {k: net.index.get(v) for k, v in I.NAMED.items()}
        self.named['MN9'] = self.mn9

    def ids(self, id_list):
        return [self.net.index[x] for x in id_list if x in self.net.index]


def build(mapping):
    """Return ({experiment name: (drive {index: Hz}, silence [indices])}, readout indices {label: index})."""
    m = mapping
    exps = {}

    def drive(ids, hz, extra=None):
        d = {i: float(hz) for i in m.ids(ids)}
        if extra:
            d.update({i: float(extra[1]) for i in m.ids(extra[0])})
        return d

    exps['sugarR_100'] = (drive(I.SUGAR_R, 100), [])
    exps['sugarL_100'] = (drive(I.SUGAR_L, 100), [])
    exps['sugarR_200'] = (drive(I.SUGAR_R, 200), [])
    exps['sugarR_50'] = (drive(I.SUGAR_R, 50), [])
    for n in SILENCE_1F:
        if m.named[n] is not None:
            exps[f'sugarR_50_sil_{n}'] = (drive(I.SUGAR_R, 50), [m.named[n]])
    for t, ids in I.screen_types().items():
        d = drive(ids, 50)
        if d:
            exps[f'screen_{t}'] = (d, [])
    for inh, ids in (('bitter', I.BITTER), ('ir94e', I.IR94E)):
        for hz in (100, 200):
            exps[f'sugarR_{hz}_{inh}_200'] = (drive(I.SUGAR_R, hz, (ids, 200)), [])
    exps['waterR_260'] = (drive(I.WATER, 260), [])
    exps['waterR_160'] = (drive(I.WATER, 160), [])
    for n in SILENCE_4C:
        if m.named[n] is not None:
            exps[f'waterR_160_sil_{n}'] = (drive(I.WATER, 160), [m.named[n]])
    exps['JON_220'] = (drive(I.JON_ALL, 220), [])
    exps['JON_160'] = (drive(I.JON_ALL, 160), [])
    for n in ('aBN1', 'aBN2'):
        if m.named[n] is not None:
            exps[f'JON_160_sil_{n}'] = (drive(I.JON_ALL, 160), [m.named[n]])
    exps['JOCE_150'] = (drive(I.JON_CE, 150), [])
    exps['JOF_150'] = (drive(I.JON_F, 150), [])
    readout = {k: v for k, v in m.named.items() if v is not None}
    if m.mn9_other is not None:
        readout['MN9_other'] = m.mn9_other
    return exps, readout


def predict(rates, mapping):
    """rates: {experiment: {readout label: mean rate in Hz}}. Returns {item: 0/1, or None if not measurable}."""
    r = rates
    p = {}

    def rate(exp, label):
        return r.get(exp, {}).get(label)

    def ratio_required(ctrl_exp, sil_exp, target):
        c, s = rate(ctrl_exp, target), rate(sil_exp, target)
        if c is None or s is None:
            return None
        return int(c > 0 and s <= 0.8 * c)

    for exp, item, contra, ipsi in (('sugarR_100', 'sugarR_contra_gt_ipsi', 'MN9', 'MN9_other'),
                                    ('sugarL_100', 'sugarL_contra_gt_ipsi', 'MN9_other', 'MN9')):
        a, b = rate(exp, contra), rate(exp, ipsi)
        p[f'1C:{item}'] = None if a is None or b is None else int(a > b)
    for n in RESPOND_1D:
        v = rate('sugarR_200', n)
        p[f'1D:{n}'] = None if v is None else int(v > 0)
    for n in SILENCE_1F:
        p[f'1F:{n}'] = ratio_required('sugarR_50', f'sugarR_50_sil_{n}', 'MN9')
    # Amendment 1: screen items are named as in Supplementary Table 3; match the pickle's spelling case-insensitively
    table3_name = {row['subject'].lower(): row['subject'] for row in load_targets().values() if row['category'] == '2A'}
    for t in I.screen_types():
        v = rate(f'screen_{t}', 'MN9')
        p[f'2A:{table3_name.get(t.lower(), t)}'] = None if v is None else int(v > 0)
    for inh in ('bitter', 'ir94e'):
        for hz, strength in ((100, 'weak'), (200, 'strong')):
            alone, both = rate(f'sugarR_{hz}', 'MN9'), rate(f'sugarR_{hz}_{inh}_200', 'MN9')
            p[f'3BC:{inh}_{strength}_sugar'] = None if alone is None or both is None else int(alone > 0 and both <= 0.1 * alone)
    for n in RESPOND_4A:
        v = rate('waterR_260', n)
        p[f'4A:{n}'] = None if v is None else int(v > 0)
    for n in SILENCE_4C:
        p[f'4C:{n}'] = ratio_required('waterR_160', f'waterR_160_sil_{n}', 'MN9')
    for n in ('aBN1', 'aDN1', 'aDN2'):
        v = rate('JON_220', n)
        p[f'5B:{n}'] = None if v is None else int(v > 0)
    for n in ('aBN1', 'aBN2'):
        p[f'5D:{n}'] = ratio_required('JON_160', f'JON_160_sil_{n}', 'aDN1')
    for exp, item in (('JOCE_150', 'aDN1_under_JO-CE'), ('JOF_150', 'aDN1_under_JO-F')):
        v = rate(exp, 'aDN1')
        p[f'5I:{item}'] = None if v is None else int(v > 0)
    return p


def load_targets():
    with open(P1 / 'data/derived/targets.csv') as f:
        return {row['item']: row for row in csv.DictReader(f)}
