"""Score a finished run: item predictions, accuracy and balanced accuracy, and agreement with Shiu et al.'s predictions.

Usage: python score.py results/v630/M     (prints a summary and writes scores.json next to rates.json)
Balanced accuracy = mean of the hit rate on items whose experimental outcome is "yes" and the correct-rejection rate on
items whose outcome is "no". Items that cannot be measured on a wiring (neurons absent from v783) are left out and listed.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments as E


def metrics(preds, targets, items):
    pairs = [(preds[i], int(targets[i]['truth'])) for i in items if preds.get(i) is not None]
    if not pairs:
        return None
    pos = [p for p, t in pairs if t == 1]
    neg = [p for p, t in pairs if t == 0]
    hit = sum(pos) / len(pos) if pos else None
    cr = sum(1 - p for p in neg) / len(neg) if neg else None
    ba = (hit + cr) / 2 if hit is not None and cr is not None else None
    return {'n': len(pairs), 'n_yes': len(pos), 'n_no': len(neg), 'accuracy': sum(p == t for p, t in pairs) / len(pairs),
            'hit_rate': hit, 'correct_rejection_rate': cr, 'balanced_accuracy': ba,
            'predicted_yes': sum(p for p, _ in pairs)}


def score_rates(rates):
    targets = E.load_targets()
    means = {exp: v['mean'] for exp, v in rates.items()}
    preds = E.predict(means, None)
    items = list(targets)
    screen = [i for i in items if targets[i]['category'] == '2A']
    other = [i for i in items if targets[i]['category'] != '2A']
    out = {'all': metrics(preds, targets, items), 'screen': metrics(preds, targets, screen),
           'non_screen': metrics(preds, targets, other),
           'by_category': {c: metrics(preds, targets, [i for i in items if targets[i]['category'] == c])
                           for c in sorted({targets[i]['category'] for i in items})},
           'not_measurable': [i for i in items if preds.get(i) is None],
           'agreement_with_shiu': None, 'predictions': preds}
    comparable = [i for i in items if preds.get(i) is not None]
    out['agreement_with_shiu'] = {
        'agree': sum(preds[i] == int(targets[i]['shiu_pred']) for i in comparable), 'n': len(comparable),
        'disagree': [i for i in comparable if preds[i] != int(targets[i]['shiu_pred'])]}
    return out


def main(run_dir):
    run_dir = Path(run_dir)
    s = score_rates(json.loads((run_dir / 'rates.json').read_text()))
    (run_dir / 'scores.json').write_text(json.dumps(s, indent=1))
    for k in ('all', 'screen', 'non_screen'):
        m = s[k]
        print(f"{k:10s} n={m['n']:3d} accuracy {m['accuracy']:.3f}  balanced {m['balanced_accuracy']:.3f}  "
              f"hits {m['hit_rate']:.3f}  correct rejections {m['correct_rejection_rate']:.3f}  predicted yes {m['predicted_yes']}")
    a = s['agreement_with_shiu']
    print(f"agreement with Shiu et al.'s published predictions: {a['agree']}/{a['n']}; differ: {a['disagree']}")
    if s['not_measurable']:
        print('not measurable on this wiring:', s['not_measurable'])


if __name__ == '__main__':
    main(sys.argv[1])
