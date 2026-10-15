"""Independent confusion-matrix replay of enriched or context policies, no fits."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path(__file__).resolve().parent.parent / 'pred_cache')
    p.add_argument('--parent', type=Path, default=Path(__file__).resolve().parent.parent / 'oof_reliability_results_v1')
    args = p.parse_args()
    inventory = json.loads((args.run / 'POLICIES_SAVED.json').read_text(encoding='utf-8'))
    reported = json.loads((args.run / 'summary.json').read_text(encoding='utf-8'))
    totals = {arm: [] for arm in reported['means_ba']}
    max_error, count, contexts = 0., 0, 0
    for item in inventory:
        port = item['port']
        with np.load(args.cache / (port + '.npz'), allow_pickle=False) as a:
            y, expected_q = a['y_ids'], a['qpos']
            base = a['base_logits'][expected_q].argmax(1)
            expert = a['expert_logits'][expected_q].argmax(1)
        with np.load(args.run / port / 'policies.npz', allow_pickle=False) as a:
            data = {k: a[k] for k in a.files}
        record = json.loads((args.run / port / 'metrics.json').read_text(encoding='utf-8'))
        parent = json.loads((args.parent / (port + '.json')).read_text(encoding='utf-8'))
        require(np.array_equal(data['query_rows'], expected_q), 'query rows differ')
        per_port = {a: [] for a in totals}
        for i, seed in enumerate(data['seeds']):
            contexts += 1
            support = data['support_rows'][i]
            ev = data['eval_masks'][i]
            require(len(support) == len(np.unique(support)) == 64, 'support budget/uniqueness')
            require(np.array_equal(support, parent['seeds'][i]['annotation_rows']), 'support changed')
            require(np.array_equal(ev, ~np.isin(expected_q, support)), 'evaluation includes support')
            true = y[expected_q][ev]
            b, e = base[ev], expert[ev]
            classes, class_counts = np.unique(true, return_counts=True)
            weights = np.zeros(8)
            weights[classes] = 1. / (len(classes) * class_counts)
            sample_weight = weights[true]
            preds = dict(zip(data['arm_names'], data['predictions'][i]))
            require(np.array_equal(preds['base'], base) and np.array_equal(preds['fusion'], expert),
                    'fixed predictor changed')
            observed = np.bincount(y[support], minlength=8) > 0
            frozen_scores = data['scores'][i] if data['scores'].ndim == 3 else data['scores']
            for name, score in zip(data['score_names'], frozen_scores):
                require(np.isfinite(score).all(), 'nonfinite frozen score')
                expected = np.where((base != expert) & observed[expert] & (score > 0), expert, base)
                require(np.array_equal(preds[name], expected), 'score/guard/prediction mismatch')
                if 'covered' in data:
                    require(np.array_equal(preds[name][~data['covered']], preds['oof_lr0'][~data['covered']]),
                            'missing context changed baseline')
            preds['oracle'] = np.where((expert == y[expected_q]) & (base != y[expected_q]), expert, base)
            oracle_ba = float(sample_weight[preds['oracle'][ev] == true].sum())
            for arm, prediction in preds.items():
                require(np.all((prediction == base) | (prediction == expert)), 'third-class override')
                prediction = prediction[ev]
                matrix = confusion_matrix(true, prediction, labels=np.arange(8))
                recalls = matrix.diagonal()[classes] / matrix.sum(axis=1)[classes]
                chosen = (b != e) & (prediction == e)
                metrics = dict(ba=float(recalls.mean()), acc=float(matrix.trace() / matrix.sum()),
                               recover_ba=float(sample_weight[chosen & (e == true)].sum()),
                               harm_ba=float(sample_weight[chosen & (b == true)].sum()),
                               oracle_gap_ba=oracle_ba - float(recalls.mean()))
                for key, val in metrics.items():
                    error = abs(val - record['seeds'][i]['metrics'][arm][key])
                    max_error = max(error, max_error)
                    require(error < 1e-12, f'metric mismatch {port}/{seed}/{arm}/{key}')
                    count += 1
                per_port[arm].append(metrics['ba'])
            for arm in ('oof_lr0', 'oof_lr64'):
                if arm not in record['seeds'][i]['metrics']:
                    continue
                require(record['seeds'][i]['metrics'][arm] == parent['seeds'][i]['metrics'][arm],
                        'inherited LR result changed')
                if arm not in preds:
                    per_port[arm].append(parent['seeds'][i]['metrics'][arm]['ba'])
        for arm, vals in per_port.items():
            totals[arm].append(float(np.mean(vals)))
    for arm, vals in totals.items():
        error = abs(float(np.mean(vals)) - reported['means_ba'][arm])
        max_error = max(error, max_error)
        require(error < 1e-12, 'aggregate mean mismatch: ' + arm)
    result = dict(status='verified', ports=len(inventory), contexts=contexts,
                  independently_recomputed_metrics=count, maximum_error=max_error,
                  reference_LR_metrics='checked against parent; saved predictions replayed where present',
                  classification_heads_retrained=False, independent_dataset=False)
    (args.run / 'VERIFICATION.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
