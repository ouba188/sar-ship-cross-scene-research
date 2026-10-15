"""Base-only readout of matched digital rotations, not a deployment benchmark."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json

import numpy as np
import pandas as pd

from prepare_learned_frequency import OUT as EXP, ROOT, SOURCE, DATASETS, load_npz
from evaluate_learned_frequency import predict
from explain_frequency_gain import metrics

OUT = ROOT/'frequency_information_content_20261009/rotation_probe'


def evaluate(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/info['dataset']/info['task']
    if (out/'EVALUATED.json').exists():
        return
    raw, emb = load_npz(folder/'target.npz'), load_npz(out/'embeddings.npz')
    rows = emb['local_rows']
    lookup = np.full(len(raw['rows']), -1, dtype=int)
    lookup[rows] = np.arange(len(rows))
    supports, query = lookup[raw['support']], lookup[emb['query']]
    if np.any(supports < 0) or np.any(query < 0) or np.intersect1d(supports, query).size:
        raise ValueError('Sample-support remapping error')
    data = {k: raw[k][rows] for k in ('phi', 'prior', 'mean', 'conditions')}
    y = (pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()[raw['rows'][rows]]
         if info['dataset'] == 'self24ports' else pd.read_csv(ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv',
                usecols=['class_id']).class_id.to_numpy()[rows])
    policies = load_npz(folder/'policies.npz')
    selective = list(policies['modes']).index('selective')
    records, classes, sensitivity = [], [], []
    for model in ('signed', 'unsigned'):
        original, rotated = emb[model+'_original'], emb[model+'_rotated']
        consensus = (original.astype(float)+rotated)/2
        consensus /= np.maximum(np.linalg.norm(consensus, axis=1, keepdims=True), 1e-12)
        query_rotated = original.copy()
        query_rotated[query] = rotated[query]
        similarity = np.sum(original[query]*rotated[query], axis=1)
        for c in [-1, *np.unique(y[query])]:
            mask = np.ones(len(query), bool) if c == -1 else y[query] == c
            sensitivity.append(dict(dataset=info['dataset'], task=info['task'], model=model,
                class_id=int(c), n=int(mask.sum()), mean_cosine=float(similarity[mask].mean())))
        for draw, support in enumerate(supports):
            baseline = raw['expected_base'][draw, rows][query]
            oldoa, oldba = metrics(y[query], baseline)
            reference = predict(data, original, support, raw['support_labels'][draw], info['eta']).argmax(1)
            cached_reference = predict(data, emb[model+'_cached'], support, raw['support_labels'][draw], info['eta']).argmax(1)
            if model == 'signed':
                np.testing.assert_array_equal(cached_reference[query], policies['base_predictions'][draw, selective, rows[query]])
            for mode, z in [('original', original), ('rotate_query', query_rotated), ('c2_mean', consensus)]:
                prediction = predict(data, z, support, raw['support_labels'][draw], info['eta']).argmax(1)
                oa, ba = metrics(y[query], prediction[query])
                records.append(dict(dataset=info['dataset'], task=info['task'], model=model, draw=draw,
                    mode=mode, n=len(query), oa=oa, ba=ba, delta_old_oa=oa-oldoa, delta_old_ba=ba-oldba,
                    precision_changed_vs_cached=int(np.sum(reference[query] != cached_reference[query])),
                    changed=int(np.sum(prediction[query] != reference[query])),
                    lost_correct=int(np.sum((reference[query] == y[query]) & (prediction[query] != y[query]))),
                    gained_correct=int(np.sum((reference[query] != y[query]) & (prediction[query] == y[query])))))
                for c in np.unique(y[query]):
                    q = query[y[query] == c]
                    classes.append(dict(dataset=info['dataset'], task=info['task'], model=model, draw=draw,
                        mode=mode, class_id=int(c), n=len(q), recall=100*float(np.mean(prediction[q] == c))))
    pd.DataFrame(records).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(out/'classes.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity.csv', index=False)
    (out/'EVALUATED.json').write_text(json.dumps(dict(dataset=info['dataset'], task=info['task'],
        original_base_replay=True, neural_retraining=False, target_label_selection=False,
        sampled_query_n=len(query), full_model=False), indent=2))
    print(f'ROTATION EVAL {info["task"]}', flush=True)


def report():
    folders = [p.parent for p in sorted(OUT.glob('*/*/EVALUATED.json'))]
    data = pd.concat([pd.read_csv(f/'metrics.csv') for f in folders], ignore_index=True)
    data.to_csv(OUT/'metrics_all.csv', index=False)
    fields = ['oa', 'ba', 'delta_old_oa', 'delta_old_ba']
    summary = data.groupby(['dataset', 'model', 'mode', 'task'])[fields].mean().groupby(['dataset', 'model', 'mode']).mean()
    summary.to_csv(OUT/'summary.csv')
    data.groupby(['dataset', 'model', 'mode'])[['n', 'changed', 'lost_correct', 'gained_correct']].sum().to_csv(OUT/'events.csv')
    data.groupby(['dataset', 'model', 'mode', 'draw'])[fields].mean().to_csv(OUT/'draw_metrics.csv')
    for name in ('sensitivity', 'classes'):
        pd.concat([pd.read_csv(f/f'{name}.csv') for f in folders], ignore_index=True).to_csv(OUT/f'{name}_all.csv', index=False)
    (OUT/'READOUT.json').write_text(json.dumps(dict(tasks=len(folders), expected=25, complete=len(folders) == 25,
        full_model=False, independent_confirmation=False, sampled_queries=True), indent=2))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        evaluate(marker.parent)
    report()
