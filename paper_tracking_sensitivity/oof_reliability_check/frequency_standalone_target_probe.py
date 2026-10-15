"""Post-report target diagnostics only; no selection, retraining or causal claims.

OA/BA are percentages; matched deltas are percentage points. Mean rows average
four band accuracies (not predictions) and event counts. Summary accuracies first
average draws within task, then tasks within dataset. Summary counts sum query-
draw events; mean-row counts are band-averaged totals, not unique sample counts.
shared_both_wrong means both standalone and integrated band predictions are wrong.
"""

import json

from frequency_standalone_experiment import OUT, EXP, ROOT, FAMILIES

import numpy as np
import pandas as pd

from prepare_learned_frequency import SOURCE, DATASETS, load_npz
from frequency_band_core import BANDS, unit_bands
from frequency_band_report import markdown_table
from frequency_reliability_core import coefficients
from frequency_reliability_run import statistics

COUNTS = ['alone_correct_integrated_wrong', 'integrated_correct_alone_wrong',
          'shared_both_wrong', 'both_correct']
VALUES = ['oa', 'ba', 'integrated_oa', 'integrated_ba', 'delta_oa', 'delta_ba']


def predict(bands, conditions, support, sy):
    cs, bs = conditions[support], bands[support]
    condition = np.exp(-np.sum((cs[:, None]-cs[None])**2, axis=2)/.5)
    prior = np.zeros((len(support), 8))
    dual = [coefficients(.25*(bs[:, b]@bs[:, b].T)*condition, sy, prior)
            for b in range(len(BANDS))]
    predictions = np.empty((len(bands), len(BANDS)), dtype=np.int8)
    for lo in range(0, len(bands), 8192):
        hi = min(lo+8192, len(bands))
        condition = np.exp(-np.sum((conditions[lo:hi, None]-cs[None])**2, axis=2)/.5)
        for b in range(len(BANDS)):
            gram = .25*(bands[lo:hi, b]@bs[:, b].T)*condition
            predictions[lo:hi, b] = (gram@dual[b]).argmax(1)
    return predictions


def main():
    marker = OUT/'REPORT_COMPLETE.json'
    if not marker.is_file():
        raise ValueError('Finish the standalone report before target diagnostics')
    status = json.loads(marker.read_text())
    if status['tasks'] != 25 or not status['labels_aligned'] or tuple(status['families']) != FAMILIES:
        raise ValueError('Expected the aligned 25-task standalone report')
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    infos = [json.loads((p/'PREPARED.json').read_text()) for p in folders]
    expected = {(i['dataset'], i['task']) for i in infos}
    complete = {(p.parent.parent.name, p.parent.name)
                for p in (OUT/'targets').glob('*/*/COMPLETE.json')}
    if len(folders) != 25 or len(expected) != 25 or complete != expected:
        raise ValueError('Expected all 25 matching frozen target tasks')
    manifests, records = {}, []
    for folder, info in zip(folders, infos):
        dataset, task = info['dataset'], info['task']
        with np.load(folder/'target.npz') as archive:
            data = {k: archive[k] for k in ('rows', 'ids', 'support', 'support_labels',
                                           'eval_masks', 'conditions')}
        with np.load(OUT/'targets'/dataset/task/'policies.npz') as archive:
            p = {k: archive[k] for k in ('rows', 'support', 'eval_masks', 'modes', 'base_predictions')}
        for key in ('rows', 'support', 'eval_masks'):
            np.testing.assert_array_equal(p[key], data[key])
        if data['support'].shape != (5, 64) or data['eval_masks'].shape != (5, len(data['rows'])):
            raise ValueError(f'Expected five 64-label support draws: {dataset}/{task}')
        id_column = 'instance_id' if dataset == 'self24ports' else 'sample_id'
        if dataset not in manifests:
            path = (SOURCE/'manifest.csv' if dataset == 'self24ports' else
                    ROOT/DATASETS[dataset][0]/'target_manifest.csv')
            manifests[dataset] = pd.read_csv(path, usecols=[id_column, 'class_id'], keep_default_na=False)
        aligned = manifests[dataset].iloc[data['rows']]
        np.testing.assert_array_equal(aligned[id_column].to_numpy(str), data['ids'])
        labels = aligned.class_id.to_numpy()
        np.testing.assert_array_equal(labels[data['support']], data['support_labels'])
        modes = list(p['modes'])
        for family in FAMILIES[1:]:
            z = load_npz(OUT/'tasks'/dataset/task/f'{family}_embedding.npz')
            np.testing.assert_array_equal(z['rows'], data['rows'])
            bands = unit_bands(z['embedding'])
            for draw, support in enumerate(data['support']):
                ev, sy = data['eval_masks'][draw], data['support_labels'][draw]
                if len(np.unique(support)) != 64 or ev[support].any():
                    raise ValueError(f'Support/query split mismatch: {dataset}/{task}/{draw}')
                prediction = predict(bands, data['conditions'], support, sy)
                y, rows = labels[ev], []
                for b, band in enumerate(BANDS):
                    alone = prediction[ev, b]
                    integrated = p['base_predictions'][draw, modes.index(f'{family}_{band}'), ev]
                    oa, ba, _ = statistics(y, alone)
                    ioa, iba, _ = statistics(y, integrated)
                    a, i = alone == y, integrated == y
                    counts = [int(mask.sum()) for mask in (a & ~i, i & ~a, ~a & ~i, a & i)]
                    rows.append(dict(dataset=dataset, task=task, draw=draw, family=family,
                        band=band, n=len(y), oa=oa, ba=ba, integrated_oa=ioa, integrated_ba=iba,
                        **dict(zip(COUNTS, counts))))
                mean = pd.DataFrame(rows)[['oa', 'ba', 'integrated_oa', 'integrated_ba', *COUNTS]].mean()
                records.extend(rows)
                records.append(dict(dataset=dataset, task=task, draw=draw, family=family,
                                    band='mean', n=len(y), **mean.to_dict()))
        print(f'TARGET STANDALONE PROBE {dataset}/{task}', flush=True)
    frame = pd.DataFrame(records)
    match = ['dataset', 'task', 'draw', 'band']
    reference = frame[frame.family.eq('independent')][match+['oa', 'ba']].rename(
        columns={'oa': 'independent_oa', 'ba': 'independent_ba'})
    frame = frame.merge(reference, on=match, how='left', validate='many_to_one')
    for value in ('oa', 'ba'):
        frame['delta_'+value] = frame[value]-frame['independent_'+value]
    frame.to_csv(OUT/'target_standalone_probe.csv', index=False)
    keys = ['dataset', 'family', 'band']
    task_means = frame.groupby([*keys, 'task'])[VALUES].mean()
    summary = task_means.groupby(keys)[VALUES].mean()
    summary['tasks'] = task_means.groupby(keys).size()
    summary['task_draws'] = frame.groupby(keys).size()
    summary['query_draw_events'] = frame.groupby(keys)['n'].sum()
    summary[COUNTS] = frame.groupby(keys)[COUNTS].sum()
    summary = summary.reset_index()
    summary.to_csv(OUT/'target_standalone_probe_summary.csv', index=False)
    print(markdown_table(summary[summary.band.eq('mean')][['dataset', 'family', *VALUES]]), flush=True)


if __name__ == '__main__':
    main()
