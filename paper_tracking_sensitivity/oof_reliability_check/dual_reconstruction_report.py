"""Classification readout after every fixed reconstruction arm has been saved."""

import json
import time

import numpy as np
import pandas as pd

from dual_reconstruction_core import OUT, ROOT, KINDS, folders, info_for, write
from compact_fusion_source import SOURCE
from backbone_ablation_report import ba
from prepare_learned_frequency import DATASETS


def report():
    tasks = folders()
    expected = json.loads((OUT/'PROTOCOL.json').read_text())['tasks']
    if [info_for(f)[1] for f in tasks] != expected or len(tasks) != 25:
        raise ValueError('Task cohort changed')
    if not all((OUT/'targets'/info_for(f)[1]/'COMPLETE.json').exists() for f in tasks):
        raise RuntimeError('Complete all25 prediction sets before classification evaluation')
    write(OUT/'ALL_PREDICTIONS_SAVED.json', dict(tasks=25, draws=125,
        saved_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), no_query_selection=True))
    metrics, effects, diagnoses, classes = [], [], [], []
    for folder in tasks:
        info, suffix = info_for(folder)
        dst = OUT/'targets'/suffix
        with np.load(dst/'predictions.npz') as z:
            p = {k: z[k] for k in z.files}
        with np.load(ROOT/'backbone_ablation_20261010/targets'/suffix/'predictions.npz') as z:
            ref = {k: z[k] for k in z.files}
        for key in ('rows', 'ids', 'support', 'eval_masks'):
            np.testing.assert_array_equal(p[key], ref[key])
        manifest, idcol = ((SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
            else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'))
        frame = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(frame[idcol].to_numpy(str), p['ids'])
        truth = frame.class_id.to_numpy(int)
        with np.load(folder/'target.npz') as z:
            np.testing.assert_array_equal(truth[p['support']], z['support_labels'])
        common = dict(dataset=info['dataset'], task=info['task'])
        for draw, ev in enumerate(p['eval_masks']):
            y = truth[ev]
            preds = {str(mode): p['predictions'][draw, mi, ev] for mi, mode in enumerate(p['modes'])}
            for kind in ('r50', 'dual_refit'):
                for head in ('expert', 'late', 'unified'):
                    mode = f'{kind}_{head}'
                    preds[mode] = ref['predictions'][draw, list(ref['modes']).index(mode), ev]
            for mode, pred in preds.items():
                metrics.append(dict(**common, draw=draw, mode=mode, oa=100*np.mean(pred == y), ba=ba(y, pred)))
            for kind in KINDS:
                for head in ('expert', 'late', 'unified'):
                    candidate = kind+'_'+head
                    pred = preds[candidate]
                    for control in ('r50_'+head, 'dual_refit_'+head):
                        old = preds[control]
                        rescue, harm = (pred == y)&(old != y), (pred != y)&(old == y)
                        effects.append(dict(**common, draw=draw, candidate=candidate, control=control,
                            delta_oa=100*np.mean((pred == y).astype(float)-(old == y)),
                            delta_ba=ba(y, pred)-ba(y, old), rescues=int(rescue.sum()), harms=int(harm.sum())))
                        if head == 'late':
                            for c in np.unique(y):
                                mask = y == c
                                classes.append(dict(**common, draw=draw, candidate=candidate, control=control,
                                    class_id=int(c), n=int(mask.sum()), rescues=int(rescue[mask].sum()), harms=int(harm[mask].sum()),
                                    delta_recall=100*np.mean((pred[mask] == c).astype(float)-(old[mask] == c))))
        complete = json.loads((dst/'COMPLETE.json').read_text())
        fit = json.loads((dst/'FIT.json').read_text())
        for kind in KINDS:
            diagnoses.append(dict(**common, kind=kind,
                source_scaled_mse=fit['source_standardized_mse'][kind],
                **{('target_'+k if k == 'source_scaled_mse' else k): v
                   for k, v in complete['feature_diagnostics'][kind].items()}))
    metrics, effects = pd.DataFrame(metrics), pd.DataFrame(effects)
    metrics.to_csv(OUT/'metrics.csv', index=False)
    effects.to_csv(OUT/'effects.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_effects.csv', index=False)
    diagnoses = pd.DataFrame(diagnoses)
    diagnoses.to_csv(OUT/'feature_diagnostics.csv', index=False)
    task_means = metrics.groupby(['dataset', 'task', 'mode'])[['oa', 'ba']].mean().reset_index()
    task_means.to_csv(OUT/'task_means.csv', index=False)
    summary = task_means.groupby(['dataset', 'mode'])[['oa', 'ba']].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    es = effects.groupby(['dataset', 'task', 'candidate', 'control'])[['delta_oa', 'delta_ba']].mean().reset_index()
    es.to_csv(OUT/'task_effects.csv', index=False)
    es = es.groupby(['dataset', 'candidate', 'control'])[['delta_oa', 'delta_ba']].mean().reset_index()
    es.to_csv(OUT/'effect_summary.csv', index=False)
    ds = ['self24ports', 'f0921', 'oss2']
    lines = ['# Source-Only Single-Backbone Reconstruction', '',
        'All25 tasks, five fixed64-label draws; final checkpoint and both mapper architectures were frozen before target classification evaluation.', '',
        'No SSL18 execution is required for these student inputs at inference. Original-view R50 and its existing C4 mean feed a fixed linear mapper or a 2,360,320-parameter MLP. The teacher is required during source training. End-to-end student latency has not been measured.', '',
        '## Classification', '', '| Mode | 24 ports OA / BA | F_0921 OA / BA | OSS2 OA / BA |',
        '| --- | ---: | ---: | ---: |']
    for kind in ('r50', 'linear', 'mlp', 'dual_refit'):
        for head in ('expert', 'late', 'unified'):
            mode = kind+'_'+head
            vals = []
            for dataset in ds:
                r = summary[(summary.dataset == dataset)&(summary['mode'] == mode)].iloc[0]
                vals.append(f'{r.oa:.5f} / {r.ba:.5f}')
            lines.append('| '+mode+' | '+' | '.join(vals)+' |')
    lines += ['', '## Transfer of Teacher Features', '',
        'R2 is variance-weighted across source-standardized teacher coordinates, using the target cohort mean only in the diagnostic denominator. It is not class accuracy, causal information or a measure of physical mechanisms. Source MSE is in-sample.', '',
        '| Dataset | Mapper | Source MSE | Target MSE | Target R2 | Centered cosine |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    grouped = diagnoses.groupby(['dataset', 'kind']).mean(numeric_only=True)
    for dataset in ds:
        for kind in KINDS:
            r = grouped.loc[dataset, kind]
            lines.append(f'| {dataset} | {kind} | {r.source_scaled_mse:.5f} | {r.target_source_scaled_mse:.5f} | {r.variance_weighted_target_r2:.5f} | {r.mean_centered_feature_cosine:.5f} |')
    lines += ['', '## Boundaries', '',
        '- Original base predictions replay exactly for every sample/draw. Original matched-refit teacher expert predictions replay on every support-union sample before replacement.',
        '- Source geometry, retained/complement normalization and 24-port stable/drift/angle operators are unchanged. A fresh source classifier is fitted on student features to avoid training/deployment feature mismatch.',
        '- Mapper source rows equal the permitted source cohort; no target query or support labels train the mapper. All variants share the original64 target labels for residual adaptation.',
        '- The GPU fitter loads inherited sklearn objects but never calls their transform/predict methods. Geometry prediction uses the original ClearSAR sklearn1.8.0 environment. Mapper CPU/GPU inference agrees on source probes.',
        '- A failed linear/MLP reconstruction cannot establish that no single network can obtain the information. A successful student would support source-learned inference compression, not teacher-free training.',
        '- Inherited development reuse and historical F_0921 SSL exposure remain. Five support draws are not five neural seeds or independent confirmation.', '']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
    write(OUT/'COMPLETE.json', dict(tasks=25, draws=125, mapper_kinds=KINDS,
        metrics_rows=len(metrics), base_replay_exact=True, no_target_selection=True,
        all_predictions_before_classification_evaluation=True, independent_confirmation=False))
    print(summary.to_string(index=False), flush=True)
    print(grouped.to_string(), flush=True)


if __name__ == '__main__':
    report()
