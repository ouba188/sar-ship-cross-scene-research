"""Remove only post-SSL source-episode tail training from the current readout."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from dual_reconstruction_core import Inputs, BATCH, folders, info_for, write, ROOT
from dual_reconstruction_predict import fit_student_head, view, kernel, base_block
from unified_head_expert import _conditions
from unified_head_core import coefficients, late
from compact_fusion_source import SOURCE
from backbone_ablation_report import ba
from prepare_learned_frequency import DATASETS

OUT = ROOT/'dual_frozen_teacher_20261010'
MODES = ('frozen_expert', 'frozen_late', 'frozen_unified')


def predict(folder):
    info, suffix = info_for(folder)
    dst = OUT/'targets'/suffix
    if (dst/'COMPLETE.json').exists():
        return
    started = time.perf_counter()
    with np.load(folder/'target.npz') as z:
        data = {k: z[k] for k in ('phi', 'mean', 'conditions', 'prior', 'rows', 'ids',
                                  'support', 'support_labels', 'eval_masks', 'source_rows')}
    with np.load(folder/'selective_embedding.npz') as z:
        np.testing.assert_array_equal(z['rows'], data['rows'])
        embed = z['embedding'].astype(float)
    inputs = Inputs(folder)
    baseline = ROOT/'neural_complement_pilot_20261007/frozen_replay'
    frozen_source = np.load(baseline/'source_extra.npy', mmap_mode='r')
    source_extra = frozen_source[inputs.source_rows]
    if info['dataset'] == 'self24ports':
        target_extra = frozen_source[inputs.rows]
        bank = np.load(inputs.ctx.parent/'target_bank.npy', mmap_mode='r')
        angles = _conditions()[inputs.rows]
        provenance = str(baseline/'source_extra.npy')
    else:
        path = (ROOT/'module_completion_20261008/oss2/reference/target_extra.npy'
                if info['dataset'] == 'oss2' else baseline/'target_extra.npy')
        target_extra = np.load(path, mmap_mode='r')
        bank, angles = None, None
        provenance = str(path)
    if target_extra.shape != inputs.target_extra.shape:
        raise ValueError('Frozen target feature rows differ')
    head = fit_student_head(inputs, None, 'frozen', source_extra=source_extra)
    del source_extra
    ref = np.unique(data['support'])
    locations = [np.searchsorted(ref, support) for support in data['support']]
    rv = view(inputs, np.array(target_extra[ref]), ref, head, bank, angles)
    kr = kernel(rv, rv)
    br = base_block(data, embed, ref, ref, info['eta'])
    alphas, saved = [], {}
    for draw, loc in enumerate(locations):
        sy = data['support_labels'][draw]
        support = data['support'][draw]
        pb, pe = data['prior'][support], rv['prior'][loc]
        kb, ke = br[np.ix_(loc, loc)], kr[np.ix_(loc, loc)]
        ab = coefficients(kb, pb, sy, 8.)
        ae = coefficients(ke, pe, sy, 8.)
        aj = coefficients(.5*(kb+ke), .5*(pb+pe), sy, 8.)
        alphas.append((ab, ae, aj))
        for name, value in [('base', ab), ('expert', ae), ('unified', aj)]:
            saved[f'{draw}_{name}'] = value
    n = len(inputs.rows)
    predictions = np.empty((5, 3, n), np.int8)
    base_pred = np.empty((5, n), np.int8)
    for lo in range(0, n, BATCH):
        s = slice(lo, min(lo+BATCH, n))
        v = view(inputs, np.array(target_extra[s]), s, head, bank, angles)
        ke = kernel(v, rv)
        kb = base_block(data, embed, s, ref, info['eta'])
        for draw, loc in enumerate(locations):
            ab, ae, aj = alphas[draw]
            b = data['prior'][s]+kb[:, loc]@ab
            e = v['prior']+ke[:, loc]@ae
            j = .5*(data['prior'][s]+v['prior'])+.5*(kb[:, loc]+ke[:, loc])@aj
            base_pred[draw, s] = b.argmax(1)
            for i, value in enumerate((e, late(b, e), j)):
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite frozen feature score')
                predictions[draw, i, s] = value.argmax(1)
    with np.load(ROOT/'backbone_ablation_20261010/targets'/suffix/'predictions.npz') as z:
        for key in ('rows', 'ids', 'support', 'eval_masks'):
            np.testing.assert_array_equal(data[key], z[key])
        np.testing.assert_array_equal(base_pred, z['predictions'][:, list(z['modes']).index('base')])
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst/'predictions.npz', predictions=predictions, modes=np.array(MODES),
        **{k: data[k] for k in ('rows', 'ids', 'support', 'eval_masks')})
    np.savez_compressed(dst/'coefficients.npz', **saved, source_coef=head['coef'], source_intercept=head['intercept'])
    write(dst/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'],
        source_features=str(baseline/'source_extra.npy'), target_features=provenance,
        source_n=len(inputs.source_rows), seconds=time.perf_counter()-started,
        source_head_refitted=True, base_prediction_exact=True, query_labels_evaluated=False))
    print('FROZEN TAIL DONE', suffix, time.perf_counter()-started, flush=True)


def report():
    if not all((OUT/'targets'/info_for(f)[1]/'COMPLETE.json').exists() for f in folders()):
        raise RuntimeError('Finish all25 frozen-tail predictions first')
    write(OUT/'ALL_PREDICTIONS_SAVED.json', dict(tasks=25, draws=125, modes=MODES))
    rows = []
    for folder in folders():
        info, suffix = info_for(folder)
        with np.load(OUT/'targets'/suffix/'predictions.npz') as z:
            p = {k: z[k] for k in z.files}
        with np.load(ROOT/'backbone_ablation_20261010/targets'/suffix/'predictions.npz') as z:
            old = {k: z[k] for k in z.files}
        manifest, idcol = ((SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
            else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'))
        frame = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(frame[idcol].to_numpy(str), p['ids'])
        truth = frame.class_id.to_numpy(int)
        for draw, ev in enumerate(p['eval_masks']):
            y = truth[ev]
            pred = {str(mode): p['predictions'][draw, i, ev] for i, mode in enumerate(p['modes'])}
            for kind in ('r50', 'dual_refit'):
                for head in ('expert', 'late', 'unified'):
                    mode = kind+'_'+head
                    pred[mode] = old['predictions'][draw, list(old['modes']).index(mode), ev]
            for mode, value in pred.items():
                rows.append(dict(dataset=info['dataset'], task=info['task'], draw=draw,
                    mode=mode, oa=100*np.mean(value == y), ba=ba(y, value)))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT/'metrics.csv', index=False)
    means = metrics.groupby(['dataset', 'task', 'mode'])[['oa', 'ba']].mean().reset_index()
    means.to_csv(OUT/'task_means.csv', index=False)
    summary = means.groupby(['dataset', 'mode'])[['oa', 'ba']].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    write(OUT/'COMPLETE.json', dict(tasks=25, draws=125, metrics=len(rows),
        no_new_neural_training=True, no_target_selection=True))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('predict', 'report', 'all'), default='all')
    args = parser.parse_args()
    path = OUT/'PROTOCOL.json'
    if not path.exists():
        write(path, dict(date='2026-10-10', tasks=[info_for(f)[1] for f in folders()],
            intervention='Replace task-trained SSL18 layer4 features with existing pre-episode-training frozen e206 features, source and target together',
            source='Same permitted source rows; refit balanced source ridge1 on replaced features',
            fixed='Full current base, geometry, normalization, same5x64 supports, residual ridge8, equal fusion',
            control='Current source-refit R50-only and source-refit trained dual from backbone_ablation_20261010',
            goal='Distinguish value present before source-episode tail training from value changed by that training',
            limitation='Does not isolate architecture depth, SSL corpus, input polarization or normalization individually',
            evaluation='All25 predictions frozen before query metrics; no target model selection'))
    with threadpool_limits(limits=2):
        if args.stage in ('predict', 'all'):
            for folder in folders():
                predict(folder)
        if args.stage in ('report', 'all'):
            report()
