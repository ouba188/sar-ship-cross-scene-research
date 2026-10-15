"""Use source-reconstructed features with the unchanged deployed expert operator."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import json
import time

import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from dual_reconstruction_core import (OUT, KINDS, Inputs, BATCH, folders, info_for,
                                      freeze_protocol, write, reconstruct, ROOT)
from compact_fusion_source import _expert_inputs
from unified_head_expert import _source_frame, _conditions
from protected_drift_activation import fit_source_heads
from unified_head_core import coefficients, late
from unified_head_run import base_kernel

MODES = ('base', *(f'{kind}_{head}' for kind in KINDS for head in ('expert', 'late', 'unified')))


def fit_student_head(inputs, mapper, kind, source_extra=None):
    c, model = inputs.ctx, inputs.model
    rows = inputs.source_rows
    labels = _source_frame().class_id.to_numpy()[rows]
    if inputs.dataset == 'self24ports':
        from direct_incidence_conditioning import extend_transforms
        from run_protected_drift_activation import make_transforms
        bank = np.array(np.load(c.parent/'source_bank.npy', mmap_mode='r'))
        for lo in range(0, len(rows), BATCH):
            s = slice(lo, min(lo+BATCH, len(rows)))
            extra = (reconstruct(inputs.source_x(s), mapper, kind) if source_extra is None
                     else np.asarray(source_extra[s]))
            bank[s, 128:192] = model['pca_ssl'].transform(extra).astype(np.float32)
        augmented = np.c_[bank, _conditions()[rows]]
        trans = extend_transforms({'student': make_transforms(c.geometry)['joint_expert']})
        head = fit_source_heads(augmented, labels, trans)['student']
    else:
        matrix = np.empty((len(rows), 192), float)
        for lo in range(0, len(rows), BATCH):
            s = slice(lo, min(lo+BATCH, len(rows)))
            extra = (reconstruct(inputs.source_x(s), mapper, kind) if source_extra is None
                     else np.asarray(source_extra[s]))
            matrix[s] = _expert_inputs(inputs.raw[rows[s]], extra, model)
        fitted = clone(model['expert']).fit(matrix, labels)
        head = dict(coef=fitted.coef_.T, intercept=fitted.intercept_)
    return head


def view(inputs, extra, index, head, target_bank, angles):
    c, model = inputs.ctx, inputs.model
    raw = inputs.target_raw_block(index)
    if inputs.dataset == 'self24ports':
        from run_protected_drift_activation import make_views
        bank = np.array(target_bank[index])
        bank[:, 128:192] = model['pca_ssl'].transform(extra).astype(np.float32)
        v = make_views(bank, raw, extra, c.geometry)
        ret, comp = v['common'][True]['retained'], v['common'][True]['complement']
        u = angles[index]
        prior = np.c_[bank, u]@head['coef']+head['intercept']
        return dict(ret=ret, comp=comp, stable=v['stable'], u=u, prior=prior)
    from evaluate_joint_residual import views
    _, ret, comp = views(raw, extra, model, c.stats, True)
    prior = (_expert_inputs(raw, extra, model)@head['coef']+head['intercept']).astype(np.float32).astype(float)
    return dict(ret=ret, comp=comp, prior=prior)


def kernel(left, right):
    k = .5*(left['ret']@right['ret'].T)+.5*(left['comp']@right['comp'].T)+1.
    if 'stable' in left:
        k += .25*(left['stable']@right['stable'].T)+.125*(left['u']@right['u'].T)
    return k


def base_block(data, embed, index, ref, eta):
    cond = np.exp(-np.sum((data['conditions'][index, None]-data['conditions'][ref][None])**2, axis=2)/.5)
    k = data['phi'][index]@data['phi'][ref].T
    k += eta*(data['mean'][index]@data['mean'][ref].T)*cond
    k += .25*(embed[index]@embed[ref].T)*cond
    return k


def predict(folder):
    info, suffix = info_for(folder)
    dst = OUT/'targets'/suffix
    if (dst/'COMPLETE.json').exists():
        print('REUSE PREDICT', suffix, flush=True)
        return
    started = time.perf_counter()
    with np.load(folder/'target.npz') as z:
        data = {k: z[k] for k in ('phi', 'mean', 'conditions', 'prior', 'rows', 'ids',
                                  'support', 'support_labels', 'eval_masks', 'source_rows')}
    with np.load(folder/'selective_embedding.npz') as z:
        np.testing.assert_array_equal(z['rows'], data['rows'])
        embed = z['embedding'].astype(float)
    with np.load(dst/'mapper.npz') as z:
        mapper = {k: z[k] for k in z.files}
    with np.load(ROOT/'backbone_ablation_20261010/targets'/suffix/'predictions.npz') as z:
        for key in ('rows', 'ids', 'support', 'eval_masks'):
            np.testing.assert_array_equal(z[key], data[key])
        oldbase = z['predictions'][:, list(z['modes']).index('base')]
        oldteacher = z['predictions'][:, list(z['modes']).index('dual_refit_expert')]
    inputs = Inputs(folder)
    np.testing.assert_array_equal(inputs.source_rows, data['source_rows'])
    target_bank, angles = None, None
    if inputs.dataset == 'self24ports':
        target_bank = np.load(inputs.ctx.parent/'target_bank.npy', mmap_mode='r')
        np.testing.assert_array_equal(np.load(inputs.ctx.parent/'target_rows.npy'), data['rows'])
        angles = _conditions()[data['rows']]
    ref = np.unique(data['support'])
    locations = [np.searchsorted(ref, s) for s in data['support']]
    with np.load(ROOT/'backbone_ablation_20261010/targets'/suffix/'coefficients.npz') as z:
        teacher_head = dict(coef=z['dual_coef'], intercept=z['dual_intercept'])
        tv = view(inputs, np.array(inputs.target_extra[ref]), ref, teacher_head, target_bank, angles)
        tk = kernel(tv, tv)
        for draw, loc in enumerate(locations):
            replay = tv['prior']+tk[:, loc]@z[f'{draw}_dual_refit_expert']
            np.testing.assert_array_equal(replay.argmax(1), oldteacher[draw, ref])
    del tv, tk, oldteacher
    kb_ref = base_block(data, embed, ref, ref, info['eta'])
    # Check the chunked base operator against the existing support-kernel API.
    small = {k: data[k][ref] for k in ('phi', 'mean', 'conditions', 'prior')}
    np.testing.assert_allclose(kb_ref[:, locations[0]],
        base_kernel(small, embed[ref], locations[0], info['eta']), atol=1e-10, rtol=1e-10)
    n = len(data['rows'])
    predictions = np.empty((5, len(MODES), n), np.int8)
    saved, diagnostics, times = {}, {}, {}
    for kind in KINDS:
        t = time.perf_counter()
        head = fit_student_head(inputs, mapper, kind)
        ref_extra = reconstruct(inputs.target_x(ref), mapper, kind)
        rv = view(inputs, ref_extra, ref, head, target_bank, angles)
        ke_ref = kernel(rv, rv)
        alphas = []
        for draw, (support, loc) in enumerate(zip(data['support'], locations)):
            sy = data['support_labels'][draw]
            if len(np.unique(support)) != 64 or data['eval_masks'][draw, support].any():
                raise ValueError('Changed support budget or support/query overlap')
            kb, ke = kb_ref[np.ix_(loc, loc)], ke_ref[np.ix_(loc, loc)]
            pb, pe = data['prior'][support], rv['prior'][loc]
            ab = coefficients(kb, pb, sy, 8.)
            ae = coefficients(ke, pe, sy, 8.)
            aj = coefficients(.5*(kb+ke), .5*(pb+pe), sy, 8.)
            alphas.append((ab, ae, aj))
            for name, alpha in [('base', ab), (kind+'_expert', ae), (kind+'_unified', aj)]:
                saved[f'{draw}_{name}'] = alpha
        error = target_energy = cosine = 0.
        ysum = np.zeros(512)
        ysq = np.zeros(512)
        sqerror = np.zeros(512)
        for lo in range(0, n, BATCH):
            s = slice(lo, min(lo+BATCH, n))
            extra = reconstruct(inputs.target_x(s), mapper, kind)
            v = view(inputs, extra, s, head, target_bank, angles)
            ke = kernel(v, rv)
            kb = base_block(data, embed, s, ref, info['eta'])
            for draw, loc in enumerate(locations):
                ab, ae, aj = alphas[draw]
                b = data['prior'][s]+kb[:, loc]@ab
                e = v['prior']+ke[:, loc]@ae
                j = .5*(data['prior'][s]+v['prior'])+.5*(kb[:, loc]+ke[:, loc])@aj
                predictions[draw, 0, s] = b.argmax(1)
                for h, value in [('expert', e), ('late', late(b, e)), ('unified', j)]:
                    predictions[draw, MODES.index(kind+'_'+h), s] = value.argmax(1)
                if not all(np.isfinite(a).all() for a in (b, e, j)):
                    raise ValueError('Nonfinite student scores')
            yt = (np.asarray(inputs.target_extra[s], float)-mapper['y_mean'])/mapper['y_std']
            yp = (extra.astype(float)-mapper['y_mean'])/mapper['y_std']
            d2 = np.sum((yp-yt)**2, axis=0)
            error += float(d2.sum())
            target_energy += float(np.sum(yt**2))
            sqerror += d2
            ysum += yt.sum(0)
            ysq += (yt**2).sum(0)
            cosine += float(np.sum(np.sum(yp*yt, axis=1)/np.maximum(
                np.linalg.norm(yp, axis=1)*np.linalg.norm(yt, axis=1), 1e-12)))
        np.testing.assert_array_equal(predictions[:, 0], oldbase)
        centered_energy = ysq-ysum**2/n
        diagnostics[kind] = dict(source_scaled_mse=error/(n*512),
            source_mean_baseline_mse=target_energy/(n*512),
            variance_weighted_target_r2=1-error/max(float(centered_energy.sum()), 1e-12),
            mean_centered_feature_cosine=cosine/n,
            target_n=n, feature_diagnostic_not_used_for_selection=True)
        saved[kind+'_source_coef'], saved[kind+'_source_intercept'] = head['coef'], head['intercept']
        times[kind] = time.perf_counter()-t
        print('PREDICTED', suffix, kind, times[kind], flush=True)
    np.savez_compressed(dst/'predictions.npz', predictions=predictions, modes=np.array(MODES),
        **{k: data[k] for k in ('rows', 'ids', 'support', 'eval_masks')})
    np.savez_compressed(dst/'coefficients.npz', **saved)
    write(dst/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'],
        kinds=KINDS, feature_diagnostics=diagnostics, prediction_seconds=times,
        total_seconds=time.perf_counter()-started, base_predictions_exact=True,
        original_teacher_support_union_predictions_exact=True,
        query_classification_labels_evaluated=False, target_mapper_fit=False))
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    freeze_protocol()
    with threadpool_limits(limits=2):
        for folder in folders():
            if not args.tasks or info_for(folder)[0]['task'] in args.tasks:
                predict(folder)
