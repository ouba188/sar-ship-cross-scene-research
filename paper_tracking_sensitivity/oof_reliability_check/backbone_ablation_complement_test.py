"""Replay supplemental coefficients using the pre-existing expert view helpers."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from backbone_ablation_core import (OUT, EPISODES, AblationContext, folders, info_for,
    load_npz, write, _conditions, _pickle, PORT_REFERENCE, _expert_inputs)
from backbone_ablation_complement import DEST
from unified_head_run import base_kernel
from unified_head_core import coefficients, late


def check(folder):
    info, suffix = info_for(folder)
    data = load_npz(folder/'target.npz')
    embedding = load_npz(folder/'selective_embedding.npz')['embedding'].astype(float)
    primary = load_npz(OUT/'targets'/suffix/'predictions.npz')
    original_coef = load_npz(OUT/'targets'/suffix/'coefficients.npz')
    saved = load_npz(DEST/'targets'/suffix/'predictions.npz')
    saved_coef = load_npz(DEST/'targets'/suffix/'coefficients.npz')
    ix = np.unique(np.r_[data['support'].ravel(), np.arange(min(32, len(data['rows'])))])
    ctx = AblationContext(folder)
    if ctx.dataset == 'self24ports':
        from run_protected_drift_activation import make_views
        g = ctx.geometry
        bank = np.load(ctx.parent/'target_bank.npy', mmap_mode='r')[ix]
        extra = np.load(EPISODES/ctx.task/'matched_joint/joint_extra.npy', mmap_mode='r')[ix]
        views = make_views(bank, ctx.raw[data['rows'][ix]], extra, g)
        retained = views['common'][True]['retained']
        complement = views['common'][True]['complement']
        u = _conditions()[data['rows'][ix]]
        bias = 1.+.25*(views['stable']@views['stable'].T)+.125*(u@u.T)
        original = _pickle(PORT_REFERENCE/ctx.task/'direct_heads.pkl')['joint_expert']
        aug = np.c_[bank, u]
        priors = dict(dual_original=aug@original['coef']+original['intercept'],
            dual_refit=aug@original_coef['dual_coef']+original_coef['dual_intercept'])
    else:
        from evaluate_joint_residual import views
        target = Path(json.loads((ctx.joint.parent/'protocol.json').read_text())['target'])
        if ctx.dataset == 'oss2':
            with np.load(target/'features.npz') as z:
                raw = z['paired_r50'][ix]
        else:
            raw = np.load(target/'paired_r50.npy', mmap_mode='r')[ix]
        extra = np.load(ctx.joint/'target_extra.npy', mmap_mode='r')[ix]
        po, retained, complement = views(raw, extra, ctx.model, ctx.stats, True)
        inputs = _expert_inputs(raw, extra, ctx.model)
        priors = dict(dual_original=po, dual_refit=(inputs@original_coef['dual_coef']+
            original_coef['dual_intercept']).astype(np.float32).astype(float))
        bias = 1.
    retained_kernel = .5*(retained@retained.T)+bias
    full_kernel = retained_kernel+.5*(complement@complement.T)
    small = {key: data[key][ix] for key in ('phi', 'mean', 'conditions', 'prior')}
    max_alpha_error = 0.
    for draw, support in enumerate(data['support']):
        s = np.searchsorted(ix, support)
        kb = base_kernel(small, embedding[ix], s, info['eta'])
        base = small['prior']+kb@original_coef[f'{draw}_base']
        for name, pe in priors.items():
            full = pe+full_kernel[:, s]@original_coef[f'{draw}_{name}_expert']
            np.testing.assert_array_equal(full.argmax(1),
                primary['predictions'][draw, list(primary['modes']).index(name+'_expert')][ix])
            ae = saved_coef[f'{draw}_{name}_expert']
            aj = saved_coef[f'{draw}_{name}_unified']
            independent = coefficients(retained_kernel[np.ix_(s, s)], pe[s], data['support_labels'][draw], 8.)
            max_alpha_error = max(max_alpha_error, float(np.max(abs(ae-independent))))
            np.testing.assert_allclose(ae, independent, atol=2e-7, rtol=1e-6)
            expert = pe+retained_kernel[:, s]@ae
            joint = .5*(small['prior']+pe)+.5*(kb+retained_kernel[:, s])@aj
            for head, value in [('expert', expert), ('late', late(base, expert)), ('unified', joint)]:
                np.testing.assert_array_equal(value.argmax(1),
                    saved['predictions'][draw, list(saved['modes']).index(name+'_retained_'+head)][ix])
    return dict(task=suffix, draws=5, rows_checked=int(len(ix)), saved_predictions_exact=True,
                refitted_coefficient_max_error=max_alpha_error)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        results = [check(f) for f in folders()]
    write(DEST/'REPLAY_TEST.json', dict(tasks=len(results), results=results,
        failure_detected_by_test='Retained-kernel scaling, omitted stable/incidence, source-prior mismatch, support-column misalignment, or reuse of old residual coefficients'))
    print(f'PASS: {len(results)} tasks x five draws, both priors, all three readouts; max coefficient error '
          f'{max(r["refitted_coefficient_max_error"] for r in results):.3g}')
