"""Fixed complement removal and paired historical four-arm expert replay."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from backbone_ablation_core import (OUT, ROOT, SOURCE, BATCH, EPISODES, AblationContext,
    folders, info_for, load_npz, write, _expert_inputs, _conditions, _pickle, PORT_REFERENCE)
from unified_head_core import score, combined, late
from unified_head_run import base_kernel
from prepare_learned_frequency import DATASETS
from backbone_ablation_report import ba

DEST = OUT/'complement_and_historical'
ARMS = ('reference', 'complement_only', 'aligned_only', 'joint')
MODES = tuple(f'{prior}_retained_{head}' for prior in ('dual_original', 'dual_refit')
              for head in ('expert', 'late', 'unified')) + tuple(
    f'historical_{arm}_{head}' for arm in ARMS for head in ('expert', 'late'))
CONTRASTS = tuple((f'complement_{prior}', f'{prior}_{head}', f'{prior}_retained_{head}')
    for prior in ('dual_original', 'dual_refit') for head in ('expert', 'late', 'unified')) + tuple(
    (factor, f'historical_{a}_{head}', f'historical_{b}_{head}')
    for factor, a, b in [('historical_complement_untrained', 'complement_only', 'reference'),
                         ('historical_alignment_retained', 'aligned_only', 'reference'),
                         ('historical_complement_aligned', 'joint', 'aligned_only'),
                         ('historical_alignment_complement', 'joint', 'complement_only'),
                         ('historical_joint_total', 'joint', 'reference')]
    for head in ('expert', 'late')) + (
    ('current_vs_historical_expert_recipe', 'dual_original_late', 'historical_joint_late'),)


def freeze_protocol():
    if (DEST/'PROTOCOL.json').exists():
        return
    write(DEST/'PROTOCOL.json', dict(frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        tasks=[info_for(f)[1] for f in folders()], modes=MODES,
        requested_after_primary_ablation_readout=True, no_parameter_selection=True,
        fixed_kernel='Retained-only removes .5*K_complement; keeps .5*K_retained + bias1 + existing port stable/incidence terms',
        priors='Both original dual prior and matched-refit dual prior separately held fixed; reuse saved coefficients, no source refit',
        support='Same original five64 draws, class balance, ridge8, equal fusion; re-solve each changed expert/unified kernel',
        historical='Reuse fully paired cached expert_scores for reference/complement_only/aligned_only/joint; combine with latest full base by equal softmax average',
        pairing='Require exact rows or sample IDs, same source cohort for ports, exact64support and eval masks in all four arms',
        historical_boundary='Historical four-arm experts predate current port drift/stable/incidence recipe. This is historical trained expert transplant into current base fusion, not new training of every latest expert component.',
        frozen_complement_boundary='Removal from current frozen joint representation is distinct from training-aligned historical2x2',
        no_new_neural_training=True, no_new_source_training=True,
        readout='All25 supplemental predictions saved before their query evaluation; primary outcomes already known',
        cost='Full base still C4 four rotations; cached-feature inference and adaptation only, no end-to-end speedup assertion'))


def retained_geometry(ctx, data, coefficients):
    reference = np.unique(data['support'])
    n = len(data['rows'])
    k = np.empty((n, len(reference)), float)
    priors = {name: np.empty((n, 8), float) for name in ('dual_original', 'dual_refit')}
    if ctx.dataset == 'self24ports':
        from run_protected_drift_activation import DRIFT_STRENGTH
        g = ctx.geometry
        m = g['model']
        rank = g['stable_rank']
        bank = np.load(ctx.parent/'target_bank.npy', mmap_mode='r')
        np.testing.assert_array_equal(np.load(ctx.parent/'target_rows.npy'), data['rows'])
        u = _conditions()[data['rows']]
        head = _pickle(PORT_REFERENCE/ctx.task/'direct_heads.pkl')['joint_expert']

        def features(indices):
            b = np.asarray(bank[indices])
            core = b[:, :128]-DRIFT_STRENGTH*(b[:, 192+rank:]@g['drift_to_xs'])
            xe = (np.c_[core, b[:, 128:192].astype(np.float32)]-m['expert_center'])/m['expert_scale']
            return np.asarray(xe/np.sqrt(192), float), b[:, 192:192+rank]/np.sqrt(rank)

        rr, rs = features(reference)
        for lo in range(0, n, BATCH):
            hi = min(n, lo+BATCH)
            ret, stable = features(slice(lo, hi))
            k[lo:hi] = .5*(ret@rr.T)+1.+.25*(stable@rs.T)+.125*(u[lo:hi]@u[reference].T)
            aug = np.c_[bank[lo:hi], u[lo:hi]]
            priors['dual_original'][lo:hi] = aug@head['coef']+head['intercept']
            priors['dual_refit'][lo:hi] = aug@coefficients['dual_coef']+coefficients['dual_intercept']
    else:
        target = Path(json.loads((ctx.joint.parent/'protocol.json').read_text())['target'])
        if ctx.dataset == 'oss2':
            with np.load(target/'features.npz') as z:
                raw = z['paired_r50']
        else:
            raw = np.load(target/'paired_r50.npy', mmap_mode='r')
        extra = np.load(ctx.joint/'target_extra.npy', mmap_mode='r')
        np.testing.assert_array_equal(data['rows'], np.arange(len(raw)))
        rr = (_expert_inputs(raw[reference], extra[reference], ctx.model)/np.sqrt(192)).astype(float)
        for lo in range(0, n, BATCH):
            hi = min(n, lo+BATCH)
            x = _expert_inputs(raw[lo:hi], extra[lo:hi], ctx.model)
            k[lo:hi] = .5*((x/np.sqrt(192)).astype(float)@rr.T)+1.
            priors['dual_original'][lo:hi] = ctx.model['expert'].decision_function(x).astype(np.float32).astype(float)
            priors['dual_refit'][lo:hi] = (x@coefficients['dual_coef']+coefficients['dual_intercept']).astype(np.float32).astype(float)
    return reference, k, priors


def historical_paths(ctx):
    if ctx.dataset == 'self24ports':
        return {arm: EPISODES/ctx.task/'matched_joint'/f'{arm}_policies.npz' for arm in ARMS}
    return {arm: ctx.joint.parent/arm/'policies.npz' for arm in ARMS}


def load_historical(ctx, data):
    scores = {}
    provenance = []
    for arm, path in historical_paths(ctx).items():
        with np.load(path) as z:
            np.testing.assert_array_equal(z['eval_mask'], data['eval_masks'])
            if ctx.dataset == 'self24ports':
                np.testing.assert_array_equal(z['query_rows'], data['rows'])
                np.testing.assert_array_equal(z['source_rows'], data['source_rows'])
                np.testing.assert_array_equal(z['support_rows'], data['rows'][data['support']])
            else:
                np.testing.assert_array_equal(z['sample_ids'], data['ids'])
                np.testing.assert_array_equal(z['support_rows'], data['support'])
            scores[arm] = z['expert_scores']
        if scores[arm].shape != (5, len(data['rows']), 8):
            raise ValueError('Historical expert score shape differs')
        provenance.append(dict(arm=arm, path=str(path), matched_support_rows_eval=True))
    return scores, provenance


def predict_task(folder):
    info, suffix = info_for(folder)
    dst = DEST/'targets'/suffix
    if (dst/'COMPLETE.json').exists():
        return
    started = time.perf_counter()
    data = load_npz(folder/'target.npz')
    emb = load_npz(folder/'selective_embedding.npz')
    np.testing.assert_array_equal(emb['rows'], data['rows'])
    prior_result = load_npz(OUT/'targets'/suffix/'predictions.npz')
    coefficients = load_npz(OUT/'targets'/suffix/'coefficients.npz')
    ctx = AblationContext(folder)
    reference, kernel, priors = retained_geometry(ctx, data, coefficients)
    historical, provenance = load_historical(ctx, data)
    predictions = np.empty((5, len(MODES), len(data['rows'])), np.int8)
    alphas = {}
    for draw, support in enumerate(data['support']):
        if data['eval_masks'][draw, support].any() or len(np.unique(support)) != 64:
            raise ValueError('Changed support/query split')
        sy = data['support_labels'][draw]
        kb = base_kernel(data, emb['embedding'].astype(float), support, info['eta'])
        base = data['prior']+kb@coefficients[f'{draw}_base']
        np.testing.assert_array_equal(base.argmax(1),
            prior_result['predictions'][draw, list(prior_result['modes']).index('base')])
        ke = kernel[:, np.searchsorted(reference, support)]
        result = {}
        for prior_name, pe in priors.items():
            expert, ae = score(ke, pe, support, sy)
            joint, aj = combined(kb, ke, data['prior'], pe, support, sy)
            result[prior_name+'_retained_expert'] = expert
            result[prior_name+'_retained_late'] = late(base, expert)
            result[prior_name+'_retained_unified'] = joint
            alphas[f'{draw}_{prior_name}_expert'] = ae
            alphas[f'{draw}_{prior_name}_unified'] = aj
        for arm, scores in historical.items():
            result[f'historical_{arm}_expert'] = scores[draw]
            result[f'historical_{arm}_late'] = late(base, scores[draw])
        for mi, name in enumerate(MODES):
            predictions[draw, mi] = result[name].argmax(1)
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst/'predictions.npz', predictions=predictions, modes=np.array(MODES))
    np.savez_compressed(dst/'coefficients.npz', **alphas)
    write(dst/'COMPLETE.json', dict(**info, draws=5, seconds=time.perf_counter()-started,
        historical=provenance, base_predictions_exact=True, source_heads_reused=True,
        target_query_metrics_computed=False, saved_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print(f'COMPLEMENT+HISTORICAL {suffix}: all5 frozen; historical4arms paired', flush=True)


def report():
    tasks = folders()
    if len(tasks) != 25 or not all((DEST/'targets'/info_for(f)[1]/'COMPLETE.json').exists() for f in tasks):
        raise ValueError('All25 supplemental tasks must be frozen first')
    write(DEST/'ALL_PREDICTIONS_SAVED.json', dict(tasks=25, draws=125,
        frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), primary_outcomes_previously_known=True))
    metrics, effects, classes = [], [], []
    for folder in tasks:
        info, suffix = info_for(folder)
        primary = load_npz(OUT/'targets'/suffix/'predictions.npz')
        extra = load_npz(DEST/'targets'/suffix/'predictions.npz')
        manifest, idcol = ((SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
            else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'))
        frame = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[primary['rows']]
        np.testing.assert_array_equal(frame[idcol].to_numpy(str), primary['ids'])
        truth = frame.class_id.to_numpy(int)
        for draw, ev in enumerate(primary['eval_masks']):
            y = truth[ev]
            present = np.unique(y)
            predictions = {str(mode): p['predictions'][draw, i, ev]
                           for p in (primary, extra) for i, mode in enumerate(p['modes'])}
            for mode, pred in predictions.items():
                metrics.append(dict(dataset=info['dataset'], task=info['task'], draw=draw, mode=mode,
                    oa=100*np.mean(pred == y), ba=ba(y, pred), query_n=len(y)))
            for factor, candidate, control in CONTRASTS:
                a, b = predictions[candidate], predictions[control]
                rescued, harmed = (a == y) & (b != y), (a != y) & (b == y)
                meta = dict(dataset=info['dataset'], task=info['task'], draw=draw,
                            factor=factor, candidate=candidate, control=control)
                effects.append(dict(**meta, delta_oa=100*np.mean((a == y).astype(float)-(b == y)),
                    delta_ba=ba(y, a)-ba(y, b), rescues=int(rescued.sum()), harms=int(harmed.sum()),
                    net_correct=int(rescued.sum()-harmed.sum())))
                for c in present:
                    mask = y == c
                    delta = 100*np.mean((a[mask] == c).astype(float)-(b[mask] == c))
                    classes.append(dict(**meta, class_id=int(c), n=int(mask.sum()), delta_recall=delta,
                        ba_contribution=delta/len(present), rescues=int(rescued[mask].sum()), harms=int(harmed[mask].sum())))
    m, e, c = pd.DataFrame(metrics), pd.DataFrame(effects), pd.DataFrame(classes)
    m.to_csv(DEST/'metrics.csv', index=False)
    e.to_csv(DEST/'effects.csv', index=False)
    c.to_csv(DEST/'class_effects.csv', index=False)
    task = m.groupby(['dataset', 'task', 'mode'])[['oa', 'ba']].mean().reset_index()
    task.to_csv(DEST/'task_means.csv', index=False)
    summary = task.groupby(['dataset', 'mode'])[['oa', 'ba']].mean().reset_index()
    summary.to_csv(DEST/'summary.csv', index=False)
    keys = ['dataset', 'factor', 'candidate', 'control']
    values = ['delta_oa', 'delta_ba', 'rescues', 'harms', 'net_correct']
    te = e.groupby([*keys, 'task'])[values].mean().reset_index()
    te.to_csv(DEST/'task_effects.csv', index=False)
    es = te.groupby(keys)[values].mean().reset_index()
    es.to_csv(DEST/'effect_summary.csv', index=False)
    de = e.groupby([*keys, 'draw'])[values].mean().reset_index()
    de.to_csv(DEST/'draw_effects.csv', index=False)
    counts = task.groupby('dataset').task.nunique()
    cc = c.groupby([*keys, 'class_id'])[['ba_contribution', 'rescues', 'harms']].sum().reset_index()
    cc.ba_contribution /= cc.dataset.map(counts)*5
    cc.to_csv(DEST/'class_contributions.csv', index=False)
    for row in es.itertuples():
        own = cc[(cc.dataset == row.dataset) & (cc.factor == row.factor) & (cc.candidate == row.candidate)]
        np.testing.assert_allclose(own.ba_contribution.sum(), row.delta_ba, atol=1e-10)
    write(DEST/'COMPLETE.json', dict(tasks=25, draws=125, new_modes=MODES,
        new_evaluated_rows=125*len(MODES), historical_all_four_arms_paired_tasks=25,
        new_neural_training=False, new_source_training=False,
        all_additional_predictions_saved_before_additional_readout=True,
        primary_results_known_when_addendum_requested=True,
        all_latest_expert_training_components_closed=False))
    print(es.to_string(index=False), flush=True)


if __name__ == '__main__':
    freeze_protocol()
    with threadpool_limits(limits=2):
        for folder in folders():
            predict_task(folder)
        report()
