"""Freeze protocol and all 25 task predictions before evaluating query truth."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

from backbone_ablation_core import (OUT, OLD, MODES, AblationContext, folders,
                                    info_for, write, load_npz)
from unified_head_run import base_kernel
from unified_head_core import score, combined, late


def freeze_protocol():
    path = OUT/'PROTOCOL.json'
    if path.exists():
        return
    tasks = [info_for(f)[1] for f in folders()]
    if len(tasks) != 25:
        raise ValueError(f'Expected latest 25 tasks, got {len(tasks)}')
    write(path, dict(date='2026-10-10', tasks=tasks, modes=MODES,
        protocol_frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        support='Original five draws, 64 labels per task/draw; no class quota changes',
        base='Exact full frozen phi + eta*mean kernel + .25*selective embedding kernel, with original condition RBF and prior',
        expert_single='R50 PCA128, full standardized R50 PCA complement2048; ports also preserve drift suppression, stable directions and incidence',
        normalization='Retained /sqrt192 and complement /sqrt2560 unchanged in both arms; source stable transform sqrt192/r unchanged; no post-removal renormalization',
        source_heads='Refit class-balanced ridge1 with unpenalized intercept on exact existing source_rows; no query labels',
        source_control='Dual refit uses same current frozen source representation as R50 refit; original dual priors separately replayed',
        source_exclusions='Outer target port or existing external identity-purged source_split; source_rows must equal full-base source_rows',
        target_fit='Class-balanced support ridge8; separate heads vs one combined kernel/prior, rho=.5',
        primary_metric='BA; OA reported together; 23 valid self24ports tasks averaged equally',
        tuning='None. No source validation selection, no target-query selection, no new neural fitting',
        evaluation='Write every task prediction and ALL_PREDICTIONS_SAVED marker before query metrics',
        label_access='ExpertContext helpers read source manifest containing all labels. Only permitted source labels enter source fit; support_labels enter target fit. Loading is not fitting.',
        contrasts='dual_refit-r50 isolates SSL18 incremental value under matched source refit; dual_refit-dual_original measures prior refit; r50-dual_original is operational replacement',
        timing='CPU2 source fit and frozen-feature kernel construction; per-draw solve+score time, excluding backbone forward, image loading, feature extraction, original training, disk writes. No end-to-end speed claim.',
        limitations=['Frozen-backbone cached-feature ablation, not exhaustive end-to-end single-model training',
            'Original F0921 SSL encoder has historical target-corpus exposure',
            'Five overlapping support draws are not independent data replications',
            'Source geometry and neural weights inherited; not representation-held validation']))


def predict_task(folder):
    info, suffix = info_for(folder)
    dst = OUT/'targets'/suffix
    if (dst/'COMPLETE.json').exists():
        print(f'RESUME {suffix}: already complete', flush=True)
        return
    started = time.perf_counter()
    data = load_npz(folder/'target.npz')
    embedding = load_npz(folder/'selective_embedding.npz')
    previous = load_npz(OLD/'targets'/suffix/'policies.npz')
    for key in ('rows', 'ids', 'support', 'eval_masks'):
        np.testing.assert_array_equal(data[key], previous[key])
    np.testing.assert_array_equal(data['rows'], embedding['rows'])
    ctx = AblationContext(folder)
    details = ctx.fit_heads(data)
    print(f'SOURCE FIT {suffix}: n={details["source_n"]} {details["source_fit_seconds"]:.1f}s', flush=True)
    reference, priors, kernels, feature_seconds = ctx.build(data)
    predictions = np.empty((5, len(MODES), len(data['rows'])), np.int8)
    alphas, timing, replay = {}, [], []
    for draw, support in enumerate(data['support']):
        if len(np.unique(support)) != 64 or data['eval_masks'][draw, support].any():
            raise ValueError('Support/query overlap or changed budget')
        labels = data['support_labels'][draw]
        columns = np.searchsorted(reference, support)
        t = time.perf_counter()
        kb = base_kernel(data, embedding['embedding'].astype(float), support, info['eta'])
        base_kernel_seconds = time.perf_counter()-t
        t = time.perf_counter()
        b, ab = score(kb, data['prior'], support, labels)
        base_seconds = time.perf_counter()-t
        alphas[f'{draw}_base'] = ab
        result = dict(base=b)
        record = dict(draw=draw, base_kernel_seconds=base_kernel_seconds, base_solve_score_seconds=base_seconds)
        for name, prior_key, kernel_key in [('dual_original', 'original', 'dual'),
                                           ('dual_refit', 'dual', 'dual'), ('r50', 'r50', 'r50')]:
            ke, pe = kernels[kernel_key][:, columns], priors[prior_key]
            t = time.perf_counter()
            expert, ae = score(ke, pe, support, labels)
            expert_seconds = time.perf_counter()-t
            t = time.perf_counter()
            joint, aj = combined(kb, ke, data['prior'], pe, support, labels)
            joint_seconds = time.perf_counter()-t
            t = time.perf_counter()
            fused = late(b, expert)
            late_seconds = time.perf_counter()-t
            result.update({name+'_expert': expert, name+'_late': fused, name+'_unified': joint})
            alphas[f'{draw}_{name}_expert'] = ae
            alphas[f'{draw}_{name}_unified'] = aj
            record.update({name+'_expert_solve_score_seconds': expert_seconds,
                           name+'_unified_solve_score_seconds': joint_seconds,
                           name+'_late_readout_seconds': late_seconds,
                           name+'_two_head_solve_score_seconds': base_seconds+expert_seconds+late_seconds})
        error = float(np.max(np.abs(result['dual_original_expert']-data['expert'][draw])))
        np.testing.assert_allclose(result['dual_original_expert'], data['expert'][draw], atol=3e-5, rtol=1e-6)
        for name, old_name in [('base', 'base'), ('dual_original_expert', 'expert'),
                ('dual_original_late', 'late_softmax_fixed'), ('dual_original_unified', 'unified_fixed')]:
            np.testing.assert_array_equal(result[name].argmax(1),
                previous['predictions'][draw, list(previous['modes']).index(old_name)])
        for mi, name in enumerate(MODES):
            if not np.isfinite(result[name]).all():
                raise ValueError(f'Nonfinite predictions in {suffix}/{name}')
            predictions[draw, mi] = result[name].argmax(1)
        replay.append(dict(draw=draw, original_expert_max_error=error, original_predictions_exact=True))
        timing.append(record)
        print(f'FROZEN {suffix} draw{draw}: old base/expert/late/unified exact', flush=True)
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst/'predictions.npz', predictions=predictions, modes=np.array(MODES),
        rows=data['rows'], ids=data['ids'], support=data['support'], eval_masks=data['eval_masks'])
    np.savez_compressed(dst/'coefficients.npz', **alphas,
        **{name+'_'+key: value for name, head in ctx.fitted.items() for key, value in head.items()})
    write(dst/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], draws=5,
        **details, shared_both_expert_kernel_build_seconds=feature_seconds,
        total_seconds=time.perf_counter()-started, timing=timing, replay=replay,
        prediction_saved_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        query_truth_metrics_computed=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--stage', choices=('predict', 'report', 'all'), default='all')
    args = parser.parse_args()
    freeze_protocol()
    with threadpool_limits(limits=2):
        if args.stage in ('predict', 'all'):
            for folder in folders():
                if not args.tasks or info_for(folder)[0]['task'] in args.tasks:
                    predict_task(folder)
        if args.stage in ('report', 'all'):
            from backbone_ablation_report import report
            report()


if __name__ == '__main__':
    main()
