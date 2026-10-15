"""Matched 64-label / 5%-handoff pilot of source-PCA-complement information."""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from budgeted_residual_handoff import score_policy
from projection_residual_expert import build_support_kernels, fit_geometry_statistics, freeze_kernel_draw
from run_opensarship2_transfer import SOURCE, WS, write_json


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
DATASETS = {'oss1': 'opensarship1_transfer_20261007',
            'oss2': 'opensarship2_transfer_20261007',
            'f0921': 'f0921_detector_recrop_20261007'}
ARMS = ('pca_reference', 'raw_joint', 'retained_half_control', 'projection_residual')


def load_target(name):
    folder = ROOT/DATASETS[name]
    frame = pd.read_csv(folder/'target_manifest.csv', keep_default_na=False)
    with np.load(folder/'policies.npz') as saved:
        policy = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(policy['sample_ids'], frame.sample_id.to_numpy(str))
    if name == 'f0921':
        base = np.load(folder/'paired_r50.npy', mmap_mode='r')
        extra = np.load(folder/'vh_ssl.npy', mmap_mode='r')
        np.testing.assert_array_equal(np.load(folder/'feature_rows.npy'), frame.row.to_numpy(int))
        source_folder = ROOT/'f0921_transfer_20261007'
    else:
        with np.load(folder/'features.npz') as features:
            np.testing.assert_array_equal(features['sample_ids'], policy['sample_ids'])
            base, extra = features['paired_r50'], features['vh_ssl']
        source_folder = folder
    with (source_folder/'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    with np.load(source_folder/'source_split.npz') as saved:
        source_rows = saved['source_rows']
    return frame, base, extra, policy, model, source_rows, source_folder


def freeze_dataset(name, out, source_base, source_extra):
    folder = out/name
    folder.mkdir()
    started = time.monotonic()
    frame, base, extra, old, model, source_rows, source_folder = load_target(name)
    stats = fit_geometry_statistics(source_base, source_extra, source_rows, model)
    np.savez_compressed(folder/'source_geometry.npz', source_rows=source_rows, **stats)
    reference_rows = np.unique(old['support_rows'])
    kernels = build_support_kernels(base, extra, model, stats, reference_rows)
    print(f'{name} SOURCE GEOMETRY / SUPPORT KERNELS READY {time.monotonic()-started:.1f}s', flush=True)
    all_draws = {arm: [] for arm in ARMS}
    max_replay_error = 0.
    for draw, support in enumerate(old['support_rows']):
        columns = np.searchsorted(reference_rows, support)
        support_labels = frame.iloc[support].class_id.to_numpy(int)
        pca = kernels['pca'][:, columns]
        choices = dict(pca_reference=pca+1., raw_joint=kernels['raw'][:, columns]+1.,
            retained_half_control=.5*pca+1.,
            projection_residual=.5*pca+.5*kernels['discarded'][:, columns]+1.)
        for arm, kernel in choices.items():
            result = freeze_kernel_draw(old['base_scores'][draw], old['prior_expert_scores'], kernel,
                support, support_labels, model['gate'], frame.target_manifest_row.to_numpy())
            np.testing.assert_array_equal(result['eval_mask'], old['eval_masks'][draw])
            if result['selected'][~result['eval_mask']].any() or result['selected'].sum() > old['caps'][draw]:
                raise ValueError('support exclusion or budget mismatch')
            np.testing.assert_array_equal(result['predictions'] != old['predictions'][draw, 1], result['selected'])
            if arm == 'pca_reference':
                error = float(np.max(np.abs(result['expert_scores']-old['expert_scores'][draw])))
                max_replay_error = max(max_replay_error, error)
                np.testing.assert_allclose(result['expert_scores'], old['expert_scores'][draw], atol=3e-5, rtol=0)
                np.testing.assert_array_equal(result['predictions'], old['predictions'][draw, 2])
                np.testing.assert_array_equal(result['selected'], old['selected'][draw])
            all_draws[arm].append(result)
        print(f'{name} DRAW {draw+1}/5 FROZEN', flush=True)
    for arm, draws in all_draws.items():
        np.savez_compressed(folder/f'{arm}_policies.npz',
            **{key: np.stack([d[key] for d in draws]) for key in draws[0]},
            support_rows=old['support_rows'], sample_ids=old['sample_ids'])
    write_json(folder/'POLICIES_SAVED.json', dict(dataset=name, arms=ARMS,
        target_rows=len(frame), source_rows=len(source_rows), source_model=str(source_folder/'source_model.pkl'),
        max_reference_score_error=max_replay_error, reference_predictions_exact=True,
        labels_per_draw=64, budget=.05, elapsed_seconds=time.monotonic()-started))
    print(f'{name} ALL ARMS SAVED BEFORE QUERY READOUT', flush=True)


def evaluate_dataset(name, out, arms=ARMS):
    folder = out/name
    frame = pd.read_csv(ROOT/DATASETS[name]/'target_manifest.csv', keep_default_na=False)
    truth = frame.class_id.to_numpy(int)
    with np.load(ROOT/DATASETS[name]/'policies.npz') as old:
        base_predictions = old['predictions'][:, 1]
    rows = []
    for arm in arms:
        with np.load(folder/f'{arm}_policies.npz') as policy:
            for draw in range(5):
                base, prediction = base_predictions[draw], policy['predictions'][draw]
                expert, ev = policy['expert_predictions'][draw], policy['eval_mask'][draw]
                selected = policy['selected'][draw]
                full = score_policy(prediction, base, truth, ev)
                expert_metrics = score_policy(expert, base, truth, ev)
                base_metrics = score_policy(base, base, truth, ev)
                rows.append(dict(dataset=name, arm=arm, draw=draw, **full,
                    expert_ba=expert_metrics['ba'], expert_acc=expert_metrics['acc'],
                    base_ba=base_metrics['ba'], base_acc=base_metrics['acc'],
                    selected=int(selected.sum()), evaluated=int(ev.sum()), cap=int(policy['cap'][draw]),
                    rescued=int(np.sum(selected & (prediction == truth) & (base != truth))),
                    harmed=int(np.sum(selected & (prediction != truth) & (base == truth)))))
    summary = []
    for arm in arms:
        own = [r for r in rows if r['arm'] == arm]
        reference = [r for r in rows if r['arm'] == 'pca_reference']
        deltas = {metric: [100*(r[metric]-b[metric]) for r, b in zip(own, reference)]
                  for metric in ('acc', 'ba', 'expert_acc', 'expert_ba')}
        measured = dict(dataset=name, arm=arm,
            **{metric: float(np.mean([r[metric] for r in own])) for metric in
               ('acc', 'ba', 'expert_acc', 'expert_ba', 'base_acc', 'base_ba', 'selected', 'rescued', 'harmed')},
            delta_vs_reference_pp={k: float(np.mean(v)) for k, v in deltas.items()},
            draw_delta_vs_reference_pp=deltas,
            ba_positive_draws=int(np.count_nonzero(np.array(deltas['ba']) > 0)))
        summary.append(measured)
        print(f'{name} {arm} FULL OA={100*measured["acc"]:.6f} BA={100*measured["ba"]:.6f}; '
              f'EXPERT OA={100*measured["expert_acc"]:.6f} BA={100*measured["expert_ba"]:.6f}; '
              f'dBA={measured["delta_vs_reference_pp"]["ba"]:+.6f} '
              f'positive={measured["ba_positive_draws"]}/5', flush=True)
    write_json(folder/'metrics.json', rows)
    write_json(folder/'summary.json', summary)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out/name).exists() for name in DATASETS):
        raise ValueError('use a new output directory; preserve existing pilots')
    write_json(args.out/'PROTOCOL.json', dict(date='2026-10-08', datasets=DATASETS, arms=ARMS,
        support='Exact existing5draws of64 labels; row-level historical development splits',
        base='Exact cached64-adapted base scores; source expert prior, gate and existing source cohort frozen',
        geometry='Coordinatewise mean/std from each identity-purged source cohort only',
        complement='(x-source_PCA_mean) minus its retained-PC reconstruction, separately R50/SSL',
        kernels='pca=Kp+1; raw=Kr+1; retained_half=.5Kp+1; projection_residual=.5Kp+.5Kd+1',
        rationale='Normalize each bias-free branch by sqrt(d); fixed equal mixture and one penalized bias',
        residual_ridge=8., gate_threshold=0., handoff_budget=.05,
        control='Half-retained control separates added complement from attenuation of the old branch',
        prohibited_inputs='No query labels, geometry missingness, AIS length or port metadata in fitting/routing',
        parameters='Frozen before this pilot readout, no hyperparameter grid or checkpoint search',
        interpretation='Development pilot; gate remains inherited and may not be calibrated for new expert',
        independent_confirmation=False, main_model_unchanged=True))
    source_base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    with np.load(WS/'b2_ssl_feats_38091.npz') as saved:
        source_extra = saved['F'].astype(np.float32)
    all_rows, all_summaries = [], []
    with threadpool_limits(limits=4):
        # Freeze every dataset's policies before reading any query outcomes.
        for name in DATASETS:
            freeze_dataset(name, args.out, source_base, source_extra)
        for name in DATASETS:
            rows, summary = evaluate_dataset(name, args.out)
            all_rows.extend(rows)
            all_summaries.extend(summary)
    write_json(args.out/'summary.json', all_summaries)
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)}
                  for row in all_rows]).to_csv(args.out/'metrics.csv', index=False)
    print('PROJECTION RESIDUAL PILOT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
