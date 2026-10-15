"""Freeze and read out the matched 2x2 integration pilot, with fixed base64."""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from budgeted_residual_handoff import score_policy
from projection_residual_expert import discarded_coordinates, freeze_kernel_draw
from run_opensarship2_transfer import target_geometry, write_json


def views(base, extra, model, stats, complement):
    _, prior, _, phi = target_geometry(model, base, extra)
    if complement:
        raw = np.column_stack((discarded_coordinates(base, model['pca']),
                               discarded_coordinates(extra, model['pca_ssl'])))
        residual = (raw-stats['discarded_center'])/stats['discarded_scale']/np.sqrt(raw.shape[1])
    else:
        residual = None
    return prior, phi[:, :-1], residual


def build_kernels(base, extra, model, stats, reference_rows, complement):
    _, retained_reference, discarded_reference = views(
        base[reference_rows], extra[reference_rows], model, stats, complement)
    prior = np.empty((len(base), 8))
    kernels = dict(retained=np.empty((len(base), len(reference_rows))))
    if complement:
        kernels['complement'] = np.empty_like(kernels['retained'])
    for lo in range(0, len(base), 2048):
        hi = min(lo+2048, len(base))
        p, retained, discarded = views(base[lo:hi], extra[lo:hi], model, stats, complement)
        prior[lo:hi] = p
        product = retained@retained_reference.T
        kernels['retained'][lo:hi] = product+1.
        if complement:
            kernels['complement'][lo:hi] = .5*product+.5*(discarded@discarded_reference.T)+1.
    return prior, kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if not (out/'TRAINING_COMPLETE.json').exists():
        raise ValueError('finish both trained arms before readout')
    if (out/'summary.json').exists():
        raise ValueError('completed results must not be overwritten')
    protocol = json.loads((out/'protocol.json').read_text())
    target, previous = Path(protocol['target']), Path(protocol['previous'])
    frame = pd.read_csv(target/'target_manifest.csv', keep_default_na=False)
    base = np.load(target/'paired_r50.npy', mmap_mode='r')
    np.testing.assert_array_equal(np.load(out/'target_feature_rows.npy'), np.load(target/'feature_rows.npy'))
    with np.load(previous/'frozen_replay/policies.npz') as saved:
        old = {key: saved[key] for key in saved.files}
    with np.load(target/'policies.npz') as saved:
        np.testing.assert_array_equal(old['predictions'][:, 1], saved['predictions'][:, 1])
        for key in ('support_rows', 'eval_masks', 'caps', 'sample_ids'):
            np.testing.assert_array_equal(old[key], saved[key])
    np.testing.assert_array_equal(old['sample_ids'], frame.sample_id.to_numpy(str))
    with (out/'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    with np.load(out/'deployment_complement.npz') as saved:
        stats = {key: saved[key] for key in ('discarded_center', 'discarded_scale')}
    reference_rows = np.unique(old['support_rows'])
    replay_error = 0.
    started = time.monotonic()
    with threadpool_limits(limits=4):
        groups = [(previous/'frozen_replay/target_extra.npy', ['reference', 'complement_only']),
                  (out/'aligned_only/target_extra.npy', ['aligned_only']),
                  (out/'joint/target_extra.npy', ['joint'])]
        for path, arms in groups:
            extra = np.load(path, mmap_mode='r')
            uses_complement = any(arm in ('joint', 'complement_only') for arm in arms)
            prior, kernels = build_kernels(base, extra, model, stats, reference_rows, uses_complement)
            for arm in arms:
                folder = out/arm
                folder.mkdir(exist_ok=True)
                key = 'complement' if arm in ('joint', 'complement_only') else 'retained'
                draws = []
                for i, support in enumerate(old['support_rows']):
                    result = freeze_kernel_draw(old['base_scores'][i], prior,
                        kernels[key][:, np.searchsorted(reference_rows, support)], support,
                        frame.iloc[support].class_id.to_numpy(int), model['gate'], frame.target_manifest_row.to_numpy())
                    np.testing.assert_array_equal(result['eval_mask'], old['eval_masks'][i])
                    np.testing.assert_array_equal(result['predictions'] != old['predictions'][i, 1], result['selected'])
                    if result['selected'][~result['eval_mask']].any() or result['selected'].sum() > old['caps'][i]:
                        raise ValueError('support/budget violation')
                    if arm == 'reference':
                        replay_error = max(replay_error, float(np.max(np.abs(result['expert_scores']-old['expert_scores'][i]))))
                        np.testing.assert_allclose(result['expert_scores'], old['expert_scores'][i], atol=3e-5, rtol=0)
                        np.testing.assert_array_equal(result['predictions'], old['predictions'][i, 2])
                        np.testing.assert_array_equal(result['selected'], old['selected'][i])
                    draws.append(result)
                np.savez_compressed(folder/'policies.npz',
                    **{key: np.stack([d[key] for d in draws]) for key in draws[0]},
                    support_rows=old['support_rows'], sample_ids=old['sample_ids'], prior_expert_scores=prior)
                print(f'FROZEN {arm} all5draws', flush=True)
            del extra, kernels, draws, prior
        write_json(out/'POLICIES_SAVED.json', dict(arms=protocol['arms'],
            all_saved_before_query_readout=True, base64_predictions_unchanged=True,
            reference_predictions_and_admissions_exact=True, maximum_replay_score_error=replay_error,
            labels_per_draw=64, budget=.05, seconds=time.monotonic()-started))
        truth = frame.class_id.to_numpy(int)
        rows = []
        for arm in protocol['arms']:
            with np.load(out/arm/'policies.npz') as policy:
                for i in range(5):
                    ev, selected = policy['eval_mask'][i], policy['selected'][i]
                    baseline = old['predictions'][i, 1]
                    pred, expert = policy['predictions'][i], policy['expert_predictions'][i]
                    full = score_policy(pred, baseline, truth, ev)
                    em = score_policy(expert, baseline, truth, ev)
                    rows.append(dict(arm=arm, draw=i, **full, expert_acc=em['acc'], expert_ba=em['ba'],
                        selected=int(selected.sum()), cap=int(policy['cap'][i]),
                        evaluated=int(ev.sum()), rescued=int(np.sum(selected & (pred == truth) & (baseline != truth))),
                        harmed=int(np.sum(selected & (pred != truth) & (baseline == truth)))))
        means = []
        measures = ('acc', 'ba', 'expert_acc', 'expert_ba', 'selected', 'rescued', 'harmed')
        for arm in protocol['arms']:
            own = [row for row in rows if row['arm'] == arm]
            mean = dict(arm=arm, **{m: float(np.mean([r[m] for r in own])) for m in measures})
            means.append(mean)
            print(f'RESULT {arm} FULL OA={100*mean["acc"]:.6f} BA={100*mean["ba"]:.6f}; '
                  f'EXPERT OA={100*mean["expert_acc"]:.6f} BA={100*mean["expert_ba"]:.6f}', flush=True)
        contrasts = []
        for candidate, control in [('complement_only', 'reference'), ('aligned_only', 'reference'),
                ('joint', 'reference'), ('joint', 'aligned_only'), ('joint', 'complement_only')]:
            left = [r for r in rows if r['arm'] == candidate]
            right = [r for r in rows if r['arm'] == control]
            delta = {m: [100*(a[m]-b[m]) for a, b in zip(left, right)]
                     for m in ('acc', 'ba', 'expert_acc', 'expert_ba')}
            contrasts.append(dict(candidate=candidate, control=control, draw_deltas_pp=delta,
                means_pp={m: float(np.mean(v)) for m, v in delta.items()},
                positive_draws={m: int(np.count_nonzero(np.asarray(v) > 0)) for m, v in delta.items()}))
        mean_lookup = {row['arm']: row for row in means}
        interaction = {m: 100*(mean_lookup['joint'][m]-mean_lookup['aligned_only'][m]
                              -mean_lookup['complement_only'][m]+mean_lookup['reference'][m])
                       for m in ('acc', 'ba', 'expert_acc', 'expert_ba')}
        write_json(out/'metrics.json', rows)
        pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]).to_csv(out/'metrics.csv', index=False)
        write_json(out/'summary.json', dict(means=means, contrasts=contrasts, interaction_pp=interaction,
            statistical_unit='One historically used dataset, five overlapping support draws, one neural seed',
            independent_confirmation=False, joint_training_operator_matches_deployment=True,
            fixed_base=True, fixed_source_geometry=True, extra_target_labels=0))
        print(json.dumps(dict(contrasts=contrasts, interaction_pp=interaction), indent=2), flush=True)
    print('MATCHED JOINT PILOT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
