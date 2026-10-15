"""CPU-only, fixed-lambda prior/kernel crossover after all requested seeds finish."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from strong_single_r50_task import OUT, ROOT, metrics, support_scores, write_json

ARMS = ('frozen_single_same_readout', 'task_adapted_single_same_readout')
MODES = ('frozen_prior_frozen_kernel', 'trained_prior_frozen_kernel',
         'frozen_prior_trained_kernel', 'trained_prior_trained_kernel')
PATHS = ((0, 0), (1, 0), (0, 1), (1, 1))
REFERENCE = ROOT/'backbone_ablation_20261010/targets/oss2/oss2/predictions.npz'


@torch.no_grad()
def cross_predictions(phis, weights, supports, support_labels):
    """Each prior uses its own representation and head, independent of kernel arm."""
    priors = [F.linear(phi, weight) for phi, weight in zip(phis, weights)]
    predictions = np.empty((len(supports), 4, len(phis[0])), dtype=np.int8)
    for draw, support in enumerate(supports):
        labels = torch.as_tensor(support_labels[draw], dtype=torch.long)
        for mode, (pi, ki) in enumerate(PATHS):
            phi, prior = phis[ki], priors[pi]
            for lo in range(0, len(phi), 1024):
                scores = support_scores(phi[support], labels, phi[lo:lo+1024], strength=8.,
                                        support_prior=prior[support], query_prior=prior[lo:lo+1024])
                predictions[draw, mode, lo:lo+len(scores)] = scores.argmax(1).numpy()
    return predictions


def check_diagonals(predictions, saved, receipt=None):
    status = {}
    for mode, arm in ((0, ARMS[0]), (3, ARMS[1])):
        index = list(saved['arms']).index(arm)
        mismatch = predictions[:, mode] != saved['predictions'][:, index]
        status[arm] = dict(exact=not bool(mismatch.any()), total_mismatches=int(mismatch.sum()),
                           mismatches_per_draw=mismatch.sum(axis=1).tolist(),
                           first_mismatch_draw_row=np.argwhere(mismatch)[:20].tolist())
    if receipt is not None:
        write_json(receipt, dict(exact=all(item['exact'] for item in status.values()), arms=status,
                                precision='float32 CPU, same original primal residual operator',
                                original_predictions_modified=False))
    for mode, arm in ((0, ARMS[0]), (3, ARMS[1])):
        index = list(saved['arms']).index(arm)
        np.testing.assert_array_equal(predictions[:, mode], saved['predictions'][:, index],
                                      err_msg=f'Crossover diagonal failed exact replay: {arm}')


def pair_flow(truth, candidate, reference):
    """Positive net counts favor candidate; never count only its rescued examples."""
    old, new = reference == truth, candidate == truth
    rescue, harm = ~old & new, old & ~new
    classes = []
    for c in np.unique(truth):
        subset = truth == c
        r, h, n = int((rescue & subset).sum()), int((harm & subset).sum()), int(subset.sum())
        classes.append(dict(class_id=int(c), n=n, rescues=r, harms=h, net_count=r-h,
                            net_sign=int(np.sign(r-h)), recall_delta_pp=100*(r-h)/n,
                            candidate_recall=100*float(new[subset].mean()),
                            reference_recall=100*float(old[subset].mean())))
    result = dict(n=len(truth), rescues=int(rescue.sum()), harms=int(harm.sum()),
                  net_oa_pp=100*float((new.astype(float)-old.astype(float)).mean()),
                  net_ba_pp=float(np.mean([row['recall_delta_pp'] for row in classes])))
    candidate_metrics, reference_metrics = metrics(truth, candidate), metrics(truth, reference)
    for key in ('oa', 'ba'):
        np.testing.assert_allclose(result[f'net_{key}_pp'], candidate_metrics[key]-reference_metrics[key], atol=1e-10)
    return result, classes


def load_reference(data, path):
    with np.load(path) as saved:
        for key in ('rows', 'ids', 'support', 'eval_masks'):
            np.testing.assert_array_equal(saved[key], data[key])
        return saved['predictions'][:, list(saved['modes']).index('dual_refit_late')].copy()


def run(out, seeds, reference=REFERENCE, target_manifest=None, id_column='sample_id'):
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate seeds would duplicate evidence')
    for seed in seeds:
        if not (out/f'seed_{seed}'/'COMPLETE.json').exists():
            raise ValueError(f'Seed {seed} is not COMPLETE; no partial-seed diagnosis')
    torch.set_num_threads(2)
    destination = out/'diagnosis'
    destination.mkdir(parents=True, exist_ok=True)
    designs, all_predictions = {}, {}
    for seed in seeds:
        folder = out/f'seed_{seed}'
        with np.load(folder/'predictions.npz') as saved:
            data = {key: saved[key] for key in ('source_rows', 'rows', 'ids', 'support',
                    'support_labels', 'eval_masks', 'arms', 'predictions')}
        state = torch.load(folder/'final_tail.pt', map_location='cpu', weights_only=False)
        np.testing.assert_array_equal(state['source_rows'], data['source_rows'])
        weights = [state['frozen_prior']['weight'].detach().clone(), state['prior']['weight'].detach().clone()]
        del state
        phis = [torch.from_numpy(np.load(folder/f'{arm}_phi.npy', mmap_mode='c')) for arm in ARMS]
        if any(phi.dtype != torch.float32 for phi in phis) or any(w.dtype != torch.float32 for w in weights):
            raise ValueError('Replay requires original float32 CPU head operations')
        predictions = cross_predictions(phis, weights, data['support'], data['support_labels'])
        all_predictions[seed] = predictions
        designs[seed] = {key: data[key] for key in ('source_rows', 'rows', 'ids', 'support', 'support_labels', 'eval_masks')}
        np.savez_compressed(destination/f'seed_{seed}_predictions.npz', predictions=predictions,
                            modes=np.array(MODES), **designs[seed])
        check_diagonals(predictions, data, destination/f'seed_{seed}_DIAGONAL_REPLAY.json')
        del phis, weights
    anchor = designs[seeds[0]]
    for seed in seeds[1:]:
        for key in anchor:
            np.testing.assert_array_equal(designs[seed][key], anchor[key])
    dual = load_reference(anchor, reference)
    # All fixed crossover predictions and exact replays are persisted before query truth.
    if target_manifest is None:
        target_manifest = ROOT/'opensarship2_transfer_20261007/target_manifest.csv'
    frame = pd.read_csv(target_manifest, usecols=[id_column, 'class_id'])
    np.testing.assert_array_equal(frame[id_column].to_numpy(str)[anchor['rows']], anchor['ids'])
    truth = frame.class_id.to_numpy()[anchor['rows']]
    records, flows, classes, interactions, dual_metrics = [], [], [], [], []
    for draw, mask in enumerate(anchor['eval_masks']):
        dual_metrics.append(dict(draw=draw, n=int(mask.sum()), **metrics(truth[mask], dual[draw, mask])))
    for seed in seeds:
        predictions = all_predictions[seed]
        for draw, mask in enumerate(anchor['eval_masks']):
            y = truth[mask]
            scores_by_mode = []
            support_counts = np.bincount(anchor['support_labels'][draw], minlength=8)
            for index, mode in enumerate(MODES):
                candidate = predictions[draw, index, mask]
                score = metrics(y, candidate)
                scores_by_mode.append(score)
                records.append(dict(seed=seed, draw=draw, mode=mode, n=len(y), **score))
                for name, ref in [('frozen_single', predictions[draw, 0, mask]),
                                  ('current_matched_dual', dual[draw, mask])]:
                    flow, by_class = pair_flow(y, candidate, ref)
                    context = dict(seed=seed, draw=draw, mode=mode, reference=name)
                    flows.append(dict(**context, **flow, count_unit='within-draw query exposures'))
                    classes.extend(dict(**context, **row, support_count=int(support_counts[row['class_id']]))
                                   for row in by_class)
            for measure in ('oa', 'ba'):
                ff, tf, ft, tt = [score[measure] for score in scores_by_mode]
                interactions.append(dict(seed=seed, draw=draw, metric=measure,
                    prior_only_pp=tf-ff, kernel_only_pp=ft-ff, joint_pp=tt-ff,
                    nonadditivity_pp=tt-tf-ft+ff,
                    prior_effect_given_trained_kernel_pp=tt-ft,
                    kernel_effect_given_trained_prior_pp=tt-tf))
    table = pd.DataFrame(records)
    table.to_csv(destination/'metrics_by_seed_draw.csv', index=False)
    flow_table = pd.DataFrame(flows)
    flow_table.to_csv(destination/'rescue_harm_by_seed_draw.csv', index=False)
    pd.DataFrame(classes).to_csv(destination/'per_class_signs.csv', index=False)
    pd.DataFrame(interactions).to_csv(destination/'interactions_by_seed_draw.csv', index=False)
    reference_table = pd.DataFrame(dual_metrics)
    reference_table.to_csv(destination/'fixed_dual_by_draw.csv', index=False)
    seed_means = table.groupby(['seed', 'mode'], as_index=False)[['oa', 'ba']].mean()
    seed_means.to_csv(destination/'neural_seed_means.csv', index=False)
    summary = seed_means.groupby('mode').agg(oa_mean=('oa', 'mean'), oa_seed_sd=('oa', 'std'),
                                             ba_mean=('ba', 'mean'), ba_seed_sd=('ba', 'std'),
                                             neural_seeds=('seed', 'nunique')).reset_index()
    summary.to_csv(destination/'neural_summary.csv', index=False)
    table.groupby(['draw', 'mode'], as_index=False)[['oa', 'ba']].mean().to_csv(
        destination/'support_draw_means.csv', index=False)
    flow_seed = flow_table.groupby(['seed', 'mode', 'reference'], as_index=False)[['net_oa_pp', 'net_ba_pp']].mean()
    flow_seed.to_csv(destination/'net_effect_neural_seed_means.csv', index=False)
    flow_seed.groupby(['mode', 'reference']).agg(
        net_oa_mean=('net_oa_pp', 'mean'), net_oa_seed_sd=('net_oa_pp', 'std'),
        net_ba_mean=('net_ba_pp', 'mean'), net_ba_seed_sd=('net_ba_pp', 'std')).to_csv(destination/'net_effect_summary.csv')
    write_report(destination, summary, seed_means, reference_table, flow_seed, seeds)
    write_json(destination/'COMPLETE.json', dict(seeds=seeds, neural_seed_count=len(seeds),
        support_draws=len(anchor['support']), residual_lambda=8., diagonal_replay_exact=True,
        query_labels_used_for_selection=False, additional_training=False, device='cpu',
        fixed_dual_neural_fits=1, fixed_dual_reused_across_comparisons=True,
        interpretation='Functional prior/kernel crossover; not physical causality or architecture-only attribution'))
    print(summary.to_string(index=False), flush=True)
    return destination


def write_report(destination, summary, seed_means, dual, effects, seeds):
    lines = ['# Single R50 Functional Crossover', '',
             'Completed fixed-lambda8 diagnosis. No selection or training uses this report.', '',
             '## Results', '',
             '| Prior / Kernel | OA mean | Neural-seed SD | BA mean | Neural-seed SD |',
             '| --- | ---: | ---: | ---: | ---: |']
    for mode in MODES:
        row = summary[summary['mode'] == mode].iloc[0]
        lines.append(f"| {mode} | {row.oa_mean:.5f} | {row.oa_seed_sd:.5f} | {row.ba_mean:.5f} | {row.ba_seed_sd:.5f} |")
    lines += ['', f'Neural seeds: {seeds}. First average five support draws within each seed;',
              'then compute mean and sample SD (ddof1) over neural-seed means.',
              'Support draws are not independent neural training runs.', '',
              f"Existing matched dual reference, five-draw mean: OA {dual.oa.mean():.5f}, BA {dual.ba.mean():.5f}.",
              'This is one existing trained reference, reused across comparisons, NOT three dual retrainings.', '',
              '## Direction Across Seeds', '']
    for name in ('frozen_single', 'current_matched_dual'):
        rows = effects[(effects['mode'] == MODES[3]) & (effects.reference == name)]
        for measure in ('oa', 'ba'):
            values = rows[f'net_{measure}_pp'].to_numpy()
            lines.append(f"- Trained full classifier versus {name}, {measure.upper()}: mean {values.mean():+.5f}pp; "
                         f"positive/zero/negative seed means: {int((values > 0).sum())}/{int((values == 0).sum())}/{int((values < 0).sum())}.")
    lines += ['', '## What The Crossover Means', '',
              '- Frozen/trained prior means each source head evaluated on its OWN corresponding embedding.',
              '  It is the complete prior function, not a head-weight-only change.',
              '- Frozen/trained kernel means the corresponding feature geometry used to relate query and supports.',
              '- All four cells re-solve the same balanced64-label residual system, with lambda8 unchanged.',
              '  The two original cells exactly reproduce stored predictions using original float32 torch CPU operations.',
              '- Prior-only and kernel-only deltas locate functional changes. Their interaction is reported;',
              '  nonlinear accuracy effects need not add. Crossed cells are diagnostics, not target-selected deployment choices.',
              '- The analysis does not identify distinct physical mechanisms, prove causality, or equate pretraining histories.',
              '  The current dual comparison is a practical reference, not a matched architecture-only intervention.', '',
              '## Error Accounting', '',
              '`rescue_harm_by_seed_draw.csv` counts candidate-correct/reference-wrong rescues and the reverse harms.',
              'OA difference and class-balanced net rescue differences are checked against exact classifier metrics.',
              '`per_class_signs.csv` includes both references, class recalls, support coverage, net counts and their signs.',
              'Counts are within-draw exposures; overlapping queries across five draws are not independent samples.',
              'Use actual seed curves, CACHE_ALIGNMENT.json and prior/tail update norms to interpret a negative result;',
              'the crossover alone does not prove a training bug or that a research direction is impossible.', '',
              '## Files', '',
              '- `metrics_by_seed_draw.csv`: four complete eight-class classifiers, each seed and support draw.',
              '- `neural_seed_means.csv` / `neural_summary.csv`: five-draw means then neural-seed mean and SD.',
              '- `support_draw_means.csv`: support-draw results averaged across neural seeds, kept separately.',
              '- `interactions_by_seed_draw.csv`: prior/kernel conditional effects and nonadditivity.',
              '- `net_effect_summary.csv`: paired metric differences versus frozen single and fixed matched dual.',
              '- `seed_*_predictions.npz`: persisted fixed crossover predictions and exact common row/support metadata.', '']
    (destination/'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seeds', nargs='+', type=int, default=[20261010, 20261011, 20261012])
    args = parser.parse_args()
    run(args.out, args.seeds)


if __name__ == '__main__':
    main()
