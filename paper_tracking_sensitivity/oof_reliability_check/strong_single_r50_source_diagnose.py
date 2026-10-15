"""One bounded source-held prior/kernel crossover; no training or target labels."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from strong_single_r50_task import (
    OUT, ROOT, SOURCE, CACHE, SingleR50, initialize_prior, pixel_tensor,
    support_scores, metrics, write_json,
)
from strong_single_r50_task_diagnose import MODES, PATHS


@torch.no_grad()
def encode_episode(net, pixels, support, query, device):
    net.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        phi = net(pixel_tensor(pixels, np.r_[support, query], device), microbatch=16)
    return phi, net.prior(phi)


def comparison_report(destination, summary, effects, checks, out):
    target = pd.read_csv(out/'diagnosis/neural_summary.csv').set_index('mode')
    source = summary.set_index('mode')
    lines = ['# Source-Held Prior/Kernel Crossover', '',
             'One bounded diagnostic using the original eight held-source episodes per neural seed.',
             'No new training, target-query labels, or target-driven parameter selection.', '',
             '| Prior / kernel | Source OA | Source BA | Target OA | Target BA |',
             '| --- | ---: | ---: | ---: | ---: |']
    for mode in MODES:
        s, t = source.loc[mode], target.loc[mode]
        lines.append(f'| {mode} | {s.oa_mean:.6f} | {s.ba_mean:.6f} | {t.oa_mean:.6f} | {t.ba_mean:.6f} |')
    lines += ['', '## Kernel Effects', '',
              '| Metric | Prior context | Source kernel change | Target kernel change |',
              '| --- | --- | ---: | ---: |']
    for metric in ('oa', 'ba'):
        for context, start, end in [('frozen', MODES[0], MODES[2]), ('trained', MODES[1], MODES[3])]:
            sdelta = source.loc[end, f'{metric}_mean']-source.loc[start, f'{metric}_mean']
            tdelta = target.loc[end, f'{metric}_mean']-target.loc[start, f'{metric}_mean']
            lines.append(f'| {metric.upper()} | {context} | {sdelta:+.6f}pp | {tdelta:+.6f}pp |')
    sf = float(source.loc[MODES[2], 'ba_mean']-source.loc[MODES[0], 'ba_mean'])
    st = float(source.loc[MODES[3], 'ba_mean']-source.loc[MODES[1], 'ba_mean'])
    tf = float(target.loc[MODES[2], 'ba_mean']-target.loc[MODES[0], 'ba_mean'])
    tt = float(target.loc[MODES[3], 'ba_mean']-target.loc[MODES[1], 'ba_mean'])
    lines += ['', '## Interpretation', '']
    if sf < 0 and st < 0:
        lines.append('The trained kernel already hurts mean held-source BA under BOTH prior contexts. '
                     'A target-only transfer explanation is insufficient. This supports investigating '
                     'joint-training objective/representation interference or held-source generalization; '
                     'it does not by itself prove an implementation bug or isolate which loss caused it.')
    elif sf > 0 and st > 0 and tf < 0 and tt < 0:
        lines.append('The trained kernel helps mean held-source BA under BOTH prior contexts but hurts '
                     'target BA. The direction is consistent with transfer-specific failure, not a kernel '
                     'that is generally worse already on held-source validation.')
    else:
        lines.append('The kernel effects depend on prior context or domain. Do not reduce this to '
                     'a single source-interference versus target-transfer label; inspect the two contexts '
                     'and per-seed/per-class records together.')
    lines += ['', 'OA already declines on held source, while mean BA increases. Therefore there is '
              'also a source-side OA/BA tradeoff; the positive source BA must not be described as '
              'uniform improvement. The additional BA sign reversal on target is a separate observation.']
    for key, name in [('kernel_only_pp', 'frozen prior'), ('kernel_given_trained_prior_pp', 'trained prior')]:
        values = np.array([row[key] for row in effects if row['metric'] == 'ba'])
        lines.append(f'Source kernel BA effects under {name}: {int((values > 0).sum())} positive, '
                     f'{int((values == 0).sum())} zero, {int((values < 0).sum())} negative neural seeds.')
    lines += ['', 'Kernel/prior swaps are functional interventions, not physical causal identification.',
              'The source run uses selected INNER checkpoints and an INNER-training fitted frozen prior.',
              'The target run uses ALL-allowed-source refits. This deliberate phase difference means a '
              'sign reversal could also involve refit behavior; the comparison cannot isolate domain '
              'shift from refitting without further evidence.',
              'Held-source episodes selected the checkpoint, so their results are development evidence, '
              'not independent confirmation.', '', '## Replay And Scope', '']
    lines.append(f"Original OA/BA diagonal replay exact (tolerance1e-9): {checks['all_metric_diagonals_exact']}; "
                 f"maximum absolute OA/BA difference: {checks['max_metric_difference_pp']:.12g}pp.")
    lines += ['The original curve stored metrics, not per-sample validation predictions. '
              'Metric equality is NOT a claim of bitwise equality of unavailable original predictions.',
              'Encoding repeats the original192-row episode, microbatch16, four rotations, bf16 forward '
              'and float32 GPU prior/residual operations. No cache/dtype algebra substitution is made.',
              'The frozen eight episode embeddings/priors are reused across seeds after checking '
              'the original saved episodes match. The source prior was fitted once on inner training only.',
              'Each seed first averages eight episodes. Means/SD then aggregate neural seeds; '
              'the shared frozen reference is not three independent frozen trainings.', '',
              '## Files', '',
              '- `by_seed_episode_mode.csv`: every source OA/BA.',
              '- `per_class.csv`: recalls, support coverage and net rescue/harm signs versus frozen single.',
              '- `neural_seed_means.csv`, `neural_summary.csv`: episode averages then neural mean/SD.',
              '- `effects_by_seed.csv`: prior-only/kernel-only/joint deltas and conditional effects.',
              '- `DIAGONAL_CHECKS.json`: original versus replay OA/BA and every discrepancy.',
              '- `seed_*_predictions.npz`: query predictions, support/query rows and source episode metadata.', '']
    (destination/'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')


@torch.no_grad()
def run(out, seeds):
    started = time.perf_counter()
    torch.set_num_threads(2)
    for seed in seeds:
        if not (out/f'seed_{seed}'/'COMPLETE.json').exists():
            raise ValueError(f'Seed {seed} is not complete')
    destination = out/'source_diagnosis'
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(out/'source_split.npz') as saved:
        inner_train, inner_val = saved['inner_train'], saved['inner_val']
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id'])
    labels = frame.class_id.to_numpy(np.int64)
    episodes = []
    with np.load(out/f'seed_{seeds[0]}'/'validation_episodes.npz') as saved:
        for index in range(8):
            support, query = saved[f'{index}_support'], saved[f'{index}_query']
            if len(support) != 64 or np.intersect1d(support, query).size:
                raise ValueError('Saved source episode support/query mismatch')
            combined = np.r_[support, query]
            if not np.isin(combined, inner_val).all() or np.intersect1d(combined, inner_train).size:
                raise ValueError('Saved source episode crosses inner training boundary')
            episodes.append((support, query))
    for seed in seeds[1:]:
        with np.load(out/f'seed_{seed}'/'validation_episodes.npz') as saved:
            for index, (support, query) in enumerate(episodes):
                np.testing.assert_array_equal(saved[f'{index}_support'], support)
                np.testing.assert_array_equal(saved[f'{index}_query'], query)
    pixels = np.load(CACHE/'source_inputs128.npy', mmap_mode='r')
    frozen_features = np.load(ROOT/'c4_base_trial_20261009/pooled_c4.npy', mmap_mode='r')
    device = torch.device('cuda')
    net = SingleR50().to(device).eval()
    initialize_prior(net, frozen_features, labels, inner_train)
    np.save(destination/'inner_frozen_prior.npy', net.prior.weight.cpu().numpy())
    frozen = []
    print(f'inner source prior fitted on {len(inner_train)} rows; encoding frozen8episodes', flush=True)
    for index, (support, query) in enumerate(episodes):
        phi, prior = encode_episode(net, pixels, support, query, device)
        frozen.append((phi.cpu(), prior.cpu()))
        print(f'frozen episode {index+1}/8 elapsed={time.perf_counter()-started:.1f}s', flush=True)
    records, class_records, diagonal_checks, effects = [], [], [], []
    for seed in seeds:
        folder = out/f'seed_{seed}'
        state = torch.load(folder/'selected_inner.pt', map_location='cpu', weights_only=False)
        selection = json.loads((folder/'SELECTION.json').read_text())
        if state['epoch'] != selection['epoch']:
            raise ValueError('Selected inner checkpoint epoch differs from source selection')
        curve = json.loads((folder/'curve.json').read_text())
        curve0 = next(row for row in curve if row['epoch'] == 0)
        chosen = next(row for row in curve if row['epoch'] == state['epoch'])
        net.layer4.load_state_dict(state['layer4'])
        net.prior.load_state_dict(state['prior'])
        predictions, saved_rows = {}, {}
        for index, (support, query) in enumerate(episodes):
            trained_phi, trained_prior = encode_episode(net, pixels, support, query, device)
            frozen_phi, frozen_prior = [x.to(device) for x in frozen[index]]
            phis, priors = [frozen_phi, trained_phi], [frozen_prior, trained_prior]
            support_labels = torch.as_tensor(labels[support], device=device)
            episode_preds, episode_scores = [], []
            context = dict(seed=seed, selected_epoch=int(state['epoch']), episode=index,
                           port=curve0['validation'][index]['port'], draw=curve0['validation'][index]['draw'])
            for mode, (pi, ki) in zip(MODES, PATHS):
                phi, prior = phis[ki], priors[pi]
                scores = support_scores(phi[:64], support_labels, phi[64:], strength=8.,
                                        support_prior=prior[:64], query_prior=prior[64:])
                prediction = scores.argmax(1).cpu().numpy()
                episode_preds.append(prediction)
                score = metrics(labels[query], prediction)
                episode_scores.append(score)
                records.append(dict(**context, mode=mode, n=len(query), **score))
                for c in np.unique(labels[query]):
                    subset = labels[query] == c
                    old, new = episode_preds[0] == labels[query], prediction == labels[query]
                    rescue, harm = int((~old & new & subset).sum()), int((old & ~new & subset).sum())
                    class_records.append(dict(**context, mode=mode, class_id=int(c), n=int(subset.sum()),
                        recall=100*float(new[subset].mean()), support_count=int((labels[support] == c).sum()),
                        rescues=rescue, harms=harm, net_count=rescue-harm, net_sign=int(np.sign(rescue-harm))))
            predictions[f'{index}_predictions'] = np.stack(episode_preds)
            saved_rows[f'{index}_support'], saved_rows[f'{index}_query'] = support, query
            for mi, original in [(0, curve0), (3, chosen)]:
                expected = original['validation'][index]
                actual = episode_scores[mi]
                differences = {key: actual[key]-expected[key] for key in ('oa', 'ba')}
                diagonal_checks.append(dict(**context, mode=MODES[mi], expected_oa=expected['oa'],
                    expected_ba=expected['ba'], actual_oa=actual['oa'], actual_ba=actual['ba'],
                    oa_difference_pp=differences['oa'], ba_difference_pp=differences['ba'],
                    metrics_exact=all(abs(value) <= 1e-9 for value in differences.values())))
            print(f'seed{seed} source episode{index+1}/8 epoch{state["epoch"]} elapsed={time.perf_counter()-started:.1f}s', flush=True)
        np.savez_compressed(destination/f'seed_{seed}_predictions.npz', modes=np.array(MODES), **predictions, **saved_rows)
        del state
    table = pd.DataFrame(records)
    table.to_csv(destination/'by_seed_episode_mode.csv', index=False)
    pd.DataFrame(class_records).to_csv(destination/'per_class.csv', index=False)
    seed_means = table.groupby(['seed', 'mode'], as_index=False)[['oa', 'ba']].mean()
    seed_means.to_csv(destination/'neural_seed_means.csv', index=False)
    summary = seed_means.groupby('mode').agg(oa_mean=('oa', 'mean'), oa_seed_sd=('oa', 'std'),
        ba_mean=('ba', 'mean'), ba_seed_sd=('ba', 'std')).reset_index()
    summary.to_csv(destination/'neural_summary.csv', index=False)
    for seed in seeds:
        values = seed_means[seed_means.seed == seed].set_index('mode')
        for metric in ('oa', 'ba'):
            ff, tf, ft, tt = [values.loc[mode, metric] for mode in MODES]
            effects.append(dict(seed=seed, metric=metric, prior_only_pp=tf-ff, kernel_only_pp=ft-ff,
                                joint_pp=tt-ff, kernel_given_trained_prior_pp=tt-tf,
                                prior_given_trained_kernel_pp=tt-ft))
    pd.DataFrame(effects).to_csv(destination/'effects_by_seed.csv', index=False)
    checks = dict(all_metric_diagonals_exact=all(row['metrics_exact'] for row in diagonal_checks),
                  max_metric_difference_pp=max(abs(row[f'{metric}_difference_pp']) for row in diagonal_checks for metric in ('oa', 'ba')),
                  original_prediction_arrays_available=False, checks=diagonal_checks)
    write_json(destination/'DIAGONAL_CHECKS.json', checks)
    comparison_report(destination, summary, effects, checks, out)
    torch.cuda.synchronize()
    write_json(destination/'COMPLETE.json', dict(seeds=seeds, source_episodes_per_seed=8,
        unique_frozen_encoded_rows=len(np.unique(np.concatenate([np.r_[s, q] for s, q in episodes]))),
        frozen_encoded_row_exposures=sum(len(s)+len(q) for s, q in episodes),
        encoded_row_exposures=(1+len(seeds))*sum(len(s)+len(q) for s, q in episodes),
        forwards_per_row=4, inner_prior_rows=len(inner_train), target_labels_used=False,
        extra_training=False, selected_inner_checkpoints=True, metric_diagonals_exact=checks['all_metric_diagonals_exact'],
        seconds=time.perf_counter()-started, residual_lambda=8.))
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(dict(diagonals_exact=checks['all_metric_diagonals_exact'], seconds=time.perf_counter()-started)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seeds', nargs='+', type=int, default=[20261010, 20261011, 20261012])
    args = parser.parse_args()
    run(args.out, args.seeds)
