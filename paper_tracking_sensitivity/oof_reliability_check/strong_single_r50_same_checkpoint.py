"""Final bounded OSS2 diagnostic with the exact source-selected INNER models."""

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
from torch.nn import functional as F

from strong_single_r50_task import OUT, ROOT, CACHE, SingleR50, pixel_tensor, support_scores, metrics, write_json
from strong_single_r50_task_diagnose import ARMS, MODES, PATHS, pair_flow


def repair_source_counts(out, first_seed):
    with np.load(out/f'seed_{first_seed}'/'validation_episodes.npz') as saved:
        arrays = [saved[key] for key in saved.files if key.endswith(('_support', '_query'))]
        exposures = sum(len(array) for array in arrays)
        unique = len(np.unique(np.concatenate(arrays)))
    path = out/'source_diagnosis/COMPLETE.json'
    receipt = json.loads(path.read_text())
    receipt['unique_frozen_encoded_rows'] = unique
    receipt['frozen_encoded_row_exposures'] = exposures
    receipt['encoded_row_exposures'] = (1+len(receipt['seeds']))*exposures
    write_json(path, receipt)
    return dict(unique_frozen_encoded_rows=unique, frozen_encoded_row_exposures=exposures)


@torch.no_grad()
def run(out, seeds):
    torch.set_num_threads(2)
    started = time.perf_counter()
    destination = out/'source_diagnosis/target_same_checkpoint'
    destination.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        if not (out/f'seed_{seed}'/'COMPLETE.json').exists():
            raise ValueError(f'Original seed{seed} is incomplete')
    counts = repair_source_counts(out, seeds[0])
    print(json.dumps(counts), flush=True)
    with np.load(out/f'seed_{seeds[0]}'/'predictions.npz') as saved:
        data = {key: saved[key] for key in ('rows', 'ids', 'support', 'support_labels', 'eval_masks')}
    np.testing.assert_array_equal(np.load(CACHE/'oss2_ids.npy'), data['ids'])
    for draw, support in enumerate(data['support']):
        if len(support) != 64 or data['eval_masks'][draw, support].any():
            raise ValueError('Original64-label design mismatch')
    for seed in seeds[1:]:
        with np.load(out/f'seed_{seed}'/'predictions.npz') as saved:
            for key in data:
                np.testing.assert_array_equal(saved[key], data[key])
    write_json(destination/'PROTOCOL.json', dict(
        checkpoints='Exact selected_inner.pt per seed, not all-source final_tail.pt',
        frozen_prior='Previously fitted on inner_train ONLY, source_diagnosis/inner_frozen_prior.npy',
        frozen_phi='Reuse initial ImageNetV2 target geometry from seed20261010 frozen comparator',
        operator='Four fixed prior/kernel cells, each prior on its own representation, lambda8 balanced residual ridge',
        source_target_checkpoint_phase_controlled=True, hyperparameter_selection=False,
        new_training=False, query_truth_read_after_all_predictions=True, seeds=seeds,
        scope='Post-hoc diagnostic only; main all-source-refit results unchanged',
        extraction='C4, batch32/microbatch16, bf16 encoder, float32 geometry and GPU residual head'))
    device = torch.device('cuda')
    frozen_phi = torch.from_numpy(np.load(out/f'seed_{seeds[0]}'/f'{ARMS[0]}_phi.npy')).to(device)
    initial_weight = torch.from_numpy(np.load(out/'source_diagnosis/inner_frozen_prior.npy')).to(device)
    frozen_prior = F.linear(frozen_phi, initial_weight)
    pixels = np.load(CACHE/'oss2_inputs128.npy', mmap_mode='r')
    if len(pixels) != len(frozen_phi):
        raise ValueError('Frozen target geometry/pixel rows mismatch')
    net = SingleR50().to(device).eval()
    all_predictions, selected_epochs = {}, {}
    for seed in seeds:
        state = torch.load(out/f'seed_{seed}'/'selected_inner.pt', map_location='cpu', weights_only=False)
        selection = json.loads((out/f'seed_{seed}'/'SELECTION.json').read_text())
        if state['epoch'] != selection['epoch']:
            raise ValueError('Wrong inner checkpoint epoch')
        selected_epochs[seed] = int(state['epoch'])
        net.layer4.load_state_dict(state['layer4'])
        net.prior.load_state_dict(state['prior'])
        del state
        trained_phi = torch.empty_like(frozen_phi)
        for lo in range(0, len(pixels), 32):
            rows = np.arange(lo, min(lo+32, len(pixels)))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                trained_phi[lo:lo+len(rows)] = net(pixel_tensor(pixels, rows, device), microbatch=16)
            if lo % 1024 == 0:
                print(f'seed{seed} target{lo+len(rows)}/{len(pixels)} elapsed={time.perf_counter()-started:.1f}s', flush=True)
        trained_prior = net.prior(trained_phi)
        np.save(destination/f'seed_{seed}_inner_target_phi.npy', trained_phi.cpu().numpy())
        phis, priors = [frozen_phi, trained_phi], [frozen_prior, trained_prior]
        predictions = np.empty((5, 4, len(pixels)), dtype=np.int8)
        for draw, support in enumerate(data['support']):
            support_labels = torch.as_tensor(data['support_labels'][draw], device=device, dtype=torch.long)
            for index, (pi, ki) in enumerate(PATHS):
                phi, prior = phis[ki], priors[pi]
                score = support_scores(phi[support], support_labels, phi, strength=8.,
                                       support_prior=prior[support], query_prior=prior)
                if not torch.isfinite(score).all():
                    raise RuntimeError('Nonfinite diagnostic target scores')
                predictions[draw, index] = score.argmax(1).cpu().numpy()
        np.savez_compressed(destination/f'seed_{seed}_predictions.npz', predictions=predictions,
                            modes=np.array(MODES), selected_inner_epoch=selected_epochs[seed], **data)
        all_predictions[seed] = predictions
        print(f'seed{seed} all fixed predictions saved', flush=True)
    # This is the first target-query truth access, after all fixed predictions are saved.
    frame = pd.read_csv(ROOT/'opensarship2_transfer_20261007/target_manifest.csv', usecols=['sample_id', 'class_id'])
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str)[data['rows']], data['ids'])
    truth = frame.class_id.to_numpy()[data['rows']]
    records, classes, flows = [], [], []
    for seed in seeds:
        predictions = all_predictions[seed]
        for draw, mask in enumerate(data['eval_masks']):
            y = truth[mask]
            for index, mode in enumerate(MODES):
                candidate = predictions[draw, index, mask]
                context = dict(seed=seed, epoch=selected_epochs[seed], draw=draw, mode=mode)
                records.append(dict(**context, n=len(y), **metrics(y, candidate)))
                flow, by_class = pair_flow(y, candidate, predictions[draw, 0, mask])
                flows.append(dict(**context, **flow))
                support_counts = np.bincount(data['support_labels'][draw], minlength=8)
                classes.extend(dict(**context, **row, support_count=int(support_counts[row['class_id']])) for row in by_class)
    table = pd.DataFrame(records)
    table.to_csv(destination/'by_seed_draw_mode.csv', index=False)
    pd.DataFrame(classes).to_csv(destination/'per_class.csv', index=False)
    pd.DataFrame(flows).to_csv(destination/'rescue_harm.csv', index=False)
    seed_means = table.groupby(['seed', 'mode'], as_index=False)[['oa', 'ba']].mean()
    seed_means.to_csv(destination/'neural_seed_means.csv', index=False)
    summary = seed_means.groupby('mode').agg(oa_mean=('oa', 'mean'), oa_seed_sd=('oa', 'std'),
        ba_mean=('ba', 'mean'), ba_seed_sd=('ba', 'std')).reset_index()
    summary.to_csv(destination/'neural_summary.csv', index=False)
    source = pd.read_csv(out/'source_diagnosis/neural_seed_means.csv')
    source_means = source.groupby('mode')[['oa', 'ba']].mean()
    comparison = []
    for seed in seeds:
        source_seed = source[source.seed == seed].set_index('mode')
        target_seed = seed_means[seed_means.seed == seed].set_index('mode')
        for measure in ('oa', 'ba'):
            for context, begin, end in [('frozen_prior', MODES[0], MODES[2]), ('trained_prior', MODES[1], MODES[3])]:
                comparison.append(dict(seed=seed, metric=measure, prior_context=context,
                    source_kernel_delta_pp=source_seed.loc[end, measure]-source_seed.loc[begin, measure],
                    target_kernel_delta_pp=target_seed.loc[end, measure]-target_seed.loc[begin, measure]))
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(destination/'same_checkpoint_kernel_effects.csv', index=False)
    effects = comparison.groupby(['metric', 'prior_context'], as_index=False)[['source_kernel_delta_pp', 'target_kernel_delta_pp']].mean()
    effects.to_csv(destination/'kernel_effect_summary.csv', index=False)
    report(destination, summary, source_means, effects, comparison, counts, selected_epochs)
    torch.cuda.synchronize()
    write_json(destination/'COMPLETE.json', dict(seeds=seeds, selected_inner_epochs=selected_epochs,
        target_rows=len(pixels), newly_encoded_row_exposures=len(seeds)*len(pixels),
        image_forwards_per_row=4, frozen_target_geometry_reused=True, residual_lambda=8.,
        same_source_target_checkpoints=True, predictions_saved_before_target_truth=True,
        target_labels_used_for_selection=False, new_training=False, main_results_modified=False,
        seconds=time.perf_counter()-started, source_count_correction=counts))
    print(summary.to_string(index=False), flush=True)
    print(effects.to_string(index=False), flush=True)
    print(f'complete {time.perf_counter()-started:.1f}s', flush=True)


def report(destination, summary, source, effects, comparison, counts, epochs):
    lines = ['# Same-Checkpoint Source To OSS2 Diagnostic', '',
             'The exact selected inner checkpoints and inner-fitted initial prior now appear on both domains.',
             'No source refit, additional training, hyperparameter selection or main-result replacement.', '',
             '| Prior / kernel | Source OA | Source BA | OSS2 OA | OSS2 BA |',
             '| --- | ---: | ---: | ---: | ---: |']
    target = summary.set_index('mode')
    for mode in MODES:
        s, t = source.loc[mode], target.loc[mode]
        lines.append(f'| {mode} | {s.oa:.6f} | {s.ba:.6f} | {t.oa_mean:.6f} | {t.ba_mean:.6f} |')
    lines += ['', '## Kernel Effects With Training Phase Controlled', '',
              '| Metric | Prior | Source delta | OSS2 delta |', '| --- | --- | ---: | ---: |']
    for row in effects.itertuples(index=False):
        lines.append(f'| {row.metric.upper()} | {row.prior_context} | {row.source_kernel_delta_pp:+.6f}pp | {row.target_kernel_delta_pp:+.6f}pp |')
    ba_effects = effects[effects.metric == 'ba']
    reversals = (ba_effects.source_kernel_delta_pp > 0) & (ba_effects.target_kernel_delta_pp < 0)
    lines += ['', '## Discriminating Conclusion', '']
    if reversals.all():
        lines.append('The BA sign reversal persists under BOTH prior contexts for the SAME inner checkpoints. '
                     'It cannot be explained solely by all-source refitting. The learned support geometry '
                     'has held-source BA benefit but fails to transfer that benefit to OSS2 in this protocol.')
    elif reversals.any():
        lines.append('The BA sign reversal persists under one prior context with the SAME inner checkpoints. '
                     'All-source refitting alone is therefore insufficient for that path; the other path '
                     'has a different prior/kernel interaction and must be reported separately.')
    else:
        lines.append('The earlier BA sign reversal does not persist with this controlled checkpoint phase. '
                     'Refit behavior becomes a plausible contributor; this alone does not prove it is the unique cause.')
    lines += ['', 'This localizes the problem to empirical cross-domain transfer of the learned relation, '
              'not a demonstrated physical causal factor or a universal impossibility of single-backbone learning.',
              'Held-source OA/BA tradeoffs remain visible. Source checkpoint selection used these episodes, '
              'so positive source BA is selection-conditioned development evidence, not independent confirmation.',
              'The crossed cells require both frozen and trained feature functions during diagnosis; they '
              'are not automatically deployable as a single-backbone, single-forward model.', '',
              '## Seed Directions', '']
    for context in ('frozen_prior', 'trained_prior'):
        rows = comparison[(comparison.metric == 'ba') & (comparison.prior_context == context)]
        for name in ('source', 'target'):
            values = rows[f'{name}_kernel_delta_pp'].to_numpy()
            lines.append(f'- {context}, {name} BA kernel effect: {int((values > 0).sum())} positive / '
                         f'{int((values == 0).sum())} zero / {int((values < 0).sum())} negative neural seeds.')
    lines += ['', '## Metadata And Boundaries', '',
              f'Selected inner epochs: {epochs}. Each seed averages five fixed64-label target draws; '
              'neural-seed SD is computed after these averages.',
              f"Source frozen cache contains {counts['unique_frozen_encoded_rows']} unique row IDs across "
              f"{counts['frozen_encoded_row_exposures']} row exposures. Four rotations are separate forwards, not unique samples.",
              'Only source diagnosis count metadata was corrected. All-source-refit training/results and original '
              'target predictions remain unchanged. Diagnostic target predictions were persisted before query truth.',
              'Three inner-model target extractions only; initial frozen target geometry was reused.', '']
    (destination/'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seeds', nargs='+', type=int, default=[20261010, 20261011, 20261012])
    args = parser.parse_args()
    run(args.out, args.seeds)
