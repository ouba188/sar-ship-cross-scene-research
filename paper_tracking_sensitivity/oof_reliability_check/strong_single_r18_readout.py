"""Fixed CPU readout control on cached frozen C4-R50 and task-trained SSL-R18.

Only source heads and the original five support residuals are fitted. No neural
training, feature extraction, validation selection or CUDA execution is used.
"""

import os
import sys
sys.dont_write_bytecode = True
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

from unified_head_core import coefficients

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
SOURCE = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
TASKS = ROOT / 'learned_frequency_residual_20261009/tasks'
OUT = ROOT / 'strong_single_multidata_20261010/r18_control'
MODES = ('r50_frozen_c4', 'r18_task_trained', 'concat_shared_head', 'equal_single_softmax')
REFERENCE = 'current_dual_reference'
CLASSES = ('bulk_carrier', 'fishing_vessel', 'general_cargo', 'product_chemical_tanker',
           'container_ship', 'crude_oil_tanker', 'tug_towing', 'offshore_supply')
DESIGN_KEYS = ('source_rows', 'rows', 'ids', 'support', 'support_labels', 'eval_masks')
BATCH = 512
torch.set_num_threads(2)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def tasks():
    result = []
    for path in sorted(TASKS.rglob('PREPARED.json')):
        info = read_json(path)
        if info['dataset'] in ('self24ports', 'f0921', 'oss2'):
            result.append((info, path.parent))
    if len(result) != 25 or sum(i['dataset'] == 'self24ports' for i, _ in result) != 23:
        raise ValueError('Expected exactly 23 LOPO + F + OSS2 task designs')
    return result


def unit(raw):
    values = torch.from_numpy(np.array(raw, dtype=np.float32))
    return F.normalize(values, dim=1, eps=1e-12).numpy().astype(np.float64)


class Inputs:
    """Feature reads are bounded blocks; global and already-subset rows stay distinct."""

    def __init__(self, info, folder, source_frame):
        self.info, self.folder = info, folder
        self.dataset, self.task = info['dataset'], info['task']
        self.suffix = f'{self.dataset}/{self.task}'
        with np.load(folder / 'target.npz', allow_pickle=False) as z:
            self.data = {key: z[key] for key in DESIGN_KEYS}
        self.source_rows, self.rows = self.data['source_rows'], self.data['rows']
        self.source_labels = source_frame.class_id.to_numpy(int)[self.source_rows]
        self.r50_source_path = ROOT / 'c4_base_trial_20261009/pooled_c4.npy'
        self.r50_source = np.load(self.r50_source_path, mmap_mode='r')
        if self.r50_source.shape != (len(source_frame), 2048):
            raise ValueError('Source C4 shape differs from established manifest order')
        self.r18_source_subset = self.dataset != 'f0921'
        if self.dataset == 'self24ports':
            directory = ROOT / 'evidence_aware_handoff_20261008/source_features' / self.task
            self.r18_source_path = directory / 'features.npy'
            self.r18_source = np.load(self.r18_source_path, mmap_mode='r')
            np.testing.assert_array_equal(np.load(directory / 'rows.npy'), self.source_rows)
            parent = ROOT / 'protected_drift_activation_20261008/ports' / self.task
            np.testing.assert_array_equal(np.load(parent / 'source_rows.npy'), self.source_rows)
            np.testing.assert_array_equal(np.load(parent / 'target_rows.npy'), self.rows)
            if (source_frame.iloc[self.source_rows].port == self.task).any():
                raise ValueError('Held target port enters source fit')
            np.testing.assert_array_equal(source_frame.instance_id.to_numpy(str)[self.rows], self.data['ids'])
            self.r50_target_path = self.r50_source_path
            self.r50_target = self.r50_source
            self.r18_target_path = ROOT / 'module_completion_20261008/ports_projection' / self.task / 'matched_joint/joint_extra.npy'
            self.r18_checkpoint = ROOT / 'module_completion_20261008/ports_projection' / self.task / 'joint/layer4.pt'
            self.source_row_file = directory / 'rows.npy'
        else:
            transfer, joint = {
                'oss2': ('opensarship2_transfer_20261007', 'module_completion_20261008/oss2/joint'),
                'f0921': ('f0921_transfer_20261007', 'joint_residual_pilot_20261008/joint'),
            }[self.dataset]
            self.source_row_file = ROOT / transfer / 'source_split.npz'
            with np.load(self.source_row_file) as z:
                np.testing.assert_array_equal(z['source_rows'], self.source_rows)
                if np.intersect1d(self.source_rows, z['excluded_source_rows']).size:
                    raise ValueError('Excluded source rows enter readout fit')
            self.r18_checkpoint = ROOT / joint / 'layer4.pt'
            self.r18_target_path = ROOT / joint / 'target_extra.npy'
            self.r18_source_path = ROOT / joint / 'source_extra.npy'
            if self.dataset == 'oss2':
                self.r18_source_path = ROOT / 'compact_fusion_20261010/cache/oss2_source_extra.npz'
                with np.load(self.r18_source_path) as z:
                    np.testing.assert_array_equal(z['rows'], self.source_rows)
                    self.r18_source = z['features']  # Only the existing 72 MiB float32 source subset.
            else:
                self.r18_source = np.load(self.r18_source_path, mmap_mode='r')
            self.r50_target_path = ROOT / 'cross_dataset_c4_20261009' / self.dataset / 'pooled_c4.npy'
            self.r50_target = np.load(self.r50_target_path, mmap_mode='r')
            np.testing.assert_array_equal(np.load(self.r50_target_path.parent / 'sample_ids.npy'), self.data['ids'])
            np.testing.assert_array_equal(self.rows, np.arange(len(self.rows)))
        self.r18_target = np.load(self.r18_target_path, mmap_mode='r')
        expected_source = len(self.source_rows) if self.r18_source_subset else len(source_frame)
        if self.r18_source.shape != (expected_source, 512) or self.r18_target.shape != (len(self.rows), 512):
            raise ValueError('R18 feature rows or widths do not match task metadata')
        if not self.r18_checkpoint.exists():
            raise FileNotFoundError(self.r18_checkpoint)
        if self.data['support'].shape != (5, 64) or self.data['eval_masks'].shape != (5, len(self.rows)):
            raise ValueError('Expected established five x64 support design')
        if self.data['eval_masks'].dtype != np.bool_:
            raise ValueError('Expected boolean query masks')
        for support, mask in zip(self.data['support'], self.data['eval_masks']):
            if len(np.unique(support)) != 64 or mask[support].any():
                raise ValueError('Repeated support or support/query overlap')
        self.reference_path = ROOT / 'backbone_ablation_20261010/targets' / self.suffix / 'predictions.npz'
        with np.load(self.reference_path) as z:
            for key in ('rows', 'ids', 'support', 'eval_masks'):
                np.testing.assert_array_equal(self.data[key], z[key])
            self.reference = z['predictions'][:, list(z['modes']).index('dual_refit_late')]

    def source_units(self, index):
        rows = self.source_rows[index]
        r18_rows = index if self.r18_source_subset else rows
        return unit(self.r50_source[rows]), unit(self.r18_source[r18_rows])

    def target_units(self, index):
        return unit(self.r50_target[self.rows[index]]), unit(self.r18_target[index])

    def provenance(self):
        return dict(dataset=self.dataset, task=self.task, source_n=len(self.source_rows), target_n=len(self.rows),
            r50_source=str(self.r50_source_path), r50_target=str(self.r50_target_path),
            r18_source=str(self.r18_source_path), r18_target=str(self.r18_target_path),
            r18_source_already_subset=self.r18_source_subset, source_rows=str(self.source_row_file),
            r18_task_checkpoint=str(self.r18_checkpoint), original_target_design=str(self.folder / 'target.npz'),
            current_dual_reference=str(self.reference_path), r50_source_shape=list(self.r50_source.shape),
            r18_source_shape=list(self.r18_source.shape), r50_target_shape=list(self.r50_target.shape),
            r18_target_shape=list(self.r18_target.shape), dtype_source_r18=str(self.r18_source.dtype))


def fit_from_moments(gram, rhs, r50_width, alpha=1.):
    """Solve in unit coordinates: alpha/s^2 equals alpha on scaled coefficients.

    The last unit coordinate is 1. Its source penalty is zero for every arm.
    Three separate heads are fitted; the joint head is not an average of heads.
    """
    end = len(gram) - 1
    columns = (np.r_[np.arange(r50_width), end], np.arange(r50_width, end + 1), np.arange(end + 1))
    heads = []
    for ix, scale_squared in zip(columns, (32., 32., 16.)):
        matrix = gram[np.ix_(ix, ix)].copy()
        matrix.flat[::len(ix) + 1] += np.r_[np.full(len(ix)-1, alpha/scale_squared), 0.]
        heads.append(cho_solve(cho_factor(matrix, lower=True, check_finite=False), rhs[ix], check_finite=False))
    return heads


def fit_heads(inputs):
    labels = inputs.source_labels
    counts = np.bincount(labels, minlength=8)
    if (counts == 0).any():
        raise ValueError('Allowed source cohort lacks a class')
    weights = len(labels) / (8 * counts)
    gram = np.zeros((2561, 2561), dtype=np.float64)
    rhs = np.zeros((2561, 8), dtype=np.float64)
    for lo in range(0, len(labels), BATCH):
        hi = min(lo+BATCH, len(labels))
        r, s = inputs.source_units(slice(lo, hi))
        design = np.c_[r, s, np.ones(hi-lo)]
        if not np.isfinite(design).all():
            raise ValueError('Nonfinite allowed-source features')
        weighted = design * weights[labels[lo:hi], None]
        gram += design.T @ weighted
        rhs += weighted.T @ (2*np.eye(8)[labels[lo:hi]] - 1)
    return fit_from_moments(gram, rhs, 2048), counts


def priors(r, s, heads):
    return (r @ heads[0][:-1] + heads[0][-1], s @ heads[1][:-1] + heads[1][-1],
            r @ heads[2][:r.shape[1]] + s @ heads[2][r.shape[1]:-1] + heads[2][-1])


def kernels(r, s, rr, ss):
    kr, ks = r @ rr.T, s @ ss.T
    return 32*kr+1, 32*ks+1, 16*(kr+ks)+1


def bounded_check(inputs):
    """Probe both ends and supports to expose row-mode and streamed-F shape mistakes."""
    ns, nt = len(inputs.source_rows), len(inputs.rows)
    si = np.unique(np.r_[np.arange(min(8, ns)), np.arange(max(0, ns-8), ns)])
    ti = np.unique(np.r_[np.arange(min(8, nt)), np.arange(max(0, nt-8), nt), inputs.data['support'][0]])
    source = inputs.source_units(si)
    target = inputs.target_units(ti)
    for arrays in (source, target):
        for values in arrays:
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite feature probe')
            norms = np.linalg.norm(values, axis=1)
            if not np.all((norms < 1e-12) | (abs(norms-1) < 2e-6)):
                raise ValueError('Unit feature probe failed')
    return dict(**inputs.provenance(), source_probe=len(si), target_probe=len(ti),
                target_block_widths=[v.shape[1] for v in target], aligned=True,
                max_feature_read_batch=BATCH, full_float64_feature_copy=False)


def predict(inputs, destination):
    predicted = destination / 'PREDICTED.json'
    if predicted.exists():
        receipt = read_json(predicted)
        if receipt['modes'] != list(MODES) or receipt['operator_version'] != 1:
            raise ValueError('Existing prediction protocol differs')
        print('REUSE ' + inputs.suffix, flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(destination / 'INPUTS.json', inputs.provenance())
    print('FIT ' + inputs.suffix + f' source={len(inputs.source_rows)} target={len(inputs.rows)}', flush=True)
    heads, counts = fit_heads(inputs)
    source_seconds = time.perf_counter()-started
    np.savez(destination / 'source_heads.npz', **{mode: head for mode, head in zip(MODES[:3], heads)},
             source_rows=inputs.source_rows, source_class_counts=counts,
             coordinate_convention=np.array('unit-feature coefficients, appended unpenalized source bias'),
             scaled_feature_norm_squared=np.array([32., 32., 16.]))
    union = np.unique(inputs.data['support'])
    rr, ss = inputs.target_units(union)
    pp = priors(rr, ss, heads)
    kk = kernels(rr, ss, rr, ss)
    states, stored_coefficients = [], {}
    for draw, support in enumerate(inputs.data['support']):
        cols = np.searchsorted(union, support)
        alpha = [coefficients(k[np.ix_(cols, cols)], p[cols], inputs.data['support_labels'][draw], 8.)
                 for p, k in zip(pp, kk)]
        states.append((cols, alpha))
        for mode, a in zip(MODES[:3], alpha):
            stored_coefficients[f'{draw}_{mode}'] = a
    np.savez(destination / 'support_coefficients.npz', **stored_coefficients)
    n = len(inputs.rows)
    scores = np.lib.format.open_memmap(destination / 'scores.npy', mode='w+', dtype=np.float64,
                                      shape=(5, 4, n, 8))
    saved_priors = np.lib.format.open_memmap(destination / 'priors.npy', mode='w+', dtype=np.float64, shape=(3, n, 8))
    predictions = np.empty((5, 4, n), dtype=np.int8)
    margins = np.empty((5, 4, n), dtype=np.float64)
    for lo in range(0, n, BATCH):
        hi = min(lo+BATCH, n)
        r, s = inputs.target_units(slice(lo, hi))
        p = priors(r, s, heads)
        k = kernels(r, s, rr, ss)
        saved_priors[:, lo:hi] = np.stack(p)
        for draw, (cols, alpha) in enumerate(states):
            values = [prior + kernel[:, cols] @ a for prior, kernel, a in zip(p, k, alpha)]
            values.append(.5*softmax(values[0], axis=1) + .5*softmax(values[1], axis=1))
            for mi, value in enumerate(values):
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite fixed readout scores')
                scores[draw, mi, lo:hi] = value
                predictions[draw, mi, lo:hi] = value.argmax(1)
                top = np.sort(value, axis=1)[:, -2:]
                margins[draw, mi, lo:hi] = top[:, 1]-top[:, 0]
        if lo and lo % (BATCH*64) == 0:
            print(f'STREAM {inputs.suffix} {hi}/{n}', flush=True)
    scores.flush()
    saved_priors.flush()
    del scores, saved_priors
    np.savez_compressed(destination / 'predictions.npz', predictions=predictions, modes=np.array(MODES),
        margins=margins, current_dual_predictions=inputs.reference,
        current_dual_mode=np.array('dual_refit_late'), **inputs.data,
        score_units=np.array(['logits', 'logits', 'logits', 'softmax_probability']))
    write_json(predicted, dict(dataset=inputs.dataset, task=inputs.task, modes=list(MODES), operator_version=1,
        source_fit_seconds=source_seconds, total_seconds=time.perf_counter()-started,
        source_alpha=1., support_lambda=8., label_budget=64, support_draws=5,
        source_class_counts=counts.tolist(), source_heads_separately_fitted=True,
        prediction_saved_before_query_evaluation=True, target_selected_hyperparameters=False,
        device='cpu', threads=2, neural_training=False, fresh_inner_validation=False,
        historical_r18_context='Tail was trained in the dual-task context, not independently optimized as R18-only',
        precision='float32 unit normalization; float64 streamed sufficient statistics, prior and support kernels',
        current_dual_new_fit=False, python=sys.version, torch=str(torch.__version__)))
    print(f'PREDICTED {inputs.suffix} source_fit={source_seconds:.1f}s total={time.perf_counter()-started:.1f}s', flush=True)


def truth(info, data, source_frame):
    if info['dataset'] == 'self24ports':
        frame, idcol = source_frame, 'instance_id'
    else:
        folder = {'oss2': 'opensarship2_transfer_20261007', 'f0921': 'f0921_transfer_20261007'}[info['dataset']]
        frame = pd.read_csv(ROOT / folder / 'target_manifest.csv', usecols=['sample_id', 'class_id'], keep_default_na=False)
        idcol = 'sample_id'
    frame = frame.iloc[data['rows']]
    np.testing.assert_array_equal(frame[idcol].to_numpy(str), data['ids'])
    y = frame.class_id.to_numpy(int)
    np.testing.assert_array_equal(y[data['support']], data['support_labels'])
    return y


def metrics(y, pred):
    return dict(oa=100*float((y == pred).mean()),
                ba=100*float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)])))


def pair_flow(y, candidate, reference):
    new, old = candidate == y, reference == y
    rescue, harm = new & ~old, old & ~new
    rows = []
    for c in np.unique(y):
        ix = y == c
        r, h = int(rescue[ix].sum()), int(harm[ix].sum())
        rows.append(dict(class_id=int(c), class_name=CLASSES[c], n=int(ix.sum()), rescue=r, harm=h,
            net_count=r-h, recall_delta_pp=100*(r-h)/int(ix.sum()),
            ba_contribution_pp=100*(r-h)/(int(ix.sum())*len(np.unique(y)))))
    diff = metrics(y, candidate)
    base = metrics(y, reference)
    effect = dict(n=len(y), rescue=int(rescue.sum()), harm=int(harm.sum()),
        both_correct=int((new & old).sum()), both_wrong=int((~new & ~old).sum()),
        net_oa_pp=diff['oa']-base['oa'], net_ba_pp=diff['ba']-base['ba'])
    np.testing.assert_allclose(effect['net_oa_pp'], 100*(effect['rescue']-effect['harm'])/len(y), atol=1e-10)
    np.testing.assert_allclose(effect['net_ba_pp'], sum(r['ba_contribution_pp'] for r in rows), atol=1e-10)
    return effect, rows


def report(out, all_tasks, source_frame):
    available = [(i, f) for i, f in all_tasks if (out / 'targets' / i['dataset'] / i['task'] / 'PREDICTED.json').exists()]
    metrics_rows, class_rows, effects, class_effects = [], [], [], []
    for info, _ in available:
        folder = out / 'targets' / info['dataset'] / info['task']
        with np.load(folder / 'predictions.npz') as z:
            data = {key: z[key] for key in z.files}
        y = truth(info, data, source_frame)
        for draw, mask in enumerate(data['eval_masks']):
            common = dict(dataset=info['dataset'], task=info['task'], draw=draw)
            yy = y[mask]
            pred = {mode: data['predictions'][draw, mi, mask] for mi, mode in enumerate(MODES)}
            pred[REFERENCE] = data['current_dual_predictions'][draw, mask]
            for mode, value in pred.items():
                metrics_rows.append(dict(**common, mode=mode, n=len(yy), **metrics(yy, value)))
                for c in np.unique(yy):
                    ix = yy == c
                    class_rows.append(dict(**common, mode=mode, class_id=int(c), class_name=CLASSES[c],
                        n=int(ix.sum()), support_count=int((data['support_labels'][draw] == c).sum()),
                        recall=100*float((value[ix] == c).mean())))
            pairs = [(mode, single) for mode in MODES[2:] for single in MODES[:2]]
            pairs += [(mode, REFERENCE) for mode in MODES] + [(MODES[1], MODES[0])]
            for candidate, reference in pairs:
                flow, by_class = pair_flow(yy, pred[candidate], pred[reference])
                context = dict(**common, candidate=candidate, reference=reference)
                effects.append(dict(**context, **flow))
                class_effects.extend(dict(**context, **row) for row in by_class)
    if not available:
        raise ValueError('No persisted task predictions to report')
    met, ef, cl, ce = map(pd.DataFrame, (metrics_rows, effects, class_rows, class_effects))
    for name, frame in [('metrics_by_task_draw', met), ('class_recall', cl), ('rescue_harm', ef), ('class_rescue_harm', ce)]:
        frame.to_csv(out / f'{name}.csv', index=False)
    task_means = met.groupby(['dataset', 'task', 'mode'], as_index=False)[['oa', 'ba']].mean()
    task_means.to_csv(out / 'task_means.csv', index=False)
    summary = task_means.groupby(['dataset', 'mode'], as_index=False)[['oa', 'ba']].mean()
    summary.to_csv(out / 'summary.csv', index=False)
    effect_task = ef.groupby(['dataset', 'task', 'candidate', 'reference'], as_index=False)[['net_oa_pp', 'net_ba_pp']].mean()
    effect_task.to_csv(out / 'task_effects.csv', index=False)
    effect_summary = effect_task.groupby(['dataset', 'candidate', 'reference'], as_index=False)[['net_oa_pp', 'net_ba_pp']].mean()
    counts = ef.groupby(['dataset', 'candidate', 'reference'], as_index=False)[['n', 'rescue', 'harm']].sum()
    effect_summary = effect_summary.merge(counts, on=['dataset', 'candidate', 'reference'])
    effect_summary.to_csv(out / 'effect_summary.csv', index=False)
    status = dict(expected_tasks=25, predicted_tasks=len(available), all25_complete=len(available) == 25,
        tasks_by_dataset=met.groupby('dataset').task.nunique().to_dict(), modes=list(MODES),
        metrics_rows=len(met), device='cpu', threads=2, source_alpha=1., residual_lambda=8.,
        source_validation_or_tuning=False, extra_neural_training=False, independent_neural_seeds=False,
        support_draws_are_not_neural_seeds=True, fixed_current_dual_reference=True,
        count_unit='within-draw query exposures; overlapping across the five draws',
        macro_average='draw mean within task, then equal task mean within dataset')
    write_json(out / 'RUN_STATUS.json', status)
    if len(available) == 25:
        write_json(out / 'COMPLETE.json', status)
    write_report(out, summary, effect_summary, effect_task, status)
    print(summary.to_string(index=False), flush=True)


def write_report(out, summary, effects, task_effects, status):
    lines = ['# Trained R18 Direct Readout Control', '',
        f"Completed prediction tasks: {status['predicted_tasks']}/25; coverage: {status['tasks_by_dataset']}.", '',
        'Fixed CPU protocol: unit-feature sqrt32 + constant for each single; concat of two unit blocks times sqrt16 + one constant;',
        'three separately fitted balanced source ridge heads (alpha1, unpenalized bias), balanced64-support residual lambda8.',
        'Fourth arm is exactly the .5/.5 mixture of the two adapted single softmax outputs, without fitting a fusion head.',
        'No target choice of arm, weights, epoch or hyperparameters. Source fitting uses only legal source rows.', '',
        '| Dataset | Fixed configuration | OA % | BA % |', '| --- | --- | ---: | ---: |']
    for row in summary.itertuples():
        lines.append(f'| {row.dataset} | {row.mode} | {row.oa:.5f} | {row.ba:.5f} |')
    lines += ['', '## Paired Error Flow', '',
        'Positive deltas favor candidate. Counts are overlapping query/draw exposures, not independent samples.', '',
        '| Dataset | Candidate | Reference | OA pp | BA pp | Rescue | Harm |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for row in effects.itertuples():
        lines.append(f'| {row.dataset} | {row.candidate} | {row.reference} | {row.net_oa_pp:+.5f} | '
                     f'{row.net_ba_pp:+.5f} | {row.rescue} | {row.harm} |')
    lines += ['', '## Port Direction', '', '| Candidate | Reference | LOPO BA positive / zero / negative |', '| --- | --- | --- |']
    for (candidate, reference), frame in task_effects[task_effects.dataset == 'self24ports'].groupby(['candidate', 'reference']):
        v = frame.net_ba_pp.to_numpy()
        lines.append(f'| {candidate} | {reference} | {int((v>1e-10).sum())} / {int((abs(v)<=1e-10).sum())} / {int((v < -1e-10).sum())} |')
    lines += ['', '## Interpretation Boundaries', '',
        '- This fills the direct trained-R18-alone readout gap. R18 was trained in the original dual-task context, not optimized independently as the best single backbone.',
        '- The R18 outer-task representation has seen inner-source labels. No new independent inner validation or hyperparameter selection is claimed.',
        '- R50 and R18 differ in pretraining, input and task training; these effects are not isolated. F has historical unlabeled SSL exposure.',
        '- The main agent audited code and weights in this round: current R50 and the historical extract_full_six_spatial_features.py use meanRGB repeat2*1.5 conv1 weights, identical for VV and VH. The task-R50 prefix is frozen.',
        '  Audit artifact: E:/Hermes/paper_tracking/sensitivity/strong_single_polarization_20261010/OBSERVABILITY.json. The float64 audit records old antisymmetric response max 1.0246394606700237e-14, new antisymmetric response RMS 1.7404581770394036, common-mode response difference max 6.375879291198316e-7, and unchanged total parameters 23521288. These supersede the earlier float32 exploratory response of 3.6e-7.',
        '  At the encoder input this collapses the two channels to their sum, whereas R18 receives VH. This establishes an observability limitation, not its performance contribution or a requirement for two backbones; input-information complementarity remains a candidate explanation.',
        '  The main-line difference-preserving single-input control uses wVV=wsum+.5*(wR-wG), wVH=wsum-.5*(wR-wG), with unchanged parameter/forward counts and source/target re-extraction plus matched readout refitting. Its performance result is pending; no such experiment was added here and original readout results are unchanged.',
        '- The two new combined arms isolate fixed readout choices on these cached feature functions. Current full dual is a reference with additional modules and a different readout, not an architecture-only intervention.',
        '- This R50 arm reuses historical pooled_c4.npy. The task-R50 frozen control re-extracts pixels through the current bf16 network.',
        '  A prior source-initialization probe measured about .004 relative geometry difference; this is not a full-cohort estimate or a proven rounding-only cause.',
        '  These cached readout arms are internally matched, but are not the identical task-script frozen control. Predictions are never forced to match its reported numbers.',
        '- The main line fixed the measured-single common-hard set to include R18-only before reading the final metrics. Only r18_task_trained is eligible; concat_shared_head and equal_single_softmax both use two backbones and must not enter that single set.',
        '- Five overlapping support draws are not five neural seeds; current R18 weights and full-dual reference are each fixed per task.',
        '- Both benefit and harm are retained. No failed arm is removed or target-best configuration promoted.', '',
        '- Increased concatenated dimension is not itself evidence of complementarity; the B=A duplicate-branch identity is checked on synthetic data.', '',
        '## Numerical And Artifact Contract', '',
        '- Float32 L2 normalization matches the original geometry normalization. Subsequent scale arithmetic, source statistics and support kernels are float64.',
        '- In unit coordinates, source alpha1 on scaled features equals penalties 1/32 for singles or 1/16 for concatenation, with zero penalty on source bias.',
        '- Support kernels are 32 Ur Us^T+1, 32 Vr Vs^T+1, and 16(Ur Us^T+Vr Vs^T)+1. The support intercept is regularized as in the original residual operator.',
        '- Source sufficient statistics and target scores are streamed in blocks of512. No full F feature matrix is copied to float64.',
        '- targets/{dataset}/{task}/predictions.npz stores modes, predictions, margins, source_rows, rows, ids, support, support_labels, eval_masks and fixed current_dual_predictions.',
        '- scores.npy is mmap-able float64 [5,4,N,8]: first three modes logits, fourth probabilities. priors.npy is [3,N,8].',
        '- source_heads.npz stores coefficients in unit coordinates, including the final bias row; support_coefficients.npz stores all fixed residual coefficients.',
        '- class_recall.csv and class_rescue_harm.csv retain task/draw/class outcomes and support coverage. Counts and macro BA contributions are not interchangeable.',
        '- SELF_TEST.json, SHAPE_CHECKS.json, PROTOCOL.json and per-task INPUTS/PREDICTED receipts document verification and provenance.', '']
    (out / 'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')


def self_test(out):
    """Independent explicit scaled-feature oracle tests all three fixed operators."""
    from aligned_residual_ops import residual_scores
    from strong_single_r50_task import fit_source_prior, geometry
    rng = np.random.default_rng(217)
    raw_r, raw_s = rng.normal(size=(41, 7)).astype(np.float32), rng.normal(size=(41, 5)).astype(np.float32)
    r, s = unit(raw_r), unit(raw_s)
    labels = np.r_[np.arange(8), rng.integers(0, 8, 33)]
    weights = len(labels)/(8*np.bincount(labels, minlength=8))[labels]
    x = np.c_[r, s, np.ones(len(r))]
    gram, rhs = x.T@(x*weights[:, None]), (x*weights[:, None]).T@(2*np.eye(8)[labels]-1)
    heads = fit_from_moments(gram, rhs, r.shape[1])
    explicit = [np.c_[r*np.sqrt(32), np.ones(len(r))], np.c_[s*np.sqrt(32), np.ones(len(r))],
                np.c_[r*4, s*4, np.ones(len(r))]]
    computed_prior = priors(r, s, heads)
    source_errors, residual_errors = [], []
    support = np.arange(16)
    support_labels = labels[support].copy()
    support_labels[support_labels == 7] = 0  # A missing support class must remain a valid eight-class head.
    kr = kernels(r, s, r[support], s[support])
    for phi, prior, kernel in zip(explicit, computed_prior, kr):
        penalty = np.diag(np.r_[np.ones(phi.shape[1]-1), 0.])
        coef = np.linalg.solve(phi.T@(weights[:, None]*phi)+penalty,
                               (weights[:, None]*phi).T@(2*np.eye(8)[labels]-1))
        expected_prior = phi@coef
        source_errors.append(float(np.max(abs(prior-expected_prior))))
        np.testing.assert_allclose(prior, expected_prior, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(kernel, phi@phi[support].T, atol=1e-12)
        a = coefficients(kernel[support], prior[support], support_labels, 8.)
        actual = prior+kernel@a
        expected = residual_scores(torch.from_numpy(phi[support]), torch.from_numpy(prior[support]),
            torch.from_numpy(support_labels), torch.from_numpy(phi), torch.from_numpy(prior), 8.).numpy()
        residual_errors.append(float(np.max(abs(actual-expected))))
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    original_coef = fit_source_prior(raw_r, labels, np.arange(len(labels)))
    original = geometry(torch.from_numpy(raw_r)).numpy().astype(float) @ original_coef.T
    normalization_rounding_difference = float(np.max(abs(computed_prior[0]-original)))
    np.testing.assert_allclose(computed_prior[0], original, atol=2e-6, rtol=2e-6)
    mixture = .5*softmax(computed_prior[0], axis=1) + .5*softmax(computed_prior[1], axis=1)
    np.testing.assert_allclose(mixture.sum(1), 1, atol=1e-14)
    # Duplicating A into B must preserve alpha1 source ridge and lambda8 residuals.
    duplicate_x = np.c_[r, r, np.ones(len(r))]
    duplicate_heads = fit_from_moments(duplicate_x.T@(duplicate_x*weights[:, None]),
        (duplicate_x*weights[:, None]).T@(2*np.eye(8)[labels]-1), r.shape[1])
    duplicate_priors = priors(r, r, duplicate_heads)
    duplicate_kernels = kernels(r, r, r[support], r[support])
    np.testing.assert_allclose(duplicate_kernels[2], kr[0], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(duplicate_priors[2], computed_prior[0], atol=1e-10, rtol=1e-10)
    duplicate_scores = []
    for k, p in ((kr[0], computed_prior[0]), (duplicate_kernels[2], duplicate_priors[2])):
        a = coefficients(k[support], p[support], support_labels, 8.)
        duplicate_scores.append(p+k@a)
    np.testing.assert_allclose(duplicate_scores[0], duplicate_scores[1], atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(duplicate_scores[0].argmax(1), duplicate_scores[1].argmax(1))
    # Asymmetric errors verify rescue direction and missing-query-class BA denominator.
    y = np.array([0, 0, 0, 2])
    f, _ = pair_flow(y, np.array([0, 1, 0, 2]), np.array([1, 0, 1, 2]))
    assert (f['rescue'], f['harm']) == (2, 1)
    np.testing.assert_allclose(f['net_ba_pp'], 100/6)
    receipt = dict(passed=True, seed=217, source_head_max_errors=source_errors,
        structured_vs_original_residual_max_errors=residual_errors,
        original_float32_scale_vs_float64_scale_prior_max_error=normalization_rounding_difference,
        duplicate_B_equals_A=dict(kernel_max_error=float(np.max(abs(duplicate_kernels[2]-kr[0]))),
            source_prior_max_error=float(np.max(abs(duplicate_priors[2]-computed_prior[0]))),
            residual_score_max_error=float(np.max(abs(duplicate_scores[1]-duplicate_scores[0]))),
            predictions_exact=True, source_alpha=1., residual_lambda=8., constant_coordinates=1),
        missing_support_class_tested=True, query_class_denominator_tested=True, device='cpu')
    write_json(out / 'SELF_TEST.json', receipt)
    print(json.dumps(receipt), flush=True)


def protocol(all_tasks):
    return dict(operator_version=1, tasks=[f"{i['dataset']}/{i['task']}" for i, _ in all_tasks], modes=list(MODES),
        singles='unit feature * sqrt32 + one constant', concat='concat(R50unit,R18unit)*sqrt16 + one constant',
        source='Separate balanced ridge alpha1, bias unpenalized, all and only legal source rows',
        support='Original five x64 draws, balanced residual lambda8', softmax_fusion='Equal .5/.5 of adapted single logits softmax, no fit',
        hyperparameter_selection=False, target_selected_main_arm=False, new_inner_validation=False,
        r18_training_context='Original dual-task trained layer4; inner-source labels were used by this representation',
        independent_best_single_claim=False, pretraining_equalized=False, historical_f_ssl_exposure=True,
        neural_training=False, feature_extraction=False, device='cpu', cpu_threads=2, chunk_rows=BATCH,
        numerical_equivalence='Unit-coordinate source penalty alpha/scale^2; structured support kernel exactly equals explicit scaled concatenation',
        precision='float32 unit normalization, float64 scale/linear algebra; not bitwise original float32 readout',
        count_unit='within-draw query exposures, not independent neural replications')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'run', 'report', 'self-test'), default='run')
    parser.add_argument('--tasks', nargs='+', help='Optional exact task names or dataset/task, e.g. Mombasa f0921')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    all_tasks = tasks()
    fixed = protocol(all_tasks)
    path = args.out / 'PROTOCOL.json'
    if path.exists():
        if read_json(path) != fixed:
            raise ValueError('Existing fixed protocol differs; do not overwrite it')
    else:
        write_json(path, fixed)
    current = None
    try:
        with threadpool_limits(limits=2):
            if args.stage == 'self-test':
                self_test(args.out)
                return
            source_frame = pd.read_csv(SOURCE / 'manifest.csv', usecols=['instance_id', 'class_id', 'port'], keep_default_na=False)
            selected = [(i, f) for i, f in all_tasks if not args.tasks or i['task'] in args.tasks
                        or f"{i['dataset']}/{i['task']}" in args.tasks]
            if not selected:
                raise ValueError('No requested tasks found')
            if args.stage == 'report':
                report(args.out, all_tasks, source_frame)
                return
            checks = []
            for info, folder in selected:
                current = f"{info['dataset']}/{info['task']}"
                inputs = Inputs(info, folder, source_frame)
                if args.stage == 'check':
                    checks.append(bounded_check(inputs))
                    print(f'CHECK {current} source={len(inputs.source_rows)} target={len(inputs.rows)} OK', flush=True)
                else:
                    predict(inputs, args.out / 'targets' / current)
                del inputs
                gc.collect()
            if args.stage == 'check':
                write_json(args.out / 'SHAPE_CHECKS.json', dict(tasks=checks, gpu_used=False, features_streamed=True))
            else:
                report(args.out, all_tasks, source_frame)
    except Exception:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        write_json(args.out / f'FAILURE_{stamp}.json', dict(task=current, stage=args.stage,
            traceback=traceback.format_exc(), hyperparameters_changed=False, partial_outputs_preserved=True))
        raise


if __name__ == '__main__':
    main()
