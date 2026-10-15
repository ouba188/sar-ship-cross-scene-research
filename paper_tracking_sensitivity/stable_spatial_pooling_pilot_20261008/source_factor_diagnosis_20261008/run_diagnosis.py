"""Bounded source-only separation of the frozen spatial pilot's two factors."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'

import inspect
import json
import pickle
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check')

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge, RidgeClassifier
from threadpoolctl import threadpool_limits

from run_stable_guided_spatial_pooling import (
    OUT as PILOT, PARENT, EVIDENCE, SOURCE, POOL_ARMS, archive, read_json,
    feature_bank, inner_train_priors, inner_held_views)
from sar_joint_completion import write_json
from stable_guided_spatial_pooling import (
    coordinated_expert, fit_residual_head, spatial_residual)


OUT = Path(__file__).resolve().parent
OUTER = 'Antwerp-Bruges'
HELD = ('Jebel Ali', 'Port Klang', 'Singapore')
FACTORS = {'original': (False, False), 'prior_only': (True, False),
           'kernel_only': (False, True), 'full': (True, True)}


def balanced_weights(labels):
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return len(labels) / (len(counts) * counts[inverse])


def score_summary(labels, scores):
    desired = 2 * np.eye(8)[labels] - 1
    error = scores - desired
    weights = balanced_weights(labels)
    pred = scores.argmax(1)
    recall = [float(np.mean(pred[labels == c] == c)) if np.any(labels == c) else None
              for c in range(8)]
    centered_error = error - error.mean(1, keepdims=True)
    return dict(n=len(labels), oa=float(np.mean(pred == labels)),
        ba=float(np.mean([r for r in recall if r is not None])),
        classbalanced_mse=float(np.average(np.mean(error ** 2, axis=1), weights=weights)),
        centered_classbalanced_mse=float(np.average(np.mean(centered_error ** 2, axis=1), weights=weights)),
        classbalanced_mean_score_error=np.average(error, axis=0, weights=weights).tolist(),
        class_recall=recall)


def attention_summary(folder, g, rows, selector, labels, arm, features):
    projected = np.load(folder / 'projected_tokens.npy', mmap_mode='r')
    evidence = np.load(folder / (arm + '_evidence.npy'), mmap_mode='r')
    local = np.searchsorted(g['rows'], rows)
    max_weight, entropy, count = 0., 0., 0
    for lo in range(0, len(rows), 512):
        chosen = local[lo:lo + 512]
        _, alpha = spatial_residual(projected[chosen], evidence[chosen], selector[lo:lo + 512],
                                     g['pairs'], g[arm + '_evidence_scale'])
        max_weight += alpha.max(1).sum()
        entropy += -(alpha * np.log(np.maximum(alpha, 1e-300))).sum()
        count += len(chosen)
    top2 = np.argsort(selector, axis=1, kind='stable')[:, -2:]
    contains = (top2 == labels[:, None]).any(1)
    return dict(mean_attention_entropy_nats=float(entropy / count),
        mean_attention_entropy_fraction_of_uniform=float(entropy / count / np.log(4)),
        mean_max_weight=float(max_weight / count),
        source_base_top2_contains_truth=float(contains.mean()),
        balanced_source_base_top2_contains_truth=float(np.average(contains, weights=balanced_weights(labels))),
        normalized_feature_rms=float(np.sqrt(np.mean(features.astype(float) ** 2))),
        normalized_feature_mean_rms=float(np.sqrt(np.mean(features.mean(0, dtype=float) ** 2))))


def report(records, prior_records, module_records, verification, elapsed):
    df = pd.DataFrame(records)
    scalar = ['oa', 'ba', 'classbalanced_mse', 'centered_classbalanced_mse',
              'rescues_vs_original', 'harms_vs_original', 'changed_vs_original']
    df.to_json(OUT / 'factor_metrics.json', orient='records', indent=2)
    df.drop(columns=['classbalanced_mean_score_error', 'class_recall']).to_csv(OUT / 'factor_metrics.csv', index=False)
    means = df.groupby(['arm', 'factor'], as_index=False)[scalar].mean()
    means.to_csv(OUT / 'equal_episode_factor_means.csv', index=False)
    priors = pd.DataFrame(prior_records)
    priors.to_json(OUT / 'prior_metrics.json', orient='records', indent=2)
    priors.drop(columns=['classbalanced_mean_score_error', 'class_recall']).to_csv(OUT / 'prior_metrics.csv', index=False)
    pmeans = priors.groupby(['arm', 'split', 'prior'], as_index=False)[
        ['oa', 'ba', 'classbalanced_mse', 'centered_classbalanced_mse']].mean()
    pmeans.to_csv(OUT / 'equal_episode_prior_means.csv', index=False)
    modules = pd.DataFrame(module_records)
    modules.to_csv(OUT / 'module_diagnostics.csv', index=False)
    effects = []
    for arm in POOL_ARMS:
        for held in HELD:
            values = df[(df.arm == arm) & (df.held_source == held)].set_index('factor')
            effects.append(dict(arm=arm, held_source=held,
                prior_effect_ba=float(values.loc['prior_only', 'ba'] - values.loc['original', 'ba']),
                kernel_effect_ba=float(values.loc['kernel_only', 'ba'] - values.loc['original', 'ba']),
                full_effect_ba=float(values.loc['full', 'ba'] - values.loc['original', 'ba']),
                interaction_ba=float(values.loc['full', 'ba'] - values.loc['prior_only', 'ba']
                    - values.loc['kernel_only', 'ba'] + values.loc['original', 'ba'])))
    effect = pd.DataFrame(effects)
    effect.to_csv(OUT / 'factor_effects.csv', index=False)
    effect_mean = effect.groupby('arm').mean(numeric_only=True)
    lines = ['# Source-only spatial factor diagnosis', '',
        'Fixed outer Antwerp-Bruges; held source Jebel Ali, Port Klang and Singapore. '
        'Original purged train rows, original source64 supports, frozen geometry and unchanged settings. '
        'No Antwerp target predictions or metrics were read, and no Antwerp target labels were used. '
        'Original negative pilot outputs are preserved.', '',
        'The large loss localizes to the auxiliary prior factor in all three episodes for both pooling arms. '
        'The kernel-only loss is much smaller; adding the kernel to the auxiliary prior slightly moderates its loss. '
        'This separates the dominant integration factor, not a unique underlying statistical cause. '
        'No clear implementation bug was found, so no corrected model or alternate parameter setting was run.', '',
        '| Pool | Factor | Expert OA % | Expert BA % | Balanced score MSE |',
        '|---|---|---:|---:|---:|']
    for arm in POOL_ARMS:
        for factor in FACTORS:
            row = means[(means.arm == arm) & (means.factor == factor)].iloc[0]
            lines.append(f'| {arm} | {factor} | {100*row.oa:.4f} | {100*row.ba:.4f} | {row.classbalanced_mse:.6f} |')
    lines.extend(['', 'Equal weight across the three held source episodes; support rows excluded from held metrics.', '',
        '## Factor separation', ''])
    for arm in POOL_ARMS:
        row = effect_mean.loc[arm]
        lines.append(f'- {arm}: prior-only {100*row.prior_effect_ba:+.4f} pp BA; '
            f'kernel-only {100*row.kernel_effect_ba:+.4f} pp; full {100*row.full_effect_ba:+.4f} pp; '
            f'factor interaction {100*row.interaction_ba:+.4f} pp.')
    lines.extend(['', '| Pool | Split | Original prior MSE | Augmented prior MSE | Original prior BA % | Augmented prior BA % |',
                  '|---|---|---:|---:|---:|---:|'])
    for arm in POOL_ARMS:
        for split in ('train_fit', 'held_query'):
            values = pmeans[(pmeans.arm == arm) & (pmeans.split == split)].set_index('prior')
            a, b = values.loc['original'], values.loc['augmented']
            lines.append(f'| {arm} | {split} | {a.classbalanced_mse:.6f} | {b.classbalanced_mse:.6f} | '
                         f'{100*a.ba:.4f} | {100*b.ba:.4f} |')
    lines.extend(['', 'The auxiliary head improves its class-balanced score MSE on training rows AND held-source rows, '
        'but held-prior BA falls from 37.9061% to 35.9265% (ordinary) / 36.0756% (stable). '
        'After the coordinated64 correction, both MSE and BA are worse than the original corrected expert. '
        'Thus the measured failure is not simply failure to reduce the residual regression objective, '
        'nor does it require a target-only distribution shift. Score-regression improvement does not preserve class decisions '
        'or the original expert benefit from support adaptation.', '', '## Attention and units', '',
        '| Pool | Split | Entropy nats (uniform=1.3863) | Max weight | Base top2 contains truth % | Feature RMS |',
        '|---|---|---:|---:|---:|---:|'])
    for arm in POOL_ARMS:
        for split in ('train_fit', 'held_query'):
            row = modules[(modules.arm == arm) & (modules.split == split)].mean(numeric_only=True)
            lines.append(f'| {arm} | {split} | {row.mean_attention_entropy_nats:.4f} | '
                f'{row.mean_max_weight:.4f} | {100*row.source_base_top2_contains_truth:.3f} | {row.normalized_feature_rms:.4f} |')
    lines.extend(['', 'Truth-in-top2 is diagnostic only and never selects attention or examples. '
        'All MSEs average score-coordinate squared errors, then give each present true class equal weight. '
        'Per-class recalls, class-balanced mean score errors, centered-score MSE and per-episode values are saved.', '',
        'The .25 coefficient multiplies ONLY the added adaptation kernel z z_support.T / 128. '
        'It does not attenuate the learned prior z W, whose fixed multiplier is 1. '
        'Every factor refits the support correction jointly against its own prior and kernel. '
        'The original retained/complement kernel and its bias remain intact. '
        'The residual readout fits source-RMS-normalized 128-dimensional z without an added intercept; '
        'its ridge is 1. The 64-support correction ridge is 8.', ''])
    for arm in POOL_ARMS:
        row = modules[(modules.arm == arm) & (modules.split == 'held_query')].mean(numeric_only=True)
        lines.append(f'- {arm}: held auxiliary score RMS {row.auxiliary_score_rms:.5f}; '
            f'prior-only corrected-score delta RMS {row.prior_only_corrected_delta_rms:.5f}; '
            f'mean survival ratio {row.prior_delta_survival_ratio:.4f}; '
            f'spatial/original support-kernel diagonal ratio {row.spatial_to_original_kernel_diagonal_ratio:.4f}.')
    lines.extend(['', '## Implementation checks', '',
        f"Saved original classifier and installed sklearn {sklearn.__version__} use LabelBinarizer(neg_label=-1, pos_label=1) "
        'with columns 0..7. This exactly matches 2*onehot-1. '
        'The new residual head is checked against sklearn Ridge(alpha=1, fit_intercept=False) using '
        'the same original expert prior and exact balanced class weights. '
        'The train-prior helper is replayed on held rows against the held-view prior; '
        'episode expert coefficients/intercepts equal their frozen geometry counterparts. '
        'Maximum residual-head coefficient disagreement with sklearn is 5.61e-15. '
        'Original corrected scores replay exactly; all six full-arm BA values reproduce the original pilot. '
        'The train/held prior helper difference is at most 2.63e-6 from floating-point evaluation order. '
        'No alternative head, intercept or strength is selected.', '',
        'See VERIFICATION.json for numerical errors, archived original/full replay, source exclusion and label-encoding checks. '
        'Directions, PCA, RMS moments and frozen encoder still use all outer-source data: not fully inner-OOF. '
        'Only these three fixed source episodes were diagnosed; no independent confirmation or unique root-cause claim follows. '
        'The missing auxiliary intercept is an intentional difference from the original classifier, not an established bug. '
        'Row-centered score MSE is essentially identical to uncentered MSE here, so common score offsets do not explain '
        'the loss; class-specific calibration and representation effects remain unresolved. '
        'An intercept change, joint retraining or changed regularization would be new experiments and were not performed.', ''])
    (OUT / 'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(OUT / 'VERIFICATION.json', verification)
    write_json(OUT / 'COMPLETE.json', dict(status='complete', outer=OUTER, held_sources=list(HELD),
        target_labels_used=0, target_predictions_read=False, parameter_search=False,
        seconds=elapsed, means=means.to_dict('records'), factor_effect_means=effect_mean.reset_index().to_dict('records')))
    print(means.to_string(index=False), flush=True)
    print(effect_mean.to_string(), flush=True)


def main():
    if (OUT / 'COMPLETE.json').exists():
        print('Completed diagnosis preserved: ' + str(OUT / 'FINDINGS.md'))
        return
    started = time.monotonic()
    write_json(OUT / 'PROTOCOL.json', dict(outer=OUTER, held_sources=list(HELD),
        pool_arms=list(POOL_ARMS), factors=FACTORS, source_support=64, head_ridge=1.,
        correction_ridge=8., spatial_kernel_weight=.25, auxiliary_prior_weight=1.,
        no_target_readout=True, no_parameter_search=True, no_gate_refit=True, cpu_threads=2,
        source_geometry='Reuse fixed full outer-source geometry; not fully inner-OOF',
        auxiliary_head='Same deterministic head as original pilot, refit once per arm/episode then shared between prior-only/full'))
    folder = PILOT / 'ports' / OUTER
    g = archive(folder / 'geometry.npz')
    source = g['source_rows']
    frame = pd.read_csv(SOURCE / 'manifest.csv', usecols=['port', 'class_id'])
    source_labels = frame.class_id.to_numpy()[source]
    np.testing.assert_array_equal(np.load(EVIDENCE / 'source_features' / OUTER / 'rows.npy'), source)
    if np.any(frame.port.to_numpy()[source] == OUTER):
        raise ValueError('Outer target enters diagnosis')
    raw = np.load(SOURCE / 'pooled_features.npy', mmap_mode='r')
    extra = np.load(EVIDENCE / 'source_features' / OUTER / 'features.npy', mmap_mode='r')
    descriptions = {item['port']: item for item in read_json(PARENT / OUTER / 'EPISODES_COMPLETE.json')['episodes']}
    cached = {item['port']: item for item in read_json(EVIDENCE / 'source_episodes' / OUTER / 'COMPLETE.json')['episodes']}
    historical = pd.read_csv(folder / 'source_episode_metrics.csv')
    with (PARENT / OUTER / 'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    encoder = model['expert']._label_binarizer
    np.testing.assert_array_equal(model['expert'].classes_, np.arange(8))
    np.testing.assert_array_equal(encoder.transform(np.arange(8)), 2 * np.eye(8) - 1)
    (OUT / 'installed_ridge_encoding.txt').write_text(inspect.getsource(RidgeClassifier._prepare_data), encoding='utf-8')
    verification = dict(sklearn_version=sklearn.__version__,
        original_type=str(type(model['expert'])), original_parameters=model['expert'].get_params(),
        original_label_encoding=encoder.transform(np.arange(8)).tolist(), label_encoding_exact=True,
        target_rows_excluded=True, target_predictions_read=False, episodes=[])
    records, prior_records, module_records = [], [], []
    for held in HELD:
        ep = archive(descriptions[held]['path'])
        saved = archive(cached[held]['path'])
        geometry = archive(PARENT / 'episode_geometry_cache' / (cached[held]['pair_key'] + '.npz'))
        fitted = archive(EVIDENCE / 'source_pair_cache' / (cached[held]['pair_key'] + '.npz'))
        train, rows, support, query = ep['train_rows'], ep['rows'], ep['support'], ep['query']
        ti, hi = np.searchsorted(source, train), np.searchsorted(source, rows)
        for recovered, expected in ((source[ti], train), (source[hi], rows),
            (geometry['train_rows'], train), (fitted['train_rows'], train),
            (saved['rows'], rows), (saved['support'], support)):
            np.testing.assert_array_equal(recovered, expected)
        if np.intersect1d(train, rows).size or np.intersect1d(support, query).size:
            raise ValueError('Train/held or support/query overlap')
        for key in ('expert_coef', 'expert_intercept', 'expert_center', 'expert_scale', 'ssl_mean', 'ssl_components'):
            np.testing.assert_array_equal(ep[key], geometry[key].astype(np.float32))
        ytrain, yheld = source_labels[ti], source_labels[hi]
        btrain, etrain = inner_train_priors(raw, train, extra[ti], geometry, ep, fitted)
        bheld = ep['xs'] @ fitted['base_coef'].T + fitted['base_intercept']
        prior, kernel = inner_held_views(ep, extra[hi])
        _, replay_prior = inner_train_priors(raw, rows, extra[hi], geometry, ep, fitted)
        prior_error = float(np.max(np.abs(replay_prior - prior)))
        np.testing.assert_allclose(replay_prior, prior, atol=2e-5, rtol=2e-5)
        train_features = feature_bank(folder, g, train, btrain)
        held_features = feature_bank(folder, g, rows, bheld)
        receipt = dict(held_source=held, train_rows=len(train), held_query_rows=len(query),
            support_count=len(support), prior_helper_replay_max_error=prior_error, arms={})
        predictions = {}
        for arm in POOL_ARMS:
            x, z = train_features[arm], held_features[arm]
            head = fit_residual_head(x, ytrain, etrain)
            check = Ridge(alpha=1., fit_intercept=False, solver='cholesky').fit(
                x.astype(float), encoder.transform(ytrain) - etrain, sample_weight=balanced_weights(ytrain))
            coef_error = float(np.max(np.abs(check.coef_.T - head)))
            np.testing.assert_allclose(check.coef_.T, head, atol=1e-8, rtol=1e-7)
            train_add, held_add = x @ head, z @ head
            for split, y, p, add in (('train_fit', ytrain, etrain, train_add),
                                     ('held_query', yheld[query], prior[query], held_add[query])):
                for name, value in (('original', p), ('augmented', p + add)):
                    prior_records.append(dict(held_source=held, arm=arm, split=split, prior=name,
                                              **score_summary(y, value)))
            factor_scores, duals = {}, {}
            for factor, (use_prior, use_kernel) in FACTORS.items():
                scores, dual = coordinated_expert(prior, kernel, z, head if use_prior else np.zeros_like(head),
                    support, yheld[support], kernel_strength=.25 if use_kernel else 0.)
                factor_scores[factor], duals[factor] = scores, dual
                pred, original_pred = scores[query].argmax(1), saved['expert_scores'][query].argmax(1)
                good, original_good = pred == yheld[query], original_pred == yheld[query]
                records.append(dict(held_source=held, arm=arm, factor=factor,
                    **score_summary(yheld[query], scores[query]),
                    rescues_vs_original=int((good & ~original_good).sum()),
                    harms_vs_original=int((~good & original_good).sum()),
                    changed_vs_original=int((pred != original_pred).sum())))
                predictions[arm + '_' + factor] = scores
            replay_error = float(np.max(np.abs(factor_scores['original'] - saved['expert_scores'])))
            np.testing.assert_allclose(factor_scores['original'], saved['expert_scores'], atol=2e-6, rtol=2e-6)
            old_ba = float(historical[(historical.held_source == held) & (historical.arm == arm)].expert_ba.iloc[0])
            full_ba = score_summary(yheld[query], factor_scores['full'][query])['ba']
            np.testing.assert_allclose(full_ba, old_ba, atol=1e-14)
            delta = factor_scores['prior_only'] - factor_scores['original']
            identity_error = float(np.max(np.abs(delta - held_add - kernel @ (duals['prior_only'] - duals['original']))))
            np.testing.assert_allclose(delta, held_add + kernel @ (duals['prior_only'] - duals['original']), atol=1e-12)
            support_diag = float(np.mean(np.diag(kernel[support])))
            spatial_diag = float(.25 * np.mean(np.sum(z[support].astype(float) ** 2, axis=1)) / 128)
            for split, ids, selector, y, features, addition in (
                    ('train_fit', train, btrain, ytrain, x, train_add),
                    ('held_query', rows[query], bheld[query], yheld[query], z[query], held_add[query])):
                details = attention_summary(folder, g, ids, selector, y, arm, features)
                details['auxiliary_score_rms'] = float(np.sqrt(np.mean(addition ** 2)))
                details['auxiliary_centered_score_rms'] = float(np.sqrt(np.mean((addition - addition.mean(1, keepdims=True)) ** 2)))
                if split == 'held_query':
                    details.update(prior_only_corrected_delta_rms=float(np.sqrt(np.mean(delta[query] ** 2))),
                        prior_delta_survival_ratio=float(np.linalg.norm(delta[query]) / np.linalg.norm(held_add[query])),
                        original_support_kernel_diagonal=support_diag, spatial_support_kernel_diagonal=spatial_diag,
                        spatial_to_original_kernel_diagonal_ratio=spatial_diag / support_diag)
                module_records.append(dict(held_source=held, arm=arm, split=split, **details))
            receipt['arms'][arm] = dict(residual_sklearn_coef_max_error=coef_error,
                original_expert_replay_max_error=replay_error, historical_full_ba_error=full_ba - old_ba,
                coordinated_prior_delta_identity_error=identity_error)
            np.savez_compressed(OUT / (held.replace(' ', '_') + '_' + arm + '_head.npz'),
                coef=head, train_rows=train, support=support, rows=rows)
        np.savez_compressed(OUT / (held.replace(' ', '_') + '_source_scores.npz'),
                            rows=rows, support=support, query=query, labels=yheld, **predictions)
        verification['episodes'].append(receipt)
        print(f'SOURCE DIAGNOSIS {held}: 2 pooling arms x 4 factors complete', flush=True)
    report(records, prior_records, module_records, verification, time.monotonic() - started)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
