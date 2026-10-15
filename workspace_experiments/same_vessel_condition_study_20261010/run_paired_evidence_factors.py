"""Separate explicit candidate factors, test their roles, then frozen-kernel use."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import time
import itertools
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from train_explicit_candidate import (
    BASE, OUT as OLD, SEEDS, SUPPORTS, load_data, metrics, save_json, phi, residual_head,
)
from train_split_evidence_kernel import OUT as PARENT
from paired_evidence_factors import (
    acquisition_table, row_weights, separate, complement, paired_covariance,
    fit_probe, score_probe, support_operator, adapted_scores, fit_episode_correction,
)


OUT = BASE/'paired_evidence_factors_20261011'
METHODS = ('paired', 'class_only', 'stable_only', 'pca', 'head')
ORIGINS = ('split_task', 'split_relation')
GAIN = (0., .5, 1.)
CLASSES = ('bulk', 'fishing', 'general', 'chemical', 'container', 'crude', 'tug', 'offshore')


def validate_correction(prior, delta, kernel, labels, validation):
    records = []
    for gain in GAIN:
        for role, episodes in validation.items():
            for draw, (s, q) in enumerate(episodes):
                op, present = support_operator(kernel[s], kernel[q], labels[s])
                score = adapted_scores(prior[s]+gain*delta[s], prior[q]+gain*delta[q], op, present, labels[s])
                records.append(dict(gain=gain, role=role, draw=draw,
                                    **metrics(score.argmax(1), labels[q])))
    df = pd.DataFrame(records)
    means = df.groupby('gain', sort=True).BA.mean()
    chosen = float(means.index[np.argmax(means.to_numpy())])
    return chosen, records


def geometry(values, meta):
    x = np.asarray(values, np.float64)
    weights = row_weights(meta)
    center = weights@x
    variance = float((weights[:, None]*np.square(x-center)).sum())
    records = []
    for scope in ('cross_port', 'same_port'):
        drift, coverage = paired_covariance(x, meta, scope)
        records.append(dict(scope=scope, dimensions=x.shape[1], variance=variance,
                            drift=float(np.trace(drift)), normalized_drift=float(np.trace(drift)/variance),
                            vessels=len(coverage), pairs=int(coverage.pairs.sum()) if len(coverage) else 0))
    return records


def paired_correctness(scores, meta):
    p, mm = acquisition_table(scores, meta)
    records = []
    for ship, group in mm.groupby('mmsi', sort=True):
        label = int(group.class_id.iloc[0])
        rivals = np.arange(8) != label
        for scope in ('cross_port', 'same_port'):
            pairs = [(i, j) for i, j in itertools.combinations(group.index, 2)
                     if mm.loc[i, 'product'] != mm.loc[j, 'product'] and
                     (mm.loc[i, 'port'] != mm.loc[j, 'port']) == (scope == 'cross_port')]
            if not pairs:
                continue
            a, b = np.asarray(pairs).T
            ma, mb = [(p[ids, label, None]-p[ids][:, rivals]) for ids in (a, b)]
            correct_a, correct_b = p[a].argmax(1) == label, p[b].argmax(1) == label
            records.append(dict(mmsi=ship, class_id=label, scope=scope, pairs=len(pairs),
                relation_both_correct=float(((ma > 0)&(mb > 0)).mean()),
                relation_flip=float(((ma > 0)!=(mb > 0)).mean()),
                relation_both_wrong=float(((ma <= 0)&(mb <= 0)).mean()),
                top1_both_correct=float((correct_a&correct_b).mean()),
                top1_disagree=float((p[a].argmax(1) != p[b].argmax(1)).mean())))
    return records


def target_readout(prior, kernel, s, q, sy):
    p = torch.as_tensor(prior, dtype=torch.float32)
    k = torch.as_tensor(kernel, dtype=torch.float32)
    beta, _, _ = residual_head(phi(k[s]), p[s], torch.tensor(sy), balanced=True)
    return (p[q]+phi(k[q])@beta).numpy()


def main():
    torch.set_num_threads(1)
    started = time.perf_counter(); OUT.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    del raw, norm, fixed, h
    fit, held = split['factor_fit'], split['test_source']
    fm, hm = [meta.iloc[ids].reset_index(drop=True) for ids in (fit, held)]
    labels = meta.class_id.to_numpy(); weights = row_weights(fm)
    validation = {}
    with np.load(OLD/'validation_episodes.npz') as f:
        for role in ('val_new_ship', 'val_new_time'):
            validation[role] = [(f[f'{role}_{j}_support'], f[f'{role}_{j}_query']) for j in range(3)]
    tasks = {}
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        with np.load(path) as f:
            tasks[path.stem.removesuffix('_split')] = tuple(f[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels'))
    save_json(OUT/'PROTOCOL.json', dict(
        origins=ORIGINS, methods=METHODS, seeds=SEEDS, rank=7, gains=GAIN,
        scope='New source-only linear factor extraction inside existing source-trained32D; no new neural encoder fits; reused development cohorts, not independent confirmation.',
        fit='Original1296 chips/525ships; only source-fit labels/metadata estimate factors. Independent-identity heldsource823 diagnostic; historical sourcevalidation selects correction gain. Not strict heldout-port source training.',
        signal='Equal classes; withinclass equal ships, withinship equal product/port acquisitions; covariance of eight classcentroids. Rankatmost7; only mean-visible linear class information.',
        drift='Real same-vessel DIFFERENT-product DIFFERENT-port acquisition mean differences. Uncentered delta secondmoment, equal coveredclasses/ships/pairs. Missing crossport classes not claimed stable.',
        extraction='Common sourceweighted whitening; generalized eig signal vs drift/(trace(drift)/32)+.1I. Top7 span QR; complement25 in WHITENED metric. No per-axis physical meaning/independence claim. Export complete mapping and both factors.',
        controls='class_only top7signal, stable_only lowest7drift, PCA7 chosen PREwhitening, centered oldhead span7 transformed to common whitened space. Headkeep parity is algebraic, NOT evidence of causality.',
        readouts='All extracted7D normalized bysqrt7; complement25 bysqrt25; common whitenedfull32 bysqrt32. Fixed classbalanced source ridge1; all pairwise7D comparisons equalcapacity/scale. Frozenhead keep/remove separately test dependence.',
        correction='Same64parameter7+intercept x8 linear residual for eachmethod. Closedform ridge8 leastsquares over ALL400old64support96query sourceepisodes through fixedcasekernel, including supportprior subtraction and absentclassmask. Gain0/.5/1 selected only oldsourcevalidationadaptedBA. Equal BA ties choose lowergain.',
        interpretation='Paired observational candidate, NOT proof of purecausal factor recovery. Stable vesselidentity can remain. Need held cross-vessel utility, conditionmatched identity controls, fullclass BA/OA and matched class_only increment.',
        target='Old3portsx3draws64labels; predictions archived before querymetrics. Frozen oldcasekernel shared throughout. Primary correction choice uses source only; probes/ablations are diagnostics, not targetselected deployment.',
        no_search='No dimension/penalty sweep, no backbone retraining, no target choice of arms or coefficients. Negative results retained.'))
    coverage, geometry_rows, source_rows, pair_rows = [], [], [], []
    selections, val_rows, cards, accounting, overlap = [], [], [], [], []
    replays, decomposition_error = 0, 0.
    for origin in ORIGINS:
        for seed in SEEDS:
            tag = dict(origin=origin, seed=seed)
            with np.load(PARENT/f'{origin}_{seed}_vectors.npz') as f:
                z = f['z']; saved_mapping = {k: f[k] for k in ('basis', 'scale', 'input_center', 'raw_center', 'raw_scale')}
                np.testing.assert_array_equal(f['obs_ids'], np.arange(len(meta)))
            state = torch.load(PARENT/f'{origin.removeprefix("split_")}_{seed}.pt', map_location='cpu', weights_only=True)['selected']
            head_w, head_b = state['head.weight'].numpy(), state['head.bias'].numpy()
            prior = F.linear(torch.from_numpy(z), state['head.weight'], state['head.bias']).numpy()
            with np.load(OLD/f'task_{seed}_vectors.npz') as f:
                kernel = f['z']
            factor = separate(z[fit], fm, head_w)
            cov_rows = factor['coverage'].copy(); cov_rows['origin'] = origin; cov_rows['seed'] = seed
            coverage.append(cov_rows)
            u = (z-factor['center'])@factor['whitener']
            hweight = factor['inverse_whitener']@(head_w-head_w.mean(0)).T
            reverse = {int(v): j for j, v in enumerate(fit)}
            with np.load(OLD/f'training_episodes_{seed}.npz') as f:
                episodes = [(np.array([reverse[int(v)] for v in s]), np.array([reverse[int(v)] for v in q]))
                            for s, q in zip(f['support_obs_ids'], f['query_obs_ids'])]
            priors = dict(original=prior)
            full_probe = fit_probe(u[fit]/np.sqrt(32), labels[fit], weights)
            priors['full_refit'] = score_probe(u/np.sqrt(32), full_probe)
            for method in METHODS:
                t = dict(**tag, method=method)
                q = factor['bases'][method]; r = complement(q)
                fpart, fremain = u@q/np.sqrt(7), u@r/np.sqrt(25)
                error = float(np.max(np.abs((u[held]@q@q.T+u[held]@r@r.T)-u[held])))
                decomposition_error = max(decomposition_error, error)
                assert error < 1e-9
                coef = fit_probe(fpart[fit], labels[fit], weights)
                rcoef = fit_probe(fremain[fit], labels[fit], weights)
                candidate_scores, remainder_scores = score_probe(fpart, coef), score_probe(fremain, rcoef)
                contribution = (u@q)@(q.T@hweight)
                kept = prior-(u@hweight-contribution)
                removed = prior-contribution
                if method == 'head':
                    np.testing.assert_allclose(kept, prior, atol=1e-6)
                correction = fit_episode_correction(fpart[fit], prior[fit], labels[fit], episodes, kernel[fit])
                delta = score_probe(fpart, correction)
                gain, trials = validate_correction(prior, delta, kernel, labels, validation)
                val_rows.extend(dict(**t, **row) for row in trials)
                selections.append(dict(**t, gain=gain, drift_scale=factor['drift_scale']))
                priors.update({method+'_probe7': candidate_scores, method+'_probe25': remainder_scores,
                               method+'_keep': kept, method+'_remove': removed,
                               method+'_corrected': prior+gain*delta})
                np.savez_compressed(OUT/f'{origin}_{method}_{seed}_vectors.npz',
                    obs_ids=np.arange(len(meta)), z_candidate=fpart.astype(np.float32),
                    z_remainder=fremain.astype(np.float32), candidate_scores=candidate_scores.astype(np.float32),
                    remainder_scores=remainder_scores.astype(np.float32))
                arrays = {k: v for k, v in factor.items() if isinstance(v, np.ndarray)}
                np.savez_compressed(OUT/f'{origin}_{method}_{seed}_mapping.npz',
                    **arrays, q=q, q_remainder=r, probe=coef, remainder_probe=rcoef,
                    correction=correction, selected_gain=gain, **{'old_'+k: v for k, v in saved_mapping.items()})
                for scope, values in [('candidate', fpart), ('remainder', fremain)]:
                    geometry_rows.extend(dict(**t, part=scope, role='held', **row) for row in geometry(values[held], hm))
                    geometry_rows.extend(dict(**t, part=scope, role='fit', **row) for row in geometry(values[fit], fm))
                for j in range(7):
                    prototype = factor['class_centroids']@q[:, j]
                    contribution_to_head = q[:, j]@hweight
                    cards.append(dict(**t, axis=j, source_high_class=CLASSES[int(prototype.argmax())],
                        source_low_class=CLASSES[int(prototype.argmin())],
                        class_signal=float(q[:, j]@factor['signal']@q[:, j]),
                        fit_crossport_drift=float(q[:, j]@factor['drift']@q[:, j]),
                        head_high_class=CLASSES[int(contribution_to_head.argmax())],
                        head_low_class=CLASSES[int(contribution_to_head.argmin())]))
                for other in METHODS:
                    oq = factor['bases'][other]
                    overlap.append(dict(**t, other=other, mean_squared_cosine=float(np.square(q.T@oq).sum()/7)))
                for name, p in [('probe7', candidate_scores), ('corrected', priors[method+'_corrected'])]:
                    pair_rows.extend(dict(**t, variant=name, **row) for row in paired_correctness(p[held], hm))
                print('FACTOR', origin, seed, method, 'source_gain', gain, flush=True)
            geometry_rows.extend(dict(**tag, method='full', part='full', role='held', **row) for row in geometry(u[held]/np.sqrt(32), hm))
            pair_rows.extend(dict(**tag, method='original', variant='original', **row) for row in paired_correctness(prior[held], hm))
            for variant, p in priors.items():
                source_rows.append(dict(**tag, variant=variant, **metrics(p[held].argmax(1), labels[held])))
            for task, (s, query, sy) in tasks.items():
                archived = dict(support_obs_ids=s, query_obs_ids=query)
                for variant, p in priors.items():
                    score = target_readout(p, kernel, s, query, sy)
                    archived[variant] = score
                    archived[variant+'_raw'] = np.asarray(p[query], np.float32)
                    if variant == 'original':
                        with np.load(PARENT/f'{task}_predictions.npz') as old:
                            np.testing.assert_array_equal(score, old[f'{origin}__{seed}'])
                            replays += 1
                # Report how much the fixed support correction cancels new evidence.
                op, present = support_operator(kernel[s], kernel[query], sy)
                for method in METHODS:
                    delta = priors[method+'_corrected']-prior
                    support_effect = -op@(delta[s]*present[None, :])
                    total = delta[query]+support_effect
                    accounting.append(dict(**tag, method=method, task=task,
                        query_delta_rms=float(np.sqrt(np.square(delta[query]).mean())),
                        support_delta_rms=float(np.sqrt(np.square(support_effect).mean())),
                        net_delta_rms=float(np.sqrt(np.square(total).mean())),
                        gain=next(row['gain'] for row in selections if all(row[k] == v for k, v in dict(**tag, method=method).items()))))
                np.savez_compressed(OUT/f'{origin}_{seed}_{task}_predictions.npz', **archived)
    pd.concat(coverage, ignore_index=True).to_csv(OUT/'fit_pair_coverage.csv', index=False)
    for name, rows in [('geometry', geometry_rows), ('source_results', source_rows), ('held_pair_correctness', pair_rows),
                       ('selection', selections), ('validation', val_rows), ('factor_cards', cards),
                       ('support_cancellation', accounting), ('subspace_overlap', overlap)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv', index=False)
    save_json(OUT/'PREDICTIONS_SAVED.json', dict(target_metrics_read=False, exact_baseline_replays=replays,
                                               max_whitened_reconstruction_error=decomposition_error))
    results, class_rows = [], []
    for origin in ORIGINS:
        for seed in SEEDS:
            for task, (_, query, _) in tasks.items():
                truth = labels[query]
                with np.load(OUT/f'{origin}_{seed}_{task}_predictions.npz') as f:
                    baseline = f['original'].argmax(1)
                    for variant in [k for k in f.files if k not in ('support_obs_ids', 'query_obs_ids') and not k.endswith('_raw')]:
                        pred, raw_pred = f[variant].argmax(1), f[variant+'_raw'].argmax(1)
                        tag = dict(origin=origin, seed=seed, task=task, port=meta.iloc[query].port.iloc[0], variant=variant)
                        raw_metrics = metrics(raw_pred, truth)
                        results.append(dict(**tag, **metrics(pred, truth), raw_BA=raw_metrics['BA'], raw_OA=raw_metrics['OA']))
                        for c in np.unique(truth):
                            mask = truth == c
                            class_rows.append(dict(**tag, class_id=int(c), count=int(mask.sum()),
                                recall=float(100*(pred[mask] == c).mean()),
                                rescued=int(((baseline != truth)&(pred == truth)&mask).sum()),
                                harmed=int(((baseline == truth)&(pred != truth)&mask).sum())))
    df = pd.DataFrame(results); df.to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'target_classes.csv', index=False)
    for name, keys in [('summary', ['origin', 'variant']), ('seed_summary', ['origin', 'variant', 'seed']),
                       ('port_summary', ['origin', 'variant', 'port'])]:
        df.groupby(keys)[['OA', 'BA', 'raw_OA', 'raw_BA']].mean().reset_index().to_csv(OUT/f'target_{name}.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-started, new_neural_fits=0,
        candidate_mappings=20, new_linear_correction_fits=20, source_only_gain_selection=True,
        exact_baseline_replays=replays, main_model_modified=False, causal_identification_proven=False))
    print(df[df.variant.str.endswith('corrected')|df.variant.eq('original')].groupby(['origin', 'variant'])[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
