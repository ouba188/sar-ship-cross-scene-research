"""Source-only crossfit/full transfer diagnostics; no fitting or gain selection."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
from pathlib import Path

import numpy as np
import pandas as pd
from explicit_candidate_projection import identity_episode
from paired_evidence_factors import support_operator, adapted_scores, score_probe


BASE = Path(__file__).resolve().parent
ROOT = BASE/'crossfit_evidence_20261011'
SEEDS = (31415, 27182)
BUNDLES = ('full', 'fold0', 'fold1', 'fold2')
ORIGINS = ('full_in', 'size_matched_in', 'crossfit')
METHODS = ('paired', 'class_only')
ROLES = ('val_new_ship', 'val_new_time', 'test_source')
METRICS = ('OA', 'BA', 'MSE', 'unbalanced_MSE')


def source_episodes(meta, roles, validation_path):
    """Retain the old validation draws; sample held-source draws once, seed-free."""
    episodes = {}
    with np.load(validation_path) as saved:
        for role in ROLES[:2]:
            episodes[role] = [(saved[f'{role}_{j}_support'].copy(),
                               saved[f'{role}_{j}_query'].copy()) for j in range(3)]
    ids = roles['test_source']
    local = meta.loc[ids].reset_index(drop=True)
    episodes['test_source'] = []
    for j in range(3):
        s, q = identity_episode(local, np.random.default_rng(91011+j), query_size=None)
        episodes['test_source'].append((ids[s], ids[q]))
    for role, draws in episodes.items():
        for s, q in draws:
            if len(s) != 64 or not len(q) or len(np.unique(s)) != 64:
                raise ValueError(f'{role}: expected 64 distinct support rows and nonempty query')
            if not np.isin(np.r_[s, q], roles[role]).all():
                raise ValueError(f'{role}: episode leaves its source role')
            support_ships = set(meta.loc[s, 'mmsi'])
            if len(support_ships) != 64 or support_ships & set(meta.loc[q, 'mmsi']):
                raise ValueError(f'{role}: support/query vessel overlap')
            remaining = roles[role][~meta.loc[roles[role], 'mmsi'].isin(support_ships).to_numpy()]
            if not np.array_equal(np.sort(q), np.sort(remaining)):
                raise ValueError(f'{role}: query must contain all remaining vessel observations')
    return episodes


def score_metrics(scores, labels):
    prediction = scores.argmax(1)
    squared = np.square(scores-(2*np.eye(8)[labels]-1)).mean(1)
    classes = np.unique(labels)
    return dict(OA=float(100*np.mean(prediction == labels)),
                BA=float(np.mean([100*np.mean(prediction[labels == c] == c) for c in classes])),
                MSE=float(np.mean([squared[labels == c].mean() for c in classes])),
                unbalanced_MSE=float(squared.mean()))


def gain_policies(selected):
    if selected not in (0., .5, 1.):
        raise ValueError('Saved full-model gain must be one of 0, .5, 1')
    return (('selected_full', selected), ('fixed_grid', 0.),
            ('fixed_grid', .5), ('forced', 1.))


def episode_scores(prior_s, prior_q, delta_s, delta_q, gain, operator, present, labels):
    changed_s, changed_q = prior_s+gain*delta_s, prior_q+gain*delta_q
    return dict(raw=changed_q,
                adapted=adapted_scores(changed_s, changed_q, operator, present, labels))


def evaluate_bundle(bundle, labels, meta, episode_ids, episodes, corrections, name, seed):
    """Arrays use compact source indices; metadata and saved episodes use global IDs."""
    prior, kernel = bundle['prior'], bundle['kernel']
    fit_ships = set(meta.loc[bundle['fit_obs_ids'], 'mmsi'])
    excluded_ships = set(meta.loc[bundle['excluded_obs_ids'], 'mmsi'])
    deltas = {(origin, method): score_probe(bundle['feature_'+method], coefficient)
              for (origin, method), (coefficient, _) in corrections.items()}
    rows = []
    for role, draws in episodes.items():
        for draw, (support, query) in enumerate(draws):
            s, q = np.searchsorted(episode_ids, support), np.searchsorted(episode_ids, query)
            sy, qy = labels[s], labels[q]
            operator, present = support_operator(kernel[s], kernel[q], sy)
            baseline = episode_scores(prior[s], prior[q], 0., 0., 0., operator, present, sy)
            reference = {readout: score_metrics(scores, qy) for readout, scores in baseline.items()}
            query_ships = set(meta.loc[query, 'mmsi'])
            support_ships = set(meta.loc[support, 'mmsi'])
            tag = dict(bundle=name, seed=seed, role=role, draw=draw,
                       support_count=len(s), query_count=len(q), query_ships=len(query_ships),
                       support_fit_ships=len(support_ships & fit_ships),
                       query_fit_ships=len(query_ships & fit_ships),
                       query_excluded_ships=len(query_ships & excluded_ships))
            for readout, result in reference.items():
                rows.append(dict(**tag, correction='baseline', method='none', policy='baseline',
                                 selected_gain=0., gain=0., readout=readout, **result,
                                 **{metric+'_change': 0. for metric in METRICS},
                                 score_delta_rms=0.))
            for (origin, method), (_, selected) in corrections.items():
                delta = deltas[origin, method]
                for policy, gain in gain_policies(selected):
                    changed = episode_scores(prior[s], prior[q], delta[s], delta[q], gain,
                                             operator, present, sy)
                    for readout, scores in changed.items():
                        result = score_metrics(scores, qy)
                        rows.append(dict(**tag, correction=origin, method=method, policy=policy,
                            selected_gain=selected, gain=gain, readout=readout, **result,
                            **{metric+'_change': result[metric]-reference[readout][metric]
                               for metric in METRICS},
                            score_delta_rms=float(np.sqrt(np.square(scores-baseline[readout]).mean()))))
    return rows


def full_fold_comparison(results):
    """Paired changes on the SAME queries; no fold ensembles or gain reselection."""
    keys = ['seed', 'role', 'draw', 'correction', 'method', 'policy', 'gain', 'readout']
    values = list(METRICS)+[metric+'_change' for metric in METRICS]+['score_delta_rms']
    full = results.loc[results.bundle.eq('full'), keys+values]
    folds = results.loc[results.bundle.ne('full')]
    paired = folds.merge(full, on=keys, suffixes=('_fold', '_full'), validate='many_to_one')
    for value in values:
        paired['full_minus_fold_'+value] = paired[value+'_full']-paired[value+'_fold']
    return paired


def main():
    with np.load(BASE/'grouped_condition_anchor_20261011/source_split.npz') as split:
        roles = {role: split[role].copy() for role in ROLES}
        source_ids = np.unique(np.concatenate([split['factor_fit']]+list(roles.values())))
    # The shared manifest supplies source metadata only; no target artifacts are opened.
    meta = pd.read_csv(BASE/'observations.csv', usecols=['mmsi', 'class_id']).loc[source_ids]
    episodes = source_episodes(meta, roles, ROOT/'validation_episodes.npz')
    episode_ids = np.unique(np.concatenate([np.r_[s, q] for draws in episodes.values() for s, q in draws]))
    labels = meta.loc[episode_ids, 'class_id'].to_numpy()
    rows, coefficient_rows = [], []
    for seed in SEEDS:
        corrections = {}
        for origin in ORIGINS:
            for method in METHODS:
                with np.load(ROOT/f'correction_{origin}_{method}_{seed}.npz') as saved:
                    coefficient = saved['coefficient'].astype(np.float64)
                    selected = float(saved['selected_gain'].item())
                if coefficient.shape != (8, 8):
                    raise ValueError('Expected 64 correction parameters, including intercept')
                gain_policies(selected)
                corrections[origin, method] = coefficient, selected
                coefficient_rows.append(dict(seed=seed, correction=origin, method=method,
                    selected_gain=selected, parameter_count=coefficient.size,
                    coefficient_norm=float(np.linalg.norm(coefficient)),
                    slope_norm=float(np.linalg.norm(coefficient[:-1])),
                    intercept_norm=float(np.linalg.norm(coefficient[-1]))))
        for name in BUNDLES:
            with np.load(ROOT/f'{name}_{seed}_outputs.npz') as saved:
                bundle = {key: saved[key][episode_ids].astype(np.float64)
                          for key in ('prior', 'kernel', 'feature_paired', 'feature_class_only')}
                bundle.update({key: saved[key].copy() for key in ('fit_obs_ids', 'excluded_obs_ids')})
            rows.extend(evaluate_bundle(bundle, labels, meta, episode_ids, episodes, corrections, name, seed))
            print('SOURCE_SHIFT', name, seed, flush=True)
    result = pd.DataFrame(rows)
    out = ROOT/'mechanism'; out.mkdir(exist_ok=True)
    episode_dump = {f'{role}_{j}_{part}': ids for role, draws in episodes.items()
                    for j, pair in enumerate(draws) for part, ids in zip(('support', 'query'), pair)}
    np.savez_compressed(out/'source_shift_episodes.npz', **episode_dump)
    result.to_csv(out/'source_shift_metrics.csv', index=False)
    groups = ['bundle', 'seed', 'role', 'correction', 'method', 'policy', 'gain', 'readout']
    values = list(METRICS)+[metric+'_change' for metric in METRICS]+['score_delta_rms']
    summary = result.groupby(groups, sort=False)[values].mean().reset_index()
    summary.to_csv(out/'source_shift_summary.csv', index=False)
    full_fold_comparison(result).to_csv(out/'source_shift_full_minus_fold.csv', index=False)
    pd.DataFrame(coefficient_rows).to_csv(out/'source_shift_coefficients.csv', index=False)
    protocol = dict(
        purpose='Distinguish crossfit-to-full transfer changes from correction generalization failure.',
        scope='Source development diagnostics only, including previously used test_source; NOT independent confirmation.',
        source_roles=ROLES, seeds=SEEDS, bundles=BUNDLES, correction_origins=ORIGINS, methods=METHODS,
        episodes='Original 3 validation draws per role. test_source uses identity_episode with seeds91011/91012/91013, 64 distinct support ships and ALL remaining ships as query. Same global IDs for every model, seed and correction.',
        selection='No fitting or selection here. Every coefficient and selected_gain is loaded from the runner; selected_full applies that saved gain unchanged to each fold. fixed_grid0/.5 and forced1 are diagnostics only, never deployment choices.',
        readout='Own frozen case kernel for each bundle; tested support_operator ridge8 with balanced support residual and absent-class mask. Correction changes BOTH support and query priors.',
        metrics='OA/BA in percent; MSE is CLASS-BALANCED mean squared error against2*onehot-1, averaged over query-present classes, matching runner MSE and training balanced_mse. unbalanced_MSE is the ordinary sample mean. Raw and adapted scores use the same query IDs. *_change subtracts the same bundle/episode/readout baseline.',
        aggregation='Arithmetic mean over3 draws per seed; overlapping draws/folds are not independent samples. No significance tests or target measurements.',
        comparison='full_minus_fold columns pair the SAME episode/seed/coefficient/gain/readout; positive BA_change difference means correction is more helpful on full, positive MSE_change difference means less helpful. Descriptive transfer evidence, not a unique causal attribution.',
        identity='val_new_time can overlap base fitting identities differently across folds; episode rows report those counts. val_new_ship/test_source are unseen-vessel source roles. New-ship validation also selected base checkpoints and gains.',
        size_matched_in='The runner fits seen corrections using fold(k+1)%3 on episode batchk; crossfit uses foldk on the same batch. Both bases use about350ships. This diagnostic loads their saved coefficients without reconstructing or refitting training views.',
        boundaries='No target files, target metrics, backbone features, model fitting, gain reselection or model changes. Training-residual loss is not recomputed here.',
        rows=len(result), correction_files=len(coefficient_rows), bundles_evaluated=len(SEEDS)*len(BUNDLES))
    (out/'SOURCE_SHIFT_PROTOCOL.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    report = summary.loc[summary.readout.eq('adapted') & summary.correction.eq('crossfit') &
                         summary.policy.isin(['selected_full', 'forced']),
                         groups+['BA_change', 'MSE_change']]
    print(report.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
