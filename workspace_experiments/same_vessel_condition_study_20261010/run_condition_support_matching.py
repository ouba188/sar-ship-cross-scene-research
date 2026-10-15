"""Small source-held classification bridge for the spatial-volatility candidate."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from condition_support_matching import ARMS, source_limits, multipliers, incremental_factors, score_episode, metrics, decision_accounting

BASE = Path(__file__).resolve().parent
OUT = BASE/'condition_support_matching_20261011'
OLD = BASE/'condition_evidence_20261011'
APP = BASE/'condition_applicability_20261011'
PARENT = BASE/'incidence_transport_20261011'
SEEDS = (31415, 27182)
GAINS = (0., .5, 1.)
INCREMENT_ONLY = False


def load_npz(path):
    with np.load(path) as f:
        return {k: f[k].copy() for k in f.files}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    start = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    meta = pd.read_csv(BASE/'observations.csv')
    labels = meta.class_id.to_numpy()
    cohorts = load_npz(PARENT/'source_cohorts.npz')
    fit, held = cohorts['fit_obs_ids'], cohorts['held_obs_ids']
    evidence = load_npz(OLD/'evidence_features.npz')
    features = np.full((len(meta), 6), np.nan)
    features[evidence['obs_ids']] = evidence['features']
    ratio = features[:, 5]
    context = np.full(len(meta), np.nan)
    available = np.isfinite(features).all(1)
    context[available] = np.logaddexp.reduce(features[available, :3], axis=1)
    models = {arm: load_npz(OLD/'models'/f'0_center_border_log_ratio_magnitude_{arm}.npz')
              for arm in ('anchor', 'capacity', 'condition', 'sham')}
    gate = load_npz(APP/'gate.npz')
    limits = source_limits(meta, fit, cohorts['fit_pairs'], ratio)
    save_json(OUT/'PROTOCOL.json', dict(
        hypothesis='Can an observed condition-related within-vessel spatial-change scale improve use of support cases from different vessels? This cross-vessel use is an experiment, not assumed valid.',
        fixed='Cached full source prior/case models, original64support3draws,2seeds; no backbone/head training or target dataset evaluation. Source-held823/302ships is historically reused development data.',
        response='Old center_border_log_ratio magnitude regressions anchor/capacity/condition/sham unchanged. New applicability gate uses sourceOOF response errors only, never held identities/ports as rules.',
        matching='Original balanced signed support residual influence O_qs is multiplied by exp(-.5*(r_q-r_s)^2/tolerance^2). Tolerance is symmetric mean of two directed forecasts, floored at source weighted25th change percentile. Fixed uses sourceweightedmedian. More predicted movement tolerates a larger observed difference; NOT suppressing all high-volatility observations.',
        increment_only=INCREMENT_ONLY,
        residual_formula='signed_target_minus_prior_v2',
        residual_erratum='Increment now uses (2*onehot-1-prior_s)*present, matching baseline. Historical unversioned runs omitted -1 in the increment; preserve their archives and replay explicitly as legacy.',
        corrective_formula='If increment_only=True, final multiplier=1+(raw_multiplier_arm-raw_multiplier_anchor), in[0,2]. This cancels generic center attenuation and isolates conditional/capacity/sham increments. Exactly1 for anchor and unknown edges. No parameter changes or new response fits. Post-negative developmental correction, NOT independentconfirmation.',
        gate='Mix A-only and angle-conditioned directed forecasts using sourceOOF shallow applicability model, then symmetrize. Both support/query images used for matching, but query labels are never inputs.',
        restriction='Only sameport/platform/pass, knownmetadata, sourceknownangle range and sourcepair maxgap. Missing/out-of-scope edge multiplier exactly1. Same-vessel learning does not ensure different-vessel applicability.',
        operator='This is post-solve support-influence calibration, not a refitted PSD kernel or changed bone/feature. Support residual coefficients fixed; originalprior preserved. Attenuation may remove helpful AND harmful signed contributions.',
        selection='Gain0/.5/1 chosen by mean BA on original3val_new_ship episodes ONLY, lower gain ties. Validation was used historically by upstream checkpoint selection, so reused development not independentconfirmation. No new-time role or held-source gain selection.',
        readouts='Matched baseline and6arms; selectedgain andforced1 diagnostic, transparent sourceOA/BA, rescues/harms and exact fixed-baseline-rival margin contributions. Forced1 not target-selected deployment.',
        scale_limits=limits))
    save_json(OUT/'LIMITS.json', limits)
    np.savez_compressed(OUT/'gate.npz', **gate)
    for arm, model in models.items():
        np.savez_compressed(OUT/f'response_{arm}.npz', **model)
    with np.load(BASE/'crossfit_evidence_20261011/validation_episodes.npz') as f:
        val = [(f[f'val_new_ship_{j}_support'], f[f'val_new_ship_{j}_query']) for j in range(3)]
    validations, choices, rows, classes, account, coverage = [], [], [], [], [], []
    baseline_error = 0.
    for seed in SEEDS:
        bundle = load_npz(PARENT/f'full_{seed}_inputs.npz')
        prior, kernel = bundle['prior'].astype(float), bundle['kernel'].astype(float)
        for draw, (s, q) in enumerate(val):
            if not np.isfinite(features[np.r_[s, q]]).all():
                raise ValueError('Validation evidence missing')
            if set(meta.iloc[s].mmsi) & set(meta.iloc[q].mmsi):
                raise ValueError('Support/query identity overlap')
            factor, detail = multipliers(meta, s, q, ratio, context, models, gate, limits)
            if INCREMENT_ONLY:
                factor = incremental_factors(factor)
            base, delta, op, residual = score_episode(prior, kernel, s, q, labels[s], factor)
            for arm in ARMS:
                for gain in GAINS:
                    validations.append(dict(seed=seed, draw=draw, arm=arm, gain=gain,
                                            **metrics(base+gain*delta[arm], labels[q])))
        frame = pd.DataFrame(validations)
        chosen = {}
        for arm in ARMS:
            means = frame[(frame.seed == seed) & (frame.arm == arm)].groupby('gain').BA.mean()
            chosen[arm] = float(means.index[np.argmax(means.to_numpy())])
            choices.append(dict(seed=seed, arm=arm, gain=chosen[arm], validation_BA=float(means.max())))
        save_json(OUT/f'SELECTION_FROZEN_{seed}.json', dict(chosen=chosen, held_source_metrics_read=False))
        print('SELECTED', seed, chosen, flush=True)
        for draw in range(3):
            old = load_npz(PARENT/f'{seed}_source_{draw}_predictions.npz')
            s, q = old['support_obs_ids'], old['query_obs_ids']
            if not set(q).issubset(set(held)) or not set(s).issubset(set(held)):
                raise ValueError('Historical held episode membership changed')
            if set(meta.iloc[s].mmsi) & set(meta.iloc[q].mmsi):
                raise ValueError('Support/query identity overlap')
            factor, detail = multipliers(meta, s, q, ratio, context, models, gate, limits)
            if INCREMENT_ONLY:
                factor = incremental_factors(factor)
            base, delta, op, residual = score_episode(prior, kernel, s, q, labels[s], factor)
            np.testing.assert_array_equal(base.argmax(1), old['baseline'].argmax(1))
            baseline_error = max(baseline_error, float(np.max(np.abs(base-old['baseline']))))
            archive = dict(support_obs_ids=s, query_obs_ids=q, support_labels=labels[s],
                           baseline=base, valid=detail['valid'])
            for arm in ARMS:
                archive[arm+'_selected'] = base+chosen[arm]*delta[arm]
                archive[arm+'_forced1'] = base+delta[arm]
                archive[arm+'_multiplier'] = factor[arm]
            np.savez_compressed(OUT/f'{seed}_source_{draw}_predictions.npz', **archive)
            # Selection is already frozen; evaluate labels only after saving predictions.
            rows.append(dict(seed=seed, draw=draw, arm='baseline', deployment='selected', gain=0., **metrics(base, labels[q])))
            for arm in ARMS:
                for dep, gain in [('selected', chosen[arm]), ('forced1', 1.)]:
                    score = archive[f'{arm}_{dep}']
                    m = 1+gain*(factor[arm]-1)
                    fixed = 1+gain*(factor['fixed']-1)
                    rows.append(dict(seed=seed, draw=draw, arm=arm, deployment=dep, gain=gain, **metrics(score, labels[q])))
                    account.append(dict(seed=seed, draw=draw, arm=arm, deployment=dep, gain=gain,
                        **decision_accounting(base, score, op, residual, m, fixed, labels[q], detail['valid'])))
                    for c in np.unique(labels[q]):
                        mask = labels[q] == c
                        classes.append(dict(seed=seed, draw=draw, arm=arm, deployment=dep, class_id=int(c),
                            count=int(mask.sum()), baseline_recall=float(100*np.mean(base[mask].argmax(1)==c)),
                            recall=float(100*np.mean(score[mask].argmax(1)==c))))
            coverage.append(dict(seed=seed, draw=draw, queries=len(q), supports=len(s),
                possible_edges=len(q)*len(s), eligible_edges=int(detail['valid'].sum()),
                eligible_queries=int(detail['valid'].any(1).sum()), mean_gate=float(detail['gate'][detail['valid']].mean())
                if detail['valid'].any() else 0.))
    for name, data in [('validation', validations), ('selection', choices), ('results', rows),
                       ('classes', classes), ('accounting', account), ('coverage', coverage)]:
        pd.DataFrame(data).to_csv(OUT/f'{name}.csv', index=False)
    frame = pd.DataFrame(rows)
    frame.groupby(['arm', 'deployment'])[['OA', 'BA', 'MSE']].mean().reset_index().to_csv(OUT/'summary.csv', index=False)
    frame.groupby(['arm', 'deployment', 'seed'])[['OA', 'BA', 'MSE']].mean().reset_index().to_csv(OUT/'seed_summary.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-start, neural_fits=0,
        original_baseline_predictions_exact=6, baseline_max_score_error=baseline_error,
        held_source_episodes=6, target_port_evaluations=0, main_model_modified=False))
    print(frame.groupby(['arm', 'deployment'])[['OA', 'BA', 'MSE']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
