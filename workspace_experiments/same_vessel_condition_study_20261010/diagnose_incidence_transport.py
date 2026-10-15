"""Source pair-response diagnostics and response-only leave-port-out fits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from incidence_transport import ARMS, fit_response, matched_pairs, predict_delta, shuffled_angles

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'incidence_transport_20261011'
OUT = ROOT / 'mechanism'


def pair_weights(table):
    """Equal classes, then equal ships within class, then directed pairs."""
    if not len(table):
        return np.empty(0, dtype=float)
    counts = table.groupby(['class_id', 'mmsi']).a.transform('size')
    ships = table[['class_id', 'mmsi']].drop_duplicates().groupby('class_id').size()
    return 1. / (table.class_id.nunique() * table.class_id.map(ships).to_numpy(float)
                 * counts.to_numpy(float))


def pair_metrics(table, actual, predicted):
    """Uncentered delta-energy reduction versus zero, with direction/scale split.

    Amplitude ratios are descriptive held-data diagnostics, never recalibration.
    All energies sum feature dimensions before class/ship/pair averaging.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.ndim != 2 or actual.shape != predicted.shape or len(actual) != len(table):
        raise ValueError('Pair table and actual/predicted delta arrays must align')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Pair deltas must be finite; do not silently drop predictions')
    pairs = table[['a', 'b']].to_numpy(dtype=np.int64)
    result = dict(directed_pairs=len(table),
                  undirected_pairs=len(np.unique(np.sort(pairs, axis=1), axis=0)),
                  ships=int(table.mmsi.nunique()), classes=int(table.class_id.nunique()),
                  ports=int(table.port.nunique()), observations=len(np.unique(pairs)))
    if not len(table):
        return result
    w = pair_weights(table)
    true_energy = np.square(actual).sum(1)
    pred_energy = np.square(predicted).sum(1)
    dot = (actual * predicted).sum(1)
    error_energy = np.square(actual - predicted).sum(1)
    norm = np.sqrt(pred_energy)
    projection = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
    parallel_error = np.square(projection - norm)
    orthogonal_error = np.maximum(true_energy - np.square(projection), 0)
    e0, ep, cross, err = (float(w @ x) for x in
                          (true_energy, pred_energy, dot, error_energy))
    result.update(
        zero_delta_energy=e0, prediction_energy=ep, error_energy=err,
        explained_percent=100 * (1 - err / e0) if e0 > 0 else np.nan,
        error_rms=float(np.sqrt(err)), observed_delta_rms=float(np.sqrt(e0)),
        prediction_rms=float(np.sqrt(ep)),
        error_rms_over_observed=float(np.sqrt(err / e0)) if e0 > 0 else np.nan,
        prediction_rms_over_observed=float(np.sqrt(ep / e0)) if e0 > 0 else np.nan,
        signed_alignment=cross,
        mean_signed_projection=float(w @ projection),
        mean_prediction_norm=float(w @ norm),
        directional_cosine=cross / np.sqrt(e0 * ep) if e0 > 0 and ep > 0 else np.nan,
        amplitude_ratio_diagnostic=cross / ep if ep > 0 else np.nan,
        parallel_error_energy=float(w @ parallel_error),
        orthogonal_error_energy=float(w @ orthogonal_error),
        nonzero_prediction_weight=float(w @ (pred_energy > 0)),
        positive_projection_weight=float(w @ ((dot > 0) & (pred_energy > 0))),
        mean_absolute_angle_gap=float(w @ table.delta_angle.abs().to_numpy(float)))
    return result


def diagnose_predictions(table, actual, predictions, *, cohort, model, seed):
    """Report identical slices for all arms, including an explicit zero baseline.

    table uses global observation IDs a/b and mmsi, class_id, port, delta_angle,
    both_scenes_seen. Rows of actual and every prediction match table exactly.
    model labels full/fold0/...; cohort labels sourcefit/heldraw explicitly.
    """
    table = table.reset_index(drop=True)
    actual = np.asarray(actual)
    arms = {'zero': np.zeros_like(actual), **predictions}
    gap = table.delta_angle.abs().to_numpy(float)
    seen = table.both_scenes_seen.to_numpy(bool)
    subsets = dict(all=np.ones(len(table), bool), angle_gap2=gap >= 2,
                   near_angle=gap < 2, both_scenes_seen=seen,
                   both_scenes_seen_gap2=seen & (gap >= 2))
    rows = []
    for subset, subset_mask in subsets.items():
        slices = [('overall', 'all', subset_mask)]
        for column in ('class_id', 'port'):
            for value in sorted(table.loc[subset_mask, column].unique()):
                slices.append((column, str(value), subset_mask & (table[column].to_numpy() == value)))
        for scope, value, mask in slices:
            for arm, predicted in arms.items():
                metrics = pair_metrics(table.loc[mask], actual[mask], np.asarray(predicted)[mask])
                rows.append(dict(model=model, seed=seed, cohort=cohort, subset=subset,
                                 scope=scope, value=value, arm=arm, **metrics))
    return pd.DataFrame(rows)


def paired_arm_comparisons(results, reference='angle_context'):
    """Compare matched rows, without treating directed pairs as independent."""
    keys = ['model', 'seed', 'cohort', 'subset', 'scope', 'value']
    selected = results.loc[results.arm == reference, keys + ['error_energy', 'explained_percent']]
    selected = selected.rename(columns={'error_energy': 'reference_error_energy',
                                         'explained_percent': 'reference_explained_percent'})
    result = results.loc[results.arm != reference].merge(selected, on=keys, validate='many_to_one')
    result['reference_arm'] = reference
    result['reference_error_reduction'] = result.error_energy - result.reference_error_energy
    result['reference_explained_gain_pp'] = result.reference_explained_percent - result.explained_percent
    return result


def write_reports(results, port_transfer=None, *, out=OUT, reference='angle_context'):
    """Write source diagnostics supplied by the main runner's confirmed API.

    port_transfer contains held-port predictions from coefficients refit after
    excluding that port. Base encoders and context maps are NOT held-port-out.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / 'pair_prediction_metrics.csv', index=False)
    paired_arm_comparisons(results, reference).to_csv(out / 'arm_comparisons.csv', index=False)
    if port_transfer is not None:
        port_transfer = port_transfer.copy()
        port_transfer['base_encoder_is_lopo'] = False
        port_transfer['response_fit_excludes_held_port'] = True
        port_transfer.to_csv(out / 'response_leave_held_port_out.csv', index=False)
    protocol = dict(
        scope='Source-only raw delta prediction; no target query access or neural training.',
        weighting='Equal represented classes, equal MMSIs within class, equal directed pairs within MMSI; reweighted within each reported slice.',
        independence='Reverse pairs and repeated endpoints are dependent. Counts include undirected pairs and unique MMSIs; no pair-level confidence intervals or bootstrap.',
        metric='Explained percent = 100*(1-weighted squared prediction error/weighted squared actual delta). Uncentered energy, not classification accuracy or causal variance explained.',
        projection='Actual delta projected onto each predicted unit direction. Zero prediction has no direction, projection is recorded as zero and its full error is orthogonal by convention.',
        amplitude='Weighted dot(actual,predicted)/weighted norm(predicted)^2. Descriptive only; never used to tune a coefficient or select an arm.',
        scene_seen='Both products occur in the corresponding base encoder fitting IDs; cohort remains identity-held where labelled heldraw.',
        lopo='Only response coefficients exclude the held port. Frozen base encoder, whitening and class context are NOT leave-port-out.',
        near_angle='Near-angle and >=2-degree results are separate; aggregate gains can reflect near-zero responses or a no-condition intercept.',
        fitting='Response-only LOPO calls main fit_response with fixed ridge and within-cell angle shuffle; no duplicated estimator, neural training, or tuning.',
        sham='All main pair_prediction_metrics, including sourcefit, use real angles. Separate sham_training_objective reports the actual shuffled-angle training objective.',
        seed_summary='Per-seed and per-base-model results retained; no cross-fold mixing of native representation axes.')
    (out / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')


def directed_table(meta, pairs, base_fitting):
    directed = np.concatenate([pairs, pairs[:, ::-1]])
    a, b = directed.T
    angle = meta.incidence_deg.to_numpy(float)
    products = meta['product'].to_numpy()
    seen = set(products[base_fitting])
    return pd.DataFrame(dict(a=a, b=b, mmsi=meta.iloc[a].mmsi.to_numpy(),
        class_id=meta.iloc[a].class_id.to_numpy(), port=meta.iloc[a].port.to_numpy(),
        delta_angle=angle[b] - angle[a], scene_a=products[a], scene_b=products[b],
        both_scenes_seen=np.isin(products[a], list(seen)) & np.isin(products[b], list(seen))))


def evaluate_models(z, context, meta, pairs, models, base_fitting, *, cohort, model, seed):
    table = directed_table(meta, pairs, base_fitting)
    a, b = table[['a', 'b']].to_numpy().T
    angle = meta.incidence_deg.to_numpy(float)
    actual = z[b] - z[a]
    predictions = {arm: predict_delta(context[a], angle[a], angle[b], response)
                   for arm, response in models.items()}
    result = diagnose_predictions(table, actual, predictions, cohort=cohort, model=model, seed=seed)
    result['evaluation_angles'] = 'real'
    return result, table, actual, predictions


def angle_coverage(meta, table, fit_table, *, cohort, model, seed):
    """Concentration of fixed class/ship/pair weighted angle-squared energy."""
    weights = pair_weights(table)
    gap = table.delta_angle.abs().to_numpy(float)
    energy = weights * (gap / 5) ** 2
    ships = table[['class_id', 'mmsi']].copy()
    ships['angle_energy'] = energy
    ships = ships.groupby(['class_id', 'mmsi'], as_index=False).angle_energy.sum()
    total = float(energy.sum())
    ships['energy_share'] = ships.angle_energy / total if total else 0.
    ships = ships.sort_values('energy_share', ascending=False)
    positive = ships.energy_share.to_numpy()
    angles = meta.incidence_deg.to_numpy(float)
    ids = np.unique(table[['a', 'b']].to_numpy())
    fit_ids = np.unique(fit_table[['a', 'b']].to_numpy())
    amin, amax = float(angles[fit_ids].min()), float(angles[fit_ids].max())
    fit_gap_max = float(fit_table.delta_angle.abs().max())
    a, b = table[['a', 'b']].to_numpy().T
    outside = (angles[a] < amin) | (angles[a] > amax) | (angles[b] < amin) | (angles[b] > amax)
    key = dict(cohort=cohort, model=model, seed=seed)
    row = dict(**key, paired_ships=len(ships), angle_energy=total,
        angle_energy_effective_ships=1 / float(positive @ positive) if total else np.nan,
        top5_ship_energy_share=float(positive[:5].sum()),
        median_absolute_gap=float(np.median(gap)), maximum_absolute_gap=float(gap.max()),
        endpoint_angle_min=float(angles[ids].min()), endpoint_angle_max=float(angles[ids].max()),
        fit_pair_angle_min=amin, fit_pair_angle_max=amax, fit_maximum_absolute_gap=fit_gap_max,
        outside_fit_pair_angle_range_weight=float(weights @ outside),
        above_fit_maximum_gap_weight=float(weights @ (gap > fit_gap_max)))
    return row, ships.assign(**key)


def load_source_bundle(path, metadata, source_ids):
    """Subset shared exports immediately; never load prior/kernel or query caches."""
    with np.load(path) as cache:
        positions = pd.Index(cache['obs_ids']).get_indexer(source_ids)
        if (positions < 0).any():
            raise ValueError('Source observation missing from input export')
        z, context = cache['z'][positions], cache['context'][positions]
        fitting = cache['fit_obs_ids'].copy()
    local_meta = metadata.iloc[source_ids].reset_index(drop=True)
    local_fit = pd.Index(source_ids).get_indexer(fitting)
    if (local_fit < 0).any():
        raise ValueError('Export fitting IDs leave the original source cohort')
    return z, context, local_meta, fitting, local_fit


def response_lopo(z, context, meta, fitting, held_pairs, source_ids, *, seed, out):
    """Refit only response coefficients, using full-base features and context."""
    tables = []
    ports = meta.iloc[held_pairs[:, 0]].port.to_numpy()
    for port_index, port in enumerate(sorted(set(ports))):
        training = fitting[meta.iloc[fitting].port.to_numpy() != port]
        train_pairs = matched_pairs(meta, training)
        evaluation = held_pairs[ports == port]
        models = {arm: fit_response(z, context, meta, training, arm, train_pairs) for arm in ARMS}
        result, table, actual, predictions = evaluate_models(z, context, meta, evaluation, models,
            fitting, cohort='heldraw_response_lopo', model='full', seed=seed)
        result['held_port'] = port
        result['response_fit_observations'] = len(training)
        result['response_fit_ships'] = meta.iloc[training].mmsi.nunique()
        result['response_fit_unordered_pairs'] = len(train_pairs)
        tables.append(result)
        payload = dict(pair_obs_ids=source_ids[table[['a', 'b']].to_numpy()], actual_delta=actual,
                       fitting_obs_ids=source_ids[training], fit_pairs=source_ids[train_pairs],
                       held_port=port, base_encoder_is_lopo=False, **predictions)
        for arm, fitted in models.items():
            payload[arm + '_coefficient'] = fitted['coefficient']
            payload[arm + '_ridge'] = fitted['ridge']
        np.savez_compressed(out / f'full_{seed}_response_lopo_port{port_index}.npz', **payload)
    return pd.concat(tables, ignore_index=True)


def legacy_reference(results, cohorts, *, out):
    legacy = BASE / 'incidence_factor_chain_20261011'
    for old_name, key in [('fit_pairs.csv', 'fit_pairs'), ('held_pairs.csv', 'held_pairs')]:
        previous = pd.read_csv(legacy / old_name)[['a', 'b']].to_numpy()
        one = sorted(map(tuple, np.sort(previous, axis=1)))
        two = sorted(map(tuple, np.sort(cohorts[key], axis=1)))
        if one != two:
            raise ValueError('Legacy pair cohort differs; do not claim matched-cohort comparison')
    old = pd.read_csv(legacy / 'representation_results.csv')
    old = old.loc[old.group.isin(['fit', 'held', 'held_gap2']),
                  ['space', 'seed', 'group', 'pairs', 'ships', 'explained_percent']].copy()
    old['weighting'] = 'ship_then_pair'
    old['orientation'] = 'chronological_only'
    old['comparison'] = 'same_unordered_cohort_different_representation_and_weighting'
    old.to_csv(out / 'legacy_cohort_reference.csv', index=False)
    full = results.loc[(results.model == 'full') & (results.scope == 'overall') &
                       results.subset.isin(['all', 'angle_gap2'])]
    full.to_csv(out / 'current_matched_cohort.csv', index=False)


def pool_lopo_predictions(out):
    """Reweight cached port-specific predictions jointly; no mean-of-port shortcut."""
    results = []
    for seed in (31415, 27182):
        table = pd.read_csv(out / f'full_{seed}_heldraw_pairs.csv')
        lookup = pd.MultiIndex.from_frame(table[['a', 'b']])
        actual = np.zeros((len(table), 32))
        predictions = {arm: np.zeros_like(actual) for arm in ARMS}
        filled = np.zeros(len(table), dtype=int)
        for path in sorted(out.glob(f'full_{seed}_response_lopo_port*.npz')):
            with np.load(path) as cache:
                rows = lookup.get_indexer(pd.MultiIndex.from_arrays(cache['pair_obs_ids'].T))
                if (rows < 0).any():
                    raise ValueError('LOPO cache pair outside held source cohort')
                np.add.at(filled, rows, 1)
                actual[rows] = cache['actual_delta']
                for arm in ARMS:
                    predictions[arm][rows] = cache[arm]
        if not (filled == 1).all():
            raise ValueError('Each held directed pair must have exactly one LOPO prediction')
        result = diagnose_predictions(table, actual, predictions,
            cohort='heldraw_response_lopo', model='full', seed=seed)
        result['evaluation_angles'] = 'real'
        result['base_encoder_is_lopo'] = False
        result['response_fit_excludes_held_port'] = True
        results.append(result)
    result = pd.concat(results, ignore_index=True)
    result.to_csv(out / 'response_lopo_pooled.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root, out = args.root, args.root / 'mechanism'
    with np.load(root / 'source_cohorts.npz') as cache:
        cohorts = {key: cache[key].copy() for key in cache.files}
    source_ids = np.union1d(cohorts['fit_obs_ids'], cohorts['held_obs_ids'])
    held_local = pd.Index(source_ids).get_indexer(cohorts['held_obs_ids'])
    metadata = pd.read_csv(BASE / 'observations.csv', usecols=[
        'obs_id', 'mmsi', 'class_id', 'port', 'product', 'incidence_deg',
        'pass_name', 'platform', 'acq_time'])
    out.mkdir(parents=True, exist_ok=True)
    results, transfers, coverages, ship_energy, sham_objectives = [], [], [], [], []
    for seed in (31415, 27182):
        for bundle in ('full', 'fold0', 'fold1', 'fold2'):
            z, context, meta, fit_global, fitting = load_source_bundle(
                root / f'{bundle}_{seed}_inputs.npz', metadata, source_ids)
            models = {}
            for arm in ARMS:
                with np.load(root / f'{bundle}_{seed}_{arm}.npz') as cache:
                    models[arm] = {key: cache[key].copy() for key in cache.files}
                np.testing.assert_array_equal(models[arm]['fitting'], fit_global)
            train_pairs = matched_pairs(meta, fitting)
            held_pairs = matched_pairs(meta, held_local)
            fit_table = directed_table(meta, train_pairs, fitting)
            for cohort, pairs in [('sourcefit', train_pairs), ('heldraw', held_pairs)]:
                result, table, actual, predictions = evaluate_models(z, context, meta, pairs,
                    models, fitting, cohort=cohort, model=bundle, seed=seed)
                results.append(result)
                coverage, ship = angle_coverage(meta, table, fit_table, cohort=cohort, model=bundle, seed=seed)
                coverages.append(coverage)
                ship_energy.append(ship)
                table.assign(a=source_ids[table.a], b=source_ids[table.b]).to_csv(
                    out / f'{bundle}_{seed}_{cohort}_pairs.csv', index=False)
                np.savez_compressed(out / f'{bundle}_{seed}_{cohort}_deltas.npz',
                    pair_obs_ids=source_ids[table[['a', 'b']].to_numpy()], actual_delta=actual, **predictions)
            a, b = fit_table[['a', 'b']].to_numpy().T
            sham_angles = shuffled_angles(meta, fitting)
            prediction = predict_delta(context[a], sham_angles[a], sham_angles[b], models['angle_sham'])
            sham_objectives.append(dict(model=bundle, seed=seed, arm='angle_sham',
                evaluation_angles='training_within_cell_shuffle',
                **pair_metrics(fit_table.assign(delta_angle=sham_angles[b] - sham_angles[a]),
                               z[b] - z[a], prediction)))
            if bundle == 'full':
                transfers.append(response_lopo(z, context, meta, fitting, held_pairs, source_ids,
                                                seed=seed, out=out))
            print(f'{bundle} seed={seed}: source pair diagnostics complete', flush=True)
    result = pd.concat(results, ignore_index=True)
    transfer = pd.concat(transfers, ignore_index=True)
    write_reports(result, transfer, out=out)
    pool_lopo_predictions(out)
    paired_arm_comparisons(transfer.assign(value=lambda t: t.held_port + ':' + t.value)).to_csv(
        out / 'response_lopo_arm_comparisons.csv', index=False)
    pd.DataFrame(coverages).to_csv(out / 'angle_energy_coverage.csv', index=False)
    pd.concat(ship_energy, ignore_index=True).to_csv(out / 'angle_energy_ship_contributions.csv', index=False)
    pd.DataFrame(sham_objectives).to_csv(out / 'sham_training_objective.csv', index=False)
    legacy_reference(result, cohorts, out=out)
    receipt = dict(source_fit_observations=len(cohorts['fit_obs_ids']),
        source_held_observations=len(cohorts['held_obs_ids']), base_bundles=8,
        response_lopo_fits=4 * transfer[['seed', 'held_port']].drop_duplicates().shape[0],
        neural_fits=0, target_query_evaluation=False, base_encoder_is_lopo=False,
        legacy_unordered_pair_cohorts_exact=True)
    (out / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

