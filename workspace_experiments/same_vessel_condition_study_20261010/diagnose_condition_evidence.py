"""Two frozen selected candidates: influence and response-only scene holdout."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from condition_evidence_response import ARMS, fit_model, predict_model, response_target
from incidence_transport import pair_weights

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'condition_evidence_20261011'
OUT = ROOT / 'mechanism'
CONTROLS = ('anchor', 'capacity', 'sham')
EXPECTED = {('center_border_log_ratio', 'magnitude'), ('middle_energy_share', 'signed')}


def product_exclusion(meta, fitting, pairs, products):
    retained = fitting[~meta.iloc[fitting]['product'].isin(products).to_numpy()]
    retained_pairs = pairs[np.isin(pairs, retained).all(1)]
    return retained, retained_pairs


def pair_errors(truth, predictions, n):
    return {arm: ((truth - pred) ** 2).reshape(2, n).mean(0)
            for arm, pred in predictions.items()}


def aggregate(meta, pairs, errors):
    if not len(pairs):
        return dict(pairs=0, ships=0, classes=0)
    weights = pair_weights(meta, pairs)
    result = dict(pairs=len(pairs), ships=int(meta.iloc[pairs[:, 0]].mmsi.nunique()),
                  classes=int(meta.iloc[pairs[:, 0]].class_id.nunique()))
    for arm, error in errors.items():
        result['mse_' + arm] = float(weights @ error)
    for arm in CONTROLS:
        result['gain_vs_' + arm] = 100 * (1 - result['mse_condition'] / max(result['mse_' + arm], 1e-12))
    return result


def accounting(meta, pairs, errors, identity):
    weights = pair_weights(meta, pairs)
    table = meta.iloc[pairs[:, 0]].reset_index(drop=True)
    rows = []
    for kind, column in [('class', 'class_id'), ('port', 'port'), ('ship', 'mmsi')]:
        for value, part in table.groupby(column):
            ids = part.index.to_numpy()
            local = aggregate(meta, pairs[ids], {k: v[ids] for k, v in errors.items()})
            row = dict(**identity, grouping=kind, group=str(value), global_weight=float(weights[ids].sum()), **local)
            for arm in CONTROLS:
                row['global_mse_reduction_vs_' + arm] = float(weights[ids] @ (errors[arm][ids] - errors['condition'][ids]))
            rows.append(row)
    return rows


def influence(meta, pairs, errors, identity):
    ships = meta.iloc[pairs[:, 0]].mmsi.to_numpy()
    full = aggregate(meta, pairs, errors)
    rows = []
    for ship in np.unique(ships):
        keep = ships != ship
        local = aggregate(meta, pairs[keep], {k: v[keep] for k, v in errors.items()})
        row = dict(**identity, removed_ship=str(ship), removed_pairs=int((~keep).sum()), **local)
        for arm in CONTROLS:
            row['gain_change_vs_' + arm] = local['gain_vs_' + arm] - full['gain_vs_' + arm]
        rows.append(row)
    return rows


def comparable_cells(meta, pairs, values, role):
    """Only within-port/class cells having both angle bins enter the comparison."""
    a, b = pairs.T
    angle = meta.incidence_deg.to_numpy()
    gap = np.abs(angle[b] - angle[a])
    table = meta.iloc[a][['port', 'class_id', 'mmsi']].reset_index(drop=True)
    table['outcome'] = np.abs(values[b] - values[a])
    table['gap_bin'] = np.where(gap < 2, 'near', np.where(gap >= 5, 'large', 'other'))
    rows = []
    for (port, cls), cell in table.groupby(['port', 'class_id']):
        near, large = cell[cell.gap_bin == 'near'], cell[cell.gap_bin == 'large']
        row = dict(role=role, port=port, class_id=int(cls), near_pairs=len(near), large_pairs=len(large),
            near_ships=int(near.mmsi.nunique()), large_ships=int(large.mmsi.nunique()),
            common_ships=len(set(near.mmsi) & set(large.mmsi)), comparable=bool(len(near) and len(large)))
        for name, part in [('near', near), ('large', large)]:
            row[name + '_mean'] = float(part.groupby('mmsi').outcome.mean().mean()) if len(part) else np.nan
        row['large_minus_near'] = row['large_mean'] - row['near_mean'] if row['comparable'] else np.nan
        # Same ships in both bins further limit identity composition, if available.
        common = sorted(set(near.mmsi) & set(large.mmsi))
        for name, part in [('near', near), ('large', large)]:
            row['common_ship_' + name + '_mean'] = float(part[part.mmsi.isin(common)].groupby('mmsi').outcome.mean().mean()) if common else np.nan
        rows.append(row)
    return rows


def main():
    start = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    if (OUT / 'COMPLETE.json').exists():
        raise ValueError('Preserve completed diagnostics')
    selected = pd.read_csv(ROOT / 'source_selection.csv').query('selected')
    assert set(zip(selected.name, selected['mode'])) == EXPECTED
    meta = pd.read_csv(BASE / 'observations.csv')
    with np.load(BASE / 'incidence_transport_20261011/source_cohorts.npz') as f:
        fitting, held, train_pairs, held_pairs = (f[k] for k in ('fit_obs_ids', 'held_obs_ids', 'fit_pairs', 'held_pairs'))
    with np.load(ROOT / 'evidence_features.npz') as f:
        features = np.full((len(meta), 6), np.nan)
        available = f['obs_ids'].copy()
        features[available] = f['features']
        names = f['feature_names'].tolist()
    context = np.full(len(meta), np.nan)
    context[available] = np.logaddexp.reduce(features[available, :3], axis=1)
    angle = meta.incidence_deg.to_numpy()
    products = meta['product'].to_numpy()
    pair_keys = [tuple(sorted(products[p].tolist())) for p in held_pairs]
    unique_keys = sorted(set(pair_keys))
    partitions = []
    coverage = []
    for fold, key in enumerate(unique_keys):
        indices = np.array([i for i, candidate in enumerate(pair_keys) if candidate == key])
        fit, pairs = product_exclusion(meta, fitting, train_pairs, key)
        coverage.append(dict(fold=fold, product_a=key[0], product_b=key[1],
            held_pairs=len(indices), held_ships=int(meta.iloc[held_pairs[indices, 0]].mmsi.nunique()),
            fit_observations=len(fit), fit_pairs=len(pairs), fit_pair_ships=int(meta.iloc[pairs[:, 0]].mmsi.nunique()),
            fit_pair_classes=int(meta.iloc[pairs[:, 0]].class_id.nunique()),
            removed_observations=len(fitting) - len(fit), removed_pairs=len(train_pairs) - len(pairs)))
        partitions.append((indices, fit, pairs))
    pd.DataFrame(coverage).to_csv(OUT / 'product_pair_coverage.csv', index=False)
    subsets = dict(all=np.ones(len(held_pairs), bool), near=np.abs(angle[held_pairs[:, 1]] - angle[held_pairs[:, 0]]) < 2,
                   gap2=np.abs(angle[held_pairs[:, 1]] - angle[held_pairs[:, 0]]) >= 2)
    source_products = set(products[fitting])
    subsets['both_new_scene'] = np.array([all(p not in source_products for p in key) for key in pair_keys])
    group_rows, influence_rows, metrics_rows, cell_rows, fold_rows = [], [], [], [], []
    fits = 0
    for row in selected.itertuples():
        identity = dict(name=row.name, mode=row.mode)
        values = features[:, names.index(row.name)]
        directed = np.r_[held_pairs, held_pairs[:, ::-1]]
        truth = response_target(values, directed, row.mode)
        with np.load(ROOT / 'predictions' / f'0_{row.name}_{row.mode}.npz') as f:
            np.testing.assert_array_equal(f['directed'], directed)
            np.testing.assert_allclose(f['truth'], truth, rtol=0, atol=0)
            original = {arm: f[arm].copy() for arm in ARMS}
        errors = pair_errors(truth, original, len(held_pairs))
        group_rows.extend(accounting(meta, held_pairs, errors, identity))
        influence_rows.extend(influence(meta, held_pairs, errors, identity))
        predictions = {arm: np.full(len(truth), np.nan) for arm in ARMS}
        for fold, (indices, fit, pairs) in enumerate(partitions):
            if not len(pairs):
                continue
            local_pairs = held_pairs[indices]
            local_directed = np.r_[local_pairs, local_pairs[:, ::-1]]
            a, b = local_directed.T
            positions = np.r_[indices, indices + len(held_pairs)]
            local_predictions = {}
            for arm in ARMS:
                model = fit_model(meta, fit, pairs, values, context, values, row.mode, arm)
                pred = predict_model(values[a], context[a], angle[a], angle[b], model)
                predictions[arm][positions] = pred
                local_predictions[arm] = pred
                fits += 1
            local_errors = pair_errors(truth[positions], local_predictions, len(indices))
            fold_rows.append(dict(**identity, fold=fold, **aggregate(meta, local_pairs, local_errors)))
        valid = np.isfinite(predictions['condition']).reshape(2, len(held_pairs)).all(0)
        new_errors = pair_errors(truth, predictions, len(held_pairs))
        for path_name, local_errors in [('original', errors), ('response_product_pair_holdout', new_errors)]:
            for subset, mask in subsets.items():
                keep = mask & valid
                metrics_rows.append(dict(**identity, path=path_name, subset=subset,
                    **aggregate(meta, held_pairs[keep], {k: v[keep] for k, v in local_errors.items()})))
        np.savez_compressed(OUT / f'{row.name}_{row.mode}_scene_holdout.npz', directed=directed,
            truth=truth, valid_pairs=valid, **predictions)
        if row.name == 'center_border_log_ratio':
            cell_rows.extend(comparable_cells(meta, train_pairs, values, 'fit'))
            cell_rows.extend(comparable_cells(meta, held_pairs, values, 'held'))
        print('DIAGNOSED', row.name, row.mode, 'response_fits', fits, flush=True)
    for name, records in [('gain_accounting', group_rows), ('leave_one_ship_influence', influence_rows),
                           ('scene_holdout_summary', metrics_rows), ('scene_holdout_folds', fold_rows),
                           ('center_large_near_cells', cell_rows)]:
        pd.DataFrame(records).to_csv(OUT / (name + '.csv'), index=False)
    receipt = dict(status='complete', seconds=time.perf_counter() - start, response_fits=fits,
        candidates=sorted(EXPECTED), product_pair_folds=len(unique_keys),
        protocol='Only the two already source-selected candidates. Remove both held-pair products from original fitting observations and pairs; refit same4arms, original ridge/design/sham recipe. No tuning or classification deployment.',
        aggregation='Two directions averaged per unordered pair; original equal-class/equal-ship/equal-pair weights. Scene-heldout predictions reassembled then globally weighted, not mean fold performance. Class/port/ship global contributions sum to original global MSE reduction. Ship influence reweights after removing that ship, without refitting.',
        comparison='Center large>=5 vs near<2 only within port/class cells with both bins; equal ship means within each cell/bin. All coverage retained including ineligible cells. Common-ship means also provided; no unadjusted pooled causal interpretation.',
        exposure='Response-layer product holdout only. Frozen historical backbone and source-defined candidate selection are not product-heldout; both-new-scene evidence limited. Repeatedly used held development data, not independent confirmation.',
        review='9 focused response tests passed before main study; no B evidence/true-class input or held-based candidate selection found. Deterioration angle design symmetric under swapping angles at fixed anchor; not relevant to selected magnitude/signed candidates. Source CV upstream is nonnested, explicitly documented.')
    (OUT / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='complete', response_fits=fits, seconds=receipt['seconds'])))


if __name__ == '__main__':
    main()
