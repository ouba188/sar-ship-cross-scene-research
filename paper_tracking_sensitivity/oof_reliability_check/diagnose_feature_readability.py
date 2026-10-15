"""Read-only information probes, explicitly separate from the 64-label method."""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from budgeted_residual_handoff import score_policy
from run_opensarship2_transfer import SOURCE, WS, write_json

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
SEED = 20261008


def geometry_features(frame, source=False):
    columns = (['target_long_side_vv', 'crop_side_vv', 'target_occupancy_vv',
                'width_vv', 'height_vv'] if source else
               ['det_len_px', 'requested_side', 'det_conf', 'vv_zero_fraction', 'vh_zero_fraction'])
    values = frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    valid = np.isfinite(values)
    values = np.where(valid, values, 0.)
    values[:, :2] = np.log1p(np.maximum(values[:, :2], 0.))
    if source:
        values[:, 3:] = np.log1p(np.maximum(values[:, 3:], 0.))
    return np.column_stack((values, (~valid).astype(np.float32)))


def load_dataset(name):
    if name.startswith('ports24'):
        frame = pd.read_csv(SOURCE/'manifest.csv', dtype={'mmsi': str}, keep_default_na=False)
        base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
        with np.load(WS/'b2_ssl_feats_38091.npz') as saved:
            extra = saved['F'].astype(np.float32)
        return frame, base, extra, frame.mmsi.to_numpy(str), None, geometry_features(frame, True), None
    folder = ROOT/({'f0921': 'f0921_detector_recrop_20261007',
                   'oss1': 'opensarship1_transfer_20261007',
                   'oss2': 'opensarship2_transfer_20261007'}[name])
    frame = pd.read_csv(folder/'target_manifest.csv', dtype=str, keep_default_na=False)
    group_key = 'mmsi' if name == 'f0921' else 'mmsi_or_ship_id'
    if name == 'f0921':
        base = np.load(folder/'paired_r50.npy', mmap_mode='r')
        extra = np.load(folder/'vh_ssl.npy', mmap_mode='r')
        np.testing.assert_array_equal(np.load(folder/'feature_rows.npy'), frame.row.to_numpy(int))
        crop = pd.read_csv(folder/'crop_metadata.csv').set_index('row').loc[frame.row.to_numpy(int)]
        geom = geometry_features(crop)
        length = pd.to_numeric(frame.len_m, errors='coerce').to_numpy(dtype=np.float32)
        model_path = ROOT/'f0921_transfer_20261007/source_model.pkl'
    else:
        with np.load(folder/'features.npz') as saved:
            np.testing.assert_array_equal(saved['sample_ids'], frame.sample_id.to_numpy(str))
            base, extra = saved['paired_r50'].astype(np.float32), saved['vh_ssl'].astype(np.float32)
        geom = None
        length = None
        model_path = folder/'source_model.pkl'
    with model_path.open('rb') as stream:
        model = pickle.load(stream)
    return frame, base, extra, frame[group_key].to_numpy(str), model, geom, length


def group_split(name, frame, groups):
    missing = (groups == '') | (groups == 'nan')
    groups = groups.astype(object)
    for row in np.flatnonzero(missing):
        groups[row] = f'unknown-row-{row}'
    split_groups = frame.port.to_numpy(str) if name == 'ports24_leaveport' else groups
    train, query = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED).split(
        np.arange(len(frame)), groups=split_groups))
    if name == 'ports24_leaveport':
        train = train[~np.isin(groups[train], groups[query])]
        products = frame['product'].to_numpy(str)
        held_products = np.unique(products[query])
        train = train[~np.isin(products[train], held_products[held_products != ''])]
    if np.intersect1d(groups[train], groups[query]).size:
        raise ValueError('vessel overlap in diagnostic split')
    return train, query, groups, int(missing.sum())


def probe(train_x, query_x, labels, query_y):
    mean, scale = train_x.mean(0), train_x.std(0)+1e-6
    d = train_x.shape[1]
    phi = np.column_stack(((train_x-mean)/scale/np.sqrt(d), np.ones(len(train_x), dtype=np.float32)))
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = (len(labels)/(len(counts)*counts[inverse])).astype(np.float32)
    desired = 2*np.eye(8, dtype=np.float32)[labels]-1
    model = Ridge(alpha=8., fit_intercept=False).fit(phi, desired, sample_weight=weights)
    predictions = np.empty(len(query_y), dtype=int)
    for lo in range(0, len(query_y), 2048):
        hi = min(lo+2048, len(query_y))
        x = query_x[lo:hi]
        qp = np.column_stack(((x-mean)/scale/np.sqrt(d), np.ones(len(x), dtype=np.float32)))
        predictions[lo:hi] = model.predict(qp).argmax(1)
    return predictions


def score(prediction, truth):
    return score_policy(prediction, np.full(len(truth), -1), truth, np.ones(len(truth), dtype=bool))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists():
        raise ValueError('completed diagnostic exists')
    datasets = ['ports24_within', 'ports24_leaveport', 'f0921', 'oss1', 'oss2']
    write_json(out/'protocol.json', dict(date='2026-10-08', datasets=datasets,
        purpose='Locate readable class information, NOT a new method result or theoretical accuracy upper bound',
        labels='64/512 shared nested diagnostic training rows;4096 additionally on two24port tasks andF0921',
        label_scope='Uses existing target training labels beyond64 in diagnostic arms; never report as64label performance',
        split='75/25vessel groups;24port leaveport uses25%port groups and purges query vessel/product from training',
        seed=SEED, repetitions=1, classifier='balanced +/-1 ridge8, penalized bias, train-only centering/scaling, /sqrt(d)',
        views=['r50_pca128', 'r50_raw2048', 'joint_pca192', 'joint_raw2560'],
        extra_diagnostics='24port/F0921 rawR50+image geometry;F0921 rawR50+privilegedAISlength',
        pca='24port fits PCA on diagnostic train-pool images only; external datasets retain existing source PCA',
        selection='fixed recipe, no tuning/checkpoint selection, no production model changes',
        caveat='Historical encoders/data. Same-vessel isolation does not imply scene isolation in within-dataset probes.'))
    results, populations = [], {}
    started = time.monotonic()
    with threadpool_limits(limits=4):
        for name in datasets:
            folder = out/name
            folder.mkdir(exist_ok=True)
            frame, base, extra, groups, source_model, geom, length = load_dataset(name)
            labels = frame.class_id.to_numpy(dtype=int)
            train_pool, query, groups, missing = group_split(name, frame, groups)
            order = np.random.default_rng(SEED).permutation(train_pool)
            budgets = [64, 512]+([4096] if name.startswith('ports24') or name == 'f0921' else [])
            if max(budgets) > len(order):
                raise ValueError(f'{name} has insufficient training rows for fixed budgets')
            fit = order[:max(budgets)]
            np.savez_compressed(folder/'split.npz', train_pool=train_pool, ordered_fit_rows=fit, query_rows=query)
            if source_model is None:
                pb = PCA(n_components=128, random_state=0).fit(base[train_pool])
                pe = PCA(n_components=64, random_state=0).fit(extra[train_pool])
            else:
                pb, pe = source_model['pca'], source_model['pca_ssl']
            bx, bq = np.asarray(base[fit], dtype=np.float32), np.asarray(base[query], dtype=np.float32)
            ex, eq = np.asarray(extra[fit], dtype=np.float32), np.asarray(extra[query], dtype=np.float32)
            px, pq = pb.transform(bx), pb.transform(bq)
            zx, zq = pe.transform(ex), pe.transform(eq)
            views = dict(r50_pca128=(px, pq), r50_raw2048=(bx, bq),
                joint_pca192=(np.column_stack((px, zx)), np.column_stack((pq, zq))),
                joint_raw2560=(np.column_stack((bx, ex)), np.column_stack((bq, eq))))
            if geom is not None:
                views['r50_plus_image_geometry'] = (np.column_stack((bx, geom[fit])), np.column_stack((bq, geom[query])))
            if length is not None:
                valid = np.isfinite(length) & (length > 0)
                length_features = np.column_stack((np.log1p(np.where(valid, length, 0.)), (~valid).astype(float))).astype(np.float32)
                views['r50_plus_AIS_length_PRIVILEGED'] = (np.column_stack((bx, length_features[fit])),
                    np.column_stack((bq, length_features[query])))
            sampled = query[:min(8192, len(query))]
            raw_sample = np.asarray(base[sampled], dtype=np.float32)-pb.mean_
            projected = raw_sample@pb.components_.T
            ratio = float(np.square(projected).sum()/np.square(raw_sample).sum())
            population = dict(total_rows=len(frame), train_pool=len(train_pool), query_rows=len(query),
                train_vessels=len(np.unique(groups[train_pool])), query_vessels=len(np.unique(groups[query])),
                missing_identity_rows=missing, query_class_counts=np.bincount(labels[query], minlength=8).tolist(),
                fit_class_counts={str(b): np.bincount(labels[fit[:b]], minlength=8).tolist() for b in budgets},
                sampled_source_pca_energy_fraction=ratio,
                held_ports=sorted(frame.iloc[query].port.unique().tolist()) if name == 'ports24_leaveport' else None)
            populations[name] = population
            write_json(folder/'population.json', population)
            predictions = {}
            for view, (train_x, query_x) in views.items():
                for budget in budgets:
                    # Metadata supplements address information missing from the
                    # image embedding, using only the larger diagnostic budget.
                    if 'plus_' in view and budget != max(budgets):
                        continue
                    pred = probe(train_x[:budget], query_x, labels[fit[:budget]], labels[query])
                    measured = score(pred, labels[query])
                    row = dict(dataset=name, view=view, labels=budget, **measured,
                        support_class_counts=np.bincount(labels[fit[:budget]], minlength=8).tolist())
                    results.append(row)
                    predictions[f'{view}_{budget}'] = pred
                    print(f'PROBE {name} {view} labels={budget} OA={100*measured["acc"]:.3f} BA={100*measured["ba"]:.3f}', flush=True)
            np.savez_compressed(folder/'predictions.npz', query_rows=query, truth=labels[query], **predictions)
            write_json(folder/'results.json', [r for r in results if r['dataset'] == name])
            pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in results]).to_csv(out/'results.csv', index=False)
            del base, extra, bx, bq, ex, eq, views
    write_json(out/'summary.json', dict(results=results, populations=populations,
        elapsed_seconds=time.monotonic()-started, diagnostic_only=True, independent_confirmation=False))
    print('FEATURE DIAGNOSIS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
