"""Identity-held diagnostics of real repeat observations; not a new main model."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = Path('E:/Hermes/paper_tracking')
SOURCE = Path('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
SEED = 20261010
GROUPS = ('SP_near', 'SP_far', 'DP_near', 'DP_far')
MODES = ('r50', 'r18', 'dual')
BATCH = 512


def write(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def angular_distance(a, b):
    return np.abs((np.asarray(a)-np.asarray(b)+90.) % 180.-90.)


def assign_group(same_port, same_platform, same_pass, angle_difference, known):
    near = known & same_platform & same_pass & (angle_difference <= 2.)
    far = known & (~same_platform | ~same_pass | (angle_difference >= 5.))
    return np.select([same_port & near, same_port & far, ~same_port & near,
                      ~same_port & far], GROUPS, default='unmatched')


def load_observations():
    cols = ['instance_id', 'mmsi', 'port', 'product', 'class_id', 'class_name',
            'image_relpath_vv', 'image_relpath_vh']
    f = pd.read_csv(SOURCE/'manifest.csv', usecols=cols,
                    dtype={'mmsi': str}, keep_default_na=False)
    f['manifest_row'] = np.arange(len(f))
    keys = ['instance_id', 'product', 'port']
    side = pd.read_csv(ROOT/'pixel_correction_review/chip_incidence_v1/incidence_by_instance.csv',
                       usecols=keys+['status', 'incidence_deg'])
    relative = pd.read_csv(ROOT/'sensitivity/relative_observation_geometry_20261009/relative_conditions.csv',
                           usecols=keys+['relative_status', 'relative_signed_deg'])
    f = f.merge(side, on=keys, how='left', validate='one_to_one').merge(
        relative, on=keys, how='left', validate='one_to_one')
    f.loc[f.status.ne('ok'), 'incidence_deg'] = np.nan
    cache = json.loads((ROOT/'s1_product_incidence.json').read_text(encoding='utf-8'))
    f['pass_name'] = f['product'].map(lambda p: cache.get(p, {}).get('pass'))
    f['platform'] = f['product'].str[:3]
    f['acq_time'] = pd.to_datetime(f['product'].str.extract(
        r'_([0-9]{8}T[0-9]{6})_', expand=False), format='%Y%m%dT%H%M%S', errors='coerce')
    valid = f.mmsi.str.fullmatch(r'[0-9]{9}') & f.mmsi.ne('000000000')
    conflicting = f.loc[valid].groupby('mmsi').class_id.nunique()
    bad = conflicting[conflicting > 1].index
    obs = f.loc[valid & ~f.mmsi.isin(bad) & f.acq_time.notna()].copy()
    # The current manifest has no repeated instance within an identity/port/time.
    if obs.duplicated(['mmsi', 'port', 'acq_time']).any():
        raise ValueError('Repeated acquisition needs aggregation before pairing')
    obs = obs.sort_values(['mmsi', 'acq_time', 'port']).reset_index(drop=True)
    obs['obs_id'] = np.arange(len(obs))
    identity = obs[['mmsi', 'class_id']].drop_duplicates('mmsi')
    rng = np.random.default_rng(SEED)
    fold_map = {}
    for _, part in identity.groupby('class_id', sort=True):
        ships = rng.permutation(np.sort(part.mmsi.to_numpy(str)))
        fold_map.update({ship: i % 3 for i, ship in enumerate(ships)})
    obs['fold'] = obs.mmsi.map(fold_map)
    obs.to_csv(OUT/'observations.csv', index=False)
    write('INVENTORY.json', dict(manifest_n=len(f), valid_observations=len(obs),
        distinct_mmsi=int(obs.mmsi.nunique()), conflicting_identities=int(len(bad)),
        cross_port_mmsi=int(obs.groupby('mmsi').port.nunique().gt(1).sum()),
        relative_direction_observations=int(obs.relative_signed_deg.notna().sum())))
    return obs


def build_pairs(obs):
    parts = []
    for ship, g in obs.groupby('mmsi', sort=False):
        if len(g) < 2:
            continue
        i, j = np.triu_indices(len(g), 1)
        row = g.obs_id.to_numpy()
        times = g.acq_time.to_numpy()
        gap = (times[j]-times[i])/np.timedelta64(1, 'D')
        use = gap > 0
        i, j, gap = i[use], j[use], gap[use]
        angle = g.incidence_deg.to_numpy(float)
        delta = np.abs(angle[i]-angle[j])
        passes = g.pass_name.fillna('').to_numpy(str)
        platform, ports = g.platform.to_numpy(str), g.port.to_numpy(str)
        sp, sat, orbit = ports[i] == ports[j], platform[i] == platform[j], passes[i] == passes[j]
        known = np.isfinite(delta) & (passes[i] != '') & (passes[j] != '')
        direction = g.relative_signed_deg.to_numpy(float)
        ddir = angular_distance(direction[i], direction[j])
        parts.append(pd.DataFrame(dict(a=row[i], b=row[j], mmsi=ship,
            class_id=int(g.class_id.iloc[0]), fold=int(g.fold.iloc[0]),
            gap_days=gap, port_a=ports[i], port_b=ports[j], same_port=sp,
            same_platform=sat, same_pass=orbit, delta_incidence=delta,
            delta_relative_direction=ddir, group=assign_group(sp, sat, orbit, delta, known),
            direction_near=np.isfinite(ddir) & (ddir <= 10.),
            angle_only_far=known & sat & orbit & (delta >= 5.))))
    pairs = pd.concat(parts, ignore_index=True)
    pairs['pair_id'] = np.arange(len(pairs))
    pairs['gap_bin'] = pd.cut(pairs.gap_days, [0, 7, 30, 90, np.inf],
                             labels=['0-7d', '7-30d', '30-90d', '>90d']).astype(str)
    assert (obs.fold.to_numpy()[pairs.a] == obs.fold.to_numpy()[pairs.b]).all()
    pairs.to_csv(OUT/'pairs.csv', index=False)
    counts = pairs.groupby('group').agg(pairs=('pair_id', 'size'),
        ships=('mmsi', 'nunique'), classes=('class_id', 'nunique'),
        median_gap_days=('gap_days', 'median'), direction_known=('delta_relative_direction', 'count'),
        direction_near=('direction_near', 'sum'))
    counts.to_csv(OUT/'coverage.csv')
    print(counts.to_string(), flush=True)
    return pairs


def standardized_features(raw, rows, train, rank):
    source = np.asarray(raw[rows[train]], dtype=np.float32)
    center = source.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(source.std(0, dtype=np.float64).astype(np.float32), 1e-6)
    source -= center
    source /= scale
    pca = PCA(rank, svd_solver='randomized', iterated_power=3, random_state=SEED)
    pca.fit(source)
    del source
    low = np.empty((len(rows), rank), np.float32)
    for lo in range(0, len(rows), BATCH):
        sl = slice(lo, min(lo+BATCH, len(rows)))
        block = (np.asarray(raw[rows[sl]], np.float32)-center)/scale
        low[sl] = pca.transform(block)/np.sqrt(rank)
    return low, center, scale, pca


def identity_class_weights(obs):
    ship_counts = obs.groupby('mmsi').mmsi.transform('size').to_numpy()
    class_ships = obs.groupby('class_id').mmsi.transform('nunique').to_numpy()
    return len(obs)/(obs.class_id.nunique()*ship_counts*class_ships)


def reference_rows(obs, train, fold):
    rng = np.random.default_rng(SEED+fold+1)
    selected = []
    for _, part in obs.iloc[train].groupby('class_id', sort=True):
        ships = rng.choice(np.sort(part.mmsi.unique()), 8, replace=False)
        for ship in ships:
            selected.append(int(rng.choice(part.loc[part.mmsi.eq(ship), 'obs_id'])))
    return np.array(selected)


def raw_pair_distance(raw, rows, pairs, center, scale):
    distances, cosine = np.empty(len(pairs)), np.empty(len(pairs))
    for lo in range(0, len(pairs), BATCH):
        hi = min(lo+BATCH, len(pairs))
        a = (np.asarray(raw[rows[pairs.a.to_numpy()[lo:hi]]], np.float32)-center)/scale
        b = (np.asarray(raw[rows[pairs.b.to_numpy()[lo:hi]]], np.float32)-center)/scale
        distances[lo:hi] = np.mean((a-b)**2, axis=1)
        norm = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
        cosine[lo:hi] = 1-np.sum(a*b, axis=1)/np.maximum(norm, 1e-12)
    return distances, cosine


def run_probes(obs, pairs):
    rows = obs.manifest_row.to_numpy()
    raw = dict(r50=np.load(SOURCE/'pooled_features.npy', mmap_mode='r'),
        r18=np.load(ROOT/'sensitivity/neural_complement_pilot_20261007/frozen_replay/source_extra.npy', mmap_mode='r'))
    if any(len(v) != 38091 for v in raw.values()):
        raise ValueError('Feature cache and manifest row contracts differ')
    scores = np.full((len(obs), 3, 8), np.nan)
    relations = np.full((len(obs), 2, 8), np.nan)
    measurements = pairs[['pair_id']].copy()
    for mode in ('r50', 'r18'):
        for field in ('raw_mse', 'cosdist', 'support_mse'):
            measurements[mode+'_'+field] = np.nan
    receipts = []
    for fold in range(3):
        started = time.perf_counter()
        train = np.flatnonzero(obs.fold.to_numpy() != fold)
        test = np.flatnonzero(obs.fold.to_numpy() == fold)
        assert not set(obs.iloc[train].mmsi) & set(obs.iloc[test].mmsi)
        assert np.array_equal(np.unique(obs.iloc[train].class_id), np.arange(8))
        ref = reference_rows(obs, train, fold)
        target_pairs = pairs.loc[pairs.fold.eq(fold)]
        low, exports = {}, dict(train_rows=rows[train], test_rows=rows[test],
                                reference_rows=rows[ref])
        for mode, rank in [('r50', 128), ('r18', 64)]:
            print(f'FOLD {fold} {mode}: source-only PCA; n={len(train)}', flush=True)
            x, center, scale, pca = standardized_features(raw[mode], rows, train, rank)
            low[mode] = x
            exports.update({mode+'_center': center, mode+'_scale': scale,
                            mode+'_pca_mean': pca.mean_, mode+'_components': pca.components_})
            mse, cosine = raw_pair_distance(raw[mode], rows, target_pairs, center, scale)
            measurements.loc[target_pairs.index, mode+'_raw_mse'] = mse
            measurements.loc[target_pairs.index, mode+'_cosdist'] = cosine
            support = x @ x[ref].T
            mi = ('r50', 'r18').index(mode)
            for c in range(8):
                relations[test, mi, c] = support[test][:, obs.class_id.to_numpy()[ref] == c].mean(1)
            delta = support[target_pairs.b.to_numpy()]-support[target_pairs.a.to_numpy()]
            measurements.loc[target_pairs.index, mode+'_support_mse'] = np.mean(delta**2, axis=1)
            del support, delta
        low['dual'] = np.concatenate([low['r50'], low['r18']], axis=1)/np.sqrt(2.)
        weights = identity_class_weights(obs.iloc[train])
        for mi, mode in enumerate(MODES):
            model = RidgeClassifier(alpha=1., solver='cholesky')
            model.fit(low[mode][train], obs.class_id.to_numpy()[train], sample_weight=weights)
            scores[test, mi] = model.decision_function(low[mode][test])
            exports[mode+'_head_coef'], exports[mode+'_head_intercept'] = model.coef_, model.intercept_
        np.savez_compressed(OUT/f'fold_{fold}_models.npz', **exports)
        receipts.append(dict(fold=fold, train_rows=len(train), test_rows=len(test),
            train_ships=int(obs.iloc[train].mmsi.nunique()), test_ships=int(obs.iloc[test].mmsi.nunique()),
            pairs=len(target_pairs), seconds=time.perf_counter()-started,
            no_train_test_identity_overlap=True, reference_identities_held_from_test=True))
        print(f'FOLD {fold} frozen in {receipts[-1]["seconds"]:.1f}s', flush=True)
        del low, exports
        gc.collect()
    if not np.isfinite(scores).all() or not np.isfinite(relations).all():
        raise ValueError('Missing/nonfinite held-identity predictions')
    np.savez_compressed(OUT/'FROZEN_PREDICTIONS.npz', scores=scores, relations=relations,
                        manifest_rows=rows, modes=np.array(MODES), folds=obs.fold.to_numpy())
    measurements.to_csv(OUT/'pair_representation.csv', index=False)
    write('PREDICTIONS_SAVED.json', dict(folds=receipts, observations=len(obs), pairs=len(pairs),
        query_metrics_computed=False, frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))


def protocol():
    write('PROTOCOL.json', dict(date='2026-10-10', seed=SEED, purpose='Mechanism diagnostic, not replacement of the main model',
        groups=GROUPS, near='Same satellite and pass; absolute incidence gap<=2 degrees; relative direction not required',
        far='Known satellite/pass/incidence; platform or pass differs, or incidence gap>=5 degrees',
        direction_subset='Additional available axial relative-direction gap<=10 degrees; no missing-value imputation',
        identity='Valid consistent-label MMSI proxy; distinct acquisition times; no physical-identity guarantee',
        pairing='All chronological within-MMSI pairs; group results average within ship then equally across ships within class, then classes',
        splits='Three class-stratified MMSI folds, seed fixed; each pair uses one identical fold model and reference set',
        representation='Existing global frozen paired ImageNet R50 and pre-episode SAR SSL R18 caches; no new image/network training',
        pca='Source-only feature center/scale then randomized PCA128/64, divide each reduced block by sqrt(width)',
        probe='R50, R18, or equal-weight feature concatenation; ridge1 with unpenalized intercept; equal class/identity training weights',
        reference='Fixed64 source-identity reference observations,8 different ships per class,shared by both feature routes; used only for relationship diagnostics',
        budget='Probe heads use all fold-training labels, NOT the main model64-target-label evaluation protocol',
        heldout='No held-MMSI labels/features in fitted standardization,PCA,readout or reference selection; labels used for fold stratification and post-freeze analysis',
        exposure='Frozen neural pretraining historical exposure not removed; no independent external confirmation',
        causal='Observational associations; background,sea state,load,direction and time gaps not jointly controlled',
        primary_readout='Equal-class/equal-ship paired change, errors, and dual rescue minus harm; consistency alone is not benefit',
        comparisons='Raw groups plus common class/time-gap cells (>=5 ships/group) and common-MMSI contrasts; none identifies causality',
        resources='CPU2; no GPU, no shared source/model edits, no subagents',
        evaluation='All three folds saved before classification/evidence query metrics; no outcome-based parameter selection'))


def main():
    started = time.perf_counter()
    protocol()
    with threadpool_limits(limits=2):
        obs = load_observations()
        pairs = build_pairs(obs)
        run_probes(obs, pairs)
        from report import report
        report(obs, pairs)
    write('COMPLETE.json', dict(seconds=time.perf_counter()-started, folds=3,
        observations=len(obs), pairs=len(pairs), main_model_changed=False))


if __name__ == '__main__':
    main()
