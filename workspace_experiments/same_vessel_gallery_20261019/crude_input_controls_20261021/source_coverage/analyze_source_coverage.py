"""Metadata-only coverage audit. No classifiers, feature arrays, or image reads."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
PILOT = ROOT / 'conditional_evidence_pilot'
LOCAL = ROOT / 'local_condition_module'
PAIRED = ROOT / 'paired_relation_probe_20261020'
FIELDS = ['relative_axis_deg', 'incidence', 'length_px', 'width_px']
# Declared without looking at test correctness; these are descriptive tolerances.
ANGLE = 15.0
INCIDENCE = 3.0
LOG_RATIO = np.log(1.25)


def save(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / (name + '.csv'), index=False)
    return frame


def pilot_functions():
    # Execute only the original split/weight functions, never training imports/code.
    tree = ast.parse((PILOT / 'run_pilot.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if (
        isinstance(n, ast.FunctionDef) and n.name in ('indices', 'weights')) or (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SEED' for t in n.targets))]
    namespace = dict(np=np, pd=pd)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PILOT / 'run_pilot.py'), 'exec'), namespace)
    return namespace['indices'], namespace['weights']


def metadata(f):
    snapshot = json.loads((LOCAL / 'view_snapshot.json').read_text(encoding='utf-8'))
    assert [v['obs_id'] for v in snapshot] == f.obs_id.tolist()
    records = []
    for view in snapshot:
        for pol in ('vv', 'vh'):
            image = view['images'][pol]
            p = image['priors']; axis = p.get('ship_axis')
            radar = p.get('radar_axis'); inc = p.get('radar_incidence')
            valid = bool(axis) and p.get('radar_kind') == 'orbit_los' and radar is not None and np.isfinite(radar)
            valid = valid and inc is not None and np.isfinite(inc)
            angle = (np.rad2deg(axis['angle'] - radar) + 90) % 180 - 90 if valid else np.nan
            records.append(dict(obs_id=view['obs_id'], pol=pol, geometry_valid=valid,
                relative_axis_deg=angle, acute_axis_deg=abs(angle), incidence=inc,
                length_px=axis['length_px'] if axis else np.nan,
                width_px=axis['width_px'] if axis else np.nan,
                image_width=image['w'], image_height=image['h'],
                match_status=p.get('match_status'), radar_kind=p.get('radar_kind')))
    geo = pd.DataFrame(records)
    # Detect stale or incorrectly keyed geometry before any coverage comparison.
    vv = geo[geo.pol.eq('vv')].reset_index(drop=True)
    np.testing.assert_allclose(np.cos(np.deg2rad(vv.relative_axis_deg * 2)), f.cos2, atol=1e-9)
    np.testing.assert_allclose(np.sin(np.deg2rad(vv.relative_axis_deg * 2)), f.sin2, atol=1e-9)
    np.testing.assert_allclose(vv.incidence, f.incidence, atol=1e-9)
    return geo


def distance(a, b):
    x = a[FIELDS].to_numpy(float); y = b[FIELDS].to_numpy(float)
    angle = np.abs((x[:, None, 0] - y[None, :, 0] + 90) % 180 - 90) / ANGLE
    inc = np.abs(x[:, None, 1] - y[None, :, 1]) / INCIDENCE
    length = np.abs(np.log(x[:, None, 2] / y[None, :, 2])) / LOG_RATIO
    width = np.abs(np.log(x[:, None, 3] / y[None, :, 3])) / LOG_RATIO
    return np.maximum.reduce([angle, inc, length, width])


def wquantile(x, w, q):
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = np.asarray(x)[good], np.asarray(w)[good]
    if not len(x):
        return np.nan
    order = np.argsort(x); x, w = x[order], w[order]
    return float(x[np.searchsorted(np.cumsum(w) / w.sum(), q, side='left')])


def main():
    f = pd.read_csv(PILOT / 'cohort.csv', dtype={'mmsi': str})
    f.index.name = 'cohort_row'
    assert f.obs_id.is_unique
    excluded = pd.read_csv(LOCAL / 'EXCLUSIONS.csv').obs_id.to_numpy()
    indices, weights = pilot_functions()
    splits = [tuple(ii[~f.iloc[ii].obs_id.isin(excluded).to_numpy()] for ii in indices(f, fold)) for fold in range(3)]
    membership = []; checks = []; class_counts = []; partitions = []; category_counts = []
    for fold, split in enumerate(splits):
        for part, ii in zip(('train', 'val', 'test'), split):
            g = f.iloc[ii].reset_index().copy()
            g['split_fold'] = fold; g['partition'] = part
            g['source_weight'] = weights(f, ii) if part == 'train' else np.nan
            membership.append(g)
            for c, h in g.groupby('class_id'):
                counts = dict(split_fold=fold, partition=part, class_id=c, class_name=h.class_name.iloc[0],
                    observations=len(h), ships=h.mmsi.nunique(), products=h['product'].nunique(),
                    ports=h.port.nunique(), platforms=h.platform.nunique(),
                    source_weight_sum=h.source_weight.sum() if part == 'train' else np.nan)
                class_counts.append(counts)
                if part == 'train':
                    np.testing.assert_allclose(h.source_weight.sum(), len(ii) / 8)
                    ship_mass = h.groupby('mmsi').source_weight.sum()
                    np.testing.assert_allclose(ship_mass, ship_mass.iloc[0])
                for category in ('port', 'platform', 'product'):
                    for value, sub in h.groupby(category):
                        category_counts.append(dict(split_fold=fold, partition=part, class_id=c,
                            category=category, value=value, observations=len(sub), ships=sub.mmsi.nunique(),
                            source_weight_sum=sub.source_weight.sum() if part == 'train' else np.nan))
            partitions.append(dict(split_fold=fold, partition=part, observations=len(g), ships=g.mmsi.nunique(),
                products=g['product'].nunique(), ports=g.port.nunique(), platforms=g.platform.nunique(),
                earliest=g.time.min(), latest=g.time.max()))
        files = sorted(PAIRED.glob(f'scores_f{fold}_*.npz'))
        assert len(files) == 21
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['rows'], split[2])
            checks.append(dict(file=path.name, split_fold=fold, n=len(split[2]), ordered_rows_exact=True))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            assert not set(f.iloc[split[a]].mmsi) & set(f.iloc[split[b]].mmsi)
            assert not set(f.iloc[split[a]]['product']) & set(f.iloc[split[b]]['product'])
    members = pd.concat(membership, ignore_index=True)
    save('split_membership', members.drop(columns=['image_vv', 'image_vh']))
    save('split_summary', partitions); save('class_coverage', class_counts)
    save('category_coverage', category_counts); save('score_rows_verification', checks)
    geometry = metadata(f)
    save('geometry_metadata', geometry)
    data = members.drop(columns=['incidence', 'match_status']).merge(geometry, on='obs_id', validate='many_to_many')
    ranges = []
    for keys, g in data.groupby(['split_fold', 'partition', 'class_id', 'pol']):
        for field in FIELDS + ['acute_axis_deg']:
            vals = g[field].dropna()
            record = dict(zip(['split_fold', 'partition', 'class_id', 'pol'], keys))
            record.update(field=field, observations=len(g), available=len(vals),
                valid_geometry=int(g.geometry_valid.sum()), minimum=vals.min(), maximum=vals.max(),
                q05=vals.quantile(.05), q50=vals.quantile(.5), q95=vals.quantile(.95))
            if keys[1] == 'train':
                for q in (.05, .5, .95):
                    record[f'ship_balanced_q{int(q*100):02d}'] = wquantile(g[field].to_numpy(), g.source_weight.to_numpy(), q)
            ranges.append(record)
    save('geometry_ranges', ranges)
    pairs = pd.read_csv(PAIRED / 'data_review/main_pairs.csv', dtype={'mmsi': str})
    pairs = pairs[pairs.class_id.eq(5)].reset_index(drop=True)
    assert len(pairs) == 100 and pairs.mmsi.nunique() == 100
    endpoints = pd.concat([pairs[['pair_id', 'mmsi', 'fold', 'obs_id_' + side]].rename(columns={
        'obs_id_' + side: 'obs_id'}).assign(side=side) for side in ('a', 'b')], ignore_index=True)
    assert endpoints.obs_id.nunique() == 200
    test = members[members.partition.eq('test')]
    assert len(test) == 4744 and test.obs_id.is_unique
    ep = endpoints.merge(test.drop(columns=['fold', 'mmsi']), on='obs_id', validate='one_to_one')
    assert ep.fold.eq(ep.split_fold).all()
    assert ep.class_id.eq(5).all()
    assert ep.mmsi.eq(ep.obs_id.map(f.set_index('obs_id').mmsi)).all()
    save('crude_endpoints', ep.drop(columns=['image_vv', 'image_vh']))
    calibration = []; reference = []; coverage = []
    for (fold, pol, c), source in data[data.partition.eq('train')].groupby(['split_fold', 'pol', 'class_id']):
        source = source[source.geometry_valid].reset_index(drop=True)
        n = len(source)
        if n:
            ds = distance(source, source)
            # Repeated views of the same vessel/product are not independent references.
            ds[source.mmsi.to_numpy()[:, None] == source.mmsi.to_numpy()[None, :]] = np.inf
            ds[source['product'].to_numpy()[:, None] == source['product'].to_numpy()[None, :]] = np.inf
            nearest = ds.min(axis=1)
            threshold = wquantile(nearest, source.source_weight.to_numpy(), .95)
            for j, row in source.iterrows():
                other = source.iloc[int(ds[j].argmin())]
                reference.append(dict(split_fold=fold, pol=pol, class_id=c, obs_id=row.obs_id,
                    mmsi=row.mmsi, product=row['product'], source_weight=row.source_weight,
                    independent_nn_distance=nearest[j], nearest_obs_id=other.obs_id,
                    nearest_mmsi=other.mmsi, nearest_product=other['product']))
            calibration.append(dict(split_fold=fold, pol=pol, class_id=c, observations=n,
                ships=source.mmsi.nunique(), products=source['product'].nunique(),
                calibration_finite=int(np.isfinite(nearest).sum()), source_ship_balanced_nn95=threshold,
                source_nn_median=wquantile(nearest, source.source_weight.to_numpy(), .5)))
        else:
            threshold = np.nan
        query = ep[ep.split_fold.eq(fold)].drop(columns=['incidence', 'match_status']).merge(
            geometry[geometry.pol.eq(pol)], on='obs_id', validate='one_to_one')
        for _, q in query.iterrows():
            r = dict(pair_id=q.pair_id, side=q.side, obs_id=q.obs_id, split_fold=fold, pol=pol,
                source_class_id=c, source_class_name=f.loc[f.class_id.eq(c), 'class_name'].iloc[0],
                query_geometry_valid=q.geometry_valid, source_geometry_observations=n,
                source_geometry_ships=source.mmsi.nunique(), source_nn95=threshold,
                same_port_source_ships=source.loc[source.port.eq(q.port), 'mmsi'].nunique(),
                same_platform_source_ships=source.loc[source.platform.eq(q.platform), 'mmsi'].nunique(),
                same_port_platform_source_ships=source.loc[source.port.eq(q.port) & source.platform.eq(q.platform), 'mmsi'].nunique())
            if q.geometry_valid and n:
                d = distance(q.to_frame().T, source)[0]
                near = source.iloc[np.flatnonzero(d <= 1)]
                best = source.iloc[int(d.argmin())]
                r.update(nearest_distance=d.min(), beyond_source_nn95=bool(d.min() > threshold),
                    fixed_near_observations=len(near), fixed_near_ships=near.mmsi.nunique(),
                    fixed_near_products=near['product'].nunique(), fixed_near_ports=near.port.nunique(),
                    fixed_near_source_class_weight_fraction=near.source_weight.sum() / (len(splits[fold][0]) / 8),
                    near_same_port_ships=near.loc[near.port.eq(q.port), 'mmsi'].nunique(),
                    near_same_platform_ships=near.loc[near.platform.eq(q.platform), 'mmsi'].nunique(),
                    nearest_obs_id=best.obs_id, nearest_mmsi=best.mmsi, nearest_product=best['product'],
                    nearest_port=best.port, nearest_platform=best.platform,
                    nearest_angle_gap=abs((q.relative_axis_deg-best.relative_axis_deg+90)%180-90),
                    nearest_incidence_gap=abs(q.incidence-best.incidence),
                    nearest_length_ratio=q.length_px/best.length_px, nearest_width_ratio=q.width_px/best.width_px)
                for field in ['acute_axis_deg', 'incidence', 'length_px', 'width_px']:
                    lo, hi = source[field].min(), source[field].max()
                    r[f'{field}_query'] = q[field]
                    r[f'{field}_source_min'] = lo; r[f'{field}_source_max'] = hi
                    r[f'{field}_out_of_range'] = bool(q[field] < lo or q[field] > hi)
            coverage.append(r)
    save('source_nn_reference', reference); save('source_calibration', calibration)
    cov = save('endpoint_source_coverage', coverage)
    summary = []
    for keys, g in cov.groupby(['pol', 'source_class_id', 'source_class_name']):
        valid = g[g.query_geometry_valid]
        summary.append(dict(zip(['pol', 'source_class_id', 'source_class_name'], keys),
            endpoints=len(g), available=len(valid), missing_geometry=len(g)-len(valid),
            fixed_no_neighbor=int(valid.fixed_near_ships.eq(0).sum()),
            fixed_near_ships_median=valid.fixed_near_ships.median(),
            beyond_source_nn95=int(valid.beyond_source_nn95.sum()),
            nearest_distance_median=valid.nearest_distance.median(),
            no_same_port_source=int(g.same_port_source_ships.eq(0).sum()),
            no_same_platform_source=int(g.same_platform_source_ships.eq(0).sum()),
            no_same_port_platform_source=int(g.same_port_platform_source_ships.eq(0).sum()),
            **{field+'_outside': int(valid[field+'_out_of_range'].sum()) for field in ['acute_axis_deg','incidence','length_px','width_px']}))
    save('endpoint_coverage_summary', summary)
    # Outcomes enter only after definitions and all coverage values are fixed.
    outcomes = []; confusion = []
    for fold, (_, _, ii) in enumerate(splits):
        for seed in range(3):
            with np.load(PAIRED / f'scores_f{fold}_baseline_s{seed}.npz', allow_pickle=False) as z:
                score = z['scores']
            y = f.iloc[ii].class_id.to_numpy(); pred = score.argmax(1)
            for rival in range(8):
                confusion.append(dict(split_fold=fold, seed=seed, prediction=rival,
                    crude_observations=int((y==5).sum()), events=int(((y==5)&(pred==rival)).sum())))
            for j, i in enumerate(ii):
                if f.iloc[i].obs_id in set(endpoints.obs_id):
                    other = score[j].copy(); other[5] = -np.inf
                    outcomes.append(dict(obs_id=f.iloc[i].obs_id, seed=seed, prediction=int(pred[j]),
                        correct=int(pred[j]==5), margin=float(score[j,5]-other.max())))
    save('crude_confusion', confusion); outcome = save('endpoint_baseline_outcomes', outcomes)
    mean = outcome.groupby('obs_id')[['correct','margin']].mean()
    own = cov[cov.source_class_id.eq(5)].merge(mean, on='obs_id', validate='many_to_one')
    own = own.merge(ep[['obs_id','port','platform']], on='obs_id', validate='many_to_one')
    save('crude_gap_cases', own[own.beyond_source_nn95.eq(True) | own.fixed_near_ships.eq(0)])
    association = []; pair_rows = []
    for pol, g in own.groupby('pol'):
        v = g[g.query_geometry_valid].copy()
        for flag in ('beyond_source_nn95', 'fixed_no_neighbor'):
            if flag == 'fixed_no_neighbor':
                v[flag] = v.fixed_near_ships.eq(0)
            for state, sub in v.groupby(flag):
                association.append(dict(pol=pol, flag=flag, state=state, endpoints=len(sub),
                    vessels=sub.pair_id.nunique(), mean_seed_correct=sub.correct.mean(), mean_margin=sub.margin.mean()))
        for pid, sub in g.groupby('pair_id'):
            a = sub[sub.side.eq('a')].iloc[0]; b = sub[sub.side.eq('b')].iloc[0]
            pair_rows.append(dict(pair_id=pid, pol=pol, split_fold=a.split_fold,
                obs_id_a=a.obs_id, obs_id_b=b.obs_id,
                both_geometry_valid=bool(a.query_geometry_valid and b.query_geometry_valid),
                delta_nearest_distance=b.nearest_distance-a.nearest_distance,
                delta_margin=b.margin-a.margin, correct_a=a.correct, correct_b=b.correct,
                distance_a=a.nearest_distance, distance_b=b.nearest_distance,
                gap_a=a.beyond_source_nn95, gap_b=b.beyond_source_nn95))
    save('coverage_outcome_associations', association)
    stratified = []
    for category in ('split_fold', 'port', 'platform'):
        for keys, sub in own.groupby(['pol', category, 'beyond_source_nn95']):
            stratified.append(dict(pol=keys[0], category=category, value=keys[1], gap=keys[2],
                endpoints=len(sub), vessels=sub.pair_id.nunique(), mean_seed_correct=sub.correct.mean(),
                mean_margin=sub.margin.mean(), median_distance=sub.nearest_distance.median()))
    save('coverage_outcome_strata', stratified)
    pp = save('paired_coverage_changes', pair_rows)
    # pandas rank-correlation avoids adding a scipy dependency.
    assoc = []
    for keys, g in pp.groupby(['pol', 'split_fold']):
        g = g[g.both_geometry_valid]
        assoc.append(dict(pol=keys[0], split_fold=keys[1], pairs=len(g),
            spearman_distance_vs_margin=g.delta_nearest_distance.rank().corr(g.delta_margin.rank())))
    save('paired_associations', assoc)
    receipt = dict(status='complete', classifier_fits=0, feature_arrays_read=0, image_reads=0,
        score_files_rows_verified=len(checks), test_observations=len(test), crude_pairs=len(pairs), crude_endpoints=len(ep),
        exclusion_ids=excluded.tolist(), weights='N_train / (8 * observations_per_ship * ships_in_class)',
        tolerances=dict(axial_degrees=ANGLE, incidence_degrees=INCIDENCE, length_width_ratio=1.25),
        distance='max(axial_gap/15, incidence_gap/3, abs(log(Lq/Ls))/log(1.25), abs(log(Wq/Ws))/log(1.25))',
        calibration='Each fold/class/polarization: source nearest neighbor excluding same MMSI and same product; weighted empirical q95 using source weights; missing geometry excluded, never imputed.',
        source_scope='Only paired_relation training rows. Validation never added. All query references are unseen ships/products by split.',
        interpretation='Coverage association only, no causal conclusion; fixed tolerances are descriptive, not validated physical cutoffs.',
        coordinates_used=False, image_and_literature_review='Reserved for main agent; not performed here.',
        github_sync='Reserved for main-agent combined delivery: writes restricted to source_coverage directory.')
    (OUT/'RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
