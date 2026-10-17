"""Exploratory factor analysis with acquisition/vessel-held-out image prediction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'oss1_polar_environment_20261017'
GEOM = HERE.parent/'oss1_joint_geometry_20261017'
FACTORS = ['ais_sog_mps', 'ground_los_3d_abs_mps', 'ground_alongflight_abs_mps',
           'ais_draught_m', 'ais_length_m', 'ais_width_m', 'incidence_deg',
           'era5_wind_speed_m_s', 'era5_hs', 'cmems_current_speed_m_s']
OUTCOMES = ['body_top5_mass', 'azimuth_range_log_ratio', 'body_neighbor_db', 'ends_sides_balance']


def corr(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[good], np.asarray(y)[good]
    return float(spearmanr(x, y).statistic) if len(x) >= 8 and np.std(x) > 1e-9 and np.std(y) > 1e-9 else np.nan


def load():
    m = pd.read_csv(HERE/'metadata/oss1_factor_metadata.csv', dtype={'mmsi': str})
    s = pd.read_csv(HERE/'native_shape_measurements.csv', dtype={'mmsi': str})
    e = pd.read_csv(OLD/'environment/sample_environment.csv')
    cols = ['sample_id', 'era5_wind_speed_m_s', 'era5_hs', 'era5_mean_wave_period',
            'era5_mean_wave_direction', 'cmems_current_speed_m_s', 'cmems_thetao', 'cmems_so']
    cols = [c for c in cols if c in e]
    m = m.merge(e[cols], on='sample_id', how='left', validate='one_to_one')
    d = pd.read_csv(OLD/'image_descriptors.csv')
    d = d[d.dataset.eq('OSS1')][['global_row', 'pol', 'representation', 'body_neighbor_db',
                                'ends_sides_balance', 'min_outside_coverage']]
    s = s.merge(d, on=['global_row', 'pol', 'representation'], how='left', validate='one_to_one')
    drop = [c for c in s if c in m and c != 'global_row']
    data = s.drop(columns=drop).merge(m, on='global_row', how='left', validate='many_to_one').copy()
    if data.loc[data.representation.eq('sigma0'), 'body_neighbor_db'].notna().sum() < 2500:
        raise AssertionError('Calibrated descriptors did not join the current image cohort')
    data['axis_cos2'] = np.cos(2*np.radians(data.ship_axis_deg))
    data['axis_sin2'] = np.sin(2*np.radians(data.ship_axis_deg))
    for col in ['ais_length_m', 'ais_width_m', 'ais_draught_m']:
        data['log_'+col] = np.log(data[col].where(data[col] > 0))
    data['radial_incidence'] = data.ground_los_3d_abs_mps*(data.incidence_deg-35)/10
    data['sog_axis_cos2'] = data.ais_sog_mps*data.axis_cos2
    data['length_axis_cos2'] = data.log_ais_length_m*data.axis_cos2
    data['draught_incidence'] = data.log_ais_draught_m*(data.incidence_deg-35)/10
    data['speed_group'] = pd.cut(data.ais_sog_knots, [-.001, .5, 3, np.inf], labels=['low<=0.5kn', '0.5-3kn', '>3kn'])
    data.to_csv(HERE/'joined_image_factors.csv', index=False)
    return data, m


def associations(data):
    records = []
    for pol, p in data[data.representation.eq('sigma0')].groupby('pol'):
        for scope, block in [('all', p), ('gap<=60s', p[p.ais_abs_timegap_s <= 60]), ('moving>3kn', p[p.ais_sog_knots > 3])]:
            for factor in FACTORS:
                for outcome in OUTCOMES:
                    f = block[block.min_outside_coverage >= .5] if outcome in OUTCOMES[2:] else block
                    controls = ['log_ais_length_m', 'log_ais_width_m', 'axis_cos2', 'axis_sin2', 'incidence_deg']
                    if factor in ('ais_length_m', 'ais_width_m'):
                        controls.remove('log_'+factor)
                    if factor == 'incidence_deg':
                        controls.remove('incidence_deg')
                    f = f.dropna(subset=[factor, outcome]+controls)
                    if len(f) < 25:
                        continue
                    group = f['product'].astype(str)+'/'+f.class_id.astype(str)
                    counts = group.map(group.value_counts())
                    f, group = f[counts >= 3], group[counts >= 3]
                    if len(f) < 25:
                        continue
                    design = pd.concat([pd.DataFrame({c: rankdata(f[c]) for c in controls}, index=f.index),
                                        pd.get_dummies(group, dtype=float)], axis=1).to_numpy(float)
                    xy = np.column_stack([rankdata(f[factor]), rankdata(f[outcome])])
                    residual = xy-design@np.linalg.lstsq(design, xy, rcond=None)[0]
                    records.append(dict(pol=pol, scope=scope, factor=factor, outcome=outcome,
                        n=len(f), ships=f.mmsi.nunique(), products=f['product'].nunique(),
                        spearman=corr(f[factor], f[outcome]), residual_rank_correlation=corr(residual[:, 0], residual[:, 1]),
                        method='within product/class with size, detected axis, incidence controls; descriptive, not causal'))
    pd.DataFrame(records).to_csv(HERE/'conditional_associations.csv', index=False)


def same_vessel(data):
    pairs = pd.read_csv(OLD/'same_vessel_pairs.csv', dtype={'mmsi': str})
    pairs = pairs[pairs.dataset.eq('OSS1') & pairs.representation.eq('sigma0')]
    records = []
    for pol, group in pairs.groupby('pol'):
        images = data[data.pol.eq(pol) & data.representation.eq('sigma0')].set_index('global_row')
        for _, p in group.iterrows():
            a, b = images.loc[p['first']], images.loc[p['second']]
            item = dict(pol=pol, mmsi=p.mmsi, first=int(p['first']), second=int(p['second']),
                        class_id=int(p.class_id), first_product=p.first_product, second_product=p.second_product,
                        max_ais_gap=max(a.ais_abs_timegap_s, b.ais_abs_timegap_s),
                        min_coverage=min(a.min_outside_coverage, b.min_outside_coverage))
            for col in FACTORS+OUTCOMES+['axis_cos2', 'axis_sin2']:
                item['delta_'+col] = b[col]-a[col]
            records.append(item)
    delta = pd.DataFrame(records)
    delta.to_csv(HERE/'same_vessel_deltas.csv', index=False)
    stats = []
    for pol, p in delta.groupby('pol'):
        for factor in FACTORS:
            for outcome in OUTCOMES:
                for scope, block in [('all', p), ('both_gap<=60s', p[p.max_ais_gap <= 60])]:
                    f = block.dropna(subset=['delta_'+factor, 'delta_'+outcome])
                    if outcome in OUTCOMES[2:]:
                        f = f[f.min_coverage >= .5]
                    stats.append(dict(pol=pol, scope=scope, factor=factor, outcome=outcome, n=len(f),
                                      spearman=corr(f['delta_'+factor], f['delta_'+outcome])))
    pd.DataFrame(stats).to_csv(HERE/'same_vessel_associations.csv', index=False)


def image_prediction(data):
    # Known class labels are diagnostic controls, never inference features.
    base = ['log_ais_length_m', 'log_ais_width_m', 'axis_cos2', 'axis_sin2', 'incidence_deg', 'length_axis_cos2']
    speed = ['ais_sog_mps']
    projected = speed+['ground_los_3d_abs_mps', 'ground_alongflight_abs_mps', 'radial_incidence', 'sog_axis_cos2']
    env = ['era5_wind_speed_m_s', 'era5_hs', 'cmems_current_speed_m_s']
    draught = ['log_ais_draught_m', 'draught_incidence']
    arms = dict(class_only=[], structure_only=base[:2], base=base, speed=base+speed, motion_geometry=base+projected,
                environment=base+env, draught=base+draught, joint=base+projected+env+draught)
    pred, splits = [], []
    for pol, p in data[data.representation.eq('sigma0')].groupby('pol'):
        for outcome in OUTCOMES:
            cohort = p[np.isfinite(p[outcome])].copy()
            if outcome in OUTCOMES[2:]:
                cohort = cohort[cohort.min_outside_coverage >= .5]
            cohort = cohort.reset_index(drop=True)
            classes = np.eye(8)[cohort.class_id.to_numpy(int)]
            for product, target in cohort.groupby('product'):
                train = cohort[~cohort['product'].eq(product) & ~cohort.mmsi.isin(target.mmsi)]
                if len(train) < 128:
                    continue
                ytrain = train[outcome].to_numpy(float)
                splits.append(dict(pol=pol, outcome=outcome, target_product=product, source_rows=len(train),
                                   source_ships=train.mmsi.nunique(), target_rows=len(target),
                                   overlapping_mmsi=len(set(train.mmsi)&set(target.mmsi))))
                for arm, columns in arms.items():
                    x = np.column_stack([cohort[columns].to_numpy(float), classes])
                    model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                                          StandardScaler(), Ridge(alpha=10.))
                    model.fit(x[train.index], ytrain)
                    prediction = model.predict(x[target.index])
                    for (_, row), value in zip(target.iterrows(), prediction):
                        pred.append(dict(pol=pol, outcome=outcome, arm=arm, product=product, global_row=int(row.global_row),
                                         mmsi=row.mmsi, class_id=int(row.class_id), truth=row[outcome], prediction=value))
    pred = pd.DataFrame(pred)
    pred['squared_error'] = (pred.prediction-pred.truth)**2
    pred.to_csv(HERE/'image_response_predictions.csv', index=False)
    pd.DataFrame(splits).to_csv(HERE/'image_response_splits.csv', index=False)
    summary = []
    for (pol, outcome), block in pred.groupby(['pol', 'outcome']):
        errors = block.pivot(index=['global_row', 'product', 'class_id'], columns='arm', values='squared_error')
        by_product = errors.groupby('product').mean()
        by_class = errors.groupby('class_id').mean()
        rng = np.random.default_rng(20261017)
        draws = rng.integers(0, len(by_product), (1000, len(by_product)))
        for arm in arms:
            gain = (by_product['base']-by_product[arm])/by_product['base'].mean()
            boot = gain.to_numpy()[draws].mean(1)*100
            values = block[block.arm.eq(arm)]
            summary.append(dict(pol=pol, outcome=outcome, arm=arm, n=len(values), products=len(by_product),
                mse=values.squared_error.mean(), product_macro_mse=by_product[arm].mean(), class_macro_mse=by_class[arm].mean(),
                product_relative_mse_reduction_pct=100*gain.mean(), bootstrap_low_pct=np.quantile(boot, .025),
                bootstrap_high_pct=np.quantile(boot, .975), products_improved=int((gain > 0).sum())))
    pd.DataFrame(summary).to_csv(HERE/'image_response_summary.csv', index=False)
    (HERE/'FACTOR_PROTOCOL.json').write_text(json.dumps(dict(
        diagnostic='predict image response, NOT ship class; class is a nuisance/structure control',
        arms=arms, ridge_alpha=10, splitting='leave one product out AND exclude its MMSIs from source',
        imputation='source fold medians and missingness indicators; no external values fabricated',
        evaluation='same cohort per outcome/pol/arm, product- and class-macro MSE',
        uncertainty='1000 product bootstrap of fixed OOF losses, exploratory, not model-refit or causal CI',
        limits='many dependent exploratory hypotheses; no best-factor causal claim'), indent=2), encoding='utf-8')


def classifier_diagnosis(meta):
    p = pd.read_csv(HERE/'classifier/predictions.csv')
    p = p.merge(meta[['global_row', 'class_name', 'mmsi', 'ais_sog_knots', 'ais_abs_timegap_s', 'ais_length_m', 'ais_draught_m']],
                on='global_row', how='left', validate='many_to_one')
    p['net'] = (p.prediction.eq(p.truth)).astype(int)-p.baseline.eq(p.truth).astype(int)
    p['prob_gain'] = p.true_probability-p.baseline_true_probability
    p['speed_group'] = pd.cut(p.ais_sog_knots, [-.001, .5, 3, np.inf], labels=['low<=0.5kn', '0.5-3kn', '>3kn'])
    p.to_csv(HERE/'classifier/factor_predictions.csv', index=False)
    p[p.net.ne(0)].to_csv(HERE/'classifier/changed_events.csv', index=False)
    summaries = []
    for dimension in ['class_name', 'speed_group', 'support_class_count', 'fold']:
        for keys, g in p.groupby(['origin', 'arm', dimension], observed=True):
            summaries.append(dict(origin=keys[0], arm=keys[1], dimension=dimension, value=keys[2],
                unique_queries=g.global_row.nunique(), events=len(g), net_events=int(g.net.sum()),
                mean_probability_gain=g.prob_gain.mean(), mean_ap_gain=(g.ap-g.baseline_ap).mean()))
    pd.DataFrame(summaries).to_csv(HERE/'classifier/group_diagnosis.csv', index=False)


def gallery(data):
    folder = HERE/'figures'
    folder.mkdir(exist_ok=True)
    m = pd.read_csv(GEOM/'geometry.csv').set_index('global_row')
    det = pd.read_csv(GEOM/'detections.csv').set_index('global_row')
    cohort = data[data.pol.eq('vv') & data.representation.eq('sigma0')]
    selected = []
    for cls, group in cohort.groupby('class_id'):
        ordered = group.sort_values(['ais_sog_knots', 'sample_id']).drop_duplicates('mmsi')
        chosen = pd.concat([ordered.head(3), ordered.tail(3)]).drop_duplicates('global_row')
        fig, axes = plt.subplots(len(chosen), 2, figsize=(7, 2.5*len(chosen)), squeeze=False)
        for rownum, (_, row) in enumerate(chosen.iterrows()):
            r, d = m.loc[row.global_row], det.loc[row.global_row]
            image = tifffile.imread(r.calibrated_patch_path)
            co, si = np.cos(d.ship_axis), np.sin(d.ship_axis)
            for col, (pol, channel) in enumerate([('VV', 1), ('VH', 0)]):
                ax = axes[rownum, col]
                ax.imshow(10*np.log10(np.maximum(image[:, :, channel], 1e-8)), cmap='gray', vmin=-35, vmax=5)
                a, b = d.length*1.25/2, d.width*1.25/2
                for scale, color in [(1, '#31b6df'), (3, '#edb64c')]:
                    u, v = np.array([-a, a, a, -a, -a])*scale, np.array([-b, -b, b, b, -b])*scale
                    ax.plot(d.cx+u*co-v*si, d.cy+u*si+v*co, color=color, lw=.8)
                ax.set_xlim(-.5, image.shape[1]-.5)
                ax.set_ylim(image.shape[0]-.5, -.5)
                ax.set_title(f'{pol} row {int(row.global_row)} | {row.ais_sog_knots:.1f} kn | gap {row.ais_abs_timegap_s:.0f}s\n'
                             f'axis {row.ship_axis_deg:.0f} deg | inc {row.incidence_deg:.1f} deg | AIS L {row.ais_length_m:.0f}m', fontsize=8)
                ax.tick_params(labelsize=6)
            selected.append(dict(class_id=int(cls), global_row=int(row.global_row), mmsi=row.mmsi,
                                 selection='three lowest and highest SOG distinct vessels within class; no outcomes',
                                 ais_sog_knots=row.ais_sog_knots, ais_abs_timegap_s=row.ais_abs_timegap_s))
        fig.suptitle(f'{chosen.class_name.iloc[0]} | calibrated sigma0 [-35, 5] dB\nCyan: expanded OBB; amber: nominal outer support, clipped by chip', fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, .96])
        fig.savefig(folder/f'class_{int(cls)}_native.png', dpi=140)
        plt.close(fig)
    pd.DataFrame(selected).to_csv(folder/'gallery_selection.csv', index=False)


def main():
    with threadpool_limits(limits=1):
        data, meta = load()
        print('JOIN', len(data), 'native repeated measurements', flush=True)
        associations(data)
        same_vessel(data)
        print('PREDICT IMAGE RESPONSE', flush=True)
        image_prediction(data)
        classifier_diagnosis(meta)
        print('GALLERY', flush=True)
        gallery(data)
        summary = pd.read_csv(HERE/'image_response_summary.csv')
        print(summary[['pol', 'outcome', 'arm', 'product_relative_mse_reduction_pct', 'bootstrap_low_pct', 'bootstrap_high_pct']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
