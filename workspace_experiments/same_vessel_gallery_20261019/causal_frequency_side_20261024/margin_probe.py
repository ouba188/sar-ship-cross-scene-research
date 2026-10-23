"""Held-vessel diagnostic of old classifier margins; not a new classifier."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'round05_margin'
SPATIAL = ['log_length', 'log_width', 'log_area', 'contrast',
           'target_saturated', 'far_mean', 'far_sd']
FREQUENCY = ['core_centroid', 'core_high', 'outer_centroid', 'outer_high']


def load():
    spec = importlib.util.spec_from_file_location('nextobs', ROOT / 'next_observation_20261021/run.py')
    nxt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nxt)
    frame, _ = nxt.load()
    freq = pd.read_csv(HERE / 'round04_frequency/measurements.csv')
    frame = frame.merge(freq, on=['obs_id', 'pol'], how='inner', validate='one_to_one')
    for field in ('length', 'width', 'area'):
        name = 'r3_area_m2' if field == 'area' else f'r3_{field}_m'
        frame['log_' + field] = np.log(frame[name])
    frame['length_squared'] = frame.log_length ** 2
    frame['width_squared'] = frame.log_width ** 2
    for name in FREQUENCY:
        frame['log_' + name] = np.log(frame[name])
    pred = pd.read_csv(ROOT / 'conditional_evidence_pilot/predictions.csv', dtype={'mmsi': str})
    pred = pred[pred.arm.isin(['image', 'interaction']) & pred.obs_id.ne(31237)]
    assert not pred.duplicated(['obs_id', 'arm']).any()
    wide = pred.pivot(index='obs_id', columns='arm', values='margin').add_prefix('margin_')
    frame = frame.merge(wide, on='obs_id', validate='many_to_one')
    prior = pd.read_csv(ROOT / 'next_observation_20261021/predictions.csv')
    prior = prior[prior.variant.eq('all') & prior.metric.eq('r3_length_m') & prior.arm.eq('transfer_geometry')]
    base = nxt.META + nxt.GEOM + SPATIAL
    arms = {'spatial_geometry': base,
            'plus_core': base + ['log_core_centroid', 'log_core_high'],
            'plus_outer': base + ['log_outer_centroid', 'log_outer_high'],
            'plus_both': base + ['log_' + n for n in FREQUENCY],
            'plus_quadratic': base + ['length_squared', 'width_squared']}
    return frame, prior, arms


def fit(frame, target, columns):
    x = frame[columns].to_numpy(float)
    y = frame[target].to_numpy(float)
    xc, yc, weight = np.empty_like(x), np.empty_like(y), np.empty_like(y)
    ids = frame.mmsi.to_numpy()
    for ship in np.unique(ids):
        use = ids == ship
        xc[use] = x[use] - x[use].mean(0)
        yc[use] = y[use] - y[use].mean()
        weight[use] = 1 / use.sum()
    scale = np.sqrt(np.average(xc ** 2, axis=0, weights=weight))
    active = scale > 1e-8
    scale = np.where(active, scale, 1)
    z = xc / scale
    z[:, ~active] = 0
    beta = np.linalg.solve(z.T @ (weight[:, None] * z) + 10 * np.eye(len(columns)),
                           z.T @ (weight * yc))
    return beta / scale * active


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, prior, arms = load()
    records, manifests = [], []
    for pol, full in frame.groupby('pol'):
        qids = prior[prior.pol.eq(pol)].obs_id.tolist()
        valid = full.dropna(subset=SPATIAL + ['log_' + n for n in FREQUENCY]).copy()
        assert set(qids).issubset(set(valid.obs_id))
        for classifier in ('image', 'interaction'):
            target = 'margin_' + classifier
            cache = {}
            for oid in qids:
                query = valid[valid.obs_id.eq(oid)].iloc[0]
                history = valid[valid.mmsi.eq(query.mmsi) & valid.timestamp.lt(query.timestamp)]
                assert len(history) >= 3
                key = (int(query.split_fold), query.time, query.overpass)
                if key not in cache:
                    source = valid[valid.split_fold.ne(query.split_fold) &
                                   valid.timestamp.lt(query.timestamp) & valid.overpass.ne(query.overpass)]
                    source = source.groupby('mmsi').filter(lambda g: len(g) >= 2)
                    assert query.mmsi not in set(source.mmsi)
                    assert source.timestamp.max() < query.timestamp
                    assert not source.overpass.eq(query.overpass).any()
                    fitted = {name: fit(source, target, cols) for name, cols in arms.items()}
                    fit_id = len(manifests)
                    manifests.append(dict(fit_id=fit_id, pol=pol, classifier=classifier,
                        excluded_fold=int(query.split_fold), before_time=query.time,
                        excluded_overpass=query.overpass, source_ids=source.obs_id.astype(int).tolist(),
                        source_ships=int(source.mmsi.nunique()),
                        coefficients={name: dict(zip(arms[name], b.tolist())) for name, b in fitted.items()}))
                    cache[key] = fitted, fit_id
                fitted, fit_id = cache[key]
                ref = float(history[target].mean())
                actual = float(query[target])
                predictions = {'history': ref}
                for name, beta in fitted.items():
                    columns = arms[name]
                    predictions[name] = ref + float((query[columns].to_numpy(float) -
                                                     history[columns].mean().to_numpy()) @ beta)
                for arm, value in predictions.items():
                    records.append(dict(obs_id=int(oid), mmsi=query.mmsi, pol=pol,
                        classifier=classifier, class_id=int(query.class_id), port=query.port,
                        overpass=query.overpass, fold=int(query.split_fold), time=query.time,
                        history_ids=' '.join(history.obs_id.astype(str)), history_margin=ref,
                        history_correct_fraction=float(history[target].gt(0).mean()),
                        latest_history_id=int(history.sort_values('timestamp').iloc[-1].obs_id),
                        latest_history_margin=float(history.sort_values('timestamp').iloc[-1][target]),
                        fit_id=fit_id, arm=arm, actual_margin=actual, predicted_margin=value,
                        abs_error=abs(value - actual), squared_error=(value - actual) ** 2,
                        sign_correct=bool((value > 0) == (actual > 0))))
            print('Completed', pol, classifier, len(qids), flush=True)
    pred = pd.DataFrame(records)
    pred.to_csv(OUT / 'predictions.csv', index=False)
    (OUT / 'fit_manifest.json').write_text(json.dumps(manifests, indent=2), encoding='utf8')
    summaries, groups = [], []
    rng = np.random.default_rng(20261024)
    for (pol, classifier), part in pred.groupby(['pol', 'classifier']):
        index = ['obs_id', 'mmsi', 'fold', 'overpass', 'class_id']
        wide = part.pivot(index=index, columns='arm', values='abs_error').reset_index()
        for baseline, arm in [('history', 'spatial_geometry'),
                              ('spatial_geometry', 'plus_core'),
                              ('spatial_geometry', 'plus_outer'),
                              ('spatial_geometry', 'plus_both'),
                              ('spatial_geometry', 'plus_quadratic'),
                              ('plus_quadratic', 'plus_core')]:
            gain = wide[baseline] - wide[arm]
            by_pass = wide.assign(gain=gain).groupby('overpass').agg(s=('gain', 'sum'), n=('gain', 'size'))
            draws = rng.integers(len(by_pass), size=(2000, len(by_pass)))
            boot = by_pass.s.to_numpy()[draws].sum(1) / by_pass.n.to_numpy()[draws].sum(1)
            summaries.append(dict(pol=pol, classifier=classifier, baseline=baseline, arm=arm,
                n=len(wide), baseline_mae=float(wide[baseline].mean()), model_mae=float(wide[arm].mean()),
                gain=float(gain.mean()), reduction_pct=float(100 * gain.mean() / wide[baseline].mean()),
                improved=int(gain.gt(0).sum()), harmed=int(gain.lt(0).sum()),
                positive_folds=int(wide.assign(gain=gain).groupby('fold').gain.mean().gt(0).sum()),
                pass_ci_low=float(np.quantile(boot, .025)), pass_ci_high=float(np.quantile(boot, .975))))
            for axis in ('class_id', 'fold'):
                for group, g in wide.assign(gain=gain).groupby(axis):
                    groups.append(dict(pol=pol, classifier=classifier, baseline=baseline, arm=arm,
                        axis=axis, group=int(group), n=len(g), baseline_mae=g[baseline].mean(),
                        model_mae=g[arm].mean(), gain=g.gain.mean()))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / 'summary.csv', index=False)
    pd.DataFrame(groups).to_csv(OUT / 'strata.csv', index=False)
    frame.to_csv(OUT / 'diagnostic_input.csv', index=False)
    cases = []
    for (pol, classifier), part in pred.groupby(['pol', 'classifier']):
        wide = part.pivot(index='obs_id', columns='arm', values='abs_error')
        gain = wide.spatial_geometry - wide.plus_core
        for direction, g in [('improved', gain[gain > 0]), ('worse', gain[gain < 0])]:
            if len(g):
                oid = int((g - g.median()).abs().idxmin())
                cases.append(dict(pol=pol, classifier=classifier, selection=direction + '_median', obs_id=oid,
                                  gain=float(g.loc[oid])))
    pd.DataFrame(cases).to_csv(OUT / 'case_selection.csv', index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
