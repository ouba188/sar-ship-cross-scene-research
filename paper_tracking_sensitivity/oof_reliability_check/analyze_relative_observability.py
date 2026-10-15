"""Angle-specific observability and same-vessel held-identity response probes."""

import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from extract_ship_axis_proxy import OUT as GEOMETRY, SOURCE
from run_direct_observation_knowledge import REFERENCE
from run_conditioned_vessel_drift import SIDECAR
from same_vessel_temporal import collect_observations, make_pairs
from conditioned_vessel_drift import split_observations, condition_differences, fit_response, response_metrics
from sar_joint_completion import write_json

OUT = GEOMETRY/'observability'
OUTCOMES = ['gray_contrast', 'true_class_margin', 'correct_fraction']
ANGLE_BINS = [0, 15, 30, 45, 60, 75, 90.0001]


def observation_table():
    manifest = pd.read_csv(SOURCE/'manifest.csv', dtype={'mmsi': str}, low_memory=False)
    frame = pd.read_csv(GEOMETRY/'relative_conditions.csv')
    np.testing.assert_array_equal(frame.instance_id, manifest.instance_id)
    frame['mmsi'] = manifest.mmsi
    frame['class_name'] = manifest.class_name
    side = pd.read_csv(SIDECAR, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    frame = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    frame.loc[frame.status.ne('ok'), 'incidence_deg'] = np.nan
    n = len(frame)
    correct, margin, counts = np.zeros(n), np.zeros(n), np.zeros(n)
    for folder in sorted((REFERENCE/'ports').iterdir()):
        with np.load(folder/'policies.npz') as f:
            p = {k: f[k] for k in f.files}
        index = list(p['mode_names']).index('direct_condition')
        rows = p['query_rows']
        truth = frame.loc[rows, 'class_id'].to_numpy()
        for draw, ev in enumerate(p['eval_mask']):
            scores = p['base_scores'][draw, index].copy()
            true_score = scores[np.arange(len(rows)), truth].copy()
            scores[np.arange(len(rows)), truth] = -np.inf
            correct[rows[ev]] += (p['predictions'][draw, index, ev] == truth[ev])
            margin[rows[ev]] += (true_score-scores.max(axis=1))[ev]
            counts[rows[ev]] += 1
    frame['correct_fraction'] = np.divide(correct, counts, out=np.full(n, np.nan), where=counts > 0)
    frame['true_class_margin'] = np.divide(margin, counts, out=np.full(n, np.nan), where=counts > 0)
    frame['query_draw_count'] = counts
    accepted = frame.relative_status.eq('ok')
    frame['gray_contrast'] = np.nan
    for row_id in frame.index[accepted]:
        contrasts = []
        for pol in ['vv', 'vh']:
            with Image.open(manifest.loc[row_id, 'image_relpath_'+pol]) as im:
                a = np.asarray(im.convert('L').resize((128, 128), Image.Resampling.BILINEAR), dtype=float)
            yy, xx = np.mgrid[:128, :128]-63.5
            rim = (np.abs(xx) > 48) | (np.abs(yy) > 48)
            center = (np.abs(xx) < 32) & (np.abs(yy) < 32)
            target, background = np.quantile(a[center], .95), np.median(a[rim])
            contrasts.append((target-background)/(target+background+1.))
        frame.loc[row_id, 'gray_contrast'] = np.mean(contrasts)
    frame['angle_bin'] = pd.cut(frame.relative_acute_deg, ANGLE_BINS, right=False,
                               labels=['0-15', '15-30', '30-45', '45-60', '60-75', '75-90'])
    frame.to_csv(OUT/'observations.csv', index=False)
    return frame


def angle_tables(frame):
    valid = frame[frame.relative_status.eq('ok') & frame.correct_fraction.notna()]
    for names, filename in [(['class_id', 'class_name', 'angle_bin'], 'class_angle_bins.csv'),
                            (['port', 'class_id', 'class_name', 'angle_bin'], 'port_class_angle_bins.csv')]:
        table = valid.groupby(names, observed=True).agg(n=('instance_id', 'size'), ships=('mmsi', 'nunique'),
            products=('product', 'nunique'), oa=('correct_fraction', 'mean'),
            margin=('true_class_margin', 'mean'), gray_contrast=('gray_contrast', 'mean'),
            mean_incidence=('incidence_deg', 'mean')).reset_index()
        table.to_csv(OUT/filename, index=False)
    table = pd.read_csv(OUT/'class_angle_bins.csv')
    names = frame[['class_id', 'class_name']].drop_duplicates().sort_values('class_id')
    heat = np.full((8, 6), np.nan)
    sample = np.zeros((8, 6), dtype=int)
    bins = ['0-15', '15-30', '30-45', '45-60', '60-75', '75-90']
    for row in table.itertuples():
        j = bins.index(row.angle_bin)
        sample[row.class_id, j] = row.n
        if row.n >= 20:
            heat[row.class_id, j] = row.oa*100
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heat, vmin=0, vmax=100, cmap='viridis', aspect='auto')
    ax.set_xticks(range(6), bins)
    ax.set_yticks(range(8), names.class_name.str.replace('_', ' ').to_list())
    for i in range(8):
        for j in range(6):
            label = f'{heat[i,j]:.1f}%\nn={sample[i,j]}' if np.isfinite(heat[i,j]) else f'n={sample[i,j]}'
            ax.text(j, i, label, ha='center', va='center', fontsize=8,
                    color='white' if np.isfinite(heat[i,j]) and heat[i,j] < 60 else 'black')
    ax.set_xlabel('Acute body / ground-range axis angle (degrees)')
    ax.set_title('Reference correctness on measurable-axis observations\nDifferent vessels/ports; descriptive, NOT an optimal-angle law', fontsize=11)
    fig.colorbar(im, ax=ax, label='Correctness (%)')
    fig.tight_layout()
    fig.savefig(OUT/'class_angle_observability.png', dpi=150)
    plt.close(fig)


def temporal_probe(frame):
    eligible = frame.relative_status.eq('ok') & frame[['incidence_deg', *OUTCOMES]].notna().all(axis=1)
    ids = frame.index[eligible].to_numpy()
    r = np.deg2rad(frame.loc[ids, 'relative_signed_deg'].to_numpy())*2
    data = np.c_[frame.loc[ids, ['incidence_deg', 'latitude', 'longitude']].to_numpy(),
                 np.cos(r), np.sin(r), frame.loc[ids, OUTCOMES].to_numpy()]
    obs, z = collect_observations(frame, ids, data)
    split = split_observations(obs)
    obs['split'] = ''
    for name, mask in split.items():
        obs.loc[mask, 'split'] = name
    obs.to_csv(OUT/'temporal_observations.csv', index=False)
    packed = {}
    coverage = []
    matched = []
    for name, mask in split.items():
        part, values = obs.loc[mask].reset_index(drop=True), z[mask]
        pairs, differences, weights = make_pairs(part, values)
        if len(pairs):
            x = condition_differences(part, values[:, :3], pairs)
            extended = np.c_[x, differences[:, 3:5]]
            y = differences[:, 5:]
            packed[name] = (pairs, x, extended, y, weights)
            a, b = pairs.obs_a.to_numpy(int), pairs.obs_b.to_numpy(int)
            position_km = 10*np.linalg.norm(x[:, 2:4], axis=1)
            angle_a = np.rad2deg(np.arctan2(values[a, 4], values[a, 3]))/2
            angle_b = np.rad2deg(np.arctan2(values[b, 4], values[b, 3]))/2
            delta_angle = np.abs((angle_b-angle_a+90) % 180-90)
            nearby = (position_km <= 1) & (np.abs(values[b, 0]-values[a, 0]) <= 2) & (delta_angle >= 15)
            for j in np.flatnonzero(nearby):
                matched.append(dict(split=name, port=pairs.iloc[j].port, mmsi=pairs.iloc[j].mmsi,
                    class_id=int(pairs.iloc[j].class_id), gap_days=pairs.iloc[j].gap_days,
                    position_km=float(position_km[j]), relative_angle_change=float(delta_angle[j]),
                    **{outcome+'_change': float(y[j, k]) for k, outcome in enumerate(OUTCOMES)}))
            pairs.to_csv(OUT/(name+'_pairs.csv'), index=False)
        coverage.append(dict(split=name, observations=len(part), pairs=len(pairs),
            ships=pairs.mmsi.nunique() if len(pairs) else 0, classes=pairs.class_id.nunique() if len(pairs) else 0))
    pd.DataFrame(coverage).to_csv(OUT/'temporal_coverage.csv', index=False)
    pd.DataFrame(matched).to_csv(OUT/'nearby_matched_pairs.csv', index=False)
    results = []
    if 'fit' in packed and len(packed['fit'][0]) >= 20:
        train = packed['fit']
        for class_id in [-1, *sorted(train[0].class_id.unique())]:
            fit_mask = np.ones(len(train[0]), bool) if class_id == -1 else train[0].class_id.eq(class_id).to_numpy()
            if fit_mask.sum() < 20 or train[0].loc[fit_mask, 'mmsi'].nunique() < 5:
                continue
            weights = train[4][fit_mask]
            weights = weights/weights.sum()
            for mode, xi in [('incidence_position', 1), ('plus_relative_angle', 2)]:
                coef = fit_response(train[xi][fit_mask], train[3][fit_mask], weights, ridge=.01)
                np.save(OUT/f'coef_class{class_id}_{mode}.npy', coef)
                for name, test in packed.items():
                    mask = np.ones(len(test[0]), bool) if class_id == -1 else test[0].class_id.eq(class_id).to_numpy()
                    if not mask.any():
                        continue
                    w = test[4][mask]
                    w = w/w.sum()
                    pred = test[xi][mask] @ coef
                    for k, outcome in enumerate(OUTCOMES):
                        score = response_metrics(test[3][mask, k:k+1], pred[:, k:k+1], w)
                        results.append(dict(class_id=int(class_id), split=name, mode=mode, outcome=outcome,
                            pairs=int(mask.sum()), ships=test[0].loc[mask, 'mmsi'].nunique(), **score))
    pd.DataFrame(results).to_csv(OUT/'temporal_response_metrics.csv', index=False)
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)
    if results:
        result = pd.DataFrame(results)
        print(result[(result.class_id == -1) & (result.split != 'fit')][['split', 'mode', 'outcome', 'pairs', 'skill']].to_string(index=False), flush=True)
    return dict(eligible_instances=len(ids), observations=len(obs), matched_nearby_pairs=len(matched), response_rows=len(results))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed observability analysis')
    write_json(OUT/'PROTOCOL.json', dict(question='Do relative-angle changes predict visibility and classification changes within the same vessel?',
        angle_bins=ANGLE_BINS, outcomes=OUTCOMES,
        contrast='mean VV/VH (center grayq95-rim grayq50)/(center grayq95+rim grayq50+1); NOT calibrated RCS',
        query_labels='Used only to diagnose frozen reference predictions; no classifier fitting or parameter selection',
        temporal='Same port/MMSI adjacent distinct acquisitions, conflicting identities excluded',
        split='fixedseed20261009;1/3ships held;months3/6/9/12held; disjoint response-fit/held identities',
        controls='incidence polynomial and local east/north displacement',
        models='controls vs controls+cos2relative,sin2relative differences; fixed ridge.01',
        caveat='Proxy-axis selection and other time-varying factors remain; not causal identification. Prior features/classifiers not nested-held for these response splits.'))
    frame = observation_table()
    angle_tables(frame)
    result = temporal_probe(frame)
    write_json(OUT/'COMPLETE.json', result)


if __name__ == '__main__':
    main()
