"""Polarization-separated native image evidence; no new classifier or selection."""
from pathlib import Path
import argparse
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Polygon
from scipy.stats import spearmanr
import tifffile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OSS = ROOT / 'oss1_joint_geometry_20261017'
PORTS = ROOT / 'obb_consistent_repair_20261016'
NAMES = ('body', 'near_ends', 'near_sides', 'far_ends', 'far_sides')


def regions(shape, cx, cy, length, width, theta):
    """Native integer pixel centers, unsigned axis; keep off-chip coverage explicit."""
    h, w = shape
    a, b = max(length * 1.25 / 2, 2), max(width * 1.25 / 2, 1.5)
    reach = int(np.ceil(3 * np.hypot(a, b))) + 2
    yy, xx = np.mgrid[int(np.floor(cy))-reach:int(np.ceil(cy))+reach+1,
                      int(np.floor(cx))-reach:int(np.ceil(cx))+reach+1]
    x, y = xx-cx, yy-cy
    u = (x*np.cos(theta)+y*np.sin(theta))/a
    v = (-x*np.sin(theta)+y*np.cos(theta))/b
    rho = np.maximum(abs(u), abs(v))
    # Pixel centers exactly on an axis/region boundary keep the same assignment after rotation.
    eps = 1e-10
    ends = abs(u) >= abs(v)-eps
    full = [rho <= 1+eps]
    for lo, hi in ((1, 2), (2, 3)):
        band = (rho > lo+eps) & (rho <= hi+eps)
        full.extend([band & ends, band & ~ends])
    visible = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    masks, coverage = {}, {}
    for name, f in zip(NAMES, full):
        m = np.zeros(shape, bool)
        selected = visible & f
        m[yy[selected], xx[selected]] = True
        masks[name] = m
        coverage[name] = float(m.sum()/f.sum()) if f.sum() else np.nan
    return masks, coverage


def summarize_image(image, masks, coverage, identity, representation):
    output = []
    for name, mask in masks.items():
        values = image[mask].astype(float)
        valid = values[np.isfinite(values)]
        row = dict(identity, representation=representation, region=name,
                   pixels=int(mask.sum()), finite_pixels=len(valid), coverage=coverage[name])
        if len(valid):
            row.update(mean=float(valid.mean()), median=float(np.median(valid)),
                       p95=float(np.quantile(valid, .95)),
                       zero_fraction=float((valid == 0).mean()),
                       negative_fraction=float((valid < 0).mean()),
                       saturated_fraction=float((valid == 255).mean()) if representation == 'uint8' else np.nan)
        output.append(row)
    return output


def read_oss1():
    meta = pd.read_csv(OSS/'geometry.csv', dtype={'mmsi': str})
    det = pd.read_csv(OSS/'detections.csv').set_index('global_row')
    stats, integrity = [], []
    for n, row in meta.iterrows():
        d = det.loc[row.global_row]
        if not d.axis_valid:
            continue
        cal = tifffile.imread(row.calibrated_patch_path)
        raw = tifffile.imread(row.raw_patch_path)
        shape = (int(d.native_h), int(d.native_w))
        if raw.shape != (*shape, 2) or cal.shape != (*shape, 2):
            raise ValueError(f'OSS1 native coordinates disagree at row {row.global_row}')
        masks, coverage = regions(shape, d.cx, d.cy, d.length, d.width, d.ship_axis)
        for pol, channel in (('vv', 1), ('vh', 0)):
            display = tifffile.imread(row['chip_path_'+pol])
            if display.shape != shape:
                raise ValueError('Polarization pixel correspondence changed')
            identity = dict(dataset='OSS1', global_row=int(row.global_row), mmsi=row.mmsi,
                            product=row['product'], class_id=int(row.class_id), pol=pol,
                            ship_axis_deg=float(np.degrees(d.ship_axis) % 180),
                            incidence_deg=row.incidence_deg, native_h=shape[0], native_w=shape[1])
            for representation, arr in (('sigma0', cal[:, :, channel]),
                                        ('amplitude', raw[:, :, channel]), ('uint8', display)):
                stats.extend(summarize_image(arr, masks, coverage, identity, representation))
            finite = np.isfinite(cal[:, :, channel]) & np.isfinite(raw[:, :, channel])
            c = cal[:, :, channel][finite]
            r = raw[:, :, channel][finite]
            disp = display[finite]
            integrity.append(dict(global_row=int(row.global_row), pol=pol,
                                  cal_vs_amplitude_rank=float(spearmanr(c, r).statistic),
                                  cal_vs_display_rank=float(spearmanr(c, disp).statistic),
                                  cal_negative_pixels=int((c < 0).sum()),
                                  finite_fraction=float(finite.mean())))
        if n % 300 == 0:
            print('OSS1', n, '/', len(meta), flush=True)
    pd.DataFrame(integrity).to_csv(HERE/'oss1_value_checks.csv', index=False)
    return pd.DataFrame(stats), meta


def read_ports():
    meta = pd.read_csv(PORTS/'inventory.csv', dtype={'mmsi': str})
    meta['mmsi'] = meta.mmsi.str.replace(r'\.0$', '', regex=True)
    det = {(d['global_row'], d['pol']): d for d in
           map(json.loads, (PORTS/'detections.jsonl').read_text().splitlines())}
    stats, exclusions = [], []
    for n, row in meta.iterrows():
        for pol in ('vv', 'vh'):
            d = det[(int(row.global_row), pol)]
            if not d['accepted']:
                exclusions.append(dict(global_row=row.global_row, pol=pol, reason='no_accepted_native_OBB'))
                continue
            cx, cy, length, width, theta = d['raw_xywhr']
            if width > length:
                length, width, theta = width, length, theta+np.pi/2
            with Image.open(row['image_relpath_'+pol]) as img:
                arr = np.array(img.convert('L'))
            if arr.shape != (d['native_h'], d['native_w']):
                raise ValueError('Native OBB and image shape disagree')
            masks, coverage = regions(arr.shape, cx, cy, length, width, theta)
            identity = dict(dataset='24port_local', global_row=int(row.global_row), mmsi=row.mmsi,
                            product=row['product'], class_id=int(row.class_id), pol=pol,
                            ship_axis_deg=float(np.degrees(theta) % 180), incidence_deg=np.nan,
                            native_h=arr.shape[0], native_w=arr.shape[1], port=row.port)
            stats.extend(summarize_image(arr, masks, coverage, identity, 'uint8'))
        if n % 600 == 0:
            print('24PORT', n, '/', len(meta), flush=True)
    pd.DataFrame(exclusions, columns=['global_row', 'pol', 'reason']).to_csv(HERE/'ports_exclusions.csv', index=False)
    return pd.DataFrame(stats), meta


def descriptors(stats):
    keys = ['dataset', 'global_row', 'pol', 'representation']
    d = stats.pivot(index=keys, columns='region', values='mean').reset_index()
    meta = stats.drop_duplicates(keys)[keys+['mmsi', 'product', 'class_id', 'ship_axis_deg', 'incidence_deg']]
    d = d.merge(meta, on=keys, validate='one_to_one')
    coverage = stats.pivot(index=keys, columns='region', values='coverage').reset_index()
    coverage['min_outside_coverage'] = coverage[list(NAMES[1:])].min(axis=1)
    d = d.merge(coverage[keys+['min_outside_coverage']], on=keys, validate='one_to_one')
    # Equal region weighting measures a spatial pattern, not total emitted energy.
    neighbor = d[list(NAMES[1:])].mean(axis=1, skipna=False)
    d['body_neighbor_contrast'] = (d.body-neighbor)/(d.body+neighbor).replace(0, np.nan)
    ends = d[['near_ends', 'far_ends']].mean(axis=1, skipna=False)
    sides = d[['near_sides', 'far_sides']].mean(axis=1, skipna=False)
    d['ends_sides_balance'] = (ends-sides)/(ends+sides).replace(0, np.nan)
    d['body_neighbor_db'] = np.where((d.representation == 'sigma0') & (d.body > 0) & (neighbor > 0),
                                    10*np.log10(d.body/neighbor), np.nan)
    q = stats.query("region == 'body'")[keys+['saturated_fraction']]
    d = d.merge(q, on=keys, validate='one_to_one')
    return d


def paired_polarization(d):
    a = d[d.pol.eq('vv')].set_index(['dataset', 'global_row', 'representation'])
    b = d[d.pol.eq('vh')].set_index(['dataset', 'global_row', 'representation'])
    ids = a.index.intersection(b.index)
    rows = []
    for key in ids:
        v, h = a.loc[key], b.loc[key]
        row = dict(zip(['dataset', 'global_row', 'representation'], key))
        row.update(mmsi=v.mmsi, product=v['product'], class_id=v.class_id,
                   contrast_vv=v.body_neighbor_contrast, contrast_vh=h.body_neighbor_contrast,
                   balance_vv=v.ends_sides_balance, balance_vh=h.ends_sides_balance,
                   min_coverage=min(v.min_outside_coverage, h.min_outside_coverage),
                   dominant_region_vv=v[list(NAMES[1:])].astype(float).idxmax(),
                   dominant_region_vh=h[list(NAMES[1:])].astype(float).idxmax())
        complete = np.isfinite(v[list(NAMES[1:])].to_numpy(float)).all() and np.isfinite(h[list(NAMES[1:])].to_numpy(float)).all()
        row['complete_regions'] = complete
        row['different_dominant_region'] = float(row['dominant_region_vv'] != row['dominant_region_vh']) if complete else np.nan
        if key[2] == 'sigma0':
            row['body_vv_over_vh_db'] = 10*np.log10(v.body/h.body) if v.body > 0 and h.body > 0 else np.nan
        rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(HERE/'paired_polarization.csv', index=False)
    summary = []
    for (dataset, representation), group in paired.groupby(['dataset', 'representation']):
        for label, g in [('all', group), ('coverage_ge_80pct', group[group.min_coverage >= .8])]:
            summary.append(dict(dataset=dataset, representation=representation, subset=label,
                                samples=len(g), products=g['product'].nunique(), ships=g.mmsi.nunique(),
                                complete_region_pairs=int(g.complete_regions.sum()),
                                different_region_fraction=g.different_dominant_region.mean(),
                                median_abs_contrast_gap=(g.contrast_vv-g.contrast_vh).abs().median(),
                                median_abs_balance_gap=(g.balance_vv-g.balance_vh).abs().median(),
                                contrast_rank_correlation=g.contrast_vv.corr(g.contrast_vh, method='spearman')))
    result = pd.DataFrame(summary)
    result.to_csv(HERE/'POLARIZATION_SUMMARY.csv', index=False)
    paired.groupby(['dataset', 'representation', 'class_id']).agg(
        samples=('global_row', 'size'), products=('product', 'nunique'),
        different_region_fraction=('different_dominant_region', 'mean'),
        contrast_vv=('contrast_vv', 'median'), contrast_vh=('contrast_vh', 'median')).to_csv(HERE/'polarization_by_class.csv')
    return paired, result


def same_vessel(d):
    records = []
    for (dataset, pol, rep, mmsi), g in d.groupby(['dataset', 'pol', 'representation', 'mmsi']):
        if g.class_id.nunique() != 1 or g['product'].nunique() < 2:
            continue
        # Product token 4 is UTC acquisition start, including S1A/S1B/S1C.
        g = g.assign(time=g['product'].str.split('_').str[4]).sort_values(['time', 'global_row']).drop_duplicates('product')
        a, b = g.iloc[0], g.iloc[-1]
        row = dict(dataset=dataset, pol=pol, representation=rep, mmsi=mmsi,
                   class_id=int(a.class_id), first=int(a.global_row), second=int(b.global_row),
                   first_product=a['product'], second_product=b['product'],
                   delta_contrast=b.body_neighbor_contrast-a.body_neighbor_contrast,
                   delta_balance=b.ends_sides_balance-a.ends_sides_balance,
                   delta_incidence=b.incidence_deg-a.incidence_deg,
                   axis_change_deg=abs((b.ship_axis_deg-a.ship_axis_deg+90) % 180-90),
                   min_coverage=min(a.min_outside_coverage, b.min_outside_coverage))
        records.append(row)
    out = pd.DataFrame(records)
    out.to_csv(HERE/'same_vessel_pairs.csv', index=False)
    out.groupby(['dataset', 'representation', 'pol']).agg(
        ships=('mmsi', 'nunique'), contrast_pairs=('delta_contrast', 'count'), balance_pairs=('delta_balance', 'count'),
        median_abs_contrast_change=('delta_contrast', lambda x: x.abs().median()),
        median_abs_balance_change=('delta_balance', lambda x: x.abs().median()),
        median_axis_change_deg=('axis_change_deg', 'median')).to_csv(HERE/'SAME_VESSEL_SUMMARY.csv')
    return out


def link_decisions(d):
    pred = pd.read_csv(OSS/'CASE_DIAGNOSTICS.csv')
    pred = pred[(pred.arm == 'joint') & (pred.policy == 'source_selected') & (pred.origin == 'cross')]
    # Each query contributes once after averaging the repeated support/seed outcomes.
    per_query = pred.groupby(['fold', 'global_row', 'truth']).agg(
        net=('net_base', 'mean'), trials=('net_base', 'size'),
        rescue_rate=('net_base', lambda v: (v > 0).mean()),
        harm_rate=('net_base', lambda v: (v < 0).mean())).reset_index()
    x = d[(d.dataset == 'OSS1') & (d.representation == 'sigma0')].pivot(
        index='global_row', columns='pol', values=['body_neighbor_db', 'ends_sides_balance', 'min_outside_coverage'])
    x.columns = ['_'.join(c) for c in x.columns]
    merged = per_query.merge(x, on='global_row', how='left', validate='one_to_one')
    merged['outcome'] = np.select([merged.net > 0, merged.net < 0], ['net_rescued', 'net_harmed'], 'net_unchanged')
    merged.to_csv(HERE/'existing_query_outcomes.csv', index=False)
    merged.groupby(['fold', 'outcome']).agg(
        queries=('global_row', 'size'), classes=('truth', 'nunique'),
        contrast_vv_db=('body_neighbor_db_vv', 'median'), contrast_vh_db=('body_neighbor_db_vh', 'median'),
        balance_vv=('ends_sides_balance_vv', 'median'), balance_vh=('ends_sides_balance_vh', 'median')).to_csv(HERE/'outcome_groups.csv')
    return merged


def geometry_associations(d):
    rows = []
    # Descriptive residualized correlations, not causal coefficients or training targets.
    for pol, group in d[(d.dataset == 'OSS1') & (d.representation == 'sigma0')].groupby('pol'):
        group = group.copy()
        group['unsigned_ship_range_angle'] = np.degrees(np.arccos(np.abs(np.cos(np.radians(group.ship_axis_deg)))))
        for condition in ('incidence_deg', 'unsigned_ship_range_angle'):
            for evidence in ('body_neighbor_db', 'ends_sides_balance'):
                g = group.dropna(subset=[condition, evidence])
                covariates = pd.get_dummies(g[['product', 'class_id']].astype(str), drop_first=True, dtype=float)
                x = np.column_stack([np.ones(len(g)), covariates.to_numpy()])
                values = g[[condition, evidence]].rank().to_numpy()
                residual = values-x@np.linalg.lstsq(x, values, rcond=None)[0]
                rows.append(dict(pol=pol, condition=condition, evidence=evidence, samples=len(g),
                                 products=g['product'].nunique(), ships=g.mmsi.nunique(),
                                 raw_rank_correlation=np.corrcoef(values.T)[0, 1],
                                 product_class_partial_rank_correlation=np.corrcoef(residual.T)[0, 1],
                                 condition_within_product_std=g.groupby('product')[condition].std().median()))
    pd.DataFrame(rows).to_csv(HERE/'geometry_associations.csv', index=False)


def oss2_inventory():
    path = Path('E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/target_manifest.csv')
    meta = pd.read_csv(path)
    rows = []
    for product, g in meta.groupby('scene_id'):
        folder = Path(g.iloc[0].source_path_vv).parents[1]
        def index(sub):
            found = {}
            for p in (folder/sub).glob('*.tif'):
                match = re.search(r'_x(\d+)_y(\d+)\.tif$', p.name, re.I)
                if match:
                    key = match.groups()
                    if key in found:
                        raise ValueError('Ambiguous OSS2 native chip identity')
                    found[key] = str(p)
            return found
        raw, cal = index('Patch'), index('Patch_Cal')
        for _, r in g.iterrows():
            match = re.search(r'_x(\d+)_y(\d+)_vv\.tif$', Path(r.source_path_vv).name, re.I)
            key = match.groups() if match else None
            rows.append(dict(sample_id=r.sample_id, product=product, product_type=product.split('_')[2],
                             class_id=r.class_id, mmsi=r.mmsi_or_ship_id, source_time=r.source_time,
                             lon=r.lon, lat=r.lat, raw_path=raw.get(key, ''), calibrated_path=cal.get(key, ''),
                             vv_exists=Path(r.chip_path_vv).is_file(), vh_exists=Path(r.chip_path_vh).is_file()))
    out = pd.DataFrame(rows)
    out.to_csv(HERE/'oss2_native_inventory.csv', index=False)
    return out


def plots(oss_meta, ports_meta, paired, repeated, query):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, dataset, rep in zip(axes, ['OSS1', '24port_local'], ['sigma0', 'uint8']):
        q = paired[(paired.dataset == dataset) & (paired.representation == rep)]
        ax.scatter(q.balance_vv, q.balance_vh, s=5, alpha=.25)
        ax.plot([-1, 1], [-1, 1], color='black', lw=.8)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel='VV: (ends - sides)/(ends + sides)',
               ylabel='VH: (ends - sides)/(ends + sides)', title=f'{dataset}, {rep}, n={len(q)}')
    fig.savefig(HERE/'polarization_patterns.png', dpi=160)
    plt.close(fig)
    # Fixed examples from the existing, outcome-blind inspection; not chosen for gains.
    selected = pd.read_csv(ROOT/'raw_data_rethink_20261017/same_vessel_selection.csv')
    detections = {(r['global_row'], r['pol']): r for r in
                  map(json.loads, (PORTS/'detections.jsonl').read_text().splitlines())}
    pd.DataFrame(selected).to_csv(HERE/'gallery_selection.csv', index=False)
    for page, start in enumerate(range(0, len(selected), 4), 1):
        fig, axes = plt.subplots(4, 4, figsize=(11, 11), layout='constrained')
        for i, r in enumerate(selected.iloc[start:start+4].itertuples()):
            for j, (key, pol) in enumerate([('first', 'vv'), ('first', 'vh'), ('second', 'vv'), ('second', 'vh')]):
                row = ports_meta.set_index('global_row').loc[getattr(r, key)]
                with Image.open(row['image_relpath_'+pol]) as im:
                    arr = np.array(im.convert('L'))
                axes[i, j].imshow(arr, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
                detection = detections[(int(row.name), pol)]
                if detection['accepted']:
                    cx, cy, w, h, theta = detection['raw_xywhr']
                    corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])*1.25
                    rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    axes[i, j].add_patch(Polygon(corners@rotate.T+[cx, cy], fill=False, edgecolor='lime', linewidth=.7))
                axes[i, j].set_title(f'MMSI {r.mmsi} | {key} {pol.upper()}\nrow {row.name} | class {r.class_id} | {arr.shape}', fontsize=8)
                axes[i, j].set_axis_off()
        fig.suptitle('24-port repeated vessel observations; fixed display scale; separate native views', fontsize=11)
        fig.savefig(HERE/f'ports_same_vessel_{page}.png', dpi=150)
        plt.close(fig)
    # Same pixels before/after stretch in the two already-discussed decision cases.
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout='constrained')
    om = oss_meta.set_index('global_row')
    for i, grow in enumerate([1038, 591]):
        row = om.loc[grow]
        cal = tifffile.imread(row.calibrated_patch_path)
        for j, (pol, channel) in enumerate([('vv', 1), ('vh', 0)]):
            a = cal[:, :, channel]
            db = np.full(a.shape, np.nan)
            db[a > 0] = 10*np.log10(a[a > 0])
            axes[i, 2*j].imshow(tifffile.imread(row['chip_path_'+pol]), cmap='gray', vmin=0, vmax=255)
            axes[i, 2*j+1].imshow(db, cmap='gray', vmin=-35, vmax=10)
            axes[i, 2*j].set_title(f'row {grow} | {pol.upper()} uint8 [0,255]', fontsize=9)
            axes[i, 2*j+1].set_title(f'row {grow} | {pol.upper()} sigma0 dB [-35,10]', fontsize=9)
        for ax in axes[i]:
            ax.set_axis_off()
    fig.suptitle('Existing rescue/harm cases: fixed radiometric display, not new classification results', fontsize=11)
    fig.savefig(HERE/'oss1_before_after_display.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    if args.reuse:
        stats = pd.read_csv(HERE/'native_region_statistics.csv', dtype={'mmsi': str, 'port': str})
        om = pd.read_csv(OSS/'geometry.csv')
        pm = pd.read_csv(PORTS/'inventory.csv')
        inventory = pd.read_csv(HERE/'oss2_native_inventory.csv')
    else:
        a, om = read_oss1()
        b, pm = read_ports()
        stats = pd.concat([a, b], ignore_index=True)
        stats.to_csv(HERE/'native_region_statistics.csv', index=False)
        inventory = oss2_inventory()
    d = descriptors(stats)
    d.to_csv(HERE/'image_descriptors.csv', index=False)
    paired, summary = paired_polarization(d)
    repeated = same_vessel(d)
    query = link_decisions(d)
    geometry_associations(d)
    plots(om, pm, paired, repeated, query)
    stats.groupby(['dataset', 'representation', 'pol', 'region']).agg(
        samples=('global_row', 'size'), median_value=('mean', 'median'),
        median_coverage=('coverage', 'median'), median_saturation=('saturated_fraction', 'median')).to_csv(HERE/'REGION_SUMMARY.csv')
    status = dict(date='2026-10-17', new_classifier_fits=0, datasets_measured=['OSS1', '24port_local'],
                  oss1_manifest_rows=len(om), ports_local_rows=len(pm),
                  oss2_inventory_rows=len(inventory), oss2_products=int(inventory['product'].nunique()),
                  interpretation='Descriptive/exploratory; no physical factor or classification gain identified by correlation alone',
                  region_definition='Native OBB times 1.25, homothetic shells 1-2 and 2-3, unsigned ends/sides; no claim of optimal region design',
                  polar_registration='OSS1 shared native grid; 24port independent per-pol native OBB, no pixelwise ratios',
                  full_model_unchanged=True, inference_external_weather=False)
    (HERE/'IMAGE_ANALYSIS_DONE.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
