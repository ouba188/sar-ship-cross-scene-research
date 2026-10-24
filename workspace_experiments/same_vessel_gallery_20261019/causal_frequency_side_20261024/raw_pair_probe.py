"""Recover paired native DN windows and test display effects on fixed readouts."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from osgeo import gdal
from scipy.stats import spearmanr

import joint_probe as jp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'round14_raw_pairs'
WORK = ROOT.parents[1]
SAFE = Path('E:/Docms/Port/00ports/Port Klang')
SUFFIXES = ('_122F', '_CEA9')


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def get_cohort():
    manifest = pd.read_csv(WORK / 'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv', low_memory=False)
    parts = [manifest[manifest['product'].str.endswith(s)].copy() for s in SUFFIXES]
    for p in parts:
        p['mmsi'] = p.mmsi.astype('int64').astype(str)
        assert not p.mmsi.duplicated().any()
    pairs = parts[0].merge(parts[1], on='mmsi', suffixes=('_a', '_b'))
    assert pairs.class_name_a.eq(pairs.class_name_b).all()
    gallery = json.loads((ROOT.parent / 'gallery_data.js').read_text(encoding='utf8').split('=', 1)[1].strip().removesuffix(';'))['observations']
    by_id = {o['instance_id']: o for o in gallery.values()}
    for e in ('a', 'b'):
        pairs[f'obs_id_{e}'] = pairs[f'instance_id_{e}'].map(lambda x: by_id[x]['id'])
    views = pd.read_csv(WORK / 'chip_quality_20261017/ais_metadata/view_metadata.csv', low_memory=False)
    ids = set(pairs.instance_id_a) | set(pairs.instance_id_b)
    views = views[views.instance_id.isin(ids)].copy()
    assert len(views) == 4 * len(pairs) and not views.duplicated(['instance_id', 'pol']).any()
    cols = ['mmsi', 'class_name_a', 'instance_id_a', 'instance_id_b', 'obs_id_a', 'obs_id_b', 'product_a', 'product_b']
    pairs[cols].to_csv(OUT / 'cohort.csv', index=False)
    return pairs, views, by_id


def recover(views, gallery):
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetCacheMax(64 * 1024 * 1024)
    arrays, records = {}, []
    cache = OUT / 'private_windows'
    cache.mkdir(exist_ok=True)
    for (product, pol), group in views.groupby(['product', 'pol']):
        paths = list((SAFE / f'{product}.SAFE/measurement').glob(f'*-{pol}-*.tiff'))
        assert len(paths) == 1
        ds = gdal.Open(str(paths[0]), gdal.GA_ReadOnly)
        assert ds.GetRasterBand(1).DataType == gdal.GDT_UInt16
        for r in group.itertuples():
            o = gallery[r.instance_id]
            oid = o['id']
            gt = json.loads(r.chip_transform_json)
            assert np.allclose([gt[1], gt[2], gt[4], gt[5]], [10, 0, 0, -10])
            png = np.asarray(Image.open(r.image_path).convert('L'), dtype=float) / 255.
            h, w = png.shape
            yy, xx = np.indices((h, w))
            mx = gt[0] + (xx + .5) * gt[1] + (yy + .5) * gt[2]
            my = gt[3] + (xx + .5) * gt[4] + (yy + .5) * gt[5]
            transform = gdal.Transformer(ds, None, [f'DST_SRS=EPSG:{int(r.epsg)}'])
            coords, success = transform.TransformPoints(True, np.c_[mx.ravel(), my.ravel(), np.zeros(mx.size)].tolist())
            assert all(success)
            sx, sy = np.floor(np.asarray(coords)[:, :2]).astype(int).T
            x0, y0, x1, y1 = int(sx.min()), int(sy.min()), int(sx.max()+1), int(sy.max()+1)
            assert 0 <= x0 < x1 <= ds.RasterXSize and 0 <= y0 < y1 <= ds.RasterYSize
            assert max(x1-x0, y1-y0) < 512
            raw = ds.ReadAsArray(x0, y0, x1-x0, y1-y0)
            dn = raw[sy-y0, sx-x0].reshape(h, w).astype(float)
            arrays[(oid, pol)] = dict(dn=dn, png=png, priors=o['images'][pol]['priors'])
            np.savez_compressed(cache / f'{oid}_{pol}.npz', dn=dn, png=png)
            rank = float(spearmanr(dn.ravel(), png.ravel()).statistic)
            # Translation sensitivity diagnoses mapping mismatch, not a new alignment fit.
            shifts = []
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                shifts.append(float(spearmanr(dn[1+dy:h-1+dy, 1+dx:w-1+dx].ravel(), png[1:-1, 1:-1].ravel()).statistic))
            records.append(dict(obs_id=oid, instance_id=r.instance_id, pol=pol, product=product,
                image_path=r.image_path, raw_path=str(paths[0]), mmsi=o['mmsi'], category=o['category'],
                width=w, height=h, source_width=x1-x0, source_height=y1-y0,
                rank_rho=rank, best_neighbor_shift_rho=max(shifts), png_sat_fraction=float((png >= 1).mean()),
                dn_max=float(dn.max()), has_axis=bool(o['images'][pol]['priors'].get('ship_axis')),
                prediction=o['predictions'].get('dual', {}).get('class_id'),
                correct=o['predictions'].get('dual', {}).get('correct'), calibrated=False))
        ds = None
        print(product[-4:], pol, len(group), 'windows recovered', flush=True)
    pd.DataFrame(records).to_csv(OUT / 'window_provenance.csv', index=False)
    return arrays, records


def masks(im, p):
    yy, xx = np.indices(im.shape)
    anchor = p['anchor']
    valid = (xx >= 6) & (xx < im.shape[1]-6) & (yy >= 6) & (yy < im.shape[0]-6)
    disk = ((xx+.5-anchor[0])**2 + (yy+.5-anchor[1])**2 <= 36) & valid
    result = {'anchor_disk60m': disk}
    if p.get('ship_axis'):
        s = p['ship_axis']
        result['obb_body'] = jp.masks_for(im, s, s['length_px'], s['width_px'])['body'] & valid
    return {k: v for k, v in result.items() if v.sum() >= 12}


def measure(im, region):
    ts = [jp.tensor(*jp.gradients(im, s), region) for s in (1., 2.)]
    energy = [np.trace(t) for t in ts]
    if min(energy) <= 1e-20:
        return None
    eig = np.linalg.eigvalsh(ts[0])
    return dict(log_ratio=float(np.log(energy[0]/energy[1])),
        anisotropy=float((eig[1]-eig[0])/eig.sum()), txx=float(ts[0][0, 0]/energy[0]),
        txy=float(ts[0][0, 1]/energy[0]), tyy=float(ts[0][1, 1]/energy[0]))


def analyze(pairs, arrays):
    rows, bounds = [], []
    for pair in pairs.itertuples():
        for pol in ('vv', 'vh'):
            ids = (pair.obs_id_a, pair.obs_id_b)
            pooled = np.concatenate([arrays[(i, pol)]['dn'].ravel() for i in ids])
            lo, q98, q995 = np.quantile(pooled, [.02, .98, .995])
            bounds.append(dict(mmsi=pair.mmsi, pol=pol, low=lo, high98=q98, high995=q995))
            for oid in ids:
                item = arrays[(oid, pol)]
                dn, p = item['dn'], item['priors']
                representations = dict(dn=dn, affine=2*dn+17,
                    clip98=np.clip((dn-lo)/(q98-lo), 0, 1),
                    clip995=np.clip((dn-lo)/(q995-lo), 0, 1), historical_png=item['png'])
                for region, mask in masks(dn, p).items():
                    for name, im in representations.items():
                        values = measure(im, mask)
                        if values is None:
                            continue
                        s = p.get('ship_axis')
                        angle = s['angle'] if s else np.nan
                        radar = p.get('radar_axis', np.nan)
                        rows.append(dict(mmsi=pair.mmsi, category=pair.class_name_a, obs_id=oid,
                            pol=pol, region=region, representation=name, n_pixels=int(mask.sum()),
                            angle=angle, radar=radar, cos2relative=float(np.cos(2*(angle-radar))),
                            incidence=p.get('radar_incidence', np.nan), **values))
    measures = pd.DataFrame(rows)
    measures.to_csv(OUT / 'measurements.csv', index=False)
    pd.DataFrame(bounds).to_csv(OUT / 'common_display_bounds.csv', index=False)
    out = []
    for pair in pairs.itertuples():
        d = measures[measures.mmsi.eq(pair.mmsi)]
        for (pol, region, rep), z in d.groupby(['pol', 'region', 'representation']):
            if len(z) != 2:
                continue
            z = z.set_index('obs_id')
            a, b = z.loc[pair.obs_id_a], z.loc[pair.obs_id_b]
            out.append(dict(mmsi=pair.mmsi, category=pair.class_name_a, a=pair.obs_id_a, b=pair.obs_id_b,
                pol=pol, region=region, representation=rep,
                delta_log_ratio=b.log_ratio-a.log_ratio, delta_anisotropy=b.anisotropy-a.anisotropy,
                delta_cos2relative=b.cos2relative-a.cos2relative, delta_incidence=b.incidence-a.incidence))
    diffs = pd.DataFrame(out)
    diffs.to_csv(OUT / 'paired_changes.csv', index=False)
    summaries = []
    for (pol, region), d in diffs.groupby(['pol', 'region']):
        pivot = d.pivot(index='mmsi', columns='representation', values='delta_log_ratio').dropna()
        for rep in ('affine', 'clip98', 'clip995', 'historical_png'):
            x, y = pivot.dn, pivot[rep]
            summaries.append(dict(pol=pol, region=region, representation=rep, n=len(x),
                median_abs_dn=float(x.abs().median()), median_abs_other=float(y.abs().median()),
                median_abs_change_error=float((x-y).abs().median()), max_abs_change_error=float((x-y).abs().max()),
                same_change_sign=int(((x*y)>0).sum()), spearman=float(spearmanr(x, y).statistic)))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / 'display_effect_summary.csv', index=False)
    affine = summary[summary.representation.eq('affine')].max_abs_change_error.max()
    assert affine < 1e-10, 'Scale descriptor must be invariant to a positive affine intensity transform'
    return measures, diffs, summary


def panels(pairs, arrays, gallery):
    from PIL import ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    selected = list(pairs.sort_values(['class_name_a', 'mmsi']).itertuples())
    for page in range((len(selected)+3)//4):
        chunk = selected[page*4:page*4+4]
        canvas = Image.new('RGB', (1440, 220*len(chunk)+36), 'white')
        draw = ImageDraw.Draw(canvas)
        titles = ['VV A PNG', 'VV A DN', 'VV B PNG', 'VV B DN', 'VH A PNG', 'VH A DN', 'VH B PNG', 'VH B DN']
        for c, title in enumerate(titles):
            draw.text((c*180+4, 4), title, fill='black', font=font)
        for row, pair in enumerate(chunk):
            ids = (pair.obs_id_a, pair.obs_id_b)
            y = 36+row*220
            states = [gallery[getattr(pair, f'instance_id_{e}')]['predictions'].get('dual', {}).get('correct') for e in ('a', 'b')]
            draw.text((4, y), f'{pair.mmsi} {pair.class_name_a} | {ids[0]}->{ids[1]} | archived dual correct {states}', fill='black', font=font)
            for pi, pol in enumerate(('vv', 'vh')):
                pool = np.concatenate([arrays[(i, pol)]['dn'].ravel() for i in ids])
                lo, hi = np.quantile(pool, [.02, .995])
                for ei, oid in enumerate(ids):
                    item = arrays[(oid, pol)]
                    for ri, name in enumerate(('png', 'dn')):
                        a = item[name]
                        if name == 'dn':
                            a = np.clip((a-lo)/(hi-lo), 0, 1)
                        tile = Image.fromarray((a*255).round().astype('uint8')).convert('RGB')
                        td = ImageDraw.Draw(tile)
                        p = item['priors']
                        if p.get('ship_axis'):
                            corners = p['detections'][p['target_index']]['corners']
                            td.line([tuple(v) for v in corners]+[tuple(corners[0])], fill=(30, 220, 100), width=1)
                        cx, cy = p['anchor']
                        td.ellipse((cx-6, cy-6, cx+6, cy+6), outline=(255, 160, 0), width=1)
                        assert tile.width <= 176 and tile.height <= 180
                        col = pi*4+ei*2+ri
                        canvas.paste(tile, (col*180+(180-tile.width)//2, y+30+(180-tile.height)//2))
        canvas.save(OUT / f'raw_pair_panel_{page+1:02d}.png')


def main():
    OUT.mkdir(exist_ok=True)
    pairs, views, gallery = get_cohort()
    arrays, provenance = recover(views, gallery)
    measures, diffs, summary = analyze(pairs, arrays)
    panels(pairs, arrays, gallery)
    p = pd.DataFrame(provenance)
    save('SUMMARY.json', dict(pairs=len(pairs), views=len(p),
        categories=pairs.class_name_a.value_counts().to_dict(),
        rank_rho_min=float(p.rank_rho.min()), rank_rho_median=float(p.rank_rho.median()),
        neighbor_shift_better_count=int((p.best_neighbor_shift_rho > p.rank_rho).sum()),
        body_axis_views=int(p.has_axis.sum()), independent_scene_pairs=1,
        raw_calibrated=False, trained=False, same_vessel_identity='MMSI matched; manual image review required'))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
