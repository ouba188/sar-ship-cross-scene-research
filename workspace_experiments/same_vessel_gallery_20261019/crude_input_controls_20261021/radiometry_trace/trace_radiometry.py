"""Bounded radiometry provenance audit; never decode a whole scene or fit a model."""
import argparse
import ast
import csv
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
GALLERY = OUT.parents[1]
ROOT = GALLERY.parent
DATA = Path('E:/SAR_AIS_T_shipchip_dataset')
META = DATA / 'v5_context_dynamic_64_128_return_20260915/metadata'
FOCUS = {32042, 32043, 30233, 30237}
PRODUCT_RE = re.compile(r'S1[ABC]_IW_GRDH_1SDV_\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6}_[0-9A-F]{4}')
SCAN_ROOTS = [Path(x) for x in (
    'F:/SAR_AIS_T', 'F:/SAR_AIS_T_ASF_missing_opposite_61_20260812',
    'E:/tif_local_done', 'E:/safe841_local_20260918',
    'E:/SAR_AIS_T_zip_without_8bit_20260703_103258',
    'E:/SAR_AIS_T_zip_archive_from_F', 'E:/0904_linshi_SAR',
    'D:/Documents/Port', 'E:/Docms/Port',
    'E:/SAR_AIS_T_server2_large_scene_png_20260917',
    'E:/Hermes/paper_tracking/port_ais_2024_2025_from_zip/products_by_primary_port',
)]


def read_csv(path):
    return pd.read_csv(path, low_memory=False).fillna('')


def save_csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def basename(path):
    return str(path).replace('\\', '/').rsplit('/', 1)[-1]


def stats(a):
    x = np.asarray(a).astype(float).ravel()
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    return dict(n=len(x), min=float(x.min()), max=float(x.max()), mean=float(x.mean()),
                std=float(x.std()), p02=float(np.quantile(x, .02)),
                p50=float(np.median(x)), p98=float(np.quantile(x, .98)),
                zero_fraction=float(np.mean(x == 0)), at255_fraction=float(np.mean(x == 255)))


def polygon(size, corners):
    a = Image.new('1', size)
    ImageDraw.Draw(a).polygon([tuple(x) for x in corners], fill=1)
    return np.asarray(a, dtype=bool)


def prepare():
    pairs = read_csv(GALLERY / 'condition_outcomes_20261021/crude_pair_details.csv')
    cohort = read_csv(GALLERY / 'conditional_evidence_pilot/cohort.csv')
    ids = set(pairs.obs_id_a) | set(pairs.obs_id_b)
    assert len(pairs) == 100 and len(ids) == 200 and pairs.mmsi.nunique() == 100
    cohort = cohort[cohort.obs_id.isin(ids)].copy()
    assert len(cohort) == 200 and cohort.obs_id.is_unique and cohort.instance_id.is_unique
    assert cohort.class_name.eq('crude_oil_tanker').all()
    v = read_csv(ROOT / 'chip_quality_20261017/ais_metadata/view_metadata.csv')
    v = v[v.instance_id.isin(cohort.instance_id)].copy()
    assert len(v) == 400 and not v.duplicated(['instance_id', 'pol']).any()
    returned = {str(p): read_csv(p) for p in sorted(META.glob('server*/manifest.csv'))}
    original = read_csv(ROOT / 'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv')
    views = read_csv(GALLERY / 'paired_relation_probe_20261020/data_review/main_pair_views.csv').set_index(['obs_id', 'pol'])
    snapshot = {x['obs_id']: x for x in json.loads((GALLERY / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if x['obs_id'] in ids}
    old_stats = read_csv(GALLERY / 'condition_outcomes_20261021/image_statistics.csv').set_index(['obs_id', 'pol'])
    co = cohort.set_index('instance_id')
    records, pixels, matches = [], [], []
    for row in v.to_dict('records'):
        c = co.loc[row['instance_id']]
        oid, pol = int(c.obs_id), row['pol']
        o = original.iloc[int(c.manifest_row)]
        assert o.instance_id == row['instance_id'] and o['product'] == c['product']
        assert int(o.mmsi) == int(c.mmsi)
        assert row['image_path'] == c['image_' + pol] == views.loc[(oid, pol), 'image_path']
        assert row['source_link'] == 'exact_returned_crop_manifest'
        manifest = returned[row['source_link_evidence']]
        m = manifest[(manifest.obj_uid == row['obj_uid']) & (manifest.source_tiff == row['source_tiff'])]
        assert len(m) == 1, (oid, pol, len(m))
        m = m.iloc[0]
        assert m['product'] == c['product'] and int(m.mmsi) == int(c.mmsi)
        assert basename(m.context_output_png) == basename(row['image_path'])
        for k in ['crop_left', 'crop_top', 'crop_side', 'target_center_x', 'target_center_y']:
            assert float(row[k]) == float(m[k]), (oid, pol, k)
        with Image.open(row['image_path']) as im:
            a = np.asarray(im)
            mode, info = im.mode, dict(im.info)
        assert a.ndim == 2 and a.shape == (int(m.crop_bottom-m.crop_top), int(m.crop_right-m.crop_left))
        pr = snapshot[oid]['images'][pol]['priors']
        t = polygon((a.shape[1], a.shape[0]), pr['detections'][pr['target_index']]['corners'])
        other = np.zeros_like(t)
        for j, d in enumerate(pr['detections']):
            if j != pr['target_index'] and d.get('shape_usable') and d['confidence'] >= .25:
                other |= polygon((a.shape[1], a.shape[0]), d['corners'])
        rest = ~(t | other)
        st = old_stats.loc[(oid, pol)]
        assert t.sum() == st.target_pixels and rest.sum() == st.rest_pixels
        np.testing.assert_allclose([a[t].mean()/255, a[rest].mean()/255], [st.target_mean, st.rest_mean], rtol=0, atol=1e-12)
        for label, mask in [('all', np.ones_like(t)), ('target', t), ('rest', rest)]:
            pixels.append(dict(obs_id=oid, pol=pol, region=label, dtype=str(a.dtype), **stats(a[mask])))
        # Compare actual pixels of any retained path aliases, not just filenames.
        aliases = [Path(o['image_relpath_' + pol])]
        for base in [DATA/'v5_context_dynamic_64_128_merged_20260915/images',
                     DATA/'v5_context_dynamic_64_128_merged_20260915/images/return_20260915',
                     DATA/'v5_context_dynamic_64_128_return_20260915/images']:
            aliases.append(base / c.port / c.class_name / basename(row['image_path']))
        for p in sorted(set(aliases)):
            if not p.is_file() or p == Path(row['image_path']):
                continue
            with Image.open(p) as im:
                b = np.asarray(im)
            matches.append(dict(obs_id=oid, pol=pol, comparison='png_alias', path=str(p),
                                same_shape=a.shape == b.shape, exact_pixels=np.array_equal(a,b)))
        row.update(obs_id=oid, manifest_row=int(c.manifest_row), png_dtype=str(a.dtype), png_mode=mode,
                   png_info=json.dumps(info, default=str), crop_right=int(m.crop_right), crop_bottom=int(m.crop_bottom),
                   original_context_output_png=m.context_output_png, focus=oid in FOCUS,
                   crop_manifest_identity_verified=True,
                   calibration_status='display_uint8_not_physical_backscatter',
                   prestretch_float_chip_found=False, stretch_actual_lo='', stretch_actual_hi='')
        records.append(row)
    save_csv('endpoint_cohort.csv', cohort)
    save_csv('view_trace.csv', records)
    save_csv('png_statistics.csv', pixels)
    save_csv('png_alias_pixel_checks.csv', matches)
    selected = [x for x in records if x['focus']]
    save_json('focus_provenance.json', selected)
    save_json('prepare_summary.json', dict(pairs=100, endpoints=200, views=400,
        products=int(cohort['product'].nunique()), ports=int(cohort.port.nunique()),
        manifest_linked=400, png_regions_replayed=400, png_alias_comparisons=len(matches),
        png_alias_exact=sum(x['exact_pixels'] for x in matches)))
    print('Prepared 200 endpoints / 400 views; all original manifests and existing PNG region statistics agree.', flush=True)


def inventory():
    cohort = read_csv(OUT / 'endpoint_cohort.csv')
    wanted = set(cohort['product'])
    found, roots = [], []
    for root in SCAN_ROOTS:
        count = 0
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if not fn.lower().endswith(('.tif','.tiff','.zip','.json','.xml','.safe','.npy','.npz','.aux.xml')):
                    continue
                full = Path(dp) / fn
                match = PRODUCT_RE.search(str(full))
                if match and match.group() in wanted:
                    found.append(dict(product=match.group(), path=str(full), bytes=full.stat().st_size,
                                      suffix=full.suffix.lower(), scan_root=str(root)))
                    count += 1
        roots.append(dict(path=str(root), exists=root.exists(), matched_files=count))
        print('inventory', root, count, flush=True)
    # Recorded paths are evidence of historical availability, not proof of availability now.
    catalog = read_csv(ROOT / 'safe841_redownload_plan_20260918/product_inventory_841.csv')
    catalog = catalog[catalog.product_id.isin(wanted)]
    save_csv('historical_product_inventory.csv', catalog)
    save_csv('local_asset_inventory.csv', found)
    save_json('local_search_roots.json', roots)
    archives = []
    for row in found:
        if row['suffix'] != '.zip':
            continue
        try:
            with zipfile.ZipFile(row['path']) as z:
                members = [dict(name=x.filename, bytes=x.file_size, compressed_bytes=x.compress_size,
                                compression=x.compress_type) for x in z.infolist()
                           if '/measurement/' in x.filename or '/calibration/' in x.filename or x.filename.endswith('manifest.safe')]
            archives.append(dict(path=row['path'], product=row['product'], members=members,
                                 policy='central_directory_only_no_raster_decode'))
        except (OSError, zipfile.BadZipFile) as exc:
            archives.append(dict(path=row['path'], product=row['product'], error=str(exc)))
    save_json('safe_archive_headers.json', archives)


def local_windows():
    from osgeo import gdal, osr
    from scipy.stats import spearmanr
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    assets = read_csv(OUT / 'local_asset_inventory.csv')
    views = read_csv(OUT / 'view_trace.csv')
    headers, comparisons, raw_stats, raw_maps = [], [], [], []
    datasets = {}
    for row in assets[assets.suffix.isin(['.tif','.tiff'])].itertuples():
        ds = gdal.Open(row.path, gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        match = re.search(r'(?:_|-)(VV|VH)(?:_|-)', row.path, re.I)
        if not match:
            continue
        pol = match.group(1).lower()
        h = dict(path=row.path, product=row.product, pol=pol, width=ds.RasterXSize,
                 height=ds.RasterYSize, dtype=gdal.GetDataTypeName(band.DataType),
                 block_size=band.GetBlockSize(), geotransform=ds.GetGeoTransform(can_return_null=True),
                 projection=ds.GetProjection(), gcp_count=ds.GetGCPCount(), gcp_projection=ds.GetGCPProjection(),
                 metadata=ds.GetMetadata(), band_metadata=band.GetMetadata(), scale=band.GetScale(),
                 offset=band.GetOffset(), unit=band.GetUnitType(), nodata=band.GetNoDataValue())
        headers.append(h)
        datasets[row.path] = h
        ds = None
    save_json('local_raster_headers.json', headers)
    snapshot = {x['obs_id']: x for x in json.loads((GALLERY/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    (OUT/'recovered_windows').mkdir(exist_ok=True)
    for r in views.itertuples():
        candidates = [h for h in headers if h['product'] == r.product and h['pol'] == r.pol]
        if not candidates:
            continue
        gt = json.loads(r.chip_transform_json)
        width, height = int(r.png_width), int(r.png_height)
        with Image.open(r.image_path) as im:
            png = np.asarray(im)
        for h in candidates:
            ds = gdal.Open(h['path'], gdal.GA_ReadOnly)
            common = dict(obs_id=r.obs_id, pol=r.pol, path=h['path'], source_dtype=h['dtype'])
            if h['geotransform'] is not None and h['projection']:
                crs = osr.SpatialReference(wkt=h['projection'])
                epsg = crs.GetAuthorityCode(None)
                g = h['geotransform']
                inv = gdal.InvGeoTransform(g)
                x, y = gdal.ApplyGeoTransform(inv, gt[0], gt[3])
                xx, yy = int(round(x)), int(round(y))
                if not (0 <= xx <= h['width']-width and 0 <= yy <= h['height']-height):
                    ds = None
                    continue
                same_grid = (epsg == str(int(r.epsg)) and np.allclose([g[1],g[2],g[4],g[5]], [gt[1],gt[2],gt[4],gt[5]],atol=1e-8,rtol=0)
                             and max(abs(x-xx),abs(y-yy)) < 1e-5)
                result = dict(common, type='geocoded_uint8_window', xoff=xx,yoff=yy,width=width,height=height,
                              same_grid=bool(same_grid), lattice_error_pixels=max(abs(x-xx),abs(y-yy)))
                if same_grid:
                    a = ds.ReadAsArray(xx, yy, width, height)
                    result.update(exact_pixels=bool(np.array_equal(a,png)),
                                  unequal_pixels=int(np.sum(a != png)),mae=float(np.abs(a.astype(float)-png).mean()),
                                  max_abs_error=float(np.abs(a.astype(float)-png).max()))
                    raw_stats.append(dict(common, representation='already_uint8_window', **stats(a)))
                    np.save(OUT/'recovered_windows'/f'{r.obs_id}_{r.pol}_uint8.npy', a)
                else:
                    result.update(exact_pixels=False, reason='different_grid_no_pixel_comparison')
                comparisons.append(result)
            elif h['gcp_count']:
                # Map chip pixel centers through the source GCP transformer, then read
                # only the enclosing native measurement rectangle. No Warp/whole read.
                transformer = gdal.Transformer(ds, None, [f'DST_SRS=EPSG:{int(r.epsg)}'])
                yy, xx = np.mgrid[:height,:width]
                mx = gt[0] + (xx+.5)*gt[1] + (yy+.5)*gt[2]
                my = gt[3] + (xx+.5)*gt[4] + (yy+.5)*gt[5]
                coords, success = transformer.TransformPoints(True, np.c_[mx.ravel(),my.ravel(),np.zeros(mx.size)].tolist())
                assert all(success), (r.obs_id,r.pol,'GCP transform failed')
                coords = np.asarray(coords)
                sx = np.floor(coords[:,0]).astype(int)
                sy = np.floor(coords[:,1]).astype(int)
                x0,y0,x1,y1 = int(sx.min()),int(sy.min()),int(sx.max()+1),int(sy.max()+1)
                assert 0 <= x0 < x1 <= h['width'] and 0 <= y0 < y1 <= h['height']
                assert x1-x0 <= 256 and y1-y0 <= 256, 'Refuse unexpectedly large source read'
                a = ds.ReadAsArray(x0,y0,x1-x0,y1-y0)
                mapped = a[sy-y0,sx-x0].reshape(height,width)
                np.save(OUT/'recovered_windows'/f'{r.obs_id}_{r.pol}_dn_source.npy', a)
                np.save(OUT/'recovered_windows'/f'{r.obs_id}_{r.pol}_dn_nearest.npy', mapped)
                mask = (png>0)&(png<255)
                slope,intercept = np.linalg.lstsq(np.c_[mapped[mask].astype(float),np.ones(mask.sum())],png[mask],rcond=None)[0]
                display = np.clip(np.floor(mapped*slope+intercept+.5),0,255)
                err = np.abs(display-png)
                # These fitted coefficients diagnose correspondence; they are NOT
                # the historical scene stretch bounds and never replace unknowns.
                result = dict(common,type='raw_SAFE_GCP_nearest',xoff=x0,yoff=y0,width=x1-x0,height=y1-y0,
                    mapped_width=width,mapped_height=height,unique_source_pixels=len(set(zip(sx,sy))),
                    exact_pixels=False, reason='raw_DN_vs_display_requires_unknown_stretch_and_original_warp',
                    rank_rho=float(spearmanr(mapped.ravel(),png.ravel()).statistic),
                    fitted_display_slope=float(slope),fitted_display_intercept=float(intercept),
                    fitted_display_mae=float(err.mean()),fitted_display_exact_fraction=float(np.mean(err==0)),
                    fitted_display_within1_fraction=float(np.mean(err<=1)),
                    source_requested_pixels=int(a.size), estimated_strip_decode_bytes=int((y1-y0)*h['width']*2),
                    calibrated=False, alignment_status='GCP_coordinate_candidate_not_exact_original_warp_replay')
                comparisons.append(result)
                pr = snapshot[r.obs_id]['images'][r.pol]['priors']
                t = polygon((width,height),pr['detections'][pr['target_index']]['corners'])
                other = np.zeros_like(t)
                for j,d in enumerate(pr['detections']):
                    if j != pr['target_index'] and d.get('shape_usable') and d['confidence']>=.25:
                        other |= polygon((width,height),d['corners'])
                for label,mask in [('all',np.ones_like(t)),('target',t),('rest',~(t|other))]:
                    raw_stats.append(dict(common,representation='uncalibrated_uint16_DN_mapped_nearest',region=label,**stats(mapped[mask])))
                raw_maps.append(dict(obs_id=r.obs_id,pol=r.pol,product=r.product,path=h['path'],
                    chip_transform=gt,epsg=int(r.epsg),source_window=[x0,y0,x1-x0,y1-y0],
                    transform_method='GDAL default GCP transformer exact point evaluation; floor nearest pixel centers',
                    qualification='spatial candidate; historical full-scene approximate transformer was not replayed'))
                print('raw window', r.obs_id,r.pol, (x1-x0,y1-y0),'rho',result['rank_rho'],flush=True)
            ds = None
    save_csv('source_pixel_comparisons.csv', comparisons)
    save_csv('source_window_statistics.csv', raw_stats)
    save_json('raw_window_mappings.json', raw_maps)
    print('Local window checks',len(comparisons),'raw SAFE candidates',len(raw_maps),flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'inventory', 'local-windows'])
    args = ap.parse_args()
    {'prepare':prepare, 'inventory':inventory, 'local-windows':local_windows}[args.stage]()


if __name__ == '__main__':
    main()
