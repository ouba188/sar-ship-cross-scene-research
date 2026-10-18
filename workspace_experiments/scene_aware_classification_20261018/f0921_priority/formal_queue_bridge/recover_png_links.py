"""Recover bounded clean-PNG lineage by replaying the documented crop, not row joins."""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal, ogr
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PNG = Path('E:/Hermes/paper_tracking/f0921_chips_clean_png')
FINAL = Path('D:/F_0921_relabel_det_final')


def observation_key(record):
    return (str(record['mmsi']), str(record['ship_name']).lower(),
            str(record['obs_time']), str(record['sensor']), str(record['code']),
            round(float(record['lon']), 6), round(float(record['lat']), 6))


def replay_crop(dataset, feature):
    fields = feature.items()
    length = float(fields.get('det_len_px') or 0)
    source = 'det' if fields.get('det_src') == 'obb' and length > 0 else 'ais'
    if source != 'det':
        length_m = float(fields.get('length_m') or 0)
        length = length_m / 10 * 1.55 if length_m > 0 else 80 / 2.4
    side = max(48, min(256, int(round(2.4 * length / 8)) * 8))
    side = min(side, dataset.RasterXSize, dataset.RasterYSize)
    geom = feature.GetGeometryRef()
    gt = dataset.GetGeoTransform()
    col = int((geom.GetX() - gt[0]) / gt[1])
    row = int((gt[3] - geom.GetY()) / -gt[5])
    if not (0 <= col < dataset.RasterXSize and 0 <= row < dataset.RasterYSize):
        return None
    left = max(0, min(col - side // 2, dataset.RasterXSize - side))
    top = max(0, min(row - side // 2, dataset.RasterYSize - side))
    alg = gdal.GRIORA_Average if side > 224 else gdal.GRIORA_Bilinear
    images = []
    for band in [1, 2]:
        a = dataset.GetRasterBand(band).ReadAsArray(left, top, side, side, 224, 224, resample_alg=alg)
        finite = a[np.isfinite(a)]
        if not finite.size:
            return None
        low, high = np.percentile(finite, [2, 98])
        if high <= low:
            low, high = float(finite.min()), float(max(finite.max(), finite.min() + 1e-6))
        images.append(np.clip((a - low) / (high - low) * 255, 0, 255).astype(np.uint8))
    return np.stack(images), dict(left=left, top=top, side=side, crop_source=source,
                                 anchor_x=col-left, anchor_y=row-top)


def main():
    start = time.perf_counter()
    HERE.mkdir(exist_ok=True)
    gdal.UseExceptions(); ogr.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetCacheMax(128 * 1024**2)
    sources = pd.read_csv(ROOT / 'recovery_31cc_2ec7/five_scene_sources.csv', dtype={'mmsi': str})
    stems = sorted({Path(p).stem.replace('.manifest', '')
                    for value in sources.source_manifests_json for p in json.loads(value)})
    by_suffix = {stem[-30:]: [] for stem in stems}
    for folder in PNG.iterdir():
        if not folder.is_dir() or folder.name == 'sheets':
            continue
        for entry in os.scandir(folder):
            if not entry.name.endswith('_VV.png'):
                continue
            parts = entry.name.split('__')
            if len(parts) != 4 or parts[2] not in by_suffix:
                continue
            by_suffix[parts[2]].append(Path(entry.path))
    source_keys = {}
    for s in sources.to_dict('records'):
        key = (str(s['mmsi']), s['class_raw'], s['observation_time_token'],
               s['product'].split('_')[0], s['product'][-4:], round(s['lon'], 6), round(s['lat'], 6))
        source_keys.setdefault(key, []).append(s)
    rows = []; summaries = []
    for stem in stems:
        date = stem[14:22]
        shp = FINAL / date[:4] / date / (stem + '.shp')
        tif = shp.with_suffix('.tif')
        paths = sorted(by_suffix[stem[-30:]])
        images = np.stack([np.stack([np.asarray(Image.open(p)),
                                    np.asarray(Image.open(str(p).replace('_VV.png', '_VH.png')))])
                           for p in paths]) if paths else np.empty((0, 2, 224, 224), np.uint8)
        raster = gdal.Open(str(tif)); vector = ogr.Open(str(shp)); layer = vector.GetLayer(0)
        counts = dict(scene=stem, png_pairs=len(paths), final_records=layer.GetFeatureCount(),
                      replayed=0, matching_records=0, exact_source_records=0)
        for feature in layer:
            fields = feature.items()
            replay = replay_crop(raster, feature)
            if replay is None:
                continue
            a, window = replay
            counts['replayed'] += 1
            candidates = np.flatnonzero(np.all(images == a[None], axis=(1, 2, 3)))
            if not len(candidates):
                continue
            counts['matching_records'] += 1
            matching_sources = source_keys.get(observation_key(fields), [])
            if len(matching_sources) == 1:
                counts['exact_source_records'] += 1
            for j in candidates:
                path = paths[j]
                source = matching_sources[0] if len(matching_sources) == 1 else {}
                rows.append(dict(clean_row=int(re.search(r'__row(\d+)__', path.name).group(1)),
                                 png_vv=str(path), png_vh=str(path).replace('_VV.png', '_VH.png'),
                                 png_class8=path.parent.name, source_shp=str(shp), source_tif=str(tif),
                                 source_fid=feature.GetFID(), scene=stem, mmsi=str(fields['mmsi']),
                                 class_raw=str(fields['ship_name']).lower(), obs_time=fields['obs_time'],
                                 sensor=fields['sensor'], code=fields['code'], lon=fields['lon'], lat=fields['lat'],
                                 product=source.get('product', ''), source_key_json=source.get('source_key_json', ''),
                                 source_candidates=len(matching_sources), png_matches_for_record=len(candidates),
                                 vv_exact=True, vh_exact=True, **window))
        summaries.append(counts)
        print(json.dumps(counts), flush=True)
        raster = None; vector = None
    links = pd.DataFrame(rows)
    if len(links):
        links['source_records_for_png'] = links.groupby('png_vv').png_vv.transform('size')
        links['lineage_unique'] = links.png_matches_for_record.eq(1) & links.source_records_for_png.eq(1)
        links['eligible_source_link'] = links.lineage_unique & links.source_candidates.eq(1)
        linked = links[links.eligible_source_link].copy()
    else:
        linked = links.copy()
    links.to_csv(HERE / 'pixel_links.csv', index=False)
    linked.to_csv(HERE / 'source_linked_png.csv', index=False)
    pd.DataFrame(summaries).to_csv(HERE / 'scene_replay.csv', index=False)
    if len(linked):
        pd.crosstab(linked.class_raw, linked.code).to_csv(HERE / 'class_by_product.csv')
        selected = linked.groupby('code', sort=True).head(1)
        canvas = Image.new('RGB', (720, 275 * len(selected)), 'white')
        draw = ImageDraw.Draw(canvas)
        for k, r in enumerate(selected.to_dict('records')):
            y = k * 275
            draw.text((8, y+3), f"clean{r['clean_row']} / {r['code']} / MMSI{r['mmsi']} / {r['class_raw']}", fill='black')
            draw.text((8, y+20), f"exact pixel replay; source side={r['side']}; crop source={r['crop_source']}", fill='black')
            for j, pol in enumerate(['vv', 'vh']):
                draw.text((8+j*355, y+36), pol.upper(), fill='black')
                canvas.paste(Image.open(r['png_'+pol]).convert('RGB'), (8+j*355, y+50))
        canvas.save(HERE / 'lineage_examples.png')
        selected.to_csv(HERE / 'viewed_examples.csv', index=False)
    receipt = dict(seconds=time.perf_counter()-start, scope='five previously selected products; source scene groups only',
                   groups=len(stems), png_pairs=sum(s['png_pairs'] for s in summaries),
                   replayed_records=sum(s['replayed'] for s in summaries), pixel_link_rows=len(links),
                   unique_source_linked_png=len(linked), distinct_mmsi=linked.mmsi.nunique() if len(linked) else 0,
                   neural_training=False, original_files_modified=False,
                   evidence='Exact two-polarization pixel replay confirms current source crop lineage, not independent vessel-class truth.')
    (HERE / 'RECEIPT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
