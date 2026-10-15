"""Label-blind detector-centered dynamic crops, retaining the original cohort."""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal, ogr
from PIL import Image, ImageDraw

ORIGINAL = Path('E:/Hermes/paper_tracking/sensitivity/f0921_transfer_20261007')
RASTERS = Path('D:/F_0921_relabel_det')
OLD_CHIPS = Path('E:/Hermes/paper_tracking/f0921_chips')


def crop_side(length):
    longest = int(math.ceil(max(1., length)))
    wanted = max(longest, min(128, max(64, math.ceil(1.8*longest))))
    return int(8*math.ceil(wanted/8))


def crop_window(cx, cy, side, width, height):
    left = max(0, min(int(round(cx-side/2)), width-side))
    top = max(0, min(int(round(cy-side/2)), height-side))
    return left, top, min(side, width), min(side, height)


def stretch(band):
    finite = np.isfinite(band)
    values = band[finite]
    if not values.size:
        return np.zeros(band.shape, dtype=np.uint8)
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        lo, hi = float(values.min()), float(max(values.max(), values.min()+1e-6))
    result = np.zeros(band.shape, dtype=np.uint8)
    result[finite] = np.clip((band[finite]-lo)/(hi-lo)*255., 0, 255).astype(np.uint8)
    return result


def resize_pair(pair):
    return np.stack([np.asarray(Image.fromarray(band).resize(
        (128, 128), Image.Resampling.BILINEAR)) for band in pair])


def match_feature(row, features, by_mmsi):
    candidates = by_mmsi.get(int(row.mmsi), [])
    if len(candidates) == 1:
        feature = candidates[0]
        return feature, ('same_fid_unique_mmsi' if feature.GetFID() == int(row.row_in_shp)
                         else 'remapped_unique_mmsi')
    # Old lon/lat came from a different metadata stream. Only use them when
    # both the stable identity and rounded location uniquely corroborate a row.
    located = [f for f in candidates
               if abs(float(f.GetField('lon'))-float(row.lon)) <= .0000051
               and abs(float(f.GetField('lat'))-float(row.lat)) <= .0000051]
    if len(located) == 1:
        return located[0], 'mmsi_and_location'
    return None, 'missing_mmsi' if not candidates else 'ambiguous_mmsi'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preview', action='store_true', help='12 label-blind spaced rows only')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'crop_recipe.json').exists():
        raise ValueError('completed crop experiment must not be overwritten')
    target = pd.read_csv(ORIGINAL/'target_manifest.csv', keep_default_na=False)
    if args.preview:
        target = target.iloc[np.linspace(0, len(target)-1, 12, dtype=int)].reset_index(drop=True)
        target.to_csv(out/'target_manifest.csv', index=False)
    else:
        shutil.copyfile(ORIGINAL/'target_manifest.csv', out/'target_manifest.csv')
    protocol = dict(source_run=str(ORIGINAL), target_rows=len(target), preview=args.preview,
        cohort='unchanged 123489 compatible rows from historically selected clean cohort',
        geometry='existing DET OBB center; source S(L) using detected long side; no AIS length/class input',
        side_rule='L=ceil(max(1,det_len_px)); S=8*ceil(max(L,min(128,max(64,ceil(1.8*L))))/8)',
        source_geometry_limit='source rounded axis extents unavailable; analogous sizing, not exact box replay',
        normalization='retain per-band per-chip finite-value percentile2/98; source upstream scaling unresolved',
        fallback='no usable OBB / empty raster pixels: retain existing 224px chip, bilinear resize128',
        identity_join='unique scene+MMSI; duplicated MMSI requires unique rounded lon/lat agreement; otherwise old-input fallback',
        downstream='reuse source_model.pkl, five original global64 support draws, ridge8, gate, 5% cap',
        no_target_query_selection=True, original_results_preserved=True,
        scope='development input-pipeline comparison, not independent confirmation or new model module')
    (out/'RECROP_PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rows = target.row.to_numpy(dtype=np.int64)
    np.save(out/'feature_rows.npy', rows)
    pixels = np.lib.format.open_memmap(out/'inputs128.npy', mode='w+', dtype=np.uint8,
                                     shape=(len(target), 2, 128, 128))
    old_vv = np.load(OLD_CHIPS/'F0921_VV.npy', mmap_mode='r')
    old_vh = np.load(OLD_CHIPS/'F0921_VH.npy', mmap_mode='r')
    gdal.UseExceptions()
    ogr.UseExceptions()
    gdal.SetCacheMax(128*1024*1024)
    records = [None]*len(target)
    started = time.monotonic()
    for scene_index, (scene, group) in enumerate(target.groupby('scene', sort=False)):
        date = scene[14:22]
        stem = RASTERS/date[:4]/date/scene
        vector = ogr.Open(str(stem.with_suffix('.shp')))
        layer = vector.GetLayer()
        features = list(layer)
        by_mmsi = {}
        for feature in features:
            by_mmsi.setdefault(int(feature.GetField('mmsi') or 0), []).append(feature)
        raster = gdal.Open(str(stem.with_suffix('.tif')))
        inv = gdal.InvGeoTransform(raster.GetGeoTransform())
        for index, row in group.iterrows():
            feature, join = match_feature(row, features, by_mmsi)
            conf = float(feature.GetField('det_conf') or 0) if feature else 0.
            length = float(feature.GetField('det_len_px') or 0) if feature else 0.
            geometry = feature.GetGeometryRef() if feature else None
            status = 'fallback_no_detection' if feature else 'fallback_'+join
            pair = None
            side, left, top, cx, cy = 224, -1, -1, float('nan'), float('nan')
            if (feature is not None and feature.GetField('det_src') == 'obb' and conf >= .25
                    and np.isfinite(length) and length > 0 and geometry is not None):
                cx, cy = gdal.ApplyGeoTransform(inv, geometry.GetX(), geometry.GetY())
                if not (0 <= cx < raster.RasterXSize and 0 <= cy < raster.RasterYSize):
                    raise ValueError(f'OBB center outside raster: {scene}/{row.row_in_shp}')
                side = crop_side(length)
                left, top, width, height = crop_window(cx, cy, side, raster.RasterXSize, raster.RasterYSize)
                raw = raster.ReadAsArray(left, top, width, height)
                if raw.shape != (2, height, width):
                    raise ValueError(f'expected paired raw raster: {scene}')
                if all(np.any(np.isfinite(band) & (band != 0)) for band in raw):
                    pair = resize_pair([stretch(band) for band in raw])
                    status = 'detector_dynamic'
                else:
                    status = 'fallback_empty_raster'
            if pair is None:
                pair = resize_pair([old_vv[int(row.row)], old_vh[int(row.row)]])
            pixels[index] = pair
            records[index] = dict(row=int(row.row), scene=scene, row_in_shp=int(row.row_in_shp),
                matched_fid=feature.GetFID() if feature else -1, identity_join=join,
                status=status, det_conf=conf, det_len_px=length, requested_side=side,
                left=left, top=top, center_x=cx, center_y=cy,
                vv_zero_fraction=float(np.mean(pair[0] == 0)),
                vh_zero_fraction=float(np.mean(pair[1] == 0)))
        raster = vector = layer = None
        if scene_index == 0 or (scene_index+1) % 200 == 0:
            print(f'CROPPED scenes={scene_index+1} elapsed={time.monotonic()-started:.1f}s', flush=True)
    pixels.flush()
    metadata = pd.DataFrame(records)
    metadata.to_csv(out/'crop_metadata.csv', index=False)
    changed = metadata.loc[metadata.status == 'detector_dynamic', 'requested_side']
    recipe = dict(**protocol, status_counts=metadata.status.value_counts().to_dict(),
        identity_join_counts=metadata.identity_join.value_counts().to_dict(),
        dynamic_side_quantiles=changed.quantile([0, .5, .9, 1]).to_dict(),
        elapsed_seconds=time.monotonic()-started)
    (out/'crop_recipe.json').write_text(json.dumps(recipe, indent=2), encoding='utf-8')
    chosen = np.linspace(0, len(target)-1, 12, dtype=int)
    sheet = Image.new('RGB', (512, len(chosen)*154), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, i in enumerate(chosen):
        row = int(rows[i])
        before = resize_pair([old_vv[row], old_vh[row]])
        for k, band in enumerate([before[0], pixels[i, 0], before[1], pixels[i, 1]]):
            sheet.paste(Image.fromarray(band).convert('RGB'), (128*k, 154*j+24))
        rec = records[i]
        draw.text((2, 154*j+3), f'row={row} S={rec["requested_side"]} {rec["status"]} | VV old/new, VH old/new', fill='black')
    sheet.save(out/'crop_comparison.png')
    print(json.dumps(recipe, indent=2), flush=True)


if __name__ == '__main__':
    main()
