"""Native-scale, detection-only scene context for a bounded three-class pilot."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal, osr
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch
from torchvision.models import resnet50

ROOT = Path('E:/临时会话')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'ports24'
AUDIT = ROOT / 'scene_aware_classification_20261018/data_image_review/ports24'
sys.path.insert(0, str(ROOT / 'scene_aware_classification_20261018/f0921_priority/land_mask'))
from land_mask import WaterMap, WC, region_stats


def read(path):
    return pd.read_csv(path, low_memory=False)


def polygon_mask(points, origin, shape):
    image = Image.new('1', (shape[1], shape[0]))
    ImageDraw.Draw(image).polygon([tuple(p) for p in np.asarray(points) - origin], fill=1)
    return np.asarray(image, dtype=bool)


def box_geometry(points, pixel_size):
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    k = lengths.argmax()
    return lengths[k] * pixel_size, lengths.min() * pixel_size, edges[k] / max(lengths[k], 1e-8)


class Reader:
    def __init__(self):
        torch.set_num_threads(2)
        model = resnet50(weights=None)
        model.load_state_dict(torch.load('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth', map_location='cpu', weights_only=True))
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu).cuda().eval()
        self.mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]

    @torch.inference_mode()
    def encode(self, a):
        # Native pixels, exact quarter turns, no interpolation or scale normalization.
        x = torch.from_numpy(np.asarray(a, dtype=np.float32) / 255)[None, None].cuda()
        x = x.expand(-1, 3, -1, -1)
        outputs = []
        for k in range(4):
            z = self.stem((x.rot90(k, (-2, -1)) - self.mean) / self.std)
            outputs.append(torch.cat([z.mean((-2, -1)), z.amax((-2, -1))], 1))
        return torch.stack(outputs).mean(0).squeeze(0).cpu().numpy()


def crop_world(ds, xy, side_m):
    gt = ds.GetGeoTransform()
    px, py = gdal.ApplyGeoTransform(gdal.InvGeoTransform(gt), *xy)
    sx, sy = abs(gt[1]), abs(gt[5])
    w, h = math.ceil(side_m / sx), math.ceil(side_m / sy)
    def pixel_floor(value):
        # Affine round trips can turn an exact edge into e.g. 6081.99999999994.
        nearest = round(value)
        return nearest if abs(value-nearest) < 1e-6 else math.floor(value)
    left, top = pixel_floor(px - w / 2), pixel_floor(py - h / 2)
    if left < 0 or top < 0 or left + w > ds.RasterXSize or top + h > ds.RasterYSize:
        return None, None
    return ds.ReadAsArray(left, top, w, h), [left, top, w, h]


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    (OUT / 'native_targets').mkdir(exist_ok=True)
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetCacheMax(64 * 1024**2)
    wm = WaterMap(sorted(WC.glob('*_Map.tif')))
    reader = Reader()
    assets = read(AUDIT / 'product_asset_availability.csv')
    legal = assets[assets.full & assets.full_candidates & assets.located_vv.gt(0)]['product']
    manifest = read(ROOT / 'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv')
    manifest['manifest_row'] = np.arange(len(manifest))
    manifest = manifest[manifest['product'].isin(legal)].copy()
    views = read(ROOT / 'chip_quality_20261017/ais_metadata/view_metadata.csv')
    views = views[views.pol.eq('vv') & views.source_transform_json.notna()].set_index('instance_id')
    rasters = read(AUDIT / 'local_rasters.csv')
    rasters = rasters[~rasters.tile]
    catalogs = read(AUDIT / 'fullscene_candidate_catalogs.csv')
    records, neighbors, rejected, all_h, all_n, all_g, all_m = [], [], [], [], [], [], []
    for product, group in manifest.groupby('product', sort=True):
        paths = {}
        for pol in ('vv', 'vh'):
            candidates = rasters[(rasters['product'] == product) & (rasters.pol == pol)].path
            candidates = sorted(candidates, key=lambda p: ('tif_local_done' not in p, p))
            paths[pol] = next((p for p in candidates if Path(p).exists()), None)
        if not all(paths.values()):
            rejected.extend(dict(instance_id=r.instance_id, reason='missing_full_polarization') for r in group.itertuples())
            continue
        cat = catalogs[(catalogs['product'] == product) & catalogs.pol.eq('vv')]
        if len(cat) != 1:
            raise ValueError(f'VV catalog ambiguity: {product}')
        # Do not read AIS matches, class prelabels, or label-dependent screening columns.
        det = pd.read_csv(cat.iloc[0].audit_path, usecols=['detection_id', 'source_pts', 'conf', 'pixel_status', 'roi_origin', 'source_image'])
        det = det[det['conf'].ge(.25) & det.pixel_status.eq('valid')].reset_index(drop=True)
        if not all(json.loads(v) == [0, 0] for v in det.roi_origin):
            raise ValueError('Catalog must use whole-image pixel coordinates')
        if not all(Path(v.replace('\\', '/')).name == Path(paths['vv']).name for v in det.source_image):
            raise ValueError('Detection raster version mismatch')
        pts = np.asarray([json.loads(v) for v in det.source_pts])
        centers = pts.mean(1)
        tree = cKDTree(centers)
        dv, dh = gdal.Open(paths['vv']), gdal.Open(paths['vh'])
        gt = dv.GetGeoTransform()
        crs = dv.GetSpatialRef()
        crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not crs.IsSame(dh.GetSpatialRef()) or gt != dh.GetGeoTransform() or (dv.RasterXSize,dv.RasterYSize) != (dh.RasterXSize,dh.RasterYSize):
            raise ValueError('This pilot requires pixel-aligned VV/VH originals')
        if gt[2] or gt[4] or not crs.IsProjected():
            raise ValueError('Expected north-up metric raster')
        scale = abs(gt[1])
        if not np.isclose(scale, abs(gt[5])):
            raise ValueError('Metric geometry assumes square pixels')
        feature_cache = {}
        for row in group.itertuples():
            if pd.isna(row.mmsi):
                rejected.append(dict(instance_id=row.instance_id, reason='missing_identity_for_group_split'))
                continue
            if row.instance_id not in views.index:
                rejected.append(dict(instance_id=row.instance_id, reason='missing_exact_georef'))
                continue
            v = views.loc[row.instance_id]
            if isinstance(v, pd.DataFrame):
                raise ValueError('Duplicate target view')
            native = Image.open(v.image_path)
            t = json.loads(v.chip_transform_json)
            source_crs = osr.SpatialReference()
            source_crs.ImportFromEPSG(int(v.epsg))
            source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            wx, wy = gdal.ApplyGeoTransform(t, native.width / 2, native.height / 2)
            if not source_crs.IsSame(crs):
                wx, wy, _ = osr.CoordinateTransformation(source_crs,crs).TransformPoint(wx,wy)
            center = np.array(gdal.ApplyGeoTransform(gdal.InvGeoTransform(gt), wx, wy))
            side_m = max(native.width, native.height) * abs(t[1])
            target_images, windows = [], []
            for ds in (dv, dh):
                image, window = crop_world(ds, (wx, wy), side_m)
                target_images.append(image)
                windows.append(window)
            if any(im is None for im in target_images) or max(target_images[0].shape) > 512:
                rejected.append(dict(instance_id=row.instance_id, reason='target_window_outside_or_oversize'))
                continue
            # Target label remains attached to its original crop, never reassigned to nearest detection.
            dself, selfid = tree.query(center)
            if dself > 5:
                rejected.append(dict(instance_id=row.instance_id, reason='target_axis_unresolved_distance_gt5px'))
                continue
            length, width, axis = box_geometry(pts[selfid], scale)
            cross = np.array([-axis[1], axis[0]])
            ids = tree.query_ball_point(center, 1000 / scale)
            ids = sorted(ids, key=lambda k: (np.linalg.norm(centers[k] - center), int(k)))
            left, top, w, h = windows[0]
            # Use a bounded mask window with extra margin for the 100 m interior rule.
            pad = math.ceil(1250 / scale)
            x0, y0 = np.maximum(0, np.floor(center).astype(int) - pad)
            x1, y1 = np.minimum([dv.RasterXSize, dv.RasterYSize], np.ceil(center).astype(int) + pad)
            land, distance = wm.window(dv, (x0, y0, x1-x0, y1-y0))
            used, truncated, candidates_outside, land_removed = [], 0, 0, 0
            for k in ids:
                cx, cy = centers[k]
                if k == selfid or (left <= cx < left+w and top <= cy < top+h):
                    continue
                candidates_outside += 1
                mask = polygon_mask(pts[k], np.array([x0,y0]), land.shape)
                stats = region_stats(land, distance, mask)
                if stats['action'] == 'exclude_land_interior':
                    land_removed += 1
                    continue
                nl, nw, naxis = box_geometry(pts[k], scale)
                node_side = max(64 * scale, np.ptp(pts[k], axis=0).max() * scale + 32 * scale)
                if node_side / scale > 512:
                    continue
                if k not in feature_cache:
                    world = gdal.ApplyGeoTransform(gt, *centers[k])
                    ims = [crop_world(ds, world, node_side)[0] for ds in (dv, dh)]
                    if any(im is None for im in ims):
                        feature_cache[k] = None
                    else:
                        feature_cache[k] = np.concatenate([reader.encode(im) for im in ims])
                if feature_cache[k] is None:
                    continue
                if len(used) >= 16:
                    truncated += 1
                    continue
                rel = (centers[k] - center) * scale
                dot = float(axis @ naxis)
                # Unsigned axis relations cannot encode an assumed bow or absolute heading.
                geom = [np.linalg.norm(rel)/1000, abs(rel@axis)/1000, abs(rel@cross)/1000,
                        math.log(max(nl,1)/max(length,1)), math.log(max(nw,1)/max(width,1)),
                        2*dot*dot-1, det.iloc[k]['conf']]
                used.append((k, geom))
                neighbors.append(dict(instance_id=row.instance_id, product=product, detection_id=det.iloc[k].detection_id,
                    source_points=json.dumps(pts[k].tolist()), cx=cx, cy=cy, side_m=node_side,
                    distance_m=float(np.linalg.norm(rel)), land_action=stats['action'],
                    overlaps_chip=bool(pts[k,:,0].min()<left+w and pts[k,:,0].max()>left and pts[k,:,1].min()<top+h and pts[k,:,1].max()>top),
                    **{f'g{j}': float(x) for j,x in enumerate(geom)}))
            hn = np.zeros((16,256), np.float32)
            gn = np.zeros((16,7), np.float32)
            mn = np.zeros(16, bool)
            for j,(k,g) in enumerate(used):
                hn[j], gn[j], mn[j] = feature_cache[k], g, True
            all_h.append(np.concatenate([reader.encode(im) for im in target_images]))
            all_n.append(hn); all_g.append(gn); all_m.append(mn)
            for pol,im in zip(('vv','vh'), target_images):
                Image.fromarray(im).save(OUT/'native_targets'/f'{row.instance_id}_{pol}.png')
            records.append(dict(instance_id=row.instance_id, manifest_row=row.manifest_row,
                class_name=row.class_name, class_id=row.class_id, mmsi=str(int(row.mmsi)),
                product=product, pass_group='_'.join(product.split('_')[6:8]), port=row.port,
                date=product.split('_')[4][:8], x=float(center[0]), y=float(center[1]), pixel_m=scale,
                target_window=json.dumps(windows[0]), original_chip=str(v.image_path),
                raster_vv=paths['vv'], raster_vh=paths['vh'], axis_match_px=float(dself),
                neighbors=len(used), outside_candidates=candidates_outside, land_removed=land_removed,
                truncated=truncated, radius_inside_raster=bool(min(center[0],center[1],dv.RasterXSize-center[0],dv.RasterYSize-center[1])*scale>=1000)))
        dv = None; dh = None
        print(product[-4:], 'targets',len(records),'neighbors',len(neighbors),flush=True)
    pd.DataFrame(records).to_csv(OUT/'targets.csv', index=False)
    pd.DataFrame(neighbors).to_csv(OUT/'neighbors.csv', index=False)
    pd.DataFrame(rejected).to_csv(OUT/'excluded.csv', index=False)
    np.savez_compressed(OUT/'features.npz', target=np.asarray(all_h), neighbor=np.asarray(all_n), geometry=np.asarray(all_g), mask=np.asarray(all_m))
    summary = dict(targets=len(records), edges=len(neighbors), excluded=len(rejected),seconds=time.perf_counter()-started,
        status='features_only_no_classification_result',visual='frozen ImageNet R50 stem mean/max, exact C4, VV/VH separate; native full raster windows')
    (OUT/'prepare_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(summary,flush=True)


if __name__ == '__main__':
    main()
