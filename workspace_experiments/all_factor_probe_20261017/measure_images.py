"""Native SAR shape measurements and matched local evidence probes."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from PIL import Image
import tifffile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT/'oss1_joint_geometry_20261017'
spec = importlib.util.spec_from_file_location('native_polar_measurements', ROOT/'oss1_polar_environment_20261017/analyze_images.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def weighted_extent(coordinate, weight):
    order = np.argsort(coordinate, kind='stable')
    c, w = coordinate[order], weight[order]
    if w.sum() <= 0:
        return np.nan
    cumulative = np.cumsum(w)/w.sum()
    return float(np.interp(.95, cumulative, c)-np.interp(.05, cumulative, c))


def image_shape(image, masks, d):
    yy, xx = np.indices(image.shape)
    dx, dy = xx-d.cx, yy-d.cy
    co, si = np.cos(d.ship_axis), np.sin(d.ship_axis)
    u, v = dx*co+dy*si, -dx*si+dy*co
    body = masks['body']
    weights = np.maximum(image[body].astype(float), 0)
    if len(weights) == 0 or weights.sum() <= 0:
        return {}
    total = weights.sum()
    top = max(1, int(np.ceil(.05*len(weights))))
    out = dict(body_top5_mass=float(np.sort(weights)[-top:].sum()/total),
               body_mean=float(weights.mean()), body_pixels=int(body.sum()))
    for label, coordinates in [('range', dx), ('azimuth', dy), ('ship_long', u), ('ship_transverse', v)]:
        z = coordinates[body]
        mean = np.dot(z, weights)/total
        out[label+'_rms_px'] = float(np.sqrt(np.dot((z-mean)**2, weights)/total))
        out[label+'_extent90_px'] = weighted_extent(z, weights)
    out['azimuth_range_log_ratio'] = float(np.log((out['azimuth_rms_px']+1e-6)/(out['range_rms_px']+1e-6)))
    out['long_extent_ratio'] = out['ship_long_extent90_px']/max(d.length, 1)
    out['transverse_extent_ratio'] = out['ship_transverse_extent90_px']/max(d.width, 1)
    outside = np.logical_or.reduce([masks[n] for n in native.NAMES[1:]])
    out['outside_mean'] = float(image[outside].mean()) if outside.any() else np.nan
    out['outside_to_body_mean'] = out['outside_mean']/out['body_mean']
    return out


def spatial_cells(shape, d):
    """36 ship-coordinate cells paired under a 180-degree axis reversal."""
    h, w = shape
    a, b = max(d.length*1.25/2, 2), max(d.width*1.25/2, 1.5)
    reach = int(np.ceil(3*np.hypot(a, b)))+2
    yy, xx = np.mgrid[int(np.floor(d.cy))-reach:int(np.ceil(d.cy))+reach+1,
                      int(np.floor(d.cx))-reach:int(np.ceil(d.cx))+reach+1]
    x, y = xx-d.cx, yy-d.cy
    u = (x*np.cos(d.ship_axis)+y*np.sin(d.ship_axis))/a
    v = (-x*np.sin(d.ship_axis)+y*np.cos(d.ship_axis))/b
    # Round only numerical boundary noise, preserving actual subpixel coordinates.
    u, v = np.round(u, 10), np.round(v, 10)
    inside = (abs(u) < 3) & (abs(v) < 3)
    iu = np.clip(np.floor(abs(u)).astype(int), 0, 2)
    iv = np.clip(np.floor(abs(v)).astype(int), 0, 2)
    label = (iu*3+iv)*2+(u*v < 0).astype(int)
    full = np.bincount(label[inside], minlength=18)
    visible = inside & (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    observed = np.bincount(label[visible], minlength=18)
    coverage = np.divide(observed, full, out=np.zeros(18), where=full > 0)
    return yy[visible], xx[visible], label[visible], observed, coverage


def read_cells(image, geometry, body_mean):
    y, x, label, counts, coverage = geometry
    sums = np.bincount(label, weights=image[y, x], minlength=18)
    mean = np.divide(sums, counts, out=np.zeros(18), where=counts > 0)
    response = np.log1p(np.maximum(mean, 0)/max(body_mean, 1e-12))
    # Missing values are accompanied by a mask and coverage, never measured zero sea.
    mask = counts > 0
    return np.stack([response, coverage, mask.astype(float)], axis=1)


def main():
    meta = pd.read_csv(OLD/'geometry.csv', dtype={'mmsi': str})
    det = pd.read_csv(OLD/'detections.csv').set_index('global_row')
    valid = det.reindex(meta.global_row).axis_valid.to_numpy(bool)
    variants = ['uint8_spatial', 'uint8_bag', 'sigma0_spatial', 'uint8_lowpass']
    features = {name: np.zeros((len(meta), 18, 6), np.float32) for name in variants}
    stats = []
    for n, r in meta.iterrows():
        d = det.loc[r.global_row]
        if not d.axis_valid:
            continue
        cal = tifffile.imread(r.calibrated_patch_path)
        raw = tifffile.imread(r.raw_patch_path)
        masks, coverage = native.regions(cal.shape[:2], d.cx, d.cy, d.length, d.width, d.ship_axis)
        cells = spatial_cells(cal.shape[:2], d)
        for pidx, (pol, channel) in enumerate([('vv', 1), ('vh', 0)]):
            display = tifffile.imread(r['chip_path_'+pol])
            size = (max(1, display.shape[1]//2), max(1, display.shape[0]//2))
            low = np.asarray(Image.fromarray(display).resize(size, Image.Resampling.BILINEAR).resize(
                (display.shape[1], display.shape[0]), Image.Resampling.BILINEAR))
            for name, image in [('amplitude', raw[:, :, channel]), ('sigma0', cal[:, :, channel]), ('uint8', display), ('uint8_lowpass', low)]:
                values = image_shape(image, masks, d)
                stats.append(dict(global_row=int(r.global_row), sample_id=r.sample_id, mmsi=r.mmsi,
                                  product=r['product'], class_id=int(r.class_id), pol=pol, representation=name,
                                  length_px=d.length, width_px=d.width, ship_axis_deg=np.degrees(d.ship_axis) % 180,
                                  incidence_deg=r.incidence_deg, min_coverage=min(coverage.values()), **values))
                if name == 'amplitude':
                    continue
                tokens = read_cells(image, cells, values['body_mean'])
                arm = {'sigma0': 'sigma0_spatial', 'uint8': 'uint8_spatial', 'uint8_lowpass': 'uint8_lowpass'}[name]
                features[arm][n, :, pidx*3:(pidx+1)*3] = tokens
        order = np.argsort(features['uint8_spatial'][n, :, [0, 3]].sum(axis=0), kind='stable')
        features['uint8_bag'][n] = features['uint8_spatial'][n, order]
        if n % 300 == 0:
            print('MEASURE', n, '/', len(meta), flush=True)
    pd.DataFrame(stats).to_csv(HERE/'native_shape_measurements.csv', index=False)
    np.savez_compressed(HERE/'spatial_descriptors.npz', sample_ids=meta.sample_id.to_numpy(str),
                        valid=valid, **{k: v.reshape(len(meta), -1) for k, v in features.items()})
    protocol = dict(samples=len(meta), valid_axes=int(valid.sum()), tokens=18, token_channels=['vv_response', 'vv_coverage', 'vv_present', 'vh_response', 'vh_coverage', 'vh_present'],
                    scalar='log1p(region_mean/body_mean)', coordinate='native OBB times1.25; -3..3 in long/transverse axes',
                    arms=variants, axis='unsigned; opposite cells paired', lowpass='half native dimensions then restore bilinear, not a new sensor simulation',
                    input='SAR pixels and image-derived OBB only; no AIS/environment labels',
                    intent='fixed diagnostic readouts, not a newly trained attention module')
    (HERE/'IMAGE_PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
