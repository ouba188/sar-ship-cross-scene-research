"""Co-located calibrated SAR descriptors for existing frozen local tokens."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'oss1_joint_geometry_20261017'
sys.path.insert(0, str(OLD))
from local_reader import regions


def measurements(image, body, weights):
    mean_body = max(float(image[body].mean()), 1e-12)
    flat = image.flatten().astype(float)
    mean = weights@flat
    std = np.sqrt(np.maximum(weights@(flat**2)-mean**2, 0))
    bright = weights@(flat > .5*mean_body).astype(float)
    return np.column_stack([np.log1p(mean/mean_body), np.log1p(std/np.maximum(mean, 1e-12)),
                            bright, np.full(len(mean), np.log(mean_body))]), mean, mean_body


def main():
    m = pd.read_csv(OLD/'geometry.csv')
    detections = pd.read_csv(OLD/'detections.csv').set_index('global_row')
    features = np.zeros((len(m), 12, 10), np.float32)
    positions = np.zeros((len(m), 12, 16), np.float32)
    for i, row in m.iterrows():
        d = detections.loc[row.global_row].to_dict()
        body, mask, pos, coverage = regions(d, float(row.radar_axis_rad) if row.radar_valid else 0.,
            float(row.incidence_rad) if row.incidence_valid else 0., bool(row.radar_valid), bool(row.incidence_valid), 'joint')
        positions[i] = pos
        if not d['axis_valid'] or not d['views_same_size']:
            continue
        image = tifffile.imread(row.calibrated_patch_path).astype(float)
        h, w = image.shape[:2]
        yy, xx = np.meshgrid((np.arange(56)+.25)*h/56-.5, (np.arange(56)+.25)*w/56-.5, indexing='ij')
        # Native radiometry is sampled at the existing conv1 center coordinates.
        pair = [map_coordinates(image[:, :, channel], [yy, xx], order=1, mode='nearest') for channel in (1, 0)]
        vv, mean_vv, body_vv = measurements(pair[0], body, mask.reshape(12, -1))
        vh, mean_vh, body_vh = measurements(pair[1], body, mask.reshape(12, -1))
        ratios = np.column_stack([np.log(np.maximum(mean_vv, 1e-12)/np.maximum(mean_vh, 1e-12)),
                                  np.full(12, np.log(body_vv/body_vh))])
        features[i] = np.column_stack([vv, vh, ratios])
        features[i, pos[:, 3] == 0] = 0
        if i % 300 == 0:
            print('QUALITY', i, '/', len(m), flush=True)
    valid = detections.reindex(m.global_row).axis_valid.to_numpy(bool) & detections.reindex(m.global_row).views_same_size.to_numpy(bool)
    np.savez_compressed(HERE/'quality.npz', sample_ids=m.sample_id.to_numpy(str), quality=features, positions=positions, valid=valid)
    (HERE/'QUALITY_PROTOCOL.json').write_text(json.dumps(dict(samples=len(m), valid=int(valid.sum()),
        dimensions=10, region='same joint12 centers/masks as cached source-only stem; no new detector/backbone',
        co_location='native calibrated response sampled at conv1 centers; not full receptive-field integration',
        channels='VH index0, VV index1; already sigma0, no squaring',
        values='eachpol log1p(local/body), log1p(localCV), fraction above .5bodymean, logbodymean; local/body VV-VH logratios',
        interpretation='observable SAR descriptors, not ground-truth reliability or pure sea/wake; no AIS/weather input'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
