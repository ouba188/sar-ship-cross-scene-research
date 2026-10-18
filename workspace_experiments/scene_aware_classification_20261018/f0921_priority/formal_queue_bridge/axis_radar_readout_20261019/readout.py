"""Unsigned, centered directional pooling on the frozen parent layer1 maps."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import resnet50

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT.parent / 'prepool_local_probe'))
from extract_local import pool, spatial_grid


def rectangle_axis(corners):
    p = np.asarray(corners, dtype=np.float64)
    edges = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(edges, axis=1)
    j = int(np.argmax(lengths))
    length, width = lengths[j], lengths[(j + 1) % 4]
    theta = np.arctan2(edges[j, 1], edges[j, 0]) % np.pi
    return p.mean(0), theta, (length - width) / max(length + width, 1e-12)


def detected_axes(meta, detections):
    output = []
    by_row = {}
    for record in detections:
        by_row.setdefault(record['row'], []).extend(record['boxes'])
    for item in meta.itertuples():
        anchor = np.array([item.anchor_x, item.anchor_y]) * 224 / item.side
        axes, centers, weights, confidences = [], [], [], []
        for pol in (0, 1):
            inside = [box for box in by_row[item.row] if box['pol'] == pol and
                      cv2.pointPolygonTest(np.asarray(box['corners'], np.float32),
                                          tuple(map(float, anchor)), False) >= 0]
            if len(inside) != 1:
                continue
            center, theta, elongation = rectangle_axis(inside[0]['corners'])
            axes.append(theta)
            centers.append(center)
            confidences.append(inside[0]['confidence'])
            weights.append(inside[0]['confidence'] * elongation)
        if axes:
            vector = np.dot(weights, np.exp(2j * np.array(axes)))
            theta = float(np.angle(vector) / 2 % np.pi)
            reliability = float(abs(vector) / sum(confidences))
            center = np.average(centers, axis=0, weights=confidences)
            disagreement = (float(np.degrees(np.arccos(np.clip(
                np.cos(2*(axes[0]-axes[1])), -1, 1)))/2) if len(axes) == 2 else np.nan)
        else:
            theta, reliability, center, disagreement = 0., 0., anchor, np.nan
        output.append(dict(row=item.row, center_x=center[0], center_y=center[1],
            ship_axis_rad=theta, axis_reliability=reliability,
            axis_polarizations=len(axes), polarization_axis_gap_deg=disagreement))
    return pd.DataFrame(output)


def angular_windows(x, y, theta, reliability):
    phi = np.arctan2(y, x)
    centers = np.arange(4) * np.pi / 4
    score = 2 * reliability * np.cos(2*(phi[None] - theta - centers[:, None, None]))
    score[:, x*x+y*y < 1e-14] = 0
    weights = np.exp(score - score.max(0, keepdims=True))
    return weights / weights.sum(0, keepdims=True)


def masks_for(axis, roles, reference, radar_angle=np.nan):
    size = roles.shape[-1]
    yy, xx = np.mgrid[:size, :size]
    # Coordinates are pixel centers in the same stored224 system as detector corners.
    x = (xx + .5) * 224 / size - .5 - axis.center_x
    y = (yy + .5) * 224 / size - .5 - axis.center_y
    theta = {'image': 0., 'ship': axis.ship_axis_rad,
             'radar': radar_angle if np.isfinite(radar_angle) else axis.ship_axis_rad}[reference]
    weights = angular_windows(x, y, theta, axis.axis_reliability)
    fg_bg = np.stack([roles[0], roles[1] + roles[2]])
    return (weights[:, None] * fg_bg[None]).reshape(8, size, size).astype(np.float32)


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    meta = pd.read_csv(PARENT/'cohort.csv')
    axes = detected_axes(meta, json.loads((PARENT/'detections.json').read_text()))
    axes.to_csv(HERE/'axes.csv', index=False)
    geometry = pd.read_csv(HERE/'geometry.csv').set_index('row').loc[meta.row]
    radar = geometry.radar_angle_image_rad.to_numpy(float, copy=True)
    valid = geometry.geometry_valid.astype(bool).to_numpy() & np.isfinite(radar)
    radar[~valid] = np.nan
    roles = np.load(PARENT/'masks.npz')['regions']
    old = np.load(PARENT/'features.npz')
    model = resnet50(weights=None).eval().cuda()
    model.load_state_dict(torch.load('C:/Users/25791/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth',
                                    map_location='cpu', weights_only=True))
    refs = ['image', 'ship', 'radar']
    output = {ref: [] for ref in refs}
    masses = {ref: [] for ref in refs}
    grid = spatial_grid(2, 2).cuda()
    replay, reconstruction = 0., 0.
    with torch.inference_mode():
        for start in range(0, len(meta), 8):
            rows = meta.iloc[start:start+8]
            images = np.stack([np.stack([np.asarray(Image.open(r.png_vv)),
                                        np.asarray(Image.open(r.png_vh))]) for r in rows.itertuples()])
            pair = torch.tensor(images, device='cuda', dtype=torch.float32)/255
            x = F.interpolate(pair, size=(128,128), mode='bilinear', align_corners=False, antialias=True)
            x = x.flatten(0,1)[:,None].expand(-1,3,-1,-1)
            x = (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            fmap = model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(x)))))
            restored, _ = pool(fmap, grid[None].expand(len(x),-1,-1,-1))
            expected = torch.tensor(old['grid2'][start:start+len(rows)].reshape(len(x),4,-1), device='cuda')
            replay = max(replay, float((restored-expected).abs().max()))
            torch.testing.assert_close(restored, expected, atol=2e-5, rtol=1e-5)
            for ref in refs:
                masks = np.stack([masks_for(axes.iloc[i], roles[i], ref, radar[i]) for i in rows.row])
                masks = torch.tensor(masks, device='cuda').repeat_interleave(2, 0)
                tokens, mass = pool(fmap, masks)
                output[ref].append(tokens.reshape(len(rows),2,8,-1).cpu().numpy())
                masses[ref].append(mass.reshape(len(rows),2,8)[:,0].cpu().numpy())
                combined = (tokens.reshape(len(x),4,2,-1)*mass.reshape(len(x),4,2,1)).sum(1)
                total = mass.reshape(len(x),4,2).sum(1)
                combined /= total[...,None].clamp_min(1e-8)
                fg_bg = torch.tensor(np.stack([roles[rows.row,0],roles[rows.row,1]+roles[rows.row,2]],1),
                                     device='cuda',dtype=torch.float32).repeat_interleave(2,0)
                direct, _ = pool(fmap, fg_bg)
                reconstruction = max(reconstruction, float((combined-direct).abs().max()))
                torch.testing.assert_close(combined, direct, atol=1e-5, rtol=1e-5)
            if start % 64 == 0:
                print('EXTRACT', start+len(rows), len(meta), flush=True)
    saved = {ref: np.concatenate(value) for ref,value in output.items()}
    saved.update({ref+'_mass': np.concatenate(value) for ref,value in masses.items()})
    np.savez_compressed(HERE/'direction_features.npz', clean_rows=meta.clean_row,
                        radar_valid=valid, **saved)
    receipt = dict(seconds=time.perf_counter()-started, observations=len(meta),
        radar_valid=int(valid.sum()), axis_polarizations=axes.axis_polarizations.value_counts().to_dict(),
        max_grid_replay_error=replay, max_region_reconstruction_error=reconstruction,
        new_detector_calls=False, backbone_training=False, frozen_layer1_inference=True)
    (HERE/'extraction.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
