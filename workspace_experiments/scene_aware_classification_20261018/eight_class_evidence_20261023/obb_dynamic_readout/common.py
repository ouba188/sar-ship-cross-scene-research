"""Local geometry and read-only interfaces for the OBB experiment."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT.parents[1] / 'same_vessel_gallery_20261019'
ATT = ROOT / 'calibrated_signed_transfer/polarization_readout/joint_distribution_readout/matched_attention_readout'
sys.path.insert(0, str(OLD / 'local_condition_module'))
from native_input import crop_pad
from prepare import masks_for, WEIGHTS
POL = ('vv', 'vh')
REGIONS = ('body', 'end', 'side')
SIZE, STRIDE, ORIGIN = 160, 4, 0
EDGES = np.r_[-.5, np.arange(2., 155., 4.), 159.5]
YY, XX = np.mgrid[:SIZE, :SIZE]


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def load_data():
    table = pd.read_csv(ROOT / 'cohort.csv', dtype={'mmsi': str})
    table['pass_group'] = table['product'].map(lambda s: '_'.join([s.split('_')[0], s.split('_')[6], s.split('_')[7]]))
    with np.load(ROOT / 'features.npz', allow_pickle=True) as z:
        np.testing.assert_array_equal(z['instance_ids'].astype(str), table.instance_id)
        valid = z['valid'].copy()
    snapshots = {s['obs_id']: s for s in json.loads((OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    return table, valid, snapshots


def readers():
    result = []
    for f in range(3):
        for tag in ['outer'] + [f'inner{k}' for k in range(4)]:
            name = f'fold{f}_{tag}'
            with np.load(ROOT / 'baseline/models' / f'{name}.npz') as z:
                result.append(dict(name=name, fold=f, fit=z['fit'].copy(),
                                   held=z['held'].copy() if 'held' in z else None))
    return result


def splits(fold):
    with np.load(ROOT / 'baseline' / f'fold{fold}_inputs.npz') as z:
        return {k: z[k].copy() for k in ('train', 'validation', 'test')}


def required(stage, n):
    roles = readers()
    take = np.zeros((len(roles), n), bool)
    for j, r in enumerate(roles):
        d = splits(r['fold'])
        if r['held'] is not None:
            if stage == 'source':
                take[j, r['held']] = True
        else:
            take[j, d['validation' if stage == 'source' else 'test']] = True
    return roles, take


def geometry(view, place):
    priors = view['priors']
    index = priors.get('target_index')
    dets = priors.get('detections', [])
    if index is None or index < 0 or index >= len(dets):
        return None
    d = dets[index]
    if not d.get('shape_usable') or d['confidence'] < .25:
        return None
    corners = np.asarray(d['corners'], float)
    edges = np.roll(corners, -1, 0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    j = int(lengths.argmax())
    center = corners.mean(0) + [place['pad_left']-place['crop_x'], place['pad_top']-place['crop_y']]
    return np.array([*center, lengths[j], lengths[(j+1) % 4],
                     np.arctan2(edges[j, 1], edges[j, 0]) % np.pi, d['confidence']])


def native(table, snapshots, i, p, pixels=True):
    row = table.iloc[i]
    view = snapshots[int(row.obs_id)]['images'][POL[p]]
    if pixels:
        with Image.open(row['image_'+POL[p]]) as im:
            raw = np.array(im.convert('L'))
    else:
        raw = np.zeros((view['h'], view['w']), np.uint8)
    assert raw.shape == (view['h'], view['w'])
    pix, masks, place = crop_pad(raw, masks_for(view), SIZE)
    assert place['retained_target_pixels'] == place['native_target_pixels']
    return pix, masks.sum(0), geometry(view, place), place


def soft_regions(g, template=None, shape=(SIZE, SIZE), offset=(0, 0)):
    if g is None:
        return np.zeros((3, *shape), np.float64)
    cx, cy, length, width, angle = g[:5]
    if template is not None:
        length, width = template
    if shape == (SIZE, SIZE) and offset == (0, 0):
        yy, xx = YY, XX
    else:
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        xx, yy = xx + offset[0], yy + offset[1]
    u = (xx-cx)*np.cos(angle) + (yy-cy)*np.sin(angle)
    v = -(xx-cx)*np.sin(angle) + (yy-cy)*np.cos(angle)
    # One-pixel-wide linear transition, centered on each rectangle edge.
    def rectangle(l, w):
        return np.clip(l/2+.5-np.abs(u), 0, 1) * np.clip(w/2+.5-np.abs(v), 0, 1)
    body = rectangle(length, width)
    return np.stack((body, np.maximum(rectangle(1.5*length, width)-body, 0),
                     np.maximum(rectangle(length, 2*width)-body, 0)))


def planned_mass(g, template=None):
    if g is None:
        return np.zeros(3)
    length, width = g[2:4] if template is None else template
    radius = int(np.ceil(np.hypot(1.5*length, 2*width)/2+2))
    start = (int(np.floor(g[0]))-radius, int(np.floor(g[1]))-radius)
    return soft_regions(g, template, (2*radius+1, 2*radius+1), start).sum((-2, -1))


def cell_mass(mask):
    """Exact integral of piecewise-constant pixels over center-Voronoi cells."""
    points = EDGES + .5
    idx = np.minimum(np.floor(points).astype(int), SIZE-1)
    frac = points - idx
    def integrate_last(a):
        prefix = np.concatenate((np.zeros((*a.shape[:-1], 1)), np.cumsum(a, axis=-1)), -1)
        values = prefix[..., idx] + a[..., idx]*frac
        return np.diff(values, axis=-1)
    out = integrate_last(mask)
    return np.swapaxes(integrate_last(np.swapaxes(out, -1, -2)), -1, -2)


def region_quality(mask, footprint, g, template=None):
    pixel = mask * footprint
    mass = pixel.sum((-2, -1))
    plan = planned_mass(g, template)
    weights = np.stack([cell_mass(np.rot90(pixel, k, (-2, -1))) for k in range(4)])
    neff = mass[None]**2 / np.maximum((weights**2).sum((-2, -1)), 1e-300)
    coverage = np.divide(mass, plan, out=np.zeros_like(mass), where=plan > 0)
    return dict(area=mass, planned=plan, coverage=coverage,
                neff=neff.mean(0), min_neff=neff.min(0), nodes=(weights > 0).sum((-2, -1)).min(0))


def rotate_geometry(g, k):
    out = g.copy()
    for _ in range(k % 4):
        out[0], out[1] = out[1], SIZE-1-out[0]
        out[4] -= np.pi/2
    return out


def structural_checks():
    g = np.array([73.2, 81.7, 27., 9., .37, .8])
    mask = soft_regions(g)
    gp = g.copy(); gp[4] += np.pi
    pierr = float(abs(mask-soft_regions(gp)).max())
    rotation = max(float(abs(np.rot90(mask, k, (-2, -1))-soft_regions(rotate_geometry(g, k))).max()) for k in range(4))
    impulse = np.zeros((160, 160)); impulse[0, 2] = 1
    weights = cell_mass(impulse)
    # Pixel x=2 straddles the true x=2 cell boundary, unlike old 4x4 bins.
    assert weights[0, 0] == .5 and weights[0, 1] == .5 and weights.sum() == 1
    assert pierr < 1e-12 and rotation < 1e-12
    np.testing.assert_allclose(cell_mass(mask).sum((-2, -1)), mask.sum((-2, -1)), atol=1e-10)
    raw = np.arange(80*90, dtype=np.int32).reshape(80, 90)
    m = np.zeros((3, 80, 90), np.float32); m[0, 25:40, 30:50] = 1; m[2] = 1-m[0]
    out, _, place = crop_pad(raw, m, 160)
    dx, dy = place['pad_left']-place['crop_x'], place['pad_top']-place['crop_y']
    assert out[30+dy, 40+dx] == raw[30, 40]
    return dict(theta_pi_max_error=pierr, c4_mask_max_error=rotation,
                known_impulse_weights=[.5, .5], crop_offset_pixel_check=True,
                stride=4, center_origin=0, voronoi_edges=EDGES.tolist(),
                layer1_theoretical_rf=35, pixel_domain=[-.5,159.5],
                note='Pixel mask values are piecewise constant; Voronoi border cells extend to canvas edge. C4 weights recomputed, no single-view feature equivariance assumed.')
