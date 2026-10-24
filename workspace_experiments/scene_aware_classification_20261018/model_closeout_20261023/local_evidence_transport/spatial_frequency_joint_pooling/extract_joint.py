"""Native spatial-frequency cross moments with frozen layer1/PCA32."""
import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import fft

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAIRED = ROOT / 'paired_scale_readout'
spec = importlib.util.spec_from_file_location(
    'joint_frequency', ROOT / 'frequency_conditioned_matching/extract_frequency.py')
frequency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frequency)
token, geo = frequency.token, frequency.geo


def place(field, placement):
    x, y = placement['crop_x'], placement['crop_y']
    w, h = placement['crop_width'], placement['crop_height']
    dx, dy = placement['pad_left'], placement['pad_top']
    canvas = np.zeros((*field.shape[:-2], 160, 160), dtype=field.dtype)
    canvas[..., dy:dy+h, dx:dx+w] = field[..., y:y+h, x:x+w]
    return canvas


def energy_field(raw, kernels):
    return np.stack([np.log1p(np.mean([
        np.abs(frequency.response(raw, psi))**2 for psi in band
    ], axis=0) / 1e-4) for band in kernels])


def grid_moments(features, weights, energy_weights):
    """All expectations use the same mask mass, never energy-normalized mass."""
    mass = weights.sum(-1)
    denominator = np.maximum(mass, 1e-20)
    mean = (weights @ features) / denominator[:, None]
    energy_mean = energy_weights.sum(-1) / denominator[:, None]
    cross = (energy_weights @ features) / denominator[:, None, None]
    marginal = energy_mean[..., None] * mean[:, None, :]
    joint = cross - marginal
    for value in (joint, marginal, mean):
        value[mass <= 0] = 0
    return joint, marginal, mean


def pool(features, masks, energy):
    weights = token.cell_mass(masks, 4).reshape(20, -1)
    energy_weights = token.cell_mass(
        masks[:, None] * energy[None], 4).reshape(20, 3, -1)
    return grid_moments(features, weights, energy_weights)


def targeted_checks():
    rng = np.random.default_rng(1024)
    features = rng.normal(size=(1600, 32))
    masks = token.slots(np.array([73.2, 81.7, 27., 9., .37, .8]),
                        np.ones((160, 160)))
    masks[-1] = 0
    masks[0] = 0
    masks[0, 0, 2] = 1
    energy = rng.uniform(.1, 3., size=(3, 160, 160))
    weights = token.cell_mass(masks, 4).reshape(20, -1)
    energy_weights = token.cell_mass(
        masks[:, None] * energy[None], 4).reshape(20, 3, -1)
    actual = grid_moments(features, weights, energy_weights)
    constant = np.broadcast_to(np.array([.25, 2., 7.])[:, None, None],
                               (3, 160, 160))
    constant_error = float(np.abs(pool(features, masks, constant)[0]).max())
    assert constant_error < 1e-10, constant_error
    permutation = rng.permutation(1600)
    permuted = grid_moments(features[permutation], weights[:, permutation],
                            energy_weights[:, :, permutation])
    permutation_error = max(float(np.abs(a-b).max())
                            for a, b in zip(actual, permuted))
    assert permutation_error < 1e-10, permutation_error
    # Independent pixel/grid overlap reference detects half-pixel normalization errors.
    pixels = np.arange(160)
    edges = geo.EDGES
    overlap = np.maximum(0., np.minimum(edges[1:, None], pixels[None]+.5)
                         - np.maximum(edges[:-1, None], pixels[None]-.5))
    pixel_features = np.einsum('iy,ijd,jx->yxd', overlap,
                               features.reshape(40, 40, 32), overlap, optimize=True)
    mass = masks.sum((-2, -1))
    den = np.maximum(mass, 1e-20)
    mean = np.einsum('ryx,yxd->rd', masks, pixel_features) / den[:, None]
    emean = np.einsum('ryx,byx->rb', masks, energy) / den[:, None]
    cross = np.einsum('ryx,byx,yxd->rbd', masks, energy, pixel_features,
                      optimize=True) / den[:, None, None]
    marginal = emean[..., None] * mean[:, None]
    reference_error = max(float(np.abs(a-b).max()) for a, b in
                          zip(actual, (cross-marginal, marginal, mean)))
    assert reference_error < 1e-10, reference_error
    assert weights[0, 0] == .5 and weights[0, 1] == .5
    mass_error = float(np.abs(weights.sum(-1)-mass).max())
    assert mass_error < 1e-10, mass_error
    # A content-only permutation preserves both marginals but changes cooccurrence.
    f = np.array([[0., 1.], [2., -1.]])
    w = np.array([[.5, .5]])
    ew = np.array([[[.5, 1.5], [1., 2.], [2., 3.]]])
    before = grid_moments(f, w, ew)
    after = grid_moments(f[::-1], w, ew)
    np.testing.assert_allclose(before[1], after[1], atol=1e-12)
    positional_change = float(np.abs(before[0]-after[0]).max())
    assert positional_change > 0
    return dict(constant_energy_covariance_max_abs=constant_error,
                simultaneous_grid_permutation_max_abs=permutation_error,
                independent_pixel_overlap_max_abs=reference_error,
                fractional_grid_mass_max_abs=mass_error,
                content_only_permutation_joint_change=positional_change,
                synthetic_invalid_slots_zero=all(np.count_nonzero(a[-1]) == 0
                                                  for a in actual), seed=1024)


class Extractor:
    def __init__(self, projection):
        self.model = token.resnet50(weights=None)
        self.model.load_state_dict(torch.load(geo.WEIGHTS, map_location='cpu',
                                              weights_only=True))
        self.model = self.model.cuda().eval().requires_grad_(False)
        self.mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
        self.center = torch.tensor(projection['mean'], device='cuda')[None, None, :]
        self.components = torch.tensor(projection['components'].T.copy(), device='cuda')

    @torch.inference_mode()
    def extract(self, images, masks, energies):
        x = torch.tensor(np.stack(images)[:, None], device='cuda')
        count = len(images)
        result = [np.zeros((count, 20, 3, 32)), np.zeros((count, 20, 3, 32)),
                  np.zeros((count, 20, 32))]
        for k in range(4):
            v = x.rot90(k, (-2, -1)).expand(-1, 3, -1, -1)
            z = self.model.relu(self.model.bn1(self.model.conv1((v-self.mean)/self.std)))
            z = self.model.layer1(self.model.maxpool(z))
            projected = ((z.flatten(2).transpose(1, 2)-self.center)
                         @ self.components).cpu().numpy().astype(np.float64)
            for b in range(count):
                values = pool(projected[b], np.rot90(masks[b], k, (-2, -1)),
                              np.rot90(energies[b], k, (-2, -1)))
                for accumulated, value in zip(result, values):
                    accumulated[b] += value / 4
        return result


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    table = pd.read_csv(PAIRED / 'table.csv', dtype={'instance_id': str, 'mmsi': str})
    ids = table.table_index.to_numpy(dtype=np.int64)
    assert len(ids) == 1805 and len(np.unique(ids)) == 1805
    with np.load(ROOT / 'projection_layer1.npz') as z:
        projection = {key: z[key].copy() for key in ('mean', 'components')}
    assert projection['mean'].shape == (256,) and projection['components'].shape == (32, 256)
    with np.load(ROOT / 'tokens.npz') as z:
        assert z['layer1'].shape[0] == 3387
        np.testing.assert_array_equal(z['instance_ids'][ids].astype(str), table.instance_id)
        valid, mass = z['valid'][ids].astype(bool), z['mass'][ids].copy()
        expected = ((z['layer1'][ids]-projection['mean']) @ projection['components'].T)
    active = valid[..., None] & (mass > 0)
    mass[~active] = 0
    expected[~active] = 0
    with np.load(PAIRED / 'views.npz') as z:
        np.testing.assert_array_equal(z['ids'], ids)
        np.testing.assert_array_equal(z['mass'], mass)
        np.testing.assert_array_equal(z['valid'], valid)
    snapshots = {int(s['obs_id']): s for s in json.loads(
        (geo.OLD / 'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))
        if int(s['obs_id']) in set(table.obs_id)}
    checks = targeted_checks()
    print('CHECKS', json.dumps(checks), flush=True)
    extractor = Extractor(projection)
    kernels = [[frequency.kernel(w, a) for a in frequency.ANGLES]
               for w in frequency.WAVELENGTHS]
    outputs = dict(joint=np.zeros((len(ids), 2, 20, 96), np.float32),
                   marginal=np.zeros((len(ids), 2, 20, 96), np.float32),
                   mean=np.zeros((len(ids), 2, 20, 32), np.float32))
    checks.update(projected_mean_replay_max_abs=0., mask_mass_relative_max=0.,
                  placement_pixel_max_abs=0.)
    last = time.perf_counter()
    extraction_started = last
    for first in range(0, len(ids), 8):
        images, masks, energies, addresses = [], [], [], []
        for j in range(first, min(first+8, len(ids))):
            for p, pol in enumerate(('vv', 'vh')):
                if not valid[j, p]:
                    continue
                pix, fp, g, placement = geo.native(table, snapshots, j, p)
                assert g is not None
                mask = token.slots(g, fp)
                actual_mass = mask.sum((-2, -1))
                np.testing.assert_array_equal(actual_mass > 0, active[j, p])
                error = float(np.max(np.abs(actual_mass-mass[j, p]) /
                                     np.maximum(actual_mass, 1e-20)))
                assert error < 1e-6, (int(ids[j]), pol, error)
                checks['mask_mass_relative_max'] = max(checks['mask_mass_relative_max'], error)
                with Image.open(table.iloc[j]['image_'+pol]) as image:
                    raw = np.array(image.convert('L'), dtype=np.float64) / 255
                pixel_error = float(np.abs(place(raw, placement)-pix/255).max())
                assert pixel_error == 0, (int(ids[j]), pol, pixel_error)
                images.append(pix.astype(np.float32)/255)
                masks.append(mask)
                energies.append(place(energy_field(raw, kernels), placement))
                addresses.append((j, p))
        if not addresses:
            continue
        values = extractor.extract(images, masks, energies)
        if first == 0:
            rotated = extractor.extract(
                [np.rot90(a) for a in images],
                [np.rot90(a, axes=(-2, -1)) for a in masks],
                [np.rot90(a, axes=(-2, -1)) for a in energies])
            checks['c4_simultaneous_rotation_max_abs'] = {
                key: float(np.abs(a-b).max())
                for key, a, b in zip(outputs, values, rotated)}
            assert max(checks['c4_simultaneous_rotation_max_abs'].values()) < 2e-4
            checks['c4_test_views'] = len(addresses)
            raw = np.arange(53*71, dtype=np.float64).reshape(53, 71) % 255 / 255
            rotation_error = float(np.abs(energy_field(np.rot90(raw), kernels)
                - np.rot90(energy_field(raw, kernels), axes=(-2, -1))).max())
            assert rotation_error < 1e-10, rotation_error
            checks['direction_averaged_gabor_rotation_max_abs'] = rotation_error
        for b, (j, p) in enumerate(addresses):
            for key, value in zip(outputs, values):
                outputs[key][j, p] = value[b].reshape(20, -1)
            error = float(np.max(np.abs(outputs['mean'][j, p, active[j, p]]
                                       - expected[j, p, active[j, p]]), initial=0))
            assert error < 2e-4, (int(ids[j]), p, error)
            checks['projected_mean_replay_max_abs'] = max(
                checks['projected_mean_replay_max_abs'], error)
        if first == 0 or time.perf_counter()-last >= 20:
            print('EXTRACT', min(first+8, len(ids)), '/', len(ids),
                  'seconds', round(time.perf_counter()-started, 1),
                  'replay', checks['projected_mean_replay_max_abs'], flush=True)
            last = time.perf_counter()
    extraction_seconds = time.perf_counter()-extraction_started
    for value in outputs.values():
        assert np.isfinite(value).all()
        assert np.count_nonzero(value[~active]) == 0
    checks.update(all_finite=True, invalid_slots_zero=True,
                  valid_only_replay_slots=int(active.sum()),
                  invalid_slots=int((~active).sum()))
    np.savez_compressed(HERE / 'features.npz', **outputs, mass=mass.astype(np.float32), ids=ids)
    with np.load(HERE / 'features.npz') as z:
        np.testing.assert_array_equal(z['ids'], ids)
        np.testing.assert_array_equal(z['mass'], mass)
        for key, value in outputs.items():
            np.testing.assert_array_equal(z[key], value)
    checks['saved_array_roundtrip'] = True
    receipt = dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        n=len(ids), original_rows=3387, valid_pol=int(valid.sum()),
        roles=table.role.value_counts().to_dict(), seconds=time.perf_counter()-started,
        extraction_seconds=extraction_seconds, device=torch.cuda.get_device_name(0),
        threads=2, batch_rows=8, backbone='frozen R50 layer1, eval, TF32 disabled',
        source_table=str(PAIRED/'table.csv'), projection=str(ROOT/'projection_layer1.npz'),
        weights=str(geo.WEIGHTS), replay_source=str(ROOT/'tokens.npz'),
        shapes={**{key: list(v.shape) for key, v in outputs.items()},
                'mass': list(mass.shape), 'ids': list(ids.shape)},
        dtype='joint/marginal/mean/mass=float32; ids=int64',
        ids='original 3387-row table_index, exact paired table order',
        axes='row, polarization [VV,VH], slot [8 body,8 side,4 end], feature',
        feature_order='wavelength-major [4,8,16], PCA-component-minor [0..31]',
        energy='log1p(mean_four_directions(abs(response)**2)/1e-4) per native pixel',
        directions_degrees=[0,45,90,135], direction_frame='fixed image; mean before log',
        native='full PNG L/255; existing reflect Gabor before original 160 crop/pad; no resize',
        projection_rule='F(u)=(layer1(u)-mean256) @ components32x256.T; no unit normalization',
        pooling='W=cell_mass(mask,4); V=cell_mass(mask*energy,4); denominator=sum(W)',
        moments='Fbar=W@F/sum(W); E[e]=sum(V)/sum(W); E[eF]=V@F/sum(W)',
        joint='C4mean(E[eF]-E[e]*Fbar)', marginal='C4mean(E[e]*Fbar)',
        mean='C4mean(Fbar); all invalid slots zero',
        c4='rotate image, masks and energy together; compute moments before averaging',
        precision='CNN/projection float32; fractional integrals/moments/C4 accumulator float64; save float32',
        grid_edges=geo.EDGES.tolist(), checks=checks, new_fits=0, new_classifiers=0,
        limitations=['Extraction only; no prediction or accuracy claim.',
                     'Filter support and CNN receptive fields cross region boundaries.',
                     'Gabor PNG response is not physical SLC phase.',
                     'Cache remains local; no Git operations.'])
    (HERE/'extraction.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print('COMPLETE', json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    with fft.set_workers(2):
        main()
