"""Frozen OSS1 stem features with chip-centered, ship-axis-independent regions."""
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'oss1_joint_geometry_20261017'
EDGES = np.array([.16, .26, .37, .49])
ANGLES = np.arange(4) * np.pi / 4


def native_grid(h, w, size=56, pad=0):
    grid = np.arange(-pad, size + pad, dtype=np.float64)
    return np.meshgrid((grid + .25) * h / size - .5,
                       (grid + .25) * w / size - .5, indexing='ij')


def regions(d, radar=0., incidence=0., radar_valid=True, incidence_valid=True,
            size=56):
    h, w = float(d['native_h']), float(d['native_w'])
    radii = EDGES * min(h, w)
    pad = int(np.ceil(radii[-1] * size / min(h, w))) + 2
    yy, xx = native_grid(h, w, size, pad)
    x, y = xx - (w - 1) / 2, yy - (h - 1) / 2
    radius, phi = np.hypot(x, y), np.arctan2(y, x)
    crop = (slice(pad, pad + size), slice(pad, pad + size))
    core = (radius <= radii[0])[crop]
    weights, positions, coverages = [], [], []
    for scale in range(3):
        shell = (radius > radii[scale]) & (radius <= radii[scale + 1])
        midradius = (radii[scale] + radii[scale + 1]) / 2
        for angle in ANGLES:
            full = shell * np.exp(4 * (np.cos(2 * (phi - angle)) - 1))
            observed = full[crop]
            mass, full_mass = float(observed.sum()), float(full.sum())
            coverage = mass / full_mass if full_mass else 0.
            nonempty = mass > 1e-10
            weight = observed / mass if nonempty else np.zeros_like(observed)
            pos = np.zeros(16, np.float32)
            pos[:4] = [scale / 2, np.log1p(midradius) / 5, coverage, nonempty]
            pos[4:6] = [np.cos(2 * angle), np.sin(2 * angle)]
            pos[6:8] = np.log1p(2 * radii[0]) / 5
            if radar_valid:
                pos[8:10] = [np.cos(2 * (angle - radar)), np.sin(2 * (angle - radar))]
                pos[14] = 1
            if incidence_valid:
                pos[12:14] = [np.sin(incidence), np.cos(incidence)]
                pos[15] = 1
            weights.append(weight.astype(np.float32))
            positions.append(pos)
            coverages.append(coverage)
    return core, np.stack(weights), np.stack(positions), np.array(coverages)


def pool(feature, core, weights):
    feature = feature.float().flatten(2).transpose(1, 2)
    core = core.flatten(1)
    count = core.sum(1, keepdim=True)
    mean = (feature * core[:, :, None]).sum(1) / count.clamp_min(1)
    peak = feature.masked_fill(~core[:, :, None], -torch.inf).amax(1)
    peak = torch.where(count > 0, peak, torch.zeros_like(peak))
    tokens = torch.bmm(weights.flatten(2), feature)
    return torch.cat((mean, peak), 1), tokens


def prior_common():
    spec = importlib.util.spec_from_file_location('oss1_universal_previous', PREVIOUS / 'common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    prior = prior_common()
    from general_domain_ssl import backbone, normalize, rgb

    manifest = pd.read_csv(prior.MANIFEST, usecols=['sample_id', 'chip_path_vv', 'chip_path_vh'])
    n = len(manifest)
    ids = manifest.sample_id.to_numpy(str)
    dims = ['native_height_vv', 'native_width_vv', 'native_height_vh', 'native_width_vh']
    columns = ['global_row', 'sample_id', 'chip_path_vv', 'chip_path_vh',
               'radar_axis_rad', 'incidence_rad', 'radar_valid', 'incidence_valid'] + dims
    geometry = pd.read_csv(PREVIOUS / 'geometry.csv', usecols=columns).set_index('global_row').loc[np.arange(n)]
    np.testing.assert_array_equal(geometry.sample_id.to_numpy(str), ids)
    pixels = np.load(prior.PIXELS, mmap_mode='r')
    np.testing.assert_array_equal(np.load(prior.PIXELS.with_name('oss1_ids.npy')).astype(str), ids)
    if pixels.shape != (n, 2, 128, 128):
        raise ValueError(f'Unexpected frozen pixel shape: {pixels.shape}')
    cores, weights, positions, records = [], [], [], []
    valid = np.isfinite(pixels).all(axis=(1, 2, 3))
    for i, geo in geometry.iterrows():
        matched = geo.native_height_vv == geo.native_height_vh and geo.native_width_vv == geo.native_width_vh
        for pol in ('vv', 'vh'):
            path = Path(geo[f'chip_path_{pol}'])
            if path != Path(manifest.iloc[i][f'chip_path_{pol}']):
                raise ValueError(f'Native path mismatch at row {i}, {pol}')
            if not path.is_file():
                valid[i] = False
                continue
            with Image.open(path) as image:
                expected = (int(geo[f'native_width_{pol}']), int(geo[f'native_height_{pol}']))
                if image.size != expected:
                    raise ValueError(f'Native dimension mismatch at row {i}, {pol}')
                valid[i] &= bool(np.isfinite(np.asarray(image)).all())
        valid[i] &= matched
        core, mask, pos, coverage = regions(
            dict(native_h=geo.native_height_vv, native_w=geo.native_width_vv),
            float(geo.radar_axis_rad), float(geo.incidence_rad),
            bool(geo.radar_valid), bool(geo.incidence_valid))
        cores.append(core)
        weights.append(mask)
        positions.append(pos)
        records.append(dict(global_row=i, sample_id=ids[i], native_h=geo.native_height_vv,
                            native_w=geo.native_width_vv, valid=bool(valid[i]),
                            mean_coverage=float(coverage.mean()), min_coverage=float(coverage.min()),
                            empty_tokens=int((pos[:, 3] == 0).sum()), core_pixels=int(core.sum())))
    cores, weights, positions = np.stack(cores), np.stack(weights), np.stack(positions)
    pd.DataFrame(records).to_csv(HERE / 'universal_region_coverage.csv', index=False)
    preparation_seconds = time.perf_counter() - started
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timings = []
    print(f'PREPARED {n} samples; valid={valid.sum()}; device={device}', flush=True)
    for fold, cache, _, _ in prior.folds():
        fold_started = time.perf_counter()
        checkpoint = cache / 'source_only_r18_best.pt'
        model, _ = backbone('r18')
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['backbone'])
        stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu).to(device).eval()
        stem.requires_grad_(False)
        body_values, token_values = [], []
        for start in range(0, n, 32):
            end = min(start + 32, n)
            pair = torch.from_numpy(np.array(pixels[start:end], np.float32)).to(device) / 255
            pair[~torch.from_numpy(valid[start:end]).to(device)] = 0
            # Identical frozen-input preprocessing and stem precision to prior extract.py.
            image = normalize(rgb(F.interpolate(pair, (112, 112), mode='bilinear',
                                                align_corners=False, antialias=True)))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                feature = stem(image)
            body, tokens = pool(feature, torch.from_numpy(cores[start:end]).to(device),
                                torch.from_numpy(weights[start:end]).to(device))
            body_values.append(body.cpu().numpy())
            token_values.append(tokens.cpu().numpy())
        body, tokens = np.concatenate(body_values), np.concatenate(token_values)
        body[~valid], tokens[~valid] = 0, 0
        for name, value in [('body', body), ('tokens', tokens), ('positions', positions)]:
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f'{fold}: invalid {name}')
        output = HERE / fold
        output.mkdir(exist_ok=True)
        feature_path = output / 'universal_features.npz'
        np.savez_compressed(feature_path, body=body, tokens=tokens, positions=positions,
                            valid=valid.astype(bool), sample_ids=ids)
        seconds = time.perf_counter() - fold_started
        timings.append(dict(fold=fold, seconds=seconds, checkpoint=str(checkpoint), path=str(feature_path)))
        print(f'FEATURES_READY {fold} {seconds:.3f}s {feature_path}', flush=True)
        del stem, model
    receipt = dict(samples=n, valid=int(valid.sum()), device=str(device),
                   preparation_seconds=preparation_seconds, folds=timings,
                   total_seconds=time.perf_counter() - started,
                   pixels=str(prior.PIXELS), target_labels_used=False, backbone_retrained=False)
    (HERE / 'UNIVERSAL_FEATURES_TIMING.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('ALL_FEATURES_READY', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
