"""Frozen paired R50 layer2 maps; independent of learned-branch training/evaluation."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from threadpoolctl import threadpool_limits

from c4_condition_descriptors import load_manifest, read_pair, MANIFESTS, ROOT


OUT = ROOT/'learned_frequency_residual_20261009/maps'


def haar_bands(x):
    """Orthonormal Haar: (B,C,H,W) -> (B,4,C,H/2,W/2).

    Band order is LL, LH, HL, HH; letters refer to height then width.
    H uses even-minus-odd samples. Both spatial dimensions must be even.
    """
    if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError('Haar input must be BCHW with even height and width')
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return torch.stack((a+b+c+d, a-b+c-d, a+b-c-d, a-b-c+d), dim=1)*0.5


def inverse_haar(bands):
    """Invert haar_bands without detaching the autograd graph."""
    if bands.ndim != 5 or bands.shape[1] != 4:
        raise ValueError('Haar bands must have shape (B,4,C,H,W)')
    ll, lh, hl, hh = bands.unbind(dim=1)
    a, b = (ll+lh+hl+hh)*0.5, (ll-lh+hl-hh)*0.5
    c, d = (ll+lh-hl-hh)*0.5, (ll-lh-hl+hh)*0.5
    top = torch.stack((a, b), dim=-1).flatten(-2)
    bottom = torch.stack((c, d), dim=-1).flatten(-2)
    return torch.stack((top, bottom), dim=-2).flatten(-3, -2)


def build_encoder(device):
    """Match extract_full_six_spatial_features.build_encoder through layer2."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    checkpoint = Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/', 1)[-1]
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Existing ImageNetV2 weight cache required: {checkpoint}')
    backbone = resnet50(weights=weights)
    first = backbone.conv1
    paired = nn.Conv2d(2, first.out_channels, kernel_size=first.kernel_size,
                       stride=first.stride, padding=first.padding,
                       bias=first.bias is not None)
    with torch.no_grad():
        paired.weight.copy_(first.weight.mean(1, keepdim=True).repeat(1, 2, 1, 1)*1.5)
        if first.bias is not None:
            paired.bias.copy_(first.bias)
    backbone.conv1 = paired
    encoder = nn.Sequential(*list(backbone.children())[:6])
    return encoder.requires_grad_(False).to(device).eval()


def extract(dataset, encoder, device, batch_size=64, limit=None):
    if batch_size <= 0 or (limit is not None and limit <= 0):
        raise ValueError('batch_size and limit must be positive')
    frame = load_manifest(dataset)
    total = len(frame)
    n = total if limit is None else min(total, limit)
    if not n:
        raise ValueError(f'Empty manifest: {dataset}')
    # Limited runs never write into the full cache or claim full completion.
    folder = OUT/dataset if limit is None else OUT.parent/'smoke'/dataset
    marker = folder/('COMPLETE.json' if limit is None else 'SMOKE.json')
    if limit is None and marker.exists():
        info = json.loads(marker.read_text(encoding='utf-8'))
        maps = np.load(folder/'maps.npy', mmap_mode='r')
        if maps.shape != (total, 512, 16, 16) or maps.dtype != np.float16:
            raise ValueError(f'Completed map layout mismatch: {folder}')
        np.testing.assert_array_equal(np.load(folder/'sample_ids.npy'), frame.sample_id.to_numpy(str))
        print(f'REUSE MAPS {dataset} {total}', flush=True)
        return info

    pixels, paths = None, None
    if dataset == 'f0921':
        directory = MANIFESTS[dataset].parent
        pixels = np.load(directory/'inputs128.npy', mmap_mode='r')
        if pixels.shape != (total, 2, 128, 128) or pixels.dtype != np.uint8:
            raise ValueError('Unexpected F0921 pixel shape or dtype')
        np.testing.assert_array_equal(
            np.load(directory/'feature_rows.npy', mmap_mode='r'), frame.row.to_numpy())
    else:
        keys = (['image_relpath_vv', 'image_relpath_vh'] if dataset == 'self24ports'
                else ['chip_path_vv', 'chip_path_vh'])
        paths = list(frame.iloc[:n][keys].itertuples(index=False, name=None))

    folder.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    result = np.lib.format.open_memmap(folder/'maps.npy', mode='w+', dtype=np.float16,
                                       shape=(n, 512, 16, 16))
    started = last_print = time.perf_counter()

    def read(i):
        return read_pair(paths[i])[0]

    encoder.eval()
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for lo in range(0, n, batch_size):
            hi = min(n, lo+batch_size)
            images = (np.array(pixels[lo:hi], copy=True) if pixels is not None
                      else np.stack(list(pool.map(read, range(lo, hi)))))
            tensor = torch.from_numpy(images).to(device=device, dtype=torch.float32)
            tensor = (tensor/255.0-0.5)/0.25
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda'):
                maps = encoder(tensor)
            values = maps.float().cpu().numpy().astype(np.float16)
            if values.shape != (hi-lo, 512, 16, 16) or not np.isfinite(values).all():
                raise ValueError(f'Invalid layer2 output at {dataset} rows {lo}:{hi}')
            result[lo:hi] = values
            now = time.perf_counter()
            if lo == 0 or hi == n or now-last_print >= 20:
                result.flush()
                elapsed = time.perf_counter()-started
                print(f'MAPS {dataset} {hi}/{n} {elapsed:.1f}s {hi/elapsed:.1f} rows/s', flush=True)
                last_print = time.perf_counter()
    result.flush()
    del result
    np.save(folder/'sample_ids.npy', frame.sample_id.iloc[:n].to_numpy(str))
    seconds = time.perf_counter()-started
    info = dict(
        dataset=dataset, rows=n, manifest_rows=total, shape=[n, 512, 16, 16],
        dtype='float16', full_population=limit is None, limit=limit,
        manifest=str(MANIFESTS[dataset]), row_order='Unfiltered manifest order',
        encoder='ResNet50 IMAGENET1K_V2: conv1,bn1,relu,maxpool,layer1,layer2',
        paired_conv='RGB weight mean repeated twice, multiplied by 1.5',
        inputs='VV,VH; PIL L and bilinear resize128; F0921 existing uint8 inputs128 mmap',
        normalization='(x/255.0-0.5)/0.25', frozen=True, batchnorm='eval',
        autocast='bfloat16' if device.type == 'cuda' else 'disabled',
        target_labels_read=False, model_fits=0, c4_averaging=False,
        batch_size=batch_size, workers=4, cpu_threads_max=2, device=str(device),
        seconds=seconds, rows_per_second=n/seconds,
        map_bytes=(folder/'maps.npy').stat().st_size,
        created_utc=datetime.now(timezone.utc).isoformat())
    marker.write_text(json.dumps(info, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(info), flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=list(MANIFESTS), default=['self24ports'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error('--batch-size and --limit must be positive')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with threadpool_limits(limits=2):
        encoder = build_encoder(device)
        for dataset in args.datasets:
            extract(dataset, encoder, device, args.batch_size, args.limit)


if __name__ == '__main__':
    main()
