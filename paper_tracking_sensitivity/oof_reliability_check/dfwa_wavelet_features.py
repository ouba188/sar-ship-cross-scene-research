"""Fixed image-level Haar evidence probe, not a DFWA-Net reproduction."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time

import numpy as np
import pywt

from c4_condition_descriptors import load_manifest, read_pair, MANIFESTS, ROOT


OUT = ROOT/'dfwa_informed_probe_20261009'
LOW_MASK = np.indices((2, 2, 3, 16))[2].ravel() == 0


def grid_mean(x):
    n, p, h, w = x.shape
    return x.reshape(n, p, 4, h//4, 4, w//4).mean(axis=(3, 5))


def one_orientation(x):
    levels = []
    ll = x
    for _ in range(2):
        ll, (horizontal, vertical, diagonal) = pywt.dwt2(
            ll, 'haar', mode='periodization', axes=(-2, -1))
        bands = [grid_mean(ll), np.sqrt(grid_mean(horizontal**2+vertical**2)),
                 np.sqrt(grid_mean(diagonal**2))]
        levels.append(np.stack(bands, axis=2).reshape(len(x), 2, 3, 16))
    return np.stack(levels, axis=2).reshape(len(x), 192)


def features(pairs):
    x = np.asarray(pairs, dtype=np.float32)/255.
    return np.mean([one_orientation(np.rot90(x, k, axes=(-2, -1))) for k in range(4)], axis=0)


def extract(dataset):
    folder = OUT/'features'/dataset
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'COMPLETE.json').exists():
        print('REUSE WAVELET '+dataset, flush=True)
        return
    frame = load_manifest(dataset)
    n = len(frame)
    pixels, paths = None, None
    if dataset == 'f0921':
        pixels = np.load(MANIFESTS[dataset].parent/'inputs128.npy', mmap_mode='r')
        np.testing.assert_array_equal(np.load(MANIFESTS[dataset].parent/'feature_rows.npy'), frame.row)
    else:
        keys = ['image_relpath_vv', 'image_relpath_vh'] if dataset == 'self24ports' else ['chip_path_vv', 'chip_path_vh']
        paths = list(frame[keys].itertuples(index=False, name=None))
    result = np.lib.format.open_memmap(folder/'features.npy', mode='w+', dtype='float32', shape=(n, 192))
    started = time.perf_counter()
    def read(i):
        return read_pair(paths[i])[0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for lo in range(0, n, 128):
            hi = min(n, lo+128)
            batch = np.asarray(pixels[lo:hi]) if pixels is not None else np.stack(list(pool.map(read, range(lo, hi))))
            result[lo:hi] = features(batch)
            if lo%4096 == 0 or hi == n:
                result.flush()
                print(f'WAVELET {dataset} {hi}/{n} {time.perf_counter()-started:.1f}s', flush=True)
    np.save(folder/'sample_ids.npy', frame.sample_id.to_numpy(str))
    result.flush()
    (folder/'COMPLETE.json').write_text(json.dumps(dict(
        rows=n, dimensions=192, low_dimensions=int(LOW_MASK.sum()), high_dimensions=int((~LOW_MASK).sum()),
        library='PyWavelets '+pywt.__version__, wavelet='haar', levels=2, grid=4,
        inputs='Existing paired VV/VH 128px uint8 display images, divided by255',
        low='Mean LL coefficients per4x4 spatial grid', high='RMS axial pair and diagonal coefficients per4x4 grid',
        layout='polarization2,level2,band3,grid16',
        c4='Average statistics over0/90/180/270display rotations',
        excludes='No WSA/WRA/Gabor/DDF training; no physical noise separation',
        seconds=time.perf_counter()-started), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['self24ports', 'oss2', 'f0921'],
                        choices=['self24ports', 'oss2', 'f0921'])
    for dataset in parser.parse_args().datasets:
        extract(dataset)
