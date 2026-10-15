"""Batch-extract paired R50 features and reuse aligned, replay-checked SSL."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

from extract_opensarship2_transfer import WS, build_encoder, write_json

DATA = Path('E:/Hermes/paper_tracking')
CHIPS = DATA/'f0921_chips'
SSL = DATA/'f0921_ssl_align/f0921_ssl_clean.npz'


def select_cached_ssl(rows):
    with np.load(SSL) as d:
        cached_rows = d['row']
        order = np.argsort(cached_rows)
        positions = np.searchsorted(cached_rows[order], rows)
        if np.any(positions >= len(order)) or not np.array_equal(cached_rows[order][positions], rows):
            raise ValueError('target row is absent from the SSL cache')
        return d['x'][order[positions]]


def replay_ssl(vh, rows, cached, out, device):
    ck = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    net = resnet18(weights=None)
    missing, unexpected = net.load_state_dict(ck['backbone'], strict=False)
    if set(missing) != {'fc.weight', 'fc.bias'} or unexpected:
        raise ValueError('SSL checkpoint is not the expected e206 backbone')
    body = torch.nn.Sequential(*list(net.children())[:-1]).to(device).eval()
    transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
    selected = np.linspace(0, len(rows)-1, 8, dtype=int)
    with torch.inference_mode():
        batch = torch.stack([transform(Image.fromarray(np.asarray(vh[rows[i]])).convert('RGB'))
                             for i in selected]).to(device)
        current = body(batch).flatten(1).cpu().numpy()
    original = cached[selected].astype(np.float32)
    relative = np.linalg.norm(current-original, axis=1)/np.maximum(np.linalg.norm(current, axis=1), 1e-12)
    write_json(out/'ssl_cache_replay.json', dict(chip_rows=rows[selected].tolist(),
        relative_l2_by_row=relative.tolist(), max_abs_error=float(np.max(np.abs(current-original))),
        cache=str(SSL), storage_dtype=str(cached.dtype), checkpoint_epoch=int(ck['epoch']),
        acceptance='relative L2 <= 0.001, allowing existing float16 storage quantization'))
    if np.max(relative) > .001:
        raise ValueError('cached SSL differs from current encoder; do not silently reuse')
    print(f'SSL CACHE REPLAY max_relative_L2={relative.max():.6g}', flush=True)
    del body, net, ck, batch
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    out = args.out
    if (out/'feature_recipe.json').exists():
        raise ValueError('completed target feature extraction must not be overwritten')
    target = pd.read_csv(out/'target_manifest.csv', keep_default_na=False)
    rows = target['row'].to_numpy(dtype=np.int64)
    vv = np.load(CHIPS/'F0921_VV.npy', mmap_mode='r')
    vh = np.load(CHIPS/'F0921_VH.npy', mmap_mode='r')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    cached_ssl = select_cached_ssl(rows)
    replay_ssl(vh, rows, cached_ssl, out, device)
    np.save(out/'vh_ssl.npy', cached_ssl)
    del cached_ssl
    progress_path = out/'feature_progress.json'
    begin = 0
    if progress_path.exists():
        if not np.array_equal(np.load(out/'feature_rows.npy'), rows):
            raise ValueError('partial feature cache belongs to another row ordering')
        begin = int(json.loads(progress_path.read_text())['completed_rows'])
    else:
        np.save(out/'feature_rows.npy', rows)
    features = np.lib.format.open_memmap(out/'paired_r50.npy', mode='r+' if begin else 'w+',
                                        dtype=np.float32, shape=(len(rows), 2048))
    encoder = build_encoder(device)

    def read_pair(row):
        return np.stack([np.asarray(Image.fromarray(np.asarray(band[row])).resize(
            (128, 128), Image.Resampling.BILINEAR)) for band in (vv, vh)])

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for lo in range(begin, len(rows), args.batch_size):
            hi = min(lo+args.batch_size, len(rows))
            pixels = np.stack(list(pool.map(read_pair, rows[lo:hi])))
            tensor = torch.from_numpy(pixels).to(device).float()
            tensor = (tensor/255.-.5)/.25
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                spatial = encoder(tensor).flatten(2).transpose(1, 2)
            pooled = spatial.float().cpu().numpy().astype(np.float16).astype(np.float32).mean(1)
            if not np.isfinite(pooled).all():
                raise ValueError('nonfinite paired features')
            features[lo:hi] = pooled
            if lo == begin or ((lo-begin)//args.batch_size) % 32 == 0 or hi == len(rows):
                features.flush()
                elapsed = time.monotonic()-start
                write_json(progress_path, dict(completed_rows=hi, total_rows=len(rows),
                                               elapsed_seconds=elapsed))
                print(f'PAIRED {hi}/{len(rows)} elapsed={elapsed:.1f}s', flush=True)
    features.flush()
    write_json(out/'feature_recipe.json', dict(rows=len(rows), base='paired VV/VH two-channel R50 V2, 2048D',
        base_input='uint8 memmap chips -> PIL bilinear128; (x/255-.5)/.25; bfloat16/spatial-float16, mean-float32',
        base_cache='paired_r50.npy float32, aligned to feature_rows.npy',
        extra='VH e206 SSL R18 512D, existing float16 cache reused after eight-row replay',
        extra_cache='vh_ssl.npy float16, converted to float32 before source PCA',
        cached_separate_R50_reused=False, chip_arrays=str(CHIPS),
        ssl_pretraining='F0921 target-corpus exposure: E:/F_0921_pool_v1; not an unseen-encoder dataset',
        torch_version=torch.__version__))
    print('F0921 FEATURES SAVED', flush=True)


if __name__ == '__main__':
    main()
