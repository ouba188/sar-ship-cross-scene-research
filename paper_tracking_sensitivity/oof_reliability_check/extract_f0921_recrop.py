"""Extract both frozen encoders from the matched raw-raster recrop."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet18

from extract_opensarship2_transfer import WS, build_encoder, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if (out/'feature_recipe.json').exists():
        raise ValueError('completed features must not be overwritten')
    crop_recipe = json.loads((out/'crop_recipe.json').read_text())
    pixels = np.load(out/'inputs128.npy', mmap_mode='r')
    n = len(pixels)
    metadata = pd.read_csv(out/'crop_metadata.csv')
    changed = metadata.status.to_numpy() == 'detector_dynamic'
    original = Path(crop_recipe['source_run'])
    np.testing.assert_array_equal(np.load(out/'feature_rows.npy'),
                                  np.load(original/'feature_rows.npy'))
    old_base = np.load(original/'paired_r50.npy', mmap_mode='r')
    old_extra = np.load(original/'vh_ssl.npy', mmap_mode='r')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    base = build_encoder(device)
    ck = torch.load(WS/'e206_out/ssl_resnet18.pt', map_location='cpu', weights_only=True)
    net = resnet18(weights=None)
    missing, unexpected = net.load_state_dict(ck['backbone'], strict=False)
    if set(missing) != {'fc.weight', 'fc.bias'} or unexpected:
        raise ValueError('unexpected SSL checkpoint')
    ssl = torch.nn.Sequential(*list(net.children())[:-1]).to(device).eval()
    del ck, net
    paired = np.lib.format.open_memmap(out/'paired_r50.npy', mode='w+',
                                     dtype=np.float32, shape=(n, 2048))
    extra = np.lib.format.open_memmap(out/'vh_ssl.npy', mode='w+',
                                    dtype=np.float32, shape=(n, 512))
    started = time.monotonic()
    with torch.inference_mode():
        for lo in range(0, n, 128):
            hi = min(n, lo+128)
            active = changed[lo:hi]
            b = np.array(old_base[lo:hi])
            e = np.array(old_extra[lo:hi], dtype=np.float32)
            if np.any(active):
                x = torch.from_numpy(np.array(pixels[lo:hi][active])).to(device).float()/255.
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    spatial = base((x-.5)/.25).flatten(2).transpose(1, 2)
                b[active] = spatial.float().cpu().numpy().astype(np.float16).astype(np.float32).mean(1)
                e[active] = ssl(x[:, 1:2].repeat(1, 3, 1, 1)).flatten(1).cpu().numpy().astype(np.float16).astype(np.float32)
            if not np.isfinite(b).all() or not np.isfinite(e).all():
                raise ValueError('nonfinite extracted features')
            paired[lo:hi], extra[lo:hi] = b, e
            if lo == 0 or lo % 4096 == 0 or hi == n:
                print(f'BOTH ENCODERS {hi}/{n} elapsed={time.monotonic()-started:.1f}s', flush=True)
    paired.flush()
    extra.flush()
    write_json(out/'feature_recipe.json', dict(rows=n, crop_recipe=crop_recipe,
        base='same frozen paired VV/VH R50 V2; bfloat16 then spatial float16, pooled float32',
        extra='same frozen VH RGB e206 R18; float32 inference then float16 rounding as original cache',
        fallback='unchanged inputs reuse exactly their original paired/SSL features',
        newly_extracted_rows=int(changed.sum()),
        input='PIL bilinear128; base (uint8/255-.5)/.25; SSL uint8/255',
        ssl_pretraining='F0921 target corpus E:/F_0921_pool_v1; not independent encoder evidence',
        torch_version=torch.__version__, elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
