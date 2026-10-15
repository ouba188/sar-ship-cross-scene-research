"""Frozen SARCLIP visual-only screening on the existing paired-pixel cohorts."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path('E:/Hermes/paper_tracking')
OUT = ROOT/'sensitivity/strong_single_sarclip_20261010'
CACHE = Path('D:/SAR_strong_single_cache_20261010')
PIXELS = Path('D:/SAR_completion_cache_20261010')
REPO = ROOT/'repos/SARCLIP'
CHECKPOINT = ROOT/'models/sarclip/BiliSakura_SARCLIP-ViT-L-14/model.safetensors'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def load_visual():
    # Load the official visual module without importing unrelated text/data loaders.
    package = types.ModuleType('sar_clip')
    package.__path__ = [str(REPO/'sar_clip')]
    sys.modules['sar_clip'] = package
    from sar_clip.transformer import VisionTransformer
    with torch.device('meta'):
        model = VisionTransformer(image_size=224, patch_size=14, width=1024,
                                  layers=24, heads=16, mlp_ratio=4, output_dim=768)
    with safe_open(CHECKPOINT, framework='pt', device='cpu') as handle:
        state = {key.removeprefix('vision_model.'): handle.get_tensor(key).to(
                     device='cuda', dtype=torch.bfloat16)
                 for key in handle.keys() if key.startswith('vision_model.')}
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model, len(state)


def encode(model, pixels):
    x = torch.as_tensor(np.array(pixels), device='cuda', dtype=torch.float32)
    x = x/255
    x = torch.cat((x, x.mean(dim=1, keepdim=True)), dim=1)
    x = torch.nn.functional.interpolate(x, (224, 224), mode='bilinear', align_corners=False)
    x = (2*x-1).to(torch.bfloat16)
    with torch.inference_mode():
        result = model(x).float()
    return result


def smoke():
    torch.set_num_threads(2)
    model, loaded = load_visual()
    pixels = np.load(PIXELS/'source_inputs128.npy', mmap_mode='r')
    batch = np.array(pixels[:16])
    encode(model, batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(4):
        feature = encode(model, batch)
    torch.cuda.synchronize()
    seconds = (time.perf_counter()-start)/4
    actual = feature.cpu().numpy()
    assert actual.shape == (16, 768) and np.isfinite(actual).all()
    assert np.min(np.linalg.norm(actual, axis=-1)) > 0
    assert float(np.std(actual, axis=0).mean()) > 0
    receipt = dict(loaded_visual_tensors=loaded, parameter_count=sum(p.numel() for p in model.parameters()),
                   feature_shape=list(actual.shape), pair_batch=16, pair_batch_seconds=seconds,
                   paired_chips_per_second=16/seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   input='Official Byte-pair layout VV,VH,mean(VV,VH); existing128 pixels resized224',
                   weights_dtype='bfloat16', normalization='unit pixels * 2 - 1',
                   same_weights_forward_per_chip=1, text_tower_loaded=False,
                   gpu=torch.cuda.get_device_name(), torch_version=torch.__version__)
    write(OUT/'SMOKE_joint.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def extract(datasets, batch):
    torch.set_num_threads(2)
    CACHE.mkdir(parents=True, exist_ok=True)
    model, loaded = load_visual()
    for dataset in datasets:
        output = CACHE/f'{dataset}_sarclip_joint.npy'
        complete = OUT/f'{dataset}_EXTRACTED.json'
        if complete.exists():
            print(f'reuse completed {dataset}', flush=True)
            continue
        if dataset == 'f0921':
            path = ROOT/'sensitivity/f0921_detector_recrop_20261007/inputs128.npy'
        else:
            path = PIXELS/f'{dataset}_inputs128.npy'
        pixels = np.load(path, mmap_mode='r')
        assert pixels.dtype == np.uint8 and pixels.shape[1:] == (2, 128, 128)
        features = np.lib.format.open_memmap(output, mode='w+', dtype=np.float32,
                                            shape=(len(pixels), 768))
        start = time.perf_counter()
        last = start
        for lo in range(0, len(pixels), batch):
            hi = min(lo+batch, len(pixels))
            z = encode(model, pixels[lo:hi]).cpu().numpy()
            if not np.isfinite(z).all():
                raise ValueError(f'Nonfinite {dataset} features at row {lo}')
            features[lo:hi] = z
            now = time.perf_counter()
            if now-last >= 25 or hi == len(pixels):
                print(f'{dataset} {hi}/{len(pixels)} {now-start:.1f}s', flush=True)
                last = now
        features.flush()
        write(complete, dict(dataset=dataset, shape=list(features.shape), input=str(path),
                            output=str(output), seconds=time.perf_counter()-start,
                            batch=batch, visual_keys=loaded, no_neural_training=True,
                            rotation_averaging=False, forwards_per_paired_chip=1,
                            input_layout='VV,VH,mean(VV,VH), matching official Byte-pair readtif',
                            boundary='Frozen external-pretrained screening, not matched-pretraining architecture proof'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['smoke', 'extract'])
    parser.add_argument('--datasets', nargs='+', default=['source', 'oss2'])
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    if args.action == 'smoke':
        smoke()
    else:
        extract(args.datasets, args.batch)
