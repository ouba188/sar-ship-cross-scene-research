"""Architecture-only extraction microbenchmark; no image I/O or accuracy claims."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, resnet18

OUT = Path(__file__).resolve().parent.parent/'completion_20261010'


def conv_macs(model, sample):
    values, handles = [], []
    def count(layer, args, output):
        values.append(output.numel()*(layer.in_channels//layer.groups)*np.prod(layer.kernel_size))
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(count))
    with torch.inference_mode():
        model(sample)
    for handle in handles:
        handle.remove()
    return int(sum(values))


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261010)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the explicitly GPU-labelled microbenchmark')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    b50, b18 = resnet50(weights=None), resnet18(weights=None)
    b50.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    r50 = nn.Sequential(*list(b50.children())[:-2], nn.AdaptiveAvgPool2d((2, 2))).cuda().eval()
    r18 = nn.Sequential(*list(b18.children())[:-2], nn.AdaptiveAvgPool2d((1, 1))).cuda().eval()
    counts = dict(r50=sum(p.numel() for p in r50.parameters()),
                  r18=sum(p.numel() for p in r18.parameters()))
    macs = dict(r50=conv_macs(r50, torch.randn(1, 2, 128, 128, device='cuda')),
                r18=conv_macs(r18, torch.randn(1, 3, 128, 128, device='cuda')))
    results = []
    for batch in (1, 64):
        x = torch.randn(batch, 2, 128, 128, device='cuda')
        def base_once():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return r50(x)
        def base_c4():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return sum(r50(torch.rot90(x, k, dims=(-2, -1))) for k in range(4))/4
        def ssl():
            return r18(x[:, 1:2].repeat(1, 3, 1, 1))
        def both():
            return base_c4(), ssl()
        functions = dict(r50_once=base_once, r50_c4=base_c4, ssl18_only=ssl, r50_c4_plus_ssl18=both)
        samples = {name: [] for name in functions}
        with torch.inference_mode():
            for fn in functions.values():
                for _ in range(5):
                    fn()
            torch.cuda.synchronize()
            rng = np.random.default_rng(87)
            for repeat in range(20):
                for name in rng.permutation(list(functions)):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = functions[name]()
                    torch.cuda.synchronize()
                    samples[name].append(1000*(time.perf_counter()-start))
                    del output
        for name, times in samples.items():
            results.append(dict(batch=batch, mode=name, median_batch_ms=float(np.median(times)),
                median_ms_per_image=float(np.median(times)/batch),
                p25_batch_ms=float(np.percentile(times, 25)), p75_batch_ms=float(np.percentile(times, 75))))
        print(f'GPU extraction microbenchmark batch={batch} completed', flush=True)
    value = dict(device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        model_weights='Random weights, same backbone operators/shapes; no checkpoint download or accuracy evaluation',
        input='Synthetic resident 128x128 VV/VH; SSL uses VH repeated into three channels',
        precision='R50 bfloat16 autocast; SSL18 float32; TF32 disabled, model storage float32',
        scope='GPU backbone extraction only; excludes disk, preprocessing, host transfers, frequency adapter, kernel construction and classifiers',
        c4='Four sequential R50 rotations, original direction shared with original-view R50 branch',
        parameters=counts, convolution_macs_per_single_image=macs,
        convolution_macs_definition='Multiply-accumulate count; BN, pooling, activations and copies excluded; not total FLOPs',
        timings=results, repeats=20, warmup=5)
    (OUT/'backbone_cost.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(value, indent=2), flush=True)


if __name__ == '__main__':
    main()
