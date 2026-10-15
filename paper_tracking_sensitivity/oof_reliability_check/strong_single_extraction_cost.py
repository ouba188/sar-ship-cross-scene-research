"""Interleaved real-pixel neural-extraction timing; not full deployment cost."""

import json
import time
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, resnet50, ResNet50_Weights

from strong_single_sarclip import ROOT, OUT, PIXELS, load_visual, encode, write
from learned_frequency_model import FrequencyResidual


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sar, _ = load_visual()
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    paired = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
    with torch.no_grad():
        paired.weight.copy_(net.conv1.weight.mean(1, keepdim=True).repeat(1, 2, 1, 1)*1.5)
    net.conv1 = paired
    prefix = nn.Sequential(*list(net.children())[:6]).cuda().eval()
    tail = nn.Sequential(net.layer3, net.layer4, nn.AdaptiveAvgPool2d((2, 2))).cuda().eval()
    ssl = resnet18(weights=None)
    ws = 'E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/e206_out/ssl_resnet18.pt'
    ssl.load_state_dict(torch.load(ws, map_location='cpu', weights_only=True)['backbone'], strict=False)
    path = ROOT/'sensitivity/module_completion_20261008/ports_projection/Antwerp-Bruges/joint/layer4.pt'
    ssl.layer4.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    ssl_prefix = nn.Sequential(*list(ssl.children())[:-3]).cuda().eval()
    ssl_tail = ssl.layer4.cuda().eval()
    freq = FrequencyResidual('selective').cuda().eval()
    folder = ROOT/'sensitivity/learned_frequency_residual_20261009/tasks/self24ports/Antwerp-Bruges'
    freq.load_state_dict(torch.load(folder/'selective.pt', map_location='cpu', weights_only=True)['state'])
    with np.load(folder/'target.npz') as data:
        rows = data['rows'][:16]
        conditions = data['conditions'][:16]
    pixels = np.array(np.load(PIXELS/'source_inputs128.npy', mmap_mode='r')[rows])
    records = []
    for batch in (1, 16):
        pair = pixels[:batch]
        cond = torch.tensor(conditions[:batch], device='cuda', dtype=torch.float32)
        def r50(dual=False):
            pixel = torch.tensor(pair, device='cuda', dtype=torch.float32)
            x = (pixel/255-.5)/.25
            orbit = []
            for rotation in range(4):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    maps = prefix(torch.rot90(x, rotation, (-2, -1)))
                    z = tail(maps)
                    if dual and rotation == 0:
                        embedding, _ = freq(maps.half().float(), cond)
                        embedding = embedding.float().cpu().numpy()
                orbit.append(z.half().float().mean((-2, -1)).cpu().numpy())
            if dual:
                extra = ssl_tail(ssl_prefix(pixel[:, 1:2].repeat(1, 3, 1, 1)/255).half().float())
                return np.mean(orbit, axis=0), extra.mean((-2, -1)).cpu().numpy(), embedding
            return np.mean(orbit, axis=0)
        methods = dict(r50_c4=lambda: r50(False), current_dual_neural=lambda: r50(True),
                       sarclip_joint=lambda: encode(sar, pair).cpu().numpy())
        samples = {arm: [] for arm in methods}
        with torch.inference_mode():
            for function in methods.values():
                for _ in range(4):
                    function()
            rng = np.random.default_rng(20261010)
            for repeat in range(12):
                for arm in rng.permutation(list(methods)):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = methods[arm]()
                    torch.cuda.synchronize()
                    samples[arm].append(1000*(time.perf_counter()-started))
                    del result
        for arm, values in samples.items():
            records.append(dict(arm=arm, batch=batch, median_batch_ms=float(np.median(values)),
                median_ms_per_chip=float(np.median(values)/batch), p25_ms=float(np.percentile(values, 25)),
                p75_ms=float(np.percentile(values, 75)), repeats=values))
        print(f'completed isolated GPU extraction timing B{batch}', flush=True)
    count = lambda models: sum(p.numel() for model in models for p in model.parameters())
    write(OUT/'EXTRACTION_COST.json', dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        sample_rows=rows.tolist(), sample_dataset='24-port/Antwerp-Bruges, 16 fixed real paired pixels',
        scope='RAM uint8 -> input transfer/resize/normalize -> neural forward -> CPU feature arrays',
        excluded='Disk I/O, source geometry, kernel construction, support fit, classification and model initialization',
        limitation='All three models resident; do not interpret combined allocated memory as per-arm deployment footprint',
        precision='SARCLIP bf16 weights/activations; R50 bf16 autocast; SSL18 fp32; TF32 disabled',
        parameters=dict(r50_c4=count([prefix, tail]), current_dual_neural=count([prefix, tail, ssl_prefix, ssl_tail, freq]),
                        sarclip_joint=count([sar])),
        forwards=dict(r50_c4=4, current_dual_neural=5, sarclip_joint=1),
        timings=records, independent_processes_on_GPU=False))
    print(json.dumps([{k: v for k, v in r.items() if k != 'repeats'} for r in records], indent=2), flush=True)


if __name__ == '__main__':
    main()
