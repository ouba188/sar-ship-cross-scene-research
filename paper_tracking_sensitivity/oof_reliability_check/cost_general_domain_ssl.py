"""Real-chip, interleaved extraction timing for the fixed G/D pilot."""

import json
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from general_domain_readout import PRIMARY_MODES
from general_domain_ssl import CACHE, OUT, PIXELS, SIZE, backbone, normalize, rgb


@torch.inference_mode()
def main():
    if not (OUT / 'ALL_FEATURES_COMPLETE.json').exists():
        raise RuntimeError('Finish the main GPU experiment before timing')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = {}
    for domain in ('G', 'D'):
        for arch in ('r18', 'r50'):
            model, _ = backbone(arch)
            if domain == 'D':
                state = torch.load(CACHE / 'models' / f'{arch}_best.pt', map_location='cpu', weights_only=False)
                model.load_state_dict(state['backbone'], strict=True)
            models[f'{domain}{arch[1:]}'] = model.cuda().eval()
    records, selected = [], {}
    for ds in ('oss2', 'f0921'):
        with np.load(OUT / f'{ds}_design.npz') as saved:
            local = np.flatnonzero(saved['eval_masks'][0])[:16]
            rows = saved['rows'][local]
        selected[ds] = rows.tolist()
        raw = np.load(PIXELS[ds], mmap_mode='r')
        resident = torch.from_numpy(np.array(raw[rows], copy=True))
        for batch in (1, 16):
            def extract(mode):
                pair = resident[:batch].to(device='cuda', dtype=torch.float32) / 255.
                pair = F.interpolate(pair, size=(SIZE, SIZE), mode='bilinear', align_corners=False, antialias=True)
                image = normalize(rgb(pair))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    values = [models[a](image).float().cpu().numpy() for a in PRIMARY_MODES[mode]]
                return values
            modes = list(PRIMARY_MODES)
            for mode in modes:
                for _ in range(3):
                    extract(mode)
            for repeat in range(12):
                order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
                if repeat % 2:
                    order.reverse()
                for mode in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    extract(mode)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter()-start)*1000.
                    records.append(dict(dataset=ds, batch=batch, repeat=repeat, mode=mode,
                                        batch_ms=elapsed, per_image_ms=elapsed/batch))
    table = pd.DataFrame(records)
    table.to_csv(OUT / 'cost_repeats.csv', index=False)
    summary = table.groupby(['dataset', 'batch', 'mode'], sort=False).agg(
        median_batch_ms=('batch_ms', 'median'), median_image_ms=('per_image_ms', 'median'))
    summary.to_csv(OUT / 'cost_summary.csv')
    result = dict(device=torch.cuda.get_device_name(), selected_rows=selected,
        parameters={m: sum(sum(p.numel() for p in models[a].parameters()) for a in archs)
                    for m, archs in PRIMARY_MODES.items()},
        scope='Real uint8 chips resident in CPU RAM -> shared RGB112 preprocessing -> backbone(s) -> CPU GAP arrays',
        excludes=['disk IO', 'model loading', 'source/support head fitting', 'classification readout'],
        all_models_resident=True, no_claim_of_peak_memory_comparison=True,
        batch_sizes=[1, 16], interleaved_repeats=12)
    (OUT / 'COST.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
