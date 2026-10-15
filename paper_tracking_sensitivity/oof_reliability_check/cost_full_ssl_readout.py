"""Measured complete neural extraction overhead of adding the fixed D18."""

import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from full_ssl_run import OUT, CACHE, OLD_CACHE, PIXELS, TARGETS, SIZE, backbone, load_r50
from full_ssl_run import views, normalize, rgb, FrequencyResidual, archive, write, frequency_embedding


@torch.inference_mode()
def main():
    if not (OUT/'READOUT_COMPLETE.json').exists():
        raise RuntimeError('Finish extraction and readout first')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    r50 = load_r50('D')
    r18, _ = backbone('r18')
    r18.load_state_dict(torch.load(OLD_CACHE/'models/r18_best.pt', map_location='cpu', weights_only=False)['backbone'])
    r18 = r18.cuda().eval()
    freq = FrequencyResidual('selective').cuda().eval()
    freq.load_state_dict(torch.load(CACHE/'D/frequency.pt', map_location='cpu', weights_only=False)['state'])
    records, selected = [], {}
    for ds in ('oss2', 'f0921'):
        design = archive(OUT/f'{ds}_design.npz')
        rows = np.flatnonzero(design['eval_masks'][0])[:16]
        selected[ds] = rows.tolist()
        pixels = np.load(PIXELS[ds], mmap_mode='r')
        resident = torch.from_numpy(np.array(pixels[rows], copy=True))
        conditions = torch.tensor(archive(TARGETS[ds]['design'])['conditions'][rows], dtype=torch.float32, device='cuda')
        for batch in (1, 16):
            def extract(dual):
                pair = resident[:batch].to(device='cuda', dtype=torch.float32)/255.
                zero, mean, spatial = views(r50, pair)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    embed, _ = frequency_embedding(freq, spatial, conditions[:batch])
                    if dual:
                        resized = F.interpolate(pair, size=(SIZE, SIZE), mode='bilinear', align_corners=False, antialias=True)
                        extra = r18(normalize(rgb(resized))).float().cpu().numpy()
                return zero.cpu().numpy(), mean.cpu().numpy(), embed.float().cpu().numpy()
            for dual in (False, True):
                for _ in range(3):
                    extract(dual)
            for repeat in range(12):
                for dual in ((False, True) if repeat % 2 == 0 else (True, False)):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    extract(dual)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter()-start)*1000.
                    records.append(dict(dataset=ds, batch=batch, repeat=repeat,
                        architecture='dual' if dual else 'single', batch_ms=elapsed, per_image_ms=elapsed/batch))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'cost_repeats.csv', index=False)
    summary = frame.groupby(['dataset', 'batch', 'architecture'])[['batch_ms', 'per_image_ms']].median().reset_index()
    summary.to_csv(OUT/'cost_summary.csv', index=False)
    common = sum(p.numel() for m in (r50, freq) for p in m.parameters())
    write(OUT/'COST.json', dict(device=torch.cuda.get_device_name(), selected_rows=selected,
        parameters={'single': common, 'dual': common+sum(p.numel() for p in r18.parameters())},
        scope='Actual uint8 chips in CPU RAM -> preprocess -> R50 C4 and frequency plus optional original-view D18 -> CPU features',
        excludes=['disk IO', 'source fitting', 'support fitting', 'CPU classification readout'],
        conditions='Known cohort ranks resident; D encoders, unchanged architecture relative G',
        all_models_resident=True, no_peak_memory_claim=True, interleaved_repeats=12))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
