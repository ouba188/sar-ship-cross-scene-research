"""Reuse frozen feature maps; retain original gates under band interventions."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from learned_frequency_model import FrequencyResidual, haar
from frequency_information_content import band_intervention

EXP = Path('E:/Hermes/paper_tracking/sensitivity/learned_frequency_residual_20261009')
OUT = Path('E:/Hermes/paper_tracking/sensitivity/frequency_information_content_20261009')
MODES = ('unsigned_bands', 'no_local_band_layout')


def encode(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'ENCODED.json').exists():
        return
    with np.load(folder/'selective_embedding.npz') as f:
        rows, gates, original = f['rows'], f['weights'], f['embedding']
    with np.load(folder/'target.npz') as f:
        conditions = f['conditions']
        np.testing.assert_array_equal(rows, f['rows'])
    maps = np.load(EXP/'maps'/info['dataset']/'maps.npy', mmap_mode='r')
    ckpt = torch.load(folder/'selective.pt', map_location='cpu', weights_only=True)
    model = FrequencyResidual('selective').cuda().eval()
    model.load_state_dict(ckpt['state'])
    arrays = {mode: np.empty_like(original) for mode in MODES}
    begin = time.perf_counter()
    replay_error = 0.
    with torch.inference_mode():
        for lo in range(0, len(rows), 128):
            ix = rows[lo:lo+128]
            x = torch.from_numpy(np.asarray(maps[ix], dtype=np.float32)).cuda()
            gate = torch.as_tensor(gates[lo:lo+len(ix)], device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if lo == 0:
                    c = torch.as_tensor(conditions[lo:lo+len(ix)], dtype=torch.float32, device='cuda')
                    replay, _ = model(x, c)
                    replay_error = float(np.max(np.abs(replay.cpu().numpy()-original[lo:lo+len(ix)])))
                    np.testing.assert_allclose(replay.cpu().numpy(), original[lo:lo+len(ix)], atol=2e-6, rtol=2e-6)
                bands = haar(x)
                for mode in MODES:
                    changed = band_intervention(bands, mode)
                    z = [block(changed[:, i]) for i, block in enumerate(model.blocks)]
                    vectors = torch.stack([F.normalize(F.adaptive_avg_pool2d(a, 2).flatten(1).float(), dim=1)
                                           for a in z], dim=1)
                    joined = (gate.sqrt().unsqueeze(-1)*vectors).flatten(1)
                    arrays[mode][lo:lo+len(ix)] = joined.cpu().numpy()
            if lo % 16384 == 0 or lo+len(ix) == len(rows):
                print(f'INFO ENCODE {info["task"]} {lo+len(ix)}/{len(rows)} {time.perf_counter()-begin:.1f}s', flush=True)
    for mode, values in arrays.items():
        np.savez_compressed(out/f'{mode}.npz', embedding=values, rows=rows)
    (out/'ENCODED.json').write_text(json.dumps(dict(dataset=info['dataset'], task=info['task'],
        modes=MODES, rows=len(rows), gate='Original per-query weights fixed',
        neural_retraining=False, replay_max_abs=replay_error, seconds=time.perf_counter()-begin), indent=2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        encode(marker.parent)
