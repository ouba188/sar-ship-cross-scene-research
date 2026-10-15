"""Matched source retraining separates removed information from head mismatch."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from learned_frequency_model import FrequencyResidual, haar, support_scores
from frequency_information_content import band_intervention
from train_learned_frequency import archive

EXP = Path('E:/Hermes/paper_tracking/sensitivity/learned_frequency_residual_20261009')
OUT = Path('E:/Hermes/paper_tracking/sensitivity/frequency_information_content_20261009/retrained')
MODES = ('unsigned_bands', 'no_local_band_layout')


class InformationResidual(FrequencyResidual):
    def __init__(self, intervention):
        super().__init__('selective')
        self.intervention = intervention

    def forward(self, x, conditions):
        bands = band_intervention(haar(x), self.intervention)
        z = [block(bands[:, i]) for i, block in enumerate(self.blocks)]
        vectors = torch.stack([F.normalize(F.adaptive_avg_pool2d(a, 2).flatten(1).float(), dim=1)
                               for a in z], dim=1)
        context = torch.cat([a.mean((-1, -2)) for a in z]+[conditions], dim=1)
        weights = self.gate(context).float().softmax(1)
        return (weights.sqrt().unsqueeze(-1)*vectors).flatten(1), weights


def process(folder, maps):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'ENCODED.json').exists():
        return
    target = archive(folder/'target.npz')
    episodes = [archive(p) for p in sorted((folder/'episodes').glob('*.npz'))]
    if any(np.setdiff1d(ep['rows'], target['source_rows']).size for ep in episodes):
        raise ValueError('Excluded source rows')
    rng, plan = np.random.default_rng(20261009), []
    for epoch in range(6):
        batches = []
        for ei in rng.permutation(len(episodes)):
            ep = episodes[ei]
            q = rng.choice(ep['query'], min(128, len(ep['query'])), replace=False)
            batches.append((ei, np.r_[ep['support'], q]))
        plan.append(batches)
    used = np.unique(np.concatenate([episodes[ei]['rows'][ix] for batches in plan for ei, ix in batches]))
    lookup = np.full(len(maps), -1, dtype=np.int64)
    lookup[used] = np.arange(len(used))
    cached = torch.empty((len(used), 512, 16, 16), dtype=torch.float16, device='cuda')
    for lo in range(0, len(used), 256):
        ix = used[lo:lo+256]
        cached[lo:lo+len(ix)].copy_(torch.from_numpy(np.array(maps[ix], copy=True)))
    eps = [{k: torch.as_tensor(ep[k], device='cuda', dtype=torch.long if k == 'labels' else torch.float32)
            for k in ('phi', 'prior', 'mean', 'conditions', 'labels')} for ep in episodes]
    for mode in MODES:
        path = out/f'{mode}.pt'
        if path.exists():
            continue
        torch.manual_seed(20261009)
        model = InformationResidual(mode).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
        logs, started = [], time.perf_counter()
        for epoch, batches in enumerate(plan):
            losses = []
            for ei, ix in batches:
                local = torch.as_tensor(ix, device='cuda')
                ri = torch.as_tensor(lookup[episodes[ei]['rows'][ix]], device='cuda')
                x = cached[ri].float()
                args = [eps[ei][key][local] for key in ('phi', 'prior', 'mean', 'conditions')]
                sy, yq = eps[ei]['labels'][local[:64]], eps[ei]['labels'][local[64:]]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    embedding, weights = model(x, args[-1])
                score = support_scores(*args, embedding, sy, info['eta'])
                counts = torch.bincount(yq, minlength=8).clamp_min(1)
                balance = 1./counts[yq].float()
                loss = (F.cross_entropy(score[64:], yq, reduction='none')*balance).sum()/balance.sum()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite source objective')
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5))
                optimizer.step()
                losses.append(float(loss.detach()))
            logs.append(dict(epoch=epoch+1, query_ce=float(np.mean(losses)), gradient_norm=norm))
        torch.save(dict(state=model.state_dict(), mode=mode, source_rows=target['source_rows'].tolist()), path)
        (out/f'{mode}_training.json').write_text(json.dumps(dict(epochs=logs,
            seconds=time.perf_counter()-started, same_seed_and_plan=True, source_only=True,
            trainable_parameters=sum(p.numel() for p in model.parameters())), indent=2))
        print(f'RETRAIN {info["task"]} {mode} {time.perf_counter()-started:.1f}s', flush=True)
        del model, optimizer
    del cached, eps
    torch.cuda.empty_cache()
    target_maps = np.load(EXP/'maps'/info['dataset']/'maps.npy', mmap_mode='r')
    models, vectors, gates = {}, {}, {}
    for mode in MODES:
        models[mode] = InformationResidual(mode).cuda().eval()
        models[mode].load_state_dict(torch.load(out/f'{mode}.pt', map_location='cpu', weights_only=True)['state'])
        vectors[mode] = np.empty((len(target['rows']), 256), dtype=np.float32)
        gates[mode] = np.empty((len(target['rows']), 4), dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for lo in range(0, len(target['rows']), 128):
            ix = target['rows'][lo:lo+128]
            x = torch.from_numpy(np.asarray(target_maps[ix], dtype=np.float32)).cuda()
            c = torch.as_tensor(target['conditions'][lo:lo+len(ix)], dtype=torch.float32, device='cuda')
            for mode in MODES:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    z, w = models[mode](x, c)
                vectors[mode][lo:lo+len(ix)], gates[mode][lo:lo+len(ix)] = z.cpu().numpy(), w.cpu().numpy()
            if lo % 32768 == 0 or lo+len(ix) == len(target['rows']):
                print(f'RETRAIN ENCODE {info["task"]} {lo+len(ix)}/{len(target["rows"])} {time.perf_counter()-started:.1f}s', flush=True)
    for mode in MODES:
        np.savez_compressed(out/f'{mode}.npz', embedding=vectors[mode], weights=gates[mode], rows=target['rows'])
    (out/'ENCODED.json').write_text(json.dumps(dict(dataset=info['dataset'], task=info['task'],
        modes=MODES, target_neural_fit=False, source_epochs=6, final_epoch=True, no_target_selection=True), indent=2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = np.load(EXP/'maps/self24ports/maps.npy', mmap_mode='r')
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        process(marker.parent, source)
