"""Source-only decision examples with untouched target-query labels."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import time

import numpy as np
import torch

from learned_frequency_model import FrequencyResidual
from train_learned_frequency import archive
from frequency_reliability_source import source_plan
from frequency_reliability_core import kernels, coefficients, strict_loo
from frequency_decision_core import decision_features

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
EXP = ROOT/'learned_frequency_residual_20261009'
OUT = ROOT/'frequency_decision_20261009'


def prepare(folder, maps):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/'source'/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'PREPARED.json').exists():
        return
    episodes = [archive(p) for p in sorted((folder/'episodes').glob('*.npz'))]
    _, used = source_plan(episodes)
    ckpt = torch.load(folder/'selective.pt', map_location='cpu', weights_only=True)
    allowed = np.asarray(ckpt['source_rows'])
    model = FrequencyResidual('selective').cuda().eval()
    model.load_state_dict(ckpt['state'])
    rng, started, records = np.random.default_rng(20261009), time.perf_counter(), []
    arrays = {key: [] for key in ('features', 'old', 'new', 'labels', 'episode', 'rows', 'weights')}
    for ei, ep in enumerate(episodes):
        available = np.setdiff1d(ep['query'], sorted(used[ei]))
        if len(available) < 32:
            continue
        q = np.sort(rng.choice(available, min(256, len(available)), replace=False))
        ix = np.r_[ep['support'], q]
        rows = ep['rows'][ix]
        if np.intersect1d(q, sorted(used[ei])).size or np.intersect1d(q, ep['support']).size:
            raise ValueError('Query reused by neural fit or support')
        if np.setdiff1d(rows, allowed).size:
            raise ValueError('Outer excluded rows in gate examples')
        data = {k: ep[k][ix] for k in ('phi', 'prior', 'mean', 'conditions')}
        chunks = []
        with torch.inference_mode():
            for lo in range(0, len(rows), 128):
                x = torch.from_numpy(np.asarray(maps[rows[lo:lo+128]], dtype=np.float32)).cuda()
                c = torch.as_tensor(data['conditions'][lo:lo+128], dtype=torch.float32, device='cuda')
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    z, _ = model(x, c)
                chunks.append(z.cpu().numpy())
        embedding = np.concatenate(chunks).astype(float)
        sy, yq, support = ep['labels'][ix[:64]], ep['labels'][q], np.arange(64)
        k0, k1 = kernels(data, embedding, support, info['eta'])
        l0, l1 = strict_loo(k0[:64], k1[:64], data['prior'][:64], sy)
        s0 = data['prior']+k0@coefficients(k0[:64], sy, data['prior'][:64])
        s1 = data['prior']+k1@coefficients(k1[:64], sy, data['prior'][:64])
        f, names, width = decision_features(s0[64:], s1[64:], (k1-k0)[64:]/.25, sy, l0, l1)
        _, inv, counts = np.unique(yq, return_inverse=True, return_counts=True)
        weights = 1./(len(counts)*counts[inv])
        values = dict(features=f, old=s0[64:], new=s1[64:], labels=yq,
            episode=np.full(len(q), ei), rows=rows[64:], weights=weights)
        for key in arrays:
            arrays[key].append(values[key])
        records.append(dict(episode=ei, port=str(ep['held_port']), n=len(q),
            disagree=int(np.sum(s0[64:].argmax(1) != s1[64:].argmax(1))),
            available_query_n=len(available), neural_query_overlap=0))
    np.savez_compressed(out/'examples.npz', **{k: np.concatenate(v) for k, v in arrays.items()})
    (out/'PREPARED.json').write_text(json.dumps(dict(dataset=info['dataset'], task=info['task'],
        episodes=records, feature_names=names, decision_width=width, source_only=True,
        numerical_precision='GPU BF16 head, FP32 normalization, FP64 solve',
        seconds=time.perf_counter()-started), indent=2))
    print(f'DECISION SOURCE {info["task"]}: {sum(r["n"] for r in records)} queries, '
          f'{sum(r["disagree"] for r in records)} disagreements, {len(records)} ports, {time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    maps = np.load(EXP/'maps/self24ports/maps.npy', mmap_mode='r')
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        prepare(marker.parent, maps)
