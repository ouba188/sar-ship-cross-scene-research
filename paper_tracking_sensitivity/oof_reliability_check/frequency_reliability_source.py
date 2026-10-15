"""Small CPU-only source screen using existing maps and frozen checkpoints."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from learned_frequency_model import FrequencyResidual
from frequency_reliability_core import (kernels, coefficients, strict_loo,
    ce_strength, balanced_accuracy, balanced_ce, select_rule, point_ce, local_strength)

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
EXP = ROOT / 'learned_frequency_residual_20261009'
OUT = ROOT / 'frequency_reliability_20261009'


def read(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def source_plan(episodes):
    # Reconstruct the exact original six-epoch sampling RNG, including ordering.
    rng = np.random.default_rng(20261009)
    used = [set() for _ in episodes]
    for _ in range(6):
        for ei in rng.permutation(len(episodes)):
            q = rng.choice(episodes[ei]['query'], min(128, len(episodes[ei]['query'])), replace=False)
            used[ei].update(q.tolist())
    available = [np.setdiff1d(ep['query'], sorted(u)) for ep, u in zip(episodes, used)]
    eligible = [i for i, q in enumerate(available) if len(q) >= 32]
    chosen = [eligible[i] for i in np.linspace(0, len(eligible)-1, min(4, len(eligible)), dtype=int)]
    if len(chosen) != 4:
        raise ValueError('Fewer than four eligible source episodes; protocol requires review')
    rng = np.random.default_rng(20261009)
    return [(i, np.sort(rng.choice(available[i], min(128, len(available[i])), replace=False))) for i in chosen], used


def process(folder, maps):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT / 'source' / info['dataset'] / info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'SELECTION.json').exists() and 'local_ba' in json.loads((out/'SELECTION.json').read_text())['episodes'][0]:
        print('REUSE SOURCE ' + info['task'], flush=True)
        return
    started = time.perf_counter()
    episodes = [read(p) for p in sorted((folder/'episodes').glob('*.npz'))]
    plan, used = source_plan(episodes)
    checkpoint = torch.load(folder/'selective.pt', map_location='cpu', weights_only=True)
    model = FrequencyResidual('selective').eval()
    model.load_state_dict(checkpoint['state'])
    allowed = np.asarray(checkpoint['source_rows'])
    records = []
    for ei, q in plan:
        ep = episodes[ei]
        ix = np.r_[ep['support'], q]
        rows = ep['rows'][ix]
        assert not set(q) & used[ei]
        assert not np.intersect1d(ep['support'], q).size
        assert not np.setdiff1d(rows, allowed).size
        data = {k: ep[k][ix] for k in ('phi', 'prior', 'mean', 'conditions')}
        cached_path = out/f'episode_{ei:02d}.npz'
        cached = read(cached_path) if cached_path.exists() else None
        if cached is not None:
            np.testing.assert_array_equal(cached['rows'], rows)
            embedding = cached['embedding'].astype(float)
        else:
            chunks = []
            with torch.inference_mode():
                for lo in range(0, len(rows), 32):
                    x = torch.from_numpy(np.asarray(maps[rows[lo:lo+32]], dtype=np.float32))
                    c = torch.as_tensor(data['conditions'][lo:lo+32], dtype=torch.float32)
                    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                        z, _ = model(x, c)
                    chunks.append(z.numpy())
            embedding = np.concatenate(chunks).astype(float)
        support = np.arange(64)
        sy, yq = ep['labels'][ix[:64]], ep['labels'][ix[64:]]
        k0, k1 = kernels(data, embedding, support, info['eta'])
        l0, l1 = (strict_loo(k0[:64], k1[:64], data['prior'][:64], sy)
            if cached is None else (cached['loo_old'], cached['loo_new']))
        gamma = ce_strength(l0, l1, sy)
        s0 = data['prior'] + k0 @ coefficients(k0[:64], sy, data['prior'][:64])
        s1 = data['prior'] + k1 @ coefficients(k1[:64], sy, data['prior'][:64])
        shrunk = s0 + gamma * (s1-s0)
        utility = point_ce(l0, sy) - point_ce(l1, sy)
        local_gamma = local_strength((k1-k0)[64:], utility)
        local_score = s0[64:] + local_gamma[:, None]*(s1-s0)[64:]
        records.append(dict(episode=ei, held_port=str(ep['held_port']), query_n=len(q),
            gamma=gamma, learned_ba=balanced_accuracy(s1[64:], yq),
            loo_ce_ba=balanced_accuracy(shrunk[64:], yq),
            local_ba=balanced_accuracy(local_score, yq),
            local_ce=balanced_ce(local_score, yq), local_gamma_mean=float(local_gamma.mean()),
            baseline_ba=balanced_accuracy(s0[64:], yq),
            learned_ce=balanced_ce(s1[64:], yq), loo_ce_ce=balanced_ce(shrunk[64:], yq),
            support_rows=ep['rows'][ep['support']].tolist(), query_rows=ep['rows'][q].tolist(),
            query_train_overlap=0, eta=float(info['eta'])))
        np.savez_compressed(out/f'episode_{ei:02d}.npz', rows=rows, labels=ep['labels'][ix],
            embedding=embedding.astype(np.float32), old_scores=s0, new_scores=s1,
            loo_old=l0, loo_new=l1, gamma=gamma, local_gamma=local_gamma, utility=utility)
    rule, gain = select_rule(records)
    result = dict(dataset=info['dataset'], task=info['task'], rule=rule,
        source_mean_delta_ba=gain, episodes=records, seconds=time.perf_counter()-started,
        selection='Equal-source-episode base BA among learned/loo_ce/local; ties in that order',
        precision='CPU BF16 convolution, FP32 normalization, FP64 ridge',
        limitation='Queries absent from neural training sample plan; other samples of same source ports trained the frozen adapter')
    (out/'SELECTION.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'SOURCE {info["task"]} {rule} deltaBA={gain:+.4f} {result["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(2)
    maps = np.load(EXP/'maps/self24ports/maps.npy', mmap_mode='r')
    for p in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        if not args.tasks or p.parent.name in args.tasks:
            process(p.parent, maps)
