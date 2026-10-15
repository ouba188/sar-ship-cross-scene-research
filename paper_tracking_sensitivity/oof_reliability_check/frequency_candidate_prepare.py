"""Reuse source query plans and frozen band heads for direct visual comparisons."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
import time
import numpy as np
import pandas as pd
import torch

from learned_frequency_model import FrequencyResidual
from train_learned_frequency import archive
from prepare_learned_frequency import ROOT, OUT as EXP, load_npz
from frequency_candidate_core import class_evidence, candidate_difference, training_pairs

OUT = ROOT/'frequency_candidate_20261009'
PREVIOUS = ROOT/'frequency_decision_20261009'


def prepare(folder, maps):
    info = json.loads((folder/'PREPARED.json').read_text())
    dst = OUT/'source'/info['dataset']/info['task']
    dst.mkdir(parents=True, exist_ok=True)
    if (dst/'PREPARED.json').exists():
        return
    src = PREVIOUS/'source'/info['dataset']/info['task']
    prior_info = json.loads((src/'PREPARED.json').read_text())
    d = load_npz(src/'examples.npz')
    episodes = sorted((folder/'episodes').glob('*.npz'))
    ckpt = torch.load(folder/'selective.pt', map_location='cpu', weights_only=True)
    model = FrequencyResidual('selective').cuda().eval()
    model.load_state_dict(ckpt['state'])
    candidate, pairs = [np.zeros((len(d['rows']), 12), np.float32) for _ in range(2)]
    eligible, train_valid = np.zeros(len(d['rows']), bool), np.zeros(len(d['rows']), bool)
    negative = np.zeros(len(d['rows']), np.int8)
    records, started = [], time.perf_counter()
    for ei in np.unique(d['episode']):
        ep = archive(episodes[ei])
        mask = d['episode'] == ei
        q = pd.Index(ep['rows']).get_indexer(d['rows'][mask])
        if (q < 0).any() or np.intersect1d(q, ep['support']).size:
            raise ValueError('Source query row mapping mismatch')
        ix = np.r_[ep['support'], q]
        rows, conditions = ep['rows'][ix], ep['conditions'][ix]
        chunks = []
        with torch.inference_mode():
            for lo in range(0, len(rows), 128):
                x = torch.from_numpy(np.asarray(maps[rows[lo:lo+128]], dtype=np.float32)).cuda()
                c = torch.as_tensor(conditions[lo:lo+128], dtype=torch.float32, device='cuda')
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    z, _ = model(x, c)
                chunks.append(z.cpu().numpy())
        z = np.concatenate(chunks)
        labels = ep['labels'][q]
        np.testing.assert_array_equal(labels, d['labels'][mask])
        e, counts = class_evidence(z[64:], z[:64], conditions[64:], conditions[:64], ep['labels'][ep['support']])
        candidate[mask], eligible[mask] = candidate_difference(e, counts,
            d['old'][mask].argmax(1), d['new'][mask].argmax(1))
        pairs[mask], train_valid[mask], negative[mask] = training_pairs(e, counts, d['old'][mask], d['new'][mask], labels)
        records.append(dict(episode=int(ei), port=str(ep['held_port']), n=int(mask.sum()),
            training_pairs=int(train_valid[mask].sum()), eligible_changes=int(eligible[mask].sum()),
            support_counts=counts.tolist()))
    np.savez_compressed(dst/'examples.npz', **d, candidate=candidate, pairs=pairs,
        eligible=eligible, train_valid=train_valid, negative_class=negative)
    meta = dict(dataset=info['dataset'], task=info['task'], episodes=records,
        source_only=True, inherited_source_plan=str(src), original_episode_count=len(prior_info['episodes']),
        seconds=time.perf_counter()-started)
    (dst/'PREPARED.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'VISUAL SOURCE {info["task"]}: {train_valid.sum()} training pairs, '
          f'{eligible.sum()} eligible changes, {time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    maps = np.load(EXP/'maps/self24ports/maps.npy', mmap_mode='r')
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        prepare(marker.parent, maps)
