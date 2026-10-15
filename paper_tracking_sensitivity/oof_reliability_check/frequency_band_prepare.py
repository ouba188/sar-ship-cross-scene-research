"""Source-only band probes, reusing the previously frozen unused-query plan."""

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
from frequency_band_core import unit_bands, old_kernel, score_embedding, choice_features

OUT = ROOT/'frequency_band_complementarity_20261009'


def prepare(folder, maps):
    info = json.loads((folder/'PREPARED.json').read_text())
    dst = OUT/'source'/info['dataset']/info['task']
    dst.mkdir(parents=True, exist_ok=True)
    if (dst/'PREPARED.json').exists():
        return
    previous = ROOT/'frequency_decision_20261009/source'/info['dataset']/info['task']
    d = load_npz(previous/'examples.npz')
    models = {}
    for mode in ('selective', 'fixed'):
        ckpt = torch.load(folder/f'{mode}.pt', map_location='cpu', weights_only=True)
        model = FrequencyResidual(mode).cuda().eval()
        model.load_state_dict(ckpt['state'])
        models[mode] = model
    allowed = np.asarray(ckpt['source_rows'])
    episodes = sorted((folder/'episodes').glob('*.npz'))
    scores = np.empty((len(d['labels']), 6, 8), np.float64)
    features = None
    started = time.perf_counter()
    for ei in np.unique(d['episode']):
        ep = archive(episodes[ei])
        mask = d['episode'] == ei
        q = pd.Index(ep['rows']).get_indexer(d['rows'][mask])
        if (q < 0).any() or np.intersect1d(q, ep['support']).size:
            raise ValueError('Source query mapping mismatch')
        ix = np.r_[ep['support'], q]
        rows = ep['rows'][ix]
        if np.setdiff1d(rows, allowed).size:
            raise ValueError('Outer excluded source row')
        data = {key: ep[key][ix] for key in ('phi', 'prior', 'mean', 'conditions')}
        embeds = {mode: [] for mode in models}
        with torch.inference_mode():
            for lo in range(0, len(rows), 128):
                x = torch.from_numpy(np.asarray(maps[rows[lo:lo+128]], dtype=np.float32)).cuda()
                c = torch.as_tensor(data['conditions'][lo:lo+128], dtype=torch.float32, device='cuda')
                for mode, model in models.items():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        z, _ = model(x, c)
                    embeds[mode].append(z.cpu().numpy())
        embeds = {m: np.concatenate(z).astype(float) for m, z in embeds.items()}
        support, sy = np.arange(64), ep['labels'][ix[:64]]
        np.testing.assert_array_equal(ep['labels'][q], d['labels'][mask])
        kernel, condition = old_kernel(data, support, info['eta'])
        reference, _ = score_embedding(data, embeds['selective'], support, sy, kernel, condition)
        np.testing.assert_allclose(reference[64:], d['new'][mask], atol=1e-10, rtol=1e-10)
        scores[mask, 0] = reference[64:]
        scores[mask, 1] = score_embedding(data, embeds['fixed'], support, sy, kernel, condition)[0][64:]
        bands = unit_bands(embeds['fixed'])
        for band in range(4):
            score, sim = score_embedding(data, bands[:, band], support, sy, kernel, condition)
            scores[mask, band+2] = score[64:]
            f, width = choice_features(reference[64:], score[64:], sim[64:], sy, data['conditions'][64:], band)
            if features is None:
                features = np.empty((len(scores), 4, f.shape[1]), np.float32)
            features[mask, band] = f
    np.savez_compressed(dst/'examples.npz', scores=scores, features=features,
        labels=d['labels'], episode=d['episode'], rows=d['rows'], weights=d['weights'])
    (dst/'PREPARED.json').write_text(json.dumps(dict(dataset=info['dataset'], task=info['task'],
        queries=len(scores), score_width=width, feature_width=features.shape[-1],
        source_plan=str(previous), representation_modes=['selective', 'fixed'],
        no_new_neural_training=True, original_source_scores_replayed=True,
        seconds=time.perf_counter()-started), indent=2), encoding='utf-8')
    print(f'BAND SOURCE {info["task"]}: {len(scores)} queries {time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    maps = np.load(EXP/'maps/self24ports/maps.npy', mmap_mode='r')
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        prepare(marker.parent, maps)
