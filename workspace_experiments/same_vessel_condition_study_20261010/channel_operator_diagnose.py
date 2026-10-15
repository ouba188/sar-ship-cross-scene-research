"""Source-only localization of attenuation between channels, block and head."""

import numpy as np
import pandas as pd
from channel_operator import (BASE, OUT, build_blocks, forward_features,
                              scene_residuals, scene_energy, metrics)


def shared_positive_energy(values, meta, scale):
    groups = scene_residuals(values/scale, meta)
    ports = {}
    for (p, _), r in groups.items():
        cross = (r.sum(0)**2-(r*r).sum(0))/(len(r)*(len(r)-1))
        ports.setdefault(p, []).append(cross)
    diagonal = np.mean([np.mean(v, axis=0) for v in ports.values()], axis=0)
    return float(np.maximum(diagonal, 0).sum())


def run():
    import torch
    torch.set_num_threads(1)
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(OUT/'split.npz') as split:
        train = split['train_obs_ids']
    meta = obs.iloc[train].reset_index(drop=True)
    maps = np.load(OUT/'boundary_maps.npy', mmap_mode='r')
    indices = np.arange(len(train))
    block = build_blocks()[1]
    h = np.asarray(maps[:len(train)]).mean((2, 3))
    hscale = np.maximum(h.std(0), 1e-6)
    with np.load(OUT/'baseline.npz') as saved:
        center, scale, w0, b0 = (saved[k] for k in ('center', 'scale', 'coef', 'bias'))
    baseline_scores = None
    result = []
    # No new fitting, tuning, or target evaluation in this diagnostic.
    for arm in ('baseline', 'targeted_fixed', 'targeted_learned', 'random_learned_0',
                'random_learned_1', 'random_learned_2'):
        with np.load(OUT/f'{arm}.npz') as saved:
            g, w, b = (saved[k] for k in ('gate', 'coef', 'bias'))
        features = forward_features(block, maps, indices, torch.from_numpy(g))
        x = (features-center)/scale/np.sqrt(512)
        fixed, refit = x@w0+b0, x@w+b
        if baseline_scores is None:
            baseline_scores = refit.copy()
        result.append(dict(arm=arm,
            input_shared_energy=shared_positive_energy(h*g, meta, hscale),
            output_shared_energy=shared_positive_energy(features, meta, scale),
            original_head_scene_energy=scene_energy(fixed, meta),
            refitted_head_scene_energy=scene_energy(refit, meta),
            original_head_source_BA=metrics(fixed.argmax(1), meta.class_id.to_numpy())['BA'],
            refitted_head_source_BA=metrics(refit.argmax(1), meta.class_id.to_numpy())['BA'],
            original_head_score_change=float(np.linalg.norm(fixed-baseline_scores)/np.linalg.norm(baseline_scores)),
            refitted_head_score_change=float(np.linalg.norm(refit-baseline_scores)/np.linalg.norm(baseline_scores)),
            gate_std=float(g[g < .999999].std()) if (g < .999999).any() else 0.))
        print('DIAGNOSE', arm, flush=True)
    frame = pd.DataFrame(result)
    for col in ('input_shared_energy', 'output_shared_energy',
                'original_head_scene_energy', 'refitted_head_scene_energy'):
        frame[col+'_ratio'] = frame[col]/frame.loc[0, col]
    frame.to_csv(OUT/'stage_attribution.csv', index=False)
    print(frame.to_string(index=False))


if __name__ == '__main__':
    run()
