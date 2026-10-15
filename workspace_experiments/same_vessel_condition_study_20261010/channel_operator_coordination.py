"""Fixed two-factor follow-up: internal gate crossed with scene-aware head fit."""

import json
import numpy as np
import pandas as pd
from channel_operator import (BASE, OUT, PORTS, build_blocks, forward_features,
    scene_residuals, scene_energy, sufficient, fit_head, sample_weights, metrics, save_json)
from scene_training import directions


def covariance(values, meta):
    groups = scene_residuals(values, meta)
    sums, counts = {}, {}
    for (p, _), r in groups.items():
        total = r.sum(0)
        cross = (np.outer(total, total)-r.T@r)/(len(r)*(len(r)-1))
        if p not in sums:
            sums[p] = np.zeros_like(cross, dtype=np.float64)
            counts[p] = 0
        sums[p] += cross
        counts[p] += 1
    result = np.mean([sums[p]/counts[p] for p in sums], axis=0)
    return (result+result.T)/2


def run():
    import torch
    torch.set_num_threads(1)
    dest = OUT/'coordination'
    dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        raise ValueError('Do not overwrite completed follow-up')
    save_json(dest/'PROTOCOL.json', dict(
        reason='Stage attribution locates loss of suppression during classification-head refit',
        design='Gate absent/present x ordinary/scene-regularized source head; original gates unchanged',
        rank=8, tau=.1, parameters='Same rank/strength as previous scene-training pilot; no search',
        fit='Source-only class/identity-weighted ridge plus trace-normalized PSD shared-scene penalty',
        query_labels='Only for evaluation after all fixed choices; no per-port winner selection',
        limitations='Sequential development follow-up, not independent confirmation; not gate retraining through added penalty'))
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(OUT/'split.npz') as split:
        train, test = split['train_obs_ids'], split['test_obs_ids']
    tm, qm = obs.iloc[train].reset_index(drop=True), obs.iloc[test].reset_index(drop=True)
    maps = np.load(OUT/'boundary_maps.npy', mmap_mode='r')
    block = build_blocks()[1]
    with np.load(OUT/'baseline.npz') as saved:
        center, scale = saved['center'], saved['scale']
        baseline_pred = saved['predictions']
    n = len(train)
    results, sensitivity, class_results, replay = [], [], [], []
    for gate_arm in ('baseline', 'targeted_learned'):
        with np.load(OUT/f'{gate_arm}.npz') as saved:
            gate = saved['gate']
            original_predictions = saved['predictions']
            original_w, original_b = saved['coef'], saved['bias']
        features = forward_features(block, maps, np.arange(len(train)+len(test)), torch.from_numpy(gate))
        x = ((features-center)/scale/np.sqrt(512)).astype(np.float64)
        stats = sufficient(x[:n], tm.class_id.to_numpy(), sample_weights(tm))
        _, scene, info = directions(covariance(x[:n], tm), rank=8)
        penalty = np.eye(512)+.1*np.trace(stats[0])/max(np.trace(scene), 1e-12)*scene
        w, b = fit_head(stats, np.eye(512), penalty)
        for head, coef, bias in [('ordinary', original_w, original_b), ('scene', w, b)]:
            score = x@coef+bias
            pred = score[n:].argmax(1)
            if head == 'ordinary':
                mismatches = int((pred != original_predictions).sum())
                replay.append(dict(gate=gate_arm, disagreements=mismatches))
                if mismatches:
                    raise ValueError('Original diagonal did not replay')
            arm = gate_arm+'__'+head
            sensitivity.append(dict(arm=arm, source_scene_energy=scene_energy(score[:n], tm),
                source_BA=metrics(score[:n].argmax(1), tm.class_id.to_numpy())['BA']))
            for port in PORTS:
                q = qm.port.eq(port).to_numpy()
                y = qm.class_id.to_numpy()[q]
                before, after = baseline_pred[q] == y, pred[q] == y
                results.append(dict(arm=arm, port=port, **metrics(pred[q], y),
                    rescue=int((~before & after).sum()), harm=int((before & ~after).sum())))
                for c in np.unique(y):
                    cc = y == c
                    class_results.append(dict(arm=arm, port=port, class_id=int(c), n=int(cc.sum()),
                        recall=float(100*after[cc].mean()), rescue=int((~before & after & cc).sum()),
                        harm=int((before & ~after & cc).sum())))
            np.savez_compressed(dest/f'{arm}.npz', coef=coef, bias=bias, predictions=pred, gate=gate)
            print(arm, json.dumps(results[-3:]), flush=True)
    pd.DataFrame(results).to_csv(dest/'results.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(dest/'sensitivity.csv', index=False)
    pd.DataFrame(class_results).to_csv(dest/'by_class.csv', index=False)
    save_json(dest/'COMPLETE.json', dict(arms=4, diagonal_replay=replay,
        no_gate_retraining=True, source_only=True))


if __name__ == '__main__':
    run()
