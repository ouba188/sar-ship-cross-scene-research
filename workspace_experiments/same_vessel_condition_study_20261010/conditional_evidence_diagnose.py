"""Replay the bounded gate and separate prior versus support-geometry effects."""

import numpy as np
import pandas as pd
import torch
from conditional_evidence_gate import (
    BASE, CACHE, OUT, SUPPORTS, SEEDS, FactorNet, EvidenceGate,
    standardize, residual_head, metrics,
)


def score(phi, prior, y):
    beta, _, _ = residual_head(phi[:64], prior[:64], y, balanced=True)
    return prior[64:]+phi[64:]@beta


def main():
    torch.set_num_threads(1)
    dest = OUT/'path_diagnosis'
    dest.mkdir(exist_ok=True)
    meta = pd.read_csv(BASE/'observations.csv')
    fixed = torch.load(OUT/'fixed_classifier.pt', map_location='cpu', weights_only=True)
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], dtype=np.float32)
    h = (torch.from_numpy(raw)-fixed['center'])/fixed['scale']/np.sqrt(512)
    factor_stats = np.load(BASE/'relational_factors_20261010/source_split.npz')
    fit_ids = np.load(OUT/'source_split.npz')['fit']
    norm = ((raw-factor_stats['center'])/factor_stats['scale']).astype(np.float32)
    net = FactorNet()
    net.load_state_dict(torch.load(BASE/'relational_factors_20261010/relations.pt', map_location='cpu', weights_only=True))
    with torch.no_grad():
        context = torch.cat([net.condition(torch.from_numpy(norm[i:i+256])) for i in range(0, len(norm), 256)]).numpy()
    context = torch.tensor(standardize(context, fit_ids), dtype=torch.float32)
    rows, replays, gain_rows, per_class = [], [], [], []
    for seed in SEEDS:
        model = EvidenceGate(fixed['basis'])
        model.load_state_dict(torch.load(OUT/f'condition_{seed}.pt', map_location='cpu', weights_only=True))
        model.eval()
        for path in sorted(SUPPORTS.glob('*_split.npz')):
            task = path.stem.removesuffix('_split')
            z = np.load(path)
            support, query = z['support_obs_ids'], z['query_obs_ids']
            y = torch.from_numpy(z['support_labels']).long()
            ids = np.r_[support, query]
            truth = meta.iloc[query].class_id.to_numpy()
            port, draw = meta.iloc[query].port.iloc[0], int(task.rsplit('draw', 1)[1])
            with torch.no_grad():
                old = h[ids]
                new = model(old, context[ids])
                phi0 = torch.cat([old, torch.ones((len(ids), 1))], 1)
                phi1 = torch.cat([new, torch.ones((len(ids), 1))], 1)
                p0 = old@fixed['weight']+fixed['bias']
                p1 = new@fixed['weight']+fixed['bias']
                scores = {'P0R0': score(phi0, p0, y), 'P1R0': score(phi0, p1, y),
                    'P0R1': score(phi1, p0, y), 'P1R1': score(phi1, p1, y)}
                saved = np.load(OUT/f'{task}.npz')
                for mode, key in [('P0R0', 'baseline_-1_disabled'), ('P1R1', f'condition_{seed}_dynamic')]:
                    maximum = np.max(np.abs(scores[mode].numpy()-saved[key]))
                    disagreement = int((scores[mode].argmax(1).numpy() != saved[key].argmax(1)).sum())
                    assert disagreement == 0
                    np.testing.assert_allclose(scores[mode], saved[key], atol=1e-6, rtol=1e-5)
                    replays.append(dict(task=task, seed=seed, mode=mode, max_score_difference=maximum,
                        prediction_disagreement=disagreement))
                gain = model.gains(context[ids])
                mean_gain = model.gains(context[fit_ids]).mean(0, keepdim=True)
                mean_context_gain = model.gains(context[fit_ids].mean(0, keepdim=True))
                gain_rows.append(dict(task=task, seed=seed, gain_RMS=float(gain.square().mean().sqrt()),
                    mean_context_gain_RMS=float(mean_context_gain.square().mean().sqrt()),
                    mean_gain_RMS=float(mean_gain.square().mean().sqrt()),
                    update_RMS_ratio=float(torch.linalg.norm(new-old)/torch.linalg.norm(old))))
            for mode, s in scores.items():
                pred = s.argmax(1).numpy()
                rows.append(dict(task=task, port=port, draw=draw, seed=seed, path=mode, **metrics(pred, truth)))
                for c in np.unique(truth):
                    per_class.append(dict(task=task, port=port, seed=seed, path=mode, class_id=int(c),
                        support_count=int((y.numpy() == c).sum()), recall=float(100*(pred[truth == c] == c).mean())))
    results = pd.DataFrame(rows)
    results.to_csv(dest/'results.csv', index=False)
    summary = results.groupby('path')[['OA', 'BA']].mean().reset_index()
    summary.to_csv(dest/'summary.csv', index=False)
    results.groupby(['port', 'path'])[['OA', 'BA']].mean().reset_index().to_csv(dest/'ports.csv', index=False)
    pd.DataFrame(per_class).to_csv(dest/'classes.csv', index=False)
    pd.DataFrame(replays).to_csv(dest/'replay.csv', index=False)
    pd.DataFrame(gain_rows).to_csv(dest/'gains.csv', index=False)
    contrasts = []
    for (task, seed), g in results.groupby(['task', 'seed']):
        values = g.set_index('path')
        for metric in ('OA', 'BA'):
            base, prior, relation, both = [values.loc[k, metric] for k in ('P0R0', 'P1R0', 'P0R1', 'P1R1')]
            contrasts.append(dict(task=task, seed=seed, metric=metric, prior_effect=prior-base,
                relation_effect=relation-base, interaction=both-prior-relation+base, combined=both-base))
    pd.DataFrame(contrasts).to_csv(dest/'contrasts.csv', index=False)
    print(summary.to_string(index=False))
    print(results.groupby(['port', 'path'])[['OA', 'BA']].mean().to_string())
    print(pd.DataFrame(gain_rows).mean(numeric_only=True).to_string())


if __name__ == '__main__':
    main()
