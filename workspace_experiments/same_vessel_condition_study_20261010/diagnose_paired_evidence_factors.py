"""Locate loss/validation tradeoffs and the information retained by each factor."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from run_paired_evidence_factors import OUT, OLD, PARENT, ORIGINS, METHODS, SEEDS
from train_explicit_candidate import load_data, metrics, save_json
from paired_evidence_factors import (
    row_weights, paired_covariance, encode_raw, score_probe, support_operator, adapted_scores,
)


def episode_loss(p, kernel, labels, episodes):
    losses = []
    for s, q in episodes:
        op, present = support_operator(kernel[s], kernel[q], labels[s])
        pred = adapted_scores(p[s], p[q], op, present, labels[s])
        counts = np.bincount(labels[q], minlength=8)
        weights = len(q)/((counts > 0).sum()*counts[labels[q]])
        target = 2*np.eye(8)[labels[q]]-1
        losses.append(float((weights[:, None]*(pred-target)**2).mean()))
    return float(np.mean(losses))


def main():
    torch.set_num_threads(1)
    out = OUT/'mechanism'; out.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    fit, held = split['factor_fit'], split['test_source']
    labels = meta.class_id.to_numpy()
    checks, objectives, classes, drifts, degeneracy = [], [], [], [], []
    validation = {}
    with np.load(OLD/'validation_episodes.npz') as f:
        for role in ('val_new_ship', 'val_new_time'):
            validation[role] = [(f[f'{role}_{j}_support'], f[f'{role}_{j}_query']) for j in range(3)]
    for origin in ORIGINS:
        for seed in SEEDS:
            with np.load(PARENT/f'{origin}_{seed}_vectors.npz') as f:
                z = f['z']
            state = torch.load(PARENT/f'{origin.removeprefix("split_")}_{seed}.pt', map_location='cpu', weights_only=True)['selected']
            p = F.linear(torch.from_numpy(z), state['head.weight'], state['head.bias']).numpy().astype(np.float64)
            with np.load(OLD/f'task_{seed}_vectors.npz') as f:
                kernel = f['z']
            with np.load(OLD/f'training_episodes_{seed}.npz') as f:
                train = list(zip(f['support_obs_ids'], f['query_obs_ids']))
            original_loss = {role: episode_loss(p, kernel, labels, episodes)
                             for role, episodes in dict(train=train, **validation).items()}
            for method in METHODS:
                tag = dict(origin=origin, seed=seed, method=method)
                with np.load(OUT/f'{origin}_{method}_{seed}_mapping.npz') as mapping, np.load(OUT/f'{origin}_{method}_{seed}_vectors.npz') as vector:
                    for part in ('candidate', 'remainder'):
                        exported = vector['z_'+part]
                        recreated = encode_raw(raw[held[:32]], mapping, part=part)
                        error = float(np.max(np.abs(exported[held[:32]]-recreated)))
                        checks.append(dict(**tag, part=part, max_raw_mapping_error=error))
                        np.testing.assert_allclose(recreated, exported[held[:32]], atol=1e-5, rtol=1e-5)
                    for part, score_key in [('candidate', 'candidate_scores'), ('remainder', 'remainder_scores')]:
                        scores = vector[score_key].astype(np.float64)
                        for role, ids in [('fit', fit), ('held', held)]:
                            mm = meta.iloc[ids].reset_index(drop=True)
                            values = vector['z_'+part][ids].astype(np.float64)
                            w = row_weights(mm); variance = float((w[:, None]*(values-w@values)**2).sum())
                            for cls, group in mm.groupby('class_id', sort=True):
                                selected = group.index.to_numpy()
                                pred, truth = scores[ids[selected]].argmax(1), labels[ids[selected]]
                                classes.append(dict(**tag, part=part, role=role, class_id=int(cls),
                                    chips=len(selected), ships=group.mmsi.nunique(),
                                    recall=float(100*(pred == truth).mean())))
                                drift, covered = paired_covariance(values[selected], group.reset_index(drop=True))
                                drifts.append(dict(**tag, part=part, role=role, class_id=int(cls),
                                    covered_ships=len(covered), pairs=int(covered.pairs.sum()) if len(covered) else 0,
                                    normalized_drift=float(np.trace(drift)/variance)))
                        centered = scores-scores.mean(1, keepdims=True)
                        degeneracy.append(dict(**tag, part=part, max_centered_logit=float(np.abs(centered).max()),
                                               held_BA=metrics(scores[held].argmax(1), labels[held])['BA']))
                    if method not in ('paired', 'class_only'):
                        continue
                    candidate = vector['z_candidate'].astype(np.float64)
                    delta = score_probe(candidate, mapping['correction'])
                    for gain in (0.5, 1.):
                        corrected = p+gain*delta
                        for role, episodes in dict(train=train, **validation).items():
                            loss = episode_loss(corrected, kernel, labels, episodes)
                            objectives.append(dict(**tag, role=role, gain=gain, baseline_loss=original_loss[role],
                                                   corrected_loss=loss, loss_change=loss-original_loss[role]))
                print('DIAGNOSED', origin, seed, method, flush=True)
    for name, rows in [('export_checks', checks), ('source_objectives', objectives), ('source_classes', classes),
                       ('class_drift', drifts), ('probe_degeneracy', degeneracy)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv', index=False)
    save_json(out/'COMPLETE.json', dict(raw_feature_mapping_checks=len(checks), new_neural_fits=0,
        source_objective_checks=len(objectives), target_selection_changed=False))
    print(pd.DataFrame(objectives).groupby(['origin', 'method', 'role', 'gain'])[['baseline_loss', 'corrected_loss', 'loss_change']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
