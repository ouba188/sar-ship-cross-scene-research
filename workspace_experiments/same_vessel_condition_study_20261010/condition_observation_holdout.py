"""Separate task-fitting observations from new observations of the same ship."""

import numpy as np
import pandas as pd
import torch
import json
from condition_task_objective import (
    BASE, CACHE, OUT as OLD, STATE_OUT, PAIR_OUT, SEEDS, SEED,
    FactorNet, source_readout, design, standardize, encode, train_arm, predict_residual,
    relation_pools, ship_weights, summarize, save_json,
)

OUT = BASE/'condition_failure_root_20261010/observation_holdout'


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    obs = pd.read_csv(BASE/'observations.csv')
    saved = np.load(BASE/'relational_factors_20261010/source_split.npz')
    meta = obs.iloc[np.r_[saved['train_obs_ids'], saved['diagnostic_obs_ids']]].reset_index(drop=True)
    folds = {f: np.flatnonzero(meta.fold.eq(f)) for f in (0, 1, 2)}
    pairs = pd.read_csv(PAIR_OUT/'train_pairs.csv')
    test = pd.read_csv(PAIR_OUT/'test_pairs.csv')
    source_targets = pd.read_csv(OLD/'source_oof_targets.csv')
    paired_ids = np.unique(np.r_[pairs.a, pairs.b])
    rng = np.random.default_rng(SEED+73)
    held = []
    for _, g in meta.iloc[paired_ids].groupby('mmsi'):
        if len(g) >= 4:
            n = min(len(g)-2, max(2, len(g)//3))
            held.extend(rng.choice(g.index.to_numpy(), n, replace=False).tolist())
    held = np.array(sorted(held), dtype=int)
    is_a, is_b = pairs.a.isin(held), pairs.b.isin(held)
    fit_ix, held_ix = np.flatnonzero(~is_a&~is_b), np.flatnonzero(is_a&is_b)
    held_ships = set(pairs.iloc[held_ix].mmsi)
    matched_ix = np.array([i for i in fit_ix if pairs.iloc[i].mmsi in held_ships])
    train_ids = np.setdiff1d(folds[2], held)
    assert not set(train_ids)&set(held)
    assert not set(np.r_[pairs.iloc[fit_ix].a, pairs.iloc[fit_ix].b])&set(held)
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], dtype=np.float32)
    classifier = source_readout(raw[folds[1]], meta.iloc[folds[1]].reset_index(drop=True))
    with torch.no_grad():
        _, scores = design(torch.from_numpy(raw), classifier)
    scores = scores.numpy().astype(float)
    ordered = np.sort(scores, axis=1)
    # State normalization is kept identical to the original diagnostic; it uses no labels.
    state = standardize(np.c_[scores, ordered[:, -1], ordered[:, -1]-ordered[:, -2]], folds[2])
    x = ((raw-saved['center'])/saved['scale']).astype(np.float32)
    initial = FactorNet()
    initial.load_state_dict(torch.load(BASE/'relational_factors_20261010/relations.pt', map_location='cpu', weights_only=True))
    reference = encode(initial.eval(), x)[1]
    residual = source_targets[['signed_residual', 'magnitude_residual']].to_numpy()[fit_ix]
    scale = np.array(json.loads((OLD/'protocol.json').read_text())['residual_scale'])
    _, pools = relation_pools(meta.iloc[train_ids].reset_index(drop=True))
    original = pd.read_csv(STATE_OUT/'predictions.csv')
    test_base = np.column_stack([original[(original.split == 'test')&(original.arm == 'state_base')&(original.task == t)].sort_values('pair').prediction.to_numpy() for t in ('signed', 'magnitude')])
    oof_base = source_targets[['signed_oof_baseline', 'magnitude_oof_baseline']].to_numpy()
    groups = dict(fit_all=(pairs.iloc[fit_ix].reset_index(drop=True), oof_base[fit_ix]),
        fit_matched_ships=(pairs.iloc[matched_ix].reset_index(drop=True), oof_base[matched_ix]),
        new_observations_same_ships=(pairs.iloc[held_ix].reset_index(drop=True), oof_base[held_ix]),
        new_ships=(test, test_base))
    pd.DataFrame([dict(group=k, pairs=len(p), ships=p.mmsi.nunique(), observations=len(np.unique(np.r_[p.a, p.b]))) for k, (p, _) in groups.items()]).to_csv(OUT/'coverage.csv', index=False)
    meta.iloc[held][['obs_id', 'mmsi', 'product', 'acq_time']].to_csv(OUT/'held_observations.csv', index=False)
    rows, predictions, logs = [], [], []
    for seed in SEEDS:
        for arm in ('scene_control', 'scene_task'):
            model, history = train_arm(arm, seed, initial.condition, reference, x, state,
                pairs.iloc[fit_ix].reset_index(drop=True), residual, scale, pools['scene'], train_ids)
            logs.extend(history)
            torch.save(model.state_dict(), OUT/f'{arm}_{seed}.pt')
            for group, (p, base) in groups.items():
                correction, _ = predict_residual(model, x, state, p, scale)
                combined = base+correction
                combined[:, 1] = np.maximum(combined[:, 1], 0)
                for j, task in enumerate(('signed', 'magnitude')):
                    target = p.delta_margin.to_numpy()
                    if j:
                        target = np.abs(target)
                    row = summarize(p, target, combined[:, j], base[:, j], p.flip.to_numpy(),
                        p.correct_a.to_numpy(), arm, task, group)
                    row.update(seed=seed, base_MSE=np.average((target-base[:, j])**2, weights=ship_weights(p)))
                    rows.append(row)
                    predictions.append(pd.DataFrame(dict(seed=seed, arm=arm, group=group, task=task,
                        mmsi=p.mmsi, target=target, base=base[:, j], prediction=combined[:, j])))
            print('EVALUATED', arm, seed, flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUT/'metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT/'predictions.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT/'training.csv', index=False)
    summary = results.groupby(['split', 'arm', 'task'])[['MSE', 'base_MSE', 'skill_vs_constant']].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    save_json(OUT/'protocol.json', dict(task_finetuning_observation_disjoint=True,
        held_observations=len(held), fit_observations=len(train_ids), seeds=SEEDS,
        notes='Initial factor encoder historically saw source fold2 observations. Only this round task/scene fine-tuning excludes held observations. OOF baselines exclude each entire identity. State normalization fixed using source unlabeled states. Not an independent unseen-image benchmark.'))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
