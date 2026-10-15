"""Matched-capacity state/context interactions for held-ship error changes."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import itertools
import numpy as np
import pandas as pd
import torch
from condition_error_link import (
    BASE, CACHE, FactorNet, encode, source_readout, design, ship_weights,
    summarize, save_json, SEED, OUT as PREVIOUS,
)

OUT = BASE/'condition_state_link_20261010'


def standardize(values, train_ids):
    mean = values[train_ids].mean(0)
    scale = np.maximum(values[train_ids].std(0), 1e-3)
    return (values-mean)/scale


def quadratic_state(state):
    i, j = np.triu_indices(state.shape[1])
    return np.c_[state, state[:, i]*state[:, j]]


def context_coordinates(values, train_ids, dimensions):
    # Source-only unsupervised capacity control, with no pair outcomes or labels.
    normalized = standardize(values, train_ids)
    mean = normalized[train_ids].mean(0)
    _, _, vt = np.linalg.svd(normalized[train_ids]-mean, full_matrices=False)
    reduced = (normalized-mean)@vt[:dimensions].T
    return standardize(reduced, train_ids)


def endpoint_blocks(state, context=None):
    blocks = [quadratic_state(state)]
    if context is not None:
        cross = (state[:, :, None]*context[:, None, :]).reshape(len(state), -1)
        blocks.append(np.c_[context, context**2, cross])
    return blocks


def pair_blocks(blocks, pairs, magnitude=False):
    diffs = [v[pairs.b.to_numpy()]-v[pairs.a.to_numpy()] for v in blocks]
    return [np.abs(v) for v in diffs] if magnitude else diffs


def fit_predict(train_blocks, query_blocks, y, w, magnitude=False):
    train, query = [], []
    for xt, xv in zip(train_blocks, query_blocks):
        scale = np.maximum(np.sqrt(np.average(xt*xt, axis=0, weights=w)), 1e-3)
        # Each block has the same normalization irrespective of context dimensionality.
        train.append(xt/scale/np.sqrt(xt.shape[1]))
        query.append(xv/scale/np.sqrt(xv.shape[1]))
    xt, xv = np.concatenate(train, 1), np.concatenate(query, 1)
    if magnitude:
        xt, xv = np.c_[xt, np.ones(len(xt))], np.c_[xv, np.ones(len(xv))]
    penalty = np.eye(xt.shape[1])*10.
    if magnitude:
        penalty[-1, -1] = 0.
    beta = np.linalg.solve(xt.T@(w[:, None]*xt)+penalty, xt.T@(w*y))
    pt, pv = xt@beta, xv@beta
    if magnitude:
        pt, pv = np.maximum(pt, 0), np.maximum(pv, 0)
    return pt, pv, beta


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    obs = pd.read_csv(BASE/'observations.csv')
    saved = np.load(BASE/'relational_factors_20261010/source_split.npz')
    meta = obs.iloc[np.r_[saved['train_obs_ids'], saved['diagnostic_obs_ids']]].reset_index(drop=True)
    folds = {f: np.flatnonzero(meta.fold.eq(f)) for f in (0, 1, 2)}
    ship_sets = {f: set(meta.iloc[ix].mmsi) for f, ix in folds.items()}
    assert all(not ship_sets[a]&ship_sets[b] for a, b in itertools.combinations(folds, 2))
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], dtype=np.float32)
    classifier = source_readout(raw[folds[1]], meta.iloc[folds[1]].reset_index(drop=True))
    with torch.no_grad():
        _, scores = design(torch.from_numpy(raw), classifier)
    scores = scores.numpy().astype(float)
    truth = meta.class_id.to_numpy()
    competitors = scores.copy()
    competitors[np.arange(len(meta)), truth] = -np.inf
    margin = scores[np.arange(len(meta)), truth]-competitors.max(1)
    sorted_scores = np.sort(scores, axis=1)
    state = standardize(np.c_[scores, sorted_scores[:, -1], sorted_scores[:, -1]-sorted_scores[:, -2]], folds[2])
    norm = ((raw-saved['center'])/saved['scale']).astype(np.float32)
    contexts = {}
    for arm, name in [('task_only', 'ordinary'), ('relations', 'condition')]:
        model = FactorNet()
        model.load_state_dict(torch.load(BASE/'relational_factors_20261010'/f'{arm}.pt', map_location='cpu', weights_only=True))
        contexts[name] = encode(model.eval(), norm)[1].astype(float)
    pairs = {s: pd.read_csv(PREVIOUS/f'{s}_pairs.csv') for s in ('train', 'test')}
    replay = []
    for split, p in pairs.items():
        np.testing.assert_array_equal(meta.iloc[p.a].obs_id, p.obs_a)
        np.testing.assert_array_equal(meta.iloc[p.b].obs_id, p.obs_b)
        delta = margin[p.b]-margin[p.a]
        np.testing.assert_allclose(delta, p.delta_margin, rtol=0, atol=1e-12)
        assert set(meta.iloc[np.r_[p.a, p.b]].fold) == {2 if split == 'train' else 0}
        replay.append(dict(split=split, pairs=len(p), max_delta_error=float(np.max(np.abs(delta-p.delta_margin)))))
    # All comparisons are specified before examining evaluation outcomes.
    variants = {'state_base': endpoint_blocks(state)}
    for dim in (32, 8):
        state_context = context_coordinates(quadratic_state(state), folds[2], dim)
        variants[f'state_matched{dim}'] = endpoint_blocks(state, state_context)
        for name, values in contexts.items():
            c = standardize(values, folds[2]) if dim == 32 else context_coordinates(values, folds[2], dim)
            variants[f'state_{name}{dim}'] = endpoint_blocks(state, c)
    train, test = pairs['train'], pairs['test']
    w = ship_weights(train)
    results, frames, coefficients = [], [], {}
    for task in ('signed', 'magnitude'):
        yt, yv = train.delta_margin.to_numpy(), test.delta_margin.to_numpy()
        magnitude = task == 'magnitude'
        if magnitude:
            yt, yv = np.abs(yt), np.abs(yv)
        constant = np.average(yt, weights=w) if magnitude else 0.
        for arm, blocks in variants.items():
            xt, xv = [pair_blocks(blocks, p, magnitude) for p in (train, test)]
            pt, pv, beta = fit_predict(xt, xv, yt, w, magnitude)
            coefficients[task+'_'+arm] = beta
            for split, p, y, prediction in [('train', train, yt, pt), ('test', test, yv, pv)]:
                row = summarize(p, y, prediction, np.full(len(y), constant), p.flip.to_numpy(),
                    p.correct_a.to_numpy(), arm, task, split)
                row['parameters'] = len(beta)
                results.append(row)
                frames.append(pd.DataFrame(dict(split=split, task=task, arm=arm, pair=np.arange(len(p)),
                    mmsi=p.mmsi.to_numpy(), target=y, prediction=prediction, same_port=p.same_port.to_numpy())))
        print(task, pd.DataFrame(results).query("task == @task and split == 'test'").to_string(index=False), flush=True)
    metrics = pd.DataFrame(results)
    predictions = pd.concat(frames, ignore_index=True)
    metrics.to_csv(OUT/'metrics.csv', index=False)
    predictions.to_csv(OUT/'predictions.csv', index=False)
    np.savez(OUT/'coefficients.npz', **coefficients)
    pd.DataFrame(replay).to_csv(OUT/'replay.csv', index=False)
    bootstrap, groups = [], []
    predictions['squared_error'] = (predictions.target-predictions.prediction)**2
    for task in ('signed', 'magnitude'):
        sub = predictions.query("task == @task and split == 'test'")
        by_ship = sub.groupby(['mmsi', 'arm']).squared_error.mean().unstack()
        draws = np.random.default_rng(SEED).integers(len(by_ship), size=(2000, len(by_ship)))
        for dim in (32, 8):
            arm = f'state_condition{dim}'
            for ref in ('state_base', f'state_matched{dim}', f'state_ordinary{dim}'):
                diff = (by_ship[arm]-by_ship[ref]).to_numpy()
                low, high = np.quantile(diff[draws].mean(1), [.025, .975])
                bootstrap.append(dict(task=task, arm=arm, reference=ref, delta_MSE=diff.mean(),
                    ci_low=low, ci_high=high, relative_error_reduction=1-by_ship[arm].mean()/by_ship[ref].mean()))
        for same, g in sub.groupby('same_port'):
            for arm, h in g.groupby('arm'):
                ship_error = h.groupby('mmsi').squared_error.mean()
                groups.append(dict(task=task, same_port=same, arm=arm, MSE=ship_error.mean(),
                    ships=len(ship_error), pairs=len(h)))
    pd.DataFrame(bootstrap).to_csv(OUT/'ship_bootstrap.csv', index=False)
    pd.DataFrame(groups).to_csv(OUT/'port_groups.csv', index=False)
    save_json(OUT/'protocol.json', dict(classifier_fold=1, predictor_fold=2, evaluation_fold=0,
        primary_context_dim=32, prespecified_low_capacity_dim=8, ridge=10,
        state='8 class scores, maximum score, top-two score gap',
        state_matched='32 or 8 source PCA coordinates of quadratic state; no new information',
        pair_features='endpoint differences of state quadratic, context, context squared, state-context products; absolute differences for magnitude',
        parameters_signed={k: sum(x.shape[1] for x in v) for k, v in variants.items()},
        normalization='source fold2 only; RMS pair scaling; each feature block divided by square root of its dimension',
        notes='Frozen encoders learned folds1+2; held ship IDs but overlapping scenes; previous diagnostic holdout reused; no target-port training or model changes.'))


if __name__ == '__main__':
    main()
