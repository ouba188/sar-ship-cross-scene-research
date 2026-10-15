"""No-retraining attribution: correction alignment and cross-fit baseline mismatch."""

import numpy as np
import pandas as pd
import torch
from condition_task_objective import (
    BASE, CACHE, OUT, STATE_OUT, PAIR_OUT, SEED, source_readout, design,
    standardize, endpoint_blocks, pair_blocks, fit_predict, ship_weights, identity_cv,
)


def main():
    torch.set_num_threads(1)
    dest = OUT/'diagnosis'
    dest.mkdir(exist_ok=True)
    obs = pd.read_csv(BASE/'observations.csv')
    saved = np.load(BASE/'relational_factors_20261010/source_split.npz')
    meta = obs.iloc[np.r_[saved['train_obs_ids'], saved['diagnostic_obs_ids']]].reset_index(drop=True)
    source = np.flatnonzero(meta.fold.eq(2))
    classifier_ids = np.flatnonzero(meta.fold.eq(1))
    train, test = [pd.read_csv(PAIR_OUT/f'{s}_pairs.csv') for s in ('train', 'test')]
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], dtype=np.float32)
    fit = source_readout(raw[classifier_ids], meta.iloc[classifier_ids].reset_index(drop=True))
    with torch.no_grad():
        _, scores = design(torch.from_numpy(raw), fit)
    scores = scores.numpy().astype(float)
    sorted_scores = np.sort(scores, axis=1)
    state_raw = np.c_[scores, sorted_scores[:, -1], sorted_scores[:, -1]-sorted_scores[:, -2]]
    part, mapping = identity_cv(meta.iloc[source], train)
    expected = pd.read_csv(OUT/'source_oof_targets.csv')
    predictions = pd.read_csv(OUT/'predictions.csv')
    fixed = pd.read_csv(STATE_OUT/'predictions.csv')
    fixed = fixed[fixed.arm == 'state_base']
    ensemble, replay = {}, []
    for k, task in enumerate(('signed', 'magnitude')):
        test_heads = []
        for fold in range(3):
            p = train.loc[part != fold].reset_index(drop=True)
            q = train.loc[part == fold].reset_index(drop=True)
            ids = np.array([i for i in source if mapping[meta.iloc[i].mmsi] != fold])
            blocks = endpoint_blocks(standardize(state_raw, ids))
            target = p.delta_margin.to_numpy()
            if k:
                target = np.abs(target)
            all_query = pd.concat([q, test], ignore_index=True)
            _, values, _ = fit_predict(pair_blocks(blocks, p, bool(k)), pair_blocks(blocks, all_query, bool(k)),
                target, ship_weights(p), bool(k))
            wanted = expected.loc[part == fold, task+'_oof_baseline'].to_numpy()
            np.testing.assert_allclose(values[:len(q)], wanted, rtol=0, atol=1e-12)
            replay.append(dict(task=task, fold=fold, max_error=float(np.max(np.abs(values[:len(q)]-wanted)))))
            test_heads.append(values[len(q):])
        ensemble[task] = np.mean(test_heads, axis=0)
    rows, decomposition, crossfit_difference = [], [], []
    for task, q in predictions.groupby('task'):
        targets = test.delta_margin.to_numpy()
        if task == 'magnitude':
            targets = np.abs(targets)
        w = ship_weights(test)
        full = fixed[(fixed.task == task)&(fixed.split == 'test')].sort_values('pair').prediction.to_numpy()
        for name, baseline in [('full_source', full), ('source_cv_ensemble', ensemble[task])]:
            rows.append(dict(task=task, baseline=name, arm='state_base', seed=-1,
                MSE=np.average((targets-baseline)**2, weights=w)))
            for (arm, seed), g in q[q.split == 'test'].groupby(['arm', 'seed']):
                g = g.sort_values('pair')
                correction = g.correction.to_numpy()
                combined = baseline+correction
                if task == 'magnitude':
                    combined = np.maximum(combined, 0)
                rows.append(dict(task=task, baseline=name, arm=arm, seed=seed,
                    MSE=np.average((targets-combined)**2, weights=w)))
        for (arm, seed), g in q.groupby(['arm', 'seed']):
            for split, pp in [('train', train), ('test', test)]:
                h = g[g.split == split].sort_values('pair')
                w = ship_weights(pp)
                r, c = h.target.to_numpy()-h.base.to_numpy(), h.correction.to_numpy()
                energy, alignment = np.average(c*c, weights=w), np.average(c*r, weights=w)
                observed = np.average((h.prediction-h.target)**2-(h.base-h.target)**2, weights=w)
                row = dict(task=task, arm=arm, seed=seed, split=split, correction_energy=energy,
                    target_inner_product=alignment, raw_delta_MSE=energy-2*alignment,
                    observed_delta_MSE=observed, oracle_scalar=alignment/energy,
                    normalized_alignment=alignment/np.sqrt(energy*np.average(r*r, weights=w)))
                if task == 'signed':
                    np.testing.assert_allclose(observed, energy-2*alignment, atol=1e-12)
                if split == 'test':
                    ship = pd.DataFrame(dict(mmsi=pp.mmsi, inner=c*r)).groupby('mmsi').inner.mean().to_numpy()
                    draws = np.random.default_rng(SEED).integers(len(ship), size=(2000, len(ship)))
                    row['inner_ci_low'], row['inner_ci_high'] = np.quantile(ship[draws].mean(1), [.025, .975])
                decomposition.append(row)
        train_full = fixed[(fixed.task == task)&(fixed.split == 'train')].sort_values('pair').prediction.to_numpy()
        train_oof = expected[task+'_oof_baseline'].to_numpy()
        wt = ship_weights(train)
        crossfit_difference.append(dict(task=task, OOF_minus_full_RMS=np.sqrt(np.average((train_oof-train_full)**2, weights=wt)),
            OOF_target_RMS=np.sqrt(np.average(expected[task+'_residual']**2, weights=wt))))
    pd.DataFrame(rows).to_csv(dest/'baseline_mismatch.csv', index=False)
    pd.DataFrame(decomposition).to_csv(dest/'correction_decomposition.csv', index=False)
    pd.DataFrame(replay).to_csv(dest/'oof_replay.csv', index=False)
    pd.DataFrame(crossfit_difference).to_csv(dest/'source_oof_difference.csv', index=False)
    print('BASELINE_CHECK', pd.DataFrame(rows).groupby(['task', 'baseline', 'arm']).MSE.mean().to_string(), flush=True)
    print('DIRECTION', pd.DataFrame(decomposition).query("arm == 'scene_task' and split == 'test'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
