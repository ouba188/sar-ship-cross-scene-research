"""Task-directed condition learning using identity-cross-fitted residual targets."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from condition_state_link import (
    BASE, CACHE, FactorNet, encode, source_readout, design, ship_weights, summarize,
    save_json, SEED, standardize, endpoint_blocks, pair_blocks, fit_predict,
    OUT as STATE_OUT, PREVIOUS as PAIR_OUT,
)
from relational_factors import relation_pools, sample_triplets, angular_gap

OUT = BASE/'condition_task_objective_20261010'
STEPS = 400
SEEDS = (SEED+21, SEED+22)
ARMS = ('ordinary_frozen', 'condition_frozen', 'scene_control', 'scene_task')


def identity_cv(meta, pairs):
    ships = meta[['mmsi', 'class_id']].drop_duplicates()
    rng = np.random.default_rng(SEED)
    mapping = {}
    for _, group in ships.groupby('class_id'):
        for i, ship in enumerate(rng.permutation(group.mmsi.to_numpy())):
            mapping[ship] = i % 3
    return pairs.mmsi.map(mapping).to_numpy(), mapping


def crossfit_baseline(state_raw, meta, source_ids, pairs):
    partition, mapping = identity_cv(meta.iloc[source_ids], pairs)
    outputs = np.empty((len(pairs), 2))
    records = []
    for fold in range(3):
        fit_pairs = pairs.loc[partition != fold].reset_index(drop=True)
        held_pairs = pairs.loc[partition == fold].reset_index(drop=True)
        fit_ships, held_ships = set(fit_pairs.mmsi), set(held_pairs.mmsi)
        assert not fit_ships&held_ships
        fit_ids = np.array([i for i in source_ids if mapping[meta.iloc[i].mmsi] != fold])
        blocks = endpoint_blocks(standardize(state_raw, fit_ids))
        for k, magnitude in enumerate((False, True)):
            y = fit_pairs.delta_margin.to_numpy()
            if magnitude:
                y = np.abs(y)
            _, prediction, _ = fit_predict(pair_blocks(blocks, fit_pairs, magnitude),
                pair_blocks(blocks, held_pairs, magnitude), y, ship_weights(fit_pairs), magnitude)
            outputs[partition == fold, k] = prediction
        records.append(dict(fold=fold, fit_ships=len(fit_ships), held_ships=len(held_ships),
            fit_pairs=len(fit_pairs), held_pairs=len(held_pairs), identity_overlap=0))
    return outputs, partition, records


class ConditionResidual(nn.Module):
    def __init__(self, condition, center, scale):
        super().__init__()
        self.condition = copy.deepcopy(condition)
        self.register_buffer('center', torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.signed_head = nn.Linear(352, 1, bias=False)
        self.magnitude_head = nn.Linear(352, 1)
        for head in (self.signed_head, self.magnitude_head):
            nn.init.zeros_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def normalize(self, z):
        return (z-self.center)/self.scale

    def pair_residual(self, za, zb, sa, sb, detach=False):
        if detach:
            za, zb = za.detach(), zb.detach()
        def features(z, s):
            z = self.normalize(z)
            cross = (s[:, :, None]*z[:, None, :]).flatten(1)
            return torch.cat([z/np.sqrt(32), cross/np.sqrt(320)], 1)
        delta = features(zb, sb)-features(za, sa)
        return torch.cat([self.signed_head(delta), self.magnitude_head(delta.abs())], 1)


def train_arm(arm, seed, initial, reference, x, state, pairs, residual, residual_scale, pools, source_ids):
    torch.manual_seed(seed)
    c = reference[source_ids].mean(0)
    s = np.maximum(reference[source_ids].std(0), .1)
    model = ConditionResidual(initial, c, s)
    frozen = arm.endswith('frozen')
    model.condition.requires_grad_(not frozen)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.0001)
    x, state, reference = [torch.as_tensor(v, dtype=torch.float32) for v in (x, state, reference)]
    target = torch.as_tensor(residual/residual_scale, dtype=torch.float32)
    probability = ship_weights(pairs)
    probability /= probability.sum()
    history = []
    for step in range(STEPS):
        rng = np.random.default_rng(seed+step)
        rows = rng.choice(len(pairs), 96, replace=True, p=probability)
        a, b = pairs.a.to_numpy()[rows], pairs.b.to_numpy()[rows]
        triples = source_ids[sample_triplets(pools, rng, 48)]
        ids = np.r_[a, b, triples.ravel()]
        z = model.condition(x[ids])
        predicted = model.pair_residual(z[:96], z[96:192], state[a], state[b], detach=arm == 'scene_control')
        losses = (predicted-target[rows]).square().mean(0)
        task_loss = losses.mean()
        scene_loss = F.relu(.2-angular_gap(z[192:], np.arange(144).reshape(48, 3))).mean()
        anchor = (model.normalize(z)-model.normalize(reference[ids])).square().mean()
        loss = task_loss if frozen else task_loss+.2*scene_loss+.1*anchor
        optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 0 or (step+1) % 100 == 0:
            row = dict(arm=arm, seed=seed, step=step+1, signed_loss=float(losses[0].detach()),
                magnitude_loss=float(losses[1].detach()), scene_loss=float(scene_loss.detach()),
                anchor=float(anchor.detach()), grad=float(grad))
            history.append(row)
            print('TRAIN', row, flush=True)
    return model.eval(), history


def predict_residual(model, x, state, pairs, scale):
    with torch.no_grad():
        z = torch.cat([model.condition(torch.from_numpy(x[i:i+256])) for i in range(0, len(x), 256)])
        state = torch.as_tensor(state, dtype=torch.float32)
        a, b = pairs.a.to_numpy(copy=True), pairs.b.to_numpy(copy=True)
        pred = model.pair_residual(z[a], z[b], state[a], state[b]).numpy()*scale
    return pred, z.numpy()


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    obs = pd.read_csv(BASE/'observations.csv')
    saved = np.load(BASE/'relational_factors_20261010/source_split.npz')
    meta = obs.iloc[np.r_[saved['train_obs_ids'], saved['diagnostic_obs_ids']]].reset_index(drop=True)
    folds = {f: np.flatnonzero(meta.fold.eq(f)) for f in (0, 1, 2)}
    sets = {f: set(meta.iloc[ix].mmsi) for f, ix in folds.items()}
    assert all(not sets[a]&sets[b] for a, b in itertools.combinations(folds, 2))
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], dtype=np.float32)
    classifier = source_readout(raw[folds[1]], meta.iloc[folds[1]].reset_index(drop=True))
    with torch.no_grad():
        _, scores = design(torch.from_numpy(raw), classifier)
    scores = scores.numpy().astype(float)
    sorted_scores = np.sort(scores, axis=1)
    state_raw = np.c_[scores, sorted_scores[:, -1], sorted_scores[:, -1]-sorted_scores[:, -2]]
    state = standardize(state_raw, folds[2])
    x = ((raw-saved['center'])/saved['scale']).astype(np.float32)
    pairs = {s: pd.read_csv(PAIR_OUT/f'{s}_pairs.csv') for s in ('train', 'test')}
    train, test = pairs['train'], pairs['test']
    for split, p in pairs.items():
        np.testing.assert_array_equal(meta.iloc[p.a].obs_id, p.obs_a)
        np.testing.assert_array_equal(meta.iloc[p.b].obs_id, p.obs_b)
        assert set(meta.iloc[np.r_[p.a, p.b]].fold) == {2 if split == 'train' else 0}
    oof, cv, cv_records = crossfit_baseline(state_raw, meta, folds[2], train)
    y_train = np.c_[train.delta_margin, np.abs(train.delta_margin)]
    residual = y_train-oof
    residual_scale = np.maximum(np.sqrt(np.average(residual**2, axis=0, weights=ship_weights(train))), .01)
    baseline = {}
    for split in ('train', 'test'):
        saved_pred = pd.read_csv(STATE_OUT/'predictions.csv')
        values = []
        for task in ('signed', 'magnitude'):
            p = saved_pred[(saved_pred.split == split)&(saved_pred.arm == 'state_base')&(saved_pred.task == task)].sort_values('pair')
            np.testing.assert_array_equal(p.mmsi, pairs[split].mmsi)
            values.append(p.prediction.to_numpy())
        baseline[split] = np.column_stack(values)
    # Replay the unchanged full-source state baseline before using it as a protected component.
    blocks = endpoint_blocks(state)
    replay = []
    for k, magnitude in enumerate((False, True)):
        pt, pv, _ = fit_predict(pair_blocks(blocks, train, magnitude), pair_blocks(blocks, test, magnitude),
            y_train[:, k], ship_weights(train), magnitude)
        np.testing.assert_allclose(pv, baseline['test'][:, k], rtol=0, atol=1e-12)
        np.testing.assert_allclose(pt, baseline['train'][:, k], rtol=0, atol=1e-12)
        replay.append(dict(task=k, maximum_difference=float(np.max(np.abs(pv-baseline['test'][:, k])))))
    initial, reference = {}, {}
    for key, name in [('ordinary', 'task_only'), ('condition', 'relations')]:
        net = FactorNet()
        net.load_state_dict(torch.load(BASE/'relational_factors_20261010'/f'{name}.pt', map_location='cpu', weights_only=True))
        initial[key] = net.condition
        reference[key] = encode(net.eval(), x)[1]
    _, pools = relation_pools(meta.iloc[folds[2]].reset_index(drop=True))
    save_json(OUT/'protocol.json', dict(classifier_fold=1, source_training_fold=2, held_identity_fold=0,
        source_cv_folds=3, training_steps=STEPS, seeds=SEEDS, learning_rate=.001,
        task_weight=1., scene_weight=.2, anchor_weight=.1, source_scene_anchors=len(pools['scene']),
        residual_scale=residual_scale.tolist(), notes='Only scene_task backpropagates task residual loss into the encoder. scene_control uses identical initial encoder, head, batches and optimizer but stops that gradient. Frozen controls learn heads only. Source OOF state normalizations exclude held ships; inherited encoder pretrained on source folds1+2. Test holdout reused for development, not new confirmation.'))
    pd.DataFrame(cv_records).to_csv(OUT/'source_cv.csv', index=False)
    pd.DataFrame(replay).to_csv(OUT/'baseline_replay.csv', index=False)
    source_targets = train[['mmsi', 'a', 'b']].copy()
    source_targets['cv_fold'] = cv
    for k, task in enumerate(('signed', 'magnitude')):
        source_targets[task+'_target'] = y_train[:, k]
        source_targets[task+'_oof_baseline'] = oof[:, k]
        source_targets[task+'_residual'] = residual[:, k]
    source_targets.to_csv(OUT/'source_oof_targets.csv', index=False)
    results, predictions, logs, diagnostics = [], [], [], []
    for seed in SEEDS:
        for arm in ARMS:
            key = 'ordinary' if arm == 'ordinary_frozen' else 'condition'
            model, history = train_arm(arm, seed, initial[key], reference[key], x, state,
                train, residual, residual_scale, pools['scene'], folds[2])
            logs.extend(history)
            torch.save(model.state_dict(), OUT/f'{arm}_{seed}.pt')
            for split, p in pairs.items():
                correction, z = predict_residual(model, x, state, p, residual_scale)
                # Training metrics use out-of-fold state predictions, exactly matching the optimized targets.
                base = oof if split == 'train' else baseline['test']
                combined = base+correction
                combined[:, 1] = np.maximum(combined[:, 1], 0)
                for k, task in enumerate(('signed', 'magnitude')):
                    target = p.delta_margin.to_numpy()
                    if k:
                        target = np.abs(target)
                    row = summarize(p, target, combined[:, k], base[:, k], p.flip.to_numpy(),
                        p.correct_a.to_numpy(), arm, task, split)
                    row['seed'] = seed
                    results.append(row)
                    predictions.append(pd.DataFrame(dict(seed=seed, arm=arm, split=split, task=task,
                        pair=np.arange(len(p)), mmsi=p.mmsi, target=target, base=base[:, k],
                        correction=correction[:, k], prediction=combined[:, k])))
                    w = ship_weights(p)
                    need = target-base[:, k]
                    diagnostics.append(dict(seed=seed, arm=arm, split=split, task=task,
                        residual_MSE=np.average((correction[:, k]-need)**2, weights=w),
                        base_MSE=np.average(need**2, weights=w),
                        correction_RMS=np.sqrt(np.average(correction[:, k]**2, weights=w)),
                        correction_target_product=np.average(correction[:, k]*need, weights=w),
                        feature_change_RMS=np.sqrt(np.mean((z[folds[2 if split == 'train' else 0]]-reference[key][folds[2 if split == 'train' else 0]])**2))))
            print('DONE', arm, seed, pd.DataFrame(results).tail(2).to_string(index=False), flush=True)
            pd.DataFrame(results).to_csv(OUT/'metrics.csv', index=False)
            pd.DataFrame(logs).to_csv(OUT/'training.csv', index=False)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(OUT/'predictions.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv', index=False)
    metrics = pd.DataFrame(results)
    summary = metrics.groupby(['split', 'arm', 'task']).agg(MSE=('MSE', 'mean'),
        MSE_min=('MSE', 'min'), MSE_max=('MSE', 'max'), skill_vs_state=('skill_vs_constant', 'mean')).reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    intervals = []
    predictions['error'] = (predictions.prediction-predictions.target)**2
    predictions['base_error'] = (predictions.base-predictions.target)**2
    for task in ('signed', 'magnitude'):
        sub = predictions[(predictions.split == 'test')&(predictions.task == task)]
        by_ship = sub.groupby(['mmsi', 'arm']).error.mean().unstack()
        by_ship['state_base'] = sub.groupby('mmsi').base_error.mean()
        draws = np.random.default_rng(SEED).integers(len(by_ship), size=(2000, len(by_ship)))
        for ref in ('state_base', 'scene_control', 'condition_frozen', 'ordinary_frozen'):
            diff = (by_ship.scene_task-by_ship[ref]).to_numpy()
            lo, hi = np.quantile(diff[draws].mean(1), [.025, .975])
            intervals.append(dict(task=task, reference=ref, delta_MSE=diff.mean(), ci_low=lo, ci_high=hi))
    pd.DataFrame(intervals).to_csv(OUT/'ship_bootstrap.csv', index=False)
    print('SUMMARY', summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
