"""Locate semantic/identity contributions and test actual classification information."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
import numpy as np
import pandas as pd
import torch
from condition_task_objective import (
    BASE, CACHE, OUT as TRAINED, STATE_OUT, PAIR_OUT, SEEDS,
    ConditionResidual, FactorNet, source_readout, design, standardize, ship_weights,
)
from condition_state_link import context_coordinates, quadratic_state
from channel_operator import metrics, save_json, SEED

OUT = BASE/'condition_failure_root_20261010'


def class_margin(scores, labels):
    other = scores.copy()
    other[np.arange(len(scores)), labels] = -np.inf
    return scores[np.arange(len(scores)), labels]-other.max(1)


def centroids(z, meta, source_ids):
    frame = pd.DataFrame(z)
    frame['mmsi'], frame['class_id'] = meta.mmsi.to_numpy(), meta.class_id.to_numpy()
    ship_mean = frame.groupby('mmsi').mean()
    source_ships = meta.iloc[source_ids].mmsi.unique()
    source = ship_mean.loc[source_ships]
    global_mean = source.drop(columns='class_id').mean().to_numpy(dtype=np.float32)
    classes = source.groupby('class_id').mean()
    class_mean = classes.loc[meta.class_id.to_numpy()].to_numpy(dtype=np.float32)
    identity_mean = ship_mean.drop(columns='class_id').loc[meta.mmsi.to_numpy()].to_numpy(dtype=np.float32)
    return global_mean, class_mean, identity_mean


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    obs = pd.read_csv(BASE/'observations.csv')
    source_split = np.load(BASE/'relational_factors_20261010/source_split.npz')
    meta = obs.iloc[np.r_[source_split['train_obs_ids'], source_split['diagnostic_obs_ids']]].reset_index(drop=True)
    folds = {f: np.flatnonzero(meta.fold.eq(f)) for f in (0, 1, 2)}
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], dtype=np.float32)
    classifier = source_readout(raw[folds[1]], meta.iloc[folds[1]].reset_index(drop=True))
    with torch.no_grad():
        _, tensor_scores = design(torch.from_numpy(raw), classifier)
    scores = tensor_scores.numpy().astype(float)
    sorted_scores = np.sort(scores, axis=1)
    state_raw = np.c_[scores, sorted_scores[:, -1], sorted_scores[:, -1]-sorted_scores[:, -2]]
    state = standardize(state_raw, folds[2])
    x = ((raw-source_split['center'])/source_split['scale']).astype(np.float32)
    true_margin = class_margin(scores, meta.class_id.to_numpy())
    pairs = {s: pd.read_csv(PAIR_OUT/f'{s}_pairs.csv') for s in ('train', 'test')}
    previous = pd.read_csv(TRAINED/'predictions.csv')
    previous = previous[previous.task == 'signed']
    scale = json.loads((TRAINED/'protocol.json').read_text())['residual_scale'][0]
    features = {'state': state_raw, 'state_matched': np.c_[state_raw, context_coordinates(quadratic_state(state), folds[2], 32)]}
    oracle, decomposition, contributions, replays = [], [], [], []
    for split, p in pairs.items():
        exact = true_margin[p.b]-true_margin[p.a]
        np.testing.assert_allclose(exact, p.delta_margin, atol=1e-12, rtol=0)
        oracle.append(dict(split=split, pairs=len(p), scores_plus_true_class_max_error=np.max(np.abs(exact-p.delta_margin))))
    # A label-free identity average is diagnostic, not available in single-chip deployment.
    for old, name in [('task_only', 'ordinary'), ('relations', 'condition')]:
        network = FactorNet()
        network.load_state_dict(torch.load(BASE/'relational_factors_20261010'/f'{old}.pt', map_location='cpu', weights_only=True))
        with torch.no_grad():
            z = network.condition(torch.from_numpy(x)).numpy()
        features[name+'_alone'] = z
        features['state_'+name] = np.c_[state_raw, z]
    for arm in ('scene_control', 'scene_task'):
        for seed in SEEDS:
            saved = torch.load(TRAINED/f'{arm}_{seed}.pt', map_location='cpu', weights_only=True)
            model = ConditionResidual(FactorNet().condition, saved['center'].numpy(), saved['scale'].numpy())
            model.load_state_dict(saved)
            model.eval()
            with torch.no_grad():
                z = torch.cat([model.condition(torch.from_numpy(x[i:i+256])) for i in range(0, len(x), 256)]).numpy()
            features[f'{arm}_{seed}_alone'] = z
            features[f'state_{arm}_{seed}'] = np.c_[state_raw, z]
            global_mean, class_mean, identity_mean = centroids(z, meta, folds[2])
            for split, p in pairs.items():
                a, b = p.a.to_numpy(copy=True), p.b.to_numpy(copy=True)
                sa, sb = [torch.tensor(state[ix], dtype=torch.float32) for ix in (a, b)]
                def response(za, zb):
                    with torch.no_grad():
                        return model.pair_residual(torch.tensor(za), torch.tensor(zb), sa, sb)[:, 0].numpy().astype(float)*scale
                global_response = response(np.tile(global_mean, (len(a), 1)), np.tile(global_mean, (len(a), 1)))
                class_response = response(class_mean[a], class_mean[b])
                identity_response = response(identity_mean[a], identity_mean[b])
                full = response(z[a], z[b])
                saved_pred = previous[(previous.arm == arm)&(previous.seed == seed)&(previous.split == split)].sort_values('pair')
                np.testing.assert_array_equal(saved_pred.mmsi, p.mmsi)
                np.testing.assert_allclose(full, saved_pred.correction, atol=1e-6, rtol=1e-5)
                base = saved_pred.base.to_numpy()
                need = p.delta_margin.to_numpy()-base
                pieces = dict(global_state=global_response, oracle_class=class_response-global_response,
                    identity_remainder=identity_response-class_response, within_identity=full-identity_response)
                np.testing.assert_allclose(sum(pieces.values()), full, atol=1e-12)
                w = ship_weights(p)
                for name, c in dict(**pieces, all_static=identity_response, full=full).items():
                    mse = np.average((need-c)**2, weights=w)
                    inner = np.average(c*need, weights=w)
                    energy = np.average(c*c, weights=w)
                    attributed = np.average(c*full-2*c*need, weights=w)
                    decomposition.append(dict(arm=arm, seed=seed, split=split, component=name,
                        MSE=mse, base_MSE=np.average(need*need, weights=w), energy=energy, target_inner_product=inner,
                        delta_MSE_attribution=attributed,
                        alignment=inner/np.sqrt(max(energy*np.average(need*need, weights=w), 1e-15))))
                    contributions.append(pd.DataFrame(dict(arm=arm, seed=seed, split=split, component=name,
                        mmsi=p.mmsi, pair=np.arange(len(p)), value=c, needed=need)))
                replays.append(dict(arm=arm, seed=seed, split=split,
                    correction_replay_max_error=np.max(np.abs(full-saved_pred.correction))))
    # Equal-label-budget multiclass probes answer whether information is useful outside the scalar proxy.
    probe_rows, probe_predictions = [], []
    subsets = dict(train=folds[2], test_all=folds[0], test_paired=np.unique(np.r_[pairs['test'].a, pairs['test'].b]))
    for name, values in features.items():
        fit = source_readout(values[folds[2]], meta.iloc[folds[2]].reset_index(drop=True))
        with torch.no_grad():
            _, output = design(torch.tensor(values, dtype=torch.float32), fit)
        pred = output.argmax(1).numpy()
        for split, ids in subsets.items():
            y = meta.iloc[ids].class_id.to_numpy()
            correct = pred[ids] == y
            row = dict(arm=name, split=split, chips=len(ids), ships=meta.iloc[ids].mmsi.nunique(), dimensions=values.shape[1],
                **metrics(pred[ids], y), ship_OA=100*pd.DataFrame(dict(mmsi=meta.iloc[ids].mmsi.to_numpy(), correct=correct)).groupby('mmsi').correct.mean().mean())
            probe_rows.append(row)
            probe_predictions.append(pd.DataFrame(dict(arm=name, split=split, obs_id=meta.iloc[ids].obs_id.to_numpy(),
                mmsi=meta.iloc[ids].mmsi.to_numpy(), true=y, pred=pred[ids])))
    pd.DataFrame(probe_rows).to_csv(OUT/'classification_probes.csv', index=False)
    pd.concat(probe_predictions, ignore_index=True).to_csv(OUT/'classification_predictions.csv', index=False)
    pd.DataFrame(decomposition).to_csv(OUT/'correction_components.csv', index=False)
    pd.concat(contributions, ignore_index=True).to_csv(OUT/'component_values.csv', index=False)
    pd.DataFrame(replays).to_csv(OUT/'model_replay.csv', index=False)
    pd.DataFrame(oracle).to_csv(OUT/'oracle_target_identity.csv', index=False)
    save_json(OUT/'protocol.json', dict(new_neural_training=False,
        classification_probe_labels='same 910 source fold2 chips / 313 ships for all heads',
        oracle_class_centroid='source fold2 class centroids, selected with evaluation true class; diagnostic only',
        identity_centroid='each identity mean across available observations, including evaluation observations; diagnostic only',
        contribution='global, class-minus-global, identity-minus-class, observation-minus-identity; exact telescoping decomposition, not physical factor identification',
        limitations='Repeated development holdout and overlapping scenes. Probe OA/BA are source identity holdout, not full-model cross-port performance.'))
    print('PROBES', pd.DataFrame(probe_rows).query("split == 'test_all'").to_string(index=False), flush=True)
    print('COMPONENTS', pd.DataFrame(decomposition).groupby(['arm', 'split', 'component'])[['MSE', 'energy', 'target_inner_product', 'delta_MSE_attribution']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
