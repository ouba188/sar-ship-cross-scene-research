"""Train-only log1p(SOG) auxiliary supervision, matched against class-mean SOG."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import copy
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from knowledge_train_only import (SEEDS, STEPS, KnowledgeStudent, SAROnlyPredictor,
    load_upstream, make_schedule, calibrate_alpha, train_arm)
from execute_knowledge_trial import initialize_attribute_head, adapt
from relative_content_condition import load_data, BASE, SUPPORTS
from scene_training import sample_weights, metrics
from probe_knowledge_attributes import write_json, fit_ridge, predict_ridge

ROOT = BASE/'motion_knowledge_20261011'
OUT = ROOT/'training'
PREVIOUS = BASE/'knowledge_train_only_20261011/training'
CACHE = BASE/'crossfit_evidence_20261011'
ARMS = ('C', 'D')


def prepare_speed(sog, valid, meta, probability):
    sog = np.asarray(sog, float)
    valid = np.asarray(valid, bool)
    if not valid.any() or not (np.isfinite(sog[valid]) & (sog[valid] >= 0) & (sog[valid] < 102.2)).all():
        raise ValueError('Exact numeric SOG is required, including valid zero; censored102.2 excluded')
    vm = meta.loc[valid].reset_index(drop=True)
    target = np.log1p(sog[valid])[:, None]
    weight = sample_weights(vm); weight /= weight.sum()
    mean = weight@target
    scale = np.sqrt(weight@(target-mean)**2).clip(1e-5)
    class_mean = np.full((8, 1), np.nan)
    for c in np.unique(vm.class_id):
        rows = vm.class_id.to_numpy() == c
        class_mean[c] = np.average(target[rows], axis=0, weights=weight[rows])
    real = np.zeros((len(meta), 1), np.float32); control = real.copy()
    real[valid] = (target-mean)/scale
    control[valid] = (class_mean[vm.class_id.to_numpy()]-mean)/scale
    importance = np.zeros(len(meta), np.float32)
    importance[valid] = weight/probability[valid]
    return dict(real=real, control=control, valid=valid, importance=importance,
                log_mean=mean, log_scale=scale, class_log_means=class_mean)


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    del norm, fixed, h
    fit, held = split['factor_fit'], split['test_source']
    fm, hm = [meta.iloc[ix].reset_index(drop=True) for ix in (fit, held)]
    with np.load(ROOT/'attributes.npz') as f:
        attributes = {k: f[k] for k in f.files}
    np.testing.assert_array_equal(attributes['fit_obs_ids'], fit)
    np.testing.assert_array_equal(attributes['held_obs_ids'], held)
    probability = sample_weights(fm); probability /= probability.sum()
    # AIS1022 is a lower-bounded speed, not an exact regression target.
    valid = attributes['fit_sog_valid'] & (attributes['fit_sog'] < 102.2)
    attrs = prepare_speed(attributes['fit_sog'], valid, fm, probability)
    at = {k: torch.from_numpy(attrs[k]) for k in ('real', 'control', 'valid', 'importance')}
    y = torch.tensor(meta.class_id.to_numpy(copy=True))
    with np.load(PREVIOUS/'evaluation_episodes.npz') as f:
        episodes = {role: [(f[f'{role}_{j}_support'], f[f'{role}_{j}_query']) for j in range(3)]
                    for role in ('validation', 'held')}
    OUT.mkdir(exist_ok=False)
    np.savez_compressed(OUT/'attribute_normalization.npz', **attrs)
    write_json(OUT/'PROTOCOL.json', dict(
        arms=list(ARMS), seeds=list(SEEDS), steps=STEPS, fitting=len(fit), valid_sog=int(valid.sum()),
        zero_speed=int((valid & (attributes['fit_sog'] == 0)).sum()),
        targets='C log1p AIS SOG; D source class-mean log1p SOG on SAME mask. No dimension targets.',
        objective='B classification+support+OOF other-product teaching plus speed auxiliary. Actual C gradient capped10pct on8fixed batches; D same coefficient.',
        initialization='32->1 auxiliary head fitted by fixed source ridge0.1; class projection/prior and schedules identical to B.',
        selection='Original3source new-ship SAR support episodes; BA then CE then earliest every20steps incl0.',
        inference='Metadata-free SAR input and64labeled SAR supports; auxiliary head removed.',
        evaluation='4fits frozen before target evaluation. B selected/last predictions reused unchanged and replayed.',
        limitations='AIS reports within3min, not exact physical synchronization; developmental ports and historical frozen features, no causal proof.',
        scope='No sweep, no main model promotion by isolated positive. SOG source-only labels, no target AIS.'))
    trace_rows, choices, gradients, deployments = [], [], [], []
    predictions, kernels = {}, {}
    for seed in SEEDS:
        upstream = load_upstream(CACHE, seed, raw, meta, fit, episodes['validation'])
        x, kernel = upstream['x'], upstream['kernel']; kernels[seed] = kernel
        torch.manual_seed(seed)
        initial = KnowledgeStudent(x[fit], y[fit], probability)
        old_initial = torch.load(PREVIOUS/f'initial_{seed}.pt', weights_only=True)
        for key, value in initial.state_dict().items():
            torch.testing.assert_close(value, old_initial[key], rtol=0, atol=0)
        initial.attribute_head = nn.Linear(32, 1)
        schedule = make_schedule(fm, probability, seed)
        with np.load(PREVIOUS/f'train_schedule_{seed}.npz') as old:
            for key in schedule:
                np.testing.assert_array_equal(schedule[key], old[key])
        calibrated = copy.deepcopy(initial)
        initialize_attribute_head(calibrated, x[fit], at, probability, 'real')
        alpha, rows = calibrate_alpha(calibrated, x[fit], y[fit], kernel[fit],
            upstream['teacher'], upstream['teacher_valid'], at, schedule)
        gradients.extend(dict(seed=seed, **row) for row in rows)
        for arm in ARMS:
            start_model = copy.deepcopy(initial)
            initialize_attribute_head(start_model, x[fit], at, probability, 'real' if arm == 'C' else 'control')
            torch.save(start_model.state_dict(), OUT/f'{arm}_{seed}_initial.pt')
            print('START', seed, arm, 'alpha', alpha, flush=True)
            model, states, trace, choice = train_arm(start_model, x, y, kernel, fit,
                upstream['teacher'], upstream['teacher_valid'], at, schedule,
                episodes['validation'], seed, arm, alpha)
            torch.save(states, OUT/f'{arm}_{seed}.pt')
            trace_rows.extend(trace); choices.append(choice)
            pd.DataFrame(trace_rows).to_csv(OUT/'training.csv', index=False)
            pd.DataFrame(choices).to_csv(OUT/'selection.csv', index=False)
            for snapshot in ('selected', 'last'):
                model.load_state_dict(states[snapshot])
                with torch.no_grad():
                    z, prior = model(x); attr = model.attribute_head(z)
                predictions[arm, seed, snapshot] = prior
                np.savez_compressed(OUT/f'{arm}_{seed}_{snapshot}.npz', z=z.numpy(), prior=prior.numpy(),
                                    attribute_prediction=attr.numpy())
            model.load_state_dict(states['selected'])
            deploy = SAROnlyPredictor.from_student(model, upstream['case_state'], upstream['raw_center'], upstream['raw_scale'])
            torch.save(deploy.state_dict(), OUT/f'deployment_{arm}_{seed}.pt')
            loaded = SAROnlyPredictor(torch.load(OUT/f'deployment_{arm}_{seed}.pt', weights_only=True))
            with torch.no_grad():
                p, k = loaded(torch.from_numpy(raw))
            np.testing.assert_array_equal(p.numpy(), predictions[arm, seed, 'selected'].numpy())
            np.testing.assert_array_equal(k.numpy(), kernel.numpy())
            assert not any('attribute' in key for key in loaded.state_dict())
            deployments.append(dict(seed=seed, arm=arm, n=len(raw), prior_error=0, kernel_error=0))
        for snapshot in ('selected', 'last'):
            with np.load(PREVIOUS/f'B_{seed}_{snapshot}.npz') as old:
                predictions['B', seed, snapshot] = torch.from_numpy(old['prior'])
        baseline = SAROnlyPredictor(torch.load(PREVIOUS/f'deployment_B_{seed}.pt', weights_only=True))
        with torch.no_grad():
            p, k = baseline(torch.from_numpy(raw))
        np.testing.assert_array_equal(p.numpy(), predictions['B', seed, 'selected'].numpy())
        np.testing.assert_array_equal(k.numpy(), kernel.numpy())
    pd.DataFrame(gradients).to_csv(OUT/'gradient_calibration.csv', index=False)
    pd.DataFrame(deployments).to_csv(OUT/'deployment_replay.csv', index=False)
    write_json(OUT/'ALL_MODELS_FROZEN.json', dict(fits=len(choices), query_metrics_read=False))
    source_rows, target_rows, class_rows = [], [], []
    for seed in SEEDS:
        kernel = kernels[seed]
        for arm in ('B',)+ARMS:
            for snapshot in ('selected', 'last'):
                prior = predictions[arm, seed, snapshot]
                source_rows.append(dict(seed=seed, arm=arm, snapshot=snapshot, readout='raw', draw=-1,
                    **metrics(prior[held].argmax(1).numpy(), y[held].numpy())))
                for j, (s, q) in enumerate(episodes['held']):
                    score = adapt(prior, kernel, s, q, y[s])
                    source_rows.append(dict(seed=seed, arm=arm, snapshot=snapshot, readout='adapted', draw=j,
                        **metrics(score.argmax(1).numpy(), y[q].numpy())))
        for path in sorted(SUPPORTS.glob('*_split.npz')):
            task = path.stem.removesuffix('_split')
            with np.load(path) as f:
                s, q, sy = [f[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
            assert not set(meta.iloc[np.r_[s, q]].mmsi) & set(meta.iloc[source].mmsi)
            assert not set(meta.iloc[s].mmsi) & set(meta.iloc[q].mmsi)
            dump = dict(support_obs_ids=s, query_obs_ids=q, support_labels=sy)
            for arm in ('B',)+ARMS:
                for snapshot in ('selected', 'last'):
                    prior = predictions[arm, seed, snapshot]
                    score = adapt(prior, kernel, s, q, torch.tensor(sy)).numpy()
                    dump[f'{arm}_{snapshot}'] = score
                    dump[f'{arm}_{snapshot}_raw'] = prior[q].numpy()
            with np.load(PREVIOUS/f'target_{task}_{seed}.npz') as old:
                np.testing.assert_array_equal(dump['B_selected'], old['B_selected'])
            np.savez_compressed(OUT/f'target_{task}_{seed}.npz', **dump)
    write_json(OUT/'ALL_TARGET_PREDICTIONS_SAVED.json', dict(query_metrics_read=False))
    for path in sorted(OUT.glob('target_*.npz')):
        with np.load(path) as f:
            q = f['query_obs_ids']; truth = y[q].numpy()
            seed = int(path.stem.rsplit('_', 1)[1]); task = path.stem[7:].rsplit('_', 1)[0]
            port = str(meta.iloc[q[0]].port)
            for arm in ('B',)+ARMS:
                for snapshot in ('selected', 'last'):
                    pred = f[f'{arm}_{snapshot}'].argmax(1)
                    bp = f[f'B_{snapshot}'].argmax(1)
                    raw_result = metrics(f[f'{arm}_{snapshot}_raw'].argmax(1), truth)
                    tag = dict(seed=seed, arm=arm, snapshot=snapshot, task=task, port=port)
                    target_rows.append(dict(**tag, **metrics(pred, truth), raw_OA=raw_result['OA'], raw_BA=raw_result['BA'],
                        rescue=int(((pred == truth)&(bp != truth)).sum()), harm=int(((pred != truth)&(bp == truth)).sum())))
                    for c in np.unique(truth):
                        mask = truth == c
                        class_rows.append(dict(**tag, class_id=int(c), n=int(mask.sum()),
                            recall=float(100*(pred[mask] == c).mean()),
                            delta_recall=float(100*((pred[mask] == c).mean()-(bp[mask] == c).mean()))))
    pd.DataFrame(source_rows).to_csv(OUT/'source_results.csv', index=False)
    pd.DataFrame(target_rows).to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'target_classes.csv', index=False)
    write_json(OUT/'COMPLETE.json', dict(completed=True, fits=len(choices), elapsed_seconds=time.perf_counter()-started))
    print('TARGET', pd.DataFrame(target_rows).groupby(['snapshot', 'arm'])[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
