"""Execute the predeclared eight source-only fits, then frozen evaluations."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import time
import copy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from knowledge_train_only import (
    ARMS, SEEDS, STEPS, KnowledgeStudent, SAROnlyPredictor, prepare_attributes,
    load_upstream, make_schedule, calibrate_alpha, train_arm, phi, balanced_ce,
)
from relative_content_condition import load_data, BASE, SUPPORTS
from scene_training import sample_weights, metrics
from explicit_candidate_projection import identity_episode
from target_response_adapter import residual_head
from probe_knowledge_attributes import write_json, fit_ridge

ROOT = BASE/'knowledge_train_only_20261011'
OUT = ROOT/'training'
CACHE = BASE/'crossfit_evidence_20261011'


def adapt(prior, kernel, support, query, labels):
    beta = residual_head(phi(kernel[support]), prior[support], labels, balanced=True)[0]
    return prior[query]+phi(kernel[query])@beta


def initialize_attribute_head(model, xfit, attributes, probability, target):
    mask = attributes['valid'].numpy()
    with torch.no_grad():
        z, before = model(xfit)
        reader = fit_ridge(z.numpy()[mask].astype(float), attributes[target].numpy()[mask].astype(float),
                          probability[mask]*attributes['importance'].numpy()[mask])
        coefficient = reader['coef']/reader['scale'][:, None]
        bias = reader['intercept']-reader['center']@coefficient
        model.attribute_head.weight.copy_(torch.from_numpy(coefficient.T))
        model.attribute_head.bias.copy_(torch.from_numpy(bias))
        _, after = model(xfit)
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def main(output=None, warmstart=False):
    global OUT
    if output is not None:
        OUT = output
    arms = ('C', 'D') if warmstart else ARMS
    torch.set_num_threads(1)
    started = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    del norm, fixed, h
    fitting, held = split['factor_fit'], split['test_source']
    fm, hm = [meta.iloc[ids].reset_index(drop=True) for ids in (fitting, held)]
    y = torch.tensor(meta.class_id.to_numpy(copy=True))
    probability = sample_weights(fm); probability /= probability.sum()
    with np.load(ROOT/'attributes.npz') as f:
        np.testing.assert_array_equal(f['fit_obs_ids'], fitting)
        attrs = prepare_attributes(f['fit_values'], f['fit_valid'], fm, probability)
    np.savez_compressed(OUT/'attribute_normalization.npz', **attrs)
    at = {k: torch.from_numpy(attrs[k]) for k in ('real', 'control', 'valid', 'importance')}
    with np.load(CACHE/'validation_episodes.npz') as f:
        validation = [(f[f'val_new_ship_{j}_support'], f[f'val_new_ship_{j}_query']) for j in range(3)]
    held_episodes = []
    for j in range(3):
        s, q = identity_episode(hm, np.random.default_rng(91011+j), query_size=None)
        held_episodes.append((held[s], held[q]))
    episodes_dump = {}
    for role, episodes in [('validation', validation), ('held', held_episodes)]:
        for j, (s, q) in enumerate(episodes):
            assert not set(meta.iloc[s].mmsi) & set(meta.iloc[q].mmsi)
            episodes_dump[f'{role}_{j}_support'] = s
            episodes_dump[f'{role}_{j}_query'] = q
    np.savez_compressed(OUT/'evaluation_episodes.npz', **episodes_dump)
    write_json(OUT/'PROTOCOL.json', dict(
        arms=list(arms), seeds=list(SEEDS), steps=STEPS, fit_observations=len(fitting),
        fitting_ships=int(fm.mmsi.nunique()), attributes_valid=int(attrs['valid'].sum()),
        selection='Source new-vessel three fixed episodes; max mean BA, then min balanced CE, then earliest; every20 incl0.',
        teacher='Native source-vessel-fold OOF prior, average other products of same source vessel; no held inputs.',
        inference='SAR features only, training-only auxiliary head removed; fixed64labeled SAR supports retained.',
        control='C real AIS reported logdims; D source class-mean logdims, identical masks and coefficient; no new exclusions by AIS type.',
        caution='Existing fine labels can disagree with AIS broad descriptions; no relabeling and no causal-identification claim.',
        alpha='Maximum0.1, reduce to at most10pct initial frame gradient on each of8fixed batches; C/D same.',
        labels='Historical reused development data; frozen backbone historical exposure unchanged.',
        evaluation='Freeze all planned fits before held/target classification; last snapshots diagnostic only, never reselect.',
        attribute_head_initialization='Source valid-row fixed ridge0.1 for each own target, no true attributes in D initialization' if warmstart else 'Shared random initialization',
        correction='One bounded post-failure corrective trial; A/B unchanged and reused from original, not independent confirmation.' if warmstart else 'Original predeclared eight fits'))
    all_traces, selections, gradients, upstream_audit, deployment_audit = [], [], [], [], []
    predictions, kernels = {}, {}
    for seed in SEEDS:
        upstream = load_upstream(CACHE, seed, raw, meta, fitting, validation)
        x, kernel = upstream['x'], upstream['kernel']
        kernels[seed] = kernel
        upstream_audit.extend(upstream['audit'])
        schedule = make_schedule(fm, probability, seed)
        np.savez_compressed(OUT/f'train_schedule_{seed}.npz', **schedule, fit_obs_ids=fitting)
        np.savez_compressed(OUT/f'teacher_{seed}.npz', fit_obs_ids=fitting, oof=upstream['oof_prior'],
                            teacher=upstream['teacher'].numpy(), valid=upstream['teacher_valid'].numpy())
        torch.manual_seed(seed)
        initial = KnowledgeStudent(x[fitting], y[fitting], probability)
        torch.save(initial.state_dict(), OUT/f'initial_{seed}.pt')
        with torch.no_grad():
            zi, pi = initial(x)
        np.savez_compressed(OUT/f'initial_{seed}_vectors.npz', obs_ids=np.arange(len(meta)), z=zi.numpy(), prior=pi.numpy())
        calibration_initial = copy.deepcopy(initial)
        if warmstart:
            initialize_attribute_head(calibration_initial, x[fitting], at, probability, 'real')
        alpha, grows = calibrate_alpha(calibration_initial, x[fitting], y[fitting], kernel[fitting],
            upstream['teacher'], upstream['teacher_valid'], at, schedule)
        gradients.extend(dict(seed=seed, **row) for row in grows)
        for arm in arms:
            arm_initial = copy.deepcopy(initial)
            if warmstart:
                initialize_attribute_head(arm_initial, x[fitting], at, probability, 'real' if arm == 'C' else 'control')
            torch.save(arm_initial.state_dict(), OUT/f'{arm}_{seed}_initial.pt')
            print('START', seed, arm, 'alpha', alpha, flush=True)
            model, states, trace, choice = train_arm(arm_initial, x, y, kernel, fitting, upstream['teacher'],
                upstream['teacher_valid'], at, schedule, validation, seed, arm, alpha)
            torch.save(states, OUT/f'{arm}_{seed}.pt')
            all_traces.extend(trace); selections.append(choice)
            pd.DataFrame(all_traces).to_csv(OUT/'training.csv', index=False)
            pd.DataFrame(selections).to_csv(OUT/'selection.csv', index=False)
            for snapshot in ('selected', 'last'):
                model.load_state_dict(states[snapshot])
                with torch.no_grad():
                    z, prior = model(x)
                    attribute_prediction = model.attribute_head(z)
                predictions[arm, seed, snapshot] = prior
                np.savez_compressed(OUT/f'{arm}_{seed}_{snapshot}.npz', obs_ids=np.arange(len(meta)),
                    z=z.numpy(), prior=prior.numpy(), attribute_prediction=attribute_prediction.numpy())
            model.load_state_dict(states['selected'])
            deploy = SAROnlyPredictor.from_student(model, upstream['case_state'],
                upstream['raw_center'], upstream['raw_scale'])
            torch.save(deploy.state_dict(), OUT/f'deployment_{arm}_{seed}.pt')
            assert not any('attribute' in k for k in deploy.state_dict())
            # Reload the actual metadata-free export and replay every raw-feature row.
            replay_model = SAROnlyPredictor(torch.load(OUT/f'deployment_{arm}_{seed}.pt', weights_only=True))
            with torch.no_grad():
                rp, rk = replay_model(torch.from_numpy(raw))
            np.testing.assert_array_equal(rp.numpy(), predictions[arm, seed, 'selected'].numpy())
            np.testing.assert_array_equal(rk.numpy(), kernel.numpy())
            deployment_audit.append(dict(arm=arm, seed=seed, rows=len(raw), prior_max_error=0., kernel_max_error=0.,
                attribute_head_removed=True, metadata_free_api=True))
            print('SELECTED', choice, flush=True)
    pd.DataFrame(gradients).to_csv(OUT/'gradient_calibration.csv', index=False)
    pd.DataFrame(upstream_audit).to_csv(OUT/'upstream_audit.csv', index=False)
    pd.DataFrame(deployment_audit).to_csv(OUT/'deployment_replay.csv', index=False)
    write_json(OUT/'ALL_MODELS_FROZEN.json', dict(models=len(selections), query_metrics_read=False))

    source_rows, target_rows, class_rows = [], [], []
    for seed in SEEDS:
        kernel = kernels[seed]
        source_dump = dict(held_obs_ids=held)
        for arm in arms:
            for snapshot in ('selected', 'last'):
                prior = predictions[arm, seed, snapshot]
                source_rows.append(dict(arm=arm, seed=seed, snapshot=snapshot, readout='raw', draw=-1,
                    **metrics(prior[held].argmax(1).numpy(), y[held].numpy())))
                source_dump[f'{arm}_{snapshot}_raw'] = prior[held].numpy()
                for j, (s, q) in enumerate(held_episodes):
                    score = adapt(prior, kernel, s, q, y[s])
                    source_dump[f'{arm}_{snapshot}_{j}'] = score.numpy()
                    source_rows.append(dict(arm=arm, seed=seed, snapshot=snapshot, readout='adapted', draw=j,
                        **metrics(score.argmax(1).numpy(), y[q].numpy()), CE=float(balanced_ce(score, y[q]))))
        np.savez_compressed(OUT/f'source_predictions_{seed}.npz', **source_dump)
        for path in sorted(SUPPORTS.glob('*_split.npz')):
            task = path.stem.removesuffix('_split')
            with np.load(path) as f:
                s, q, sy = [f[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
            assert not set(meta.iloc[s].mmsi) & set(meta.iloc[q].mmsi)
            assert not set(meta.iloc[np.r_[s, q]].mmsi) & set(meta.iloc[source].mmsi)
            dump = dict(support_obs_ids=s, query_obs_ids=q, support_labels=sy)
            for arm in arms:
                for snapshot in ('selected', 'last'):
                    prior = predictions[arm, seed, snapshot]
                    dump[f'{arm}_{snapshot}'] = adapt(prior, kernel, s, q, torch.tensor(sy)).numpy()
                    dump[f'{arm}_{snapshot}_raw'] = prior[q].numpy()
            np.savez_compressed(OUT/f'target_{task}_{seed}.npz', **dump)
    # Target query metrics are read only after every model and task prediction is frozen.
    write_json(OUT/'ALL_TARGET_PREDICTIONS_SAVED.json', dict(query_metrics_read=False))
    for path in sorted(OUT.glob('target_*.npz')):
        with np.load(path) as f:
            q = f['query_obs_ids']; truth = y[q].numpy()
            seed = int(path.stem.rsplit('_', 1)[1])
            task = path.stem[len('target_'):].rsplit('_', 1)[0]
            port = str(meta.iloc[q[0]].port)
            for arm in arms:
                for snapshot in ('selected', 'last'):
                    scores = f[f'{arm}_{snapshot}']; pred = scores.argmax(1)
                    raw_scores = f[f'{arm}_{snapshot}_raw']
                    tag = dict(seed=seed, arm=arm, snapshot=snapshot, task=task, port=port)
                    raw_result = metrics(raw_scores.argmax(1), truth)
                    target_rows.append(dict(**tag, **metrics(pred, truth), raw_OA=raw_result['OA'], raw_BA=raw_result['BA']))
                    for c in np.unique(truth):
                        class_rows.append(dict(**tag, class_id=int(c), count=int((truth == c).sum()),
                            recall=float(100*(pred[truth == c] == c).mean())))
    pd.DataFrame(source_rows).to_csv(OUT/'source_results.csv', index=False)
    pd.DataFrame(target_rows).to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'target_classes.csv', index=False)
    write_json(OUT/'COMPLETE.json', dict(elapsed_seconds=time.perf_counter()-started,
        fits=len(selections), completed=True, main_model_changed=False))
    print('SOURCE', pd.DataFrame(source_rows).groupby(['snapshot', 'readout', 'arm'])[['OA', 'BA']].mean().to_string(), flush=True)
    print('TARGET', pd.DataFrame(target_rows).groupby(['snapshot', 'arm'])[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
