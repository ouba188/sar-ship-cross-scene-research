"""Matched source-only geometry-teacher trial; no angle required at deployment."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import copy
import time
import numpy as np
import pandas as pd
import torch
from geometry_train_teacher import ARMS, ANGLE_SCALE, make_teachers
from knowledge_train_only import (SEEDS, STEPS, KnowledgeStudent, SAROnlyPredictor,
    load_upstream, make_schedule, train_arm, balanced_ce)
from execute_knowledge_trial import adapt
from relative_content_condition import load_data, BASE, SUPPORTS
from scene_training import sample_weights, metrics
from probe_knowledge_attributes import write_json

OUT = BASE/'geometry_train_teacher_20261011'
CACHE = BASE/'crossfit_evidence_20261011'
PREVIOUS = BASE/'knowledge_train_only_20261011/training'


def main(output=None, fit_arms=('B', 'far', 'near', 'sham')):
    global OUT
    if output is not None:
        OUT = output
    arms = tuple(dict.fromkeys(('B',)+tuple(fit_arms)))
    torch.set_num_threads(1)
    start = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    del norm, fixed, h
    fitting, held = split['factor_fit'], split['test_source']
    fm = meta.iloc[fitting].reset_index(drop=True)
    y = torch.tensor(meta.class_id.to_numpy(copy=True))
    probability = sample_weights(fm); probability /= probability.sum()
    with np.load(PREVIOUS/'evaluation_episodes.npz') as z:
        episodes = {role: [(z[f'{role}_{j}_support'], z[f'{role}_{j}_query']) for j in range(3)]
                    for role in ('validation', 'held')}
    np.savez_compressed(OUT/'evaluation_episodes.npz', **{
        f'{role}_{j}_{side}': ids for role, ep in episodes.items()
        for j, pair in enumerate(ep) for side, ids in zip(('support', 'query'), pair)})
    write_json(OUT/'PROTOCOL.json', dict(
        arms=list(arms), trained_arms=list(fit_arms), seeds=list(SEEDS), steps=STEPS, angle_scale_degrees=ANGLE_SCALE,
        baseline='B same-vessel other-product OOF teaching; original classification/support losses retained.',
        intervention='far: favor different incidence; near: favor similar incidence; sham: far after within-vessel permutation.',
        weights='Known donor weights 1..2 then normalized to preserve known donor total mass; unknowns unchanged. At least2known donors.',
        metadata='Only source-fit incidence builds teachers. No target/held metadata in training or model selection.',
        selection='Same3new-vessel SAR support episodes, BA then CE then earliest, every20steps including0.',
        inference='SAR only, same64SAR support labels. No incidence/MMSI/product/history/auxiliary head at prediction.',
        limitation='Reused developmental ports, frozen historical R18; teacher weighting is not a causal intervention. Angle/port correlations not eliminated.',
        budget='2seeds, no angle-scale or loss-weight sweep; freeze all before held/target metrics. Last snapshots diagnostic only.',
        followup='sham_near is a post-original-result specificity control, not an independent confirmation.' if 'sham_near' in fit_arms else 'Original4arms8fits.'))
    dummy = dict(real=torch.zeros(len(fm), 2), control=torch.zeros(len(fm), 2),
                 valid=torch.zeros(len(fm), dtype=torch.bool), importance=torch.zeros(len(fm)))
    traces, selections, replays, audits, teacher_quality = [], [], [], [], []
    kernels, predictions = {}, {}
    for seed in SEEDS:
        upstream = load_upstream(CACHE, seed, raw, meta, fitting, episodes['validation'])
        audits.extend(upstream['audit'])
        x, kernel = upstream['x'], upstream['kernel']; kernels[seed] = kernel
        teacher, valid, coverage, edges = make_teachers(upstream['oof_prior'], fm)
        np.testing.assert_array_equal(teacher['B'], upstream['teacher'].numpy())
        np.testing.assert_array_equal(valid, upstream['teacher_valid'].numpy())
        coverage['obs_id'] = fitting[coverage.local_id]
        coverage.to_csv(OUT/f'teacher_coverage_{seed}.csv', index=False)
        edges.to_csv(OUT/f'teacher_edges_{seed}.csv', index=False)
        np.savez_compressed(OUT/f'teachers_{seed}.npz', **teacher, valid=valid, fit_obs_ids=fitting,
                            original_oof=upstream['oof_prior'])
        changed = coverage[(coverage.arm == 'far') & coverage.weights_changed].local_id.to_numpy()
        for arm in arms:
            for name, ix in [('all_multiview', np.flatnonzero(valid)), ('changed_far', changed)]:
                truth = y[fitting[ix]].numpy(); score = teacher[arm][ix]
                base = teacher['B'][ix].argmax(1); pred = score.argmax(1)
                correct = score[np.arange(len(ix)), truth]
                rival = score.copy(); rival[np.arange(len(ix)), truth] = -np.inf
                teacher_quality.append(dict(seed=seed, arm=arm, cohort=name, n=len(ix),
                    **metrics(pred, truth), margin=float(np.mean(correct-rival.max(1))),
                    rescue=int(((base != truth)&(pred == truth)).sum()),
                    harm=int(((base == truth)&(pred != truth)).sum())))
        schedule = make_schedule(fm, probability, seed)
        with np.load(PREVIOUS/f'train_schedule_{seed}.npz') as old:
            for key in schedule:
                np.testing.assert_array_equal(schedule[key], old[key])
        torch.manual_seed(seed)
        initial = KnowledgeStudent(x[fitting], y[fitting], probability)
        old_initial = torch.load(PREVIOUS/f'initial_{seed}.pt', weights_only=True)
        for key, value in initial.state_dict().items():
            torch.testing.assert_close(value, old_initial[key], rtol=0, atol=0)
        torch.save(initial.state_dict(), OUT/f'initial_{seed}.pt')
        if 'B' not in fit_arms:
            for snapshot in ('selected', 'last'):
                with np.load(PREVIOUS/f'B_{seed}_{snapshot}.npz') as old:
                    predictions['B', seed, snapshot] = torch.from_numpy(old['prior'])
        for arm in fit_arms:
            print('START', seed, arm, flush=True)
            # All arms use exactly the B objective, changing only its fixed teacher.
            model, states, trace, choice = train_arm(copy.deepcopy(initial), x, y, kernel, fitting,
                torch.from_numpy(teacher[arm]), torch.from_numpy(valid), dummy, schedule,
                episodes['validation'], seed, 'B', 0.)
            for row in trace:
                row['arm'] = arm
            choice['arm'] = arm
            traces.extend(trace); selections.append(choice)
            torch.save(states, OUT/f'{arm}_{seed}.pt')
            pd.DataFrame(traces).to_csv(OUT/'training.csv', index=False)
            pd.DataFrame(selections).to_csv(OUT/'selection.csv', index=False)
            for snapshot in ('selected', 'last'):
                model.load_state_dict(states[snapshot])
                with torch.no_grad():
                    z, prior = model(x)
                predictions[arm, seed, snapshot] = prior
                np.savez_compressed(OUT/f'{arm}_{seed}_{snapshot}.npz', z=z.numpy(), prior=prior.numpy())
            model.load_state_dict(states['selected'])
            deploy = SAROnlyPredictor.from_student(model, upstream['case_state'],
                upstream['raw_center'], upstream['raw_scale'])
            torch.save(deploy.state_dict(), OUT/f'deployment_{arm}_{seed}.pt')
            reloaded = SAROnlyPredictor(torch.load(OUT/f'deployment_{arm}_{seed}.pt', weights_only=True))
            with torch.no_grad():
                p, k = reloaded(torch.from_numpy(raw))
            np.testing.assert_array_equal(p.numpy(), predictions[arm, seed, 'selected'].numpy())
            np.testing.assert_array_equal(k.numpy(), kernel.numpy())
            assert not any('attribute' in key for key in reloaded.state_dict())
            if arm == 'B':
                with np.load(PREVIOUS/f'B_{seed}_selected.npz') as old:
                    np.testing.assert_array_equal(old['prior'], p.numpy())
            replays.append(dict(seed=seed, arm=arm, rows=len(raw), max_prior_error=0,
                                max_kernel_error=0, metadata_free=True))
            print('SELECTED', choice, flush=True)
    pd.DataFrame(replays).to_csv(OUT/'deployment_replay.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT/'upstream_audit.csv', index=False)
    pd.DataFrame(teacher_quality).to_csv(OUT/'fit_teacher_quality.csv', index=False)
    write_json(OUT/'ALL_MODELS_FROZEN.json', dict(fits=len(selections), target_query_metrics_read=False))
    source_rows, target_rows, class_rows, events = [], [], [], []
    for seed in SEEDS:
        kernel = kernels[seed]
        for arm in arms:
            for snapshot in ('selected', 'last'):
                prior = predictions[arm, seed, snapshot]
                source_rows.append(dict(seed=seed, arm=arm, snapshot=snapshot, readout='raw', draw=-1,
                    **metrics(prior[held].argmax(1).numpy(), y[held].numpy())))
                for j, (s, q) in enumerate(episodes['held']):
                    score = adapt(prior, kernel, s, q, y[s])
                    source_rows.append(dict(seed=seed, arm=arm, snapshot=snapshot, readout='adapted', draw=j,
                        **metrics(score.argmax(1).numpy(), y[q].numpy()), CE=float(balanced_ce(score, y[q]))))
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
    write_json(OUT/'ALL_TARGET_PREDICTIONS_SAVED.json', dict(target_query_metrics_read=False))
    for path in sorted(OUT.glob('target_*.npz')):
        with np.load(path) as f:
            q = f['query_obs_ids']; truth = y[q].numpy()
            seed = int(path.stem.rsplit('_', 1)[1])
            task = path.stem[len('target_'):].rsplit('_', 1)[0]
            port = str(meta.iloc[q[0]].port)
            for arm in arms:
                for snapshot in ('selected', 'last'):
                    pred = f[f'{arm}_{snapshot}'].argmax(1)
                    baseline = f[f'B_{snapshot}'].argmax(1)
                    raw_score = f[f'{arm}_{snapshot}_raw']
                    tag = dict(seed=seed, arm=arm, snapshot=snapshot, task=task, port=port)
                    raw_metrics = metrics(raw_score.argmax(1), truth)
                    target_rows.append(dict(**tag, **metrics(pred, truth),
                        raw_OA=raw_metrics['OA'], raw_BA=raw_metrics['BA'],
                        rescue=int(((baseline != truth)&(pred == truth)).sum()),
                        harm=int(((baseline == truth)&(pred != truth)).sum())))
                    for c in np.unique(truth):
                        mask = truth == c
                        class_rows.append(dict(**tag, class_id=int(c), n=int(mask.sum()),
                            recall=float(100*(pred[mask] == c).mean()),
                            delta_recall=float(100*((pred[mask] == c).mean()-(baseline[mask] == c).mean()))))
                    if arm != 'B' and snapshot == 'selected':
                        for j in np.flatnonzero((pred == truth) != (baseline == truth)):
                            events.append(dict(**tag, obs_id=int(q[j]), mmsi=int(meta.iloc[q[j]].mmsi),
                                class_id=int(truth[j]), base=int(baseline[j]), prediction=int(pred[j]),
                                event='rescue' if pred[j] == truth[j] else 'harm'))
    pd.DataFrame(source_rows).to_csv(OUT/'source_results.csv', index=False)
    pd.DataFrame(target_rows).to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'target_classes.csv', index=False)
    pd.DataFrame(events).to_csv(OUT/'target_events.csv', index=False)
    write_json(OUT/'COMPLETE.json', dict(completed=True, fits=len(selections),
        elapsed_seconds=time.perf_counter()-start, main_model_changed=False))
    print('SOURCE', pd.DataFrame(source_rows).groupby(['snapshot', 'readout', 'arm'])[['OA', 'BA']].mean().to_string(), flush=True)
    print('TARGET', pd.DataFrame(target_rows).groupby(['snapshot', 'arm'])[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
