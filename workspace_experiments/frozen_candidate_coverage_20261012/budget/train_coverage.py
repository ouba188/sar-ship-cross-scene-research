"""Prepare each allowed task's source operators, then frozen-recipe correctors."""

import run_coverage as coverage
from run_coverage import (OUT, STUDY, ROOT, SOURCE, TASKS, OLD_PORTS, BUDGETS, DOSES,
                          SEEDS, Features, archive, read_json, write, task_folders,
                          np, pd, Path, time, threadpool_limits)
import argparse
import gc
import json
import os
import torch
from sklearn.base import clone
from sklearn.linear_model import RidgeClassifier
from case_support_completion import COLS, weights, predict
from support_task_loss import ResidualCorrector, common_class_targets
from unified_head_core import coefficients, late
from unified_head_run import base_kernel
from direct_incidence_conditioning import extend_transforms
from protected_drift_activation import fit_source_heads
from run_protected_drift_activation import make_transforms
from budget_fusion_source_base import _features
from conditional_direction_residual import rank_conditions, normalize_direction
from cross_dataset_c4 import base_coordinates, base_phi as external_base_phi
import run_support_task_loss as recipe


def eligible_paths(requested):
    audit = pd.read_csv(OUT/'coverage_audit.csv')
    valid = set(audit.loc[audit.unchanged_episode_recipe_feasible.eq(True), 'task'])
    return [p for p in task_folders() if read_json(p)['task'] in valid
            and (read_json(p)['task'] in requested if requested else read_json(p)['task'] not in OLD_PORTS)]


def source_prepare(path):
    info = read_json(path)
    ds, task = info['dataset'], info['task']
    dest = OUT/'training'/ds/task
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'SOURCE_PREPARED.json').exists():
        return
    started = time.perf_counter()
    meta = pd.read_csv(STUDY/'observations.csv')
    original = archive(STUDY/'crossfit_evidence_20261011/source_folds.npz')
    features = Features(path.parent)
    keep = np.isin(meta.iloc[original['fit_obs_ids']].manifest_row, features.source_rows)
    ids, assignment = original['fit_obs_ids'][keep], original['assignment'][keep]
    mm = meta.iloc[ids].reset_index(drop=True)
    rows = mm.manifest_row.to_numpy(int)
    pos = features.index.get_indexer(rows)
    assert (pos >= 0).all()
    fold_ships = [mm.loc[assignment == k].mmsi.nunique() for k in range(3)]
    if min(fold_ships) < 160:
        raise ValueError('Original episode recipe not feasible; no silent resizing')
    original_maps = archive(STUDY/'case_relation_headroom_20261012/Los Angeles/maps.npz')
    held_ids = original_maps['obs_ids'][len(original['fit_obs_ids']):]
    held_ships = meta.iloc[held_ids].mmsi.unique()
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'class_id', 'port', 'mmsi'])
    sf = manifest.iloc[features.source_rows]
    if ds == 'self24ports':
        assert not sf.port.eq(task).any() and not mm.port.eq(task).any()
    cf = pd.read_csv(ROOT/'c4_condition_diagnosis_20261009/descriptors/self24ports.csv')
    np.testing.assert_array_equal(cf.instance_id, manifest.instance_id)
    conditions = np.empty((len(manifest), 2))
    for _, ix in manifest.groupby('port', sort=False).groups.items():
        conditions[ix] = rank_conditions(cf.loc[ix])
    averaged = np.load(ROOT/'c4_base_trial_20261009/pooled_c4.npy', mmap_mode='r')
    phi = features.source_phi(rows)
    if ds == 'self24ports':
        g = features.geometry
        augmented = np.c_[features.bank_source, features.u[features.source_rows]]
        bt, base_augmented = _features(rows, features.bank_source[pos], g, averaged,
                                       features.u, conditions)
        transforms = extend_transforms({k: v for k, v in make_transforms(g).items()
                                        if k in ('joint_base', 'joint_expert')})
        def fit_pair(use):
            heads = fit_source_heads(augmented[use], sf.class_id.to_numpy()[use], transforms)
            bp = base_augmented@heads['joint_base']['coef']+heads['joint_base']['intercept']
            ep = augmented[pos]@heads['joint_expert']['coef']+heads['joint_expert']['intercept']
            return bp, ep
    else:
        from unified_head_expert import ExpertContext
        from compact_fusion_source import _expert_inputs
        context = ExpertContext(path.parent)
        context._prepare_source()
        np.testing.assert_array_equal(context.source_rows, features.source_rows)
        bmodel = coverage.load_pickle(Path(info['model_path']))
        x0 = base_coordinates(features.raw_source[features.source_rows], bmodel)
        x4 = base_coordinates(averaged[rows], bmodel)
        bt = dict(phi=external_base_phi(x4, bmodel),
                  mean=normalize_direction((x4-bmodel['residual_center'])/bmodel['residual_scale']),
                  conditions=conditions[rows])
        ex = _expert_inputs(features.raw_source[rows], features.extra_source[pos], context.model)
        fit_ex = np.empty((len(features.source_rows), 192), dtype=float)
        for lo in range(0, len(fit_ex), 2048):
            rr = features.source_rows[lo:lo+2048]
            fit_ex[lo:lo+len(rr)] = _expert_inputs(features.raw_source[rr],
                features.extra_source[lo:lo+len(rr)], context.model)
        def fit_pair(use):
            bh = RidgeClassifier(alpha=1., class_weight='balanced').fit(x0[use], sf.class_id.to_numpy()[use])
            # Match the stronger current head: fit on post-tail deployment inputs.
            eh = clone(context.model['expert']).fit(fit_ex[use], sf.class_id.to_numpy()[use])
            np.testing.assert_array_equal(bh.classes_, np.arange(8))
            np.testing.assert_array_equal(eh.classes_, np.arange(8))
            return bh.decision_function(x4), eh.decision_function(ex).astype(np.float32).astype(float)
    base_prior = np.empty((len(mm), 8))
    expert_prior = np.empty_like(base_prior)
    audits = []
    for k in range(3):
        val = assignment == k
        excluded = np.r_[held_ships, mm.loc[val].mmsi.unique()]
        use = ~sf.mmsi.isin(excluded).to_numpy()
        assert not np.isin(sf.mmsi.to_numpy()[use], excluded).any()
        bp, ep = fit_pair(use)
        base_prior[val], expert_prior[val] = bp[val], ep[val]
        audits.append(dict(fold=k, train_rows=int(use.sum()), validation_rows=int(val.sum()),
                           fold_ships=int(fold_ships[k]), excluded_vessel_overlap=0))
    bridge = dict(original_fit_rows=len(original['fit_obs_ids']), fit_rows=len(mm),
        removed_rows=int((~keep).sum()), removed_obs_ids=original['fit_obs_ids'][~keep].tolist(),
        retained_obs_ids=ids.tolist(), fixed_original_fold_assignment=True,
        excluded_held_ships=len(held_ships), reason='Intersection with this outer task allowed source rows',
        same_input_for_all_arms=True, not_same_weights_as_three_port_pilot=True,
        interpretation='New task-specific recipe refit; source cohort and geometry change together versus old3 tasks')
    write(dest/'SOURCE_COHORT_BRIDGE.json', bridge)
    np.savez_compressed(dest/'source.npz', obs_ids=ids, assignment=assignment, phi=phi,
        prior=expert_prior, base_prior=base_prior, base_phi=bt['phi'], mean=bt['mean'],
        conditions=bt['conditions'], eta=info['eta'], rows=rows)
    if task in OLD_PORTS:
        old = recipe.prepare_context(task, meta, original, 'cpu')
        bd = archive(recipe.REFERENCE/'base_cache'/task/'base.npz')
        errors = {}
        for key, new, expected in [('phi', phi, old['phi']), ('expert_prior', expert_prior, old['prior']),
                                  ('base_prior', base_prior, old['base_prior']),
                                  ('base_phi', bt['phi'], bd['phi'][:len(mm)]),
                                  ('mean', bt['mean'], bd['mean'][:len(mm)]),
                                  ('conditions', bt['conditions'], bd['conditions'][:len(mm)])]:
            error = float(abs(new-expected).max())
            np.testing.assert_allclose(new, expected, rtol=1e-7, atol=1e-7, err_msg=key)
            errors[key] = error
        write(dest/'SOURCE_REPLAY.json', errors)
    write(dest/'SOURCE_PREPARED.json', dict(dataset=ds, task=task, source_rows=len(mm),
        folds=audits, geometry_task=task, source_heads_refitted=6,
        external_head_recipe='post-tail matched refit' if ds != 'self24ports' else None,
        gpu_used=False, seconds=time.perf_counter()-started))
    print('SOURCE PREPARED', ds, task, len(mm), round(time.perf_counter()-started, 2), flush=True)
    del features
    gc.collect()


def encode(path, device):
    info = read_json(path)
    dest = OUT/'training'/info['dataset']/info['task']
    if (dest/'FREQUENCY_READY.json').exists():
        return
    data = archive(dest/'source.npz')
    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from learned_frequency_model import FrequencyResidual
    model = FrequencyResidual('selective').to(device).eval()
    model.load_state_dict(torch.load(path.parent/'selective.pt', map_location='cpu', weights_only=True)['state'])
    model.requires_grad_(False)
    maps_root = ROOT/'learned_frequency_residual_20261009/maps/self24ports'
    maps = np.load(maps_root/'maps.npy', mmap_mode='r')
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id'])
    np.testing.assert_array_equal(np.load(maps_root/'sample_ids.npy')[data['rows']],
                                  manifest.iloc[data['rows']].instance_id.to_numpy(str))
    result = np.empty((len(data['rows']), 256), np.float32)
    start = time.perf_counter()
    with torch.inference_mode():
        for lo in range(0, len(result), 128):
            rows = data['rows'][lo:lo+128]
            x = torch.as_tensor(np.asarray(maps[rows], dtype=np.float32), device=device)
            c = torch.as_tensor(data['conditions'][lo:lo+len(rows)], dtype=torch.float32, device=device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                z, _ = model(x, c)
            result[lo:lo+len(rows)] = z.cpu().numpy()
    replay = None
    if info['task'] in OLD_PORTS:
        legacy = archive(recipe.REFERENCE/'base_cache'/info['task']/'base.npz')
        expected = legacy['embedding'][:len(result)]
        replay = float(abs(result-expected).max())
        boundary = len(result)//128*128
        evidence = dict(before_max=replay, prefix_max=float(abs(result[:boundary]-expected[:boundary]).max()),
                        final_batch=len(result)-boundary, original_batch=128)
        # Old cache encoded fit+held together; match its final fit batch exactly.
        if boundary < len(result):
            meta = pd.read_csv(STUDY/'observations.csv', usecols=['manifest_row'])
            rr = meta.iloc[legacy['obs_ids'][boundary:boundary+128]].manifest_row.to_numpy()
            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                x = torch.as_tensor(np.asarray(maps[rr], dtype=np.float32), device=device)
                cc = torch.as_tensor(legacy['conditions'][boundary:boundary+128], dtype=torch.float32, device=device)
                last, _ = model(x, cc)
            result[boundary:] = last[:len(result)-boundary].cpu().numpy()
        evidence['after_max'] = float(abs(result-expected).max())
        write(dest/'FREQUENCY_LAYOUT_PROBE.json', evidence)
        replay = evidence['after_max']
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)
    np.save(dest/'embedding.npy', result)
    write(dest/'FREQUENCY_READY.json', dict(device=device, rows=len(result), seconds=time.perf_counter()-start,
        frozen_checkpoint=str(path.parent/'selective.pt'), neural_training=False, original_replay_error=replay))
    print('SOURCE FREQUENCY', info['task'], len(result), round(time.perf_counter()-start, 2), flush=True)


def train(path):
    info = read_json(path)
    dest = OUT/'training'/info['dataset']/info['task']
    if (dest/'MODELS_FROZEN.json').exists():
        return
    source = archive(dest/'source.npz')
    meta = pd.read_csv(STUDY/'observations.csv')
    mm = meta.iloc[source['obs_ids']].reset_index(drop=True)
    x, phi = source['phi'][:, COLS], source['phi']
    w = weights(mm)
    wanted = common_class_targets(x, mm, w)
    initial_root = 'tasks' if info['dataset'] == 'self24ports' else 'strong_external'
    initial = archive(OUT/initial_root/info['dataset']/info['task']/'initial.npz')
    components = dict(phi=source['base_phi'], mean=source['mean'], conditions=source['conditions'])
    kb = base_kernel(components, np.load(dest/'embedding.npy').astype(float), np.arange(len(mm)), float(source['eta']))
    t = lambda a, integer=False: recipe.tensor(a, 'cpu', integer)
    data = dict(phi=phi, prior=source['prior'], base_prior=source['base_prior'], kb=kb,
        y=mm.class_id.to_numpy(), initial=initial, cols=t(COLS, True), tphi=t(phi), tx=t(x),
        tprior=t(source['prior']), ty=t(mm.class_id.to_numpy(), True), twanted=t(wanted),
        tweight=t(w), mse_scale=float(np.average(np.square(wanted).mean(1), weights=w)))
    torch.set_num_threads(2)
    traces, episodes_log, costs = [], [], []
    for seed in SEEDS:
        definitions = recipe.training_episodes(meta, source['obs_ids'], source['assignment'], seed)
        assert len(definitions) == 90
        episodes = recipe.attach_base(data, definitions, 'cpu')
        if seed == SEEDS[0]:
            write(dest/'INITIAL_SOLVER_REPLAY.json', recipe.verify_initial(data, episodes, 'cpu'))
        for step, ep in enumerate(definitions):
            episodes_log.append(dict(seed=seed, step=step+1, budget=ep['budget'], fold=ep['fold'],
                support_obs_ids=json.dumps(source['obs_ids'][ep['s']].tolist()),
                query_obs_ids=json.dumps(source['obs_ids'][ep['q']].tolist())))
        for mode, budget in [('task', None)]+[('isolated', b) for b in BUDGETS]:
            torch.manual_seed(seed)
            model = ResidualCorrector(initial).to('cpu')
            optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.LR, weight_decay=1e-4)
            chosen = [ep for ep in episodes if budget is None or ep['budget'] == budget]
            assert len(chosen) == (90 if budget is None else 30)
            started = time.perf_counter()
            for step, ep in enumerate(chosen, 1):
                _, loss, mse, _ = recipe.losses(model, data, ep)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = sum(p.grad.square().sum() for p in model.parameters()).sqrt()
                if not torch.isfinite(loss) or not torch.isfinite(norm) or float(norm) == 0:
                    raise ValueError('Invalid task objective/gradient')
                optimizer.step()
                traces.append(dict(seed=seed, mode=mode, isolated_budget=budget, step=step,
                    budget=ep['budget'], ce=float(loss.detach()), mse=float(mse.detach()), grad=float(norm)))
            filename = f'task_{seed}.npz' if budget is None else f'isolated_{seed}_{budget}.npz'
            np.savez_compressed(dest/filename, **model.numpy_state())
            costs.append(dict(seed=seed, mode=mode, budget=budget, steps=len(chosen),
                              seconds=time.perf_counter()-started))
            print('TRAINED', info['task'], mode, seed, budget, round(costs[-1]['seconds'], 2), flush=True)
    pd.DataFrame(traces).to_csv(dest/'training_trace.csv', index=False)
    pd.DataFrame(episodes_log).to_csv(dest/'training_episodes.csv', index=False)
    pd.DataFrame(costs).to_csv(dest/'training_cost.csv', index=False)
    write(dest/'MODELS_FROZEN.json', dict(dataset=info['dataset'], task=info['task'], fits=len(costs),
        optimizer='AdamW', lr=recipe.LR, weight_decay=1e-4, shared_steps=90, isolated_steps=30,
        seeds=SEEDS, no_checkpoint_selection=True, backbone_training=False, cpu_threads=2,
        same_episode_ids_and_order=True, source_fit_rows=len(mm)))


def predict_trained(path):
    info = read_json(path)
    model_dir = OUT/'training'/info['dataset']/info['task']
    dest = OUT/'trained_predictions'/info['dataset']/info['task']
    if (dest/'PREDICTIONS_FROZEN.json').exists():
        return
    if not (model_dir/'MODELS_FROZEN.json').exists():
        raise ValueError('Training incomplete')
    started = time.perf_counter()
    dest.mkdir(parents=True, exist_ok=True)
    feature = Features(path.parent)
    if info['dataset'] != 'self24ports':
        from strong_external import CachedStrongFeatures
        feature = CachedStrongFeatures(info['dataset'])
    index = pd.Index(feature.target_rows)
    for draw in range(5):
        for budget in BUDGETS:
            origin = OUT/('tasks' if info['dataset'] == 'self24ports' else 'strong_external')
            original = origin/info['dataset']/info['task']/f'{draw}_{budget}.npz'
            z = archive(original)
            s, q = index.get_indexer(z['support_ids']), index.get_indexer(z['query_ids'])
            assert (s >= 0).all() and (q >= 0).all()
            sp, ps = feature.target(s)
            for seed in SEEDS:
                for mode in ('task', 'isolated'):
                    name = f'task_{seed}.npz' if mode == 'task' else f'isolated_{seed}_{budget}.npz'
                    model = archive(model_dir/name)
                    delta = predict(model, sp[:, COLS], np.zeros(budget, int), 'image_history')
                    corrected = coverage.adapt_support(sp, delta, DOSES[budget])
                    alpha = coefficients(corrected@corrected.T, ps, z['support_labels'], budget/8.)
                    score = np.empty((len(q), 8))
                    for lo in range(0, len(q), 2048):
                        qp, pq = feature.target(q[lo:lo+2048])
                        score[lo:lo+len(qp)] = pq+(qp@corrected.T)@alpha
                    z[f'expert_{mode}_{seed}'] = score
                    z[f'fusion_{mode}_{seed}'] = late(z['base'], score)
            np.savez_compressed(dest/f'{draw}_{budget}.npz', **z)
    receipt = read_json(origin/info['dataset']/info['task']/'PREDICTIONS_FROZEN.json')
    receipt['baseline_stage_seconds'] = receipt.pop('seconds', None)
    receipt.update(new_shared_or_isolated_fits=8, shared_isolated_coverage='task-specific frozen recipe refit')
    receipt['seconds'] = time.perf_counter()-started
    write(dest/'PREDICTIONS_FROZEN.json', receipt)
    print('TRAINED PREDICTIONS FROZEN', info['task'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'encode', 'train', 'predict'])
    parser.add_argument('--task', action='append')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        for path in eligible_paths(args.task):
            if args.stage == 'prepare':
                source_prepare(path)
            elif args.stage == 'encode':
                encode(path, args.device)
            elif args.stage == 'train':
                train(path)
            else:
                predict_trained(path)
