"""Matched reconstruction/task training of the same support corrector."""

from full_teacher_data import BASE, SOURCE, PORTS, write
from case_support_completion import COLS, predict, weights as reconstruction_weights
from run_support_budget_strength import corrected_score, BUDGETS
from unified_head_core import coefficients, late
from unified_head_run import base_kernel
from evaluate_probability_layers import cached, probability_rows
from support_task_loss import ResidualCorrector, common_class_targets, torch_scores, fusion_ce
import run_support_budget as previous

import argparse
import json
import time
import numpy as np
import pandas as pd
import torch


OUT = BASE/'support_task_loss_20261012'
REFERENCE = BASE/'support_budget_fusion_select_20261012'
SEEDS = (31415, 27182)
DOSES = {16: 1., 64: .5, 128: .25}
STEPS = 90
LR = 1e-4
ARMS = ('reconstruction', 'task')


def tensor(x, device, integer=False):
    return torch.as_tensor(x, dtype=torch.long if integer else torch.float64, device=device)


def training_episodes(meta, fit_ids, assignment, seed, rounds=10):
    rng = np.random.default_rng(seed)
    episodes = []
    for round_id in range(rounds):
        for fold in rng.permutation(3):
            ids = np.flatnonzero(assignment == fold)
            ships = meta.iloc[fit_ids[ids]].mmsi.to_numpy()
            ordered = rng.permutation(np.unique(ships))
            assert len(ordered) >= 160
            q = np.array([rng.choice(ids[ships == ship]) for ship in ordered[:32]])
            full_s = np.array([rng.choice(ids[ships == ship]) for ship in ordered[32:160]])
            assert not set(meta.iloc[fit_ids[q]].mmsi) & set(meta.iloc[fit_ids[full_s]].mmsi)
            for budget in rng.permutation(BUDGETS):
                episodes.append(dict(fold=int(fold), round=int(round_id), budget=int(budget),
                                     s=full_s[:budget], q=q))
    return episodes


def prepare_context(context, meta, folds, device):
    maps = cached(previous.TEACHER/context/'maps.npz')
    nfit = len(folds['fit_obs_ids'])
    np.testing.assert_array_equal(maps['obs_ids'][:nfit], folds['fit_obs_ids'])
    phi = maps['single'][:nfit]
    prior = cached(BASE/'case_relation_student_20261012'/context/'source_readers.npz')['fit_prior']
    bd = cached(REFERENCE/'base_cache'/context/'base.npz')
    np.testing.assert_array_equal(maps['obs_ids'], bd['obs_ids'])
    components = {k: bd[k][:nfit] for k in ('phi', 'mean', 'conditions')}
    kb = base_kernel(components, bd['embedding'][:nfit].astype(float), np.arange(nfit), float(bd['eta']))
    frame = meta.iloc[folds['fit_obs_ids']].reset_index(drop=True)
    x = phi[:, COLS]
    weights = reconstruction_weights(frame)
    wanted = common_class_targets(x, frame, weights)
    scale = float(np.average(np.square(wanted).mean(1), weights=weights))
    initial = cached(previous.CONTROL/context/'final_model.npz')
    result = dict(phi=phi, prior=prior, base_prior=bd['prior_oof'], kb=kb, initial=initial,
                  y=frame.class_id.to_numpy(), cols=tensor(COLS, device, True),
                  tphi=tensor(phi, device), tx=tensor(x, device), tprior=tensor(prior, device),
                  ty=tensor(frame.class_id.to_numpy(), device, True), twanted=tensor(wanted, device),
                  tweight=tensor(weights, device), mse_scale=scale)
    return result


def attach_base(data, episodes, device):
    answer = []
    for ep in episodes:
        s, q, b = ep['s'], ep['q'], ep['budget']
        alpha = coefficients(data['kb'][np.ix_(s, s)], data['base_prior'][s], data['y'][s], previous.strength(b))
        base = data['base_prior'][q]+data['kb'][np.ix_(q, s)]@alpha
        answer.append(dict(**ep, ts=tensor(s, device, True), tq=tensor(q, device, True),
                           ti=tensor(np.r_[s, q], device, True), tbase=tensor(base, device)))
    return answer


def losses(model, data, ep):
    s, q, ix = ep['ts'], ep['tq'], ep['ti']
    delta = model(data['tx'][s])
    expert = torch_scores(data['tphi'][s], data['tphi'][q], data['tprior'][s], data['tprior'][q],
                          data['ty'][s], delta, data['cols'], DOSES[ep['budget']], previous.strength(ep['budget']))
    task = fusion_ce(ep['tbase'], expert, data['ty'][q])
    row_error = (model(data['tx'][ix])-data['twanted'][ix]).square().mean(1)
    w = data['tweight'][ix]
    reconstruction = (row_error*w).sum()/w.sum()
    return reconstruction/data['mse_scale'], task, reconstruction, expert


def fitting_probe(model, data, probes):
    records = []
    with torch.no_grad():
        for ep in probes:
            normalized, task, mse, expert = losses(model, data, ep)
            y = data['y'][ep['q']]
            e = expert.cpu().numpy()
            full = late(ep['tbase'].cpu().numpy(), e)
            records.append(dict(budget=ep['budget'], fold=ep['fold'], task_ce=float(task),
                feature_mse=float(mse), normalized_mse=float(normalized),
                expert_ba=previous.ba(y, e), fusion_ba=previous.ba(y, full)))
    return records


def verify_initial(data, episodes, device):
    model = ResidualCorrector(data['initial']).to(device)
    errors = []
    for budget in BUDGETS:
        ep = next(e for e in episodes if e['budget'] == budget)
        s, q = ep['s'], ep['q']
        delta = predict(data['initial'], data['phi'][s][:, COLS], np.zeros(len(s), int), 'image_history')
        expected = corrected_score(data['phi'], data['prior'], s, q, data['y'][s], delta, DOSES[budget])
        with torch.no_grad():
            actual = losses(model, data, ep)[3].cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
        errors.append(dict(budget=budget, maximum_error=float(abs(actual-expected).max())))
    return errors


def train():
    OUT.mkdir(exist_ok=True)
    device = 'cuda'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = pd.read_csv(BASE/'observations.csv')
    folds = cached(BASE/'crossfit_evidence_20261011/source_folds.npz')
    definitions = {seed: training_episodes(meta, folds['fit_obs_ids'], folds['assignment'], seed) for seed in SEEDS}
    episode_rows = []
    for seed, episodes in definitions.items():
        assert len(episodes) == STEPS
        for step, ep in enumerate(episodes):
            episode_rows.append(dict(seed=seed, step=step, budget=ep['budget'], fold=ep['fold'],
                support_obs_ids=json.dumps(folds['fit_obs_ids'][ep['s']].tolist()),
                query_obs_ids=json.dumps(folds['fit_obs_ids'][ep['q']].tolist())))
    pd.DataFrame(episode_rows).to_csv(OUT/'training_episodes.csv', index=False)
    write(OUT/'FROZEN_TRAINING.json', dict(seeds=SEEDS, steps=STEPS, learning_rate=LR, weight_decay=1e-4,
        optimizer='AdamW', doses=DOSES, label_budgets=BUDGETS, initial='previous class-common final_model',
        fitting_rows=1296, fitting_labels_available_to_both_objectives=True,
        reconstruction_target='Same-class OTHER-vessel mean from all1296 fitting rows; not same-vessel history',
        previous_initializer_fitting_rows=1040, fitting_data_expansion_matched_between_objectives=True,
        objective='Class-balanced CE of actual equal probability fusion versus normalized feature residual MSE',
        objective_scale='Feature MSE divided by fixed source target energy; positive constant preserves its minimizer',
        port_reweighting_changed=False, checkpoint='Final90 updates only; no held selection',
        frozen='Backbones, base, source readers, query features, dose schedule, centroid/energy constraints',
        task_support_solver='Recompute KSS/KQS and differentiate through solve, signed one-hot targets',
        target_labels_used=False, trainable_parameters=576*576+576,
        acceptance='Repeatable task-scoped net benefit worth cost; not universal port/class/draw wins'))
    traces, probe_rows, checks, costs = [], [], [], []
    for context in PORTS:
        data = prepare_context(context, meta, folds, device)
        probe_defs = training_episodes(meta, folds['fit_obs_ids'], folds['assignment'], 125012, rounds=1)
        probes = attach_base(data, probe_defs, device)
        checks.extend(dict(context=context, **r) for r in verify_initial(data, probes, device))
        for seed in SEEDS:
            episodes = attach_base(data, definitions[seed], device)
            for arm in ARMS:
                torch.manual_seed(seed)
                model = ResidualCorrector(data['initial']).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
                for r in fitting_probe(model, data, probes):
                    probe_rows.append(dict(context=context, seed=seed, arm=arm, step=0, **r))
                torch.cuda.synchronize()
                started = time.perf_counter()
                for step, ep in enumerate(episodes, 1):
                    mse, task, raw_mse, _ = losses(model, data, ep)
                    loss = mse if arm == 'reconstruction' else task
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite objective')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = sum(p.grad.square().sum() for p in model.parameters()).sqrt()
                    if not torch.isfinite(grad) or float(grad) == 0:
                        raise ValueError('Invalid correction gradient')
                    optimizer.step()
                    traces.append(dict(context=context, seed=seed, arm=arm, step=step, budget=ep['budget'],
                        objective=float(loss.detach()), task_ce=float(task.detach()),
                        feature_mse=float(raw_mse.detach()), gradient_norm=float(grad.detach())))
                    if step in (30, 60, 90):
                        for r in fitting_probe(model, data, probes):
                            probe_rows.append(dict(context=context, seed=seed, arm=arm, step=step, **r))
                torch.cuda.synchronize()
                seconds = time.perf_counter()-started
                folder = OUT/'models'/context/str(seed)
                folder.mkdir(parents=True, exist_ok=True)
                state = model.numpy_state()
                np.savez_compressed(folder/(arm+'.npz'), **state)
                costs.append(dict(context=context, seed=seed, arm=arm, seconds=seconds,
                    parameters=sum(p.numel() for p in model.parameters()),
                    coef_relative_change=float(np.linalg.norm(state['image_coef']-data['initial']['image_coef'])/
                                               np.linalg.norm(data['initial']['image_coef']))))
                print('TRAINED', context, seed, arm, round(seconds, 2), flush=True)
                del model, optimizer
        del data
        torch.cuda.empty_cache()
    for name, rows in [('training_trace', traces), ('fitting_probe', probe_rows), ('training_cost', costs),
                       ('initial_solver_replay', checks)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False)
    write(OUT/'MODELS_FROZEN.json', dict(models=len(costs), neural_backbone_fits=0,
        affine_corrector_fits=len(costs), target_labels_used=False, held_labels_used=False))


def freeze(family):
    meta = pd.read_csv(BASE/'observations.csv') if family == 'source' else pd.read_csv(SOURCE/'manifest.csv')
    draws = 3 if family == 'source' else 5
    checks = []
    for context in PORTS:
        path = previous.TEACHER/context/'maps.npz' if family == 'source' else BASE/'full_teacher_integration_20261012'/context/'no_training/predictions.npz'
        maps = cached(path)
        ids = maps['obs_ids'] if family == 'source' else maps['rows']
        phi = maps['single'] if family == 'source' else maps['phi']
        prior, index = maps['prior'], pd.Index(ids)
        for seed in SEEDS:
            models = {a: cached(OUT/'models'/context/str(seed)/(a+'.npz')) for a in ARMS}
            dest = OUT/family/context/str(seed)
            dest.mkdir(parents=True, exist_ok=True)
            for draw in range(draws):
                for budget in BUDGETS:
                    old = cached(REFERENCE/family/context/f'{draw}_{budget}.npz')
                    s, q = index.get_indexer(old['support_ids']), index.get_indexer(old['query_ids'])
                    assert (s >= 0).all() and (q >= 0).all()
                    sy = meta.iloc[old['support_ids']].class_id.to_numpy()
                    values = dict(query_ids=old['query_ids'], support_ids=old['support_ids'], base=old['base'],
                        expert_raw=old['expert_raw'], fusion_raw=old['fusion_raw'],
                        expert_initial=old['expert_repair_expert_selected'],
                        fusion_initial=old['fusion_repair_expert_selected'],
                        expert_simple=old['expert_simple_fusion_selected'],
                        fusion_simple=old['fusion_simple_fusion_selected'])
                    for arm, model in models.items():
                        delta = predict(model, phi[s][:, COLS], np.zeros(len(s), int), 'image_history')
                        expert = corrected_score(phi, prior, s, q, sy, delta, DOSES[budget])
                        values['expert_'+arm], values['fusion_'+arm] = expert, late(old['base'], expert)
                    if seed == SEEDS[0]:
                        initial = cached(previous.CONTROL/context/'final_model.npz')
                        delta = predict(initial, phi[s][:, COLS], np.zeros(len(s), int), 'image_history')
                        replay = corrected_score(phi, prior, s, q, sy, delta, DOSES[budget])
                        np.testing.assert_allclose(replay, values['expert_initial'], atol=1e-10, rtol=0)
                        checks.append(dict(context=context, draw=draw, budget=budget,
                                           error=float(abs(replay-values['expert_initial']).max())))
                    np.savez_compressed(dest/f'{draw}_{budget}.npz', **values)
        print('FROZEN', family, context, flush=True)
    write(OUT/(family.upper()+'_PREDICTIONS_FROZEN.json'), dict(seeds=SEEDS,
        context_episode_count=len(PORTS)*len(SEEDS)*draws*len(BUDGETS), initial_replays=checks))


def evaluate(family):
    meta = pd.read_csv(BASE/'observations.csv') if family == 'source' else pd.read_csv(SOURCE/'manifest.csv')
    draws = 3 if family == 'source' else 5
    rows, pairs, classes = [], [], []
    for context in PORTS:
        for seed in SEEDS:
            for draw in range(draws):
                for budget in BUDGETS:
                    z = cached(OUT/family/context/str(seed)/f'{draw}_{budget}.npz')
                    qm = meta.iloc[z['query_ids']]
                    y = qm.class_id.to_numpy()
                    info = dict(context=context, seed=seed, draw=draw, budget=budget)
                    scores = {}
                    for layer in ('expert', 'fusion'):
                        kind = 'logits' if layer == 'expert' else 'probabilities'
                        for arm in ('raw', 'initial', 'simple', *ARMS):
                            p = probability_rows(z[layer+'_'+arm], y, kind)
                            cell = pd.DataFrame(dict(port=qm.port.to_numpy(), y=y, brier=p['brier'], nll=p['nll']))
                            scores[(layer, arm)] = dict(ba=previous.ba(y, z[layer+'_'+arm]),
                                oa=100*p['correct'].mean(), nll=p['nll'].mean(), brier=p['brier'].mean(),
                                portclass_brier=cell.groupby(['port', 'y']).brier.mean().groupby(level=0).mean().mean())
                            rows.append(dict(**info, layer=layer, arm=arm, **scores[(layer, arm)]))
                            for c in np.unique(y):
                                classes.append(dict(**info, layer=layer, arm=arm, class_id=int(c),
                                    recall=100*p['correct'][y == c].mean(), n=int((y == c).sum())))
                        for arm, ref in [('initial', 'raw'), ('simple', 'raw'), ('reconstruction', 'initial'),
                                         ('task', 'raw'), ('task', 'initial'), ('task', 'reconstruction'), ('task', 'simple')]:
                            a, b = scores[(layer, arm)], scores[(layer, ref)]
                            pairs.append(dict(**info, layer=layer, comparison=arm+'_vs_'+ref,
                                delta_ba=a['ba']-b['ba'], delta_oa=a['oa']-b['oa'],
                                nll_gain=b['nll']-a['nll'], brier_gain=b['brier']-a['brier'],
                                portclass_brier_gain=b['portclass_brier']-a['portclass_brier']))
    frame, paired = pd.DataFrame(rows), pd.DataFrame(pairs)
    dest = OUT/(family+'_evaluation')
    dest.mkdir(exist_ok=True)
    frame.to_csv(dest/'metrics.csv', index=False)
    paired.to_csv(dest/'paired.csv', index=False)
    pd.DataFrame(classes).to_csv(dest/'classes.csv', index=False)
    frame.groupby(['budget', 'layer', 'arm'])[['oa', 'ba', 'nll', 'brier', 'portclass_brier']].mean().reset_index().to_csv(dest/'summary.csv', index=False)
    pair_metrics = ['delta_ba', 'delta_oa', 'nll_gain', 'brier_gain', 'portclass_brier_gain']
    paired.groupby(['seed', 'budget', 'layer', 'comparison'])[pair_metrics].mean().reset_index().to_csv(dest/'seed_summary.csv', index=False)
    paired.groupby(['context', 'budget', 'layer', 'comparison'])[pair_metrics].mean().reset_index().to_csv(dest/'context_summary.csv', index=False)
    avg = paired.groupby(['budget', 'layer', 'comparison'])[pair_metrics].mean().reset_index()
    avg.to_csv(dest/'paired_summary.csv', index=False)
    variation = []
    for key, group in paired.groupby(['context', 'seed', 'budget', 'layer', 'comparison']):
        variation.append(dict(zip(['context', 'seed', 'budget', 'layer', 'comparison'], key),
            delta_ba_mean=group.delta_ba.mean(), delta_ba_std=group.delta_ba.std(ddof=0),
            wins=int((group.delta_ba > 1e-10).sum()), ties=int((abs(group.delta_ba) <= 1e-10).sum()),
            losses=int((group.delta_ba < -1e-10).sum())))
    pd.DataFrame(variation).to_csv(dest/'support_variation.csv', index=False)
    print(family.upper(), avg[(avg.layer == 'fusion')].to_string(index=False), flush=True)
    write(dest/'COMPLETE.json', dict(status='complete', primary='Paired BA with OA and probability-risk diagnostics',
        training_episode_seeds=SEEDS, draws=draws,
        context_role='shared-vessel geometry versions' if family == 'source' else 'distinct target ports, shared source',
        historical_development=True, universal_wins_required=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['source', 'target'], default='source')
    args = parser.parse_args()
    if args.stage == 'source':
        train()
    freeze(args.stage)
    evaluate(args.stage)
