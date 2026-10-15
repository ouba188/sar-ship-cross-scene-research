"""Isolate support budgets while preserving every original task episode."""

import run_support_task_loss as old
from full_teacher_data import BASE, SOURCE, PORTS, write
from case_support_completion import COLS, predict
from run_support_budget_strength import corrected_score, BUDGETS
from support_task_loss import ResidualCorrector
from unified_head_core import late
from evaluate_probability_layers import cached, probability_rows
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch


OUT = BASE/'support_budget_isolation_20261012'
ARMS = ('raw', 'initial', 'simple', 'reconstruction', 'task', 'isolated')


def matched_episodes(meta, folds):
    saved = pd.read_csv(old.OUT/'training_episodes.csv')
    definitions = {}
    rows = []
    for seed in old.SEEDS:
        episodes = old.training_episodes(meta, folds['fit_obs_ids'], folds['assignment'], seed)
        original = saved[saved.seed == seed].sort_values('step')
        assert len(original) == len(episodes) == 90
        for step, (ep, row) in enumerate(zip(episodes, original.itertuples())):
            assert ep['budget'] == row.budget and ep['fold'] == row.fold
            np.testing.assert_array_equal(folds['fit_obs_ids'][ep['s']], json.loads(row.support_obs_ids))
            np.testing.assert_array_equal(folds['fit_obs_ids'][ep['q']], json.loads(row.query_obs_ids))
            rows.append(dict(seed=seed, shared_step=step+1, budget=ep['budget'], fold=ep['fold'],
                             support_obs_ids=row.support_obs_ids, query_obs_ids=row.query_obs_ids))
        definitions[seed] = episodes
    frame = pd.DataFrame(rows)
    frame['isolated_step'] = frame.groupby(['seed', 'budget']).cumcount()+1
    assert (frame.groupby(['seed', 'budget']).size() == 30).all()
    frame.to_csv(OUT/'matched_episodes.csv', index=False)
    return definitions


def train():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda'
    meta = pd.read_csv(BASE/'observations.csv')
    folds = cached(BASE/'crossfit_evidence_20261011/source_folds.npz')
    definitions = matched_episodes(meta, folds)
    write(OUT/'FROZEN_PROTOCOL.json', dict(seeds=old.SEEDS, budgets=BUDGETS,
        source_rows=1296, steps_per_budget=30, total_steps_per_context_seed=90,
        learning_rate=old.LR, weight_decay=1e-4, optimizer='AdamW', doses=old.DOSES,
        initialization='Same original class-common corrector for every isolated budget',
        episode_match='Exact old source observation IDs and within-budget order',
        changed='Budget sharing removed; parameters and Adam moments now separate',
        checkpoint='Last30 within-budget steps; no checkpoint selection',
        query_backbone_changes=False, target_labels_used=False,
        scope='Historical development; supports use existing identity protocol',
        acceptance='Repeatable task-scoped net benefit; universal wins not required'))
    traces, probes, costs = [], [], []
    for context in PORTS:
        data = old.prepare_context(context, meta, folds, device)
        probe_defs = old.training_episodes(meta, folds['fit_obs_ids'], folds['assignment'], 125012, rounds=1)
        fixed_probe = old.attach_base(data, probe_defs, device)
        for seed in old.SEEDS:
            episodes = old.attach_base(data, definitions[seed], device)
            for budget in BUDGETS:
                selected = [(i+1, ep) for i, ep in enumerate(episodes) if ep['budget'] == budget]
                budget_probe = [ep for ep in fixed_probe if ep['budget'] == budget]
                torch.manual_seed(seed)
                model = ResidualCorrector(data['initial']).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=old.LR, weight_decay=1e-4)
                for r in old.fitting_probe(model, data, budget_probe):
                    probes.append(dict(context=context, seed=seed, step=0, **r))
                torch.cuda.synchronize()
                start = time.perf_counter()
                for step, (shared_step, ep) in enumerate(selected, 1):
                    _, loss, mse, _ = old.losses(model, data, ep)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = sum(p.grad.square().sum() for p in model.parameters()).sqrt()
                    if not torch.isfinite(loss) or not torch.isfinite(grad) or float(grad) == 0:
                        raise ValueError('Nonfinite loss or invalid gradient')
                    optimizer.step()
                    traces.append(dict(context=context, seed=seed, budget=budget, step=step,
                        shared_step=shared_step, task_ce=float(loss.detach()), feature_mse=float(mse.detach()),
                        gradient_norm=float(grad.detach())))
                    if step in (10, 20, 30):
                        for r in old.fitting_probe(model, data, budget_probe):
                            probes.append(dict(context=context, seed=seed, step=step, **r))
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                folder = OUT/'models'/context/str(seed)
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder/f'{budget}.npz', **model.numpy_state())
                costs.append(dict(context=context, seed=seed, budget=budget, seconds=elapsed,
                                  parameters=sum(p.numel() for p in model.parameters()), steps=len(selected)))
                print('TRAINED', context, seed, budget, round(elapsed, 3), flush=True)
                del model, optimizer
        del data
        torch.cuda.empty_cache()
    for name, records in [('training_trace', traces), ('fitting_probe', probes), ('training_cost', costs)]:
        pd.DataFrame(records).to_csv(OUT/(name+'.csv'), index=False)
    write(OUT/'MODELS_FROZEN.json', dict(status='complete', isolated_fits=len(costs),
        total_training_updates=sum(r['steps'] for r in costs), backbone_fits=0,
        held_or_target_label_selection=False))


def freeze(family):
    meta = pd.read_csv(BASE/'observations.csv' if family == 'source' else SOURCE/'manifest.csv',
                       usecols=['class_id', 'port'])
    for context in PORTS:
        path = old.previous.TEACHER/context/'maps.npz' if family == 'source' else BASE/'full_teacher_integration_20261012'/context/'no_training/predictions.npz'
        maps = cached(path)
        ids = maps['obs_ids'] if family == 'source' else maps['rows']
        phi = maps['single'] if family == 'source' else maps['phi']
        index = pd.Index(ids)
        for seed in old.SEEDS:
            models = {b: cached(OUT/'models'/context/str(seed)/f'{b}.npz') for b in BUDGETS}
            dest = OUT/family/context/str(seed)
            dest.mkdir(parents=True, exist_ok=True)
            for draw in range(3 if family == 'source' else 5):
                for budget in BUDGETS:
                    values = dict(cached(old.OUT/family/context/str(seed)/f'{draw}_{budget}.npz'))
                    s, q = index.get_indexer(values['support_ids']), index.get_indexer(values['query_ids'])
                    assert (s >= 0).all() and (q >= 0).all()
                    delta = predict(models[budget], phi[s][:, COLS], np.zeros(len(s), int), 'image_history')
                    e = corrected_score(phi, maps['prior'], s, q,
                        meta.iloc[values['support_ids']].class_id.to_numpy(), delta, old.DOSES[budget])
                    values['expert_isolated'] = e
                    values['fusion_isolated'] = late(values['base'], e)
                    np.savez_compressed(dest/f'{draw}_{budget}.npz', **values)
        print('FROZEN', family, context, flush=True)
    write(OUT/(family.upper()+'_PREDICTIONS_FROZEN.json'), dict(status='complete',
        frozen_models=18, support_and_query_ids='Copied unchanged from previous matched pilot'))


def evaluate(family):
    meta = pd.read_csv(BASE/'observations.csv' if family == 'source' else SOURCE/'manifest.csv',
                       usecols=['class_id', 'port'])
    rows, pairs, cells, replays = [], [], [], []
    for context in PORTS:
        for seed in old.SEEDS:
            for draw in range(3 if family == 'source' else 5):
                for budget in BUDGETS:
                    z = cached(OUT/family/context/str(seed)/f'{draw}_{budget}.npz')
                    qm = meta.iloc[z['query_ids']]
                    y = qm.class_id.to_numpy()
                    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
                    weights = 1/(len(counts)*counts[inverse])
                    info = dict(context=context, seed=seed, draw=draw, budget=budget)
                    for layer in ('expert', 'fusion'):
                        scores, probabilities = {}, {}
                        for arm in ARMS:
                            p = probability_rows(z[layer+'_'+arm], y, 'logits' if layer == 'expert' else 'probabilities')
                            probabilities[arm] = p
                            cell = pd.DataFrame(dict(port=qm.port.to_numpy(), y=y, brier=p['brier']))
                            scores[arm] = dict(oa=100*p['correct'].mean(), ba=old.previous.ba(y, z[layer+'_'+arm]),
                                nll=p['nll'].mean(), brier=p['brier'].mean(),
                                class_nll=weights@p['nll'], class_brier=weights@p['brier'],
                                portclass_brier=cell.groupby(['port', 'y']).brier.mean().groupby(level=0).mean().mean())
                            rows.append(dict(**info, layer=layer, arm=arm, **scores[arm]))
                        a = scores['isolated']
                        ap = probabilities['isolated']
                        for ref in ARMS[:-1]:
                            b, bp = scores[ref], probabilities[ref]
                            rescue = 100*weights@(~bp['correct'] & ap['correct'])
                            harm = 100*weights@(bp['correct'] & ~ap['correct'])
                            np.testing.assert_allclose(rescue-harm, a['ba']-b['ba'], atol=1e-10, rtol=0)
                            pairs.append(dict(**info, layer=layer, reference=ref,
                                delta_ba=a['ba']-b['ba'], delta_oa=a['oa']-b['oa'], rescue_ba=rescue, harm_ba=harm,
                                nll_gain=b['nll']-a['nll'], brier_gain=b['brier']-a['brier'],
                                class_nll_gain=b['class_nll']-a['class_nll'],
                                class_brier_gain=b['class_brier']-a['class_brier'],
                                portclass_brier_gain=b['portclass_brier']-a['portclass_brier']))
                            frame = pd.DataFrame(dict(port=qm.port.to_numpy(), y=y,
                                recall=100*(ap['correct'].astype(float)-bp['correct']),
                                nll=bp['nll']-ap['nll'], brier=bp['brier']-ap['brier']))
                            for (port, c), group in frame.groupby(['port', 'y']):
                                cells.append(dict(**info, layer=layer, reference=ref, actual_port=port,
                                    class_id=c, n=len(group), delta_recall=group.recall.mean(),
                                    nll_gain=group.nll.mean(), brier_gain=group.brier.mean()))
    frame, paired = pd.DataFrame(rows), pd.DataFrame(pairs)
    previous = pd.read_csv(old.OUT/(family+'_evaluation')/'metrics.csv')
    keys = ['context', 'seed', 'draw', 'budget', 'layer', 'arm']
    replay = frame[frame.arm != 'isolated'].merge(previous, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    for metric in ('oa', 'ba', 'nll', 'brier', 'portclass_brier'):
        error = float(abs(replay[metric+'_new']-replay[metric+'_old']).max())
        assert error < 1e-10, (metric, error)
        replays.append(dict(metric=metric, max_error=error))
    dest = OUT/(family+'_evaluation')
    dest.mkdir(exist_ok=True)
    frame.to_csv(dest/'metrics.csv', index=False)
    paired.to_csv(dest/'paired.csv', index=False)
    pd.DataFrame(cells).to_csv(dest/'port_classes.csv', index=False)
    metrics = ['oa', 'ba', 'nll', 'brier', 'class_nll', 'class_brier', 'portclass_brier']
    frame.groupby(['budget', 'layer', 'arm'])[metrics].mean().reset_index().to_csv(dest/'summary.csv', index=False)
    changes = ['delta_ba', 'delta_oa', 'rescue_ba', 'harm_ba', 'nll_gain', 'brier_gain',
               'class_nll_gain', 'class_brier_gain', 'portclass_brier_gain']
    for name, groups in [('paired_summary', ['budget', 'layer', 'reference']),
                         ('seed_summary', ['seed', 'budget', 'layer', 'reference']),
                         ('context_summary', ['context', 'budget', 'layer', 'reference'])]:
        paired.groupby(groups)[changes].mean().reset_index().to_csv(dest/(name+'.csv'), index=False)
    summary = paired.groupby(['budget', 'layer', 'reference'])[changes].mean().reset_index()
    print(family.upper(), summary[summary.layer == 'fusion'].to_string(index=False), flush=True)
    write(dest/'COMPLETE.json', dict(status='complete', prediction_units=len(frame)//(len(ARMS)*2),
        original_metric_replays=replays, error_flow_verified=True,
        geometry_versions_not_independent=family == 'source', historical_development=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['source', 'target'], default='source')
    args = parser.parse_args()
    if args.stage == 'source':
        train()
    freeze(args.stage)
    evaluate(args.stage)
