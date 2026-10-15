"""Bounded sourceFIT gradient and actual AdamW interference sidecar."""

import sys

sys.dont_write_bytecode = True

import copy
import json
import time

import run_support_task_loss as original
import numpy as np
import pandas as pd
import torch


OUT = original.BASE/'support_budget_isolation_20261012'/'diagnostics'
CHECKPOINTS = (0, 30, 90)
PROBE_SEED = 125012
WEIGHT_DECAY = 1e-4


def state_error(before, after):
    """Compare snapshots directly, including optimizer moments and step counters."""
    if isinstance(before, torch.Tensor):
        assert before.shape == after.shape and before.dtype == after.dtype
        return float((before-after).abs().max()) if before.numel() else 0.
    if isinstance(before, dict):
        assert before.keys() == after.keys()
        return max((state_error(before[k], after[k]) for k in before), default=0.)
    if isinstance(before, (list, tuple)):
        assert len(before) == len(after)
        return max((state_error(a, b) for a, b in zip(before, after)), default=0.)
    assert before == after
    return 0.


def diagnose(model, optimizer, data, probes, info, rows):
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    grads_before = copy.deepcopy([p.grad for p in model.parameters()])
    before_ce, gradients = [], []
    for ep in probes:
        ce = original.losses(model, data, ep)[1]
        gradient = torch.cat([g.detach().reshape(-1) for g in
                              torch.autograd.grad(ce, tuple(model.parameters()))])
        assert torch.isfinite(gradient).all()
        before_ce.append(float(ce.detach()))
        gradients.append(gradient)
        rows['gradient_norms'].append(dict(**info, fold=ep['fold'], budget=ep['budget'],
            task_ce=before_ce[-1], gradient_norm=float(gradient.norm())))

    for i, driver in enumerate(probes):
        branch = copy.deepcopy(model)
        branch_optimizer = torch.optim.AdamW(branch.parameters(), lr=original.LR,
                                              weight_decay=WEIGHT_DECAY)
        # Loading a fresh deep copy prevents optimizer tensors being shared by branches.
        branch_optimizer.load_state_dict(copy.deepcopy(optimizer_before))
        assert state_error(optimizer_before, branch_optimizer.state_dict()) == 0.
        branch_optimizer.zero_grad(set_to_none=True)
        driver_ce = original.losses(branch, data, driver)[1]
        driver_ce.backward()
        branch_optimizer.step()
        update = torch.cat([(p.detach()-model_before[name]).reshape(-1)
                            for name, p in branch.named_parameters()])
        update_norm = float(update.norm())
        rows['parameter_updates'].append(dict(**info, driver_fold=driver['fold'],
            driver_budget=driver['budget'], update_norm=update_norm,
            optimizer_steps_before=json.dumps([float(s['step']) for s in
                                              optimizer_before['state'].values()]),
            optimizer_steps_after=json.dumps([float(s['step']) for s in
                                             branch_optimizer.state_dict()['state'].values()])))
        with torch.no_grad():
            for j, probe in enumerate(probes):
                after_ce = float(original.losses(branch, data, probe)[1])
                pair = dict(**info, driver_fold=driver['fold'], driver_budget=driver['budget'],
                    probe_fold=probe['fold'], probe_budget=probe['budget'],
                    fold_relation='same_fold' if driver['fold'] == probe['fold'] else 'different_fold',
                    budget_relation='same_budget' if driver['budget'] == probe['budget'] else 'cross_budget')
                denom = gradients[i].norm()*gradients[j].norm()
                cosine = float(torch.dot(gradients[i], gradients[j])/denom) if float(denom) else float('nan')
                rows['gradient_cosines'].append(dict(**pair, cosine=cosine))
                rows['actual_ce_effects'].append(dict(**pair, ce_before=before_ce[j],
                    ce_after=after_ce, ce_gain=before_ce[j]-after_ce, update_norm=update_norm))
        del branch, branch_optimizer

    errors = dict(model_max_error=state_error(model_before, model.state_dict()),
        optimizer_max_error=state_error(optimizer_before, optimizer.state_dict()),
        existing_grad_max_error=state_error(grads_before, [p.grad for p in model.parameters()]))
    rows['branch_isolation'].append(dict(**info, **errors))
    assert max(errors.values()) == 0., 'Diagnostic branch mutated the replay state'


def grouped(frame, keys, value):
    return frame.groupby(keys)[value].agg(mean='mean', minimum='min', maximum='max',
        count='size', positive=lambda s: int((s > 0).sum()),
        negative=lambda s: int((s < 0).sum()), zero=lambda s: int((s == 0).sum())).reset_index()


def finish(rows, started):
    frames = {name: pd.DataFrame(values) for name, values in rows.items()}
    for name, frame in frames.items():
        frame.to_csv(OUT/(name+'.csv'), index=False)
    pair_keys = ['step', 'fold_relation', 'driver_budget', 'probe_budget']
    summaries = {
        'gradient_norm_summary': grouped(frames['gradient_norms'], ['step', 'budget'], 'gradient_norm'),
        'cosine_summary': grouped(frames['gradient_cosines'], pair_keys, 'cosine'),
        'actual_ce_summary': grouped(frames['actual_ce_effects'], pair_keys, 'ce_gain'),
        'update_norm_summary': grouped(frames['parameter_updates'], ['step', 'driver_budget'], 'update_norm'),
    }
    for name, frame in summaries.items():
        frame.to_csv(OUT/(name+'.csv'), index=False)

    cosine = summaries['cosine_summary']
    cosine = cosine[(cosine.step.isin([0, 90])) & (cosine.driver_budget < cosine.probe_budget)]
    effects = summaries['actual_ce_summary']
    effects = effects[(effects.step.isin([0, 90])) & (effects.driver_budget != effects.probe_budget)]
    matrix = effects.pivot(index=['step', 'fold_relation', 'driver_budget'],
                           columns='probe_budget', values='mean')
    replay = frames['replay_checks']
    isolation = frames['branch_isolation']
    report = [
        '# Bounded sourceFIT budget interference diagnosis', '',
        'Exact original mixed-budget task replay; no new method or model selection. '
        'Three geometry contexts share source vessels; they are not independent target ports.', '',
        'All probes use the original 1296-row source fitting pool and fixed seed 125012, rounds=1. '
        'They are NOT independent heldout data. Within a fold, B16/64/128 supports are nested '
        'and share the same 32 query vessels, disjoint from support vessels.', '',
        'Each driver starts from the checkpoint model AND a deep copy of its actual AdamW state. '
        'The nine driver steps are independent branches. Positive CE gain means before-minus-after '
        'improvement; negative means harm. Gradient norms alone do not prove budget dominance. '
        'Late effects include the inherited mixed-budget optimizer moments and weight decay.', '',
        'Means are equally weighted over contexts, seeds and applicable fold pairs. Initial '
        'seeds share initialization and probes, so these duplicated rows are not independent replicates. '
        'Raw CSVs retain every direction and sign; summaries include minima, maxima and sign counts. '
        'Task-scoped benefit does not require universal wins, and these one-step effects do not '
        'establish heldout, target or full-training benefit.', '',
        '## Initial / late pairwise gradient cosines', '', '```text',
        cosine.to_string(index=False, float_format=lambda x: f'{x:.7g}'), '```', '',
        '## Actual cross-budget CE gain (rows: driver; columns: probe)', '', '```text',
        matrix.to_string(float_format=lambda x: f'{x:+.7g}', na_rep='same-budget'), '```', '',
        'Same-budget effects and step 30 remain in the complete CSVs.', '',
        '## Replay and isolation', '', '```text', replay.to_string(index=False), '```', '',
        f"Maximum branch model/optimizer/gradient mutation: {isolation[['model_max_error', 'optimizer_max_error', 'existing_grad_max_error']].to_numpy().max():.17g}",
        f"Maximum final archive replay error: {replay.max_error.max():.17g} (atol=1e-10, rtol=0).", '',
    ]
    (OUT/'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    original.write(OUT/'COMPLETE.json', dict(status='complete', seconds=time.perf_counter()-started,
        row_counts={name: len(frame) for name, frame in frames.items()},
        replay_max_error=float(replay.max_error.max()),
        branch_state_max_error=float(isolation[['model_max_error', 'optimizer_max_error',
                                               'existing_grad_max_error']].to_numpy().max()),
        archive_atol=1e-10, archive_rtol=0, sourcefit_not_independent_heldout=True,
        universal_wins_required=False, target_evaluation=False, model_selection=False))
    print('\nINITIAL/LATE COSINES\n'+cosine.to_string(index=False), flush=True)
    print('\nCROSS-BUDGET ACTUAL CE GAINS\n'+matrix.to_string(), flush=True)
    print('\nREPLAY\n'+replay.to_string(index=False), flush=True)
    print('COMPLETE', str(OUT), flush=True)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert original.SEEDS == (31415, 27182) and len(original.PORTS) == 3
    assert original.STEPS == 90 and original.LR == 1e-4
    assert set(original.BUDGETS) == {16, 64, 128}
    meta = pd.read_csv(original.BASE/'observations.csv')
    folds = original.cached(original.BASE/'crossfit_evidence_20261011/source_folds.npz')
    assert len(folds['fit_obs_ids']) == 1296
    definitions = {seed: original.training_episodes(meta, folds['fit_obs_ids'], folds['assignment'], seed)
                   for seed in original.SEEDS}
    probe_defs = original.training_episodes(meta, folds['fit_obs_ids'], folds['assignment'],
                                             PROBE_SEED, rounds=1)
    assert len(probe_defs) == 9
    for fold in range(3):
        nested = {ep['budget']: ep for ep in probe_defs if ep['fold'] == fold}
        for budget in (16, 64, 128):
            np.testing.assert_array_equal(nested[budget]['q'], nested[128]['q'])
            np.testing.assert_array_equal(nested[budget]['s'], nested[128]['s'][:budget])
    pd.DataFrame([dict(fold=ep['fold'], budget=ep['budget'],
        support_obs_ids=json.dumps(folds['fit_obs_ids'][ep['s']].tolist()),
        query_obs_ids=json.dumps(folds['fit_obs_ids'][ep['q']].tolist()))
        for ep in probe_defs]).to_csv(OUT/'probe_episodes.csv', index=False)
    original.write(OUT/'DIAGNOSTIC_PROTOCOL.json', dict(source_fitting_rows=1296,
        seeds=original.SEEDS, contexts=original.PORTS, checkpoints=CHECKPOINTS,
        budgets=[16, 64, 128], replay_steps=90, probe_seed=PROBE_SEED, probe_rounds=1,
        optimizer='AdamW', learning_rate=original.LR, weight_decay=WEIGHT_DECAY,
        model='ResidualCorrector', objective='original class-balanced fusion CE',
        original_doses=original.DOSES, branch_drivers=9, probes_per_driver=9,
        original_archive=str(original.OUT/'models'), sourcefit_not_independent_heldout=True,
        state_isolation='deepcopy model and optimizer state for each independent one-step branch',
        ce_sign='before minus after: positive benefit, negative harm',
        context_role='shared-source geometry contexts, not target evaluation',
        model_selection=False, universal_wins_required=False))
    rows = {name: [] for name in ('gradient_norms', 'gradient_cosines', 'actual_ce_effects',
            'parameter_updates', 'branch_isolation', 'replay_checks', 'replay_trace')}
    for context in original.PORTS:
        data = original.prepare_context(context, meta, folds, device)
        probes = original.attach_base(data, probe_defs, device)
        for seed in original.SEEDS:
            episodes = original.attach_base(data, definitions[seed], device)
            assert len(episodes) == 90
            torch.manual_seed(seed)
            model = original.ResidualCorrector(data['initial']).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=original.LR, weight_decay=WEIGHT_DECAY)
            diagnose(model, optimizer, data, probes, dict(context=context, seed=seed, step=0), rows)
            for step, ep in enumerate(episodes, 1):
                task = original.losses(model, data, ep)[1]
                assert torch.isfinite(task)
                optimizer.zero_grad(set_to_none=True)
                task.backward()
                grad_norm = sum(p.grad.square().sum() for p in model.parameters()).sqrt()
                assert torch.isfinite(grad_norm) and float(grad_norm) > 0
                optimizer.step()
                rows['replay_trace'].append(dict(context=context, seed=seed, step=step,
                    fold=ep['fold'], budget=ep['budget'], task_ce=float(task.detach()),
                    gradient_norm=float(grad_norm.detach())))
                if step in CHECKPOINTS:
                    diagnose(model, optimizer, data, probes, dict(context=context, seed=seed, step=step), rows)
                    print('DIAGNOSED', context, seed, step, flush=True)
            archived = original.cached(original.OUT/'models'/context/str(seed)/'task.npz')
            errors = {}
            for key, actual in model.numpy_state().items():
                errors[key] = float(np.abs(actual-archived[key]).max())
                np.testing.assert_allclose(actual, archived[key], atol=1e-10, rtol=0)
            rows['replay_checks'].append(dict(context=context, seed=seed,
                max_error=max(errors.values()), **{key+'_max_error': value for key, value in errors.items()}))
            print('REPLAY_MATCH', context, seed, max(errors.values()), flush=True)
            del model, optimizer, episodes
        del data, probes
        torch.cuda.empty_cache()
    finish(rows, started)


if __name__ == '__main__':
    main()
