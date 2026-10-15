"""Single-observation distillation of source-only repeated-vessel class evidence."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import copy
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import relative_content_condition as previous
from relative_content_condition import (
    BASE, SEEDS, SUPPORTS, FactorNet, BoundedOperator, encode_two,
    standardize, sample_weights, metrics, save_json, residual_head,
    vessel_targets, load_data,
)
from decision_condition_consistency import pair_metrics

OUT = BASE/'shared_evidence_distillation_20261011'
ARMS = ('supervised', 'self_teacher', 'other_acquisition_teacher')
STEPS = 400


def centered(x):
    return x-x.mean(-1, keepdim=True)


class EvidenceStudent(nn.Module):
    def __init__(self, amplitude):
        super().__init__()
        self.register_buffer('amplitude', amplitude.clone())
        self.network = nn.Sequential(nn.Linear(512, 64), nn.GELU(), nn.Linear(64, 8))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x, prior):
        delta = .5*self.amplitude*torch.tanh(self.network(x))
        return prior+centered(delta)


def objective(scores, labels, teacher, valid, arm):
    supervised = F.mse_loss(scores, 2*F.one_hot(labels, 8).float()-1)
    distillation = scores.sum()*0
    if arm != 'supervised' and valid.any():
        distillation = F.mse_loss(centered(scores[valid]), centered(teacher[valid]))
    return supervised+.5*distillation, supervised, distillation


def train_student(x, prior, teacher, valid, labels, meta, fit, roles, arm, seed):
    torch.manual_seed(seed)
    amplitude = centered(prior[fit]).square().mean(0).sqrt()
    model = EvidenceStudent(amplitude)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    probability = sample_weights(meta.iloc[fit].reset_index(drop=True))
    probability /= probability.sum()
    best, best_score, selected_step = copy.deepcopy(model.state_dict()), -np.inf, 0
    trace = []

    def validate(step, losses=None):
        nonlocal best, best_score, selected_step
        row = dict(arm=arm, seed=seed, step=step, **(losses or {}))
        bas = []
        with torch.no_grad():
            for role in ('val_new_ship', 'val_new_time'):
                ids = roles[role]
                scores = model(x[ids], prior[ids])
                result = metrics(scores.argmax(1).numpy(), labels[ids].numpy())
                row.update({role+'_'+k: v for k, v in result.items()})
                bas.append(result['BA'])
        row['selection_BA'] = float(np.mean(bas))
        trace.append(row)
        if row['selection_BA'] > best_score+1e-10:
            best, best_score, selected_step = copy.deepcopy(model.state_dict()), row['selection_BA'], step

    validate(0)
    for step in range(1, STEPS+1):
        rng = np.random.default_rng(seed+step)
        ids = rng.choice(fit, 128, replace=True, p=probability)
        scores = model(x[ids], prior[ids])
        loss, supervised, distillation = objective(scores, labels[ids], teacher[ids], valid[ids], arm)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 20 == 0:
            validate(step, dict(loss=float(loss.detach()), supervised=float(supervised.detach()),
                                distillation=float(distillation.detach())))
    last = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    torch.save(dict(selected=best, last=last), OUT/f'{arm}_{seed}.pt')
    choice = dict(arm=arm, seed=seed, selected_step=selected_step, selection_BA=best_score,
                  parameters=sum(p.numel() for p in model.parameters()))
    print('SELECTED', choice, flush=True)
    return model.eval(), trace, choice


def main(export_only=False):
    from shared_evidence_diagnose import diagnose_scores
    started = time.perf_counter()
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=export_only)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    fit, test = split['factor_fit'], split['test_source']
    labels = torch.tensor(meta.class_id.to_numpy(copy=True))
    xt = torch.from_numpy(norm)
    save_json(OUT/'PROTOCOL.json', dict(
        seeds=SEEDS, arms=ARMS, steps=STEPS, training_rows=len(fit),
        baseline='Frozen historical B factor plus previously selected additive condition operator. Retain changed feature geometry and fixed source classifier.',
        intervention='Single-image 512-64-8 GELU residual on class scores; zero output initialization, centered delta=0.5*source per-class centered-score RMS*tanh. No learned replacement of full512 features or stable128.',
        objectives='All use same source class MSE. self_teacher adds0.5 centered score MSE to the same image frozen prior; other_acquisition_teacher uses equal-acquisition mean frozen prior from other products of the SAME source-fit ship. Both teacher arms use identical multi-acquisition rows. supervised has no teacher loss.',
        training='All three arms same source1296, minibatches, seeds, Adam.001,400 steps. Larger supervised fitting set than historical gate462; supervised control is essential for attributing any benefit.',
        selection='Earliest best mean source new-ship/new-time BA every20steps including0. Source test823 and target query labels never select.',
        deployment='One image only. No vessel identity, target multi-view mean or target metadata. Same original64 support labels, ridge8, fixed baseline feature kernel. New module changes prior only.',
        limitations='Empirical multi-view teacher may be wrong. Task evidence distillation, not causal feature identification. Repeated development3ports x3support draws x2network seeds, not independent validation/full main model.',
        hypothesis='Class evidence from other acquisitions transfers better than same-image self-distillation, without destroying discriminability. Otherwise isolate teacher quality, fitting and transfer.'))
    traces, choices, diagnostics, teacher_rows = [], [], [], []
    all_scores, frozen, students = {}, {}, {}
    for seed in SEEDS:
        factor = FactorNet()
        state = torch.load(previous.old.OUT/f'anchor_{seed}_factor.pt', map_location='cpu', weights_only=True)
        factor.load_state_dict(state['model'])
        factor.eval()
        _, condition = encode_two(factor, norm)
        context = torch.from_numpy(standardize(condition, roles['fit']).astype(np.float32))
        amplitude = (h[roles['fit']]@fixed['basis']).square().mean(0).sqrt()
        operator = BoundedOperator(fixed['basis'], amplitude, 'residual')
        operator.load_state_dict(torch.load(previous.OUT/f'anchor_residual_{seed}_operator.pt', map_location='cpu', weights_only=True))
        with torch.no_grad():
            changed = operator(h, context)
            prior = changed@fixed['weight']+fixed['bias']
        frozen[seed] = (operator, context)
        all_scores['baseline', seed] = prior
        if export_only:
            for arm in ARMS:
                model = EvidenceStudent(centered(prior[fit]).square().mean(0).sqrt())
                model.load_state_dict(torch.load(OUT/f'{arm}_{seed}.pt', map_location='cpu', weights_only=True)['selected'])
                students[arm, seed] = model.eval()
            continue
        teacher = torch.zeros_like(prior)
        valid = torch.zeros(len(meta), dtype=torch.bool)
        _, other, ok = vessel_targets(prior[fit].numpy(), meta.iloc[fit].reset_index(drop=True))
        teacher[fit] = torch.from_numpy(other)
        valid[fit] = torch.from_numpy(ok)
        for role, ids in [('fit', fit), ('test_source', test)]:
            mm = meta.iloc[ids].reset_index(drop=True)
            _, t, v = vessel_targets(prior[ids].numpy(), mm)
            base_score = prior[ids].numpy()[v]
            truth = mm.class_id.to_numpy()[v]
            class_direction = centered(torch.from_numpy(2*np.eye(8)[truth]-1-base_score)).numpy()
            teacher_direction = centered(torch.from_numpy(t[v]-base_score)).numpy()
            norm_product = np.linalg.norm(class_direction, axis=1)*np.linalg.norm(teacher_direction, axis=1)
            dot = (class_direction*teacher_direction).sum(1)
            nonzero = norm_product > 1e-12
            for name, score in [('single', prior[ids].numpy()), ('other_acquisitions', t)]:
                prediction = score[v].argmax(1)
                base_prediction = base_score.argmax(1)
                teacher_rows.append(dict(seed=seed, role=role, mode=name, n=int(v.sum()),
                    **metrics(prediction, truth),
                    rescue=int(((base_prediction != truth)&(prediction == truth)).sum()),
                    harm=int(((base_prediction == truth)&(prediction != truth)).sum()),
                    teacher_label_direction_cosine=float(np.mean(dot[nonzero]/norm_product[nonzero])),
                    teacher_label_direction_conflict_pct=float(100*np.mean(dot[nonzero] < 0))))
        for arm in ARMS:
            model, trace, choice = train_student(xt, prior, prior if arm == 'self_teacher' else teacher,
                valid, labels, meta, fit, roles, arm, seed)
            traces.extend(trace)
            choices.append(choice)
            with torch.no_grad():
                all_scores[arm, seed] = model(xt, prior)
                last = torch.load(OUT/f'{arm}_{seed}.pt', map_location='cpu', weights_only=True)['last']
                model.load_state_dict(last)
                all_scores[arm+'_last', seed] = model(xt, prior)
                model.load_state_dict(torch.load(OUT/f'{arm}_{seed}.pt', map_location='cpu', weights_only=True)['selected'])
                students[arm, seed] = model
        for role, ids in [('fit', fit), ('test_source', test)]:
            mm = meta.iloc[ids].reset_index(drop=True)
            base = prior[ids].numpy()
            _, teacher_diagnostic, ok = vessel_targets(base, mm)
            for arm in ('baseline',)+ARMS+tuple(a+'_last' for a in ARMS):
                score = all_scores[arm, seed][ids].numpy()
                detail = diagnose_scores(score, base, teacher_diagnostic, mm, ok)
                detail.update(pair_metrics(score, mm))
                diagnostics.append(dict(arm=arm, seed=seed, role=role, **detail))
                if role == 'test_source':
                    np.savez_compressed(OUT/f'{arm}_{seed}_source_scores.npz', obs_ids=ids, scores=score)
            if role == 'test_source':
                unseen = ~mm['product'].isin(meta.iloc[fit]['product']).to_numpy()
                if unseen.any():
                    for arm in ('baseline',)+ARMS:
                        score = all_scores[arm, seed][ids].numpy()[unseen]
                        diagnostics.append(dict(arm=arm, seed=seed, role='test_unseen_product', n=int(unseen.sum()),
                            **metrics(score.argmax(1), mm.class_id.to_numpy()[unseen])))
    if not export_only:
        pd.DataFrame(traces).to_csv(OUT/'training.csv', index=False)
        pd.DataFrame(choices).to_csv(OUT/'selection.csv', index=False)
        pd.DataFrame(diagnostics).to_csv(OUT/'source_diagnostics.csv', index=False)
        pd.DataFrame(teacher_rows).to_csv(OUT/'teacher_quality.csv', index=False)
    tasks = []
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task = path.stem.removesuffix('_split')
        tasks.append(task)
        with np.load(path) as z:
            support, query, sy = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
        assert len(support) == 64
        assert not set(meta.iloc[np.r_[support, query]].mmsi)&set(meta.iloc[source].mmsi)
        assert not set(meta.iloc[support].mmsi)&set(meta.iloc[query].mmsi)
        ids = np.r_[support, query]
        archive = dict(support_obs_ids=support, query_obs_ids=query)
        for seed in SEEDS:
            operator, context = frozen[seed]
            # Match the historical task-local forward, including its GEMM shape.
            with torch.no_grad():
                changed = operator(h[ids], context[ids])
                base_prior = changed@fixed['weight']+fixed['bias']
            phi = torch.cat([changed, torch.ones((len(ids), 1))], 1)
            for arm in ('baseline',)+ARMS:
                with torch.no_grad():
                    prior = base_prior if arm == 'baseline' else students[arm, seed](xt[ids], base_prior)
                beta, _, _ = residual_head(phi[:64], prior[:64], torch.tensor(sy), balanced=True)
                score = (prior[64:]+phi[64:]@beta).numpy()
                archive[f'{arm}__{seed}'] = score
                if arm == 'baseline':
                    with np.load(previous.OUT/f'{task}_predictions.npz') as z:
                        np.testing.assert_array_equal(score, z[f'anchor__residual__both__{seed}'])
        np.savez_compressed(OUT/f'{task}_predictions.npz', **archive)
    save_json(OUT/'PREDICTIONS_SAVED.json', dict(tasks=len(tasks), target_query_metrics_read=False))
    rows, classes = [], []
    for task in tasks:
        with np.load(OUT/f'{task}_predictions.npz') as z:
            query = z['query_obs_ids']
            qm = meta.iloc[query].reset_index(drop=True)
            y = qm.class_id.to_numpy()
            for seed in SEEDS:
                base = z[f'baseline__{seed}'].argmax(1)
                for arm in ('baseline',)+ARMS:
                    scores = z[f'{arm}__{seed}']
                    pred = scores.argmax(1)
                    desc = dict(arm=arm, seed=seed, task=task, port=qm.port.iloc[0])
                    rows.append(dict(**desc, **metrics(pred, y), rescue=int(((base != y)&(pred == y)).sum()),
                        harm=int(((base == y)&(pred != y)).sum()), **pair_metrics(scores, qm)))
                    for cls in np.unique(y):
                        classes.append(dict(**desc, class_id=int(cls), recall=float(100*(pred[y == cls] == cls).mean())))
    data = pd.DataFrame(rows)
    data.to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'target_classes.csv', index=False)
    for name, group in [('summary', ['arm']), ('seed_summary', ['arm', 'seed']), ('port_summary', ['arm', 'port'])]:
        data.groupby(group).mean(numeric_only=True).reset_index().to_csv(OUT/f'target_{name}.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(final_invocation_seconds=time.perf_counter()-started, total_student_fits=6,
        export_from_saved_checkpoints=export_only,
        baseline_exact_replays=len(tasks)*len(SEEDS), target_tasks=len(tasks), main_model_modified=False))
    print(data.groupby('arm')[['OA', 'BA', 'rescue', 'harm']].mean().to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-only', action='store_true', help='Export existing selected checkpoints without retraining.')
    main(parser.parse_args().export_only)
