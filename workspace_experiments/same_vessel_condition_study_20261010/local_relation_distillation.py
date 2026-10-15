"""Bounded positive relation teaching with source-only matched donor controls."""

import copy
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from shared_evidence_distillation import (
    BASE, OUT as PREVIOUS, SEEDS, SUPPORTS, EvidenceStudent, centered, previous,
    FactorNet, BoundedOperator, encode_two, standardize, load_data, sample_weights,
    vessel_targets, residual_head, metrics, save_json,
)
from shared_evidence_diagnose import diagnose_scores
from decision_condition_consistency import pair_metrics

FIRST = BASE/'local_relation_distillation_20261011'
OUT = FIRST/'angle_matched'
ARMS = ('full_matched', 'local_matched', 'other_matched', 'local_all')
REFERENCES = dict(full_all='other_acquisition_teacher', supervised='supervised', self_teacher='self_teacher')


def positive_relations(teacher, labels):
    margin = teacher.gather(1, labels[:, None])-teacher
    mask = margin > 0
    mask.scatter_(1, labels[:, None], False)
    return margin.clamp(max=2.), mask


def auxiliary_rows(scores, teacher, labels, mode):
    if mode == 'full':
        return (centered(scores)-centered(teacher)).square().mean(1)
    target, mask = positive_relations(teacher, labels)
    margins = scores.gather(1, labels[:, None])-scores
    errors = F.relu(target-margins).square()*mask
    return errors.sum(1)/mask.sum(1).clamp_min(1)


def initial_gradient_scale(prior, teacher, labels, valid, probability, mode):
    weights = torch.tensor(probability[valid], dtype=prior.dtype)
    weights /= weights.sum()
    norms = {}
    for name in ('full', mode):
        score = prior[valid].detach().clone().requires_grad_(True)
        losses = auxiliary_rows(score, teacher[valid], labels[valid], name)
        gradient = torch.autograd.grad(losses.sum(), score)[0]
        norms[name] = float((weights*gradient.square().sum(1)).sum().sqrt())
    assert norms[mode] > 0
    return norms['full']/norms[mode], norms


def matched_means(prior, matches):
    same, other = np.zeros_like(prior), np.zeros_like(prior)
    valid = np.zeros(len(prior), dtype=bool)
    for anchor, group in matches.groupby('anchor'):
        same[anchor] = np.mean([prior[list(ids)].mean(0) for ids in group.same_indices], axis=0)
        other[anchor] = np.mean([prior[list(ids)].mean(0) for ids in group.other_indices], axis=0)
        valid[anchor] = True
    return torch.from_numpy(same), torch.from_numpy(other), torch.from_numpy(valid)


def train_one(x, prior, teacher, valid, labels, meta, fit, roles, arm, seed, coefficient):
    torch.manual_seed(seed)
    model = EvidenceStudent(centered(prior[fit]).square().mean(0).sqrt())
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    probability = sample_weights(meta.iloc[fit].reset_index(drop=True))
    probability /= probability.sum()
    mode = 'full' if arm == 'full_matched' else 'local'
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
    for step in range(1, 401):
        rng = np.random.default_rng(seed+step)
        ids = rng.choice(fit, 128, replace=True, p=probability)
        score = model(x[ids], prior[ids])
        supervised = F.mse_loss(score, 2*F.one_hot(labels[ids], 8).float()-1)
        rows = auxiliary_rows(score, teacher[ids], labels[ids], mode)
        auxiliary = rows[valid[ids]].mean() if valid[ids].any() else score.sum()*0
        loss = supervised+.5*coefficient*auxiliary
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 20 == 0:
            validate(step, dict(loss=float(loss.detach()), supervised=float(supervised.detach()),
                auxiliary=float(auxiliary.detach())))
    last = copy.deepcopy(model.state_dict())
    torch.save(dict(selected=best, last=last), OUT/f'{arm}_{seed}.pt')
    model.load_state_dict(best)
    selection = dict(arm=arm, seed=seed, selected_step=selected_step, selection_BA=best_score,
        auxiliary_coefficient=coefficient, parameters=sum(p.numel() for p in model.parameters()))
    print('SELECTED', selection, flush=True)
    return model.eval(), trace, selection


def main():
    from relation_teacher_matching import build_matches, summarize_matches
    torch.set_num_threads(1)
    started = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    fit, test = split['factor_fit'], split['test_source']
    fm = meta.iloc[fit].reset_index(drop=True)
    matches = build_matches(fm)
    matches.to_pickle(OUT/'source_matches.pkl')
    match_audit = summarize_matches(fm, matches)
    save_json(OUT/'MATCHING.json', match_audit)
    print('MATCHING', match_audit, flush=True)
    labels = torch.tensor(meta.class_id.to_numpy(copy=True))
    xt = torch.from_numpy(norm)
    save_json(OUT/'PROTOCOL.json', dict(seeds=SEEDS, arms=ARMS, steps=400, cap=2.,
        purpose='Test task-anchored selective teaching before modifying a content encoder. Does same-vessel correspondence add to matched same-class information?',
        matching='Only source-fit data. Prefer same teacher-product/same-port different-vessel same-class, now exclude known angle gaps>2deg even within a scene; otherwise same-port/platform/pass observed and incidence within2deg. Exact-scene unknown angles remain unknown. Exclude anchor vessel and anchor whole product. Deterministic metadata-only matching, no scores for donor selection.',
        revision='Initial complete run retained in parent directory. Match audit exposed same-scene known incidence gaps up to13.573deg. This repair was decided from metadata before reading its target summary. local_all does not depend on matching and is reused exactly;6 matched-arm fits are new.',
        controls='full_matched/local_matched/other_matched share matched source anchor cohort, paired donor counts, all1296 supervised training rows, seeds/batches/capacity. local_all keeps original full same-vessel teaching cohort. Original full_all/supervised/self_teacher reused separately.',
        positive_support='Matched anchor eligibility intersects positive-relation availability for BOTH teacher types and BOTH seeds. Each teacher recomputes its own positive edges; local loss averages per positive edge within image, then eligible images. Not a shared same-vessel edge mask.',
        auxiliary='Full:centered8score MSE. Local: true-vs-each-rival positive teacher margins capped2 (one-vs-all target margin), squared one-sided shortfall; do not pull a larger student margin down. No handpicked class pairs.',
        scale='Source-only initial per-image logit-gradient RMS normalized to same-vessel fullMSE on same cohort; .5 outer coefficient. Matches initial magnitude only, not full optimization dynamics.',
        model='Original frozen512 features and B additive condition correction; unchanged feature kernel, same33352param512-64-8 zero-init bounded residual prior. New module not a purified content encoder.',
        selection='Source new-ship299/new-time149 meanBA every20steps earliest best including0; source-test823 and target queries not used.',
        deployment='Single image; original64 target labels and balancedridge8,3ports x3draws. No extra target metadata/identity/multi-view access.',
        boundary='Reused development data; conditional matching does not equal randomized intervention; sample-level matching cannot eliminate unmeasured confounding.'))
    frozen, priors, teachers, students = {}, {}, {}, {}
    common_valid = torch.ones(len(fit), dtype=torch.bool)
    for seed in SEEDS:
        factor = FactorNet()
        factor.load_state_dict(torch.load(previous.old.OUT/f'anchor_{seed}_factor.pt', map_location='cpu', weights_only=True)['model'])
        factor.eval()
        _, condition = encode_two(factor, norm)
        context = torch.from_numpy(standardize(condition, roles['fit']).astype(np.float32))
        operator = BoundedOperator(fixed['basis'], (h[roles['fit']]@fixed['basis']).square().mean(0).sqrt(), 'residual')
        operator.load_state_dict(torch.load(previous.OUT/f'anchor_residual_{seed}_operator.pt', map_location='cpu', weights_only=True))
        with torch.no_grad():
            prior = operator(h, context)@fixed['weight']+fixed['bias']
        frozen[seed], priors[seed] = (operator, context), prior
        same, other, valid = matched_means(prior[fit].numpy(), matches)
        _, all_t, all_valid = vessel_targets(prior[fit].numpy(), fm)
        teachers[seed] = dict(same=same, other=other, all=torch.from_numpy(all_t),
                              all_valid=torch.from_numpy(all_valid))
        common_valid &= valid
        for values in (same, other):
            common_valid &= positive_relations(values, labels[fit])[1].any(1)
    save_json(OUT/'ELIGIBILITY.json', dict(matched_anchors=int(common_valid.sum()),
        ships=int(fm.loc[common_valid.numpy(), 'mmsi'].nunique()),
        class_counts=fm.loc[common_valid.numpy()].groupby('class_id').size().to_dict()))
    histories, choices, gradients, source_rows, source_classes, teacher_quality = [], [], [], [], [], []
    probability = sample_weights(fm); probability /= probability.sum()
    for seed in SEEDS:
        prior, teach = priors[seed], teachers[seed]
        for name, value in [('same', teach['same']), ('other', teach['other'])]:
            pred = value[common_valid].argmax(1).numpy()
            teacher_quality.append(dict(seed=seed, mode=name,
                **metrics(pred, labels[fit][common_valid].numpy()),
                positive_edges=float(positive_relations(value[common_valid], labels[fit][common_valid])[1].sum(1).float().mean())))
        for arm in ARMS:
            local_valid = teach['all_valid'] if arm == 'local_all' else common_valid
            values = teach['all'] if arm == 'local_all' else teach['other' if arm == 'other_matched' else 'same']
            reference = teach['all'] if arm == 'local_all' else teach['same']
            mode = 'full' if arm == 'full_matched' else 'local'
            coefficient, norms = initial_gradient_scale(prior[fit], values, labels[fit], local_valid.numpy(), probability, mode)
            if arm == 'other_matched':
                _, ref_norm = initial_gradient_scale(prior[fit], reference, labels[fit], local_valid.numpy(), probability, 'full')
                coefficient = ref_norm['full']/norms[mode]
            gradients.append(dict(arm=arm, seed=seed, coefficient=coefficient, own_full_rms=norms['full'],
                raw_aux_rms=norms[mode], scaled_aux_rms=coefficient*norms[mode]))
            teacher = torch.zeros_like(prior); teacher[fit] = values
            valid = torch.zeros(len(meta), dtype=torch.bool); valid[fit] = local_valid
            if arm == 'local_all':
                state = torch.load(FIRST/f'{arm}_{seed}.pt', map_location='cpu', weights_only=True)
                model = EvidenceStudent(centered(prior[fit]).square().mean(0).sqrt())
                model.load_state_dict(state['selected']); model.eval()
                torch.save(state, OUT/f'{arm}_{seed}.pt')
                choice = pd.read_csv(FIRST/'selection.csv').query('arm == @arm and seed == @seed').iloc[0].to_dict()
                choice['reused_from_parent'] = True
                trace = []
            else:
                model, trace, choice = train_one(xt, prior, teacher, valid, labels, meta, fit, roles, arm, seed, coefficient)
            students[arm, seed] = model
            histories.extend(trace); choices.append(choice)
            with torch.no_grad():
                score = model(xt[test], prior[test]).numpy()
            qm = meta.iloc[test].reset_index(drop=True)
            _, td, ok = vessel_targets(prior[test].numpy(), qm)
            source_rows.append(dict(arm=arm, seed=seed, snapshot='selected',
                **diagnose_scores(score, prior[test].numpy(), td, qm, ok), **pair_metrics(score, qm)))
            np.savez_compressed(OUT/f'{arm}_{seed}_source_scores.npz', obs_ids=test, scores=score)
            y, bp, pred = labels[test].numpy(), prior[test].argmax(1).numpy(), score.argmax(1)
            for c in np.unique(y):
                mask = y == c
                source_classes.append(dict(arm=arm, seed=seed, class_id=int(c), count=int(mask.sum()),
                    recall=float(100*(pred[mask] == c).mean()), rescue=int((mask&(bp != y)&(pred == y)).sum()),
                    harm=int((mask&(bp == y)&(pred != y)).sum())))
            state = torch.load(OUT/f'{arm}_{seed}.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(state['last'])
            with torch.no_grad():
                last = model(xt[test], prior[test]).numpy()
            source_rows.append(dict(arm=arm, seed=seed, snapshot='last', **metrics(last.argmax(1), y)))
            model.load_state_dict(state['selected'])
        for arm, old in REFERENCES.items():
            model = EvidenceStudent(centered(prior[fit]).square().mean(0).sqrt())
            model.load_state_dict(torch.load(PREVIOUS/f'{old}_{seed}.pt', map_location='cpu', weights_only=True)['selected'])
            students[arm, seed] = model.eval()
    pd.DataFrame(histories).to_csv(OUT/'training.csv', index=False)
    pd.DataFrame(choices).to_csv(OUT/'selection.csv', index=False)
    pd.DataFrame(gradients).to_csv(OUT/'initial_gradients.csv', index=False)
    pd.DataFrame(source_rows).to_csv(OUT/'source_results.csv', index=False)
    pd.DataFrame(source_classes).to_csv(OUT/'source_classes.csv', index=False)
    pd.DataFrame(teacher_quality).to_csv(OUT/'teacher_quality.csv', index=False)
    arms = ('baseline',)+tuple(REFERENCES)+ARMS
    tasks = []
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task = path.stem.removesuffix('_split'); tasks.append(task)
        with np.load(path) as z:
            support, query, sy = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
        assert len(support) == 64
        assert not set(meta.iloc[np.r_[support, query]].mmsi)&set(meta.iloc[source].mmsi)
        assert not set(meta.iloc[support].mmsi)&set(meta.iloc[query].mmsi)
        ids = np.r_[support, query]
        archive = dict(support_obs_ids=support, query_obs_ids=query)
        with np.load(PREVIOUS/f'{task}_predictions.npz') as old:
            for seed in SEEDS:
                operator, context = frozen[seed]
                with torch.no_grad():
                    changed = operator(h[ids], context[ids])
                    base_prior = changed@fixed['weight']+fixed['bias']
                phi = torch.cat([changed, torch.ones((len(ids), 1))], 1)
                for arm in arms:
                    with torch.no_grad():
                        prior = base_prior if arm == 'baseline' else students[arm, seed](xt[ids], base_prior)
                    beta, _, _ = residual_head(phi[:64], prior[:64], torch.tensor(sy), balanced=True)
                    score = (prior[64:]+phi[64:]@beta).numpy()
                    archive[f'{arm}__{seed}'] = score
                    if arm == 'baseline' or arm in REFERENCES:
                        np.testing.assert_array_equal(score, old[f'{REFERENCES.get(arm,arm)}__{seed}'])
                    if arm == 'local_all':
                        with np.load(FIRST/f'{task}_predictions.npz') as original:
                            np.testing.assert_array_equal(score, original[f'{arm}__{seed}'])
        np.savez_compressed(OUT/f'{task}_predictions.npz', **archive)
    save_json(OUT/'PREDICTIONS_SAVED.json', dict(tasks=len(tasks), target_query_metrics_read=False))
    rows, classes = [], []
    for task in tasks:
        with np.load(OUT/f'{task}_predictions.npz') as z:
            qm = meta.iloc[z['query_obs_ids']].reset_index(drop=True)
            y = qm.class_id.to_numpy()
            for seed in SEEDS:
                bp = z[f'baseline__{seed}'].argmax(1)
                for arm in arms:
                    score = z[f'{arm}__{seed}']; pred = score.argmax(1)
                    descriptor = dict(arm=arm, seed=seed, task=task, port=qm.port.iloc[0])
                    rows.append(dict(**descriptor, **metrics(pred,y), rescue=int(((bp != y)&(pred == y)).sum()),
                        harm=int(((bp == y)&(pred != y)).sum()), **pair_metrics(score,qm)))
                    for c in np.unique(y):
                        classes.append(dict(**descriptor, class_id=int(c), recall=float(100*(pred[y == c] == c).mean())))
    data = pd.DataFrame(rows)
    data.to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'target_classes.csv', index=False)
    for name, keys in [('summary',['arm']),('seed_summary',['arm','seed']),('port_summary',['arm','port'])]:
        data.groupby(keys).mean(numeric_only=True).reset_index().to_csv(OUT/f'target_{name}.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-started, new_student_fits=6,
        reused_local_all_fits=2, initial_parent_fits=8,
        exact_reference_target_replays=len(tasks)*len(SEEDS)*5, tasks=len(tasks), main_model_modified=False))
    print(data.groupby('arm')[['OA','BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
