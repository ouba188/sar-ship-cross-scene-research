"""Fixed trial of repeated-condition relation intervals alongside full teaching."""

import copy
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from local_relation_distillation import (
    BASE, OUT as LOCAL, PREVIOUS, SEEDS, SUPPORTS, EvidenceStudent, centered,
    previous, FactorNet, BoundedOperator, encode_two, standardize, load_data,
    sample_weights, vessel_targets, residual_head, metrics, save_json,
)
from shared_evidence_distillation import objective
from shared_evidence_diagnose import diagnose_scores
from relation_teacher_matching import build_matches, summarize_matches
from reliable_relation_targets import build_targets, interval_rows

OUT = BASE/'reliable_relation_distillation_20261011'
ARMS = ('mean', 'same', 'joint')


def gradient_scale(prior, teacher, labels, target, probability):
    valid = target['eligible']
    p = torch.tensor(probability[valid], dtype=prior.dtype)
    p /= p.sum()
    norms = {}
    for mode in ('full', 'interval'):
        score = prior[valid].detach().clone().requires_grad_(True)
        if mode == 'full':
            rows = (centered(score)-centered(teacher[valid])).square().mean(1)
        else:
            rows = interval_rows(score, labels[valid], target['lower'][valid],
                                 target['upper'][valid], target['mask'][valid])
        grad = torch.autograd.grad(rows.sum(), score)[0]
        norms[mode] = float((p*grad.square().sum(1)).sum().sqrt())
    if norms['interval'] <= 1e-12:
        raise ValueError('No initial interval teaching signal; inspect coverage instead of training.')
    return norms['full']/norms['interval'], norms


def train(x, prior, teacher, valid, labels, fit, roles, probability, target, seed, arm, scale):
    torch.manual_seed(seed)
    model = EvidenceStudent(centered(prior[fit]).square().mean(0).sqrt())
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    best, best_ba, best_step = copy.deepcopy(model.state_dict()), -np.inf, 0
    trace = []

    def validate(step, losses=None):
        nonlocal best, best_ba, best_step
        row = dict(seed=seed, arm=arm, step=step, **(losses or {}))
        bas = []
        with torch.no_grad():
            for role in ('val_new_ship', 'val_new_time'):
                ids = roles[role]
                result = metrics(model(x[ids], prior[ids]).argmax(1).numpy(), labels[ids].numpy())
                row.update({role+'_'+k: v for k, v in result.items()})
                bas.append(result['BA'])
        row['selection_BA'] = float(np.mean(bas))
        trace.append(row)
        if row['selection_BA'] > best_ba+1e-10:
            best, best_ba, best_step = copy.deepcopy(model.state_dict()), row['selection_BA'], step

    validate(0)
    for step in range(1, 401):
        ids = np.random.default_rng(seed+step).choice(fit, 128, replace=True, p=probability)
        score = model(x[ids], prior[ids])
        base_loss, supervised, full = objective(score, labels[ids], teacher[ids], valid[ids],
                                                'other_acquisition_teacher')
        eligible = target['eligible'][ids]
        rows = interval_rows(score, labels[ids], target['lower'][ids], target['upper'][ids],
                             target['mask'][ids])
        interval = rows[eligible].mean() if eligible.any() else score.sum()*0
        loss = base_loss+.25*scale*interval
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
        if step % 20 == 0:
            validate(step, dict(loss=float(loss.detach()), supervised=float(supervised.detach()),
                                full=float(full.detach()), interval=float(interval.detach())))
    state = dict(selected=best, last=copy.deepcopy(model.state_dict()))
    torch.save(state, OUT/f'{arm}_{seed}.pt')
    model.load_state_dict(best)
    selection = dict(seed=seed, arm=arm, step=best_step, selection_BA=best_ba, scale=scale)
    print('SELECTED', selection, flush=True)
    return model.eval(), trace, selection


def relation_diagnostics(score, labels, target, weights):
    margin = score.gather(1, labels[:, None])-score
    mask = target['mask']
    count = mask.sum(1)
    valid = target['eligible'] & (count > 0)
    if not valid.any():
        return dict(anchors=0, edges=0, positive_pct=np.nan, interval_loss=np.nan)
    p = torch.tensor(weights, dtype=score.dtype)[valid]; p /= p.sum()
    positive = ((margin > 0)&mask).sum(1)/count.clamp_min(1)
    loss = interval_rows(score, labels, target['lower'], target['upper'], mask)
    return dict(anchors=int(valid.sum()), edges=int(mask.sum()),
                positive_pct=float(100*(p*positive[valid]).sum()),
                interval_loss=float((p*loss[valid]).sum()))


def tensors(target):
    return {k: torch.from_numpy(v) for k, v in target.items()}


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    fit, held = split['factor_fit'], split['test_source']
    fm = meta.iloc[fit].reset_index(drop=True)
    hm = meta.iloc[held].reset_index(drop=True)
    matches = pd.read_pickle(LOCAL/'source_matches.pkl')
    held_matches = build_matches(hm)
    held_matches.to_pickle(OUT/'held_matches.pkl')
    save_json(OUT/'HELD_MATCHING.json', summarize_matches(hm, held_matches))
    save_json(OUT/'PROTOCOL.json', dict(
        seeds=SEEDS, arms=ARMS, steps=400, full_weight=.5, interval_weight=.25,
        reference='Reuse exact full_all checkpoint and original predictions; same original MSE, source1296,400steps, batches,33352params.',
        hypotheses='Does recurrence-based reliability improve over positive mean teaching when full-score stability is retained? Does evidence from different same-class vessels further help?',
        eligibility='Metadata-only common cohort: at least2 distinct same-vessel donor acquisitions and2 distinct other donor identities, excluding anchor vessel/product via previous repaired matcher. Never relax on target outcomes.',
        intervals='mean: equal same/other scene-average margins as point target if positive. same: min/max positive margins across all same-vessel donors. joint: min/max across pooled same and other donors, all margins positive. Bounds clip[0,2], truth edge excluded. Different masks/widths intentionally tested together; not pure mask-only ablation.',
        objective='Original supervised MSE +.5 original all-other-acquisition centered8score MSE +.25 scaled interval squared distance. Interval averages edges per image then common eligible images, zero-edge rows contribute zero. No auxiliary replacement.',
        scale='Fixed source-only initial logit-gradient RMS of interval matched to fullMSE within eligible cohort; does not equalize later dynamics. No hyperparameter sweep.',
        validation='Earliest best mean original source new-ship/new-time BA every20steps incl0. Held-source823 not for selection. All target predictions saved before readout.',
        deployment='Single chip, no target identity or multi-view information. Original3ports x3draws x64labels, fixed feature kernel and ridge8.',
        diagnosis='Held-source donor intervals use held labels only for posthoc conditional relation diagnostics, NOT model training, deployment or unbiased causal estimates.',
        boundary='Previously reused development data; R18 frozen side prototype, not complete dual model. No causal identification or independent confirmation.'))
    x = torch.from_numpy(norm)
    labels = torch.tensor(meta.class_id.to_numpy(copy=True))
    probability = sample_weights(fm); probability /= probability.sum()
    students, frozen, source_scores = {}, {}, {}
    traces, choices, gradients, coverage, diagnostics, source_rows = [], [], [], [], [], []
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
        frozen[seed] = operator, context
        _, values, ok = vessel_targets(prior[fit].numpy(), fm)
        teacher = torch.zeros_like(prior); teacher[fit] = torch.from_numpy(values)
        valid = torch.zeros(len(meta), dtype=torch.bool); valid[fit] = torch.from_numpy(ok)
        targets, audit = build_targets(prior[fit].numpy(), fm, matches)
        audit.to_csv(OUT/f'fit_coverage_{seed}.csv', index=False)
        held_targets, held_audit = build_targets(prior[held].numpy(), hm, held_matches)
        held_audit.to_csv(OUT/f'held_coverage_{seed}.csv', index=False)
        for role, mm, ts in [('fit', fm, targets), ('held', hm, held_targets)]:
            for mode, tt in ts.items():
                mask = tt['mask']; eligible = tt['eligible']
                for c in sorted(mm.class_id.unique()):
                    rows = mm.class_id.to_numpy() == c
                    active = rows & mask.any(1)
                    coverage.append(dict(seed=seed, role=role, mode=mode, class_id=int(c),
                        eligible_anchors=int((rows&eligible).sum()), active_anchors=int(active.sum()),
                        active_ships=int(mm.loc[active,'mmsi'].nunique()), edges=int(mask[rows].sum())))
        reference = EvidenceStudent(centered(prior[fit]).square().mean(0).sqrt())
        reference.load_state_dict(torch.load(PREVIOUS/f'other_acquisition_teacher_{seed}.pt', map_location='cpu', weights_only=True)['selected'])
        students['full', seed] = reference.eval()
        for arm in ARMS:
            target = tensors(targets[arm])
            scale, norms = gradient_scale(prior[fit], teacher[fit], labels[fit], target, probability)
            gradients.append(dict(seed=seed, arm=arm, scale=scale, **norms))
            global_target = {k: torch.zeros((len(meta),)+tuple(v.shape[1:]), dtype=v.dtype) for k,v in target.items()}
            for k,v in target.items():
                global_target[k][fit] = v
            model, trace, selection = train(x, prior, teacher, valid, labels, fit, roles,
                                            probability, global_target, seed, arm, scale)
            students[arm,seed] = model
            traces.extend(trace); choices.append(selection)
        for role, ids, mm, ts in [('fit', fit, fm, targets), ('held', held, hm, held_targets)]:
            w = sample_weights(mm)
            _, other, good = vessel_targets(prior[ids].numpy(), mm)
            for arm in ('baseline', 'full')+ARMS:
                with torch.no_grad():
                    score = prior[ids] if arm == 'baseline' else students[arm,seed](x[ids], prior[ids])
                source_scores[f'{role}_{arm}_{seed}'] = score.numpy()
                source_rows.append(dict(seed=seed, role=role, arm=arm,
                    **diagnose_scores(score.numpy(), prior[ids].numpy(), other, mm, good)))
                for mode, target in ts.items():
                    diagnostics.append(dict(seed=seed, role=role, arm=arm, mode=mode,
                        **relation_diagnostics(score, labels[ids], tensors(target), w)))
    np.savez_compressed(OUT/'source_scores.npz', fit_obs_ids=fit, held_obs_ids=held, **source_scores)
    for name, rows in [('training',traces),('selection',choices),('gradients',gradients),
                        ('coverage',coverage),('relations',diagnostics),('source_results',source_rows)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv', index=False)
    tasks = []
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task = path.stem.removesuffix('_split'); tasks.append(task)
        with np.load(path) as z:
            support, query, sy = [z[k] for k in ('support_obs_ids','query_obs_ids','support_labels')]
        assert len(support) == 64
        assert not set(meta.iloc[np.r_[support,query]].mmsi)&set(meta.iloc[source].mmsi)
        assert not set(meta.iloc[support].mmsi)&set(meta.iloc[query].mmsi)
        ids = np.r_[support,query]
        archive = dict(support_obs_ids=support,query_obs_ids=query)
        with np.load(PREVIOUS/f'{task}_predictions.npz') as old:
            for seed in SEEDS:
                operator, context = frozen[seed]
                with torch.no_grad():
                    changed = operator(h[ids],context[ids])
                    base = changed@fixed['weight']+fixed['bias']
                phi = torch.cat([changed,torch.ones((len(ids),1))],1)
                for arm in ('baseline','full')+ARMS:
                    with torch.no_grad():
                        score = base if arm == 'baseline' else students[arm,seed](x[ids],base)
                    beta,_,_ = residual_head(phi[:64],score[:64],torch.tensor(sy),balanced=True)
                    result = (score[64:]+phi[64:]@beta).numpy()
                    if arm in ('baseline','full'):
                        key = 'baseline' if arm == 'baseline' else 'other_acquisition_teacher'
                        np.testing.assert_array_equal(result,old[f'{key}__{seed}'])
                    archive[f'{arm}__{seed}'] = result
        np.savez_compressed(OUT/f'{task}_predictions.npz',**archive)
    save_json(OUT/'PREDICTIONS_SAVED.json',dict(tasks=len(tasks),query_metrics_read=False))
    results, classes = [], []
    for task in tasks:
        with np.load(OUT/f'{task}_predictions.npz') as z:
            qm = meta.iloc[z['query_obs_ids']]
            y = qm.class_id.to_numpy()
            for seed in SEEDS:
                bp = z[f'full__{seed}'].argmax(1)
                for arm in ('baseline','full')+ARMS:
                    pred = z[f'{arm}__{seed}'].argmax(1)
                    row = dict(task=task,port=qm.port.iloc[0],seed=seed,arm=arm)
                    results.append(dict(**row,**metrics(pred,y),rescue=int(((bp!=y)&(pred==y)).sum()),
                                        harm=int(((bp==y)&(pred!=y)).sum())))
                    for c in np.unique(y):
                        classes.append(dict(**row,class_id=int(c),count=int((y==c).sum()),
                                            recall=float(100*(pred[y==c]==c).mean())))
    df = pd.DataFrame(results)
    df.to_csv(OUT/'target_results.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'target_classes.csv',index=False)
    for name, keys in [('summary',['arm']),('seed_summary',['arm','seed']),('port_summary',['arm','port'])]:
        df.groupby(keys)[['OA','BA']].mean().reset_index().to_csv(OUT/f'target_{name}.csv',index=False)
    save_json(OUT/'COMPLETE.json',dict(seconds=time.perf_counter()-started,new_fits=6,
        exact_reference_replays=36,main_model_modified=False))
    print(df.groupby('arm')[['OA','BA']].mean().to_string(),flush=True)


if __name__ == '__main__':
    main()
