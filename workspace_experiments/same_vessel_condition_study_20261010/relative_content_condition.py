"""Source-evidenced shared-content anchors and matched bounded operators."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import copy
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import grouped_condition_anchor as old
from relative_factor_separation import (
    BASE, CACHE, FactorNet, relation_pools, sample_triplets, angular_gap,
    sample_weights, scene_cells, sample_scenes, group_center, confusion_loss,
    encode_two, diagnose, standardize, SEEDS, metrics, save_json,
)
from conditional_evidence_gate import SUPPORTS
from target_response_adapter import residual_head
from condition_state_link import context_coordinates
from decision_condition_consistency import readout_inputs, pair_metrics

OUT = BASE/'relative_content_condition_20261011'
FACTOR_ARMS = ('self_aux', 'group_aux')
STEPS = 200


def vessel_targets(x, meta):
    """Equal acquisition means; repeated chips in one scene do not dominate."""
    center, other = np.zeros_like(x), np.zeros_like(x)
    valid = np.zeros(len(x), dtype=bool)
    for _, group in meta.reset_index(drop=True).groupby('mmsi'):
        scenes = [(g.index.to_numpy(), x[g.index].mean(0))
                  for _, g in group.groupby('product')]
        if len(scenes) < 2:
            continue
        mean = np.mean([v for _, v in scenes], axis=0)
        for ids, local in scenes:
            center[ids] = mean
            other[ids] = (len(scenes)*mean-local)/(len(scenes)-1)
            valid[ids] = True
    return center, other, valid


def load_data():
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(old.OUT/'source_split.npz') as z:
        split = {k: z[k].copy() for k in z.files}
    fit, test = split['factor_fit'], split['test_source']
    roles = {k: split[k] for k in ('fit', 'val_new_ship', 'val_new_time', 'test_source')}
    source = np.unique(np.r_[fit, roles['val_new_ship'], roles['val_new_time'], test])
    assert not set(fit)&set(np.r_[roles['val_new_ship'], roles['val_new_time'], test])
    assert not set(meta.iloc[fit].mmsi)&set(meta.iloc[np.r_[roles['val_new_ship'], test]].mmsi)
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], np.float32)
    norm = ((raw-split['center'])/split['scale']).astype(np.float32)
    fixed = torch.load(old.OUT/'fixed_classifier.pt', map_location='cpu', weights_only=True)
    h = (torch.from_numpy(raw)-fixed['center'])/fixed['scale']/np.sqrt(512)
    return meta, split, roles, source, raw, norm, fixed, h


def probe():
    OUT.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    rows = []
    for group in ('factor_fit', 'test_source'):
        ids = split[group]
        mm = meta.iloc[ids].reset_index(drop=True)
        center, other, valid = vessel_targets(h[ids].numpy(), mm)
        y = mm.class_id.to_numpy()[valid]
        prior = (h[ids][valid]@fixed['weight']+fixed['bias']).numpy()
        originals = prior.argmax(1)
        for name, values in [('single', h[ids].numpy()), ('same_vessel_mean', center),
                             ('other_acquisitions', other)]:
            scores = values[valid]@fixed['weight'].numpy()+fixed['bias'].numpy()
            pred = scores.argmax(1)
            correct = scores[np.arange(len(y)), y]
            rival = scores.copy()
            rival[np.arange(len(y)), y] = -np.inf
            rows.append(dict(group=group, mode=name, chips=int(valid.sum()),
                ships=int(mm.loc[valid, 'mmsi'].nunique()), **metrics(pred, y),
                correct_margin=float(np.mean(correct-rival.max(1))),
                rescue=int(((originals != y)&(pred == y)).sum()),
                harm=int(((originals == y)&(pred != y)).sum())))
    pd.DataFrame(rows).to_csv(OUT/'shared_content_probe.csv', index=False)
    save_json(OUT/'PROBE_PROTOCOL.json', dict(
        purpose='Does real repeated observation provide recoverable shared class evidence?',
        restriction='Source classifier frozen; source fit and identity holdout only. No target evaluation.',
        operator='Mean each vessel/product, then equal product mean. Other_acquisitions excludes the target product.',
        boundary='Multi-observation diagnostic, not deployable single-chip accuracy or causal identification. Mean can retain identity/background and biased condition coverage.'))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def auxiliary_objective(stable_prediction, condition_prediction, truth, shared, arm):
    if arm == 'self_aux':
        return F.mse_loss(stable_prediction+condition_prediction, truth)
    return (F.mse_loss(stable_prediction, shared)
            +F.mse_loss(condition_prediction, truth-shared))


def train_factors(x, meta, anchor_data, arm, seed):
    torch.manual_seed(seed)
    model = FactorNet()
    delta_head = nn.Linear(32, 512, bias=False)
    adversary = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 8))
    anchor_head = nn.Linear(32, 3)
    state = torch.load(old.OUT/f'anchor_{seed}_factor.pt', map_location='cpu', weights_only=True)
    for name, network in [('model', model), ('delta_head', delta_head),
                          ('adversary', adversary), ('anchor_head', anchor_head)]:
        network.load_state_dict(state[name])
    shared_head, residual_decoder = nn.Linear(128, 512), nn.Linear(32, 512)
    parameters = (list(model.parameters())+list(delta_head.parameters())+list(anchor_head.parameters())
                  +list(shared_head.parameters())+list(residual_decoder.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=.001)
    adv_optimizer = torch.optim.Adam(adversary.parameters(), lr=.001)
    xt = torch.from_numpy(x)
    yt = torch.tensor(meta.class_id.to_numpy(copy=True))
    shared, _, valid = vessel_targets(x, meta)
    shared = torch.from_numpy(shared)
    probability = sample_weights(meta)
    probability /= probability.sum()
    pairs, pools = relation_pools(meta)
    assert valid[pairs].all()
    pair_ships = meta.iloc[pairs[:, 0]].mmsi.to_numpy()
    counts = pd.Series(pair_ships).value_counts()
    pw = np.array([1/counts[s] for s in pair_ships])
    pw /= pw.sum()
    delta_scale = float(np.average(((x[pairs[:, 1]]-x[pairs[:, 0]])**2).mean(1), weights=pw))
    cells = scene_cells(meta)
    angle, observed, orbit, orbit_weights = anchor_data
    history = []
    for step in range(1, STEPS+1):
        rng = np.random.default_rng(seed+step)
        ix = rng.choice(len(x), 96, replace=True, p=probability)
        pair = pairs[rng.choice(len(pairs), 32, replace=True, p=pw)]
        sem, scene = [sample_triplets(pools[k], rng) for k in ('semantic', 'scene')]
        ci, cg, cmask, cw = sample_scenes(cells, rng)
        s, c, correction, _, reconstruction = model(xt[ix])
        classification = F.cross_entropy(model.classifier(s), yt[ix])
        ids = np.r_[pair.ravel(), sem.ravel(), scene.ravel()]
        rs, rc, _, _, _ = model(xt[ids])
        ps = F.normalize(rs[:64], dim=1).reshape(32, 2, -1)
        invariant = (1-(ps[:, 0]*ps[:, 1]).sum(1)).mean()
        triples = np.arange(96).reshape(32, 3)
        semantic = F.relu(.2-angular_gap(rs[64:160], triples)).mean()
        scene_loss = F.relu(.2-angular_gap(rc[160:256], triples)).mean()
        pc = rc[:64].reshape(32, 2, -1)
        delta_loss = F.mse_loss(delta_head(pc[:, 1]-pc[:, 0]), xt[pair[:, 1]]-xt[pair[:, 0]])/delta_scale
        condition_scene = group_center(model.condition(xt[ci]), cg)
        adv_optimizer.zero_grad()
        adv_logits = adversary(condition_scene.detach()).masked_fill(~cmask, -1e4)
        adv_loss = (cw*F.cross_entropy(adv_logits, yt[ci], reduction='none')).sum()
        adv_loss.backward()
        adv_optimizer.step()
        for parameter in adversary.parameters():
            parameter.requires_grad_(False)
        confusion = confusion_loss(adversary(condition_scene), cmask, cw)
        self_recon = F.mse_loss(reconstruction, xt[ix])
        cross_recon = F.mse_loss(model.decoder(old.swap_inputs(rs[:64], rc[:64])), xt[pair.ravel()])
        auxiliary = auxiliary_objective(shared_head(rs[:64]), residual_decoder(rc[:64]),
                                         xt[pair.ravel()], shared[pair.ravel()], arm)
        anchor = old.anchor_loss(anchor_head(c), angle[ix], observed[ix], orbit[ix], orbit_weights)
        loss = (classification+.05*(self_recon+cross_recon)+.01*correction.square().mean()
                +.2*(invariant+semantic+scene_loss+delta_loss)+.1*(confusion+anchor+auxiliary))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.)
        optimizer.step()
        for parameter in adversary.parameters():
            parameter.requires_grad_(True)
        if step == 1 or step % 50 == 0:
            row = dict(arm=arm, seed=seed, step=step, classification=float(classification.detach()),
                invariant=float(invariant.detach()), delta=float(delta_loss.detach()),
                self_reconstruction=float(self_recon.detach()), cross_reconstruction=float(cross_recon.detach()),
                auxiliary=float(auxiliary.detach()), anchor=float(anchor.detach()), loss=float(loss.detach()))
            history.append(row)
            print('FACTOR', row, flush=True)
    torch.save(dict(model=model.state_dict(), delta_head=delta_head.state_dict(),
        adversary=adversary.state_dict(), anchor_head=anchor_head.state_dict(),
        shared_head=shared_head.state_dict(), residual_decoder=residual_decoder.state_dict()),
        OUT/f'{arm}_{seed}_factor.pt')
    return model.eval(), history


class BoundedOperator(nn.Module):
    def __init__(self, basis, amplitude, operator):
        super().__init__()
        self.register_buffer('basis', basis.clone())
        self.register_buffer('amplitude', amplitude.clone())
        self.operator = operator
        self.map = nn.Linear(32, basis.shape[1])
        nn.init.zeros_(self.map.weight)
        nn.init.zeros_(self.map.bias)

    def gains(self, context):
        return .1*torch.tanh(self.map(context))

    def forward(self, features, context):
        amplitude = features@self.basis if self.operator == 'scale' else self.amplitude
        return features+(amplitude*self.gains(context))@self.basis.T


def train_operator(h, context, labels, meta, roles, fixed, arm, operator, seed):
    basis, weight, bias = [fixed[k] for k in ('basis', 'weight', 'bias')]
    amplitude = (h[roles['fit']]@basis).square().mean(0).sqrt()
    model = BoundedOperator(basis, amplitude, operator)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    target = 2*torch.eye(8)[labels]-1
    fit = roles['fit']
    probability = sample_weights(meta.iloc[fit].reset_index(drop=True))
    probability /= probability.sum()
    best, best_score, best_step = copy.deepcopy(model.state_dict()), -np.inf, 0
    history = []

    def validate(step, loss=np.nan):
        nonlocal best, best_score, best_step
        row = dict(arm=arm, operator=operator, seed=seed, step=step, loss=loss)
        bas = []
        with torch.no_grad():
            for group in ('val_new_ship', 'val_new_time'):
                ids = roles[group]
                scores = model(h[ids], context[ids])@weight+bias
                result = metrics(scores.argmax(1).numpy(), labels[ids].numpy())
                row.update({group+'_'+k: v for k, v in result.items()})
                bas.append(result['BA'])
        row['selection_BA'] = float(np.mean(bas))
        history.append(row)
        if row['selection_BA'] > best_score+1e-10:
            best, best_score, best_step = copy.deepcopy(model.state_dict()), row['selection_BA'], step

    validate(0)
    for step in range(1, 401):
        rng = np.random.default_rng(seed+step)
        ids = rng.choice(fit, 128, replace=True, p=probability)
        score = model(h[ids], context[ids])@weight+bias
        loss = (score-target[ids]).square().mean()+.01*model.gains(context[ids]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 20 == 0:
            validate(step, float(loss.detach()))
    model.load_state_dict(best)
    if arm == 'anchor' and operator == 'scale':
        historical = torch.load(old.OUT/f'anchor_{seed}_gate.pt', map_location='cpu', weights_only=True)
        for key, value in historical.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    torch.save(model.state_dict(), OUT/f'{arm}_{operator}_{seed}_operator.pt')
    selected = dict(arm=arm, operator=operator, seed=seed, selected_step=best_step,
                    validation_BA=best_score, parameters=sum(p.numel() for p in model.parameters()))
    print('OPERATOR', selected, flush=True)
    return model.eval(), history, selected


def train():
    start = time.perf_counter()
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    fit, test = split['factor_fit'], split['test_source']
    targets = old.metadata_targets(meta, fit)
    anchor_data = tuple(t[fit] for t in targets[:3])+(targets[3],)
    np.savez(OUT/'source_split.npz', **split)
    torch.save(fixed, OUT/'fixed_classifier.pt')
    save_json(OUT/'PROTOCOL.json', dict(seeds=SEEDS, factor_arms=FACTOR_ARMS, factor_steps=STEPS,
        initialization='Both new arms resume the same historical B/anchor factor checkpoint with fresh identical auxiliary heads and Adam state.',
        factor_change='Both use existing joint self/cross reconstruction and metadata anchor losses. New .1 auxiliary: self_aux reconstructs x using Ds(s)+Dc(c); group_aux separately predicts equal-acquisition same-vessel source mean with Ds and x-minus-mean with Dc. Heads match, only targets differ.',
        group_proxy='Training-only per-vessel empirical mean, not true invariant/causal content. Only vessels with >=2 source acquisitions enter auxiliary batches. No target groups used in training or inference.',
        factor_selection='Fixed200 continuation steps, no target or diagnostic checkpoint selection.',
        operators='264params each: scale uses per-image projection; residual uses fixed source-fit RMS projection amplitude. Both bounded0.1*tanh, same Adam.01/400steps/MSE/.01gain regularization. Different feasible operators, not equal per-image perturbation norm.',
        contexts='Historical B/anchor, new self_aux, new group_aux, and unsupervised source-only PCA32 ordinary features. Context normalization uses original gate-fit462 samples.',
        primary='Full changed feature geometry and prior with unchanged64-label ridge8 support residual. Prior-only is fixed secondary computational-path diagnostic.',
        selection='Source new-ship299/new-time149 mean BA, every20steps earliest max including0. Source identity holdout823 is diagnostic only.',
        scope='Three ports x3 existing64 supports, two network seeds. Repeated development, not causal identification or full main model. Fixed all arms, no target selection.',
        evidence_file='shared_content_probe.csv', metadata=targets[4]))
    histories, diagnostics, contexts, encoders = [], [], {}, {}
    ordinary = context_coordinates(norm, fit, 32).astype(np.float32)
    contexts['ordinary', 0] = torch.from_numpy(standardize(ordinary, roles['fit']).astype(np.float32))
    for seed in SEEDS:
        for arm in ('anchor', 'joint')+FACTOR_ARMS:
            if arm in FACTOR_ARMS:
                model, trace = train_factors(norm[fit], meta.iloc[fit].reset_index(drop=True), anchor_data, arm, seed)
                histories.extend(trace)
            else:
                model = FactorNet()
                model.load_state_dict(torch.load(old.OUT/f'{arm}_{seed}_factor.pt', map_location='cpu', weights_only=True)['model'])
                model.eval()
            stable, condition = encode_two(model, norm)
            for factor, values in [('stable', stable), ('condition', condition)]:
                diagnostics.append(diagnose(values, norm, meta, fit, test, arm, seed, factor))
            source_archive = dict(obs_ids=source, stable=stable[source], condition=condition[source])
            np.savez_compressed(OUT/f'{arm}_{seed}_source_factors.npz', **source_archive)
            if arm != 'joint':
                contexts[arm, seed] = torch.from_numpy(standardize(condition, roles['fit']).astype(np.float32))
            encoders[arm, seed] = model
            pd.DataFrame(histories).to_csv(OUT/'factor_training.csv', index=False)
            pd.DataFrame(diagnostics).to_csv(OUT/'factor_diagnosis.csv', index=False)
    pd.DataFrame(diagnostics).groupby(['arm', 'factor']).mean(numeric_only=True).reset_index().to_csv(OUT/'factor_summary.csv', index=False)
    labels = torch.tensor(meta.class_id.to_numpy(copy=True))
    variants, traces, choices, source_rows = [], [], [], []
    weight, bias = fixed['weight'], fixed['bias']
    for arm in ('anchor', 'self_aux', 'group_aux', 'ordinary'):
        for seed in SEEDS:
            context = contexts[arm, 0 if arm == 'ordinary' else seed]
            for operator in ('scale', 'residual'):
                model, trace, selected = train_operator(h, context, labels, meta, roles, fixed, arm, operator, seed)
                variants.append((arm, operator, seed, model, context))
                traces.extend(trace)
                choices.append(selected)
                with torch.no_grad():
                    changed = model(h[test], context[test])
                    score = changed@weight+bias
                    base = (h[test]@weight+bias).argmax(1).numpy()
                    y = labels[test].numpy()
                    pred = score.argmax(1).numpy()
                    source_rows.append(dict(arm=arm, operator=operator, seed=seed,
                        **metrics(pred, y), rescue=int(((base != y)&(pred == y)).sum()),
                        harm=int(((base == y)&(pred != y)).sum()),
                        update_rms=float((changed-h[test]).square().mean().sqrt()),
                        gain_rms=float(model.gains(context[test]).square().mean().sqrt())))
                    np.savez_compressed(OUT/f'{arm}_{operator}_{seed}_source_scores.npz', obs_ids=test, scores=score.numpy())
    pd.DataFrame(traces).to_csv(OUT/'operator_training.csv', index=False)
    pd.DataFrame(choices).to_csv(OUT/'selection.csv', index=False)
    pd.DataFrame(source_rows).to_csv(OUT/'source_operator_results.csv', index=False)
    # Freeze and persist all target predictions before any target query-label metrics.
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task = path.stem.removesuffix('_split')
        with np.load(path) as z:
            support, query, sy = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
        assert len(support) == 64
        assert not set(meta.iloc[support].mmsi)&set(meta.iloc[query].mmsi)
        assert not set(meta.iloc[np.r_[support, query]].mmsi)&set(meta.iloc[source].mmsi)
        ids = np.r_[support, query]
        archive = dict(support_obs_ids=support, query_obs_ids=query)
        with torch.no_grad():
            for arm, operator, seed, model, context in [('baseline', 'none', -1, None, None)]+variants:
                changed = h[ids] if model is None else model(h[ids], context[ids])
                for route in (('both',) if model is None else ('both', 'prior')):
                    phi, prior = readout_inputs(h[ids], changed, weight, bias, route)
                    beta, _, _ = residual_head(phi[:64], prior[:64], torch.tensor(sy), balanced=True)
                    score = (prior[64:]+phi[64:]@beta).numpy()
                    key = f'{arm}__{operator}__{route}__{seed}'
                    archive[key] = score
                    if (arm, operator, route) == ('anchor', 'scale', 'both'):
                        with np.load(old.OUT/f'{task}.npz') as z:
                            np.testing.assert_array_equal(score, z[f'anchor_{seed}'])
        np.savez_compressed(OUT/f'{task}_predictions.npz', **archive)
        print('FROZEN_TARGET', task, flush=True)
    save_json(OUT/'PREDICTIONS_SAVED.json', dict(tasks=9, target_query_metrics_read=False))
    evaluate(meta)
    save_json(OUT/'TRAIN_COMPLETE.json', dict(seconds=time.perf_counter()-start, new_factor_fits=4,
        operator_fits=16, historical_factor_reuses=4, historical_target_exact_replays=18,
        resources='CPU single thread', main_model_modified=False))


def evaluate(meta):
    rows, classes, pairs = [], [], []
    for path in sorted(OUT.glob('*_predictions.npz')):
        task = path.stem.removesuffix('_predictions')
        with np.load(path) as z:
            query, support = z['query_obs_ids'], z['support_obs_ids']
            qm = meta.iloc[query].reset_index(drop=True)
            y = qm.class_id.to_numpy()
            sy = meta.iloc[support].class_id.to_numpy()
            base = z['baseline__none__both__-1'].argmax(1)
            for key in z.files:
                if '__' not in key:
                    continue
                arm, operator, route, seed = key.split('__')
                scores = z[key]
                pred = scores.argmax(1)
                descriptor = dict(arm=arm, operator=operator, route=route, seed=int(seed), task=task, port=qm.port.iloc[0])
                rows.append(dict(**descriptor, **metrics(pred, y),
                    rescue=int(((base != y)&(pred == y)).sum()), harm=int(((base == y)&(pred != y)).sum())))
                pairs.append(dict(**descriptor, **pair_metrics(scores, qm)))
                for cls in np.unique(y):
                    classes.append(dict(**descriptor, class_id=int(cls), support_count=int((sy == cls).sum()),
                        recall=float(100*(pred[y == cls] == cls).mean())))
    data = pd.DataFrame(rows)
    data.to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'target_classes.csv', index=False)
    pd.DataFrame(pairs).to_csv(OUT/'target_pairs.csv', index=False)
    for name, extra in [('summary', []), ('seed_summary', ['seed']), ('port_summary', ['port'])]:
        data.groupby(['arm', 'operator', 'route']+extra)[['OA', 'BA', 'rescue', 'harm']].mean().reset_index().to_csv(OUT/f'target_{name}.csv', index=False)
    print(data.groupby(['arm', 'operator', 'route'])[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'train'])
    args = parser.parse_args()
    probe() if args.stage == 'probe' else train()
