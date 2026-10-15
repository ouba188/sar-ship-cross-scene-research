"""Bounded rescaling of existing classifier evidence, not error-target regression."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from condition_state_link import context_coordinates, quadratic_state, standardize
from relational_factors import BASE, CACHE, FactorNet, source_readout, design, sample_weights, metrics, save_json
from target_response_adapter import residual_head

OUT = BASE/'conditional_evidence_gate_20261010'
SUPPORTS = BASE/'target_response_adapter_balanced_20261010'
SEEDS = (31415, 27182)
ARMS = ('static', 'state', 'ordinary', 'condition')
STEPS = 400


class EvidenceGate(nn.Module):
    def __init__(self, basis):
        super().__init__()
        self.register_buffer('basis', basis.clone())
        self.map = nn.Linear(32, basis.shape[1])
        nn.init.zeros_(self.map.weight)
        nn.init.zeros_(self.map.bias)

    def gains(self, context):
        return .1*torch.tanh(self.map(context))

    def forward(self, features, context, fixed_gain=None):
        gain = self.gains(context) if fixed_gain is None else fixed_gain
        return features+((features@self.basis)*gain)@self.basis.T


def source_partitions(meta, fold2):
    rng = np.random.default_rng(20261010+181)
    group = meta.iloc[fold2]
    identity = []
    for _, g in group[['mmsi', 'class_id']].drop_duplicates().groupby('class_id'):
        identity.extend(rng.choice(g.mmsi.to_numpy(), max(1, int(np.ceil(len(g)*.25))), replace=False))
    identity_ids = group.index[group.mmsi.isin(identity)].to_numpy(copy=True)
    remaining = group[~group.mmsi.isin(identity)]
    temporal = []
    for _, g in remaining.groupby('mmsi'):
        scenes = np.sort(g['product'].unique())
        if len(scenes) >= 2:
            scene = rng.choice(scenes)
            temporal.extend(g.index[g['product'].eq(scene)].tolist())
    temporal = np.array(sorted(temporal), dtype=int)
    fit = np.setdiff1d(remaining.index.to_numpy(), temporal)
    assert not set(meta.iloc[fit].mmsi)&set(meta.iloc[identity_ids].mmsi)
    assert set(meta.iloc[temporal].mmsi).issubset(set(meta.iloc[fit].mmsi))
    assert not set(fit)&set(temporal)
    return dict(fit=fit, val_new_ship=identity_ids, val_new_time=temporal)


def evaluate_scores(features, context, model, weight, bias, mode='dynamic', mean_context=None, mean_gain=None):
    if model is None:
        changed = features
    elif mode == 'mean_context':
        changed = model(features, mean_context.expand(len(features), -1))
    elif mode == 'mean_gain':
        changed = model(features, context, fixed_gain=mean_gain)
    else:
        changed = model(features, context)
    return changed, changed@weight+bias


def train_gate(h, context, labels, meta, roles, weight, bias, basis, arm, seed):
    model = EvidenceGate(basis)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    target = 2*torch.eye(8)[labels]-1
    fit_ids = roles['fit']
    probability = sample_weights(meta.iloc[fit_ids].reset_index(drop=True))
    probability /= probability.sum()
    history, best_score, best_step = [], -np.inf, 0
    best = copy.deepcopy(model.state_dict())
    def validate(step, training_loss=np.nan):
        nonlocal best, best_score, best_step
        row = dict(arm=arm, seed=seed, step=step, training_loss=training_loss)
        bas = []
        with torch.no_grad():
            for group in ('val_new_ship', 'val_new_time'):
                ids = roles[group]
                changed = model(h[ids], context[ids])
                result = metrics((changed@weight+bias).argmax(1).numpy(), labels[ids].numpy())
                row.update({group+'_'+k: v for k, v in result.items()})
                bas.append(result['BA'])
        score = float(np.mean(bas))
        row['selection_BA'] = score
        history.append(row)
        if score > best_score+1e-10:
            best_score, best_step = score, step
            best = copy.deepcopy(model.state_dict())
    validate(0)
    for step in range(1, STEPS+1):
        rng = np.random.default_rng(seed+step)
        ids = rng.choice(fit_ids, 128, replace=True, p=probability)
        changed = model(h[ids], context[ids])
        score = changed@weight+bias
        loss = (score-target[ids]).square().mean()+.01*model.gains(context[ids]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 20 == 0:
            validate(step, float(loss.detach()))
    model.load_state_dict(best)
    print('SELECTED', arm, seed, best_step, best_score, flush=True)
    return model.eval(), history, dict(arm=arm, seed=seed, selected_step=best_step, validation_BA=best_score)


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    meta = pd.read_csv(BASE/'observations.csv')
    saved = np.load(BASE/'relational_factors_20261010/source_split.npz')
    source_ids = np.r_[saved['train_obs_ids'], saved['diagnostic_obs_ids']]
    folds = {f: source_ids[meta.iloc[source_ids].fold.to_numpy() == f] for f in (0, 1, 2)}
    roles = source_partitions(meta, folds[2])
    roles['test_source'] = folds[0]
    assert not set(meta.iloc[np.r_[folds[1], folds[2]]].mmsi)&set(meta.iloc[folds[0]].mmsi)
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], dtype=np.float32)
    classifier = source_readout(raw[folds[1]], meta.iloc[folds[1]].reset_index(drop=True))
    center, scale, weight, bias = classifier
    h = (torch.from_numpy(raw)-center)/scale/np.sqrt(512)
    prior = h@weight+bias
    basis = torch.linalg.svd(weight, full_matrices=False).U
    np.testing.assert_allclose((basis.T@basis).numpy(), np.eye(8), atol=1e-5)
    norm = ((raw-saved['center'])/saved['scale']).astype(np.float32)
    contexts = {'static': torch.zeros((len(meta), 32))}
    for old, arm in [('task_only', 'ordinary'), ('relations', 'condition')]:
        encoder = FactorNet()
        encoder.load_state_dict(torch.load(BASE/'relational_factors_20261010'/f'{old}.pt', map_location='cpu', weights_only=True))
        with torch.no_grad():
            values = torch.cat([encoder.condition(torch.from_numpy(norm[i:i+256])) for i in range(0, len(norm), 256)]).numpy()
        contexts[arm] = torch.tensor(standardize(values, roles['fit']), dtype=torch.float32)
    ordered = np.sort(prior.numpy(), axis=1)
    state = standardize(np.c_[prior.numpy(), ordered[:, -1], ordered[:, -1]-ordered[:, -2]], roles['fit'])
    contexts['state'] = torch.tensor(context_coordinates(quadratic_state(state), roles['fit'], 32), dtype=torch.float32)
    labels = torch.from_numpy(meta.class_id.to_numpy(copy=True)).long()
    coverage = []
    for group, ids in dict(classifier=folds[1], **roles).items():
        coverage.append(dict(group=group, chips=len(ids), ships=meta.iloc[ids].mmsi.nunique(), classes=meta.iloc[ids].class_id.nunique()))
    pd.DataFrame(coverage).to_csv(OUT/'source_coverage.csv', index=False)
    np.savez(OUT/'source_split.npz', classifier=folds[1], **roles)
    torch.save(dict(center=center, scale=scale, weight=weight, bias=bias, basis=basis), OUT/'fixed_classifier.pt')
    models, histories, selections = {}, [], []
    for arm in ARMS:
        for seed in SEEDS:
            model, history, selection = train_gate(h, contexts[arm], labels, meta, roles, weight, bias, basis, arm, seed)
            models[(arm, seed)] = model
            histories.extend(history)
            selections.append(selection)
            torch.save(model.state_dict(), OUT/f'{arm}_{seed}.pt')
    pd.DataFrame(histories).to_csv(OUT/'training.csv', index=False)
    pd.DataFrame(selections).to_csv(OUT/'selection.csv', index=False)
    variants = [('baseline', -1, 'disabled', None, 'static', None, None)]
    for (arm, seed), model in models.items():
        mean_context = contexts[arm][roles['fit']].mean(0, keepdim=True)
        with torch.no_grad():
            mean_gain = model.gains(contexts[arm][roles['fit']]).mean(0, keepdim=True)
        for mode in (('dynamic',) if arm == 'static' else ('dynamic', 'mean_context', 'mean_gain')):
            variants.append((arm, seed, mode, model, arm, mean_context, mean_gain))
    records, predictions, diagnostics = [], [], []
    with torch.no_grad():
        for arm, seed, mode, model, key, mc, mg in variants:
            for group, ids in roles.items():
                changed, scores = evaluate_scores(h[ids], contexts[key][ids], model, weight, bias, mode, mc, mg)
                pred, truth = scores.argmax(1).numpy(), labels[ids].numpy()
                row = dict(arm=arm, seed=seed, mode=mode, group=group, **metrics(pred, truth))
                base_ok, now_ok = prior[ids].argmax(1).numpy() == truth, pred == truth
                row.update(rescue=int((~base_ok&now_ok).sum()), harm=int((base_ok&~now_ok).sum()))
                records.append(row)
                predictions.append(pd.DataFrame(dict(arm=arm, seed=seed, mode=mode, group=group,
                    obs_id=meta.iloc[ids].obs_id.to_numpy(), mmsi=meta.iloc[ids].mmsi.to_numpy(), true=truth, pred=pred)))
                diagnostics.append(dict(arm=arm, seed=seed, mode=mode, group=group,
                    relative_update=float(torch.linalg.norm(changed-h[ids])/torch.linalg.norm(h[ids])),
                    max_relative_update=float((torch.linalg.norm(changed-h[ids], dim=1)/torch.linalg.norm(h[ids], dim=1).clamp_min(1e-8)).max())))
    pd.DataFrame(records).to_csv(OUT/'source_results.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT/'source_predictions.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'source_diagnostics.csv', index=False)
    target_rows, target_classes = [], []
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task = path.stem.removesuffix('_split')
        with np.load(path) as saved_task:
            support, query, y = [saved_task[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
        assert len(support) == 64
        assert not set(meta.iloc[support].mmsi)&set(meta.iloc[query].mmsi)
        assert not set(meta.iloc[np.r_[support, query]].mmsi)&set(meta.iloc[source_ids].mmsi)
        ids = np.r_[support, query]
        truth = meta.iloc[query].class_id.to_numpy()
        port, draw = meta.iloc[query].port.iloc[0], int(task.rsplit('draw', 1)[1])
        archive = dict(support_obs_ids=support, query_obs_ids=query)
        base_pred = None
        with torch.no_grad():
            for arm, seed, mode, model, key, mc, mg in variants:
                changed, local_prior = evaluate_scores(h[ids], contexts[key][ids], model, weight, bias, mode, mc, mg)
                phi = torch.cat([changed, torch.ones((len(ids), 1))], 1)
                beta, _, _ = residual_head(phi[:64], local_prior[:64], torch.from_numpy(y).long(), balanced=True)
                scores = local_prior[64:]+phi[64:]@beta
                pred = scores.argmax(1).numpy()
                if model is None:
                    base_pred = pred
                base_ok, now_ok = base_pred == truth, pred == truth
                target_rows.append(dict(task=task, port=port, draw=draw, arm=arm, seed=seed, mode=mode,
                    **metrics(pred, truth), rescue=int((~base_ok&now_ok).sum()), harm=int((base_ok&~now_ok).sum())))
                for c in np.unique(truth):
                    target_classes.append(dict(task=task, arm=arm, seed=seed, mode=mode, class_id=int(c),
                        support_count=int((y == c).sum()), recall=float(100*(pred[truth == c] == c).mean())))
                archive[f'{arm}_{seed}_{mode}'] = scores.numpy()
        np.savez_compressed(OUT/f'{task}.npz', **archive)
        print('TARGET', task, flush=True)
    source_summary = pd.DataFrame(records).groupby(['group', 'arm', 'mode'])[['OA', 'BA', 'rescue', 'harm']].mean().reset_index()
    target_results = pd.DataFrame(target_rows)
    target_summary = target_results.groupby(['arm', 'mode'])[['OA', 'BA', 'rescue', 'harm']].mean().reset_index()
    source_summary.to_csv(OUT/'source_summary.csv', index=False)
    target_results.to_csv(OUT/'target_results.csv', index=False)
    target_summary.to_csv(OUT/'target_summary.csv', index=False)
    pd.DataFrame(target_classes).to_csv(OUT/'target_classes.csv', index=False)
    target_results.groupby(['port', 'arm', 'mode'])[['OA', 'BA']].mean().reset_index().to_csv(OUT/'target_port_summary.csv', index=False)
    save_json(OUT/'protocol.json', dict(gate_parameters=264, static_effective_parameters=8,
        rank=8, gain_bound=.1, steps=STEPS, seeds=SEEDS, learning_rate=.01,
        selection='earliest checkpoint maximizing mean BA of source new-ship/new-time groups; step0 included',
        target_labels_per_task=64, target_head='same fixed fold1 source classifier for every arm; original 64-support residual head with ridge8',
        notes='Small frozen-R18 prototype, not full dual model. Target baseline differs from earlier all-source baseline because it uses the same fold1 classifier as the source diagnostic. Frozen context encoders historically used source folds1+2 including gate-validation identities. Final source fold0 and target labels are not used for checkpoint selection; repeated development test sets remain non-confirmatory.'))
    print('SOURCE', source_summary[source_summary.group == 'test_source'].to_string(index=False), flush=True)
    print('TARGET_SUMMARY', target_summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
