"""Source same-vessel decision consistency and prior-only conditional correction."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import copy
import itertools
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from grouped_condition_anchor import BASE, CACHE, OUT as PREVIOUS, FactorNet, SEEDS, metrics, save_json
from conditional_evidence_gate import EvidenceGate, SUPPORTS
from relative_factor_separation import sample_weights, standardize
from target_response_adapter import residual_head

OUT = BASE/'decision_condition_consistency_20261011'
STEPS = 400
CONSISTENCY_WEIGHT = .05


def same_ship_pairs(meta):
    scenes = meta['product'].to_numpy()
    pairs = []
    for _, group in meta.reset_index(drop=True).groupby('mmsi'):
        pairs.extend((a, b) for a, b in itertools.combinations(group.index, 2) if scenes[a] != scenes[b])
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def pair_weights(meta, pairs):
    ships = meta.iloc[pairs[:, 0]].mmsi.to_numpy()
    counts = pd.Series(ships).value_counts()
    weights = np.array([1/counts[v] for v in ships])
    return weights/weights.sum()


def js_divergence(a, b):
    loga, logb = F.log_softmax(a, dim=-1), F.log_softmax(b, dim=-1)
    logmean = torch.logaddexp(loga, logb)-np.log(2.)
    return .5*((loga.exp()*(loga-logmean)).sum(-1)+(logb.exp()*(logb-logmean)).sum(-1))


def readout_inputs(raw, changed, weight, bias, path):
    prior = changed@weight+bias
    geometry = raw if path == 'prior' else changed
    return torch.cat([geometry, torch.ones((len(geometry), 1), dtype=geometry.dtype)], 1), prior


def pair_metrics(scores, meta, pairs=None):
    pairs = same_ship_pairs(meta) if pairs is None else pairs
    if not len(pairs):
        return dict(pairs=0, paired_ships=0)
    weights = pair_weights(meta, pairs)
    scores = torch.as_tensor(scores)
    pred = scores.argmax(1).numpy()
    y = meta.class_id.to_numpy()
    a, b = pairs.T
    assert np.array_equal(y[a], y[b])
    pa, pb = pred[a], pred[b]
    ca, cb = pa == y[a], pb == y[b]
    centered = scores-scores.mean(1, keepdim=True)
    directions = F.normalize(centered, dim=1)
    p = scores.softmax(1)
    entropy = -(p*F.log_softmax(scores, dim=1)).sum(1).numpy()
    magnitude = centered.square().mean(1).sqrt().numpy()
    return dict(pairs=len(pairs), paired_ships=int(meta.iloc[a].mmsi.nunique()),
        js=float(weights@js_divergence(scores[a], scores[b]).numpy()),
        direction_distance=float(weights@(1-(directions[a]*directions[b]).sum(1)).numpy()),
        logit_rms=float(weights@(.5*(magnitude[a]+magnitude[b]))),
        entropy=float(weights@(.5*(entropy[a]+entropy[b]))),
        disagree=float(100*weights@(pa != pb)), both_correct=float(100*weights@(ca&cb)),
        same_wrong=float(100*weights@((pa == pb)&~ca&~cb)),
        one_correct=float(100*weights@(ca^cb)), both_wrong_different=float(100*weights@((pa != pb)&~ca&~cb)))


def train(h, context, labels, meta, roles, fixed, paired, probability_pair, js_scale, objective, seed):
    gate = EvidenceGate(fixed['basis'])
    optimizer = torch.optim.Adam(gate.parameters(), lr=.01)
    weight, bias = fixed['weight'], fixed['bias']
    target = 2*torch.eye(8)[labels]-1
    fit = roles['fit']
    probability = sample_weights(meta.iloc[fit].reset_index(drop=True))
    probability /= probability.sum()
    coefficient = CONSISTENCY_WEIGHT if objective == 'consistent' else 0.
    best, best_score, best_step = copy.deepcopy(gate.state_dict()), -np.inf, 0
    history = []

    def validate(step, losses=None):
        nonlocal best, best_score, best_step
        row = dict(objective=objective, seed=seed, step=step, **(losses or {}))
        bas = []
        with torch.no_grad():
            for group in ('val_new_ship', 'val_new_time'):
                ids = roles[group]
                score = gate(h[ids], context[ids])@weight+bias
                result = metrics(score.argmax(1).numpy(), labels[ids].numpy())
                row.update({group+'_'+k: v for k, v in result.items()})
                bas.append(result['BA'])
        row['selection_BA'] = float(np.mean(bas))
        history.append(row)
        if row['selection_BA'] > best_score+1e-10:
            best_score, best_step = row['selection_BA'], step
            best = copy.deepcopy(gate.state_dict())

    validate(0)
    for step in range(1, STEPS+1):
        rng = np.random.default_rng(seed+step)
        ids = rng.choice(fit, 128, replace=True, p=probability)
        pairs = paired[rng.choice(len(paired), 32, replace=True, p=probability_pair)]
        changed = gate(h[ids], context[ids])
        score = changed@weight+bias
        classification = (score-target[ids]).square().mean()
        ps = (gate(h[pairs.ravel()], context[pairs.ravel()])@weight+bias).reshape(-1, 2, 8)
        js = js_divergence(ps[:, 0], ps[:, 1]).mean()
        regularizer = .01*gate.gains(context[ids]).square().mean()
        loss = classification+regularizer
        if coefficient:
            loss = loss+coefficient*js/js_scale
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.)
        optimizer.step()
        if step % 20 == 0:
            validate(step, dict(classification=float(classification.detach()), js=float(js.detach()),
                                normalized_js=float(js.detach()/js_scale), loss=float(loss.detach())))
    torch.save(gate.state_dict(), OUT/f'{objective}_{seed}_last.pt')
    gate.load_state_dict(best)
    torch.save(gate.state_dict(), OUT/f'{objective}_{seed}.pt')
    selection = dict(objective=objective, seed=seed, selected_step=best_step, validation_BA=best_score,
                     coefficient=coefficient, pair_js_normalizer=js_scale)
    print('SELECTED', selection, flush=True)
    return gate.eval(), history, selection


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    started = time.perf_counter()
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(PREVIOUS/'source_split.npz') as z:
        split = {k: z[k].copy() for k in z.files}
    roles = {k: split[k] for k in ('val_new_ship', 'val_new_time', 'test_source')}
    roles['fit'] = split['factor_fit']
    fit = roles['fit']
    assert not set(fit)&set(np.r_[roles['val_new_ship'], roles['val_new_time'], roles['test_source']])
    assert not set(meta.iloc[fit].mmsi)&set(meta.iloc[np.r_[roles['val_new_ship'], roles['test_source']]].mmsi)
    source_ids = np.unique(np.concatenate(list(roles.values())))
    tm = meta.iloc[fit].reset_index(drop=True)
    pairs = same_ship_pairs(tm)
    weights_pair = pair_weights(tm, pairs)
    paired = fit[pairs]
    np.savez(OUT/'source_split.npz', **roles, classifier=split['classifier'], center=split['center'], scale=split['scale'])
    pd.DataFrame(dict(a=paired[:, 0], b=paired[:, 1], probability=weights_pair,
                      mmsi=meta.iloc[paired[:, 0]].mmsi.to_numpy(), class_id=meta.iloc[paired[:, 0]].class_id.to_numpy())).to_csv(OUT/'fit_pairs.csv', index=False)
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], np.float32)
    norm = (raw-split['center'])/split['scale']
    fixed = torch.load(PREVIOUS/'fixed_classifier.pt', map_location='cpu', weights_only=True)
    torch.save(fixed, OUT/'fixed_classifier.pt')
    h = (torch.from_numpy(raw)-fixed['center'])/fixed['scale']/np.sqrt(512)
    weight, bias = fixed['weight'], fixed['bias']
    prior = h@weight+bias
    js_scale = max(float(weights_pair@js_divergence(prior[paired[:, 0]], prior[paired[:, 1]]).numpy()), 1e-4)
    labels = torch.from_numpy(meta.class_id.to_numpy(copy=True)).long()
    save_json(OUT/'PROTOCOL.json', dict(steps=STEPS, seeds=SEEDS, consistency_weight=CONSISTENCY_WEIGHT,
        js_normalizer=js_scale, fit_chips=len(fit), fit_ships=int(tm.mmsi.nunique()),
        fit_pairs=len(pairs), paired_ships=int(tm.iloc[pairs[:, 0]].mmsi.nunique()),
        objective='Original class/ship-balanced supervised squared loss +.01gainL2; consistent adds .05 same-ship JS normalized by source raw-prior mean JS, floor1e-4. Vessel-balanced pairs32, supervised128, Adam.01.',
        paths='both: corrected source prior and corrected support geometry; prior: corrected source prior but raw original512 geometry. All64 residual heads refit, ridge8.',
        encoder='Frozen previous anchor seed-matched factor checkpoints. Context standardization refit using common1296 source pool. No encoder training.',
        selection='Earliest best mean new-ship/new-observation source BA among0,20,...400. Identical gate weights for two deployment paths;4unique fits yield8seed/path cells.',
        labels='Original source labels only; same9x64 target supports. No query selection or per-class handcrafted rule.',
        caveats='Expanded source gate pool differs from previous462, so retrained within-run reference is primary. This prototype is not the full dual model; repeated-development data and shared scenes remain. JS agreement can include both-wrong cases; diagnosed separately.'))
    models, contexts, histories, selections, source_rows, source_pairs, source_predictions = {}, {}, [], [], [], [], []
    for seed in SEEDS:
        encoder = FactorNet()
        encoder.load_state_dict(torch.load(PREVIOUS/f'anchor_{seed}_factor.pt', map_location='cpu', weights_only=True)['model'])
        encoder.eval().requires_grad_(False)
        with torch.no_grad():
            c = torch.cat([encoder.condition(torch.from_numpy(norm[i:i+256])) for i in range(0, len(norm), 256)]).numpy()
        context = torch.from_numpy(standardize(c, fit).astype(np.float32))
        contexts[seed] = context
        np.savez_compressed(OUT/f'context_normalization_{seed}.npz', center=c[fit].mean(0), scale=np.maximum(c[fit].std(0), 1e-3))
        for objective in ('reference', 'consistent'):
            gate, history, selection = train(h, context, labels, meta, roles, fixed, paired, weights_pair, js_scale, objective, seed)
            models[objective, seed] = gate
            histories.extend(history)
            selections.append(selection)
    pd.DataFrame(histories).to_csv(OUT/'training.csv', index=False)
    pd.DataFrame(selections).to_csv(OUT/'selection.csv', index=False)
    variants = [('baseline', -1, None)]+[(objective, seed, gate) for (objective, seed), gate in models.items()]
    with torch.no_grad():
        for objective, seed, gate in variants:
            for group in ('fit', 'val_new_ship', 'val_new_time', 'test_source'):
                ids = roles[group]
                changed = h[ids] if gate is None else gate(h[ids], contexts[seed][ids])
                scores = changed@weight+bias
                pred, y = scores.argmax(1).numpy(), labels[ids].numpy()
                raw_ok = prior[ids].argmax(1).numpy() == y
                source_rows.append(dict(objective=objective, seed=seed, group=group, **metrics(pred, y),
                                        rescue=int((~raw_ok&(pred == y)).sum()), harm=int((raw_ok&(pred != y)).sum())))
                source_pairs.append(dict(objective=objective, seed=seed, group=group,
                                         **pair_metrics(scores, meta.iloc[ids].reset_index(drop=True))))
                source_predictions.append(pd.DataFrame(dict(objective=objective, seed=seed, group=group,
                    obs_id=ids, true=y, pred=pred)))
                if group in ('fit', 'test_source'):
                    np.savez_compressed(OUT/f'{objective}_{seed}_{group}_scores.npz', obs_ids=ids, scores=scores.numpy())
    pd.DataFrame(source_rows).to_csv(OUT/'source_results.csv', index=False)
    pd.DataFrame(source_pairs).to_csv(OUT/'source_pair_diagnosis.csv', index=False)
    pd.concat(source_predictions, ignore_index=True).to_csv(OUT/'source_predictions.csv', index=False)
    rows, classes, target_pairs, replays = [], [], [], []
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task = path.stem.removesuffix('_split')
        with np.load(path) as z:
            support, query, y = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
        assert len(support) == 64
        assert not set(meta.iloc[support].mmsi)&set(meta.iloc[query].mmsi)
        assert not set(meta.iloc[np.r_[support, query]].mmsi)&set(meta.iloc[source_ids].mmsi)
        ids = np.r_[support, query]
        qm = meta.iloc[query].reset_index(drop=True)
        truth = qm.class_id.to_numpy()
        paired_query = same_ship_pairs(qm)
        archive = dict(support_obs_ids=support, query_obs_ids=query)
        raw_pred = None
        with torch.no_grad():
            for objective, seed, gate in variants:
                changed = h[ids] if gate is None else gate(h[ids], contexts[seed][ids])
                for route in (('both',) if gate is None else ('both', 'prior')):
                    phi, local_prior = readout_inputs(h[ids], changed, weight, bias, route)
                    beta, _, _ = residual_head(phi[:64], local_prior[:64], torch.tensor(y), balanced=True)
                    scores = local_prior[64:]+phi[64:]@beta
                    pred = scores.argmax(1).numpy()
                    if gate is None:
                        raw_pred = pred
                        with np.load(PREVIOUS/f'{task}.npz') as old:
                            np.testing.assert_array_equal(query, old['query_obs_ids'])
                            np.testing.assert_array_equal(support, old['support_obs_ids'])
                            np.testing.assert_array_equal(scores.numpy(), old['baseline_-1'])
                        replays.append(dict(task=task, queries=len(query), max_score_difference=0.))
                    arm = objective+'_'+route if gate is not None else 'baseline'
                    rows.append(dict(arm=arm, seed=seed, task=task, port=qm.port.iloc[0], **metrics(pred, truth),
                                     rescue=int(((raw_pred != truth)&(pred == truth)).sum()), harm=int(((raw_pred == truth)&(pred != truth)).sum())))
                    for cls in np.unique(truth):
                        classes.append(dict(arm=arm, seed=seed, task=task, class_id=int(cls), support_count=int((y == cls).sum()),
                                            recall=float(100*(pred[truth == cls] == cls).mean())))
                    target_pairs.append(dict(arm=arm, seed=seed, task=task, **pair_metrics(scores, qm, paired_query)))
                    archive[f'{arm}_{seed}'] = scores.numpy()
        np.savez_compressed(OUT/f'{task}.npz', **archive)
        print('TARGET', task, flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'target_classes.csv', index=False)
    pd.DataFrame(target_pairs).to_csv(OUT/'target_pair_diagnosis.csv', index=False)
    pd.DataFrame(replays).to_csv(OUT/'baseline_replay.csv', index=False)
    for name, group in [('target_summary', ['arm']), ('target_seed_summary', ['arm', 'seed']), ('target_port_summary', ['arm', 'port'])]:
        data.groupby(group)[['OA', 'BA', 'rescue', 'harm']].mean().reset_index().to_csv(OUT/f'{name}.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-started, gate_fits=4, target_rows=len(data), baseline_replays=len(replays), resources='CPU1', parent_modified=False))
    print('SUMMARY', data.groupby('arm')[['OA', 'BA']].mean().to_string(), flush=True)
    print('SOURCE_PAIRS', pd.DataFrame(source_pairs).query('group == "test_source"').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
