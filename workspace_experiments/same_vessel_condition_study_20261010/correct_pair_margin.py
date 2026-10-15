"""Label-anchored, scale-invariant same-vessel correction: local matched pilot."""

import copy
import time
import decision_condition_consistency as old
from decision_condition_consistency import np, pd, torch, F, BASE, CACHE, PREVIOUS
from decision_condition_consistency import FactorNet, EvidenceGate, SEEDS, SUPPORTS
from decision_condition_consistency import metrics, save_json, sample_weights, standardize
from decision_condition_consistency import same_ship_pairs, pair_weights, pair_metrics, readout_inputs, residual_head

OUT = BASE/'correct_pair_margin_20261011'
ARMS = ('point', 'paired', 'rewired')
MARGIN = .1
COEFFICIENT = .05
STEPS = 400


def normalized_margin(scores, labels):
    centered = scores-scores.mean(-1, keepdim=True)
    direction = F.normalize(centered, dim=-1, eps=1e-8)
    correct = direction.gather(-1, labels[..., None]).squeeze(-1)
    mask = F.one_hot(labels, scores.shape[-1]).bool()
    rival = direction.masked_fill(mask, -torch.inf).max(-1).values
    return correct-rival


def auxiliary_loss(scores, labels, arm):
    deficit = F.relu(MARGIN-normalized_margin(scores, labels))
    return deficit.mean() if arm == 'point' else deficit.max(1).values.mean()


def rewire(pairs, labels, ships):
    """Keep both endpoint multisets and labels; change identities where possible."""
    result = pairs.copy()
    for cls in np.unique(labels[pairs[:, 0]]):
        positions = np.flatnonzero(labels[pairs[:, 0]] == cls)
        right = pairs[positions, 1]
        candidates = [np.roll(right, k) for k in range(len(right))]
        counts = [np.count_nonzero(ships[pairs[positions, 0]] != ships[r]) for r in candidates]
        result[positions, 1] = candidates[int(np.argmax(counts))]
    return result


def train(h, context, labels, meta, roles, fixed, paired, pair_probability, scale, arm, seed):
    gate = EvidenceGate(fixed['basis'])
    optimizer = torch.optim.Adam(gate.parameters(), lr=.01)
    weight, bias = fixed['weight'], fixed['bias']
    target = 2*torch.eye(8)[labels]-1
    fit = roles['fit']
    probability = sample_weights(meta.iloc[fit].reset_index(drop=True))
    probability /= probability.sum()
    y, ships = labels.numpy(), meta.mmsi.to_numpy()
    best, best_score, best_step = copy.deepcopy(gate.state_dict()), -np.inf, 0
    history, rewired_count = [], 0

    def validate(step, losses=None):
        nonlocal best, best_score, best_step
        row = dict(arm=arm, seed=seed, step=step, **(losses or {}))
        bas = []
        with torch.no_grad():
            for group in ('val_new_ship', 'val_new_time'):
                ids = roles[group]
                score = gate(h[ids], context[ids])@weight+bias
                result = metrics(score.argmax(1).numpy(), y[ids])
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
        pairs = paired[rng.choice(len(paired), 32, replace=True, p=pair_probability)]
        if arm == 'rewired':
            pairs = rewire(pairs, y, ships)
            rewired_count += int((ships[pairs[:, 0]] != ships[pairs[:, 1]]).sum())
        score = gate(h[ids], context[ids])@weight+bias
        classification = (score-target[ids]).square().mean()
        pair_scores = (gate(h[pairs.ravel()], context[pairs.ravel()])@weight+bias).reshape(-1, 2, 8)
        auxiliary = auxiliary_loss(pair_scores, labels[pairs], arm)
        loss = classification+.01*gate.gains(context[ids]).square().mean()+COEFFICIENT*auxiliary/scale
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.)
        optimizer.step()
        if step % 20 == 0:
            validate(step, dict(classification=float(classification.detach()), auxiliary=float(auxiliary.detach()), loss=float(loss.detach())))
    torch.save(gate.state_dict(), OUT/f'{arm}_{seed}_last.pt')
    gate.load_state_dict(best)
    torch.save(gate.state_dict(), OUT/f'{arm}_{seed}.pt')
    selection = dict(arm=arm, seed=seed, selected_step=best_step, validation_BA=best_score,
                     rewired_cross_identity_fraction=rewired_count/(STEPS*32) if arm == 'rewired' else 0.)
    print('SELECTED', selection, flush=True)
    return gate.eval(), history, selection


def margin_metrics(scores, meta):
    labels = torch.tensor(meta.class_id.to_numpy(copy=True))
    margin = normalized_margin(scores, labels).numpy()
    pairs = same_ship_pairs(meta)
    result = dict(mean_margin=float(margin.mean()), margin_covered=float(100*(margin >= MARGIN).mean()))
    if len(pairs):
        w = pair_weights(meta, pairs)
        result.update(worst_pair_margin=float(w@margin[pairs].min(1)),
                      both_margin_covered=float(100*w@(margin[pairs].min(1) >= MARGIN)))
    return result


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    started = time.perf_counter()
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(old.OUT/'source_split.npz') as z:
        split = {k: z[k].copy() for k in z.files}
    roles = {k: split[k] for k in ('fit', 'val_new_ship', 'val_new_time', 'test_source')}
    fit = roles['fit']
    tm = meta.iloc[fit].reset_index(drop=True)
    pairs = same_ship_pairs(tm)
    probability_pair = pair_weights(tm, pairs)
    paired = fit[pairs]
    source_ids = np.unique(np.concatenate(list(roles.values())))
    assert not set(fit)&set(np.r_[roles['val_new_ship'], roles['val_new_time'], roles['test_source']])
    assert not set(tm.mmsi)&set(meta.iloc[np.r_[roles['val_new_ship'], roles['test_source']]].mmsi)
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], np.float32)
    norm = (raw-split['center'])/split['scale']
    fixed = torch.load(old.OUT/'fixed_classifier.pt', map_location='cpu', weights_only=True)
    h = (torch.from_numpy(raw)-fixed['center'])/fixed['scale']/np.sqrt(512)
    weight, bias = fixed['weight'], fixed['bias']
    labels = torch.tensor(meta.class_id.to_numpy(copy=True))
    prior = h@weight+bias
    deficits = F.relu(MARGIN-normalized_margin(prior[paired], labels[paired])).mean(1).numpy()
    scale = max(float(probability_pair@deficits), 1e-4)
    np.savez(OUT/'source_split.npz', **split)
    save_json(OUT/'PROTOCOL.json', dict(seeds=SEEDS, steps=STEPS, margin=MARGIN, coefficient=COEFFICIENT,
        common_auxiliary_scale=scale, normalization='Center scores per sample, divide by L2 norm (epsilon1e-8). True-class minus strongest-rival margin.',
        objectives='point: mean hinge(.1-margin) across identical pair endpoints; paired: worst endpoint hinge per true same-vessel pair; rewired: same worst loss, same endpoint multisets within every batch but class-preserving identity re-pairing.',
        pairing='Same-vessel different-product pairs, equal vessel sampling32 pairs, source supervised128 class/ship balanced. Rewire chooses circular shift with most distinct identities within class; unbreakable same-identity pairs retained and fraction reported.',
        fixed='Frozen B factor encoders, original classifier, +-10% gate capacity, original400step schedule, Adam.01, source pool1296, all source validations unchanged. No new target labels.',
        selection='Earliest highest source validation meanBA every20steps including0. Six new gate fits, old reference and JS checkpoints replayed without retraining. No target choice of coefficient or checkpoint.',
        paths='both and prior-only; same9 original64-support draws, ridge8. Scores and pair diagnostics saved. Local prototype, repeated development, not full model or physical causal claim.'))
    models, contexts, history, selections = {}, {}, [], []
    for seed in SEEDS:
        encoder = FactorNet()
        encoder.load_state_dict(torch.load(PREVIOUS/f'anchor_{seed}_factor.pt', map_location='cpu', weights_only=True)['model'])
        encoder.eval().requires_grad_(False)
        with torch.no_grad():
            c = torch.cat([encoder.condition(torch.from_numpy(norm[i:i+256])) for i in range(0, len(norm), 256)]).numpy()
        context = torch.from_numpy(standardize(c, fit).astype(np.float32))
        contexts[seed] = context
        for arm in ('reference', 'consistent'):
            gate = EvidenceGate(fixed['basis'])
            gate.load_state_dict(torch.load(old.OUT/f'{arm}_{seed}.pt', map_location='cpu', weights_only=True))
            models[arm, seed] = gate.eval()
        for arm in ARMS:
            gate, hist, selection = train(h, context, labels, meta, roles, fixed, paired, probability_pair, scale, arm, seed)
            models[arm, seed] = gate
            history.extend(hist)
            selections.append(selection)
    pd.DataFrame(history).to_csv(OUT/'training.csv', index=False)
    pd.DataFrame(selections).to_csv(OUT/'selection.csv', index=False)
    source_rows, source_pairs, replay = [], [], []
    with torch.no_grad():
        for (arm, seed), gate in models.items():
            for group, ids in roles.items():
                scores = gate(h[ids], contexts[seed][ids])@weight+bias
                mm = meta.iloc[ids].reset_index(drop=True)
                source_rows.append(dict(arm=arm, seed=seed, group=group,
                    **metrics(scores.argmax(1).numpy(), labels[ids].numpy()), **margin_metrics(scores, mm),
                    saturation_fraction=float((gate.gains(contexts[seed][ids]).abs() > .095).float().mean())))
                source_pairs.append(dict(arm=arm, seed=seed, group=group, **pair_metrics(scores, mm)))
                if group in ('fit', 'test_source'):
                    np.savez_compressed(OUT/f'{arm}_{seed}_{group}.npz', obs_ids=ids, scores=scores.numpy())
                    if arm in ('reference', 'consistent'):
                        with np.load(old.OUT/f'{arm}_{seed}_{group}_scores.npz') as z:
                            np.testing.assert_array_equal(scores.numpy(), z['scores'])
    pd.DataFrame(source_rows).to_csv(OUT/'source_results.csv', index=False)
    pd.DataFrame(source_pairs).to_csv(OUT/'source_pair_diagnosis.csv', index=False)
    rows, classes, target_pairs = [], [], []
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task = path.stem.removesuffix('_split')
        with np.load(path) as z:
            support, query, sy = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
        assert len(support) == 64
        assert not set(meta.iloc[support].mmsi)&set(meta.iloc[query].mmsi)
        assert not set(meta.iloc[np.r_[support, query]].mmsi)&set(meta.iloc[source_ids].mmsi)
        ids = np.r_[support, query]
        qm = meta.iloc[query].reset_index(drop=True)
        truth = qm.class_id.to_numpy()
        archive = dict(support_obs_ids=support, query_obs_ids=query)
        with torch.no_grad():
            for (arm, seed), gate in models.items():
                changed = gate(h[ids], contexts[seed][ids])
                for route in ('both', 'prior'):
                    phi, local_prior = readout_inputs(h[ids], changed, weight, bias, route)
                    beta, _, _ = residual_head(phi[:64], local_prior[:64], torch.tensor(sy), balanced=True)
                    scores = local_prior[64:]+phi[64:]@beta
                    key = f'{arm}_{route}_{seed}'
                    archive[key] = scores.numpy()
                    if arm in ('reference', 'consistent'):
                        with np.load(old.OUT/f'{task}.npz') as z:
                            np.testing.assert_array_equal(scores.numpy(), z[key])
                        replay.append(dict(task=task, arm=arm, route=route, seed=seed, max_error=0.))
                    pred = scores.argmax(1).numpy()
                    rows.append(dict(arm=arm, route=route, seed=seed, task=task, port=qm.port.iloc[0], **metrics(pred, truth)))
                    target_pairs.append(dict(arm=arm, route=route, seed=seed, task=task,
                        **pair_metrics(scores, qm), **margin_metrics(scores, qm)))
                    for cls in np.unique(truth):
                        classes.append(dict(arm=arm, route=route, seed=seed, task=task, class_id=int(cls),
                                            support_count=int((sy == cls).sum()), recall=float(100*(pred[truth == cls] == cls).mean())))
        np.savez_compressed(OUT/f'{task}.npz', **archive)
        print('TARGET', task, flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(OUT/'target_results.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'target_classes.csv', index=False)
    pd.DataFrame(target_pairs).to_csv(OUT/'target_pair_diagnosis.csv', index=False)
    pd.DataFrame(replay).to_csv(OUT/'replay.csv', index=False)
    for name, groups in [('summary', ['arm', 'route']), ('seed_summary', ['arm', 'route', 'seed']), ('port_summary', ['arm', 'route', 'port'])]:
        data.groupby(groups)[['OA', 'BA']].mean().reset_index().to_csv(OUT/f'target_{name}.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-started, new_gate_fits=6, target_rows=len(data), exact_target_replays=len(replay), parent_modified=False))
    print(data.groupby(['arm', 'route'])[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
