"""Matched source-only factor objectives and frozen-feature classification gates."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from relational_factors import (
    BASE, CACHE, FactorNet, relation_pools, sample_triplets, angular_gap,
    source_readout, design, sample_weights, metrics, save_json,
)
from conditional_evidence_gate import source_partitions, train_gate, SEEDS, SUPPORTS
from condition_state_link import standardize
from target_response_adapter import residual_head

OUT = BASE/'relative_factor_separation_20261010'
ARMS = ('relations', 'decoupled', 'delta', 'purified')
STEPS = 400


def scene_cells(meta):
    cells = []
    for _, group in meta.groupby('product'):
        if group.class_id.nunique() > 1:
            cells.append({int(c): g.index.to_numpy() for c, g in group.groupby('class_id')})
    return cells


def sample_scenes(cells, rng, count=8):
    ids, groups, masks, weights = [], [], [], []
    for j, cell_index in enumerate(rng.choice(len(cells), count, replace=True)):
        cell = cells[cell_index]
        mask = np.zeros(8, bool)
        mask[list(cell)] = True
        for c in sorted(cell):
            ids.extend(rng.choice(cell[c], 2, replace=True))
            groups.extend([j, j])
            masks.extend([mask, mask])
            weights.extend([1/(count*len(cell)*2)]*2)
    return (np.array(ids), torch.tensor(groups), torch.from_numpy(np.array(masks)),
            torch.tensor(weights, dtype=torch.float32))


def group_center(values, groups):
    means = torch.zeros((int(groups.max())+1, values.shape[1]), dtype=values.dtype)
    means.index_add_(0, groups, values)
    counts = torch.bincount(groups).to(values.dtype)
    return values-means[groups]/counts[groups, None]


def confusion_loss(logits, masks, weights):
    logp = F.log_softmax(logits.masked_fill(~masks, -1e4), dim=1)
    target = masks.to(logits.dtype)/masks.sum(1, keepdim=True)
    return (weights*((target*(-logp)).sum(1)-masks.sum(1).float().log())).sum()


def train_factors(x, meta, arm, seed):
    torch.manual_seed(seed)
    model = FactorNet()
    delta_head = nn.Linear(32, 512, bias=False)
    adversary = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 8))
    optimizer = torch.optim.Adam(list(model.parameters())+list(delta_head.parameters()), lr=.001)
    adv_optimizer = torch.optim.Adam(adversary.parameters(), lr=.001)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(meta.class_id.to_numpy(copy=True)).long()
    probability = sample_weights(meta)
    probability /= probability.sum()
    pairs, pools = relation_pools(meta)
    pair_ships = meta.iloc[pairs[:, 0]].mmsi.to_numpy()
    counts = pd.Series(pair_ships).value_counts()
    pair_weights = np.array([1/counts[s] for s in pair_ships])
    pair_weights /= pair_weights.sum()
    delta_scale = float(np.average(((x[pairs[:, 1]]-x[pairs[:, 0]])**2).mean(1), weights=pair_weights))
    cells = scene_cells(meta)
    history = []
    for step in range(1, STEPS+1):
        rng = np.random.default_rng(seed+step)
        ix = rng.choice(len(x), 96, replace=True, p=probability)
        pair = pairs[rng.choice(len(pairs), 32, replace=True, p=pair_weights)]
        sem, scene = [sample_triplets(pools[k], rng) for k in ('semantic', 'scene')]
        ci, cg, cmask, cw = sample_scenes(cells, rng)
        s, c, correction, corrected, reconstruction = model(xt[ix])
        stable_ce = F.cross_entropy(model.classifier(s), yt[ix])
        classification = (.5*(stable_ce+F.cross_entropy(model.classifier(corrected), yt[ix]))
                          if arm == 'relations' else stable_ce)
        ids = np.r_[pair.ravel(), sem.ravel(), scene.ravel()]
        rs, rc, _, rr, _ = model(xt[ids])
        ps = F.normalize(rs[:64], dim=1).reshape(32, 2, -1)
        invariant = (1-(ps[:, 0]*ps[:, 1]).sum(1)).mean()
        if arm == 'relations':
            pr = F.normalize(rr[:64], dim=1).reshape(32, 2, -1)
            invariant = .5*(invariant+(1-(pr[:, 0]*pr[:, 1]).sum(1)).mean())
        triples = np.arange(96).reshape(32, 3)
        semantic = F.relu(.2-angular_gap(rs[64:160], triples)).mean()
        scene_loss = F.relu(.2-angular_gap(rc[160:256], triples)).mean()
        pc = rc[:64].reshape(32, 2, -1)
        difference = delta_head(pc[:, 1]-pc[:, 0])
        delta_loss = F.mse_loss(difference, xt[pair[:, 1]]-xt[pair[:, 0]])/delta_scale
        condition_scene = group_center(model.condition(xt[ci]), cg)
        adv_optimizer.zero_grad()
        adv_logits = adversary(condition_scene.detach()).masked_fill(~cmask, -1e4)
        adv_loss = (cw*F.cross_entropy(adv_logits, yt[ci], reduction='none')).sum()
        adv_loss.backward()
        adv_optimizer.step()
        for parameter in adversary.parameters():
            parameter.requires_grad_(False)
        confusion = confusion_loss(adversary(condition_scene), cmask, cw)
        loss = classification+.1*F.mse_loss(reconstruction, xt[ix])+.01*correction.square().mean()
        loss = loss+.2*invariant+.2*semantic+.2*scene_loss
        if arm in ('delta', 'purified'):
            loss = loss+.2*delta_loss
        if arm == 'purified':
            loss = loss+.1*confusion
        optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(delta_head.parameters()), 5.)
        optimizer.step()
        for parameter in adversary.parameters():
            parameter.requires_grad_(True)
        if step == 1 or step % 100 == 0:
            row = dict(arm=arm, seed=seed, step=step, classification=float(classification.detach()),
                       invariant=float(invariant.detach()), semantic=float(semantic.detach()),
                       scene=float(scene_loss.detach()), delta=float(delta_loss.detach()),
                       confusion=float(confusion.detach()), adversary=float(adv_loss.detach()), gradient=float(grad))
            history.append(row)
            print('FACTOR', row, flush=True)
    torch.save(dict(model=model.state_dict(), delta_head=delta_head.state_dict(), adversary=adversary.state_dict()),
               OUT/f'{arm}_{seed}_factor.pt')
    return model.eval(), history


def encode_two(model, x):
    with torch.no_grad():
        stable, condition = [], []
        for lo in range(0, len(x), 256):
            a = torch.from_numpy(x[lo:lo+256])
            stable.append(model.stable(a).numpy())
            condition.append(model.condition(a).numpy())
    return np.concatenate(stable), np.concatenate(condition)


def matched_scene_triples(meta):
    rng = np.random.default_rng(41031)
    out = []
    y, scene, port, ship = [meta[k].to_numpy() for k in ('class_id', 'product', 'port', 'mmsi')]
    for a in range(len(meta)):
        positives = np.flatnonzero((scene == scene[a])&(y != y[a])&(ship != ship[a]))
        rng.shuffle(positives)
        for p in positives:
            negatives = np.flatnonzero((scene != scene[a])&(port == port[a])&(y == y[p])&
                                       (ship != ship[a])&(ship != ship[p]))
            if len(negatives):
                out.append((a, p, rng.choice(negatives)))
                break
    return np.array(out, dtype=int).reshape(-1, 3)


def ship_average(values, ships):
    return float(pd.DataFrame(dict(value=values, ship=ships)).groupby('ship').value.mean().mean())


def pair_delta_probe(train_values, test_values, train_x, test_x, tm, vm):
    tp, _ = relation_pools(tm)
    vp, _ = relation_pools(vm)
    ts = tm.iloc[tp[:, 0]].mmsi.to_numpy()
    vs = vm.iloc[vp[:, 0]].mmsi.to_numpy()
    n = pd.Series(ts).value_counts()
    weights = np.array([1/n[s] for s in ts])
    weights *= len(weights)/weights.sum()
    a = train_values[tp[:, 1]]-train_values[tp[:, 0]]
    b = test_values[vp[:, 1]]-test_values[vp[:, 0]]
    scale = np.maximum(np.sqrt(np.average(a*a, axis=0, weights=weights)), 1e-3)
    a, b = a/scale/np.sqrt(a.shape[1]), b/scale/np.sqrt(b.shape[1])
    target = train_x[tp[:, 1]]-train_x[tp[:, 0]]
    truth = test_x[vp[:, 1]]-test_x[vp[:, 0]]
    beta = np.linalg.solve(a.T@(weights[:, None]*a)+10*np.eye(a.shape[1]), a.T@(weights[:, None]*target))
    error = ship_average(((b@beta-truth)**2).mean(1), vs)
    zero = ship_average((truth**2).mean(1), vs)
    return dict(delta_skill=100*(1-error/zero), delta_mse=error, delta_zero_mse=zero)


def centered_scene_data(values, meta):
    xs, ys, masks, weights = [], [], [], []
    for _, group in meta.groupby('product'):
        classes = np.sort(group.class_id.unique())
        if len(classes) < 2:
            continue
        ids = group.index.to_numpy()
        # Equal class weighting matches the training scene batches; diagnostic labels only.
        center = np.mean([values[group.index[group.class_id.eq(c)]] .mean(0) for c in classes], axis=0)
        xs.append(values[ids]-center)
        ys.extend(group.class_id)
        mask = np.zeros(8, bool)
        mask[classes] = True
        masks.extend([mask]*len(ids))
        counts = group.class_id.value_counts()
        weights.extend([1/(len(classes)*counts[c]) for c in group.class_id])
    weights = np.array(weights, np.float32)
    weights /= weights.sum()
    return (np.concatenate(xs).astype(np.float32), np.array(ys), np.array(masks), weights)


def leakage_probe(train_values, test_values, tm, vm):
    tx, ty, tmask, tw = centered_scene_data(train_values, tm)
    vx, vy, vmask, vw = centered_scene_data(test_values, vm)
    scale = np.maximum(tx.std(0), 1e-3)
    tx, vx = torch.from_numpy(tx/scale), torch.from_numpy(vx/scale)
    ty, tmask, tw = torch.tensor(ty), torch.tensor(tmask), torch.tensor(tw)
    results = {}
    for kind in ('linear', 'nonlinear'):
        torch.manual_seed(771)
        model = (nn.Linear(tx.shape[1], 8) if kind == 'linear' else
                 nn.Sequential(nn.Linear(tx.shape[1], 32), nn.GELU(), nn.Linear(32, 8)))
        optimizer = torch.optim.Adam(model.parameters(), lr=.005, weight_decay=.01)
        for _ in range(200):
            logits = model(tx).masked_fill(~tmask, -1e4)
            loss = (tw*F.cross_entropy(logits, ty, reduction='none')).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = model(vx).masked_fill(~torch.tensor(vmask), -1e4)
            pred = logits.argmax(1).numpy()
            nll = F.cross_entropy(logits, torch.tensor(vy), reduction='none').numpy()
        results[kind+'_scene_class_accuracy'] = float(100*np.sum(vw*(pred == vy)))
        results[kind+'_scene_class_nll_skill'] = float(np.sum(vw*(np.log(vmask.sum(1))-nll)))
    results['scene_class_chance'] = float(100*np.sum(vw/vmask.sum(1)))
    results['scene_class_chips'] = len(vy)
    return results


def diagnose(values, norm, meta, fit, test, arm, seed, factor):
    tm, vm = [meta.iloc[ix].reset_index(drop=True) for ix in (fit, test)]
    head = source_readout(values[fit], tm)
    with torch.no_grad():
        _, score = design(torch.from_numpy(values[test]), head)
    record = dict(arm=arm, seed=seed, factor=factor,
                  **metrics(score.argmax(1).numpy(), vm.class_id.to_numpy()))
    centered = values[test]-values[fit].mean(0)
    v = F.normalize(torch.from_numpy(centered), dim=1)
    pairs, pools = relation_pools(vm)
    distance = (1-(v[pairs[:, 0]]*v[pairs[:, 1]]).sum(1)).numpy()
    drift = ship_average(distance, vm.iloc[pairs[:, 0]].mmsi.to_numpy())
    semantic = sample_triplets(pools['semantic'], np.random.default_rng(41032), 1024)
    same_class_dist = (1-(v[semantic[:, 0]]*v[semantic[:, 1]]).sum(1)).numpy().mean()
    scene = matched_scene_triples(vm)
    record.update(centered_same_ship_distance=drift, same_class_other_ship_distance=float(same_class_dist),
                  drift_ratio=float(drift/same_class_dist),
                  semantic_triplet_accuracy=float(100*(angular_gap(v, semantic).numpy() > 0).mean()),
                  matched_scene_accuracy=float(100*(angular_gap(v, scene).numpy() > 0).mean()),
                  matched_scene_triplets=len(scene))
    valid_t = np.isfinite(tm.incidence_deg.to_numpy(float))
    valid_v = np.isfinite(vm.incidence_deg.to_numpy(float))
    a = values[fit][valid_t]
    b = values[test][valid_v]
    mean, std = a.mean(0), np.maximum(a.std(0), 1e-3)
    a = np.c_[(a-mean)/std/np.sqrt(a.shape[1]), np.ones(len(a))]
    b = np.c_[(b-mean)/std/np.sqrt(b.shape[1]), np.ones(len(b))]
    y, truth = tm.incidence_deg.to_numpy(float)[valid_t], vm.incidence_deg.to_numpy(float)[valid_v]
    penalty = np.eye(a.shape[1])*10
    penalty[-1, -1] = 0
    coef = np.linalg.solve(a.T@a+penalty, a.T@y)
    record['incidence_skill'] = float(100*(1-np.mean((b@coef-truth)**2)/np.mean((y.mean()-truth)**2)))
    if factor == 'condition':
        record.update(pair_delta_probe(values[fit], values[test], norm[fit], norm[test], tm, vm))
        record.update(leakage_probe(values[fit], values[test], tm, vm))
    unseen = ~vm['product'].isin(tm['product'])
    record['unseen_scene_chips'] = int(unseen.sum())
    if unseen.any():
        record.update({'unseen_scene_'+k: val for k, val in
                       metrics(score.argmax(1).numpy()[unseen], vm.class_id.to_numpy()[unseen]).items()})
    return record


def target_evaluate(meta, h, weight, bias, variants, source_ids):
    records, classes = [], []
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task = path.stem.removesuffix('_split')
        with np.load(path) as z:
            support, query, y = [z[k] for k in ('support_obs_ids', 'query_obs_ids', 'support_labels')]
        assert len(support) == 64
        assert not set(meta.iloc[support].mmsi)&set(meta.iloc[query].mmsi)
        assert not set(meta.iloc[np.r_[support, query]].mmsi)&set(meta.iloc[source_ids].mmsi)
        ids = np.r_[support, query]
        truth = meta.iloc[query].class_id.to_numpy()
        archive = dict(support_obs_ids=support, query_obs_ids=query)
        baseline = None
        with torch.no_grad():
            for arm, seed, model, context in [('baseline', -1, None, None)]+variants:
                changed = h[ids] if model is None else model(h[ids], context[ids])
                prior = changed@weight+bias
                phi = torch.cat([changed, torch.ones((len(ids), 1))], 1)
                beta, _, _ = residual_head(phi[:64], prior[:64], torch.tensor(y), balanced=True)
                score = prior[64:]+phi[64:]@beta
                pred = score.argmax(1).numpy()
                if model is None:
                    baseline = pred
                row = dict(arm=arm, seed=seed, task=task, port=meta.iloc[query].port.iloc[0],
                           **metrics(pred, truth), rescue=int(((baseline != truth)&(pred == truth)).sum()),
                           harm=int(((baseline == truth)&(pred != truth)).sum()))
                records.append(row)
                for cls in np.unique(truth):
                    classes.append(dict(arm=arm, seed=seed, task=task, class_id=int(cls),
                                        support_count=int((y == cls).sum()),
                                        recall=float(100*(pred[truth == cls] == cls).mean())))
                archive[f'{arm}_{seed}'] = score.numpy()
        np.savez_compressed(OUT/f'{task}.npz', **archive)
        print('TARGET', task, flush=True)
    return pd.DataFrame(records), pd.DataFrame(classes)


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    start = time.perf_counter()
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(BASE/'relational_factors_20261010/source_split.npz') as z:
        source_ids = np.r_[z['train_obs_ids'], z['diagnostic_obs_ids']]
    folds = {k: source_ids[meta.iloc[source_ids].fold.to_numpy() == k] for k in (0, 1, 2)}
    roles = source_partitions(meta, folds[2])
    roles['test_source'] = folds[0]
    factor_fit = np.sort(np.r_[folds[1], roles['fit']])
    assert not set(factor_fit)&set(np.r_[roles['val_new_ship'], roles['val_new_time'], folds[0]])
    assert not set(meta.iloc[factor_fit].mmsi)&set(meta.iloc[np.r_[roles['val_new_ship'], folds[0]]].mmsi)
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], np.float32)
    center, scale = raw[factor_fit].mean(0), np.maximum(raw[factor_fit].std(0), 1e-3)
    norm = (raw-center)/scale
    classifier = source_readout(raw[folds[1]], meta.iloc[folds[1]].reset_index(drop=True))
    rc, rs, weight, bias = classifier
    h = (torch.from_numpy(raw)-rc)/rs/np.sqrt(512)
    basis = torch.linalg.svd(weight, full_matrices=False).U
    labels = torch.from_numpy(meta.class_id.to_numpy(copy=True)).long()
    np.savez(OUT/'source_split.npz', factor_fit=factor_fit, classifier=folds[1], center=center, scale=scale, **roles)
    torch.save(dict(center=rc, scale=rs, weight=weight, bias=bias, basis=basis), OUT/'fixed_classifier.pt')
    save_json(OUT/'PROTOCOL.json', dict(arms=ARMS, seeds=SEEDS, factor_steps=STEPS,
        factor_fit_chips=len(factor_fit), factor_fit_ships=int(meta.iloc[factor_fit].mmsi.nunique()),
        source_only=True, target_labels=64, query_selection=False,
        training='Fresh identical FactorNet initialization per seed; no historical condition weights. Factor fit fold1 plus gate-fit subset of fold2; fixed400 steps, Adam.001. Gate same prior experiment fixed264params and source-only selection.',
        objectives='relations: original; decoupled: stable-only CE and consistency, no corrected CE; delta: +.2 normalized same-ship signed feature-difference reconstruction; purified: +.1 within-scene class-uniform confusion against detached-trained nonlinear adversary. All retain .1reconstruction+.2same-ship+.2semantic+.2scene.',
        probes='Fresh global class ridge; class-balanced scene-centered linear/nonlinear probes with class masks, diagnostic labels only; same-port class-matched scene triplets; source-fit delta ridge and incidence regression; mean-centered stability.',
        limitations='Repeated-development source fold0/three targets, not independent confirmation; many train/test scenes shared; new-scene subset reported. Matching scene is only a proxy. Different encoder fit cohort from historical gates: paired new arms are primary. No main-model change.'))
    histories, gate_history, selections, diagnoses, variants, source_rows = [], [], [], [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            model, trace = train_factors(norm[factor_fit], meta.iloc[factor_fit].reset_index(drop=True), arm, seed)
            histories.extend(trace)
            stable, condition = encode_two(model, norm)
            for factor, values in [('stable', stable), ('condition', condition)]:
                row = diagnose(values, norm, meta, factor_fit, folds[0], arm, seed, factor)
                diagnoses.append(row)
                print('DIAG', row, flush=True)
            context = torch.from_numpy(standardize(condition, roles['fit']).astype(np.float32))
            gate, trace, selection = train_gate(h, context, labels, meta, roles, weight, bias, basis, arm, seed)
            variants.append((arm, seed, gate, context))
            gate_history.extend(trace)
            selections.append(selection)
            torch.save(gate.state_dict(), OUT/f'{arm}_{seed}_gate.pt')
            np.savez_compressed(OUT/f'{arm}_{seed}_source_factors.npz', obs_ids=source_ids,
                                stable=stable[source_ids], condition=condition[source_ids])
            with torch.no_grad():
                ids = folds[0]
                pred = (gate(h[ids], context[ids])@weight+bias).argmax(1).numpy()
            source_rows.append(dict(arm=arm, seed=seed, **metrics(pred, meta.iloc[ids].class_id.to_numpy())))
            for name, data in [('factor_training', histories), ('gate_training', gate_history),
                               ('selection', selections), ('factor_diagnosis', diagnoses), ('source_gate_results', source_rows)]:
                pd.DataFrame(data).to_csv(OUT/f'{name}.csv', index=False)
    results, classes = target_evaluate(meta, h, weight, bias, variants, source_ids)
    results.to_csv(OUT/'target_results.csv', index=False)
    classes.to_csv(OUT/'target_classes.csv', index=False)
    for name, group in [('target_summary', ['arm']), ('target_seed_summary', ['arm', 'seed']), ('target_port_summary', ['arm', 'port'])]:
        results.groupby(group)[['OA', 'BA', 'rescue', 'harm']].mean().reset_index().to_csv(OUT/f'{name}.csv', index=False)
    pd.DataFrame(diagnoses).groupby(['arm', 'factor']).mean(numeric_only=True).reset_index().to_csv(OUT/'factor_summary.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-start, factor_fits=8, gate_fits=8,
                                       target_rows=len(results), resources='CPU1', parent_files_modified=False))
    print('SUMMARY', results.groupby('arm')[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
