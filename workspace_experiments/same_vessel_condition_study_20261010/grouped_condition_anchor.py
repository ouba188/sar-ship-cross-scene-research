"""Paired content swapping and source metadata anchors, with matched controls."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import relative_factor_separation as previous
from relative_factor_separation import (
    BASE, CACHE, FactorNet, relation_pools, sample_triplets, angular_gap,
    sample_weights, scene_cells, sample_scenes, group_center, confusion_loss,
    encode_two, diagnose, standardize, train_gate, SEEDS, metrics, save_json,
    ship_average,
)

PREVIOUS = previous.OUT
OUT = BASE/'grouped_condition_anchor_20261011'
ARMS = ('current', 'swap', 'anchor', 'joint')
STEPS = 400


def metadata_targets(meta, fit):
    angle = meta.incidence_deg.to_numpy(float)
    valid = np.isfinite(angle)
    center = float(angle[fit][valid[fit]].mean())
    scale = float(angle[fit][valid[fit]].std())
    standardized = np.where(valid, (angle-center)/scale, 0).astype(np.float32)
    orbit = meta.pass_name.map({'Ascending': 0, 'Descending': 1}).fillna(-1).to_numpy(np.int64)
    counts = np.bincount(orbit[fit][orbit[fit] >= 0], minlength=2)
    weights = counts.sum()/(2*counts)
    return (torch.from_numpy(standardized), torch.from_numpy(valid), torch.from_numpy(orbit),
            torch.tensor(weights, dtype=torch.float32), dict(angle_center=center, angle_scale=scale,
                angle_valid=int(valid[fit].sum()), orbit_valid=int((orbit[fit] >= 0).sum()),
                orbit_counts=counts.tolist(), inference_metadata_required=False))


def anchor_loss(pred, angle, valid, orbit, orbit_weights):
    losses = []
    if valid.any():
        losses.append(F.mse_loss(pred[valid, 0], angle[valid]))
    available = orbit >= 0
    if available.any():
        losses.append(F.cross_entropy(pred[available, 1:], orbit[available], weight=orbit_weights)/np.log(2.))
    return torch.stack(losses).mean() if losses else pred.sum()*0


def swap_inputs(stable, condition):
    # Each pair is [a,b]. Only the other acquisition provides content for the target.
    content = stable.reshape(-1, 2, stable.shape[1]).flip(1).reshape_as(stable)
    return torch.cat([content, condition], dim=1)


def train_factors(x, meta, anchor_data, arm, seed):
    torch.manual_seed(seed)
    model = FactorNet()
    delta_head = nn.Linear(32, 512, bias=False)
    adversary = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 8))
    # Allocate after the original heads so the control preserves its exact initialization.
    anchor_head = nn.Linear(32, 3)
    parameters = list(model.parameters())+list(delta_head.parameters())+list(anchor_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=.001)
    adv_optimizer = torch.optim.Adam(adversary.parameters(), lr=.001)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(meta.class_id.to_numpy(copy=True)).long()
    angle, valid, orbit, orbit_weights = anchor_data
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
        stable, condition, correction, _, reconstruction = model(xt[ix])
        classification = F.cross_entropy(model.classifier(stable), yt[ix])
        ids = np.r_[pair.ravel(), sem.ravel(), scene.ravel()]
        rs, rc, _, _, _ = model(xt[ids])
        ps = F.normalize(rs[:64], dim=1).reshape(32, 2, -1)
        invariant = (1-(ps[:, 0]*ps[:, 1]).sum(1)).mean()
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
        for p in adversary.parameters():
            p.requires_grad_(False)
        confusion = confusion_loss(adversary(condition_scene), cmask, cw)
        self_recon = F.mse_loss(reconstruction, xt[ix])
        cross_recon = F.mse_loss(model.decoder(swap_inputs(rs[:64], rc[:64])), xt[pair.ravel()])
        recon = .5*(self_recon+cross_recon) if arm in ('swap', 'joint') else self_recon
        anchor = anchor_loss(anchor_head(condition), angle[ix], valid[ix], orbit[ix], orbit_weights)
        loss = classification+.1*recon+.01*correction.square().mean()
        loss = loss+.2*invariant+.2*semantic+.2*scene_loss
        loss = loss+.2*delta_loss
        loss = loss+.1*confusion
        if arm in ('anchor', 'joint'):
            loss = loss+.1*anchor
        optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(parameters, 5.)
        optimizer.step()
        for p in adversary.parameters():
            p.requires_grad_(True)
        if step == 1 or step % 100 == 0:
            row = dict(arm=arm, seed=seed, step=step, classification=float(classification.detach()),
                       invariant=float(invariant.detach()), semantic=float(semantic.detach()),
                       scene=float(scene_loss.detach()), delta=float(delta_loss.detach()),
                       confusion=float(confusion.detach()), adversary=float(adv_loss.detach()),
                       self_reconstruction=float(self_recon.detach()), cross_reconstruction=float(cross_recon.detach()),
                       anchor=float(anchor.detach()), gradient=float(grad))
            history.append(row)
            print('FACTOR', row, flush=True)
    state = dict(model=model.state_dict(), delta_head=delta_head.state_dict(),
                 adversary=adversary.state_dict(), anchor_head=anchor_head.state_dict())
    if arm == 'current':
        old = torch.load(PREVIOUS/f'purified_{seed}_factor.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(value, old['model'][key]) for key, value in state['model'].items()), 'Control replay changed'
    torch.save(state, OUT/f'{arm}_{seed}_factor.pt')
    return model.eval(), anchor_head.eval(), history


def decoder_diagnose(model, values, norm, meta, fit, test, arm, seed):
    tm = meta.iloc[test].reset_index(drop=True)
    pairs, _ = relation_pools(tm)
    own = pairs.ravel()
    other = pairs[:, ::-1].ravel()
    ships = tm.iloc[own].mmsi.to_numpy()
    stable, condition = [torch.from_numpy(v[test]) for v in values]
    stable_mean, condition_mean = [torch.from_numpy(v[fit].mean(0)).expand(len(own), -1) for v in values]
    target = torch.from_numpy(norm[test][own])
    versions = dict(self=(stable[own], condition[own]), swap=(stable[other], condition[own]),
                    no_content=(stable_mean, condition[own]), no_condition=(stable[other], condition_mean))
    rows = []
    with torch.no_grad():
        for mode, (s, c) in versions.items():
            reconstructed = model.decoder(torch.cat([s, c], dim=1))
            errors = (reconstructed-target).square().mean(1).numpy()
            rows.append(dict(arm=arm, seed=seed, mode=mode, mse=ship_average(errors, ships),
                             observations=len(own), ships=int(pd.Series(ships).nunique())))
    return rows


def anchor_diagnose(head, condition, targets, stats, test, arm, seed):
    angle, valid, orbit, _, _ = targets
    with torch.no_grad():
        pred = head(torch.from_numpy(condition[test]))
    av = valid[test].numpy()
    pv = orbit[test].numpy() >= 0
    truth = orbit[test].numpy()[pv]
    error = pred[:, 0].numpy()[av]-angle[test].numpy()[av]
    result = metrics(pred[:, 1:].argmax(1).numpy()[pv], truth)
    return dict(arm=arm, seed=seed, head_trained=arm in ('anchor', 'joint'),
                angle_rmse_deg=float(np.sqrt(np.mean(error**2))*stats['angle_scale']),
                angle_skill=float(100*(1-np.mean(error**2)/np.mean(angle[test].numpy()[av]**2))),
                orbit_OA=result['OA'], orbit_BA=result['BA'], angle_n=int(av.sum()), orbit_n=int(pv.sum()))


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    start = time.perf_counter()
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(PREVIOUS/'source_split.npz') as z:
        split = {k: z[k].copy() for k in z.files}
    factor_fit, test = split['factor_fit'], split['test_source']
    roles = {k: split[k] for k in ('fit', 'val_new_ship', 'val_new_time', 'test_source')}
    source_ids = np.unique(np.r_[factor_fit, roles['val_new_ship'], roles['val_new_time'], test])
    assert not set(factor_fit)&set(np.r_[roles['val_new_ship'], roles['val_new_time'], test])
    assert not set(meta.iloc[factor_fit].mmsi)&set(meta.iloc[np.r_[roles['val_new_ship'], test]].mmsi)
    np.savez(OUT/'source_split.npz', **split)
    raw = np.array(np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')[meta.manifest_row], np.float32)
    norm = (raw-split['center'])/split['scale']
    fixed = torch.load(PREVIOUS/'fixed_classifier.pt', map_location='cpu', weights_only=True)
    h = (torch.from_numpy(raw)-fixed['center'])/fixed['scale']/np.sqrt(512)
    weight, bias, basis = [fixed[k] for k in ('weight', 'bias', 'basis')]
    torch.save(fixed, OUT/'fixed_classifier.pt')
    labels = torch.from_numpy(meta.class_id.to_numpy(copy=True)).long()
    targets = metadata_targets(meta, factor_fit)
    fit_targets = tuple(t[factor_fit] for t in targets[:3])+(targets[3],)
    save_json(OUT/'PROTOCOL.json', dict(arms=ARMS, seeds=SEEDS, steps=STEPS, metadata=targets[4],
        source_fit_chips=len(factor_fit), source_fit_ships=int(meta.iloc[factor_fit].mmsi.nunique()),
        swap='Half original self reconstruction replaced with paired same-ship cross reconstruction. Total coefficient remains .1. Each target uses content from the other acquisition, never itself.',
        anchor='Auxiliary linear32to3 predicts source-standardized incidence and binary pass. Missing metadata masked. Mean available-task loss with coefficient .1; pass class weights source-fit only.',
        unchanged='Original purified losses, architecture,400steps Adam.001,264parameter evidence gate and source-only checkpoint selection, original64 target supports, fixed source classifier and residual ridge8. CPU1.',
        controls='All heads allocated after original initialization; current encoder and predictions must reproduce prior purified arm exactly. Metadata labels only used in source training; no inference metadata input.',
        limits='Repeated-development data, not causal identification. Source identity holdout shares many scenes. No main-model modifications.'))
    histories, diagnoses, variants, gate_traces, choices, source_results, decoders, anchors = [], [], [], [], [], [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            model, anchor_head, trace = train_factors(norm[factor_fit], meta.iloc[factor_fit].reset_index(drop=True), fit_targets, arm, seed)
            histories.extend(trace)
            stable, condition = encode_two(model, norm)
            for factor, values in [('stable', stable), ('condition', condition)]:
                row = diagnose(values, norm, meta, factor_fit, test, arm, seed, factor)
                diagnoses.append(row)
                print('DIAG', row, flush=True)
            decoders.extend(decoder_diagnose(model, (stable, condition), norm, meta, factor_fit, test, arm, seed))
            anchors.append(anchor_diagnose(anchor_head, condition, targets, targets[4], test, arm, seed))
            context = torch.from_numpy(standardize(condition, roles['fit']).astype(np.float32))
            gate, trace, selection = train_gate(h, context, labels, meta, roles, weight, bias, basis, arm, seed)
            variants.append((arm, seed, gate, context))
            gate_traces.extend(trace)
            choices.append(selection)
            torch.save(gate.state_dict(), OUT/f'{arm}_{seed}_gate.pt')
            np.savez_compressed(OUT/f'{arm}_{seed}_source_factors.npz', obs_ids=source_ids, stable=stable[source_ids], condition=condition[source_ids])
            with torch.no_grad():
                pred = (gate(h[test], context[test])@weight+bias).argmax(1).numpy()
            source_results.append(dict(arm=arm, seed=seed, **metrics(pred, meta.iloc[test].class_id.to_numpy())))
            for name, rows in [('factor_training', histories), ('factor_diagnosis', diagnoses), ('gate_training', gate_traces),
                               ('selection', choices), ('source_gate_results', source_results), ('decoder_diagnosis', decoders), ('anchor_diagnosis', anchors)]:
                pd.DataFrame(rows).to_csv(OUT/f'{name}.csv', index=False)
    # Redirect only this process's output destination; no historical source files are changed.
    previous.OUT = OUT
    results, classes = previous.target_evaluate(meta, h, weight, bias, variants, source_ids)
    results.to_csv(OUT/'target_results.csv', index=False)
    classes.to_csv(OUT/'target_classes.csv', index=False)
    replay_rows = []
    for path in sorted(OUT.glob('*_draw*.npz')):
        with np.load(path) as now, np.load(PREVIOUS/path.name) as old:
            for key, old_key in [('baseline_-1', 'baseline_-1')]+[(f'current_{seed}', f'purified_{seed}') for seed in SEEDS]:
                np.testing.assert_array_equal(now['query_obs_ids'], old['query_obs_ids'])
                np.testing.assert_array_equal(now['support_obs_ids'], old['support_obs_ids'])
                np.testing.assert_array_equal(now[key], old[old_key])
                replay_rows.append(dict(task=path.stem, version=key, queries=len(now['query_obs_ids']), max_score_difference=0.))
    pd.DataFrame(replay_rows).to_csv(OUT/'control_replay.csv', index=False)
    for name, group in [('target_summary', ['arm']), ('target_seed_summary', ['arm', 'seed']), ('target_port_summary', ['arm', 'port'])]:
        results.groupby(group)[['OA', 'BA', 'rescue', 'harm']].mean().reset_index().to_csv(OUT/f'{name}.csv', index=False)
    pd.DataFrame(diagnoses).groupby(['arm', 'factor']).mean(numeric_only=True).reset_index().to_csv(OUT/'factor_summary.csv', index=False)
    save_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-start, factor_fits=8, gate_fits=8,
                                       target_rows=len(results), exact_replays=len(replay_rows), resources='CPU1'))
    print('SUMMARY', results.groupby('arm')[['OA', 'BA']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
