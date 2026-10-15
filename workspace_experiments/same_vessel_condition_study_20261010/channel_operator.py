"""Bounded CPU channel intervention before a frozen nonlinear SAR residual block."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scene_training import sufficient, fit_head, sample_weights, metrics

BASE = Path(__file__).resolve().parent
OUT = BASE/'channel_operator_20261010'
PREVIOUS = BASE/'scene_training_alignment_20261010'
CACHE = Path('E:/Hermes/paper_tracking/sensitivity/neural_complement_pilot_20261007')
SEED = 20261010
PORTS = ('Santos', 'Port Klang', 'Los Angeles')
EPOCHS = 3
BATCH = 16
CHANNELS = 64


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def loading_to_channels(model, mode):
    """Exact derivative of source-standardized pooled features to scene coordinates."""
    comp = model[mode+'_components'].astype(float)
    u = model[mode+'_directions']
    return comp.T@u / model[mode+'_scale'][:, None] / np.sqrt(len(comp))


def trace_existing():
    records = []
    for port in PORTS:
        task = 'heldport_'+port.replace(' ', '_')
        with np.load(PREVIOUS/f'{task}_models.npz') as model:
            for mode in ('r50', 'r18'):
                loading = loading_to_channels(model, mode)
                # Standard-deviation changes, not incomparable raw activation units.
                loading_std = loading*model[mode+'_scale'][:, None]
                comp = model[mode+'_components'].astype(float)
                w = model[mode+'_baseline_coef']
                w = w-w.mean(1, keepdims=True)
                class_loading = comp.T@w/np.sqrt(len(comp))
                scene = np.sum(loading_std**2, axis=1)
                decision = np.sum(class_loading**2, axis=1)
                for channel in range(len(scene)):
                    records.append(dict(port=port, mode=mode, channel=channel,
                        scene_direction_loading=scene[channel],
                        classifier_loading=decision[channel]))
    pd.DataFrame(records).to_csv(OUT/'pooled_channel_trace.csv', index=False)


def select_source(obs):
    target = obs.port.isin(PORTS)
    forbidden = set(obs.loc[target, 'mmsi'])
    source = obs.loc[~target & ~obs.mmsi.isin(forbidden)]
    rng = np.random.default_rng(SEED)
    selected = set()
    # Keep whole scene/class cells so shared changes can be estimated across classes.
    for _, pm in source.groupby('port'):
        cells = pm.groupby(['product', 'class_id']).mmsi.nunique()
        scenes = cells[cells.ge(3)].groupby(level=0).size()
        scenes = sorted(scenes[scenes.ge(2)].index)
        if not scenes:
            continue
        take = rng.choice(scenes, min(4, len(scenes)), replace=False)
        for _, cell in pm[pm['product'].isin(take)].groupby(['product', 'class_id']):
            unique = cell.drop_duplicates('mmsi')
            if len(unique) >= 3:
                selected.update(rng.choice(unique.index, min(4, len(unique)), replace=False).tolist())
    for _, cell in source.groupby('class_id'):
        unique = cell.drop_duplicates('mmsi')
        have = cell.loc[cell.index.intersection(list(selected))].mmsi.nunique()
        if have < 48:
            extra = unique.loc[~unique.index.isin(selected)]
            selected.update(rng.choice(extra.index, min(48-have, len(extra)), replace=False).tolist())
    train = np.array(sorted(selected), dtype=np.int64)
    test = np.flatnonzero(target.to_numpy())
    assert not set(obs.iloc[train].mmsi) & set(obs.iloc[test].mmsi)
    assert not set(obs.iloc[train]['product']) & set(obs.iloc[test]['product'])
    return train, test


def scene_residuals(values, meta):
    meta = meta.reset_index(drop=True)
    cells = {}
    for key, g in meta.groupby(['port', 'product', 'class_id']):
        if g.mmsi.nunique() >= 3:
            cells[key] = values[g.index].mean(0)
    reference = {}
    for (p, _, c), value in cells.items():
        reference.setdefault((p, c), []).append(value)
    groups = {}
    for (p, s, c), value in cells.items():
        ref = reference[p, c]
        if len(ref) >= 3:
            residual = value-(np.sum(ref, axis=0)-value)/(len(ref)-1)
            groups.setdefault((p, s), []).append(residual)
    return {k: np.stack(v) for k, v in groups.items() if len(v) >= 2}


def channel_ranking(values, meta):
    center, scale = values.mean(0), np.maximum(values.std(0), 1e-6)
    z = (values-center)/scale
    groups = scene_residuals(z, meta)
    ports = {}
    for (p, _), r in groups.items():
        cross = (r.sum(0)**2-(r*r).sum(0))/(len(r)*(len(r)-1))
        ports.setdefault(p, []).append(cross)
    if not ports:
        raise ValueError('No supported cross-class scene cells')
    signed = np.mean([np.mean(v, axis=0) for v in ports.values()], axis=0)
    between = np.stack([z[meta.class_id.to_numpy() == c].mean(0) for c in range(8)])
    semantic = between.var(0)
    score = np.maximum(signed, 0)/(semantic+np.median(semantic)+1e-8)
    return score, signed, semantic, dict(scenes=len(groups), ports=len(ports))


def build_blocks():
    import torch
    from torch import nn
    from torchvision.models.resnet import BasicBlock
    down = nn.Sequential(nn.Conv2d(256, 512, 1, stride=2, bias=False), nn.BatchNorm2d(512))
    blocks = nn.Sequential(BasicBlock(256, 512, stride=2, downsample=down), BasicBlock(512, 512))
    state = torch.load(CACHE/'initial_tail.pt', map_location='cpu', weights_only=True)
    state = {k[len('layer4.'):]: v for k, v in state.items() if k.startswith('layer4.')}
    blocks.load_state_dict(state, strict=True)
    blocks.eval().requires_grad_(False)
    return blocks


def extract_boundary(blocks, rows):
    import torch
    prefix = np.load(CACHE/'source_layer3.npy', mmap_mode='r')
    maps = np.lib.format.open_memmap(OUT/'boundary_maps.npy', mode='w+', dtype=np.float32,
                                    shape=(len(rows), 512, 4, 4))
    pooled = np.empty((len(rows), 512), np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for lo in range(0, len(rows), BATCH):
            x = torch.from_numpy(np.array(prefix[rows[lo:lo+BATCH]], dtype=np.float32))
            h = blocks[0](x)
            maps[lo:lo+len(x)] = h.numpy()
            pooled[lo:lo+len(x)] = blocks[1](h).mean((2, 3)).numpy()
            if lo % 512 == 0:
                print('BOUNDARY', lo, '/', len(rows), 'seconds', round(time.perf_counter()-started, 1), flush=True)
    maps.flush()
    original = np.load(CACHE/'frozen_replay/source_extra.npy', mmap_mode='r')
    ref = np.asarray(original[rows], dtype=np.float32)
    relative = float(np.linalg.norm(pooled-ref)/max(np.linalg.norm(ref), 1e-12))
    error = float(np.max(abs(pooled-ref)))
    save_json(OUT/'REPLAY.json', dict(relative_l2=relative, max_abs=error,
        rows=len(rows), reference='Original frozen-replay R18; all selected rows'))
    if relative > .001:
        raise ValueError('R18 weight/cache mismatch: cannot interpret interventions')
    del prefix, ref
    return maps, pooled


def forward_features(block, maps, indices, gate):
    import torch
    result = np.empty((len(indices), 512), np.float32)
    with torch.inference_mode():
        for lo in range(0, len(indices), BATCH):
            idx = indices[lo:lo+BATCH]
            x = torch.from_numpy(np.array(maps[idx]))
            result[lo:lo+len(idx)] = block(x*gate[None, :, None, None]).mean((2, 3)).numpy()
    return result


def fit_classifier(features, meta, center, scale):
    x = (features-center)/scale/np.sqrt(512)
    w, b = fit_head(sufficient(x, meta.class_id.to_numpy(), sample_weights(meta)), np.eye(512), np.eye(512))
    return w.astype(np.float32), b.astype(np.float32)


def scene_energy(scores, meta):
    # Cross-class shared residual mean energy; descriptive, not pure noise variance.
    scores = scores-scores.mean(1, keepdims=True)
    groups = scene_residuals(scores, meta)
    ports = {}
    for (p, _), r in groups.items():
        ports.setdefault(p, []).append(float(np.sum(r.mean(0)**2)))
    return float(np.mean([np.mean(v) for v in ports.values()]))


def train_gate(block, maps, train_meta, selected, center, scale, arm):
    import torch
    n = len(train_meta)
    indices = np.arange(n)
    gate = torch.ones(512, requires_grad=True)
    mask = torch.zeros(512)
    mask[selected] = 1.
    c, s = torch.from_numpy(center), torch.from_numpy(scale)
    labels = torch.from_numpy(train_meta.class_id.to_numpy(copy=True))
    target = 2*torch.eye(8)[labels]-1
    weights = torch.from_numpy(sample_weights(train_meta).astype(np.float32))
    history = []
    for epoch in range(EPOCHS):
        before = forward_features(block, maps, indices, gate)
        w, b = fit_classifier(before, train_meta, center, scale)
        wt, bt = torch.from_numpy(w), torch.from_numpy(b)
        order = np.random.default_rng(SEED+epoch).permutation(n)
        losses, max_grad = [], 0.
        for lo in range(0, n, BATCH):
            idx = order[lo:lo+BATCH]
            x = torch.from_numpy(np.array(maps[idx]))
            f = block(x*gate[None, :, None, None]).mean((2, 3))
            logits = ((f-c)/s/np.sqrt(512))@wt+bt
            per_row = (logits-target[idx]).square().mean(1)
            classification = (per_row*weights[idx]).sum()/weights[idx].sum()
            loss = classification+.05*gate[selected].square().mean()
            loss.backward()
            max_grad = max(max_grad, float(gate.grad.abs().max()))
            with torch.no_grad():
                gate.sub_(2.*gate.grad*mask).clamp_(.5, 1.)
            gate.grad = None
            losses.append(float(classification.detach()))
        history.append(dict(arm=arm, epoch=epoch+1, classification=float(np.mean(losses)),
            selected_gate_mean=float(gate[selected].mean().detach()),
            selected_gate_min=float(gate[selected].min().detach()), max_gate_gradient=max_grad))
        print('TRAIN', json.dumps(history[-1]), flush=True)
    return gate.detach(), history


def run():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Completed experiment must not be overwritten')
    started = time.perf_counter()
    obs = pd.read_csv(BASE/'observations.csv')
    train, test = select_source(obs)
    train_meta = obs.iloc[train].reset_index(drop=True)
    test_meta = obs.iloc[test].reset_index(drop=True)
    np.savez(OUT/'split.npz', train_obs_ids=train, test_obs_ids=test)
    save_json(OUT/'PROTOCOL.json', dict(seed=SEED, ports=list(PORTS),
        split='All three previous pilot ports jointly held out; not three independent LOPO fits',
        source_sampling='Up to4 rich scenes/port, up to4 distinct vessels/class/scene, minimum48/class where available',
        train=len(train), test=len(test), source_class_counts=train_meta.class_id.value_counts().sort_index().to_dict(),
        target_labels_for_fit=False, identity_overlap=0, scene_overlap=0,
        location='R18 layer4[1] input, after first residual block, before frozen convolutions/BN/ReLU/addition',
        recipe='e206 pre-episode R18 and original float16 layer3 cache; no new image encoding',
        selection='Positive distinct-class shared-scene diagonal, divided by source class-centroid variance plus its median; top64/512',
        training='Three alternating gate-gradient/source-ridge-refit epochs, batch16, gate LR2, shrink penalty.05, gate bounds[.5,1]',
        control='Fixed.75 selected gates, learned selected gates, three fixed random-channel draws with identical training',
        limits='Development operator pilot; historical pretraining exposure retained; not main64-support model or pure-noise identification',
        resources='CPU1 only; no GPU or parent-file modifications'))
    trace_existing()
    blocks = build_blocks()
    ids = np.concatenate([train, test])
    maps, baseline_features = extract_boundary(blocks, obs.iloc[ids].manifest_row.to_numpy())
    n = len(train)
    input_means = np.asarray(maps[:n]).mean((2, 3))
    score, scene, semantic, coverage = channel_ranking(input_means, train_meta)
    selected = np.argsort(score, kind='stable')[-CHANNELS:]
    pd.DataFrame(dict(channel=np.arange(512), selection_score=score,
        shared_scene_component=scene, class_centroid_variance=semantic,
        selected=np.isin(np.arange(512), selected))).to_csv(OUT/'internal_channel_ranking.csv', index=False)
    save_json(OUT/'CHANNEL_COVERAGE.json', coverage)
    center = baseline_features[:n].mean(0)
    scale = np.maximum(baseline_features[:n].std(0), 1e-3)
    arms = [('baseline', None, False), ('targeted_fixed', selected, False),
            ('targeted_learned', selected, True)]
    for seed in range(3):
        random = np.random.default_rng(SEED+100+seed).choice(512, CHANNELS, replace=False)
        arms.append((f'random_learned_{seed}', random, True))
    results, classes, history, sensitivities = [], [], [], []
    base_pred = None
    base_source_scores = None
    baseline_w = None
    for arm, channels, learn in arms:
        if learn:
            gate, records = train_gate(blocks[1], maps, train_meta, channels, center, scale, arm)
            history.extend(records)
        else:
            gate = torch.ones(512)
            if channels is not None:
                gate[channels] = .75
        features = baseline_features if channels is None else forward_features(blocks[1], maps, np.arange(len(ids)), gate)
        w, b = fit_classifier(features[:n], train_meta, center, scale)
        scores = ((features-center)/scale/np.sqrt(512))@w+b
        pred = scores[n:].argmax(1)
        if base_pred is None:
            base_pred, base_source_scores, baseline_w = pred.copy(), scores[:n].copy(), w.copy()
        energy = scene_energy(scores[:n], train_meta)
        means = np.stack([scores[:n][train_meta.class_id.to_numpy() == c].mean(0) for c in range(8)])
        means -= means.mean(0)
        sensitivities.append(dict(arm=arm, source_scene_score_energy=energy,
            source_centroid_score_energy=float((means*means).sum()),
            source_BA=metrics(scores[:n].argmax(1), train_meta.class_id.to_numpy())['BA'],
            source_score_relative_change=float(np.linalg.norm(scores[:n]-base_source_scores)/np.linalg.norm(base_source_scores)),
            effective_head_relative_change=float(np.linalg.norm(w-baseline_w)/np.linalg.norm(baseline_w)),
            gate_mean=float(gate.mean()), gate_min=float(gate.min())))
        for port in PORTS:
            q = test_meta.port.eq(port).to_numpy()
            y = test_meta.class_id.to_numpy()[q]
            before, after = base_pred[q] == y, pred[q] == y
            results.append(dict(arm=arm, port=port, n=int(q.sum()), **metrics(pred[q], y),
                rescue=int((~before & after).sum()), harm=int((before & ~after).sum())))
            for c in np.unique(y):
                cc = y == c
                classes.append(dict(arm=arm, port=port, class_id=int(c), n=int(cc.sum()),
                    recall=float(100*after[cc].mean()), rescue=int((~before & after & cc).sum()),
                    harm=int((before & ~after & cc).sum())))
        np.savez_compressed(OUT/f'{arm}.npz', predictions=pred, gate=gate.numpy(),
            selected_channels=np.array([] if channels is None else channels, dtype=int),
            coef=w, bias=b, center=center, scale=scale)
        pd.DataFrame(results).to_csv(OUT/'results.csv', index=False)
        pd.DataFrame(classes).to_csv(OUT/'by_class.csv', index=False)
        pd.DataFrame(history).to_csv(OUT/'training_history.csv', index=False)
        pd.DataFrame(sensitivities).to_csv(OUT/'sensitivity.csv', index=False)
        print('RESULT', arm, json.dumps(results[-3:]), flush=True)
        if channels is not None:
            del features
        gc.collect()
    save_json(OUT/'COMPLETE.json', dict(arms=[a[0] for a in arms],
        seconds=time.perf_counter()-started, train=len(train), test=len(test), cpu_threads=1,
        all_required_tasks_complete=True))


if __name__ == '__main__':
    run()
