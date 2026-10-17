"""Paired source-only training of fixed and instance-conditioned local evidence."""
from pathlib import Path
import importlib.util
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from local_evidence import EvidenceReader, retrieval_loss

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'oss1_joint_geometry_20261017'
sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location('prior_local_training', OLD/'train.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
spec2 = importlib.util.spec_from_file_location('prior_fixed_readout', HERE.parent/'all_factor_probe_20261017/probe_classifier.py')
probe = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(probe)
public = old.public
ARMS = ('uniform', 'static', 'dynamic')
ALPHA = 1.
STEPS = 200


def scale_quality(raw, positions, valid, train):
    observed = raw[train][(positions[train, :, 3] > 0) & valid[train, None]]
    mean, std = observed.mean(0), observed.std(0).clip(.05)
    normalized = (raw-mean)/std
    normalized *= (positions[:, :, 3] > 0)[:, :, None]
    normalized[~valid] = 0
    return normalized.astype(np.float32), mean, std


def forward(model, features, ids, gate_override=None, zero_quality=False):
    quality = features['quality'][ids]
    if zero_quality:
        quality = torch.zeros_like(quality)
    return model(features['body'][ids], features['tokens'][ids], features['positions'][ids], quality, gate_override)


def task_loss(model, features, task, labels):
    raw, weights, _ = forward(model, features, task['ids'])
    valid = features['valid'][task['ids']]
    probability, h = old.predict(raw, valid, task, ALPHA)
    classification = old.risk(probability, labels[task['q']])
    relation = retrieval_loss(h, labels[task['ids']], task['ns'], valid)
    return classification+.05*relation, classification, relation, weights


@torch.no_grad()
def evaluate_source(model, features, tasks, labels):
    output = []
    for task in tasks:
        loss, brier, relation, weights = task_loss(model, features, task, labels)
        output.append((float(loss), float(brier), float(relation), float(weights[:, -1].mean())))
    return np.mean(output, axis=0)


def fit(arm, seed, features, fit_tasks, val_tasks, labels, folder):
    torch.manual_seed(20261017+seed)
    model = EvidenceReader(arm)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(20261017+seed)
    history = []
    for step in range(STEPS+1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i in rng.choice(len(fit_tasks), min(2, len(fit_tasks)), replace=False):
                losses.append(task_loss(model, features, fit_tasks[i], labels)[0])
            loss = torch.stack(losses).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        if step in (0, 50, 100, 200):
            values = evaluate_source(model, features, val_tasks, labels)
            history.append(dict(step=step, source_objective=values[0], source_brier=values[1],
                source_retrieval_ce=values[2], source_null_weight=values[3],
                training_loss=np.nan if step == 0 else float(loss.detach()),
                gradient_norm=np.nan if step == 0 else float(norm)))
    torch.save(model.state_dict(), folder/'model.pt')
    pd.DataFrame(history).to_csv(folder/'training.csv', index=False)
    return model


def prepare_fold(args, manifest, quality):
    fold, cache, episode, prefix = args
    data = public.load_fold('oss1', fold, cache, episode, prefix, manifest,
        np.load(old.FULL/'oss1_c4.npy'), np.load(old.FULL/'oss1_zero.npy'))
    source = np.r_[data['train'], data['val']]
    g, d, c = [public.pca(raw, source) for raw in (data['raw']['G50'], data['raw']['D18'], data['c4'])]
    base, z = public.design_block(c, np.arange(len(c))), public.design_block((g, d), np.arange(len(g)))
    prior_cache = {}
    fit_episodes, val_episodes = old.make_episodes(data, 'fit'), old.make_episodes(data, 'selection')
    fit_tasks = [old.make_task(data, base, z, ep, prior_cache) for ep in fit_episodes]
    val_tasks = [old.make_task(data, base, z, ep, prior_cache) for ep in val_episodes]
    prior_cache[tuple(sorted(source))] = (base@public.fit_head(c, data['labels'], source),
        z@public.fit_head((g, d), data['labels'], source), public.class_map(z[:, :-1], data['labels'], data['train']))
    target_tasks = []
    for (origin, draw), support in data['support'].items():
        task = old.make_task(data, base, z, dict(q=data['query'], s=support, prior_rows=source,
            origin=origin, acquisition=str(data['query_frame'].acquisition.iloc[0])), prior_cache)
        task['draw'] = int(draw)
        target_tasks.append(task)
    cached = np.load(OLD/fold/'local_features.npz')
    np.testing.assert_array_equal(cached['sample_ids'], quality['sample_ids'])
    np.testing.assert_allclose(cached['joint_positions'], quality['positions'], atol=1e-6)
    rows = data['rows']
    valid = cached['valid'][rows] & quality['valid'][rows]
    normalized, mean, std = scale_quality(quality['quality'][rows], cached['joint_positions'][rows], valid, data['train'])
    features = dict(body=torch.tensor(cached['body'][rows]), tokens=torch.tensor(cached['joint_tokens'][rows]),
                    positions=torch.tensor(cached['joint_positions'][rows]), quality=torch.tensor(normalized),
                    valid=torch.tensor(valid))
    return data, features, fit_tasks, val_tasks, target_tasks, mean, std


@torch.no_grad()
def all_features(model, features, **kwargs):
    parts = [forward(model, features, np.arange(start, min(start+256, len(features['body']))), **kwargs)
             for start in range(0, len(features['body']), 256)]
    return tuple(torch.cat([p[i] for p in parts]) for i in range(3))


@torch.no_grad()
def score_model(model, data, features, tasks, arm, seed, folder):
    raw, weights, corrections = all_features(model, features)
    train = data['train'][features['valid'][data['train']].numpy()]
    source_mask = torch.cat([features['positions'][train, :, 3] > 0, torch.ones(len(train), 1, dtype=torch.bool)], 1)
    constant = (corrections[train]*source_mask).sum(0)/source_mask.sum(0).clamp_min(1)
    variants = [(arm, raw, weights)]
    if arm == 'dynamic':
        const_raw, const_weights, _ = all_features(model, features, gate_override=constant)
        zero_raw, zero_weights, _ = all_features(model, features, zero_quality=True)
        variants.extend([('dynamic_constant_gate', const_raw, const_weights), ('dynamic_no_quality', zero_raw, zero_weights)])
    np.savez_compressed(folder/'readouts.npz', global_rows=data['rows'], raw=raw.numpy(), weights=weights.numpy(),
                        correction=corrections.numpy(), source_mean_correction=constant.numpy())
    records, predictions = [], []
    for name, variant_raw, variant_weights in variants:
        for task in tasks:
            ids, q, ns = task['ids'], task['q'], task['ns']
            before = task['old'].numpy()
            tic = time.perf_counter()
            probability, h = old.predict(variant_raw[ids], features['valid'][ids], task, ALPHA)
            solve_time = time.perf_counter()-tic
            p, y = probability.numpy(), data['labels'][q]
            sy = data['labels'][task['s']]
            active = features['valid'][q].numpy() & (int(features['valid'][task['s']].sum()) >= 2)
            k0 = task['qs'].numpy()
            k1 = k0+ALPHA*(h[ns:]@h[:ns].T).numpy()
            k1[~active] = k0[~active]
            ap0, ap1 = probe.map_at_r(k0, y, sy), probe.map_at_r(k1, y, sy)
            result, baseline = old.scores(y, p), old.scores(y, before)
            macro = float(old.risk(probability, torch.tensor(y)))
            macro0 = float(old.risk(task['old'], torch.tensor(y)))
            result.update(fold=data['fold'], arm=name, seed=seed, origin=task['origin'], draw=task['draw'],
                query=len(q), valid_query=int(active.sum()), baseline_oa=baseline['oa'], baseline_ba=baseline['ba'],
                delta_oa=result['oa']-baseline['oa'], delta_ba=result['ba']-baseline['ba'],
                delta_brier=result['brier']-baseline['brier'], delta_nll=result['nll']-baseline['nll'],
                macro_brier=macro, baseline_macro_brier=macro0, kernel_risk_gain=macro0-macro,
                support_map_r=float(np.nanmean(ap1)), baseline_support_map_r=float(np.nanmean(ap0)),
                retrieval_eligible=int(np.isfinite(ap1).sum()), null_weight=float(variant_weights[q, -1].mean()),
                descriptor_energy=float(h[ns:].square().sum(1).mean()), cached_solve_seconds=solve_time,
                rescued=int(((before.argmax(1) != y) & (p.argmax(1) == y)).sum()),
                harmed=int(((before.argmax(1) == y) & (p.argmax(1) != y)).sum()))
            records.append(result)
            for j, index in enumerate(q):
                predictions.append(dict(fold=data['fold'], arm=name, seed=seed, origin=task['origin'], draw=task['draw'],
                    global_row=int(data['rows'][index]), truth=int(y[j]), baseline=int(before[j].argmax()),
                    prediction=int(p[j].argmax()), baseline_true_probability=float(before[j, y[j]]),
                    true_probability=float(p[j, y[j]]), baseline_ap=ap0[j], ap=ap1[j],
                    same_class_support=int((sy == y[j]).sum()), null_weight=float(variant_weights[index, -1]),
                    **{f'p{k}': float(p[j, k]) for k in range(8)}))
    return records, predictions


def main():
    torch.set_num_threads(1)
    quality = dict(np.load(HERE/'quality.npz'))
    manifest = pd.read_csv(old.MANIFEST, dtype={'mmsi': str})
    all_metrics, all_predictions, fits, replay = [], [], [], []
    with threadpool_limits(limits=1):
        for args in old.folds():
            fold = args[0]
            folder = HERE/fold
            folder.mkdir(exist_ok=True)
            data, features, fit_tasks, val_tasks, tasks, mean, std = prepare_fold(args, manifest, quality)
            labels = torch.tensor(data['labels'], dtype=torch.long)
            previous = pd.read_csv(old.FULL/'oss1'/f'{fold}_metrics.csv')
            for task in tasks:
                baseline = old.scores(data['labels'][task['q']], task['old'].numpy())
                row = previous[previous.arm.eq('c4_class_fixed') & previous.origin.eq(task['origin']) & previous.draw.eq(task['draw'])].iloc[0]
                error = max(abs(baseline['oa']-row.chip_oa), abs(baseline['ba']-row.chip_ba))
                if error > 1e-7:
                    raise AssertionError('Historical full baseline does not replay')
                replay.append(dict(fold=fold, origin=task['origin'], draw=task['draw'], error=error))
            (folder/'data_protocol.json').write_text(json.dumps(dict(fit_tasks=len(fit_tasks), selection_tasks=len(val_tasks),
                train=data['rows'][data['train']].tolist(), selection=data['rows'][data['val']].tolist(),
                quality_mean=mean.tolist(), quality_std=std.tolist(), alpha=ALPHA, terminal_step=STEPS), indent=2), encoding='utf-8')
            print('FOLD', fold, 'tasks', len(fit_tasks), len(val_tasks), flush=True)
            for seed in range(3):
                for arm in ARMS:
                    out = folder/f'{arm}_{seed}'
                    out.mkdir(exist_ok=True)
                    tic = time.perf_counter()
                    model = fit(arm, seed, features, fit_tasks, val_tasks, labels, out)
                    seconds = time.perf_counter()-tic
                    records, predictions = score_model(model, data, features, tasks, arm, seed, out)
                    all_metrics.extend(records)
                    all_predictions.extend(predictions)
                    fits.append(dict(fold=fold, arm=arm, seed=seed, seconds=seconds,
                                     trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
                    pd.DataFrame(all_metrics).to_csv(HERE/'metrics.csv', index=False)
                    pd.DataFrame(all_predictions).to_csv(HERE/'predictions.csv', index=False)
                    pd.DataFrame(fits).to_csv(HERE/'fits.csv', index=False)
                    print('FIT', fold, arm, seed, round(seconds, 2), flush=True)
    pd.DataFrame(replay).to_csv(HERE/'baseline_replay.csv', index=False)
    print(pd.DataFrame(all_metrics).groupby(['origin', 'arm'])[['oa', 'ba', 'delta_oa', 'delta_ba', 'kernel_risk_gain']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
