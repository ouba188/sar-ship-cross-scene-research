"""Source-only scene-direction training diagnostics and small held-port pilot."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parent
OUT = BASE/'scene_training_alignment_20261010'
SEED = 20261010
MODES = ('r50', 'r18', 'dual')
ARMS = ('baseline', 'half_test_only', 'half_refit', 'remove_test_only', 'remove_refit',
        'isotropic_regularized', 'scene_consistency', 'semantic_protected_consistency')
TAU = .1
RANK = 8


def sample_weights(meta):
    repeats = meta.groupby('mmsi').mmsi.transform('size').to_numpy()
    vessels = meta.groupby('class_id').mmsi.transform('nunique').to_numpy()
    return len(meta)/(meta.class_id.nunique()*repeats*vessels)


def sufficient(x, y, weights):
    x = np.asarray(x, np.float64)
    weights = np.asarray(weights, np.float64)
    target = 2*np.eye(8)[y]-1
    xm = np.average(x, axis=0, weights=weights)
    ym = np.average(target, axis=0, weights=weights)
    xc = x-xm
    gram = xc.T@(weights[:, None]*xc)
    rhs = xc.T@(weights[:, None]*(target-ym))
    return gram, rhs, xm, ym


def fit_head(stats, transform, regularizer):
    gram, rhs, xm, ym = stats
    lhs = transform.T@gram@transform+regularizer
    beta = np.linalg.solve(lhs, transform.T@rhs)
    coef = transform@beta
    bias = ym-xm@coef
    return coef, bias


def shared_scene_covariance(x, meta):
    """Average distinct-class cross-products; no held-out/query rows accepted."""
    meta = meta.reset_index(drop=True)
    cells = {}
    for key, g in meta.groupby(['port', 'product', 'class_id']):
        if g.mmsi.nunique() >= 3:
            cells[key] = x[g.index.to_numpy()].mean(0)
    reference = {}
    for (port, _, cls), value in cells.items():
        reference.setdefault((port, cls), []).append(value)
    sums = {k: np.sum(v, axis=0) for k, v in reference.items()}
    counts = {k: len(v) for k, v in reference.items()}
    grouped = {}
    for (port, scene, cls), value in cells.items():
        n = counts[port, cls]
        if n >= 3:
            residual = value-(sums[port, cls]-value)/(n-1)
            grouped.setdefault((port, scene), []).append(residual)
    ports = {}
    kept = 0
    for (port, _), residuals in grouped.items():
        if len(residuals) < 2:
            continue
        r = np.stack(residuals).astype(np.float64)
        total = r.sum(0)
        cross = (np.outer(total, total)-r.T@r)/(len(r)*(len(r)-1))
        ports.setdefault(port, []).append(cross)
        kept += 1
    covariance = np.mean([np.mean(v, axis=0) for v in ports.values()], axis=0)
    return (covariance+covariance.T)/2, dict(scenes=kept, ports=len(ports), cells=len(cells))


def directions(covariance, rank=RANK):
    values, vectors = np.linalg.eigh(covariance)
    ix = np.argsort(values)[::-1]
    positive = ix[values[ix] > 1e-10]
    take = positive[:rank]
    u, ev = vectors[:, take], values[take]
    return u, (u*ev)@u.T, dict(positive_rank=len(positive), used_rank=len(take),
        positive_energy_fraction=float(ev.sum()/values[positive].sum()))


def semantic_projector(x, y, weights):
    means = np.stack([np.average(x[y == c], axis=0, weights=weights[y == c]) for c in range(8)])
    means -= means.mean(0)
    _, s, vt = np.linalg.svd(means, full_matrices=False)
    basis = vt[s > s[0]*1e-7]
    return basis.T@basis


def penalties(gram, scene, semantic):
    d = len(gram)
    eye = np.eye(d)
    scale = TAU*np.trace(gram)
    perpendicular = eye-semantic
    protected = perpendicular@scene@perpendicular
    return dict(isotropic_regularized=eye+scale/d*eye,
                scene_consistency=eye+scale/max(np.trace(scene), 1e-12)*scene,
                semantic_protected_consistency=eye+scale/max(np.trace(protected), 1e-12)*protected)


def transform_cached(raw, rows, model, name):
    comp = model[name+'_components']
    low = np.empty((len(rows), len(comp)), np.float32)
    offset = model[name+'_pca_mean']@comp.T
    for lo in range(0, len(rows), 128):
        sl = slice(lo, min(lo+128, len(rows)))
        block = (np.asarray(raw[rows[sl]], np.float32)-model[name+'_center'])/model[name+'_scale']
        low[sl] = (block@comp.T-offset)/np.sqrt(len(comp))
    return low


def source_pca(raw, rows, train, rank):
    source = np.empty((len(train), raw.shape[1]), np.float32)
    for lo in range(0, len(train), 128):
        source[lo:lo+128] = raw[rows[train[lo:lo+128]]]
    center = source.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(source.std(0, dtype=np.float64).astype(np.float32), 1e-6)
    source -= center
    source /= scale
    pca = PCA(rank, svd_solver='randomized', iterated_power=3, random_state=SEED, copy=False)
    pca.fit(source)
    del source
    result = dict(center=center, scale=scale, pca_mean=pca.mean_, components=pca.components_)
    return result


def metrics(pred, y):
    recalls = [100*np.mean(pred[y == c] == c) for c in np.unique(y)]
    return dict(OA=float(100*np.mean(pred == y)), BA=float(np.mean(recalls)))


def run_task(obs, raw, task, train, test, fold=None):
    started = time.perf_counter()
    train_meta = obs.iloc[train].reset_index(drop=True)
    assert not set(train_meta.mmsi) & set(obs.iloc[test].mmsi)
    if fold is None:
        assert not set(train_meta['product']) & set(obs.iloc[test]['product'])
        assert not set(train_meta.port) & set(obs.iloc[test].port)
    rows = obs.manifest_row.to_numpy()
    if fold is not None:
        with np.load(BASE/f'fold_{fold}_models.npz') as z:
            model = {k: z[k] for k in z.files}
    else:
        model = {}
        for name, rank in [('r50', 128), ('r18', 64)]:
            fit = source_pca(raw[name], rows, train, rank)
            model.update({name+'_'+k: v for k, v in fit.items()})
            print(task, name, 'source-only PCA complete', flush=True)
    x = np.empty((len(obs), 192), np.float32)
    for name, sl in [('r50', slice(0, 128)), ('r18', slice(128, 192))]:
        x[:, sl] = transform_cached(raw[name], rows, model, name)
    full_cov, coverage = shared_scene_covariance(x[train], train_meta)
    weight = sample_weights(train_meta)
    y_train = train_meta.class_id.to_numpy()
    predictions, results, by_class, sensitivity, saved = [], [], [], [], {}
    # Labels of held-out rows enter only the evaluation below, not fitting/prediction.
    for mi, mode in enumerate(MODES):
        if mode == 'r50':
            xx, covariance = x[:, :128], full_cov[:128, :128]
        elif mode == 'r18':
            xx, covariance = x[:, 128:], full_cov[128:, 128:]
        else:
            xx, covariance = x/np.sqrt(2.), full_cov/2.
        xt, xq = xx[train], xx[test]
        stats = sufficient(xt, y_train, weight)
        u, scene, info = directions(covariance)
        semantic = semantic_projector(xt, y_train, weight)
        d, eye = xt.shape[1], np.eye(xt.shape[1])
        projector = u@u.T
        half, remove = eye-.5*projector, eye-projector
        base_w, base_b = fit_head(stats, eye, eye)
        raw_score = xq@base_w+base_b
        if fold is not None:
            old = xq@model[mode+'_head_coef'].T+model[mode+'_head_intercept']
            replay_error = float(np.max(abs(old-raw_score)))
            replay_disagree = int(np.sum(old.argmax(1) != raw_score.argmax(1)))
        else:
            replay_error, replay_disagree = None, None
        heads = dict(baseline=(base_w, base_b),
                     half_test_only=(half@base_w, base_b),
                     remove_test_only=(remove@base_w, base_b),
                     half_refit=fit_head(stats, half, eye),
                     remove_refit=fit_head(stats, remove, eye))
        for arm, regularizer in penalties(stats[0], scene, semantic).items():
            heads[arm] = fit_head(stats, eye, regularizer)
        saved.update({mode+'_scene_cov': scene, mode+'_directions': u, mode+'_semantic_projector': semantic})
        base_pred = raw_score.argmax(1)
        y_test = obs.iloc[test].class_id.to_numpy()
        class_means = np.stack([np.average(xt[y_train == c], axis=0, weights=weight[y_train == c]) for c in range(8)])
        class_means -= class_means.mean(0)
        for arm in ARMS:
            coef, bias = heads[arm]
            score = xq@coef+bias
            pred = score.argmax(1)
            before, after = base_pred == y_test, pred == y_test
            result = dict(task=task, split='identity' if fold is not None else 'held_port', mode=mode,
                          arm=arm, train=len(train), test=len(test), **metrics(pred, y_test))
            result['rescue'] = float(100*np.mean(~before & after))
            result['harm'] = float(100*np.mean(before & ~after))
            results.append(result)
            predictions.append(pred)
            for cls in np.unique(y_test):
                q = y_test == cls
                by_class.append(dict(task=task, mode=mode, arm=arm, class_id=int(cls), n=int(q.sum()),
                    recall=float(100*np.mean(after[q])), rescue=float(100*np.mean((~before & after)[q])),
                    harm=float(100*np.mean((before & ~after)[q]))))
            cc = coef-coef.mean(1, keepdims=True)
            train_score = xt@coef+bias
            sensitivity.append(dict(task=task, mode=mode, arm=arm,
                scene_score_energy=float(np.trace(cc.T@scene@cc)),
                class_centroid_score_energy=float(np.sum((class_means@cc)**2)),
                train_BA=metrics(train_score.argmax(1), y_train)['BA'],
                weighted_train_loss=float(np.average(np.mean((train_score-(2*np.eye(8)[y_train]-1))**2, axis=1), weights=weight))))
            saved[mode+'_'+arm+'_coef'], saved[mode+'_'+arm+'_bias'] = coef, bias
        info.update(task=task, mode=mode, scene_class_subspace_overlap=float(np.trace(u.T@semantic@u)/len(u.T)),
                    replay_max_error=replay_error, replay_prediction_disagreements=replay_disagree)
        (OUT/f'{task}_{mode}_directions.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    pd.DataFrame(results).to_csv(OUT/f'{task}_results.csv', index=False)
    pd.DataFrame(by_class).to_csv(OUT/f'{task}_by_class.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/f'{task}_sensitivity.csv', index=False)
    np.savez_compressed(OUT/f'{task}_predictions.npz', predictions=np.stack(predictions), test_obs_ids=test,
                        train_obs_ids=train, modes=np.array(MODES), arms=np.array(ARMS))
    saved.update({k: v for k, v in model.items() if any(k.endswith(s) for s in ('center', 'scale', 'pca_mean', 'components'))})
    np.savez_compressed(OUT/f'{task}_models.npz', **saved)
    receipt = dict(task=task, train=len(train), test=len(test), train_scene_coverage=coverage,
        query_labels_used_for_fit=False, identity_overlap=0,
        scene_overlap=len(set(train_meta['product']) & set(obs.iloc[test]['product'])),
        seconds=time.perf_counter()-started)
    (OUT/f'{task}_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)
    print(pd.DataFrame(results).pivot(index='arm', columns='mode', values='BA').round(3).to_string(), flush=True)
    del x
    gc.collect()


def run():
    OUT.mkdir(exist_ok=True)
    obs = pd.read_csv(BASE/'observations.csv')
    counts = obs.groupby(['port', 'product', 'class_id']).mmsi.nunique().reset_index(name='n')
    scene_classes = counts[counts.n.ge(3)].groupby(['port', 'product']).class_id.nunique()
    eligible_ports = obs.groupby('port').class_id.nunique().eq(8)
    ranking = scene_classes[scene_classes.ge(3)].groupby('port').size().sort_values(ascending=False)
    ports = [p for p in ranking.index if eligible_ports[p]][:3]
    protocol = dict(seed=SEED, ports=ports, port_selection='Top3 eight-class ports by number of scenes with >=3 classes each >=3 vessels; metadata only',
        tasks='3 existing identity folds +3 held-port pilots',
        shared_direction_estimate='Cross-products of distinct-class leave-scene-out within-port residual centroids; scenes equal within port, ports equal; positive spectral part top8',
        projection='0.5 shrink and1.0 removal, each test-only versus source-refit; mechanism controls only',
        consistency='Additional ridge penalty on source shared-scene score variance, trace =0.1*source Gram trace',
        protected='Project scene covariance away from source between-class-centroid span before penalty; not full semantic protection guarantee',
        isotropic_control='Same penalty trace, no scene directional information',
        query_labels='No query labels, query scene statistics, or target support used in fitting/prediction; labels only evaluate',
        source_fit='Held-port centering, scaling, PCA, nuisance directions, weights, heads all source-only; source identities overlapping target removed',
        exposure='Neural encoders frozen from historical caches; pretraining domain exposure not removed',
        limitations='Development pilot; not main64-label model, not causal identification, no best-arm selection or independent confirmation',
        resources='CPU1 batch128; no GPU/backbone training')
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    raw = dict(r50=np.load('E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/pooled_features.npy', mmap_mode='r'),
               r18=np.load('E:/Hermes/paper_tracking/sensitivity/neural_complement_pilot_20261007/frozen_replay/source_extra.npy', mmap_mode='r'))
    tasks = []
    for fold in range(3):
        task = f'identity_{fold}'
        run_task(obs, raw, task, np.flatnonzero(obs.fold.to_numpy() != fold),
                 np.flatnonzero(obs.fold.to_numpy() == fold), fold)
        tasks.append(task)
    for port in ports:
        task = 'heldport_'+port.replace(' ', '_')
        is_test = obs.port.eq(port)
        forbidden = set(obs.loc[is_test, 'mmsi'])
        train = np.flatnonzero((~is_test & ~obs.mmsi.isin(forbidden)).to_numpy())
        run_task(obs, raw, task, train, np.flatnonzero(is_test.to_numpy()))
        tasks.append(task)
    for suffix in ('results', 'by_class', 'sensitivity'):
        pd.concat([pd.read_csv(OUT/f'{t}_{suffix}.csv') for t in tasks], ignore_index=True).to_csv(OUT/f'all_{suffix}.csv', index=False)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(tasks=tasks, completed_tasks=len(tasks),
        arms=list(ARMS), modes=list(MODES), query_labels_used_for_fit=False), indent=2), encoding='utf-8')


if __name__ == '__main__':
    run()
