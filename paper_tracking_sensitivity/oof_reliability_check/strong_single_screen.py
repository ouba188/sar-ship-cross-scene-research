"""Source-selected paired-feature screening and matched rescue accounting."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from unified_head_core import coefficients

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
SOURCE = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
OUT = ROOT/'strong_single_sarclip_20261010'
CACHE = Path('D:/SAR_strong_single_cache_20261010')
TASKS = ROOT/'learned_frequency_residual_20261009/tasks'
ALPHAS = (1., 10., 100.)
LAMBDAS = (.1, 1., 8.)
SEED = 20261010


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def ba(y, pred):
    return 100*np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])


def normalize(raw):
    x = np.asarray(raw, dtype=np.float32)
    x = x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    if x.ndim == 3:
        x = x.reshape(len(x), -1)/np.sqrt(2.)
    return x


def geometry(x, rows):
    total = np.zeros(x.shape[1], dtype=np.float64)
    square = 0.
    for lo in range(0, len(rows), 1024):
        block = np.asarray(x[rows[lo:lo+1024]], dtype=np.float64)
        total += block.sum(0)
        square += np.sum(block*block)
    center = total/len(rows)
    scale = np.sqrt(max(square/len(rows)-np.sum(center*center), 1e-12))
    return center, scale


def source_heads(x, labels, rows, center, scale):
    """Weighted ridge with unpenalized intercept, without copying full float64 X."""
    counts = np.bincount(labels[rows], minlength=8)
    if not np.all(counts > 0):
        raise ValueError('An inner-source class is missing')
    weights = len(rows)/(8*counts)
    d = x.shape[1]
    gram = np.zeros((d, d))
    rhs = np.zeros((d, 8))
    sx, sy, sw = np.zeros(d), np.zeros(8), 0.
    for lo in range(0, len(rows), 1024):
        r = rows[lo:lo+1024]
        z = (np.asarray(x[r], dtype=np.float64)-center)/scale
        y = 2*np.eye(8)[labels[r]]-1
        w = weights[labels[r]]
        wz = z*w[:, None]
        gram += z.T@wz
        rhs += wz.T@y
        sx += wz.sum(0)
        sy += (y*w[:, None]).sum(0)
        sw += w.sum()
    gram -= np.outer(sx, sx)/sw
    rhs -= np.outer(sx, sy)/sw
    heads = {}
    for alpha in ALPHAS:
        matrix = gram.copy()
        matrix.flat[::d+1] += alpha
        coef = cho_solve(cho_factor(matrix, lower=True, check_finite=False), rhs,
                         check_finite=False)
        intercept = sy/sw-(sx/sw)@coef
        heads[alpha] = (coef, intercept)
    return heads


def validation_splits(frame, rows):
    subset = frame.iloc[rows]
    counts = subset.port.value_counts()
    eligible = sorted(counts[counts > 128].index)
    ports = [eligible[i] for i in np.linspace(0, len(eligible)-1, 3, dtype=int)]
    result = []
    for port in ports:
        val = rows[subset.port.to_numpy() == port]
        identities = set(frame.iloc[val].mmsi)
        train = rows[(subset.port.to_numpy() != port) & ~subset.mmsi.isin(identities).to_numpy()]
        assert not np.intersect1d(train, val).size
        assert not set(frame.iloc[train].mmsi).intersection(identities)
        rng = np.random.default_rng(SEED)
        supports = [rng.choice(len(val), 64, replace=False) for _ in range(3)]
        result.append((port, train, val, supports))
    return result


def select(x, frame, rows):
    labels = frame.class_id.to_numpy()
    records = []
    for port, train, val, supports in validation_splits(frame, rows):
        center, scale = geometry(x, train)
        heads = source_heads(x, labels, train, center, scale)
        z = (np.asarray(x[val], dtype=float)-center)/scale
        for ai, alpha in enumerate(ALPHAS):
            coef, intercept = heads[alpha]
            prior = z@coef+intercept
            for draw, s in enumerate(supports):
                kernel = z@z[s].T+1
                mask = np.ones(len(val), dtype=bool)
                mask[s] = False
                for strength in LAMBDAS:
                    a = coefficients(kernel[s], prior[s], labels[val[s]], strength)
                    pred = (prior+kernel@a).argmax(1)
                    records.append(dict(port=port, draw=draw, source_alpha=alpha,
                        residual_lambda=strength, ba=ba(labels[val][mask], pred[mask]),
                        oa=100*np.mean(labels[val][mask] == pred[mask]),
                        inner_train_rows=len(train), validation_rows=len(val)))
        print(f'  source validation {port} train={len(train)} val={len(val)}', flush=True)
    records = pd.DataFrame(records)
    means = records.groupby(['source_alpha', 'residual_lambda'], sort=True).ba.mean()
    alpha, strength = means.idxmax()
    return float(alpha), float(strength), records


def inputs(arm, dataset):
    if arm == 'sarclip':
        if not (OUT/'source_EXTRACTED.json').exists():
            raise FileNotFoundError('Source feature extraction is incomplete')
        source = normalize(np.load(CACHE/'source_sarclip_joint.npy', mmap_mode='r'))
        if dataset == 'self24ports':
            return source, source
        if not (OUT/f'{dataset}_EXTRACTED.json').exists():
            raise FileNotFoundError('Target feature extraction is incomplete')
        target = normalize(np.load(CACHE/f'{dataset}_sarclip_joint.npy', mmap_mode='r'))
    else:
        source = normalize(np.load(ROOT/'c4_base_trial_20261009/pooled_c4.npy', mmap_mode='r'))
        target = source if dataset == 'self24ports' else normalize(np.load(
            ROOT/f'cross_dataset_c4_20261009/{dataset}/pooled_c4.npy', mmap_mode='r'))
    return source, target


def predict(folder, only_arm=None):
    info = json.loads((folder/'PREPARED.json').read_text())
    dataset = info['dataset']
    task = folder.name
    dest = OUT/'targets'/dataset/task
    if (dest/'PREDICTED.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(folder/'target.npz') as raw:
        data = {k: raw[k] for k in ('source_rows', 'rows', 'ids', 'support', 'support_labels', 'eval_masks')}
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'class_id', 'port', 'mmsi'],
                        dtype={'mmsi': str}, keep_default_na=False)
    source_ids = np.load('D:/SAR_completion_cache_20261010/source_ids.npy')
    np.testing.assert_array_equal(source_ids, frame.instance_id.to_numpy(str))
    if dataset == 'self24ports':
        np.testing.assert_array_equal(source_ids[data['rows']], data['ids'])
    elif dataset == 'oss2':
        np.testing.assert_array_equal(np.load('D:/SAR_completion_cache_20261010/oss2_ids.npy')[data['rows']], data['ids'])
    elif dataset == 'f0921':
        target_ids = pd.read_csv(ROOT/'f0921_detector_recrop_20261007/target_manifest.csv',
                                 usecols=['sample_id']).sample_id.to_numpy(str)
        np.testing.assert_array_equal(target_ids[data['rows']], data['ids'])
    np.testing.assert_array_equal(data['support'].shape, (5, 64))
    assert all(not data['eval_masks'][i, s].any() for i, s in enumerate(data['support']))
    predictions = []
    for arm in ('sarclip', 'r50_c4_common_readout'):
        if only_arm is not None and arm != only_arm:
            continue
        if (dest/f'{arm}_predictions.npy').exists() and (dest/f'{arm}_selection.json').exists():
            continue
        started = time.perf_counter()
        source, target = inputs(arm, dataset)
        alpha, strength, validation = select(source, frame, data['source_rows'])
        validation.to_csv(dest/f'{arm}_source_validation.csv', index=False)
        center, scale = geometry(source, data['source_rows'])
        heads = source_heads(source, frame.class_id.to_numpy(), data['source_rows'], center, scale)
        coef, intercept = heads[alpha]
        z = (np.asarray(target[data['rows']], dtype=float)-center)/scale
        prior = z@coef+intercept
        preds = []
        for draw, support in enumerate(data['support']):
            ks = z[support]@z[support].T+1
            a = coefficients(ks, prior[support], data['support_labels'][draw], strength)
            p = np.empty(len(z), dtype=np.int8)
            for lo in range(0, len(z), 1024):
                hi = min(lo+1024, len(z))
                p[lo:hi] = (prior[lo:hi]+(z[lo:hi]@z[support].T+1)@a).argmax(1)
            preds.append(p)
        np.save(dest/f'{arm}_predictions.npy', np.stack(preds))
        np.savez(dest/f'{arm}_head.npz', center=center, scale=scale, coef=coef,
                 intercept=intercept, alpha=alpha, residual_lambda=strength)
        write(dest/f'{arm}_selection.json', dict(source_alpha=alpha, residual_lambda=strength,
            validation_ba=float(validation.query('source_alpha == @alpha and residual_lambda == @strength').ba.mean()),
            seconds=time.perf_counter()-started, input_dim=source.shape[1], target_labels_used=False))
        print(f'{dataset}/{task} {arm} alpha={alpha} lambda={strength} {time.perf_counter()-started:.1f}s', flush=True)
        del source, target, z, heads, prior
        gc.collect()
    if not all((dest/f'{arm}_predictions.npy').exists() and (dest/f'{arm}_selection.json').exists()
               for arm in ('sarclip', 'r50_c4_common_readout')):
        return
    predictions = [np.load(dest/f'{arm}_predictions.npy')
                   for arm in ('sarclip', 'r50_c4_common_readout')]
    np.savez_compressed(dest/'predictions.npz', predictions=np.stack(predictions, axis=1),
        modes=np.array(['sarclip', 'r50_c4_common_readout']), **data)
    write(dest/'PREDICTED.json', dict(dataset=dataset, task=task, support_draws=5, neural_seeds=0))


def report():
    records, flows, classes = [], [], []
    receipts = sorted((OUT/'targets').glob('*/*/PREDICTED.json'))
    expected = {('self24ports', p.name) for p in (TASKS/'self24ports').iterdir()
                if (p/'PREPARED.json').exists()} | {('oss2', 'oss2')}
    if (OUT/'f0921_EXTRACTED.json').exists():
        expected.add(('f0921', 'f0921'))
    actual = {(p.parent.parent.name, p.parent.name) for p in receipts}
    if actual != expected:
        raise RuntimeError(f'Complete predefined tasks before reporting: missing={expected-actual}, extra={actual-expected}')
    for receipt in receipts:
        dest = receipt.parent
        dataset, task = dest.parent.name, dest.name
        with np.load(dest/'predictions.npz') as p:
            data = {k: p[k] for k in p.files}
        path = (SOURCE/'manifest.csv' if dataset == 'self24ports' else
                ROOT/f'{"opensarship2_transfer_20261007" if dataset == "oss2" else "f0921_detector_recrop_20261007"}/target_manifest.csv')
        idcol = 'instance_id' if dataset == 'self24ports' else 'sample_id'
        frame = pd.read_csv(path, usecols=[idcol, 'class_id'])
        np.testing.assert_array_equal(frame[idcol].to_numpy(str)[data['rows']], data['ids'])
        y = frame.class_id.to_numpy()[data['rows']]
        np.testing.assert_array_equal(y[data['support']], data['support_labels'])
        with np.load(ROOT/'backbone_ablation_20261010/targets'/dataset/task/'predictions.npz') as p:
            for key in ('rows', 'ids', 'support', 'eval_masks'):
                np.testing.assert_array_equal(p[key], data[key])
            dual = p['predictions'][:, list(p['modes']).index('dual_refit_late')]
            r50old = p['predictions'][:, list(p['modes']).index('r50_late')]
        for draw, mask in enumerate(data['eval_masks']):
            modes = {name: data['predictions'][draw, i] for i, name in enumerate(data['modes'])}
            modes.update(current_dual=dual[draw], current_r50_only=r50old[draw])
            for arm, all_pred in modes.items():
                pred = all_pred[mask]
                truth = y[mask]
                records.append(dict(dataset=dataset, task=task, draw=draw, arm=arm,
                    n=len(truth), oa=100*np.mean(truth == pred), ba=ba(truth, pred)))
                for c in np.unique(truth):
                    cm = truth == c
                    classes.append(dict(dataset=dataset, task=task, draw=draw, arm=arm,
                        class_id=int(c), n=int(cm.sum()), recall=100*np.mean(pred[cm] == c)))
            for arm in data['modes']:
                pred = modes[arm]
                rescue = mask & (dual[draw] == y) & (pred != y)
                harm = mask & (dual[draw] != y) & (pred == y)
                net_ba = 100*np.mean([(rescue[y == c].sum()-harm[y == c].sum())/np.sum(mask & (y == c))
                                    for c in np.unique(y[mask])])
                delta = ba(y[mask], dual[draw][mask])-ba(y[mask], pred[mask])
                np.testing.assert_allclose(net_ba, delta, atol=1e-10)
                flows.append(dict(dataset=dataset, task=task, draw=draw, compared_single=arm,
                    dual_rescues=int(rescue.sum()), dual_harms=int(harm.sum()),
                    net_oa_pp=100*(rescue.sum()-harm.sum())/mask.sum(), net_ba_pp=net_ba,
                    count_unit='query exposures within this support draw'))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(flows).to_csv(OUT/'rescue_harm.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'per_class.csv', index=False)
    summary = metrics.groupby(['dataset', 'arm'])[['oa', 'ba']].mean()
    summary.to_csv(OUT/'summary.csv')
    write(OUT/'EVALUATED.json', dict(tasks=metrics[['dataset', 'task']].drop_duplicates().to_dict('records'),
        metric_rows=len(metrics), query_labels_used_for_selection=False,
        scope='Frozen backbone screening only; not completed strong-single training qualification'))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['predict', 'report'])
    parser.add_argument('--dataset', choices=['oss2', 'self24ports', 'f0921'], default='oss2')
    parser.add_argument('--task')
    parser.add_argument('--arm', choices=['sarclip', 'r50_c4_common_readout'])
    args = parser.parse_args()
    if args.action == 'report':
        report()
    elif args.dataset == 'self24ports':
        for folder in sorted((TASKS/'self24ports').iterdir()):
            if (folder/'PREPARED.json').exists() and (args.task is None or folder.name == args.task):
                predict(folder, args.arm)
    else:
        predict(TASKS/args.dataset, args.arm)
