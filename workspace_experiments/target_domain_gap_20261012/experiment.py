"""Matched target-domain experience probes, separate from production experiments."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
WS = OUT.parent/'visual_reliable_baseline'
SOURCE = WS/'artifacts/eight_class_adaptive_20260916'
PUBLIC = ROOT/'opensarship2_transfer_20261007'
TASKS = ('Los Angeles', 'Port Klang', 'Santos', 'OpenSARShip2')
SEEDS = (31415, 27182)
ARMS = ('S', 'S_U', 'S_T')
N_SOURCE = 38091


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def load_frames():
    source = pd.read_csv(SOURCE/'manifest.csv', keep_default_na=False,
                         dtype={'mmsi': str}, low_memory=False)
    source['row'] = np.arange(len(source))
    public = pd.read_csv(PUBLIC/'target_manifest.csv', keep_default_na=False,
                         dtype={'mmsi_or_ship_id': str})
    public = public.rename(columns={'mmsi_or_ship_id': 'mmsi', 'scene_id': 'product'})
    public['port'] = 'OpenSARShip2'
    public['row'] = np.arange(len(public))+len(source)
    public['instance_id'] = public.sample_id
    columns = ['row', 'instance_id', 'port', 'product', 'mmsi', 'class_id', 'class_name']
    frame = pd.concat([source[columns], public[columns]], ignore_index=True)
    valid = frame.mmsi.str.fullmatch(r'[0-9]{9}') & frame.mmsi.ne('000000000')
    conflict = frame.loc[valid].groupby('mmsi').class_id.nunique()
    frame['eligible'] = valid & ~frame.mmsi.isin(conflict[conflict > 1].index)
    return frame


def one_per_ship(frame, rng):
    order = rng.permutation(len(frame))
    return frame.iloc[order].drop_duplicates('mmsi')


def audit_split(frame, split):
    query = frame.iloc[split['query']]
    support = frame.iloc[split['support']]
    t = frame.iloc[split['target_add']]
    s = frame.iloc[split['source']]
    u = frame.iloc[split['source_add']]
    for name, rows in [('support', support), ('target_add', t), ('source', s), ('source_add', u)]:
        if set(rows.mmsi) & set(query.mmsi) or set(rows['product']) & set(query['product']):
            raise ValueError(f'{name} overlaps query identity or product')
    for a, b in [(support, t), (s, u), (s, support), (u, support), (s, t), (u, t)]:
        if set(a.mmsi) & set(b.mmsi):
            raise ValueError('Training/support identities overlap')
    np.testing.assert_array_equal(np.bincount(t.class_id, minlength=8),
                                  np.bincount(u.class_id, minlength=8))
    assert len(support) == 64 and len(t) >= 16
    return {name: dict(rows=len(rows), ships=int(rows.mmsi.nunique()),
                       products=int(rows['product'].nunique()),
                       class_counts=np.bincount(rows.class_id, minlength=8).tolist())
            for name, rows in [('source', s), ('source_add', u), ('target_add', t),
                               ('support', support), ('query', query)]}


def prepare():
    frame = load_frames()
    frame.to_csv(OUT/'manifest.csv', index=False)
    audits = []
    for task in TASKS:
        target = frame.loc[frame.port.eq(task) & frame.eligible]
        source = frame.loc[(frame.row < N_SOURCE) & frame.port.ne(task) & frame.eligible]
        all_target = frame.loc[frame.port.eq(task)]
        source = source.loc[~source.mmsi.isin(all_target.mmsi) &
                            ~source['product'].isin(all_target['product'])]
        if task == 'OpenSARShip2':
            with np.load(PUBLIC/'source_split.npz') as z:
                source = source.loc[source.row.isin(z['source_rows'])]
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            products = rng.permutation(sorted(target['product'].unique()))
            q_products = products[:max(2, int(np.ceil(.2*len(products))))]
            query = target.loc[target['product'].isin(q_products)]
            donor = target.loc[~target['product'].isin(q_products) & ~target.mmsi.isin(query.mmsi)]
            donor = one_per_ship(donor, rng)
            if len(donor) < 80:
                raise ValueError(f'Insufficient target donors for predeclared split: {task}/{seed} {len(donor)}')
            support = donor.iloc[:64]
            t = donor.iloc[64:320]
            needed = np.bincount(t.class_id, minlength=8)
            s_rows, u_rows = [], []
            for c in range(8):
                candidates = one_per_ship(source.loc[source.class_id.eq(c)], rng)
                if len(candidates) <= needed[c]:
                    raise ValueError(f'Insufficient matched source class {c}')
                u_rows.extend(candidates.iloc[:needed[c]].row)
                s_rows.extend(candidates.iloc[needed[c]:needed[c]+256].row)
            split = dict(source=np.array(s_rows, dtype=int), source_add=np.array(u_rows, dtype=int),
                         target_add=t.row.to_numpy(), support=support.row.to_numpy(), query=query.row.to_numpy())
            audit = audit_split(frame, split)
            folder = OUT/task/str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(folder/'split.npz', **split)
            write(folder/'SPLIT.json', audit)
            audits.append(dict(task=task, seed=seed, **audit))
            print('SPLIT', task, seed, {k:v['rows'] for k,v in audit.items()}, flush=True)
    write(OUT/'SPLITS_COMPLETE.json', audits)


def load_raw():
    source_base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    source_extra = np.load(ROOT/'neural_complement_pilot_20261007/frozen_replay/source_extra.npy')
    public_extra = np.load(OUT/'cache/public_initial_extra.npy')
    with np.load(PUBLIC/'features.npz') as z:
        f = pd.read_csv(PUBLIC/'target_manifest.csv', usecols=['sample_id'])
        np.testing.assert_array_equal(z['sample_ids'], f.sample_id.to_numpy(str))
        return dict(r50=np.concatenate([source_base, z['paired_r50']]).astype(np.float32),
                    r18=np.concatenate([source_extra, public_extra]).astype(np.float32))


def geometry(raw, train, rank):
    center = raw[train].mean(0)
    scale = np.maximum(raw[train].std(0), 1e-6)
    x = (raw[train]-center)/scale
    pca = PCA(n_components=rank, svd_solver='randomized', random_state=17, iterated_power=3).fit(x)
    low = pca.transform(x)
    low_center = low.mean(0)
    low_scale = np.maximum(low.std(0), 1e-6)*np.sqrt(rank)
    matrix = (pca.components_.T/low_scale).astype(np.float32)
    bias = ((-pca.mean_@pca.components_.T-low_center)/low_scale).astype(np.float32)
    return dict(center=center, scale=scale, matrix=matrix, bias=bias)


def project(raw, geo):
    return ((raw-geo['center'])/geo['scale'])@geo['matrix']+geo['bias']


def fit_head(x, labels, mass):
    counts = np.bincount(labels, minlength=8)
    w = 1/np.maximum(counts[labels], 1)
    w *= mass/w.sum()
    model = RidgeClassifier(alpha=1., solver='cholesky')
    model.fit(x, labels, sample_weight=w)
    np.testing.assert_array_equal(model.classes_, np.arange(8))
    return dict(coef=model.coef_.T.astype(np.float32), bias=model.intercept_.astype(np.float32))


def softmax(x):
    e = np.exp(x-x.max(1, keepdims=True))
    return e/e.sum(1, keepdims=True)


def residual(x, prior, support_positions, labels):
    phi = np.c_[x, np.ones(len(x))]
    ps = phi[support_positions]
    counts = np.bincount(labels, minlength=8)
    weights = 1/np.maximum(counts[labels], 1)
    weights *= len(labels)/weights.sum()
    wanted = 2*np.eye(8)[labels]-1
    residual = (wanted-prior[support_positions])*(counts > 0)
    coefficient = np.linalg.solve(ps@ps.T+8*np.diag(1/weights), residual)
    return prior+(phi@ps.T)@coefficient


def predictions(features, fit_positions, labels, support_positions, support_labels, mass):
    dual = np.c_[features['r50'], features['r18']]/np.sqrt(2.)
    modes = dict(r50=features['r50'], r18=features['r18'], dual=dual)
    output, heads = {}, {}
    for mode, x in modes.items():
        h = fit_head(x[fit_positions], labels, mass)
        prior = x@h['coef']+h['bias']
        adapted = residual(x, prior, support_positions, support_labels)
        output[mode+'_prior'] = prior
        output[mode+'_adapted'] = adapted
        heads[mode] = h
    output['fusion'] = .5*softmax(output['r50_adapted'])+.5*softmax(output['dual_adapted'])
    return output, heads


def run_frozen():
    frame = pd.read_csv(OUT/'manifest.csv', dtype={'mmsi': str})
    y = frame.class_id.to_numpy()
    raw = load_raw()
    for task in TASKS:
        for seed in SEEDS:
            folder = OUT/task/str(seed)
            split = dict(np.load(folder/'split.npz'))
            audit_split(frame, split)
            use = np.unique(np.concatenate(list(split.values())))
            loc = pd.Index(use)
            geos, feat, exported = {}, {}, {}
            for mode, rank in [('r50', 128), ('r18', 64)]:
                geos[mode] = geometry(raw[mode], split['source'], rank)
                feat[mode] = project(raw[mode][use], geos[mode])
                exported.update({mode+'_'+k:v for k,v in geos[mode].items()})
            np.savez(folder/'geometry.npz', **exported)
            np.savez(folder/'features.npz', rows=use, **feat)
            for arm in ARMS:
                train = split['source'] if arm == 'S' else np.r_[split['source'], split['source_add' if arm == 'S_U' else 'target_add']]
                pred, heads = predictions(feat, loc.get_indexer(train), y[train],
                    loc.get_indexer(split['support']), y[split['support']], len(split['source']))
                query_pos = loc.get_indexer(split['query'])
                dest = folder/'frozen'/arm
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest/'predictions.npz', rows=split['query'],
                                    **{k:v[query_pos] for k,v in pred.items()})
                np.savez(dest/'heads.npz', **{m+'_'+k:v for m,h in heads.items() for k,v in h.items()})
            write(folder/'FROZEN_COMPLETE.json', dict(rows=len(use), arms=ARMS))
            print('FROZEN', task, seed, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'frozen'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        {'prepare': prepare, 'frozen': run_frozen}[args.stage]()


if __name__ == '__main__':
    main()
