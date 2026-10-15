"""Fixed external Qs/Qd/incidence transfer; frozen neural features, source fits."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from external_knowledge_metadata import ROOT, OUT, TARGETS, write
from compact_fusion_source import SOURCE, EXTERNAL, CACHE, _expert_inputs
from conditional_invariant_directions import nuisance_penalized_basis
from stable_port_directions import fit_direction_bank
from latent_usage_attribution import span_basis
from protected_drift_activation import protected_drift_basis
from direct_incidence_conditioning import condition_features
from unified_head_expert import ExpertContext
from unified_head_core import score, combined, late
from unified_head_run import base_kernel

TASKS = ROOT/'learned_frequency_residual_20261009/tasks'
SOURCE_C4 = ROOT/'c4_base_trial_20261009/pooled_c4.npy'
SOURCE_ANGLE = ROOT.parent/'pixel_correction_review/chip_incidence_v1/incidence_by_instance.csv'
ARMS = ('original', 'matched_refit', 'drift_only', 'stable_only', 'stable_drift',
        'angle_only', 'joint', 'pca_rankmatched', 'pca_drift')
READOUTS = ('base', 'expert', 'fusion', 'unified')
CONFIG = {'matched_refit': (False, '', False), 'drift_only': (True, '', False),
          'stable_only': (False, 's', False), 'stable_drift': (True, 's', False),
          'angle_only': (False, '', True), 'joint': (True, 's', True),
          'pca_rankmatched': (False, 'p', False), 'pca_drift': (True, 'p', False)}


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (keys or z.files)}


def unpickle(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def fit_geometry(raw, labels, ports):
    center = raw.mean(0, dtype=np.float64)
    scale = raw.std(0, dtype=np.float64)+1e-6
    z = ((raw-center)/scale).astype(np.float32)
    bank = fit_direction_bank(z, labels, ports)
    candidate = nuisance_penalized_basis(z, labels, ports, bank['pairs'], bank['pooled_basis'])
    qs, spectrum = span_basis(candidate['basis'])
    cell_means, cell_labels = [], []
    for label in np.unique(labels):
        means = [z[(labels == label) & (ports == port)].mean(0, dtype=float)
                 for port in np.unique(ports)
                 if np.sum((labels == label) & (ports == port)) >= 5]
        if len(means) >= 3:
            cell_means.extend(means)
            cell_labels.extend([label]*len(means))
    drift = protected_drift_basis(np.asarray(cell_means), np.asarray(cell_labels), qs, rank=16)
    qd = drift['q_drift']
    ordinary = PCA(n_components=qs.shape[1], svd_solver='randomized', random_state=20261010).fit(z)
    qp = ordinary.components_.T.astype(float)
    result = dict(center=center, scale=scale, qs=qs, qd=qd, qp=qp,
                  pairs=bank['pairs'], spectrum=spectrum)
    for name, basis in [('s', qs), ('p', qp)]:
        values = z@basis
        result[name+'center'] = values.mean(0)
        result[name+'scale'] = values.std(0)+1e-6
    np.testing.assert_allclose(qs.T@qd, 0, atol=1e-8)
    result['diagnostics'] = dict(stable_rank=qs.shape[1], drift_rank=qd.shape[1],
        candidate_pairs=len(bank['pairs']), protected_overlap=drift['protected_overlap_max'],
        nuisance_energy_in_stable=drift['nuisance_energy_in_stable_fraction'],
        drift_selected_energy=drift['selected_energy_fraction'], pca_rank=qp.shape[1])
    return result


def coordinates(raw, model, geo, extra=None):
    branches = [name for name in ('s', 'p') if 'q'+name in geo]
    result = {k: [] for k in ('x', *branches, 'd')}
    for lo in range(0, len(raw), 2048):
        values = np.asarray(raw[lo:lo+2048])
        z = (values-geo['center'])/geo['scale']
        if extra is None:
            x = ((model['pca'].transform(values)-model['center'])/model['scale']).astype(np.float32)
        else:
            x = _expert_inputs(values, extra[lo:lo+len(values)], model)
        result['x'].append(x)
        for name in branches:
            result[name].append((z@geo['q'+name]-geo[name+'center'])/geo[name+'scale'])
        result['d'].append(z@geo['qd'])
    return {k: np.concatenate(v) for k, v in result.items()}


def maps(model, stats, geo):
    raw = geo['qd'].T*geo['scale'][None, :]
    pc = model['pca'].components_.astype(float)
    dx = raw@pc.T/model['scale']
    retained = np.zeros((len(raw), 192))
    retained[:, :128] = dx/model['expert_scale'][:128]/np.sqrt(192)
    residual = np.zeros((len(raw), 2560))
    residual[:, :2048] = (raw-(raw@pc.T)@pc)/stats['discarded_scale'][:2048]/np.sqrt(2560)
    return dx, retained, residual


def expert_drift_moment(raw, extra, model, stats, coordinates_e, ar, ac):
    moment = np.empty((len(raw), len(ar)))
    for lo in range(0, len(raw), 2048):
        hi = min(lo+2048, len(raw))
        blocks = []
        for values, pca in ((raw[lo:hi], model['pca']), (extra[lo:hi], model['pca_ssl'])):
            centered = np.asarray(values, float)-pca.mean_
            pc = pca.components_.astype(float)
            blocks.append(centered-(centered@pc.T)@pc)
        comp = (np.c_[blocks[0], blocks[1]]-stats['discarded_center'])/stats['discarded_scale']/np.sqrt(2560)
        moment[lo:hi] = .5*(coordinates_e['x'][lo:hi]/np.sqrt(192)@ar.T+comp@ac.T)
    return moment


def inputs(coordinates_, arm, dx, u, head, model):
    suppress, branch, angle = CONFIG[arm]
    x = coordinates_['x'].astype(float).copy()
    if suppress:
        correction = coordinates_['d']@dx
        if head == 'expert':
            x[:, :128] -= .5*correction/model['expert_scale'][:128]
        else:
            x -= .5*correction
    parts = [x]
    if branch:
        z = coordinates_[branch]
        parts.append(z*np.sqrt(x.shape[1]/z.shape[1]))
    if angle:
        parts.append(u)
    return np.column_stack(parts)


def source_fit(dataset):
    dest = OUT/dataset
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'SOURCE_READY.json').exists():
        print('REUSE SOURCE '+dataset, flush=True)
        return
    started = time.perf_counter()
    folder = TASKS/dataset
    info = json.loads((folder/'PREPARED.json').read_text())
    source_dir, joint_dir = EXTERNAL[dataset]
    rows = load(ROOT/source_dir/'source_split.npz', ['source_rows'])['source_rows']
    np.testing.assert_array_equal(rows, load(folder/'target.npz', ['source_rows'])['source_rows'])
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'port', 'class_id'])
    raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')[rows]
    labels, ports = frame.class_id.to_numpy()[rows], frame.port.to_numpy()[rows]
    geo = fit_geometry(raw, labels, ports)
    print('GEOMETRY '+dataset+' '+str(geo['diagnostics']), flush=True)
    base_model = unpickle(Path(info['model_path']))
    expert_model = unpickle(ROOT/joint_dir/'source_model.pkl')
    stats = load(ROOT/joint_dir/'deployment_complement.npz')
    np.testing.assert_array_equal(rows, stats['source_rows'])
    extra_path = ROOT/joint_dir/'joint/source_extra.npy'
    if extra_path.exists():
        extra = np.load(extra_path, mmap_mode='r')[rows]
    else:
        cached = load(CACHE/f'{Path(joint_dir).name}_source_extra.npz')
        np.testing.assert_array_equal(cached['rows'], rows)
        extra = cached['features']
    source_c4 = np.load(SOURCE_C4, mmap_mode='r')[rows]
    cb = coordinates(source_c4, base_model, geo)
    ce = coordinates(raw, expert_model, geo, extra)
    side = pd.read_csv(SOURCE_ANGLE, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    meta = frame.merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    angles = meta.incidence_deg.to_numpy()[rows]
    angles[meta.status.to_numpy()[rows] != 'ok'] = np.nan
    u = condition_features(angles)
    dx_b = maps(base_model, stats, geo)[0]
    dx_e = maps(expert_model, stats, geo)[0]
    heads = {}
    for arm in CONFIG:
        heads[arm] = {}
        for head, coords, model, dx in [('base', cb, base_model, dx_b), ('expert', ce, expert_model, dx_e)]:
            values = inputs(coords, arm, dx, u, head, model)
            fitted = RidgeClassifier(alpha=1., class_weight='balanced').fit(values, labels)
            np.testing.assert_array_equal(fitted.classes_, np.arange(8))
            heads[arm][head] = dict(coef=fitted.coef_.T, intercept=fitted.intercept_)
        print(f'SOURCE {dataset} {arm}: input-matched priors fitted', flush=True)
    with (dest/'source_fit.pkl').open('wb') as stream:
        pickle.dump(dict(geometry=geo, heads=heads, source_rows=rows), stream)
    write(dest/'SOURCE_READY.json', dict(dataset=dataset, source_rows=len(rows),
        **geo['diagnostics'], source_angle_coverage=float(np.isfinite(angles).mean()),
        source_only=True, target_query_labels_read=False, neural_training=False,
        source_prior='Base C4 inputs; expert deployed frozen-tail inputs; balanced ridge1; matched refit control',
        seconds=time.perf_counter()-started))


def predict(dataset):
    dest = OUT/dataset
    if (dest/'FROZEN.json').exists():
        print('REUSE TARGET '+dataset, flush=True)
        return
    started = time.perf_counter()
    folder = TASKS/dataset
    info = json.loads((folder/'PREPARED.json').read_text())
    fit = unpickle(dest/'source_fit.pkl')
    geo, heads = fit['geometry'], fit['heads']
    data = load(folder/'target.npz', ['rows', 'ids', 'support', 'support_labels', 'eval_masks',
                                    'phi', 'mean', 'conditions', 'prior', 'expert', 'source_rows'])
    np.testing.assert_array_equal(data['source_rows'], fit['source_rows'])
    saved = load(folder/'selective_embedding.npz', ['rows', 'embedding'])
    np.testing.assert_array_equal(saved['rows'], data['rows'])
    embedding = saved['embedding'].astype(float)
    del saved
    meta = pd.read_csv(OUT/f'{dataset}_incidence.csv', keep_default_na=False)
    np.testing.assert_array_equal(meta.sample_id.to_numpy(str), data['ids'])
    angles = pd.to_numeric(meta.incidence_deg, errors='coerce').to_numpy()
    valid = np.isfinite(angles)
    u = condition_features(angles)
    context = ExpertContext(folder)
    model_e, stats = context.model, context.stats
    model_b = unpickle(Path(info['model_path']))
    target = ROOT/TARGETS[dataset]
    raw = (np.load(target/'paired_r50.npy', mmap_mode='r') if dataset == 'f0921'
           else load(target/'features.npz', ['paired_r50'])['paired_r50'])
    extra = np.load(context.joint/'target_extra.npy', mmap_mode='r')
    c4 = np.load(ROOT/'cross_dataset_c4_20261009'/dataset/'pooled_c4.npy', mmap_mode='r')
    cb = coordinates(c4, model_b, geo)
    ce = coordinates(raw, model_e, geo, extra)
    dx_b = maps(model_b, stats, geo)[0]
    dx_e, ar, ac = maps(model_e, stats, geo)
    moment = expert_drift_moment(raw, extra, model_e, stats, ce, ar, ac)
    norm = .5*(ar@ar.T+ac@ac.T)
    core = cb['x'].astype(float)-.5*cb['d']@dx_b
    drift_phi = np.c_[(core-model_b['residual_center'])/model_b['residual_scale']/np.sqrt(128),
                      np.ones(len(core))]
    expected_phi = np.c_[(cb['x']-model_b['residual_center'])/model_b['residual_scale']/np.sqrt(128),
                         np.ones(len(core))]
    np.testing.assert_allclose(expected_phi, data['phi'], atol=1e-12)
    priors = {}
    for arm in CONFIG:
        priors[arm] = []
        for head, coords, model, dx in [('base', cb, model_b, dx_b), ('expert', ce, model_e, dx_e)]:
            h = heads[arm][head]
            priors[arm].append(inputs(coords, arm, dx, u, head, model)@h['coef']+h['intercept'])
    original = load(ROOT/'unified_head_20261010/targets'/dataset/dataset/'policies.npz')
    for key in ('ids', 'support', 'eval_masks'):
        np.testing.assert_array_equal(data[key], original[key])
    predictions = np.empty((5, len(ARMS), len(READOUTS), len(raw)), np.int8)
    checks, support_coverage = [], []
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        if len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('Support/query overlap')
        pb0 = data['prior']
        pe0, ke0 = context.target_components(data, support)
        kb0 = base_kernel(data, embedding, support, info['eta'])
        kb_d = kb0-data['phi']@data['phi'][support].T+drift_phi@drift_phi[support].T
        zd = ce['d']
        ke_d = ke0-.5*(zd@moment[support].T+moment@zd[support].T)+.25*zd@norm@zd[support].T
        frozen_scores = {}
        for ai, arm in enumerate(ARMS):
            if arm == 'original':
                kb, ke, pb, pe = kb0, ke0, pb0, pe0
            else:
                suppress, branch, angle = CONFIG[arm]
                kb = (kb_d if suppress else kb0).copy()
                ke = (ke_d if suppress else ke0).copy()
                pb, pe = priors[arm]
                if branch:
                    b, e = cb[branch], ce[branch]
                    kb += (b@b[support].T)/b.shape[1]
                    ke += .25*(e@e[support].T)/e.shape[1]
                if angle:
                    kb += (u@u[support].T)/2
                    ke += .25*(u@u[support].T)/2
            b = score(kb, pb, support, sy)[0]
            e = score(ke, pe, support, sy)[0]
            j = combined(kb, ke, pb, pe, support, sy)[0]
            if arm in ('angle_only', 'joint'):
                fallback = 'matched_refit' if arm == 'angle_only' else 'stable_drift'
                fb, fe, fj = frozen_scores[fallback]
                b[~valid], e[~valid], j[~valid] = fb[~valid], fe[~valid], fj[~valid]
            frozen_scores[arm] = (b, e, j)
            results = (b, e, late(b, e), j)
            for ri, values in enumerate(results):
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite scores')
                predictions[draw, ai, ri] = values.argmax(1)
            if arm == 'original':
                np.testing.assert_allclose(e, data['expert'][draw], atol=3e-5, rtol=1e-6)
                for ri, mode in enumerate(('base', 'expert', 'late_softmax_fixed', 'unified_fixed')):
                    ref = original['predictions'][draw, list(original['modes']).index(mode)]
                    np.testing.assert_array_equal(predictions[draw, ai, ri], ref)
                checks.append(dict(draw=draw, original_all_readouts_exact=True,
                                   expert_max_error=float(np.max(abs(e-data['expert'][draw])))))
        support_coverage.append(dict(draw=draw, support_valid=int(valid[support].sum()),
                                     query_valid=int((valid & ev).sum()), query_missing=int((~valid & ev).sum())))
        print(f'TARGET {dataset} draw{draw}: {len(ARMS)} arms, four readouts; original exact', flush=True)
        del frozen_scores
    np.savez_compressed(dest/'policies.npz', predictions=predictions, arms=np.array(ARMS),
        readouts=np.array(READOUTS), ids=data['ids'], rows=data['rows'], support=data['support'],
        support_labels=data['support_labels'], eval_masks=data['eval_masks'], angle_valid=valid)
    pd.DataFrame(support_coverage).to_csv(dest/'draw_coverage.csv', index=False)
    write(dest/'FROZEN.json', dict(dataset=dataset, rows=len(raw), draws=5, arms=ARMS,
        readouts=READOUTS, target_query_labels_read=False, checks=checks,
        missing_angle_policy='Exact matching non-angle scores for missing queries; zero condition basis for missing support metadata',
        seconds=time.perf_counter()-started))


def report():
    if not all((OUT/t/'FROZEN.json').exists() for t in TARGETS):
        raise ValueError('Freeze both external datasets before reading query labels')
    records, classes = [], []
    for dataset, directory in TARGETS.items():
        p = load(OUT/dataset/'policies.npz')
        frame = pd.read_csv(ROOT/directory/'target_manifest.csv', usecols=['sample_id', 'class_id'])
        np.testing.assert_array_equal(frame.sample_id.to_numpy(str), p['ids'])
        truth = frame.class_id.to_numpy()
        np.testing.assert_array_equal(truth[p['support']], p['support_labels'])
        for draw, ev in enumerate(p['eval_masks']):
            for subset, subset_mask in [('all', np.ones(len(truth), bool)),
                                        ('angle_supported', p['angle_valid']), ('angle_missing', ~p['angle_valid'])]:
                mask = ev & subset_mask
                if not mask.any():
                    continue
                y = truth[mask]
                present = np.unique(y)
                for ai, arm in enumerate(ARMS):
                    for ri, readout in enumerate(READOUTS):
                        pred = p['predictions'][draw, ai, ri, mask]
                        ref = p['predictions'][draw, 0, ri, mask]
                        oa = 100*np.mean(pred == y)
                        ba = np.mean([100*np.mean(pred[y == c] == c) for c in present])
                        records.append(dict(dataset=dataset, draw=draw, subset=subset, arm=arm,
                            readout=readout, n=len(y), classes=len(present), oa=oa, ba=ba,
                            rescues=int(np.sum((pred == y) & (ref != y))),
                            harms=int(np.sum((pred != y) & (ref == y)))))
                        for c in present:
                            classes.append(dict(dataset=dataset, draw=draw, subset=subset, arm=arm,
                                readout=readout, class_id=int(c), n=int((y == c).sum()),
                                recall=100*np.mean(pred[y == c] == c)))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    summary = metrics.groupby(['dataset', 'subset', 'arm', 'readout'])[['oa', 'ba', 'rescues', 'harms']].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    print(summary[(summary.subset == 'all') & summary.readout.isin(['fusion', 'unified'])].to_string(index=False), flush=True)
    write(OUT/'COMPLETE.json', dict(datasets=list(TARGETS), draws=5, arms=ARMS, readouts=READOUTS,
        all_policies_frozen_before_query_readout=True, target_selection=False, cpu_threads=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['source', 'target', 'report', 'all'], default='all')
    parser.add_argument('--datasets', nargs='+', choices=list(TARGETS), default=list(TARGETS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT/'PROTOCOL.json').exists():
        write(OUT/'PROTOCOL.json', dict(arms=ARMS, readouts=READOUTS,
            stable='Equal-port class contrasts, source nuisance shrinkage1, SVD span, source normalization',
            drift='Source within-class port means outside Qs; rank<=16; attenuation0.5',
            stable_kernel_weights=dict(base=1., expert=.25), incidence_kernel_weights=dict(base=.5, expert=.125),
            frequency_weight=.25, source_ridge=1., support_ridge=8., fusion_weight=.5,
            incidence='t=(degrees-38)/5,t^2; source and target same transform; no heading',
            prior='All new arms refit balanced source heads on corresponding deployment inputs',
            frozen='Backbones, expert tail, PCA/complement scales, conditional mean kernel and trained frequency embeddings',
            pca_control='Ordinary rankmatched standardized raw source PCA; same normalization/weights/prior fitting',
            cohort='Exact original source_split and latest recrop target IDs; five original64 supports',
            target_query_labels='Read only after all target arms frozen; never select parameters',
            limits='Development data, inherited F0921 SSL exposure; source first-moment candidates not causal invariants',
            common_core24='Not run here; external integration first', no_gpu=True))
    with threadpool_limits(limits=2):
        if args.stage in ('source', 'all'):
            for dataset in args.datasets:
                source_fit(dataset)
                gc.collect()
        if args.stage in ('target', 'all'):
            for dataset in args.datasets:
                predict(dataset)
                gc.collect()
        if args.stage in ('report', 'all'):
            report()


if __name__ == '__main__':
    main()
