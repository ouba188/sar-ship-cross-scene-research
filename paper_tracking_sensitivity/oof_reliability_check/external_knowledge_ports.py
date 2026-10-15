"""Matched 24-port common-core readout with fixed existing frequency features."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import gc
import json
import pickle
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from external_knowledge_run import (ROOT, OUT, SOURCE, SOURCE_C4, SOURCE_ANGLE, TASKS,
    load, unpickle, coordinates, inputs, maps, expert_drift_moment, CONFIG, READOUTS)
from external_knowledge_metadata import write
from direct_incidence_conditioning import condition_features
from unified_head_run import base_kernel
from unified_head_core import score, combined, late

PORT_ARMS = ('original', 'matched_refit', 'drift_only', 'stable_only', 'stable_drift', 'angle_only', 'joint')
PORT_OUT = OUT/'self24ports'


def common_expert_kernel(raw, extra, coordinates_e, model, stats, reference):
    def residual(values, pca):
        x = np.asarray(values, float)-pca.mean_
        pc = pca.components_.astype(float)
        return x-(x@pc.T)@pc
    def comp(ix):
        r = np.c_[residual(raw[ix], model['pca']), residual(extra[ix], model['pca_ssl'])]
        return (r-stats['discarded_center'])/stats['discarded_scale']/np.sqrt(2560)
    cref = comp(reference)
    retained = coordinates_e['x']/np.sqrt(192)
    answer = np.empty((len(raw), len(reference)))
    for lo in range(0, len(raw), 2048):
        hi = min(lo+2048, len(raw))
        answer[lo:hi] = .5*(retained[lo:hi]@retained[reference].T+comp(slice(lo, hi))@cref.T)+1
    return answer


def freeze_port(folder, frame, raw_all, c4_all, angles):
    port = folder.name
    dest = PORT_OUT/port
    if (dest/'FROZEN.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    parent = ROOT/'protected_drift_activation_20261008/ports'/port
    g = unpickle(parent/'geometry.pkl')
    model = dict(g['model'])
    model['residual_center'], model['residual_scale'] = g['base_center'], g['base_scale']
    geo = dict(center=g['feature_center'], scale=g['feature_scale'], qs=g['qs'], qd=g['qd'],
               scenter=g['stable_center'], sscale=g['stable_scale'])
    rows = np.load(parent/'source_rows.npy')
    source_path = ROOT/'evidence_aware_handoff_20261008/source_features'/port
    np.testing.assert_array_equal(rows, np.load(source_path/'rows.npy'))
    data = load(folder/'target.npz', ['rows', 'ids', 'support', 'support_labels', 'eval_masks',
                                    'phi', 'mean', 'conditions', 'prior', 'source_rows'])
    np.testing.assert_array_equal(rows, data['source_rows'])
    if np.intersect1d(rows, data['rows']).size:
        raise ValueError('Outer source/target overlap')
    labels = frame.class_id.to_numpy()[rows]
    if (frame.port.to_numpy()[rows] == port).any():
        raise ValueError('Held port in source')
    sx = np.load(source_path/'features.npy', mmap_mode='r')
    cb_source = coordinates(c4_all[rows], model, geo)
    ce_source = coordinates(raw_all[rows], model, geo, sx)
    u_source = condition_features(angles[rows])
    dx, ar, ac = maps(model, g['stats'], geo)
    heads = {}
    for arm in PORT_ARMS[1:]:
        heads[arm] = {}
        for head, coords in [('base', cb_source), ('expert', ce_source)]:
            values = inputs(coords, arm, dx, u_source, head, model)
            fitted = RidgeClassifier(alpha=1., class_weight='balanced').fit(values, labels)
            np.testing.assert_array_equal(fitted.classes_, np.arange(8))
            heads[arm][head] = dict(coef=fitted.coef_.T, intercept=fitted.intercept_)
    del cb_source, ce_source, sx, values
    with (dest/'source_heads.pkl').open('wb') as stream:
        pickle.dump(heads, stream)
    target_rows = data['rows']
    np.testing.assert_array_equal(target_rows, np.load(parent/'target_rows.npy'))
    np.testing.assert_array_equal(data['ids'], frame.instance_id.to_numpy(str)[target_rows])
    raw = raw_all[target_rows]
    extra = np.load(ROOT/'module_completion_20261008/ports_projection'/port/'matched_joint/joint_extra.npy', mmap_mode='r')
    cb = coordinates(c4_all[target_rows], model, geo)
    ce = coordinates(raw, model, geo, extra)
    u = condition_features(angles[target_rows])
    valid = np.isfinite(angles[target_rows])
    reference = np.unique(data['support'])
    ke_all = common_expert_kernel(raw, extra, ce, model, g['stats'], reference)
    moment = expert_drift_moment(raw, extra, model, g['stats'], ce, ar, ac)
    norm = .5*(ar@ar.T+ac@ac.T)
    phi = np.c_[(cb['x']-g['base_center'])/g['base_scale']/np.sqrt(128), np.ones(len(raw))]
    core = cb['x'].astype(float)-.5*cb['d']@dx
    dphi = np.c_[(core-g['base_center'])/g['base_scale']/np.sqrt(128), np.ones(len(raw))]
    saved = load(folder/'selective_embedding.npz', ['rows', 'embedding'])
    np.testing.assert_array_equal(saved['rows'], target_rows)
    embedding = saved['embedding'].astype(float)
    info = json.loads((folder/'PREPARED.json').read_text())
    original = load(ROOT/'unified_head_20261010/targets/self24ports'/port/'policies.npz')
    for key in ('ids', 'support', 'eval_masks'):
        np.testing.assert_array_equal(data[key], original[key])
    priors = {}
    for arm in PORT_ARMS[1:]:
        priors[arm] = []
        for head, coords in [('base', cb), ('expert', ce)]:
            h = heads[arm][head]
            priors[arm].append(inputs(coords, arm, dx, u, head, model)@h['coef']+h['intercept'])
    predictions = np.empty((5, len(PORT_ARMS), 4, len(raw)), np.int8)
    for draw, support in enumerate(data['support']):
        ev, sy = data['eval_masks'][draw], data['support_labels'][draw]
        if ev[support].any() or len(np.unique(support)) != 64:
            raise ValueError('Support/query overlap')
        for ri, mode in enumerate(('base', 'expert', 'late_softmax_fixed', 'unified_fixed')):
            predictions[draw, 0, ri] = original['predictions'][draw, list(original['modes']).index(mode)]
        old_kernel = base_kernel(data, embedding, support, info['eta'])
        # Remove the complete old phi term; retain the identical mean/frequency relations.
        unchanged = old_kernel-data['phi']@data['phi'][support].T
        kb0 = unchanged+phi@phi[support].T
        kbd = unchanged+dphi@dphi[support].T
        ke0 = ke_all[:, np.searchsorted(reference, support)]
        zd = ce['d']
        ked = ke0-.5*(zd@moment[support].T+moment@zd[support].T)+.25*zd@norm@zd[support].T
        for ai, arm in enumerate(PORT_ARMS[1:], 1):
            suppress, branch, angle = CONFIG[arm]
            kb, ke = (kbd if suppress else kb0).copy(), (ked if suppress else ke0).copy()
            pb, pe = priors[arm]
            if branch:
                kb += cb['s']@cb['s'][support].T/cb['s'].shape[1]
                ke += .25*ce['s']@ce['s'][support].T/ce['s'].shape[1]
            if angle:
                kb += u@u[support].T/2
                ke += .25*u@u[support].T/2
            b = score(kb, pb, support, sy)[0]
            e = score(ke, pe, support, sy)[0]
            j = combined(kb, ke, pb, pe, support, sy)[0]
            # Same missing-query policy as the external fixed integration.
            if arm == 'angle_only':
                b[~valid], e[~valid], j[~valid] = common_scores[0][~valid], common_scores[1][~valid], common_scores[2][~valid]
            elif arm == 'joint':
                b[~valid], e[~valid], j[~valid] = stable_scores[0][~valid], stable_scores[1][~valid], stable_scores[2][~valid]
            if arm == 'matched_refit':
                common_scores = b, e, j
            elif arm == 'stable_drift':
                stable_scores = b, e, j
            for ri, scores in enumerate((b, e, late(b, e), j)):
                if not np.isfinite(scores).all():
                    raise ValueError('Nonfinite port scores')
                predictions[draw, ai, ri] = scores.argmax(1)
    np.savez_compressed(dest/'policies.npz', predictions=predictions, arms=np.array(PORT_ARMS),
        readouts=np.array(READOUTS), ids=data['ids'], rows=target_rows, support=data['support'],
        support_labels=data['support_labels'], eval_masks=data['eval_masks'], angle_valid=valid,
        source_rows=rows)
    write(dest/'FROZEN.json', dict(port=port, source_rows=len(rows), target_rows=len(raw),
        target_query_labels_used=False, source_prior_refitted=True,
        source_angle_fraction=float(np.isfinite(angles[rows]).mean()), target_angle_fraction=float(valid.mean()),
        stable_rank=geo['qs'].shape[1], drift_rank=geo['qd'].shape[1],
        original='Existing unified_head original predictions copied with ID/support alignment, not a new replay',
        seconds=time.perf_counter()-started))
    print(f'PORT {port}: seven arms, five draws, four readouts frozen ({time.perf_counter()-started:.1f}s)', flush=True)


def report(frame, folders):
    if not all((PORT_OUT/f.name/'FROZEN.json').exists() for f in folders):
        raise ValueError('Freeze all23 port tasks before query evaluation')
    records, classes = [], []
    for folder in folders:
        p = load(PORT_OUT/folder.name/'policies.npz')
        y = frame.class_id.to_numpy()[p['rows']]
        np.testing.assert_array_equal(y[p['support']], p['support_labels'])
        for draw, ev in enumerate(p['eval_masks']):
            truth = y[ev]
            present = np.unique(truth)
            for ai, arm in enumerate(PORT_ARMS):
                for ri, readout in enumerate(READOUTS):
                    pred = p['predictions'][draw, ai, ri, ev]
                    ref = p['predictions'][draw, 1, ri, ev]
                    records.append(dict(port=folder.name, draw=draw, arm=arm, readout=readout,
                        query_n=len(truth), oa=100*np.mean(pred == truth),
                        ba=np.mean([100*np.mean(pred[truth == c] == c) for c in present]),
                        rescues=int(np.sum((pred == truth) & (ref != truth))), harms=int(np.sum((pred != truth) & (ref == truth)))))
                    for c in present:
                        classes.append(dict(port=folder.name, draw=draw, arm=arm, readout=readout,
                            class_id=int(c), n=int((truth == c).sum()), recall=100*np.mean(pred[truth == c] == c)))
    m = pd.DataFrame(records)
    m.to_csv(PORT_OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(PORT_OUT/'class_metrics.csv', index=False)
    means = m.groupby(['port', 'arm', 'readout'])[['oa', 'ba', 'rescues', 'harms']].mean().reset_index()
    means.to_csv(PORT_OUT/'port_means.csv', index=False)
    summary = means.groupby(['arm', 'readout'])[['oa', 'ba']].mean().reset_index()
    summary.to_csv(PORT_OUT/'summary.csv', index=False)
    write(PORT_OUT/'COMPLETE.json', dict(tasks=23, draws=115, source_priors_refitted=True,
        all_predictions_frozen_before_query_readout=True, independent_confirmation=False))
    print(summary[summary.readout.isin(['fusion', 'unified'])].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'target', 'report'], default='all')
    args = parser.parse_args()
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'product', 'port', 'class_id'])
    side = pd.read_csv(SOURCE_ANGLE, usecols=['instance_id', 'product', 'status', 'incidence_deg'])
    meta = frame[['instance_id', 'product']].merge(side, on=['instance_id', 'product'], how='left', sort=False, validate='one_to_one')
    angles = meta.incidence_deg.to_numpy(copy=True)
    angles[meta.status.to_numpy() != 'ok'] = np.nan
    folders = sorted((TASKS/'self24ports').glob('*/PREPARED.json'))
    folders = [f.parent for f in folders]
    if len(folders) != 23:
        raise ValueError('Expected23 inherited tasks')
    write(PORT_OUT/'PROTOCOL.json', dict(arms=PORT_ARMS, readouts=READOUTS,
        common_core='PCA retained/complement expert; C4 base; identical mean-condition and frequency kernels; no Qs/Qd/incidence',
        source_fits='Source-purged per-port rows; base C4 and deployed expert features; balanced ridge1 for every arm',
        adapted='Same64 supports, ridge8, equal probability fusion and equal-kernel unified head',
        fixed='Old source geometry/scales, expert neural weights and learned frequency embeddings',
        limitation='Frozen-front-end counterfactual: frequency was historically trained with old knowledge inputs. Not full source neural retraining.',
        weights='Same as external: drift0.5, stable base1/expert0.25, angle base0.5/expert0.125',
        selection='No target outcome selection; all seven arms retained'))
    with threadpool_limits(limits=2):
        if args.stage in ('all', 'target'):
            raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
            c4 = np.load(SOURCE_C4, mmap_mode='r')
            for folder in folders:
                freeze_port(folder, frame, raw, c4, angles)
                gc.collect()
        if args.stage in ('all', 'report'):
            report(frame, folders)


if __name__ == '__main__':
    main()
