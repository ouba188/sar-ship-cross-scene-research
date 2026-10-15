"""Replay frozen classifiers and recover case/deletion evidence, without labels."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
STUDY = ROOT/'same_vessel_condition_study_20261010'
PREV = ROOT/'correction_episode_rebuild_20261013'
BUDGET = ROOT/'frozen_candidate_coverage_20261012/budget'
sys.path.insert(0, str(ROOT/'joint_support_correction_20261013'))
import joint_core as j
from run_joint import support_metadata
from strong_external import CachedStrongFeatures
from evidence_core import contributions, deletion_features, NAMES
c = j.c


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(xs, ps, sy, groups, model, kb, bps):
    started = time.perf_counter()
    xi = j.corrected_support(xs, model)
    ar = c.coefficients(xs@xs.T, ps, sy, 2.)
    ai = c.coefficients(xi@xi.T, ps, sy, 2.)
    ab = c.coefficients(kb, bps, sy, 2.)
    original_seconds = time.perf_counter()-started
    started = time.perf_counter()
    ew, bw, remaining = [], [], []
    for g in np.unique(groups):
        keep = np.flatnonzero(groups != g)
        assert len(keep) > 0
        raw = xs[keep]
        changed = j.corrected_support(raw, model)
        wr = raw.T@c.coefficients(raw@raw.T, ps[keep], sy[keep], 2.)
        wi = changed.T@c.coefficients(changed@changed.T, ps[keep], sy[keep], 2.)
        b = np.zeros((len(sy), 8))
        b[keep] = c.coefficients(kb[np.ix_(keep, keep)], bps[keep], sy[keep], 2.)
        ew.extend([wr, wi]); bw.append(b); remaining.append(keep)
    return dict(xi=xi, ar=ar, ai=ai, ab=ab, ew=np.concatenate(ew, 1),
                bw=np.concatenate(bw, 1), remaining=remaining,
                original_fit_seconds=original_seconds,
                deletion_fit_seconds=time.perf_counter()-started)


def compute_unit(u, z, xs, ps, sy, groups, model, kb, bps, get_block, kernel_check):
    started = time.perf_counter()
    fitted = prepare(xs, ps, sy, groups, model, kb, bps)
    # A real reduced-support kernel checks that slicing preserves this kernel.
    keep = fitted['remaining'][0]
    actual = kernel_check(keep)
    np.testing.assert_allclose(actual, kb[np.ix_(keep, keep)], atol=1e-10, rtol=0)
    n, nd = len(z['query_ids']), len(np.unique(groups))
    evidence = np.empty((n, 14))
    replay = {'raw': 0., 'initial': 0., 'case_sum': 0.}
    timing = dict(feature_kernel_seconds=0., original_query_seconds=0.,
                  case_seconds=0., deletion_query_seconds=0.)
    for lo in range(0, n, 2048):
        hi = min(lo+2048, n)
        ts = time.perf_counter()
        xq, pq, bq, kq = get_block(lo, hi)
        timing['feature_kernel_seconds'] += time.perf_counter()-ts
        ts = time.perf_counter()
        kr, ki = xq@xs.T, xq@fitted['xi'].T
        er, ei = pq+kr@fitted['ar'], pq+ki@fitted['ai']
        bp = softmax(bq+kq@fitted['ab'], axis=1)
        raw, initial = .5*(bp+softmax(er, axis=1)), .5*(bp+softmax(ei, axis=1))
        timing['original_query_seconds'] += time.perf_counter()-ts
        for name, value in [('raw',raw), ('initial',initial)]:
            np.testing.assert_allclose(value, z[name][lo:hi], atol=1e-9, rtol=0)
            np.testing.assert_array_equal(value.argmax(1), z[name][lo:hi].argmax(1))
            replay[name] = max(replay[name], float(abs(value-z[name][lo:hi]).max()))
        r, i = raw.argmax(1), initial.argmax(1)
        ts = time.perf_counter()
        ev, terms = contributions(kr, ki, fitted['ar'], fitted['ai'], r, i, sy, groups)
        rows = np.arange(hi-lo)
        expected = (ei-er)[rows, i]-(ei-er)[rows, r]
        np.testing.assert_allclose(terms.sum(1), expected, atol=1e-9, rtol=0)
        replay['case_sum'] = max(replay['case_sum'], float(abs(terms.sum(1)-expected).max()))
        evidence[lo:hi, :8] = ev
        timing['case_seconds'] += time.perf_counter()-ts
        ts = time.perf_counter()
        e = (xq@fitted['ew']).reshape(hi-lo, nd, 2, 8)+pq[:,None,None,:]
        b = (kq@fitted['bw']).reshape(hi-lo, nd, 8)+bq[:,None,:]
        deleted = .5*(softmax(e,axis=3)+softmax(b,axis=2)[:,:,None,:])
        evidence[lo:hi, 8:] = deletion_features(deleted[:,:,0], deleted[:,:,1], r, i)
        timing['deletion_query_seconds'] += time.perf_counter()-ts
    assert np.isfinite(evidence).all()
    path = OUT/'features'/(u['uid']+'.npz')
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, query_ids=z['query_ids'], support_ids=z['support_ids'],
                        evidence=evidence, groups=groups, support_labels=sy)
    info = dict(uid=u['uid'], family=u.get('family','source_fit'),
                task=u.get('task',u.get('context')), cohort=u.get('cohort','expanded'),
                query_n=n, support_vessels=nd, unknown_support_rows=int(sum(str(g).startswith('row:') for g in groups)),
                original_fit_seconds=fitted['original_fit_seconds'],
                deletion_fit_seconds=fitted['deletion_fit_seconds'], **timing,
                wall_seconds=time.perf_counter()-started, **{'replay_'+k:v for k,v in replay.items()})
    write(path.with_suffix('.json'), info)
    print('EVIDENCE', u['uid'], n, round(info['wall_seconds'],2), flush=True)
    return info


def source(stage):
    units = ([u for u in read(PREV/'SOURCE_READY.json')['units'] if u['kind']=='expanded']
             if stage=='source' else [u for u in read(PREV/'PREDICTIONS_FROZEN.json')['units'] if u['family']=='source_held'])
    receipts = []
    for ctx in ('Los Angeles','Port Klang','Santos'):
        subset = [u for u in units if u.get('context',u.get('task'))==ctx]
        maps = load(STUDY/'case_relation_headroom_20261012'/ctx/'maps.npz', ['obs_ids','single','prior'])
        bd = load(STUDY/'support_budget_fusion_select_20261012/base_cache'/ctx/'base.npz')
        np.testing.assert_array_equal(maps['obs_ids'], bd['obs_ids'])
        idx = pd.Index(maps['obs_ids'])
        fp = load(STUDY/'case_relation_student_20261012'/ctx/'source_readers.npz', ['fit_prior'])['fit_prior']
        for u in subset:
            receipt = OUT/'features'/(u['uid']+'.json')
            if receipt.exists():
                receipts.append(read(receipt)); continue
            z = load(u['path']); s,q = idx.get_indexer(z['support_ids']),idx.get_indexer(z['query_ids'])
            assert (s>=0).all() and (q>=0).all()
            sm = support_metadata(STUDY/'observations.csv', np.sort(z['support_ids']))
            sy = sm.loc[z['support_ids'],'class_id'].to_numpy(int)
            groups = sm.loc[z['support_ids'],'group'].to_numpy(str)
            prior,bprior = (fp,bd['prior_oof']) if stage=='source' else (maps['prior'],bd['prior_held'])
            model = load(STUDY/'case_completion_class_control_20261012'/ctx/
                         (f"fold_{u['fold']}_model.npz" if stage=='source' else 'final_model.npz'))
            eta = float(bd['eta'])
            kb = c.base_block(bd,bd['embedding'],np.empty(0,int),s,eta)
            def block(lo,hi):
                qi=q[lo:hi]
                return maps['single'][qi],prior[qi],bprior[qi],c.base_block(bd,bd['embedding'],qi,s,eta)[len(s):]
            def check(keep):
                return c.base_block(bd,bd['embedding'],np.empty(0,int),s[keep],eta)
            receipts.append(compute_unit(u,z,maps['single'][s],prior[s],sy,groups,model,kb,bprior[s],block,check))
    write(OUT/(stage.upper()+'_READY.json'), dict(units=receipts, query_labels_used=False))


def targets():
    units = [u for u in read(PREV/'PREDICTIONS_FROZEN.json')['units'] if u['family']!='source_held']
    receipts = []
    for task in ('Los Angeles','Port Klang','Santos','f0921','oss2'):
        subset = [u for u in units if u['task']==task]
        isport = task not in ('f0921','oss2')
        folder = c.TASKS/'self24ports'/task if isport else c.TASKS/task
        directory = BUDGET/'tasks/self24ports'/task if isport else BUDGET/'strong_external'/task/task
        feat = c.Features(folder) if isport else CachedStrongFeatures(task)
        idx = pd.Index(feat.target_rows)
        data = load(folder/'target.npz',['rows','phi','mean','conditions','prior'])
        emb = load(folder/'selective_embedding.npz',['rows','embedding'])
        np.testing.assert_array_equal(data['rows'],emb['rows'])
        np.testing.assert_array_equal(data['rows'],feat.target_rows)
        model = load(directory/'initial.npz'); eta = read(folder/'PREPARED.json')['eta']
        for u in subset:
            receipt = OUT/'features'/(u['uid']+'.json')
            if receipt.exists():
                receipts.append(read(receipt)); continue
            z = load(u['path']); s,q = idx.get_indexer(z['support_ids']),idx.get_indexer(z['query_ids'])
            assert (s>=0).all() and (q>=0).all()
            sm = support_metadata(u['label_path'],np.sort(z['support_ids']))
            sy = sm.loc[z['support_ids'],'class_id'].to_numpy(int)
            groups = sm.loc[z['support_ids'],'group'].to_numpy(str)
            xs,ps = feat.target(s)
            kb = c.base_block(data,emb['embedding'],np.empty(0,int),s,eta)
            def block(lo,hi):
                qi=q[lo:hi]; xq,pq=feat.target(qi)
                return xq,pq,data['prior'][qi],c.base_block(data,emb['embedding'],qi,s,eta)[len(s):]
            def check(keep):
                return c.base_block(data,emb['embedding'],np.empty(0,int),s[keep],eta)
            receipts.append(compute_unit(u,z,xs,ps,sy,groups,model,kb,data['prior'][s],block,check))
        del feat,data,emb
        gc.collect()
    write(OUT/'TARGET_READY.json',dict(units=receipts,query_labels_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['source','held','targets','summary'])
    stage=parser.parse_args().stage
    if stage in ('source','held'):
        source(stage)
    elif stage=='targets':
        targets()
    else:
        records=[read(p) for p in sorted((OUT/'features').glob('*.json'))]
        pd.DataFrame(records).to_csv(OUT/'extraction_timings.csv',index=False)
        write(OUT/'FEATURES_READY.json',dict(units=len(records),names=NAMES,query_labels_used=False))
