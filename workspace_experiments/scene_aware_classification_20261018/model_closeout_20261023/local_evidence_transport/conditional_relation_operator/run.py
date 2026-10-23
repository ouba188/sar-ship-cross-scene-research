"""Body-conditioned neighborhood transfer with an exact pairing-risk reference."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import itertools
import json
import time
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location('conditional_existing_edge', ROOT/'edge_difference/run.py')
edge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edge)
dep = edge.dep
ARMS = ('independent', 'transfer', 'excess', 'excess_shuffle101', 'excess_shuffle307')


def center(b, n, legal):
    count = legal.sum(-1)
    b = np.asarray(b, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    bm = (b*legal[..., None]).sum(-2)/np.maximum(count[..., None], 1)
    nm = (n*legal[..., None]).sum(-2)/np.maximum(count[..., None], 1)
    return (b-bm[..., None, :])*legal[..., None], (n-nm[..., None, :])*legal[..., None], count


def transfer(qb, qn, sb, sn):
    """Unpadded scalar trace, independently used for exact-null verification."""
    qb, qn = qb-qb.mean(0), qn-qn.mean(0)
    sb, sn = sb-sb.mean(0), sn-sn.mean(0)
    m = len(sb)
    lam = max(float(np.sum(sb*sb)/m), 1e-8)
    A = np.linalg.solve(sb@sb.T+lam*np.eye(m), sb@qb.T).T
    pred = A@sn
    per_edge = 2*np.sum(qn*pred, -1)-np.sum(pred*pred, -1)
    null = np.sum(sn*sn)/(m-1)*(np.sum(A*A, -1)-A.sum(-1)**2/m)
    return dict(A=A, prediction=pred, transfer=per_edge, null=null, excess=per_edge+null)


def checks():
    rng = np.random.default_rng(47)
    qb, qn = rng.normal(size=(2,3,5))
    sb, sn = rng.normal(size=(2,4,5))
    v = transfer(qb, qn, sb, sn)
    values, errors = [], []
    for p in itertools.permutations(range(4)):
        vp = transfer(qb, qn, sb, sn[list(p)])
        values.append(vp['excess'])
        errors.append(np.sum((qn-qn.mean(0)-vp['prediction'])**2, -1))
    real_error = np.sum((qn-qn.mean(0)-v['prediction'])**2, -1)
    err = float(np.max(np.abs(np.mean(errors, 0)-real_error-v['excess'])))
    zero = float(np.max(np.abs(np.mean(values, 0))))
    flip = transfer(qb[::-1], qn[::-1], sb[::-1], sn[::-1])
    fliperr = abs(float(flip['excess'].mean()-v['excess'].mean()))
    shifted = transfer(qb+2, qn-3, sb+7, sn-9)
    shifterr = float(np.max(np.abs(shifted['excess']-v['excess'])))
    const = float(np.max(np.abs(transfer(qb, qn, sb, np.ones_like(sn))['excess'])))
    assert max(err, zero, fliperr, shifterr, const)<1e-11
    return dict(exact_null_error=err, permutation_mean=zero, axis_reversal_error=fliperr,
                additive_offset_error=shifterr, constant_neighbor=const)


def raw_scores(b, n, legal, bank, ids):
    """Equal valid reference means; queries never influence reference operators."""
    b, n, counts = center(b, n, legal)
    result = {a:np.zeros((len(ids),8,4)) for a in ARMS[:3]}
    refs = {a:np.zeros((len(ids),4,8,8)) for a in ARMS[:3]}
    valid = np.zeros((len(ids),8,4), bool)
    for g in range(4):
        qb, qn = b[ids,g], n[ids,g]
        mq = np.maximum(counts[ids,g],1)
        rv = counts[bank,g]>=2
        for c in range(8):
            for j in np.flatnonzero(rv[c]):
                ref = bank[c,j]
                keep = legal[ref,g]
                sb, sn = b[ref,g,keep], n[ref,g,keep]
                m = len(sb)
                lam = max(float(np.sum(sb*sb)/m),1e-8)
                solve = np.linalg.solve(sb@sb.T+lam*np.eye(m), sb)
                A = qb@solve.T
                pred = A@sn
                T = (2*np.sum(qn*pred,axis=(1,2))-np.sum(pred*pred,axis=(1,2)))/mq
                V = np.sum(sn*sn)/(m-1)*(np.sum(A*A,axis=(1,2))-np.sum(A.sum(-1)**2,axis=1)/m)/mq
                lamn = max(float(np.sum(sn*sn)/m),1e-8)
                An = qn@np.linalg.solve(sn@sn.T+lamn*np.eye(m),sn).T
                predn = An@sn
                independent = (2*np.sum(qn*predn,axis=(1,2))-np.sum(predn*predn,axis=(1,2)))/mq
                for a, value in [('independent',independent),('transfer',T),('excess',T+V)]:
                    refs[a][:,g,c,j] = np.where(counts[ids,g]>=2,value,0.)
            valid[:,c,g] = (counts[ids,g]>=2) & rv[c].any()
            for a in result:
                result[a][:,c,g] = refs[a][:,g,c].sum(-1)/max(int(rv[c].sum()),1)
    return result, valid, refs


def main():
    tick = time.perf_counter()
    torch.set_num_threads(2)
    verification = checks()
    table,x,mass,bank = edge.support.load()
    idx = np.flatnonzero(table.role.isin(['utility_train','selection','evaluation']))
    roles = table.iloc[idx].role.to_numpy()
    train, selection, evaluation = [np.flatnonzero(roles==r) for r in ('utility_train','selection','evaluation')]
    assert [len(train),len(selection),len(evaluation)] == [644,612,485]
    y = table.iloc[idx].class_id.to_numpy()
    b,n,legal,_ = edge.edges(x,mass)
    raw,valid,refs = raw_scores(b,n,legal,bank,idx)
    # Compare vectorized implementation to unpadded reconstruction, not itself.
    trace_error = 0.
    for g in range(4):
        for qi in (0,100,700):
            i = idx[qi]
            ref = bank[0,0]
            if min(legal[i,g].sum(),legal[ref,g].sum())<2:
                continue
            trace = transfer(b[i,g,legal[i,g]],n[i,g,legal[i,g]],b[ref,g,legal[ref,g]],n[ref,g,legal[ref,g]])
            trace_error = max(trace_error,abs(float(trace['excess'].mean()-refs['excess'][qi,g,0,0])))
    assert trace_error<1e-7,trace_error
    verification['vector_scalar_error'] = trace_error
    missing = mass[idx,1].sum(-1)==0
    assert missing.any() and not valid[missing,:,2:].any()
    assert np.count_nonzero(raw['excess'][missing,:,2:])==0
    for seed in (101,307):
        bs,ns,ls,_ = edge.edges(x,mass,seed)
        np.testing.assert_array_equal(ls,legal)
        sr,sv,_ = raw_scores(bs,ns,ls,bank,idx)
        np.testing.assert_array_equal(sv,valid)
        raw[f'excess_shuffle{seed}'] = sr['excess']
    features,scales = {},[]
    for a in ARMS:
        v = raw[a]
        centered = np.where(valid,v-(v*valid).sum(1,keepdims=True)/np.maximum(valid.sum(1,keepdims=True),1),0.)
        divisor,rms,count = dep.rms_valid(centered,valid,train)
        features[a] = dep.standardize(centered,valid,divisor)
        scales.extend(dict(arm=a,group=edge.GROUPS[g],feature=g,divisor=divisor[g],rms=rms[g],n=count[g]) for g in range(4))
    np.savez_compressed(HERE/'features.npz',idx=idx,bank=bank,valid=valid,**features,
                        **{f'{a}_raw':v for a,v in raw.items()},**{f'{a}_refs':v for a,v in refs.items()})
    feature_seconds = time.perf_counter()-tick
    print('features and checks',feature_seconds,verification,flush=True)
    results,receipts,curves,classes,pairs,coefficients,weights = [],[],[],[],[],[],{}
    for seed in edge.SEEDS:
        old = np.load(ROOT/f'reference_dependence/frozen_marginal/predictions_{seed}.npz')
        np.testing.assert_array_equal(old['idx'],idx)
        saved = {base:old[base].copy() for base in ('full','anchor')}
        for base in ('full','anchor'):
            for arm in ARMS:
                start = time.perf_counter()
                score,delta,w,states,history,step,error = dep.fit(features[arm],old[base][idx],y,train,selection)
                key = f'{base}__{arm}'
                saved[key] = old[base].copy()
                saved[key][idx] = score
                weights[f'{key}_{seed}'] = w
                receipts.append(dict(seed=seed,base=base,arm=arm,selected_step=step,parameters=4,replay_error=error,seconds=time.perf_counter()-start))
                curves.extend(dict(seed=seed,base=base,arm=arm,**r) for r in history)
                coefficients.extend(dict(seed=seed,base=base,arm=arm,group=edge.GROUPS[g],coefficient=w[g]) for g in range(4))
                print('fit',seed,key,step,dep.metrics(y[evaluation],score[evaluation]),flush=True)
        for arm,score in saved.items():
            for role,inds in [('utility_train',train),('selection',selection),('evaluation',evaluation)]:
                results.append(dict(seed=seed,arm=arm,role=role,**dep.metrics(y[inds],score[idx[inds]])))
            correct = score[idx[evaluation]].argmax(-1)==y[evaluation]
            for base in ('full','anchor'):
                previous = saved[base][idx[evaluation]].argmax(-1)==y[evaluation]
                pairs.append(dict(seed=seed,arm=arm,reference=base,rescued=int((correct&~previous).sum()),harmed=int((~correct&previous).sum()),net=int(correct.sum()-previous.sum())))
            for c in range(8):
                keep = y[evaluation]==c
                classes.append(dict(seed=seed,arm=arm,class_id=c,n=int(keep.sum()),correct=int(correct[keep].sum())))
        np.savez_compressed(HERE/f'predictions_{seed}.npz',idx=idx,**saved)
    np.savez_compressed(HERE/'weights.npz',**weights)
    for name,rows in [('metrics',results),('training_receipts',receipts),('selection_curve',curves),('class_results',classes),('paired_results',pairs),('coefficients',coefficients),('scales',scales)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    pd.DataFrame(results).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean().to_csv(HERE/'summary.csv')
    (HERE/'execution.json').write_text(json.dumps(dict(feature_seconds=feature_seconds,total_seconds=time.perf_counter()-tick,checks=verification),indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
