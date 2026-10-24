"""Source-only test of whole-axis ordering versus unconstrained fragment pairing."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import itertools
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location('axial_source_inputs', ROOT/'conditional_relation_operator/learned/run.py')
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)
SUBSETS = np.array(list(itertools.combinations(range(4), 3)))
PATTERNS = {}
for name in ('ordered', 'free'):
    patterns = []
    for q in SUBSETS:
        for s in SUBSETS:
            for perm in ([s] if name == 'ordered' else itertools.permutations(s)):
                patterns.append((q, perm))
    PATTERNS[name] = np.asarray(patterns)


def align(q, s, qm, sm, method):
    # Paired side costs retain transverse identity; half-turn flips both axes.
    values = []
    for flip in (False, True):
        sr = s[..., ::-1, ::-1, :] if flip else s
        mr = sm[..., ::-1, ::-1] if flip else sm
        dot = np.einsum('bpid,cpjd->bcpij', q.reshape(len(q), 2, 8, 32), sr.reshape(len(s), 2, 8, 32))
        dot = dot.reshape(len(q), len(s), 2, 4, 2, 4, 2)
        # Match the two sides jointly, not one side to whichever is brighter.
        cost = .5*(dot[..., 0, :, 0]+dot[..., 1, :, 1])
        qvalid = (qm > 0).all(-1)
        svalid = (mr > 0).all(-1)
        qi, si = PATTERNS[method][:, 0], PATTERNS[method][:, 1]
        part = cost[..., qi, si].mean(-1)
        legal = qvalid[:, None, :, qi].all(-1) & svalid[None, :, :, si].all(-1)
        values.append(np.where(legal, part, -np.inf).max(-1))
    return np.maximum(values[0], values[1])


def checks():
    rng = np.random.default_rng(37)
    q = rng.normal(size=(2,2,4,2,32)); s = rng.normal(size=(3,2,4,2,32))
    qm = np.ones((2,2,4,2)); sm = np.ones((3,2,4,2))
    errs = []
    for method in PATTERNS:
        a = align(q,s,qm,sm,method)
        b = align(q[...,::-1,::-1,:],s,qm[...,::-1,::-1],sm,method)
        errs.append(float(abs(a-b).max()))
        # Independent scalar enumeration checks coordinate reshaping and side pairing.
        explicit = []
        for flip in (False,True):
            t = s[0,0,::-1,::-1] if flip else s[0,0]
            for qi,si in PATTERNS[method]:
                explicit.append(sum(np.dot(q[0,0,i,h],t[j,h]) for i,j in zip(qi,si) for h in range(2))/6)
        errs.append(abs(float(max(explicit)-a[0,0,0])))
    assert max(errs)<1e-10,errs
    assert np.all(align(q,s,qm,sm,'free')>=align(q,s,qm,sm,'ordered')-1e-10)
    return dict(max_error=max(errs), patterns={k:len(v) for k,v in PATTERNS.items()}, half_turn_invariant=True)


def main():
    check = checks()
    table,x,mass,bank,idx,tr,va,te = exp.source_data()
    ids = idx[np.r_[tr,va]]
    assert table.iloc[ids].role.isin(['utility_train','selection']).all()
    for field in ('mmsi','pass_group'):
        assert not set(table.iloc[ids][field]) & set(table.iloc[bank.ravel()][field])
    b=x[:,:,:8].reshape(len(x),2,4,2,32)
    m=mass[:,:,:8].reshape(len(x),2,4,2)
    rng=np.random.default_rng(619)
    permutation=np.stack([rng.permutation(4) for _ in range(len(x)*2)]).reshape(len(x),2,4)
    bs=np.take_along_axis(b,permutation[...,None,None],axis=2)
    ms=np.take_along_axis(m,permutation[...,None],axis=2)
    features={}; references={}; valid=None
    for name, values, areas, method in [('ordered',b,m,'ordered'),('free',b,m,'free'),('shuffled_order',bs,ms,'ordered')]:
        allscores=[]
        for start in range(0,len(ids),64):
            part=ids[start:start+64]
            allscores.append(align(values[part],values[bank.ravel()],areas[part],areas[bank.ravel()],method))
        score=np.concatenate(allscores).reshape(len(ids),8,8,2)
        good=np.isfinite(score)
        count=good.sum(2)
        mean=np.where(good,score,0.).sum(2)/np.maximum(count,1)
        if valid is None:
            valid=count>0
        else:
            np.testing.assert_array_equal(valid,count>0)
        references[name]=score
        features[name]=mean
    rows=[]
    y=table.iloc[ids].class_id.to_numpy(int)
    roles=table.iloc[ids].role.to_numpy()
    for name,z in features.items():
        for p,pol in enumerate(('VV','VH')):
            masked=np.where(valid[:,:,p],z[:,:,p],-np.inf)
            true=masked[np.arange(len(ids)),y]
            rivals=np.where(np.arange(8)[None]==y[:,None],-np.inf,masked)
            best=rivals.max(-1)
            mean=np.where(np.isfinite(rivals),rivals,0.).sum(-1)/np.maximum(np.isfinite(rivals).sum(-1),1)
            for role in ('utility_train','selection'):
                for c in range(8):
                    keep=(roles==role)&(y==c)&valid[np.arange(len(ids)),y,p]&np.isfinite(best)
                    rows.append(dict(method=name,pol=pol,role=role,class_id=c,n=int(keep.sum()),
                        top1=float((masked.argmax(-1)[keep]==c).mean()), mean_gap=float((true[keep]-mean[keep]).mean()),
                        best_gap=float((true[keep]-best[keep]).mean())))
    df=pd.DataFrame(rows)
    df.to_csv(HERE/'classes.csv',index=False)
    summary=df.groupby(['role','pol','method'])[['top1','mean_gap','best_gap']].mean()
    summary.to_csv(HERE/'summary.csv')
    sample=[]
    for p,pol in enumerate(('VV','VH')):
        free=features['free'][:,:,p]; ordered=features['ordered'][:,:,p]
        best=np.where(np.arange(8)[None]==y[:,None],-np.inf,np.where(valid[:,:,p],free,-np.inf)).argmax(-1)
        for j,i in enumerate(ids):
            if not valid[j,y[j],p]:
                continue
            sample.append(dict(obs_id=int(table.iloc[i].obs_id),role=roles[j],pol=pol,class_id=int(y[j]),rival=int(best[j]),
                true_loss=float(free[j,y[j]]-ordered[j,y[j]]), rival_loss=float(free[j,best[j]]-ordered[j,best[j]]),
                order_advantage=float(free[j,best[j]]-ordered[j,best[j]]-free[j,y[j]]+ordered[j,y[j]])))
    pd.DataFrame(sample).to_csv(HERE/'samples.csv',index=False)
    np.savez_compressed(HERE/'scores.npz',ids=ids,bank=bank,valid=valid,**features,**{'refs_'+k:v for k,v in references.items()})
    (HERE/'checks.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(summary.to_string())


if __name__=='__main__':
    main()
