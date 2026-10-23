"""Source-only equal-mass edge matching and eighteen frozen-anchor heads."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


support = module('edge_source_support', ROOT/'support_transfer_learning/run.py')
dep = module('edge_source_fit', ROOT/'reference_dependence/run.py')
ARMS = ('body', 'additive', 'mixed', 'mixed_shuffle101', 'mixed_shuffle211', 'mixed_shuffle307')
GROUPS = ('VV_side', 'VV_end', 'VH_side', 'VH_end')
SEEDS = (17, 29, 43)
TAU = 1/np.sqrt(32)
BODY = (np.arange(8), np.array([0,1,6,7]))
NEIGHBOR = (np.arange(8,16), np.arange(16,20))
REV = np.r_[np.arange(8)[::-1],np.arange(8,16)[::-1],np.arange(16,20)[::-1]]


def edges(x, mass, shuffle=None):
    n = len(x)
    qb = np.zeros((n,4,8,32), np.float32)
    qn = np.zeros_like(qb)
    legal = np.zeros((n,4,8), bool)
    neighbors = np.full((n,4,8), -1, np.int16)
    rng = np.random.default_rng(shuffle) if shuffle is not None else None
    for g in range(4):
        p, kind = divmod(g,2)
        body, near = BODY[kind], NEIGHBOR[kind]
        e = len(body)
        valid = (mass[:,p,body]>0) & (mass[:,p,near]>0)
        mapping = np.broadcast_to(near,(n,e)).copy()
        if rng is not None:
            for i in range(n):
                positions = np.flatnonzero(valid[i])
                mapping[i,positions] = rng.permutation(near[positions])
        qb[:,g,:e] = x[:,p,body]
        qn[:,g,:e] = x[np.arange(n)[:,None],p,mapping]
        legal[:,g,:e] = valid
        neighbors[:,g,:e] = mapping
    return qb, qn, legal, neighbors


def embedding(qb, qn, arm):
    if arm == 'body':
        return qb
    if arm == 'additive':
        return np.concatenate([qb,qn],axis=-1)/np.sqrt(np.float32(2))
    return (qb-qn)/np.sqrt(np.float32(2))


def score_groups(z, legal, bank, ids, device='cuda'):
    """Uniform valid edges within each reference, then uniform valid references."""
    scores, valid, per_edge = [], [], []
    with torch.inference_mode():
        for g in range(4):
            ref_valid = legal[bank,g].any(-1)
            edge_count = legal[bank,g].sum(-1)
            ref_count = ref_valid.sum(-1)
            logmass = -np.log(np.maximum(edge_count,1))[:,:,None] - np.log(np.maximum(ref_count,1))[:,None,None]
            logmass = np.where(legal[bank,g],logmass,-np.inf).astype(np.float32)
            s = torch.tensor(z[bank,g],device=device)
            lm = torch.tensor(logmass,device=device)
            values = []
            for start in range(0,len(ids),64):
                q = torch.tensor(z[ids[start:start+64],g],device=device)
                sim = torch.einsum('bid,crjd->bcirj',q,s)/TAU
                values.append(torch.logsumexp((sim+lm[None,:,None]).flatten(-2),-1).cpu().numpy())
            edge = np.concatenate(values)
            mask = legal[ids,g]
            has = mask.any(-1)[:,None] & (ref_count>0)[None]
            edge = np.where(mask[:,None] & has[:,:,None],edge,0.)
            score = edge.sum(-1)/np.maximum(mask.sum(-1)[:,None],1)
            scores.append(score)
            valid.append(has)
            per_edge.append(edge)
    return np.stack(scores,-1),np.stack(valid,-1),np.stack(per_edge,1)


def trace_query(table_index, arm, x=None, mass=None, bank=None):
    """Exact four-point/query-edge/reference decomposition for subsequent chip review."""
    if x is None:
        _,x,mass,bank = support.load()
    seed = int(arm.split('shuffle')[1]) if 'shuffle' in arm else None
    qb,qn,legal,mapping = edges(x,mass,seed)
    out = dict(table_index=np.array(table_index),bank=bank,neighbor_map=mapping[np.r_[table_index,bank.ravel()]],
               legal_edges=legal[np.r_[table_index,bank.ravel()]])
    scores = []
    for g in range(4):
        b,n = qb[table_index,g].astype(float),qn[table_index,g].astype(float)
        sb,sn = qb[bank,g].astype(float),qn[bank,g].astype(float)
        points = np.stack([np.einsum('id,crjd->cirj',b,sb),np.einsum('id,crjd->cirj',n,sn),
                           np.einsum('id,crjd->cirj',b,sn),np.einsum('id,crjd->cirj',n,sb)])
        sim = points[0] if arm=='body' else .5*(points[0]+points[1])
        if arm.startswith('mixed'):
            sim = .5*(points[0]+points[1]-points[2]-points[3])
        refmask = legal[bank,g]
        refcount = refmask.any(-1).sum(-1)
        lm = -np.log(np.maximum(refmask.sum(-1),1))[:,:,None]-np.log(np.maximum(refcount,1))[:,None,None]
        lm = np.where(refmask,lm,-np.inf)
        log_reference = logsumexp(sim/TAU+lm[:,None],axis=-1)
        edge_score = logsumexp(log_reference,axis=-1)
        querymask = legal[table_index,g]
        has = querymask.any() & (refcount>0)
        edge_score = np.where(querymask[None] & has[:,None],edge_score,0.)
        score = edge_score.sum(-1)/max(querymask.sum(),1)
        responsibility = np.exp(sim/TAU+lm[:,None]-edge_score[:,:,None,None])
        responsibility = np.where(querymask[None,:,None,None]&has[:,None,None,None],responsibility,0.)
        out.update({f'{GROUPS[g]}_four_points':points, f'{GROUPS[g]}_similarity':sim,
                    f'{GROUPS[g]}_log_reference':log_reference,f'{GROUPS[g]}_edge_score':edge_score,
                    f'{GROUPS[g]}_responsibility':responsibility})
        scores.append(score)
    out['raw_scores'] = np.stack(scores,-1)
    return out


def checks(x,mass,bank):
    rng = np.random.default_rng(17)
    q,s = rng.normal(size=(2,32)),rng.normal(size=(2,32))
    block = q@s.T
    kernel = np.array([[1.,-1.],[-1.,1.]])/2
    expanded = float((block*kernel).sum())
    difference = float((q[0]-q[1])@(s[0]-s[1])/2)
    contaminated = block+rng.normal(size=(2,1))+rng.normal(size=(1,2))
    arithmetic = abs(expanded-difference)
    bias_error = abs(float((contaminated*kernel).sum())-expanded)
    assert max(arithmetic,bias_error)<1e-12
    qb,qn,legal,_ = edges(x,mass)
    missing = np.flatnonzero(mass[:,1].sum(-1)==0)
    ids = np.unique(np.r_[0,bank.ravel()[:2],missing[:1]])
    assert len(missing)>0 and not legal[missing[0],2:].any()
    flipb,flipn,flipl,_ = edges(x[:,:,REV],mass[:,:,REV])
    for g in range(4):
        e = len(BODY[g%2])
        np.testing.assert_array_equal(legal[:,g,:e][:,::-1],flipl[:,g,:e])
        np.testing.assert_array_equal(qb[:,g,:e][:,::-1],flipb[:,g,:e])
        np.testing.assert_array_equal(qn[:,g,:e][:,::-1],flipn[:,g,:e])
    flip_error = 0.
    for arm in ('body','additive','mixed'):
        before,v,_ = score_groups(embedding(qb,qn,arm),legal,bank,ids)
        after,v2,_ = score_groups(embedding(flipb,flipn,arm),flipl,bank,ids)
        np.testing.assert_array_equal(v,v2)
        flip_error = max(flip_error,float(np.max(np.abs(before-after))))
        assert not v[ids==missing[0],:,2:].any()
        assert not before[ids==missing[0],:,2:].any()
    assert flip_error<5e-6,flip_error
    for shuffle in (101,211,307):
        b,n,l,mapping = edges(x,mass,shuffle)
        np.testing.assert_array_equal(l,legal)
        np.testing.assert_array_equal(b,qb)
        # Only legal neighbors are permuted; masks and equal weights stay fixed.
        for i in ids:
            for g in range(4):
                keep=legal[i,g]
                original=NEIGHBOR[g%2][np.flatnonzero(keep)]
                np.testing.assert_array_equal(np.sort(mapping[i,g,keep]),original)
                assert np.all(mass[i,g//2,mapping[i,g,keep]]>0)
    return dict(four_point_error=arithmetic,row_column_bias_error=bias_error,
                axis_flip_score_error=flip_error,checked_indices=ids.tolist(),
                missing_vh_index=int(missing[0]),shuffle_mass='uniform unchanged; positive mass validity only')


def main():
    tick = time.perf_counter()
    torch.set_num_threads(2)
    table,x,mass,bank = support.load()
    idx=np.flatnonzero(table.role.isin(['utility_train','selection','evaluation']))
    roles=table.iloc[idx].role.to_numpy()
    train,selection,evaluation=[np.flatnonzero(roles==name) for name in ('utility_train','selection','evaluation')]
    assert [len(train),len(selection),len(evaluation)]==[644,612,485]
    y=table.iloc[idx].class_id.to_numpy()
    verify=checks(x,mass,bank)
    print('protocol checks',verify,flush=True)
    features,raws,valids,edge_scores,maps,scales={},{},{},{},{},[]
    for arm in ARMS:
        shuffle=int(arm.split('shuffle')[1]) if 'shuffle' in arm else None
        qb,qn,legal,mapping=edges(x,mass,shuffle)
        raw,valid,edge=score_groups(embedding(qb,qn,arm),legal,bank,np.arange(len(table)))
        center=np.where(valid,raw,0).sum(1,keepdims=True)/np.maximum(valid.sum(1,keepdims=True),1)
        centered=np.where(valid,raw-center,np.nan)
        divisor,rms,count=dep.rms_valid(centered,valid,idx[train])
        features[arm]=dep.standardize(centered,valid,divisor)
        raws[arm],valids[arm],edge_scores[arm],maps[arm]=raw,valid,edge,mapping
        scales.extend(dict(arm=arm,group=GROUPS[g],feature=g,divisor=divisor[g],
                           utility_rms=rms[g],valid_count=int(count[g])) for g in range(4))
        print('scores',arm,raw.shape,flush=True)
    np.savez_compressed(HERE/'features.npz',idx=idx,bank=bank,**features)
    np.savez_compressed(HERE/'localcache.npz',bank=bank,instance_ids=table.instance_id.to_numpy(dtype=str),
                        legal=legal,**{f'{a}_raw':v for a,v in raws.items()},
                        **{f'{a}_valid':v for a,v in valids.items()},**{f'{a}_edge':v for a,v in edge_scores.items()},
                        **{f'{a}_neighbor_map':v for a,v in maps.items()})
    pd.DataFrame(scales).to_csv(HERE/'scales.csv',index=False)
    results,receipts,curves,classes,pairs,coefficients=[],[],[],[],[],[]
    weights={}
    for seed in SEEDS:
        old=np.load(ROOT/f'reference_dependence/frozen_marginal/predictions_{seed}.npz')
        np.testing.assert_array_equal(old['idx'],idx)
        anchor=old['anchor'][idx]
        saved=dict(anchor=old['anchor'].copy(),full=old['full'].copy())
        for arm in ARMS:
            start=time.perf_counter()
            score,delta,w,states,history,step,error=dep.fit(features[arm][idx],anchor,y,train,selection)
            saved[arm]=old['anchor'].copy()
            saved[arm][idx]=score
            weights[f'{arm}_{seed}']=w
            weights[f'{arm}_{seed}_checkpoints']=states
            receipts.append(dict(seed=seed,arm=arm,selected_step=step,parameters=len(w),
                                 replay_error=error,seconds=time.perf_counter()-start))
            curves.extend(dict(seed=seed,arm=arm,**row) for row in history)
            coefficients.extend(dict(seed=seed,arm=arm,group=GROUPS[j],coefficient=float(v)) for j,v in enumerate(w))
            print('fit',seed,arm,'step',step,dep.metrics(y[evaluation],score[evaluation]),flush=True)
        for arm,score in saved.items():
            for role,select in [('utility_train',train),('selection',selection),('evaluation',evaluation)]:
                results.append(dict(seed=seed,arm=arm,role=role,**dep.metrics(y[select],score[idx[select]])))
            correct=score[idx[evaluation]].argmax(-1)==y[evaluation]
            for reference in ('anchor','body','additive','mixed'):
                previous=saved[reference][idx[evaluation]].argmax(-1)==y[evaluation]
                pairs.append(dict(seed=seed,arm=arm,reference=reference,rescued=int((correct&~previous).sum()),
                                  harmed=int((~correct&previous).sum()),net=int(correct.sum()-previous.sum())))
            for c in range(8):
                select=y[evaluation]==c
                classes.append(dict(seed=seed,arm=arm,class_id=c,n=int(select.sum()),correct=int(correct[select].sum()),
                                    **{k:v for k,v in dep.metrics(y[evaluation][select],score[idx[evaluation]][select]).items() if k!='n'}))
        np.savez_compressed(HERE/f'predictions_{seed}.npz',idx=idx,bank=bank,**saved)
    np.savez_compressed(HERE/'weights.npz',**weights)
    for name,rows in [('metrics',results),('training_receipts',receipts),('selection_curve',curves),
                      ('class_results',classes),('paired_results',pairs),('coefficients',coefficients)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    summary=pd.DataFrame(results).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean()
    summary.to_csv(HERE/'summary.csv')
    # Small scalar trace is independent of the GPU embedding/aggregation path.
    full_vh=idx[evaluation][mass[idx[evaluation],1].sum(-1)>0][0]
    no_vh=idx[evaluation][mass[idx[evaluation],1].sum(-1)==0][0]
    trace_error=0.
    for i in (full_vh,no_vh):
        for arm in ARMS:
            trace=trace_query(int(i),arm,x,mass,bank)
            trace_error=max(trace_error,float(np.max(np.abs(trace['raw_scores']-raws[arm][i]))))
            if arm=='mixed':
                np.savez_compressed(HERE/f'trace_mixed_{i}.npz',**trace)
    assert trace_error<5e-6,trace_error
    assert len(receipts)==18
    receipt=dict(status='complete',fits=18,seeds=list(SEEDS),arms=list(ARMS),source_n=len(table),
                 roles={'utility_train':644,'selection':612,'evaluation':485},parameters_per_head=4,
                 tau=TAU,edge_mass='uniform over legal edges; mass only tests positivity',
                 centering='valid classes within each group',scale='per arm/group RMS on utility644 only',
                 equal_reference_weight=True,difference_unit_normalized=False,source_only=True,
                 no_external=True,no_git=True,no_images=True,checks=verify,trace_error=trace_error,
                 max_prediction_replay_error=max(r['replay_error'] for r in receipts),
                 trace_example_indices=[int(full_vh),int(no_vh)],
                 outside_idx_predictions='unchanged frozen anchor; all3387 local scores still cached',
                 seconds=time.perf_counter()-tick)
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(summary.to_string(),flush=True)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
