"""Shared symmetric rank-four correction, source-only matched-capacity probe."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('fixed_edge_metric',ROOT/'edge_difference/run.py')
edge=importlib.util.module_from_spec(spec)
spec.loader.exec_module(edge)
ARMS=('body','additive','mixed','mixed_shuffle101','mixed_shuffle307')
SEEDS=(17,29,43)
CHECKPOINTS=(0,20,50,100,200)
BATCH=128
INIT_SEED=17
INIT_STD=.01


class Metric(torch.nn.Module):
    def __init__(self,device='cuda',dtype=torch.float32):
        super().__init__()
        generator=torch.Generator().manual_seed(INIT_SEED)
        self.U=torch.nn.Parameter((torch.randn(32,4,generator=generator)*INIT_STD).to(device=device,dtype=dtype))
        self.V=torch.nn.Parameter(torch.zeros(32,4,device=device,dtype=dtype))
        self.a=torch.nn.Parameter(torch.zeros(4,device=device,dtype=dtype))
        self.register_buffer('I',torch.eye(32,device=device,dtype=dtype))

    def matrix(self):
        uv=self.U@self.V.T
        return self.I+(uv+uv.T)*.5

    def regularizer(self):
        return .01*self.a.square().sum()+.01*(self.matrix()-self.I).square().sum()/32

    def state(self):
        return {key:value.detach().cpu().numpy().copy() for key,value in
                [('U',self.U),('V',self.V),('a',self.a),('W',self.matrix())]}

    def restore(self,state):
        with torch.no_grad():
            for name in ('U','V','a'):
                getattr(self,name).copy_(torch.as_tensor(state[name],device=self.U.device,dtype=self.U.dtype))


class Reader:
    def __init__(self,x,mass,bank,arm,divisor,device='cuda',dtype=torch.float32):
        shuffle=int(arm.split('shuffle')[1]) if 'shuffle' in arm else None
        qb,qn,legal,mapping=edge.edges(x,mass,shuffle)
        self.arm=arm
        self.bank=bank
        self.mapping=mapping
        self.numpy_legal=legal
        self.b=torch.as_tensor(qb,device=device,dtype=dtype)
        self.n=torch.as_tensor(qn,device=device,dtype=dtype)
        self.legal=torch.as_tensor(legal,device=device)
        self.divisor=torch.as_tensor(divisor,device=device,dtype=dtype)
        sm=legal[bank].transpose(2,0,1,3)
        count=sm.sum(-1)
        references=(count>0).sum(-1)
        logmass=-np.log(np.maximum(count,1))[...,None]-np.log(np.maximum(references,1))[:, :,None,None]
        logmass=np.where(sm,logmass,-np.inf)
        self.logmass=torch.as_tensor(logmass,device=device,dtype=dtype)
        self.ref_valid=torch.as_tensor(references>0,device=device)
        # Support flattening must be reference class, reference image, edge.
        self.sb=self.b[bank].permute(2,0,1,3,4).reshape(4,512,32).transpose(1,2).contiguous()
        self.sn=self.n[bank].permute(2,0,1,3,4).reshape(4,512,32).transpose(1,2).contiguous()

    def scores(self,ids,W):
        qb=self.b[ids].permute(1,0,2,3)
        qn=self.n[ids].permute(1,0,2,3)
        n=qb.shape[1]
        def bilinear(q,s):
            return torch.bmm((q@W).reshape(4,-1,32),s).reshape(4,n,8,8,8,8).permute(1,0,2,3,4,5)
        if self.arm=='body':
            sim=bilinear(qb,self.sb)
        elif self.arm=='additive':
            sim=.5*(bilinear(qb,self.sb)+bilinear(qn,self.sn))
        else:
            sim=.5*bilinear(qb-qn,self.sb-self.sn)
        logits=sim/edge.TAU+self.logmass[None,:,None]
        token=torch.logsumexp(logits.flatten(-2),dim=-1)
        mask=self.legal[ids]
        valid=mask.any(-1)[:,:,None]&self.ref_valid[None]
        token=torch.where(mask[:,:,:,None]&valid[:,:,None],token,0.)
        raw=token.sum(2)/mask.sum(-1).clamp_min(1)[:,:,None]
        center=torch.where(valid,raw,0.).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
        features=torch.where(valid,(raw-center)/self.divisor[None,:,None],0.)
        return features.transpose(1,2),raw.transpose(1,2),valid.transpose(1,2)


def delta(features,a):
    value=(features*a).sum(-1)
    return value-value.mean(-1,keepdim=True)


def infer(reader,model,ids,anchor):
    scores,features,raws=[],[],[]
    with torch.inference_mode():
        W=model.matrix()
        for start in range(0,len(ids),BATCH):
            part=ids[start:start+BATCH]
            z,raw,_=reader.scores(part,W)
            d=delta(z,model.a).cpu().numpy()
            scores.append(anchor[start:start+len(part)]+d)
            features.append(z.cpu().numpy())
            raws.append(raw.cpu().numpy())
    return np.concatenate(scores),np.concatenate(features),np.concatenate(raws)


def trace_query(table_index,arm,W,x=None,mass=None,bank=None):
    """Same keys/axes as fixed edge.trace_query, with one W in all four terms."""
    if x is None:
        _,x,mass,bank=edge.support.load()
    shuffle=int(arm.split('shuffle')[1]) if 'shuffle' in arm else None
    qb,qn,legal,mapping=edge.edges(x,mass,shuffle)
    W=np.asarray(W,dtype=float)
    out=dict(table_index=np.array(table_index),bank=bank,neighbor_map=mapping[np.r_[table_index,bank.ravel()]],
             legal_edges=legal[np.r_[table_index,bank.ravel()]])
    scores=[]
    for g in range(4):
        b,n=qb[table_index,g].astype(float),qn[table_index,g].astype(float)
        sb,sn=qb[bank,g].astype(float),qn[bank,g].astype(float)
        points=np.stack([np.einsum('id,crjd->cirj',b@W,sb),np.einsum('id,crjd->cirj',n@W,sn),
                         np.einsum('id,crjd->cirj',b@W,sn),np.einsum('id,crjd->cirj',n@W,sb)])
        sim=points[0] if arm=='body' else .5*(points[0]+points[1])
        if arm.startswith('mixed'):
            sim=.5*(points[0]+points[1]-points[2]-points[3])
        mask=legal[bank,g]
        references=mask.any(-1).sum(-1)
        lm=-np.log(np.maximum(mask.sum(-1),1))[:,:,None]-np.log(np.maximum(references,1))[:,None,None]
        lm=np.where(mask,lm,-np.inf)
        logref=logsumexp(sim/edge.TAU+lm[:,None],axis=-1)
        query=legal[table_index,g]
        has=query.any()&(references>0)
        value=logsumexp(logref,axis=-1)
        value=np.where(query[None]&has[:,None],value,0.)
        score=value.sum(-1)/max(query.sum(),1)
        responsibility=np.exp(sim/edge.TAU+lm[:,None]-value[:,:,None,None])
        responsibility=np.where(query[None,:,None,None]&has[:,None,None,None],responsibility,0.)
        out.update({f'{edge.GROUPS[g]}_four_points':points,f'{edge.GROUPS[g]}_similarity':sim,
                    f'{edge.GROUPS[g]}_log_reference':logref,f'{edge.GROUPS[g]}_edge_score':value,
                    f'{edge.GROUPS[g]}_responsibility':responsibility})
        scores.append(score)
    out['raw_scores']=np.stack(scores,-1)
    return out


def source_data():
    table,x,mass,bank=edge.support.load()
    idx=np.flatnonzero(table.role.isin(['utility_train','selection','evaluation']))
    role=table.iloc[idx].role.to_numpy()
    train,selection,evaluation=[np.flatnonzero(role==r) for r in ('utility_train','selection','evaluation')]
    assert [len(train),len(selection),len(evaluation)]==[644,612,485]
    scales=pd.read_csv(ROOT/'edge_difference/scales.csv')
    divisors={arm:scales[scales.arm==arm].sort_values('feature').divisor.to_numpy() for arm in ARMS}
    return table,x,mass,bank,idx,train,selection,evaluation,divisors


def smoke(data):
    table,x,mass,bank,idx,tr,va,te,divisors=data
    utility=idx[tr]
    missing=utility[mass[utility,1].sum(-1)==0]
    ids=np.unique(np.r_[utility[:5],missing[:1]])
    checks={}
    cache=np.load(ROOT/'edge_difference/localcache.npz')
    fixed_features=np.load(ROOT/'edge_difference/features.npz')
    for arm in ARMS:
        reader=Reader(x,mass,bank,arm,divisors[arm])
        model=Metric()
        with torch.inference_mode():
            z,raw,valid=reader.scores(ids,model.matrix())
        error=float(np.max(np.abs(raw.cpu().numpy()-cache[f'{arm}_raw'][ids])))
        zerror=float(np.max(np.abs(z.cpu().numpy()-fixed_features[arm][ids])))
        np.testing.assert_array_equal(valid.cpu().numpy(),cache[f'{arm}_valid'][ids])
        assert max(error,zerror)<1e-5,(arm,error,zerror)
        checks[f'{arm}_initial_raw']=error
        checks[f'{arm}_initial_features']=zerror
        del reader,model
    # Nonzero a and V make this test sensitive to the whole pre-pooling gradient path.
    reader=Reader(x,mass,bank,'mixed',divisors['mixed'],device='cpu',dtype=torch.float64)
    model=Metric(device='cpu',dtype=torch.float64)
    with torch.no_grad():
        model.a.copy_(torch.tensor([.1,-.2,.15,.07],dtype=torch.float64))
        model.V.copy_(torch.arange(128,dtype=torch.float64).reshape(32,4)*.0001)
    anchor=np.load(ROOT/'reference_dependence/frozen_marginal/predictions_17.npz')['anchor'][ids]
    labels=table.class_id.to_numpy(dtype=int)
    counts=np.bincount(labels[utility],minlength=8)
    global_weights=.5/len(utility)+.5/(8*counts[labels[ids]])
    b=torch.tensor(anchor,dtype=torch.float64)
    y=torch.tensor(labels[ids])
    w=torch.tensor(global_weights)
    def objective():
        z,_,_=reader.scores(ids,model.matrix())
        return (F.cross_entropy(b+delta(z,model.a),y,reduction='none')*w).sum()+model.regularizer()
    objective().backward()
    for name,pos in [('V',(0,0)),('a',(0,))]:
        parameter=getattr(model,name)
        exact=float(parameter.grad[pos])
        center=float(parameter[pos].detach())
        eps=1e-5
        with torch.no_grad():
            parameter[pos]=center+eps
            plus=float(objective())
            parameter[pos]=center-eps
            minus=float(objective())
            parameter[pos]=center
        numeric=(plus-minus)/(2*eps)
        assert abs(numeric-exact)<1e-7,(name,numeric,exact)
        checks[f'{name}_gradient']={'analytic':exact,'finite_difference':numeric,'error':abs(numeric-exact)}
    W=model.matrix().detach().numpy()
    rng=np.random.default_rng(17)
    q,s=rng.normal(size=(2,32)),rng.normal(size=(2,32))
    exchange=abs(float(q[0]@W@s[0]-s[0]@W@q[0]))
    four=.5*(q[0]@W@s[0]+q[1]@W@s[1]-q[0]@W@s[1]-q[1]@W@s[0])
    diff=.5*(q[0]-q[1])@W@(s[0]-s[1])
    assert max(exchange,abs(four-diff))<1e-12
    reverse=Reader(x[:,:,edge.REV],mass[:,:,edge.REV],bank,'mixed',divisors['mixed'],device='cpu',dtype=torch.float64)
    with torch.no_grad():
        before=reader.scores(ids,model.matrix())[0]
        after=reverse.scores(ids,model.matrix())[0]
    axis_error=float((before-after).abs().max())
    assert axis_error<1e-10
    checks.update(exchange_error=exchange,four_point_error=abs(float(four-diff)),axis_flip_error=axis_error,
                  checked_utility_indices=ids.tolist(),global_weight_sum=float((.5/644+.5/(8*counts[labels[utility]])).sum()),
                  no_evaluation_computed=True)
    (HERE/'smoke.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks),flush=True)


def fit(reader,anchor,table,idx,tr,va,seed,arm):
    model=Metric()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    y=table.class_id.to_numpy(dtype=int)
    ids=idx[tr]
    counts=np.bincount(y[ids],minlength=8)
    global_weights=.5/len(ids)+.5/(8*counts[y[ids]])
    b=torch.tensor(anchor[ids],device='cuda')
    labels=torch.tensor(y[ids],device='cuda')
    lw=torch.tensor(global_weights,device='cuda',dtype=torch.float32)
    best=edge.dep.exact_j(y[idx[va]],anchor[idx[va]].argmax(-1))
    chosen=model.state()
    selected=0
    history,states=[],{}
    for step in range(201):
        if step>0:
            optimizer.zero_grad(set_to_none=True)
            for start in range(0,len(ids),BATCH):
                end=min(start+BATCH,len(ids))
                z,_,_=reader.scores(ids[start:end],model.matrix())
                loss=(F.cross_entropy(b[start:end]+delta(z,model.a),labels[start:end],reduction='none')*lw[start:end]).sum()
                loss.backward()
            model.regularizer().backward()
            optimizer.step()
        if step not in CHECKPOINTS:
            continue
        utility,_,_=infer(reader,model,ids,anchor[ids])
        selection,_,_=infer(reader,model,idx[va],anchor[idx[va]])
        um=edge.dep.metrics(y[ids],utility)
        sm=edge.dep.metrics(y[idx[va]],selection)
        j=edge.dep.exact_j(y[idx[va]],selection.argmax(-1))
        state=model.state()
        states.update({f'{key}_{step}':value for key,value in state.items()})
        reg=float(model.regularizer().detach())
        history.append(dict(seed=seed,arm=arm,step=step,train_CE=.5*(um['NLL']+um['balanced_NLL']),
                            regularizer=reg,train_objective=.5*(um['NLL']+um['balanced_NLL'])+reg,
                            **{f'utility_{k}':v for k,v in um.items()},**{f'selection_{k}':v for k,v in sm.items()},
                            W_correction_norm=float(np.linalg.norm(state['W']-np.eye(32))),a_norm=float(np.linalg.norm(state['a'])),
                            exact_j=str(j)))
        if j>best:
            best,selected,chosen=j,step,state
        print('checkpoint',seed,arm,step,'utility_CE',history[-1]['train_CE'],'selection_J',sm['J'],flush=True)
    np.savez_compressed(HERE/f'checkpoints_{arm}_{seed}.npz',**states)
    model.restore(chosen)
    return model,chosen,selected,history


def main(data):
    started=time.perf_counter()
    table,x,mass,bank,idx,tr,va,te,divisors=data
    y=table.class_id.to_numpy(dtype=int)
    allids=np.arange(len(table))
    pd.read_csv(ROOT/'edge_difference/scales.csv').query('arm in @ARMS').to_csv(HERE/'scales.csv',index=False)
    results,receipts,curves,classes,pairs,margins=[],[],[],[],[],[]
    weights={}
    replay_max=0.
    for seed in SEEDS:
        old=np.load(ROOT/f'reference_dependence/frozen_marginal/predictions_{seed}.npz')
        np.testing.assert_array_equal(old['idx'],idx)
        anchor=old['anchor'].astype(np.float32)
        predictions=dict(anchor=anchor,full=old['full'])
        features={}
        for arm in ARMS:
            tick=time.perf_counter()
            reader=Reader(x,mass,bank,arm,divisors[arm])
            model,state,step,history=fit(reader,anchor,table,idx,tr,va,seed,arm)
            # Evaluation and full-table outputs are computed only after source selection.
            score,z,raw=infer(reader,model,allids,anchor)
            predictions[arm]=score
            features[arm]=z
            weights.update({f'{arm}_{seed}_{key}':value for key,value in state.items()})
            trace_idx=int(idx[te][0])
            trace=trace_query(trace_idx,arm,state['W'],x,mass,bank)
            replay=float(np.max(np.abs(trace['raw_scores']-raw[trace_idx])))
            assert replay<1e-5,replay
            replay_max=max(replay_max,replay)
            receipts.append(dict(seed=seed,arm=arm,selected_step=step,optimizer_steps=200,parameters=260,
                                 seconds=time.perf_counter()-tick,trace_error=replay,
                                 W_correction_norm=float(np.linalg.norm(state['W']-np.eye(32)))))
            curves.extend(history)
            fixed=np.load(ROOT/f'edge_difference/predictions_{seed}.npz')[arm]
            for role,select in [('utility_train',tr),('selection',va),('evaluation',te)]:
                results.append(dict(seed=seed,arm=arm,role=role,**edge.dep.metrics(y[idx[select]],score[idx[select]])))
                if role=='evaluation':
                    for reference_name,ref in [('anchor',anchor),('fixed_same_arm',fixed)]:
                        new=score[idx[select]].argmax(-1)==y[idx[select]]
                        prior=ref[idx[select]].argmax(-1)==y[idx[select]]
                        pairs.append(dict(seed=seed,arm=arm,reference=reference_name,rescued=int((new&~prior).sum()),
                                          harmed=int((~new&prior).sum()),net=int(new.sum()-prior.sum())))
            rival_scores=anchor[idx].copy()
            rival_scores[np.arange(len(idx)),y[idx]]=-np.inf
            rival=rival_scores.argmax(-1)
            initial=np.load(ROOT/'edge_difference/features.npz')[arm]
            for j,i in enumerate(idx):
                for g,name in enumerate(edge.GROUPS):
                    gap=z[i,y[i],g]-z[i,rival[j],g]
                    margins.append(dict(seed=seed,arm=arm,table_index=int(i),obs_id=int(table.iloc[i].obs_id),
                                        role=table.iloc[i].role,true_class=int(y[i]),rival=int(rival[j]),group=name,
                                        true_score=float(z[i,y[i],g]),rival_score=float(z[i,rival[j],g]),gap=float(gap),
                                        initial_gap=float(initial[i,y[i],g]-initial[i,rival[j],g]),
                                        coefficient=float(state['a'][g]),margin_contribution=float(gap*state['a'][g])))
            for c in range(8):
                select=idx[te][y[idx[te]]==c]
                classes.append(dict(seed=seed,arm=arm,class_id=c,**edge.dep.metrics(y[select],score[select])))
            pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
            pd.DataFrame(curves).to_csv(HERE/'training_curve.csv',index=False)
            np.savez_compressed(HERE/'weights.npz',**weights)
            print('SELECTED',seed,arm,step,edge.dep.metrics(y[idx[te]],score[idx[te]]),flush=True)
            del reader,model
        for arm in ('anchor','full'):
            for role,select in [('utility_train',tr),('selection',va),('evaluation',te)]:
                results.append(dict(seed=seed,arm=arm,role=role,**edge.dep.metrics(y[idx[select]],predictions[arm][idx[select]])))
        for arm in ARMS:
            current=predictions[arm][idx[te]].argmax(-1)==y[idx[te]]
            for reference in ('body','additive','mixed'):
                prior=predictions[reference][idx[te]].argmax(-1)==y[idx[te]]
                pairs.append(dict(seed=seed,arm=arm,reference=reference,rescued=int((current&~prior).sum()),
                                  harmed=int((~current&prior).sum()),net=int(current.sum()-prior.sum())))
        np.savez_compressed(HERE/f'predictions_{seed}.npz',idx=idx,bank=bank,**predictions)
        np.savez_compressed(HERE/f'features_{seed}.npz',idx=idx,bank=bank,**features)
    for name,rows in [('metrics',results),('class_results',classes),('paired_results',pairs),('query_group_margins',margins)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    summary=pd.DataFrame(results).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean()
    summary.to_csv(HERE/'summary.csv')
    receipt=dict(status='complete',fits=len(receipts),optimizer_steps_per_fit=200,batch_size=BATCH,
                 global_utility_count=644,global_classes=8,regularizer_once_per_optimizer_step=True,
                 rank=4,channel_parameters=256,summary_parameters=4,U_init_seed=INIT_SEED,U_init_std=INIT_STD,
                 RMS='fixed source utility RMS at W=I, copied from edge_difference/scales.csv',
                 initial_V_zero=True,initial_a_zero=True,checkpoint_steps=list(CHECKPOINTS),
                 evaluation_after_selection_only=True,no_external=True,no_git=True,no_images=True,
                 max_trace_error=replay_max,seconds=time.perf_counter()-started)
    assert len(receipts)==15
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(summary.to_string(),flush=True)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    data=source_data()
    if '--smoke' in sys.argv:
        smoke(data)
    else:
        assert (HERE/'smoke.json').exists(),'Run --smoke before training.'
        main(data)
