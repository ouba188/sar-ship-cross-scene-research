"""Nine source-only antisymmetric heads; no target/test prediction."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys
sys.dont_write_bytecode=True
import itertools
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT))
import train_heads as source
P=source.parent
ARMS=['global','marginal','paired']
SEEDS=[17,29,43]
PAIRS=np.array(list(itertools.combinations(range(8),2)),dtype=np.int64)
SWAP=np.array([0,1,2,4,3,6,5,8,7,10,9,11,12,13,14])
SIGN=np.ones(15);SIGN[[0,1,2,12,14]]=-1


def raw_input(base,mean,marginal,pair,pairs=PAIRS):
    c,d=pairs.T
    cols=[base[:,c]-base[:,d],mean[:,0,c]-mean[:,0,d],mean[:,1,c]-mean[:,1,d]]
    for pol in range(2):
        for stat in range(2):cols.extend([marginal[:,pol,c,stat],marginal[:,pol,d,stat]])
    for pol in range(2):cols.extend([pair[:,pol,:,0],pair[:,pol,:,1]])
    return np.stack(cols,-1)


def flip(raw):return raw[...,SWAP]*SIGN


def ablate(raw,arm):
    out=raw.copy()
    out[...,{'global':3,'marginal':11,'paired':15}[arm]:]=0
    return out


def tokens(raw,mu,scale):
    return tuple(torch.tensor((v-mu)/scale,dtype=torch.float32) for v in [raw,flip(raw)])


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(15,12),nn.GELU(),nn.Linear(12,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        incidence=torch.zeros(28,8)
        incidence[torch.arange(28),torch.tensor(PAIRS[:,0])]=1
        incidence[torch.arange(28),torch.tensor(PAIRS[:,1])]=-1
        self.register_buffer('incidence',incidence)

    def flux(self,forward,reverse):
        return .5*(self.net(forward)-self.net(reverse)).squeeze(-1)

    def forward(self,forward,reverse):return self.flux(forward,reverse)@self.incidence/7


def restore(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=Head().eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck,model


def load_data():
    with np.load(HERE.parent/'source_moments.npz') as z:
        cache={k:z[k] for k in ['sample_ids','pairs','pair_moments','marginal','mean','train','validation','reader']}
    np.testing.assert_array_equal(cache['pairs'],PAIRS)
    allrows=pd.read_csv(ROOT/'cohort.csv',dtype={'mmsi':str})
    lookup=pd.Series(np.arange(len(allrows)),index=allrows.sample_id)
    mapping=lookup.loc[cache['sample_ids']].to_numpy()
    table=allrows.iloc[mapping].reset_index(drop=True)
    tr,va=cache['train'],cache['validation']
    assert len(tr)==1542 and len(va)==279 and len(table)==1821
    np.testing.assert_array_equal(np.sort(np.r_[tr,va]),np.arange(len(table)))
    with np.load(ROOT/'readers/fold0_inputs.npz') as z:
        np.testing.assert_array_equal(mapping[tr],z['train'])
        np.testing.assert_array_equal(mapping[va],z['validation'])
    for key in ['mmsi','pass_group']:
        assert not set(table.iloc[tr][key])&set(table.iloc[va][key])
    local={int(v):i for i,v in enumerate(mapping)}
    for inner in range(4):
        with np.load(ROOT/'readers/models'/f'fold0_inner{inner}.npz') as z:
            held,fit=z['held'],z['fit']
        assert np.isin(fit,mapping[tr]).all()
        ix=np.array([local[int(v)] for v in held])
        assert (cache['reader'][ix]==inner+1).all()
        for key in ['mmsi','pass_group']:
            assert not set(allrows.iloc[held][key])&set(allrows.iloc[fit][key])
    assert (cache['reader'][va]==0).all()
    return table,cache,mapping


def seed_base(seed,cache,mapping):
    tr,va=cache['train'],cache['validation']
    with np.load(ROOT/'calibrated'/f'fold0_seed{seed}.npz') as z:
        np.testing.assert_array_equal(z['sample_ids'][mapping],cache['sample_ids'])
        b=np.empty((len(mapping),8),float)
        b[tr]=z['train_base'][mapping[tr]]
        b[va]=z['base'][mapping[va]]
    return b


def permuted_raw(base,cache,ix,perm):
    loc={tuple(v):i for i,v in enumerate(PAIRS)}
    a,b=perm[PAIRS[:,0]],perm[PAIRS[:,1]]
    take=np.array([loc[tuple(sorted((int(c),int(d))))] for c,d in zip(a,b)])
    pm=cache['pair_moments'][ix][:,:,take].copy()
    pm[...,1]*=np.where(a<b,1.,-1.)[None,None,:]
    return raw_input(base[ix][:,perm],cache['mean'][ix][:,:,perm],
                     cache['marginal'][ix][:,:,perm],pm)


def structural(model,raw,base,cache,ix,arm,mu,scale):
    x,xr=tokens(raw[ix],mu,scale)
    pm=cache['pair_moments'][ix].copy();pm[...,1]*=-1
    independently_reversed=ablate(raw_input(base[ix],cache['mean'][ix],cache['marginal'][ix],pm,PAIRS[:,::-1]),arm)
    np.testing.assert_array_equal(flip(raw[ix]),independently_reversed)
    with torch.no_grad():
        h=model.flux(x,xr);d=model(x,xr)
        reverse_error=float((model.flux(xr,x)+h).abs().max())
        sum_error=float(d.sum(1).abs().max())
        perm=np.array([3,0,7,1,6,4,2,5])
        pr=ablate(permuted_raw(base,cache,ix,perm),arm)
        dp=model(*tokens(pr,mu,scale))
        perm_error=float((dp-d[:,perm]).abs().max())
    assert reverse_error<1e-6 and sum_error<2e-6 and perm_error<2e-6
    return dict(raw_flip_exact=True,flux_flip_error=reverse_error,
        delta_sum_error=sum_error,class_permutation_error=perm_error,max_abs_h=float(h.abs().max()))


def main():
    started=time.perf_counter()
    assert (HERE/'PROTOCOL_ZH.md').exists()
    (HERE/'heads').mkdir(exist_ok=True);(HERE/'validation_flux').mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    table,cache,mapping=load_data()
    tr,va=cache['train'],cache['validation'];y=table.class_id.to_numpy()
    source_rows=table.copy();source_rows['global_row']=mapping
    source_rows['source_row']=np.arange(len(table));source_rows['reader']=cache['reader']
    source_rows['role']=np.where(np.isin(np.arange(len(table)),tr),'train','validation')
    source_rows.to_csv(HERE/'source_rows.csv',index=False)
    selected,curves,checks=[],[],[]
    for seed in SEEDS:
        base=seed_base(seed,cache,mapping)
        raw_all=raw_input(base,cache['mean'],cache['marginal'],cache['pair_moments'])
        initial=None
        for arm in ARMS:
            tick=time.perf_counter();raw=ablate(raw_all,arm)
            fit=np.concatenate([raw[tr],flip(raw[tr])],axis=1)
            mu,scale=fit.mean((0,1)),np.maximum(fit.std((0,1)),1e-8)
            x,xr=tokens(raw[tr],mu,scale);xv,xvr=tokens(raw[va],mu,scale)
            torch.manual_seed(seed);model=Head()
            assert sum(p.numel() for p in model.parameters())==205
            if initial is None:initial=P.frozen_state(model)
            else:
                for k,v in model.state_dict().items():assert torch.equal(v,initial[k])
            with torch.no_grad():assert not model(xv,xvr).any()
            best,epoch_best=P.frozen_state(model),0
            best_j=P.exact_j(y[va],base[va].argmax(1))
            meta=dict(fold=0,seed=seed,arm=arm)
            curves.append(dict(**meta,epoch=0,train_loss=np.nan,**P.assessment(y[va],base[va])))
            opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
            b=torch.tensor(base[tr],dtype=torch.float32);yy=torch.tensor(y[tr])
            for epoch in range(1,201):
                opt.zero_grad();loss=P.joint_loss(b+model(x,xr),yy)
                assert torch.isfinite(loss)
                loss.backward();opt.step()
                if epoch%5:continue
                with torch.no_grad():score=base[va]+model(xv,xvr).numpy().astype(float)
                j=P.exact_j(y[va],score.argmax(1))
                curves.append(dict(**meta,epoch=epoch,train_loss=float(loss.detach()),**P.assessment(y[va],score)))
                if j>best_j:best_j,epoch_best,best=j,epoch,P.frozen_state(model)
            path=HERE/'heads'/f'fold0_{arm}_seed{seed}.pt'
            torch.save(dict(**meta,state=best,mu=mu,scale=scale,pairs=PAIRS,epoch=epoch_best,
                            parameters=205,normalization=7,base_seed=seed),path)
            ck,replay=restore(path)
            with torch.no_grad():score=base[va]+replay(*tokens(raw[va],ck['mu'],ck['scale'])).numpy().astype(float)
            assert P.exact_j(y[va],score.argmax(1))==best_j
            check=structural(replay,raw,base,cache,va,arm,mu,scale)
            checks.append(dict(**meta,**check,checkpoint_replay=True))
            selected.append(dict(**meta,epoch=epoch_best,parameters=205,validation_J=float(best_j),
                seconds=time.perf_counter()-tick,path=str(path)))
            print(seed,arm,epoch_best,float(best_j),flush=True)
    pd.DataFrame(selected).to_csv(HERE/'selections.csv',index=False)
    pd.DataFrame(curves).to_csv(HERE/'selection_curves.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'checks.csv',index=False)
    assert len(selected)==9
    (HERE/'selection_frozen.json').write_text(json.dumps(dict(heads=9,test_evaluated=False)),encoding='utf-8')
    frames,metrics,pathrows=[],[],[]
    for seed in SEEDS:
        base=seed_base(seed,cache,mapping)
        full=raw_input(base,cache['mean'],cache['marginal'],cache['pair_moments'])
        for arm in ARMS:
            ck,model=restore(HERE/'heads'/f'fold0_{arm}_seed{seed}.pt')
            raw=ablate(full,arm)
            for role,ix in [('train',tr),('validation',va)]:
                xx,rr=tokens(raw[ix],ck['mu'],ck['scale'])
                with torch.no_grad():
                    h=model.flux(xx,rr).numpy().astype(float)
                    delta=model(xx,rr).numpy().astype(float)
                score=base[ix]+delta
                g=source_rows.iloc[ix][['sample_id','class_id','class_name','mmsi','pass_group','scene_id','chip_path_vv','chip_path_vh','global_row','source_row']].copy()
                g=g.rename(columns={'sample_id':'instance_id','scene_id':'product'})
                for k,v in dict(fold=0,seed=seed,arm=arm,role=role,epoch=ck['epoch']).items():g[k]=v
                g['base_prediction']=base[ix].argmax(1);g['prediction']=score.argmax(1)
                g['net']=g.prediction.eq(g.class_id).astype(int)-g.base_prediction.eq(g.class_id).astype(int)
                g['outcome']=np.select([g.net.eq(1),g.net.eq(-1),g.prediction.eq(g.class_id)],['rescued','harmed','CC'],default='WW')
                for c in range(8):g[f'b{c}'],g[f's{c}']=base[ix,c],score[:,c]
                frames.append(g)
                a=P.assessment(y[ix],score);bmet=P.assessment(y[ix],base[ix])
                metrics.append(dict(seed=seed,arm=arm,role=role,net=int(g.net.sum()),**a,
                    **{f'base_{k}':v for k,v in bmet.items() if k!='n'},
                    **{f'delta_{k}':a[k]-bmet[k] for k in ['OA','BA','NLL','balanced_NLL','Brier']}))
                if role=='validation':
                    np.savez_compressed(HERE/'validation_flux'/f'{arm}_{seed}.npz',sample_ids=table.iloc[ix].sample_id.to_numpy(str),
                        source_rows=ix,global_rows=mapping[ix],pairs=PAIRS,h=h,B=base[ix],s=score,
                        class_ids=y[ix],seed=seed,epoch=ck['epoch'])
                    hm=np.zeros((len(ix),8,8));hm[:,PAIRS[:,0],PAIRS[:,1]]=h;hm[:,PAIRS[:,1],PAIRS[:,0]]=-h
                    for j,ii in enumerate(ix):
                        truth=int(y[ii]);rival=int(np.argmax(np.where(np.arange(8)==truth,-np.inf,score[j])))
                        old=int(np.argmax(np.where(np.arange(8)==truth,-np.inf,base[ii])))
                        direct=2*hm[j,truth,rival]/7
                        third=sum(hm[j,truth,k]-hm[j,rival,k] for k in range(8) if k not in [truth,rival])/7
                        actual=delta[j,truth]-delta[j,rival]
                        assert abs(direct+third-actual)<2e-6
                        pathrows.append(dict(seed=seed,arm=arm,instance_id=table.iloc[ii].sample_id,class_id=truth,
                            rival=rival,old_rival=old,direct_pair_delta=direct,third_class_delta=third,
                            margin_delta=actual,closure_error=direct+third-actual,
                            base_margin=base[ii,truth]-base[ii,rival],final_margin=score[j,truth]-score[j,rival]))
    predictions=pd.concat(frames,ignore_index=True);predictions.to_csv(HERE/'predictions.csv',index=False)
    met=pd.DataFrame(metrics);met.to_csv(HERE/'metrics.csv',index=False)
    paths=pd.DataFrame(pathrows);paths.to_csv(HERE/'validation_paths.csv',index=False)
    paired=[]
    for seed in SEEDS:
        for role in ['train','validation']:
            part=met[met.seed.eq(seed)&met.role.eq(role)].set_index('arm')
            for left,right in [('paired','marginal'),('paired','global'),('marginal','global')]:
                paired.append(dict(seed=seed,role=role,left=left,right=right,
                    **{f'delta_{k}':part.loc[left,k]-part.loc[right,k] for k in ['OA','BA','NLL','balanced_NLL','Brier','net']}))
    pd.DataFrame(paired).to_csv(HERE/'paired_metrics.csv',index=False)
    cases=[]
    for arm in ARMS:
        for outcome in ['rescued','harmed','CC','WW']:
            candidates=predictions[predictions.seed.eq(17)&predictions.role.eq('validation')&predictions.arm.eq(arm)&predictions.outcome.eq(outcome)].sort_values('instance_id')
            if len(candidates):
                row=candidates.iloc[0].to_dict();row['available']=True
                row.update(paths[paths.seed.eq(17)&paths.arm.eq(arm)&paths.instance_id.eq(row['instance_id'])].iloc[0].to_dict())
                cases.append(row)
            else:cases.append(dict(seed=17,role='validation',arm=arm,outcome=outcome,available=False))
    pd.DataFrame(cases).to_csv(HERE/'sourcecases.csv',index=False)
    info=dict(heads=9,parameters=205,steps=1800,source_train=1542,source_validation=279,test_evaluated=False,
        seconds=time.perf_counter()-started,zero_epoch_heads=sum(s['epoch']==0 for s in selected),
        max_flip_error=max(c['flux_flip_error'] for c in checks),
        max_delta_sum_error=max(c['delta_sum_error'] for c in checks),
        max_class_permutation_error=max(c['class_permutation_error'] for c in checks),
        max_margin_closure_error=float(paths.closure_error.abs().max()),
        note='Source development only; train heads and T are not second-level OOF')
    (HERE/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(met[met.role.eq('validation')][['arm','seed','net','OA','BA','NLL','delta_OA','delta_BA']].to_string(index=False),flush=True)
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
