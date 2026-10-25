"""Matched independent/joint polar readouts on frozen M3 matching evidence."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import importlib.util
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
LOCAL=BASE/'local_evidence_transport'
SEEDS=(17,29,43)
LEARNED=('vv','vh','joint','joint_wide','context')
ARMS=LEARNED+('late_average',)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


previous=load('polar_previous',HERE.parent/'intervention_m3_20261025/train.py')
prior=previous.prior
metrics,write_json=previous.metrics,previous.write_json


class Data:
    def __init__(self):
        self.table=pd.read_csv(HERE.parent/'native_validation/table.csv',dtype={'mmsi':str})
        self.n=1741;self.up=self.table.up_index.to_numpy(int)
        self.tr=np.flatnonzero(self.table.role.eq('utility_train'))
        self.va=np.flatnonzero(self.table.role.eq('selection'))
        self.te=np.flatnonzero(self.table.role.eq('evaluation'))
        assert (len(self.tr),len(self.va),len(self.te))==(644,612,485)
        for a,b in ((self.tr,self.va),(self.tr,self.te),(self.va,self.te)):
            for field in ('mmsi','pass_group'):
                assert not set(self.table.iloc[a][field])&set(self.table.iloc[b][field])


class Head(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        indices={'vv':list(range(6))+[12,13,14],
                 'vh':list(range(6,15)), 'context':[12,13,14]}.get(arm,list(range(15)))
        self.register_buffer('indices',torch.tensor(indices))
        width=16 if arm=='joint_wide' else 12
        self.net=nn.Sequential(nn.Linear(len(indices),width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        d=self.net(x[...,self.indices]).squeeze(-1)
        return d-d.mean(-1,keepdim=True)


def make_model(arm,seed):
    torch.manual_seed(seed)
    return Head(arm).cuda()


def raw_features(local,base):
    z=base-base.mean(-1,keepdims=True)
    other=np.stack([np.delete(z,c,axis=1).max(1) for c in range(8)],1)
    return np.concatenate((local.reshape(len(local),8,12),
        np.stack((z,other,z-other),-1)), -1)


def normalize(raw,tr,scaler=None):
    if scaler is None:
        mu=raw[tr].mean((0,1));sd=raw[tr].std((0,1));active=sd>1e-6
        scaler=(mu,np.where(active,sd,1),active)
    mu,sd,active=scaler
    x=np.where(active,(raw-mu)/sd,0).astype(np.float32)
    return torch.tensor(x,device='cuda'),scaler


def prepare(data,seed):
    reader=prior.Reader('pair_joint',seed)
    lookup={int(v):i for i,v in enumerate(reader.ids)}
    ix=np.array([lookup[int(v)] for v in data.up[:data.n]])
    z=np.load(LOCAL/'pair_conditioned_joint_metric/results'/f'pair_joint_{seed}.npz')
    local=z['local'][ix].copy();oldscore=z['score'][ix].copy()
    base=reader.base[ix].copy();strong=reader.strong[ix].copy()
    plain=reader.oldlocal[ix].cpu().numpy()
    query_valid=(reader.mass[ix].sum(-1)>0).cpu().numpy()
    with torch.no_grad():
        replay=(torch.tensor(base,device='cuda')+
            reader.delta(torch.tensor(local,device='cuda'),torch.tensor(ix,device='cuda'))).cpu().numpy()
    error=float(abs(replay-oldscore).max());assert error<3e-5,error
    x,scaler=normalize(raw_features(local,base),data.tr)
    np.savez_compressed(HERE/'results'/f'inputs_{seed}.npz',local=local,plain=plain,base=base,
        oldscore=oldscore,strong=strong,query_valid=query_valid,ix=ix,up_index=data.up[:data.n],
        mu=scaler[0],scale=scaler[1],active=scaler[2])
    return reader,ix,x,base,scaler,error


def fit(data,seed,arm,x,base):
    tick=time.perf_counter();model=make_model(arm,seed)
    y=data.table.iloc[:data.n].class_id.to_numpy(int)
    yt=torch.tensor(y[data.tr],device='cuda')
    counts=np.bincount(y[data.tr],minlength=8)
    weights=torch.tensor(.5/len(data.tr)+.5/(8*counts[y[data.tr]]),device='cuda',dtype=torch.float32)
    bt=torch.tensor(base[data.tr],device='cuda');xt=x[data.tr]
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    best=prior.old.exact_j(y[data.va],base[data.va].argmax(1));chosen=0
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    curves=[dict(seed=seed,arm=arm,step=0,loss=None,**metrics(y[data.va],base[data.va]))]
    for step in range(1,201):
        opt.zero_grad(set_to_none=True)
        loss=(F.cross_entropy(bt+model(xt),yt,reduction='none')*weights).sum()
        loss.backward();assert torch.isfinite(loss);opt.step()
        if step%5==0:
            with torch.no_grad():s=base[data.va]+model(x[data.va]).cpu().numpy()
            m=metrics(y[data.va],s);j=prior.old.exact_j(y[data.va],s.argmax(1))
            curves.append(dict(seed=seed,arm=arm,step=step,loss=float(loss.detach()),**m))
            if j>best:
                best=j;chosen=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    receipt=dict(seed=seed,arm=arm,selected_step=chosen,parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter()-tick,source_J=float(best))
    torch.save(dict(state=state,receipt=receipt),HERE/'models'/f'{arm}_{seed}.pt')
    torch.save(dict(state=model.state_dict(),receipt=receipt,source_only=True),HERE/'models'/f'final_{arm}_{seed}.pt')
    pd.DataFrame(curves).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv',index=False)
    print('FIT',receipt,flush=True)
    return receipt


def restore(seed,arm):
    model=make_model(arm,seed)
    ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state']);model.eval()
    return model


def checks(data,seed,x,base):
    idx=data.tr[:4];result={}
    for arm in LEARNED:
        model=make_model(arm,seed)
        with torch.no_grad():
            assert float(model(x[idx]).abs().max())==0
            model.net[-1].weight.normal_(0,.1)
            altered=x[idx].clone()
            omitted=sorted(set(range(15))-set(model.indices.cpu().tolist()))
            if omitted:
                altered[...,omitted]+=3.14159
                assert float((model(altered)-model(x[idx])).abs().max())==0
        result[arm]=dict(parameters=sum(p.numel() for p in model.parameters()),input_indices=model.indices.cpu().tolist())
    model=make_model('joint',seed)
    yy=torch.tensor(data.table.iloc[idx].class_id.to_numpy(int),device='cuda')
    bb=torch.tensor(base[idx],device='cuda')
    F.cross_entropy(bb+model(x[idx]),yy).backward()
    grad=model.net[-1].weight.grad;col=int(grad.abs().argmax());auto=float(grad[0,col]);values=[]
    with torch.no_grad():
        for d in (.001,-.001):
            model.net[-1].weight[0,col]=d
            values.append(float(F.cross_entropy(bb+model(x[idx]),yy)))
    num=(values[0]-values[1])/.002
    assert abs(auto-num)<max(1e-4,.02*abs(auto)),(auto,num)
    result['gradient']=dict(auto=auto,numeric=num)
    return result


@torch.inference_mode()
def local_cross(reader,metric,ix):
    output={k:[] for k in ('plain','K_only','T_only','both')}
    h=metric.project(reader.d)
    for first in range(0,len(ix),64):
        ii=torch.tensor(ix[first:first+64],device='cuda');ref=reader.refs[ii]
        q,s=reader.x[ii],reader.x[ref];hq,hs=h[ii],h[ref]
        a,b=reader.mass[ii],reader.mass[ref];parts={k:[] for k in output}
        for r in prior.old.REGIONS:
            qi=q[:,:,r][:,None,None,:,:,None,:];sj=s[...,r,:][...,None,:,:]
            hi=hq[:,:,r][:,None,None,:,:,None,:];hj=hs[...,r,:][...,None,:,:]
            k0=(qi*sj).sum(-1);k1=prior.similarity(qi,sj,metric.weights(hi,hj))
            qa=a[:,None,None,:,r];sa=b[...,r]
            t0,*_=prior.old.sinkhorn(k0,qa,sa);t1,*_=prior.old.sinkhorn(k1,qa,sa)
            for name,t,k in (('plain',t0,k0),('K_only',t0,k1),('T_only',t1,k0),('both',t1,k1)):
                parts[name].append(prior.old.pair_score(t,k,qa,sa))
        for name in output:
            output[name].append(torch.stack(parts[name],-2).mean(2).cpu().numpy())
    return {k:np.concatenate(v) for k,v in output.items()}


def main(check_only=False):
    tick=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for folder in ('models','results','analysis'):(HERE/folder).mkdir(exist_ok=True)
    data=Data();receipts=[];verification={}
    for seed in SEEDS:
        reader,ix,x,base,scaler,replay=prepare(data,seed)
        verification[seed]=dict(old_head_replay=replay,**checks(data,seed,x,base))
        write_json(HERE/'checks.json',verification)
        if check_only:print(verification,flush=True);return
        for arm in LEARNED:
            receipts.append(fit(data,seed,arm,x,base))
            pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
        del reader,x;torch.cuda.empty_cache()
    write_json(HERE/'selection_frozen.json',dict(fits=len(receipts),note='All source choices frozen before development evaluation.'))
    rows=[];times=[]
    for seed in SEEDS:
        reader=prior.Reader('pair_joint',seed)
        z=np.load(HERE/'results'/f'inputs_{seed}.npz')
        ix,base=z['ix'],z['base'];scaler=(z['mu'],z['scale'],z['active'])
        metric=prior.Metric('pair_joint').cuda()
        ck=torch.load(LOCAL/'pair_conditioned_joint_metric/models'/f'pair_joint_{seed}.pt',map_location='cpu',weights_only=False)
        metric.load_state_dict(ck['state']);metric.eval().requires_grad_(False)
        start=time.perf_counter();variants=local_cross(reader,metric,ix)
        times.append(dict(seed=seed,stage='four_KT_variants_cached_features',n=data.n,seconds=time.perf_counter()-start))
        np.testing.assert_allclose(variants['both'],z['local'],atol=2e-5,rtol=1e-5)
        np.testing.assert_allclose(variants['plain'],z['plain'],atol=2e-5,rtol=1e-5)
        np.savez_compressed(HERE/'results'/f'local_cross_{seed}.npz',**variants,up_index=data.up[:data.n])
        x,_=normalize(raw_features(variants['both'],base),data.tr,scaler)
        scores={'old_full':z['oldscore'].copy(),'base':base,'strong':z['strong'].copy()}
        models={a:restore(seed,a) for a in LEARNED}
        with torch.no_grad():
            for arm,model in models.items():scores[arm]=base+model(x).cpu().numpy()
        scores['late_average']=(scores['vv']+scores['vh'])*.5
        # Evaluate the identical local change through each frozen fitted head.
        crossscores={}
        for mode,local in variants.items():
            xx,_=normalize(raw_features(local,base),data.tr,scaler)
            with torch.no_grad():
                cc={a:base+m(xx).cpu().numpy() for a,m in models.items()}
                cc['late_average']=(cc['vv']+cc['vh'])*.5
                cc['old_full']=(torch.tensor(base,device='cuda')+
                    reader.delta(torch.tensor(local,device='cuda'),torch.tensor(ix,device='cuda'))).cpu().numpy()
            for arm,s in cc.items():crossscores[arm+'__'+mode]=s
        np.savez_compressed(HERE/'results'/f'cross_scores_{seed}.npz',**crossscores)
        for arm,s in scores.items():
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',score=s,up_index=data.up[:data.n])
            for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
                rows.append(dict(seed=seed,arm=arm,role=role,**metrics(data.table.iloc[ids].class_id.to_numpy(int),s[ids])))
        del reader,models,x;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(times).to_csv(HERE/'stage_cost.csv',index=False)
    write_json(HERE/'complete.json',dict(fits=len(receipts),seconds=time.perf_counter()-tick))
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean().to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');main(p.parse_args().check_only)
