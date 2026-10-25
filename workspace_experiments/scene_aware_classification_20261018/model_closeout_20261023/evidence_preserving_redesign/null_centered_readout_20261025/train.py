"""Matched null-centered versus missing-only masked polar readout training."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import importlib.util
import argparse
import time
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'polar_readout_diagnosis_20261025'
BASE=HERE.parents[1];LOCAL=BASE/'local_evidence_transport'
SEEDS=(17,29,43)
LEARNED=('center_vv','center_vh','mask_vv','mask_vh','center_joint')
ARMS=LEARNED+('center_late','mask_late')
COMPARATORS=('old_full','base','strong','prior_vv','prior_vh','prior_late','prior_joint','prior_wide')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod);return mod

old=load('null_previous_train',PREVIOUS/'train.py')
Data,metrics,write_json=old.Data,old.metrics,old.write_json

def inputs(data,seed,local=None):
    z=np.load(PREVIOUS/'results'/f'inputs_{seed}.npz')
    local=z['local'] if local is None else local
    x,_=old.normalize(old.raw_features(local,z['base']),data.tr,(z['mu'],z['scale'],z['active']))
    available=torch.tensor(local[...,1].sum((1,3))>0,device='cuda')
    null=np.where(z['active'],-z['mu']/z['scale'],0).astype(np.float32)
    return z,x,available,null

class Head(nn.Module):
    def __init__(self,arm,null):
        super().__init__();self.arm=arm
        self.pol=0 if arm.endswith('vv') else 1 if arm.endswith('vh') else None
        indices=list(range(15)) if self.pol is None else list(range(self.pol*6,self.pol*6+6))+[12,13,14]
        self.register_buffer('indices',torch.tensor(indices))
        self.register_buffer('null',torch.tensor(null))
        width=16 if self.pol is None else 12
        self.net=nn.Sequential(nn.Linear(len(indices),width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,available):
        value=self.net(x[...,self.indices]).squeeze(-1)
        if self.arm.startswith('center'):
            empty=x.clone();empty[...,:12]=self.null[:12]
            value=value-self.net(empty[...,self.indices]).squeeze(-1)
        value=value-value.mean(-1,keepdim=True)
        if self.arm.startswith('mask'):
            value=value*available[:,self.pol,None]
        return value

def make_model(arm,seed,null):
    torch.manual_seed(seed);return Head(arm,null).cuda()

def fit(data,seed,arm,x,available,null,base):
    tick=time.perf_counter();m=make_model(arm,seed,null)
    y=data.table.iloc[:data.n].class_id.to_numpy(int);yt=torch.tensor(y[data.tr],device='cuda')
    count=np.bincount(y[data.tr],minlength=8)
    w=torch.tensor(.5/len(data.tr)+.5/(8*count[y[data.tr]]),device='cuda',dtype=torch.float32)
    b=torch.tensor(base[data.tr],device='cuda');opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01)
    best=old.prior.old.exact_j(y[data.va],base[data.va].argmax(1));chosen=0
    state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    curve=[dict(step=0,loss=None,**metrics(y[data.va],base[data.va]))]
    for step in range(1,201):
        opt.zero_grad(set_to_none=True)
        loss=(F.cross_entropy(b+m(x[data.tr],available[data.tr]),yt,reduction='none')*w).sum()
        loss.backward();assert torch.isfinite(loss);opt.step()
        if step%5==0:
            with torch.no_grad():s=base[data.va]+m(x[data.va],available[data.va]).cpu().numpy()
            curve.append(dict(step=step,loss=float(loss.detach()),**metrics(y[data.va],s)))
            j=old.prior.old.exact_j(y[data.va],s.argmax(1))
            if j>best:
                best=j;chosen=step;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    receipt=dict(seed=seed,arm=arm,selected_step=chosen,parameters=sum(p.numel() for p in m.parameters()),
        source_J=float(best),seconds=time.perf_counter()-tick)
    torch.save(dict(state=state,receipt=receipt),HERE/'models'/f'{arm}_{seed}.pt')
    pd.DataFrame(curve).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv',index=False)
    print('FIT',receipt,flush=True);return receipt

def restore(arm,seed,null):
    m=make_model(arm,seed,null)
    ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(ck['state']);return m.eval()

def checks(data,seed,x,available,null,base):
    result={};empty=x[:12].clone();empty[...,:12]=torch.tensor(null[:12],device='cuda')
    for arm in LEARNED:
        m=make_model(arm,seed,null)
        with torch.no_grad():
            assert float(m(x[:12],available[:12]).abs().max())==0
            m.net[-1].weight.normal_(0,.2)
            error=float(m(empty,torch.zeros_like(available[:12])).abs().max());assert error==0,error
            if m.pol is not None:
                missing=~available[:,m.pol]
                if missing.any():assert float(m(x[missing],available[missing]).abs().max())==0
                changed=x[:12].clone();other=1-m.pol;changed[...,other*6:other*6+6]+=2
                assert float((m(changed,available[:12])-m(x[:12],available[:12])).abs().max())==0
        result[arm]=dict(null_max_error=error,parameters=sum(p.numel() for p in m.parameters()))
    m=make_model('center_vh',seed,null);ids=data.tr[:16]
    y=torch.tensor(data.table.iloc[ids].class_id.to_numpy(int),device='cuda');b=torch.tensor(base[ids],device='cuda')
    F.cross_entropy(b+m(x[ids],available[ids]),y).backward()
    g=m.net[-1].weight.grad;col=int(g.abs().argmax());auto=float(g[0,col]);assert abs(auto)>1e-7
    val=[]
    with torch.no_grad():
        for sign in (1,-1):
            m.net[-1].weight[0,col]=sign*.001
            val.append(float(F.cross_entropy(b+m(x[ids],available[ids]),y)))
    numeric=(val[0]-val[1])/.002;assert abs(auto-numeric)<max(.0002,abs(auto)*.03),(auto,numeric)
    result['gradient']=dict(auto=auto,numeric=numeric)
    return result

@torch.inference_mode()
def predict(models,x,a,base):
    scores={name:base+m(x,a).cpu().numpy() for name,m in models.items()}
    for prefix in ('center','mask'):scores[prefix+'_late']=(scores[prefix+'_vv']+scores[prefix+'_vh'])*.5
    return scores

def main(check_only=False):
    tick=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for f in ('models','results','analysis'):(HERE/f).mkdir(exist_ok=True)
    data=Data();receipts=[];tests={}
    for seed in SEEDS:
        z,x,a,null=inputs(data,seed)
        tests[seed]=checks(data,seed,x,a,null,z['base']);write_json(HERE/'checks.json',tests)
        if check_only:print(tests,flush=True);return
        for arm in LEARNED:
            receipts.append(fit(data,seed,arm,x,a,null,z['base']))
            pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
    write_json(HERE/'selection_frozen.json',dict(fits=len(receipts),note='All source selections frozen before evaluation.'))
    rows=[];nullchecks=[]
    for seed in SEEDS:
        z,x,a,null=inputs(data,seed);models={arm:restore(arm,seed,null) for arm in LEARNED}
        scores=predict(models,x,a,z['base'])
        names={'old_full':'old_full','base':'base','strong':'strong','prior_vv':'vv','prior_vh':'vh',
            'prior_late':'late_average','prior_joint':'joint','prior_wide':'joint_wide'}
        for name,oldname in names.items():
            oldz=np.load(PREVIOUS/'results'/f'{oldname}_{seed}.npz')
            np.testing.assert_array_equal(oldz['up_index'],data.up[:data.n]);scores[name]=oldz['score']
        for arm in LEARNED:
            if models[arm].pol is not None:
                missing=~a[:,models[arm].pol]
                with torch.no_grad():err=float(models[arm](x[missing],a[missing]).abs().max()) if missing.any() else 0.
                assert err==0;nullchecks.append(dict(seed=seed,arm=arm,missing=int(missing.sum()),error=err))
        variants=np.load(PREVIOUS/'results'/f'local_cross_{seed}.npz');cross={}
        for mode in ('plain','K_only','T_only','both'):
            _,xx,aa,_=inputs(data,seed,variants[mode])
            for arm,s in predict(models,xx,aa,z['base']).items():cross[arm+'__'+mode]=s
        np.savez_compressed(HERE/'results'/f'cross_scores_{seed}.npz',**cross)
        for arm,s in scores.items():
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',score=s,up_index=data.up[:data.n])
            for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
                rows.append(dict(seed=seed,arm=arm,role=role,**metrics(data.table.iloc[ids].class_id.to_numpy(int),s[ids])))
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv',index=False)
    write_json(HERE/'null_checks.json',nullchecks)
    write_json(HERE/'complete.json',dict(fits=len(receipts),seconds=time.perf_counter()-tick))
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean().to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');main(p.parse_args().check_only)
