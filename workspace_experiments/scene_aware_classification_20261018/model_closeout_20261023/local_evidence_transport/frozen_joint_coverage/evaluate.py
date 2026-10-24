"""Frozen-checkpoint inference with matched query/reference coverage."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
EXT=ROOT/'external_confirmation'
UP=ROOT.parent/'regional_utility_pilot/upstream'
spec=importlib.util.spec_from_file_location('frozen_train',ROOT/'transport_only_joint_metric/train.py')
train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
old=train.old
ARMS=('strong','metric_only','both')
SEEDS=(17,29,43)

def body(x,m):
    z=(x[:,:,:8]*m[:,:,:8,None]).sum(2)/np.maximum(m[:,:,:8].sum(2)[...,None],1e-8)
    return old.unit(z).reshape(len(x),-1)

def project(tokens,pr):
    x=old.unit((tokens['layer1']-pr['mean'])@pr['components'].T).astype(np.float32)
    x[tokens['mass']<=0]=0
    return x

def reference_plans(qx,qm,bx,bm):
    cosine=np.einsum('nd,ckd->nck',body(qx,qm),body(bx.reshape(64,2,20,32),bm.reshape(64,2,20)).reshape(8,8,-1))
    order=np.argsort(-cosine,axis=-1)
    plans={'nearest2':order[...,:2],'rank34':order[...,2:4]}
    for seed in range(1001,1006):
        rng=np.random.default_rng(seed)
        picks=np.stack([rng.choice(8,2,replace=False) for _ in range(8)])
        plans[f'random{seed}']=np.broadcast_to(picks,(len(qx),8,2)).copy()
    return plans

def standardize(d,m,scaler):
    mu,sd,active=(np.asarray(scaler[k]) for k in ('mu','scale','active'))
    out=np.where(active,(d-mu)/sd,0).astype(np.float32)
    out[m<=0]=0
    return torch.tensor(out,device='cuda')

@torch.inference_mode()
def local_all(qx,qm,qd,bx,bm,bd,models):
    """Score every legal bank edge once, before selecting reference pairs."""
    x,m,s,b=(torch.tensor(v,device='cuda') for v in (qx,qm,bx,bm))
    h={a:models[a].project(qd) for a in models}
    hs={a:models[a].project(bd) for a in models}
    outputs={a:np.empty((len(qx),8,8,2,3,2),np.float32) for a in ARMS}
    for first in range(0,len(qx),24):
        stop=min(first+24,len(qx));groups={a:[] for a in ARMS}
        for r in old.REGIONS:
            q=x[first:stop,:,r][:,None,None,:,:,None,:]
            v=s[...,r,:][None,...,None,:,:]
            qa=m[first:stop,:,r][:,None,None];sa=b[...,r][None]
            k0=(q*v).sum(-1);t0=old.sinkhorn(k0,qa,sa)[0]
            groups['strong'].append(old.pair_score(t0,k0,qa,sa))
            for arm,model in models.items():
                hi=h[arm][first:stop,:,r][:,None,None,:,:,None,:]
                hj=hs[arm][...,r,:][None,...,None,:,:]
                k1=train.previous.similarity(q,v,model.weights(hi,hj))
                t=t0 if arm=='metric_only' else old.sinkhorn(k1,qa,sa)[0]
                groups[arm].append(old.pair_score(t,k1,qa,sa))
        for arm in ARMS:outputs[arm][first:stop]=torch.stack(groups[arm],-2).cpu().numpy()
    return outputs

def aggregate(values,picks):
    return np.take_along_axis(values,picks[...,None,None,None],axis=2).mean(2)

@torch.inference_mode()
def classify(local,base,ck):
    z=base-base.mean(-1,keepdims=True)
    rival=np.stack([np.delete(z,c,axis=1).max(1) for c in range(8)],1)
    raw=np.concatenate([local.reshape(len(base),8,12),np.stack([z,rival,z-rival],-1)],-1)
    features=np.where(ck['active'],(raw-ck['mu'])/ck['scale'],0).astype(np.float32)
    head=old.Head().eval();head.load_state_dict(ck['state'])
    return base+head(torch.from_numpy(features)).numpy()

def main():
    tick=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=HERE/'results';out.mkdir(exist_ok=True)
    source=pd.read_csv(UP/'table.csv',dtype={'mmsi':str})
    compact=pd.read_csv(ROOT/'paired_scale_readout/table.csv',dtype={'mmsi':str})
    source_tokens=np.load(ROOT/'tokens.npz');pr=np.load(ROOT/'projection_layer1.npz')
    source_x=project(source_tokens,pr);bank=pr['bank'];bm=source_tokens['mass'][bank];bx=source_x[bank]
    inverse={int(v):i for i,v in enumerate(compact.table_index)}
    bank_compact=np.array([[inverse[int(i)] for i in row] for row in bank])
    source_joint=np.load(ROOT/'spatial_frequency_joint_pooling/features.npz')
    np.testing.assert_array_equal(source_joint['ids'],compact.table_index)
    bd_raw=source_joint['joint'][bank_compact]
    dev_rows=np.flatnonzero(compact.role.eq('evaluation'))
    external=pd.read_csv(EXT/'table.csv',dtype={'mmsi':str})
    for col in ('obs_id','mmsi','pass_group'):assert not set(external[col])&set(source[col]),col
    ext_tokens=np.load(EXT/'tokens.npz');ext_joint=np.load(HERE/'joint_external.npz')
    np.testing.assert_array_equal(ext_joint['instance_ids'].astype(str),external.instance_id)
    datasets={
        'development':(compact.iloc[dev_rows].reset_index(drop=True),source_x[compact.table_index.iloc[dev_rows]],source_tokens['mass'][compact.table_index.iloc[dev_rows]],source_joint['joint'][dev_rows]),
        'expanded':(external,project(ext_tokens,pr),ext_tokens['mass'],ext_joint['joint'])}
    checks=[];metrics=[];timings=[]
    for name,(table,qx,qm,d_raw) in datasets.items():
        table.to_csv(HERE/f'{name}_table.csv',index=False)
        for col in ('mmsi','pass_group'):assert not set(table[col])&set(source.iloc[bank.ravel()][col]),(name,col)
        plans=reference_plans(qx,qm,bx,bm)
        expected=(np.load(EXT/'local_scores.npz')['refs'] if name=='expanded' else np.load(ROOT/'scores_layer1.npz')['refs'][table.table_index])
        actual=np.take_along_axis(np.broadcast_to(bank,(len(table),8,8)),plans['nearest2'],-1)
        np.testing.assert_array_equal(actual,expected)
        np.savez_compressed(HERE/f'{name}_references.npz',bank=bank,**plans)
        for seed in SEEDS:
            start=time.perf_counter();models={};checkpoints={}
            for arm,path in [('metric_only',ROOT/'transport_only_joint_metric/models'/f'metric_only_{seed}.pt'),('both',ROOT/'pair_conditioned_joint_metric/models'/f'pair_joint_{seed}.pt')]:
                ck=torch.load(path,map_location='cpu',weights_only=False);checkpoints[arm]=ck
                model=train.Metric(arm).cuda().eval().requires_grad_(False);model.load_state_dict(ck['state']);models[arm]=model
            assert checkpoints['metric_only']['scaler']==checkpoints['both']['scaler']
            scaler=checkpoints['metric_only']['scaler'];qd=standardize(d_raw,qm,scaler);bd=standardize(bd_raw,bm,scaler)
            locals_=local_all(qx,qm,qd,bx,bm,bd,models)
            base=(np.load(EXT/f'strategies_seed{seed}.npz')['logits'][:,2] if name=='expanded' else np.load(UP/f'strategies_seed{seed}.npz')['logits'][table.table_index,2]).astype(np.float32)
            head=torch.load(ROOT/'models'/f'layer1_independent_{seed}.pt',map_location='cpu',weights_only=False)
            for plan,picks in plans.items():
                payload={}
                for arm in ARMS:
                    local=aggregate(locals_[arm],picks);score=classify(local,base,head)
                    payload[arm]=score;payload[arm+'_local']=local
                    # Labels enter only here, after references and predictions are fixed.
                    scopes={'all':np.ones(len(table),bool)}
                    if name=='expanded':scopes.update(original_fold0=table.original_test_fold.eq(0).to_numpy(),additional_folds=table.original_test_fold.ne(0).to_numpy())
                    for scope,mask in scopes.items():
                        metrics.append(dict(dataset=name,plan=plan,seed=seed,arm=arm,scope=scope,**old.metrics(table.class_id.to_numpy(int)[mask],score[mask])))
                if plan=='nearest2':
                    archived=(np.load(EXT/'results'/f'independent_{seed}.npz')['score'] if name=='expanded' else np.load(ROOT/'results'/f'layer1_independent_{seed}.npz')['score'][table.table_index])
                    err=float(abs(payload['strong']-archived).max());assert err<3e-5,err
                    checks.append(dict(dataset=name,seed=seed,test='strong_original_replay',max_error=err))
                    if name=='development':
                        for arm,path in [('metric_only',ROOT/'transport_only_joint_metric/results'/f'metric_only_{seed}.npz'),('both',ROOT/'pair_conditioned_joint_metric/results'/f'pair_joint_{seed}.npz')]:
                            original=np.load(path)['score'][dev_rows];err=float(abs(payload[arm]-original).max());assert err<3e-5,(arm,err)
                            checks.append(dict(dataset=name,seed=seed,test=arm+'_original_replay',max_error=err))
                if seed==17:
                    err=float(abs(payload['metric_only']-payload['strong']).max());assert err<3e-5,err
                    checks.append(dict(dataset=name,seed=seed,test='identity_new_reference_'+plan,max_error=err))
                if plan=='random1001':
                    reversed_score=classify(aggregate(locals_['metric_only'],picks[:,:,::-1]),base,head)
                    err=float(abs(payload['metric_only']-reversed_score).max());assert err<3e-5,err
                    checks.append(dict(dataset=name,seed=seed,test='reference_order',max_error=err))
                np.savez_compressed(out/f'{name}_{plan}_{seed}.npz',**payload)
            timings.append(dict(dataset=name,seed=seed,n=len(table),seconds=time.perf_counter()-start))
            print('EVALUATED',name,seed,'seconds',round(time.perf_counter()-start,1),flush=True)
            del locals_,models,qd,bd;torch.cuda.empty_cache()
    pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'checks.csv',index=False);pd.DataFrame(timings).to_csv(HERE/'inference_receipts.csv',index=False)
    (HERE/'complete.json').write_text(json.dumps(dict(seconds=time.perf_counter()-tick,new_fits=0,reference_plans=list(plans),source_bank=64,references_per_query=16),indent=2),encoding='utf-8')
    frame=pd.DataFrame(metrics)
    print(frame.query("plan=='nearest2' and scope=='all'").groupby(['dataset','arm'])[['OA','BA','NLL','Brier']].mean().to_string(),flush=True)

if __name__=='__main__':main()
