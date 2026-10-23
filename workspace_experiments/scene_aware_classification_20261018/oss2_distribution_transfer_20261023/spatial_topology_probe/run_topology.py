"""One source-only extraction and nine fixed-budget topology readouts."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from prepare_coverage import footprint, graph

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import train_heads as parent
from extract_features import network, load_batch, feature_map, spatial_weights
from extract_moments import weighted_moments
P=parent.parent
ARMS=['marginal','complete','local']
SEEDS=[17,29,43]


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(19,12),nn.GELU(),nn.Linear(12,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        value=self.net(x).squeeze(-1)
        return value-value.mean(1,keepdim=True)


def local_energy(r,weight):
    rr=r.reshape(-1,8,14,14);ww=weight.reshape(-1,1,14,14)
    wh=ww[:,:,:,:-1]*ww[:,:,:,1:];wv=ww[:,:,:-1,:]*ww[:,:,1:,:]
    total=wh.sum((-2,-1))+wv.sum((-2,-1))
    assert (total>0).all()
    num=(wh*(rr[:,:,:,:-1]-rr[:,:,:,1:]).square()).sum((-2,-1))
    num+=(wv*(rr[:,:,:-1,:]-rr[:,:,1:,:]).square()).sum((-2,-1))
    return num/(2*total)


def explicit_check(r,weight):
    w=weight.ravel();keep=w>0;w=w[keep];x=r[:,keep]
    ii,jj=np.triu_indices(len(w),1);a=w[ii]*w[jj]
    e=(a[None]*(x[:,ii]-x[:,jj])**2).sum(1)/(2*a.sum())
    prob=w/w.sum();var=((x-(x*prob).sum(1,keepdims=True))**2*prob).sum(1)
    analytic=var/(1-prob@prob)
    return float(abs(e-analytic).max())


def extract(table,cache):
    n=len(table);assigned=cache['reader']
    readers=[]
    for tag in ['outer']+[f'inner{k}' for k in range(4)]:
        with np.load(ROOT/'readers/models'/f'fold0_{tag}.npz') as z:readers.append(z['W'].reshape(8,2,1024).transpose(1,0,2))
    W=torch.tensor(np.stack(readers),device='cuda',dtype=torch.float64)
    model=network()
    means=np.zeros((n,2,8));moments=np.zeros((n,2,8,2));ratios=np.zeros((n,2,8,2))
    zeros=np.zeros((n,2,4,8),bool);kish=np.zeros((n,2));maps=np.zeros((n,2,8,14,14),np.float32)
    points={};errors=[];chosen={int(cache['train'][0]),int(cache['validation'][0])}
    with torch.inference_mode():
        for first in range(0,n,16):
            stop=min(n,first+16);x,mask=load_batch(table,first,stop)
            wp=W[np.repeat(assigned[first:stop],2),np.tile([0,1],stop-first)]
            for rotation in range(4):
                z=feature_map(model,x,rotation)
                wt=spatial_weights(mask,z,rotation).flatten(2).double()
                response=torch.einsum('bck,bkp->bcp',wp,z.flatten(2).double())
                moment,mu,r=weighted_moments(response,wt)
                prob=wt/wt.sum(-1,keepdim=True)
                var=(r.square()*prob).sum(-1)
                denom=1-prob.square().sum(-1)
                complete=var/denom
                local=local_energy(r,wt)
                good=var>0
                R=torch.zeros((*var.shape,2),device='cuda',dtype=torch.float64)
                R[...,0][good]=(complete[good]/var[good])
                R[...,1][good]=(local[good]/var[good])
                assert torch.isfinite(R).all()
                ratios[first:stop]+=R.cpu().numpy().reshape(stop-first,2,8,2)/4
                means[first:stop]+=mu.cpu().numpy().reshape(stop-first,2,8)/4
                moments[first:stop]+=moment.cpu().numpy().reshape(stop-first,2,8,2)/4
                zeros[first:stop,:,rotation]=(~good).cpu().numpy().reshape(stop-first,2,8)
                if rotation==0:
                    kish[first:stop]=(1/prob.square().sum(-1)).cpu().numpy().reshape(stop-first,2)
                    maps[first:stop]=r.cpu().numpy().reshape(stop-first,2,8,14,14)
                for i in sorted(chosen & set(range(first,stop))):
                    rr=r[(i-first)*2:(i-first+1)*2].cpu().numpy()
                    ww=wt[(i-first)*2:(i-first+1)*2].cpu().numpy()
                    points[f'row{i}_rot{rotation}_r']=rr
                    points[f'row{i}_rot{rotation}_weight']=ww
                    points[f'row{i}_rot{rotation}_R']=R[(i-first)*2:(i-first+1)*2].cpu().numpy()
                    for p in range(2):errors.append(explicit_check(rr[p],ww[p]))
            if first%256==0:print('source extraction',stop,'/',n,flush=True)
    mean_error=float(abs(means-cache['mean']).max());moment_error=float(abs(moments-cache['marginal']).max())
    np.testing.assert_allclose(means,cache['mean'],atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(moments,cache['marginal'],atol=1e-5,rtol=1e-5)
    assert max(errors)<1e-8
    np.savez_compressed(HERE/'onepoint_replay.npz',**points)
    np.savez_compressed(HERE/'topology_features.npz',sample_ids=cache['sample_ids'],reader=assigned,
        train=cache['train'],validation=cache['validation'],ratios=ratios,kish=kish,zero_variance=zeros)
    info=dict(mean_max_abs=mean_error,moment_max_abs=moment_error,Eall_identity_max_abs=max(errors),
              explicit_checks=len(errors),zero_variance_class_pol_views=int(zeros.sum()),
              zero_variance_samples=int(zeros.any((1,2,3)).sum()),invalid_edge_samples=0)
    (HERE/'extraction_checks.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    del model,W
    torch.cuda.empty_cache()
    return ratios,kish,maps,info


def descriptor(base,cache,kish,ratio):
    cols=[base-base.mean(1,keepdims=True)]
    for p in range(2):
        for value in [cache['mean'][:,p],cache['marginal'][:,p,:,0],cache['marginal'][:,p,:,1]]:
            cols.extend([value,value-value.mean(1,keepdims=True)])
    cols.extend([np.ones_like(base),np.ones_like(base)])
    cols.extend([np.broadcast_to(kish[:,p,None],base.shape) for p in range(2)])
    cols.extend([ratio[:,p] for p in range(2)])
    return np.stack(cols,-1)


def restore(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=Head().eval().requires_grad_(False);model.load_state_dict(ck['state'])
    return ck,model


def base_for(seed,cache,mapping):
    tr,va=cache['train'],cache['validation'];base=np.zeros((len(mapping),8))
    with np.load(ROOT/'calibrated'/f'fold0_seed{seed}.npz') as z:
        np.testing.assert_array_equal(z['sample_ids'][mapping],cache['sample_ids'])
        base[tr]=z['train_base'][mapping[tr]];base[va]=z['base'][mapping[va]]
    return base


def math_example():
    fields=[np.array([[1.,1.],[-1.,-1.]]),np.array([[1.,-1.],[-1.,1.]])]
    out=[]
    for label,a in zip(['stripes','checkerboard'],fields):
        differences=np.r_[(a[:,1:]-a[:,:-1]).ravel(),(a[1:,:]-a[:-1,:]).ravel()]
        energy=float(np.mean(differences**2)/2)
        allpairs=[(a.ravel()[i]-a.ravel()[j])**2 for i in range(4) for j in range(i+1,4)]
        out.append(dict(field=label,mean=float(a.mean()),variance=float(a.var()),kappa=float((a*abs(a)).mean()),
                        local=energy,complete=float(np.mean(allpairs)/2)))
    assert [v['local'] for v in out]==[1.,2.]
    assert all(v['complete']==4/3 and v['variance']==1 for v in out)
    pd.DataFrame(out).to_csv(HERE/'math_example.csv',index=False)


def render_cases(predictions,table,maps):
    cases=[]
    val=predictions[predictions.seed.eq(17)&predictions.arm.eq('local')&predictions.role.eq('validation')]
    for outcome in ['rescued','harmed','CC','WW']:
        g=val[val.outcome.eq(outcome)].sort_values('instance_id')
        if not len(g):cases.append(dict(outcome=outcome,available=False));continue
        rec=g.iloc[0].to_dict();i=int(rec['source_row']);row=table.iloc[i];truth=int(row.class_id)
        fig,axes=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
        for p,pol in enumerate(['vv','vh']):
            with Image.open(row[f'chip_path_{pol}']) as im:raw=np.array(im)
            weight=footprint(*raw.shape);nodes,edges=graph(weight)
            pos=np.stack([nodes%14,nodes//14],-1)
            axes[p,0].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[p,0].set_title(f'{pol.upper()} native {raw.shape[1]}x{raw.shape[0]}')
            value=np.ma.masked_where(weight==0,maps[i,p,truth]);limit=max(float(abs(value).max()),1e-12)
            ax=axes[p,1];im=ax.imshow(value,cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
            ax.set_facecolor('.85');ax.add_collection(LineCollection(pos[edges],colors='k',linewidths=.55,alpha=.6))
            ax.scatter(pos[:,0],pos[:,1],s=3+8*weight.ravel()[nodes],color='k')
            ax.set_title(f'True class {truth}: r, rotation 0\n{len(nodes)} nodes / {len(edges)} local edges',fontsize=10)
            fig.colorbar(im,ax=ax,fraction=.04)
            ii,jj=np.triu_indices(len(nodes),1);ax=axes[p,2]
            ax.add_collection(LineCollection(np.stack([pos[ii],pos[jj]],axis=1),colors='#557788',linewidths=.3,alpha=.16))
            ax.scatter(pos[:,0],pos[:,1],s=3+8*weight.ravel()[nodes],color='black')
            ax.set_xlim(-.5,13.5);ax.set_ylim(13.5,-.5);ax.set_aspect('equal')
            ax.set_title(f'Complete graph: {len(ii)} edges\nR_all depends only on Kish (v>0)',fontsize=10)
            for ax in axes[p]:ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f"{row.sample_id} {outcome}; y={truth}, B={rec['base_prediction']}, local={rec['prediction']}\nEdges use w_i*w_j; graphs are coarse RF cells, not physical boundaries",fontsize=11)
        path=HERE/f'case_{outcome}.png';fig.savefig(path,dpi=130);plt.close(fig)
        rec.update(available=True,panel=str(path),chip_path_vv=row.chip_path_vv,chip_path_vh=row.chip_path_vh)
        cases.append(rec)
        np.savez_compressed(HERE/f'case_{outcome}_map.npz',r_rotation0=maps[i],sample_id=str(row.sample_id))
    pd.DataFrame(cases).to_csv(HERE/'cases.csv',index=False)


def main():
    started=time.perf_counter();assert (HERE/'PROTOCOL_ZH.md').exists()
    (HERE/'heads').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    math_example()
    with np.load(ROOT/'pair_local_evidence/source_moments.npz') as z:
        cache={k:z[k] for k in ['sample_ids','mean','marginal','reader','train','validation']}
    allrows=pd.read_csv(ROOT/'cohort.csv',dtype={'mmsi':str})
    mapping=pd.Series(np.arange(len(allrows)),index=allrows.sample_id).loc[cache['sample_ids']].to_numpy()
    table=allrows.iloc[mapping].reset_index(drop=True);tr,va=cache['train'],cache['validation'];y=table.class_id.to_numpy()
    with np.load(ROOT/'readers/fold0_inputs.npz') as z:
        np.testing.assert_array_equal(mapping[tr],z['train']);np.testing.assert_array_equal(mapping[va],z['validation'])
    assert len(tr)==1542 and len(va)==279
    ratios,kish,maps,checks=extract(table,cache)
    chosen,curves,frames,metrics=[],[],[],[]
    for seed in SEEDS:
        base=base_for(seed,cache,mapping);initial=None
        common=None
        for arm in ARMS:
            ratio=np.zeros_like(ratios[...,0]) if arm=='marginal' else ratios[...,ARMS.index(arm)-1]
            raw=descriptor(base,cache,kish,ratio)
            if common is None:common=raw[...,:17].copy()
            else:np.testing.assert_array_equal(raw[...,:17],common)
            mu=raw[tr].mean((0,1),keepdims=True);scale=raw[tr].std((0,1),keepdims=True).clip(1e-8)
            x=torch.tensor((raw-mu)/scale,dtype=torch.float32)
            torch.manual_seed(seed);model=Head();assert sum(p.numel() for p in model.parameters())==253
            if initial is None:initial=P.frozen_state(model)
            else:
                for k,v in model.state_dict().items():assert torch.equal(v,initial[k])
            with torch.no_grad():assert not model(x[va]).any()
            best,epoch_best=P.frozen_state(model),0;best_j=P.exact_j(y[va],base[va].argmax(1))
            meta=dict(seed=seed,arm=arm,fold=0)
            curves.append(dict(**meta,epoch=0,train_loss=np.nan,**P.assessment(y[va],base[va])))
            opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
            bb=torch.tensor(base[tr],dtype=torch.float32);yy=torch.tensor(y[tr],dtype=torch.long)
            for epoch in range(1,201):
                opt.zero_grad();loss=P.joint_loss(bb+model(x[tr]),yy);assert torch.isfinite(loss)
                loss.backward();opt.step()
                if epoch%5:continue
                with torch.no_grad():score=base[va]+model(x[va]).numpy().astype(float)
                j=P.exact_j(y[va],score.argmax(1))
                curves.append(dict(**meta,epoch=epoch,train_loss=float(loss.detach()),**P.assessment(y[va],score)))
                if j>best_j:best_j,epoch_best,best=j,epoch,P.frozen_state(model)
            path=HERE/'heads'/f'{arm}_{seed}.pt'
            torch.save(dict(**meta,state=best,mu=mu,scale=scale,epoch=epoch_best,parameters=253),path)
            ck,replay=restore(path)
            with torch.inference_mode():score=base+replay(x).numpy().astype(float)
            assert P.exact_j(y[va],score[va].argmax(1))==best_j
            chosen.append(dict(**meta,epoch=epoch_best,validation_J=float(best_j),replay_J_exact=True,parameters=253))
            for role,ix in [('train',tr),('validation',va)]:
                g=table.iloc[ix][['sample_id','class_id','class_name','mmsi','pass_group','chip_path_vv','chip_path_vh']].copy().rename(columns={'sample_id':'instance_id'})
                g['source_row']=ix;g['global_row']=mapping[ix]
                for k,v in dict(**meta,role=role,epoch=epoch_best).items():g[k]=v
                g['base_prediction']=base[ix].argmax(1);g['prediction']=score[ix].argmax(1)
                g['net']=g.prediction.eq(g.class_id).astype(int)-g.base_prediction.eq(g.class_id).astype(int)
                g['outcome']=np.select([g.net.eq(1),g.net.eq(-1),g.prediction.eq(g.class_id)],['rescued','harmed','CC'],default='WW')
                for c in range(8):g[f'b{c}']=base[ix,c];g[f's{c}']=score[ix,c]
                frames.append(g)
                before=P.assessment(y[ix],base[ix]);after=P.assessment(y[ix],score[ix])
                metrics.append(dict(**meta,role=role,net=int(g.net.sum()),**after,
                    **{f'base_{k}':v for k,v in before.items() if k!='n'},
                    **{f'delta_{k}':after[k]-before[k] for k in ['OA','BA','NLL','balanced_NLL','Brier']}))
            print('selected',seed,arm,epoch_best,float(best_j),flush=True)
    predictions=pd.concat(frames,ignore_index=True);met=pd.DataFrame(metrics)
    predictions.to_csv(HERE/'predictions.csv',index=False);met.to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(chosen).to_csv(HERE/'selections.csv',index=False);pd.DataFrame(curves).to_csv(HERE/'selection_curves.csv',index=False)
    comparisons=[]
    for (seed,role),part in met.groupby(['seed','role']):
        part=part.set_index('arm')
        for control in ['marginal','complete']:
            comparisons.append(dict(seed=seed,role=role,comparison=f'local_minus_{control}',
                **{f'delta_{m}':float(part.loc['local',m]-part.loc[control,m]) for m in ['OA','BA','NLL','balanced_NLL','Brier','net']}))
    pd.DataFrame(comparisons).to_csv(HERE/'paired_metrics.csv',index=False)
    classrows=[]
    for (seed,arm,role,c),g in predictions.groupby(['seed','arm','role','class_id']):
        classrows.append(dict(seed=seed,arm=arm,role=role,class_id=c,n=len(g),net=int(g.net.sum()),
            baseline_correct=int(g.base_prediction.eq(g.class_id).sum()),correct=int(g.prediction.eq(g.class_id).sum())))
    pd.DataFrame(classrows).to_csv(HERE/'classes.csv',index=False)
    render_cases(predictions,table,maps)
    info=dict(heads=9,steps=1800,source_train=1542,source_validation=279,test_evaluated=False,
        seconds=time.perf_counter()-started,zero_epoch_heads=sum(c['epoch']==0 for c in chosen),
        parameters=253,arms=ARMS,random_null_trained=False,**checks)
    (HERE/'training_summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(met[met.role.eq('validation')][['seed','arm','net','OA','BA','delta_OA','delta_BA']].to_string(index=False))
    print(json.dumps(info))


if __name__=='__main__':main()
