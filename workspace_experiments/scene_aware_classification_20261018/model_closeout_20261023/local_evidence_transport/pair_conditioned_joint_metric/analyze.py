"""Paired acquisition analysis and exact frozen-head case decompositions."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import importlib.util
import json
import math
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

train=load('pair_train',HERE/'train.py')
ext=load('frequency_extract',train.ROOT/'frequency_conditioned_matching/extract_frequency.py')
COMPS=('strong','separable_joint','pair_marginal','independent_joint','frequency')

def margin(s,y):
    return s[np.arange(len(y)),y]-np.where(np.arange(8)[None]==y[:,None],-np.inf,s).max(1)

@torch.no_grad()
def cross(reader,model,j,c,rival):
    idx=torch.tensor([j],device='cuda');refs=reader.refs[idx]
    q,s=reader.x[idx],reader.x[refs];h=model.project(reader.d);hq,hs=h[idx],h[refs]
    a,b=reader.mass[idx],reader.mass[refs];locals={k:[] for k in ('old','K_only','T_only','both')};weights=[]
    for ri,r in enumerate(train.old.REGIONS):
        qi=q[:,:,r][:,None,None,:,:,None,:];sj=s[...,r,:][...,None,:,:]
        hi=hq[:,:,r][:,None,None,:,:,None,:];hj=hs[...,r,:][...,None,:,:]
        w=model.weights(hi,hj);k=train.similarity(qi,sj,w);k0=(qi*sj).sum(-1)
        qa=a[:,None,None,:,r];sa=b[...,r]
        t,_,_,_=train.old.sinkhorn(k,qa,sa);t0,_,_,_=train.old.sinkhorn(k0,qa,sa)
        for name,tt,kk in [('old',t0,k0),('K_only',t0,k),('T_only',t,k0),('both',t,k)]:
            locals[name].append(train.old.pair_score(tt,kk,qa,sa))
        for cl in (c,rival):
            for p in range(2):
                ww=w[0,cl,:,p];tt=t[0,cl,:,p]
                vec=(tt[...,None]*(ww-1)).sum((0,1,2))/tt.sum().clamp_min(1e-10)
                weights.append(dict(class_id=cl,polarization=('VV','VH')[p],region=('body','side','end')[ri],channel_changes=vec.cpu().tolist(),channel_spread=float(vec.std()),transport_mass=float(tt.sum())))
    values={}
    for name,parts in locals.items():
        loc=torch.stack(parts,-2).mean(2)
        score=(reader.anchor[idx]+reader.delta(loc,idx)-reader.old_delta[idx])[0]
        values[name]=dict(fixed_margin=float(score[c]-score[rival]),prediction=int(score.argmax()))
    return values,weights

def main():
    torch.set_num_threads(2)
    f=pd.read_csv(train.ROOT/'paired_scale_readout/table.csv',dtype={'mmsi':str})
    ids=f.table_index.to_numpy(int);te=np.flatnonzero(f.role.eq('evaluation'));y=f.class_id.to_numpy(int)
    scores={};parts=[];paired=[];candidates=[]
    for seed in train.SEEDS:
        scores[seed,'strong']=np.load(train.ROOT/'results'/f'layer1_independent_{seed}.npz')['score'][ids]
        scores[seed,'frequency']=np.load(train.ROOT/'frequency_conditioned_matching/results'/f'frequency_{seed}.npz')['score']
        scores[seed,'independent_joint']=np.load(train.ROOT/'spatial_frequency_joint_pooling/results'/f'joint_{seed}.npz')['score']
        for arm in train.ARMS:scores[seed,arm]=np.load(HERE/'results'/f'{arm}_{seed}.npz')['score']
        for arm in (*COMPS,'pair_joint'):
            s=scores[seed,arm];d=f.iloc[te][['obs_id','class_id','class_name','mmsi','pass_group']].copy()
            d['seed']=seed;d['arm']=arm;d['prediction']=s[te].argmax(1);d['correct']=d.prediction.eq(d.class_id)
            d['margin']=margin(s[te],y[te]);parts.append(d)
        for comp in COMPS:
            a=scores[seed,'pair_joint'][te].argmax(1)==y[te];b=scores[seed,comp][te].argmax(1)==y[te];delta=a.astype(float)-b
            paired.append(dict(seed=seed,comparator=comp,rescued=int((a&~b).sum()),harmed=int((b&~a).sum()),net=int(delta.sum()),OA_pp=100*delta.mean(),BA_pp=100*np.mean([delta[y[te]==c].mean() for c in range(8)])))
        b=scores[seed,'strong'][te];a=scores[seed,'pair_joint'][te];change=margin(a,y[te])-margin(b,y[te])
        for k,j in enumerate(te):
            bc=b[k].argmax()==y[j];ac=a[k].argmax()==y[j]
            status='CC' if bc and ac else ('WW' if not bc and not ac else ('rescue' if ac else 'harm'))
            candidates.append(dict(seed=seed,local_index=int(j),case=status,change=float(change[k])))
    detail=pd.concat(parts,ignore_index=True);detail.to_csv(HERE/'predictions.csv',index=False)
    pd.DataFrame(paired).to_csv(HERE/'paired.csv',index=False)
    detail.groupby(['seed','arm','class_id','class_name']).agg(n=('correct','size'),accuracy=('correct','mean'),margin=('margin','mean')).to_csv(HERE/'classes.csv')
    detail.groupby(['seed','arm','pass_group']).agg(n=('correct','size'),accuracy=('correct','mean')).to_csv(HERE/'passes.csv')
    pg=f.iloc[te].pass_group.to_numpy();unique=np.unique(pg);rng=np.random.default_rng(1024)
    draws=rng.integers(len(unique),size=(2000,len(unique)));ci=[]
    counts=np.array([[((pg==g)&(y[te]==c)).sum() for c in range(8)] for g in unique])
    for comp in COMPS:
        d=np.mean([(scores[s,'pair_joint'][te].argmax(1)==y[te]).astype(float)-(scores[s,comp][te].argmax(1)==y[te]) for s in train.SEEDS],0)
        sums=np.array([[d[(pg==g)&(y[te]==c)].sum() for c in range(8)] for g in unique])
        ct=counts[draws].sum(1);sm=sums[draws].sum(1);ok=(ct>0).all(1)
        oa=100*sm.sum(1)/ct.sum(1);ba=100*(sm[ok]/ct[ok]).mean(1)
        ci.append(dict(comparator=comp,OA_low=float(np.quantile(oa,.025)),OA_high=float(np.quantile(oa,.975)),BA_low=float(np.quantile(ba,.025)),BA_high=float(np.quantile(ba,.975)),BA_draws=int(ok.sum())))
    pd.DataFrame(ci).to_csv(HERE/'uncertainty.csv',index=False)
    cf=pd.DataFrame(candidates);chosen=[]
    for status in ('rescue','harm','CC','WW'):
        pool=cf[cf.case.eq(status)]
        if len(pool):
            pool=pool[pool.seed.eq(pool.seed.min())];chosen.append(pool.loc[pool.change.abs().idxmax()].to_dict())
    for obs in (30232,4531):
        j=int(np.flatnonzero(f.obs_id.eq(obs))[0])
        if not any(int(c['local_index'])==j and int(c['seed'])==29 for c in chosen):
            chosen.append(dict(seed=29,local_index=j,case='previous_probe',change=float(margin(scores[29,'pair_joint'][j:j+1],y[j:j+1])[0]-margin(scores[29,'strong'][j:j+1],y[j:j+1])[0])))
    # The two crude-tanker losses account for the mean BA sign change.
    for obs in (2080,29687):
        j=int(np.flatnonzero(f.obs_id.eq(obs))[0])
        chosen.append(dict(seed=43,local_index=j,case='BA_boundary',change=float(margin(scores[43,'pair_joint'][j:j+1],y[j:j+1])[0]-margin(scores[43,'strong'][j:j+1],y[j:j+1])[0])))
    table=pd.read_csv(train.old.UP/'table.csv',dtype={'mmsi':str});refs=np.load(train.ROOT/'scores_layer1.npz')['refs'];original=np.load(train.ROOT/'tokens.npz')
    snapshots={int(s['obs_id']):s for s in json.loads((ext.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    out=HERE/'analysis';out.mkdir(exist_ok=True);records=[]
    for case in chosen:
        j=int(case['local_index']);i=int(ids[j]);seed=int(case['seed']);c=int(y[j]);obs=int(f.iloc[j].obs_id)
        before=scores[seed,'strong'][j];after=scores[seed,'pair_joint'][j]
        rival=int(np.where(np.arange(8)==c,-np.inf,before).argmax());rr=refs[i,[c,rival]].ravel()
        fig,ax=plt.subplots(2,5,figsize=(16,7));provenance=[]
        for p in range(2):
            for k,ri in enumerate([i,*rr]):
                pix,fp,g,_=ext.geo.native(table,snapshots,int(ri),p)
                ax[p,k].imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest');valid=int(original['valid'][ri,p])
                if valid and g is not None:
                    mask=ext.geo.soft_regions(g)[0]*fp
                    if mask.max()>.5:ax[p,k].contour(mask,levels=[.5],colors=['#20a780'],linewidths=.6)
                label='QUERY' if k==0 else ('TRUE' if k<3 else 'RIVAL')
                ax[p,k].set_title(f'{("VV","VH")[p]} {label} {int(table.iloc[ri].obs_id)} valid={valid}\n{table.iloc[ri].class_name}',fontsize=8);ax[p,k].set(xticks=[],yticks=[])
                provenance.append(dict(index=int(ri),pol=p,path=table.iloc[ri]['image_'+('vv','vh')[p]]))
        fig.suptitle(f"{case['case']} seed{seed}: strong {before.argmax()} -> pair {after.argmax()} | separable {scores[seed,'separable_joint'][j].argmax()}\ntrue margin {margin(before[None],y[j:j+1])[0]:+.4f} -> {margin(after[None],y[j:j+1])[0]:+.4f}")
        fig.text(.02,.02,'Native160 crop/pad; no resize. Green=body region. Appearance alone does not establish a physical cause.',fontsize=9)
        fig.subplots_adjust(top=.83,bottom=.09,wspace=.12,hspace=.15);path=out/f"{case['case']}_{obs}.png";fig.savefig(path,dpi=120);plt.close(fig)
        reader=train.Reader('pair_joint',seed);z=np.load(HERE/'results'/f'pair_joint_{seed}.npz')
        new=torch.tensor(z['local'][j:j+1],device='cuda');oldlocal=reader.oldlocal[j:j+1];idx=torch.tensor([j],device='cuda');values=[]
        with torch.no_grad():
            for subset in range(64):
                loc=oldlocal.clone()
                for k in range(6):
                    if subset&(1<<k):loc[:,:,k//3,k%3]=new[:,:,k//3,k%3]
                s=(reader.anchor[idx]+reader.delta(loc,idx)-reader.old_delta[idx])[0];values.append(float(s[c]-s[rival]))
        attribution=[]
        for k in range(6):
            val=0.
            for subset in range(64):
                if subset&(1<<k):continue
                n=subset.bit_count();val+=math.factorial(n)*math.factorial(5-n)/math.factorial(6)*(values[subset|(1<<k)]-values[subset])
            attribution.append(val)
        assert abs(sum(attribution)-(values[-1]-values[0]))<1e-6
        np.testing.assert_allclose(values[-1],after[c]-after[rival],atol=2e-5)
        model=train.Metric('pair_joint').cuda();ck=torch.load(HERE/'models'/f'pair_joint_{seed}.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state'])
        crosses,weights=cross(reader,model,j,c,rival)
        np.testing.assert_allclose([crosses['old']['fixed_margin'],crosses['both']['fixed_margin']],[values[0],values[-1]],atol=2e-5)
        records.append(dict(**case,obs_id=obs,true_class=c,rival=rival,old_pred=int(before.argmax()),new_pred=int(after.argmax()),comparator_preds={a:int(scores[seed,a][j].argmax()) for a in COMPS},old_margin=float(margin(before[None],y[j:j+1])[0]),new_margin=float(margin(after[None],y[j:j+1])[0]),region_shapley=attribution,cross_decomposition=crosses,pair_weights=weights,local_true_change=(z['local'][j,c]-reader.oldlocal[j,c].cpu().numpy()).tolist(),local_rival_change=(z['local'][j,rival]-reader.oldlocal[j,rival].cpu().numpy()).tolist(),reference_obs=table.iloc[rr].obs_id.astype(int).tolist(),provenance=provenance,path=str(path)))
        del reader,model;torch.cuda.empty_cache()
    (out/'cases.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (HERE/'analysis_receipt.json').write_text(json.dumps(dict(queries=len(te),ships=f.iloc[te].mmsi.nunique(),passes=len(unique),cases=len(records),bootstrap='2000 paired acquisition clusters; seed average first; BA conditional on all8 classes present'),indent=2),encoding='utf-8')
    print(pd.DataFrame(paired).to_string(index=False));print(pd.DataFrame(ci).to_string(index=False))
    print(pd.DataFrame(records)[['case','seed','obs_id','old_margin','new_margin','region_shapley']].to_string(index=False))

if __name__=='__main__':main()
