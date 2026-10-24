"""Evaluate frozen candidates and trace original-image matched cases."""
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
from PIL import Image

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

train=load('conditioned_train',HERE/'train.py')
ext=load('frequency_extract',HERE/'extract_frequency.py')

def margin(s,y):
    return s[np.arange(len(y)),y]-np.where(np.arange(8)[None]==y[:,None],-np.inf,s).max(1)

def main():
    f=pd.read_csv(train.ROOT/'paired_scale_readout/table.csv',dtype={'mmsi':str})
    ids=f.table_index.to_numpy(int);te=np.flatnonzero(f.role.eq('evaluation'));y=f.class_id.to_numpy(int)
    original=np.load(train.ROOT/'tokens.npz');valid=original['valid'][ids];geom=original['geometry'][ids]
    freq=np.load(HERE/'features.npz');mass=original['mass'][ids]
    length=(geom[:,:,2]*valid).sum(1)/np.maximum(valid.sum(1),1)
    coverage=(freq['coverage'][...,2]*mass).sum((1,2))/np.maximum(mass.sum((1,2)),1e-8)
    threshold=float(np.quantile(length[f.role.eq('utility_train')],.25))
    scores={};parts=[];paired=[]
    for seed in train.SEEDS:
        scores[seed,'strong']=np.load(train.ROOT/'results'/f'layer1_independent_{seed}.npz')['score'][ids]
        scores[seed,'full']=np.load(train.UP/f'strategies_seed{seed}.npz')['logits'][ids,2]
        for arm in train.ARMS:scores[seed,arm]=np.load(HERE/'results'/f'{arm}_{seed}.npz')['score']
        for arm in ('full','strong',*train.ARMS):
            s=scores[seed,arm];d=f.iloc[te].copy();d['local_index']=te;d['seed']=seed;d['arm']=arm
            d['prediction']=s[te].argmax(1);d['correct']=d.prediction.eq(d.class_id);d['margin']=margin(s[te],y[te])
            d['length']=length[te];d['small']=length[te]<=threshold;d['coverage16']=coverage[te];d['edge_limited']=coverage[te]<.9
            parts.append(d)
        for comparator in ('strong','content','full'):
            a=scores[seed,'frequency'][te].argmax(1)==y[te];b=scores[seed,comparator][te].argmax(1)==y[te]
            delta=a.astype(float)-b
            paired.append(dict(seed=seed,comparator=comparator,rescued=int((a&~b).sum()),harmed=int((b&~a).sum()),
                net=int(delta.sum()),OA_pp=100*delta.mean(),BA_pp=100*np.mean([delta[y[te]==c].mean() for c in range(8)])))
    detail=pd.concat(parts,ignore_index=True);detail.to_csv(HERE/'predictions.csv',index=False)
    pd.DataFrame(paired).to_csv(HERE/'paired.csv',index=False)
    detail.groupby(['seed','arm','class_id','class_name']).agg(n=('correct','size'),accuracy=('correct','mean'),margin=('margin','mean')).to_csv(HERE/'classes.csv')
    detail.groupby(['seed','arm','pass_group']).agg(n=('correct','size'),accuracy=('correct','mean')).to_csv(HERE/'passes.csv')
    groups=[]
    for key in ('small','edge_limited'):
        for (seed,arm,value),sub in detail.groupby(['seed','arm',key]):
            groups.append(dict(group=key,value=bool(value),seed=seed,arm=arm,n=len(sub),OA=sub.correct.mean(),BA=sub.groupby('class_id').correct.mean().mean()))
    pd.DataFrame(groups).to_csv(HERE/'groups.csv',index=False)
    # Cluster paired changes after averaging configurations; do not triple sample size.
    pg=f.iloc[te].pass_group.to_numpy();unique=np.unique(pg);rng=np.random.default_rng(1024)
    draws=rng.integers(len(unique),size=(2000,len(unique)));ci=[]
    counts=np.array([[( (pg==g)&(y[te]==c)).sum() for c in range(8)] for g in unique])
    for comp in ('strong','content'):
        d=np.mean([(scores[s,'frequency'][te].argmax(1)==y[te]).astype(float)-(scores[s,comp][te].argmax(1)==y[te]) for s in train.SEEDS],0)
        sums=np.array([[d[(pg==g)&(y[te]==c)].sum() for c in range(8)] for g in unique])
        ct=counts[draws].sum(1);sm=sums[draws].sum(1);ok=(ct>0).all(1)
        oa=100*sm.sum(1)/ct.sum(1);ba=100*(sm[ok]/ct[ok]).mean(1)
        ci.append(dict(comparator=comp,OA_low=float(np.quantile(oa,.025)),OA_high=float(np.quantile(oa,.975)),BA_low=float(np.quantile(ba,.025)),BA_high=float(np.quantile(ba,.975)),BA_draws=int(ok.sum())))
    pd.DataFrame(ci).to_csv(HERE/'uncertainty.csv',index=False)
    candidates=[]
    for seed in train.SEEDS:
        b=scores[seed,'strong'][te];a=scores[seed,'frequency'][te];change=margin(a,y[te])-margin(b,y[te])
        for k,j in enumerate(te):
            bc=b[k].argmax()==y[j];ac=a[k].argmax()==y[j]
            status='CC' if bc and ac else ('WW' if not bc and not ac else ('rescue' if ac else 'harm'))
            candidates.append(dict(seed=seed,local_index=int(j),case=status,change=float(change[k])))
    cf=pd.DataFrame(candidates);chosen=[]
    for status in ('rescue','harm','CC','WW'):
        pool=cf[cf.case.eq(status)]
        if len(pool):
            pool=pool[pool.seed.eq(pool.seed.min())];chosen.append(pool.loc[pool.change.abs().idxmax()].to_dict())
    table=pd.read_csv(train.UP/'table.csv',dtype={'mmsi':str});refs=np.load(train.ROOT/'scores_layer1.npz')['refs']
    snapshots={int(s['obs_id']):s for s in json.loads((ext.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    out=HERE/'analysis';out.mkdir(exist_ok=True);records=[]
    for case in chosen:
        j=int(case['local_index']);i=int(ids[j]);seed=int(case['seed']);c=int(y[j]);obs=int(f.iloc[j].obs_id)
        before=scores[seed,'strong'][j];after=scores[seed,'frequency'][j]
        rival=int(np.where(np.arange(8)==c,-np.inf,before).argmax());rr=refs[i,[c,rival]].ravel()
        fig,ax=plt.subplots(2,7,figsize=(20,7));provenance=[]
        for p in range(2):
            pix,fp,g,placement=ext.geo.native(table,snapshots,i,p)
            ax[p,0].imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            ax[p,0].set_title(f'{("VV","VH")[p]} query {obs} valid={int(valid[j,p])}\n{f.iloc[j].class_name}',fontsize=8)
            if valid[j,p]:
                with Image.open(table.iloc[i]['image_'+('vv','vh')[p]]) as im:raw=np.asarray(im.convert('L'),dtype=np.float64)/255
                maps=[]
                for angle in (g[4],0.):
                    energy=np.mean([abs(ext.response(raw,ext.kernel(8.,angle+a)))**2 for a in ext.ANGLES],0)
                    canvas=np.zeros((160,160));dx,dy=placement['pad_left'],placement['pad_top'];w,h=placement['crop_width'],placement['crop_height'];xx,yy=placement['crop_x'],placement['crop_y']
                    canvas[dy:dy+h,dx:dx+w]=np.log1p(energy[yy:yy+h,xx:xx+w]/1e-4);maps.append(canvas)
                vmax=max(float(np.max(maps)),1e-8)
                for k,m in enumerate(maps,1):ax[p,k].imshow(m,cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
                region=ext.geo.soft_regions(g)[0]*fp
                for k in range(3):
                    if region.max()>.5:ax[p,k].contour(region,levels=[.5],colors=['#20a780'],linewidths=.6)
            ax[p,1].set_title('8px ship-relative response\nmean of 4 directions',fontsize=8)
            ax[p,2].set_title('8px image-axis response\nmean of 4 directions',fontsize=8)
            provenance.append(dict(index=i,pol=p,path=table.iloc[i]['image_'+('vv','vh')[p]]))
            for k,ri in enumerate(rr,3):
                r,rfp,rg,_=ext.geo.native(table,snapshots,int(ri),p)
                ax[p,k].imshow(r,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if rg is not None and original['valid'][ri,p]:
                    mask=ext.geo.soft_regions(rg)[0]*rfp
                    if mask.max()>.5:ax[p,k].contour(mask,levels=[.5],colors=['#20a780'],linewidths=.6)
                ax[p,k].set_title(f'{"TRUE" if k<5 else "RIVAL"} ref{int(table.iloc[ri].obs_id)} valid={int(original["valid"][ri,p])}\n{table.iloc[ri].class_name}',fontsize=8)
                provenance.append(dict(index=int(ri),pol=p,path=table.iloc[ri]['image_'+('vv','vh')[p]]))
        for a in ax.ravel():a.set(xticks=[],yticks=[])
        fig.suptitle(f"{case['case']} | seed{seed} | strong {before.argmax()} -> frequency {after.argmax()} | content {scores[seed,'content'][j].argmax()}\n"
            f"true margin {margin(before[None],y[j:j+1])[0]:+.4f} -> {margin(after[None],y[j:j+1])[0]:+.4f}",fontsize=11)
        fig.text(.02,.025,'Native160 crop/pad; response maps are fixed filters, not learned attention. Same response range within each polarization.',fontsize=9)
        fig.subplots_adjust(top=.83,bottom=.09,wspace=.12,hspace=.18)
        path=out/f"{case['case']}_{obs}.png";fig.savefig(path,dpi=130);plt.close(fig)
        reader=train.Reader('frequency',seed);z=np.load(HERE/'results'/f'frequency_{seed}.npz')
        new=torch.tensor(z['local'][j:j+1],device='cuda');old=reader.oldlocal[j:j+1]
        idx=torch.tensor([j],device='cuda');values=[]
        with torch.no_grad():
            for subset in range(64):
                local=old.clone()
                for k in range(6):
                    if subset&(1<<k):local[:,:,k//3,k%3]=new[:,:,k//3,k%3]
                s=(reader.anchor[idx]+reader.delta(local,idx)-reader.old_delta[idx])[0]
                values.append(float(s[c]-s[rival]))
        attribution=[]
        for k in range(6):
            val=0.
            for subset in range(64):
                if subset&(1<<k):continue
                n=subset.bit_count();val+=math.factorial(n)*math.factorial(5-n)/math.factorial(6)*(values[subset|(1<<k)]-values[subset])
            attribution.append(val)
        assert abs(sum(attribution)-(values[-1]-values[0]))<1e-6
        np.testing.assert_allclose(values[-1],after[c]-after[rival],atol=2e-5)
        record=dict(**case,obs_id=obs,true_class=c,rival=rival,old_pred=int(before.argmax()),new_pred=int(after.argmax()),content_pred=int(scores[seed,'content'][j].argmax()),
            old_margin=float(margin(before[None],y[j:j+1])[0]),new_margin=float(margin(after[None],y[j:j+1])[0]),
            fixed_rival_old_margin=values[0],fixed_rival_new_margin=values[-1],region_shapley=attribution,
            local_true_change=(z['local'][j,c]-reader.oldlocal[j,c].cpu().numpy()).tolist(),local_rival_change=(z['local'][j,rival]-reader.oldlocal[j,rival].cpu().numpy()).tolist(),
            diagnostic_true=z['diagnostic'][j,c].tolist(),diagnostic_rival=z['diagnostic'][j,rival].tolist(),
            reference_obs=table.iloc[rr].obs_id.astype(int).tolist(),coverage16=float(coverage[j]),length=float(length[j]),provenance=provenance,path=str(path))
        records.append(record);del reader;torch.cuda.empty_cache()
    (out/'cases.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (HERE/'analysis_receipt.json').write_text(json.dumps(dict(small_length_source_q25=threshold,query_count=len(te),ships=f.iloc[te].mmsi.nunique(),passes=len(unique),cases=len(records),bootstrap='2000 paired acquisition clusters; BA conditional on all8 classes present'),indent=2),encoding='utf-8')
    print(pd.DataFrame(paired).to_string(index=False));print(pd.DataFrame(ci).to_string(index=False))
    print(pd.DataFrame(records)[['case','seed','obs_id','old_margin','new_margin','content_pred','region_shapley']].to_string(index=False))

if __name__=='__main__':main()
