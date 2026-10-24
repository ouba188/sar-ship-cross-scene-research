"""Paired outcome tables and native-query/reference inspection panels."""
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
from scipy.ndimage import gaussian_filter
spec=importlib.util.spec_from_file_location('paired_fit',HERE/'fit.py')
fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)
pixel=fit.pixel

def margin(s,y):
    return s[np.arange(len(y)),y]-np.where(np.arange(8)[None]==y[:,None],-np.inf,s).max(1)

def main():
    f=pd.read_csv(HERE/'table.csv',dtype={'mmsi':str});te=np.flatnonzero(f.role.eq('evaluation'))
    y=f.class_id.to_numpy(int);frames=[];scores={}
    for seed in fit.SEEDS:
        for arm in ('full','old_independent',*fit.ARMS):
            s=np.load(HERE/'results'/f'{arm}_{seed}.npz')['score'];scores[seed,arm]=s
            d=f.iloc[te].copy();d['seed']=seed;d['arm']=arm;d['local_index']=te
            d['prediction']=s[te].argmax(1);d['correct']=d.prediction.eq(d.class_id)
            d['margin']=margin(s[te],y[te]);frames.append(d)
    detail=pd.concat(frames,ignore_index=True);detail.to_csv(HERE/'predictions.csv',index=False)
    classes=detail.groupby(['seed','arm','class_id','class_name']).agg(n=('correct','size'),accuracy=('correct','mean'),margin=('margin','mean')).reset_index()
    classes.to_csv(HERE/'classes.csv',index=False)
    detail.groupby(['seed','arm','pass_group']).agg(n=('correct','size'),accuracy=('correct','mean')).to_csv(HERE/'passes.csv')
    paired=[];uncertainty=[]
    groups=f.iloc[te].pass_group.to_numpy();unique=np.unique(groups)
    rng=np.random.default_rng(1024);draw=rng.integers(len(unique),size=(2000,len(unique)))
    for comparator in ('full','old_independent','repeat','multiscale','mix'):
        deltas=[]
        for seed in fit.SEEDS:
            before=scores[seed,comparator][te].argmax(1)==y[te]
            after=scores[seed,'difference'][te].argmax(1)==y[te]
            dd=after.astype(float)-before;deltas.append(dd)
            paired.append(dict(comparator=comparator,seed=seed,rescued=int((after&~before).sum()),harmed=int((before&~after).sum()),net=int(dd.sum()),OA_pp=100*dd.mean(),BA_pp=100*np.mean([dd[y[te]==c].mean() for c in range(8)])))
        delta=np.mean(deltas,axis=0)
        counts=np.array([[( (groups==g)&(y[te]==c)).sum() for c in range(8)] for g in unique])
        sums=np.array([[delta[(groups==g)&(y[te]==c)].sum() for c in range(8)] for g in unique])
        ct=counts[draw].sum(1);sm=sums[draw].sum(1)
        oa=100*sm.sum(1)/ct.sum(1)
        valid=(ct>0).all(1);ba=100*(sm[valid]/ct[valid]).mean(1)
        uncertainty.append(dict(comparator=comparator,clusters=len(unique),OA_low=float(np.quantile(oa,.025)),OA_high=float(np.quantile(oa,.975)),BA_low=float(np.quantile(ba,.025)),BA_high=float(np.quantile(ba,.975)),BA_complete_draws=int(valid.sum())))
    pd.DataFrame(paired).to_csv(HERE/'paired.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(HERE/'uncertainty.csv',index=False)
    # A transparent explanatory sample, not a prevalence estimate.
    candidates=[]
    for seed in fit.SEEDS:
        b=scores[seed,'old_independent'][te];a=scores[seed,'difference'][te]
        for k,j in enumerate(te):
            bc=b[k].argmax()==y[j];ac=a[k].argmax()==y[j]
            status='CC' if bc and ac else ('WW' if not bc and not ac else ('rescue' if ac else 'harm'))
            candidates.append(dict(seed=seed,local_index=int(j),case=status,change=float(margin(a[k:k+1],y[j:j+1])[0]-margin(b[k:k+1],y[j:j+1])[0])))
    cf=pd.DataFrame(candidates);chosen=[]
    for status in ('rescue','harm','CC','WW'):
        pool=cf[cf.case.eq(status)]
        if len(pool):
            # Prefer seed17, then the largest absolute margin change in that status.
            pool=pool[pool.seed.eq(pool.seed.min())]
            chosen.append(pool.loc[pool.change.abs().idxmax()].to_dict())
    table=pd.read_csv(pixel.UP/'table.csv',dtype={'mmsi':str})
    refs=np.load(fit.ROOT/'scores_layer1.npz')['refs']
    valid=np.load(fit.ROOT/'tokens.npz')['valid']
    snapshots={int(s['obs_id']):s for s in json.loads((pixel.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    out=HERE/'analysis';out.mkdir(exist_ok=True);records=[]
    local=np.load(HERE/'local_difference.npz')['local']
    for case in chosen:
        j=int(case['local_index']);i=int(f.iloc[j].table_index);seed=int(case['seed']);c=int(y[j])
        before=scores[seed,'old_independent'][j];after=scores[seed,'difference'][j]
        rival=int(np.where(np.arange(8)==c,-np.inf,before).argmax())
        rr=refs[i,[c,rival]].ravel()
        fig,ax=plt.subplots(2,8,figsize=(21,7));provenance=[]
        for p in range(2):
            pix,fp,g,placement=pixel.geo.native(table,snapshots,i,p)
            with Image.open(table.iloc[i]['image_'+('vv','vh')[p]]) as im:raw=np.asarray(im.convert('L'),dtype=np.float32)/255
            images=[pix.astype(np.float32)/255]+[pixel.place(gaussian_filter(raw,s,mode='reflect'),placement) for s in (1,4,12)]
            for k,image in enumerate(images):
                ax[p,k].imshow(image,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                if g is not None and valid[i,p]:
                    mask=pixel.geo.soft_regions(g)[0]*fp
                    if mask.max()>.5:ax[p,k].contour(mask,levels=[.5],colors=['#20a780'],linewidths=.6)
                ax[p,k].set_title(f'{("VV","VH")[p]} query {int(f.iloc[j].obs_id)}\n'+('raw' if k==0 else f'Gaussian sigma={(1,4,12)[k-1]}'),fontsize=8)
            provenance.append(dict(index=i,pol=p,path=table.iloc[i]['image_'+('vv','vh')[p]]))
            for k,ri in enumerate(rr,4):
                r,rfp,rg,_=pixel.geo.native(table,snapshots,int(ri),p)
                ax[p,k].imshow(r,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if rg is not None and valid[ri,p]:
                    mask=pixel.geo.soft_regions(rg)[0]*rfp
                    if mask.max()>.5:ax[p,k].contour(mask,levels=[.5],colors=['#20a780'],linewidths=.6)
                ax[p,k].set_title(f'{"TRUE" if k<6 else "RIVAL"} ref{int(table.iloc[ri].obs_id)} valid={int(valid[ri,p])}\n{table.iloc[ri].class_name}',fontsize=8)
                provenance.append(dict(index=int(ri),pol=p,path=table.iloc[ri]['image_'+('vv','vh')[p]]))
        for a in ax.ravel():a.set(xticks=[],yticks=[])
        fig.suptitle(f"{case['case']} | seed{seed} | {f.iloc[j].class_name} | old local {before.argmax()} -> scale difference {after.argmax()}\n"
            f"true margin {margin(before[None],y[j:j+1])[0]:+.3f} -> {margin(after[None],y[j:j+1])[0]:+.3f}; raw reference views shown, matching uses identical scales",fontsize=11)
        fig.text(.02,.025,'Native pixels, crop/pad160, no resize or added contrast stretch. Invalid polarization is excluded from matching.',fontsize=10)
        fig.subplots_adjust(top=.83,bottom=.09,wspace=.12,hspace=.18)
        path=out/f"{case['case']}_{int(f.iloc[j].obs_id)}.png";fig.savefig(path,dpi=130);plt.close(fig)
        # Exact four-channel Shapley attribution in the frozen head. Zero means utility mean, not no physical evidence.
        base=scores[seed,'full'][j:j+1];z=base-base.mean(-1,keepdims=True)
        comp=np.stack([np.delete(z,k,axis=1).max(1) for k in range(8)],1)
        raw=np.concatenate([local[j:j+1].reshape(1,8,48),z[...,None],comp[...,None],(z-comp)[...,None]],-1)
        ck=torch.load(HERE/'models'/f'difference_{seed}.pt',map_location='cpu',weights_only=False)
        x=np.where(ck['active'],(raw-ck['mu'])/ck['scale'],0).astype(np.float32)
        model=fit.Head();model.load_state_dict(ck['state']);model.eval();values=[]
        with torch.no_grad():
            for subset in range(16):
                xx=x.copy()
                for k in range(4):
                    if not subset&(1<<k):xx[:,:,k*12:(k+1)*12]=0
                s=(base+model(torch.tensor(xx)).numpy())[0];values.append(float(s[c]-s[rival]))
        attrib=[]
        for k in range(4):
            value=0.
            for subset in range(16):
                if subset&(1<<k):continue
                n=subset.bit_count();w=math.factorial(n)*math.factorial(3-n)/math.factorial(4)
                value+=w*(values[subset|(1<<k)]-values[subset])
            attrib.append(value)
        assert abs(sum(attrib)-(values[15]-values[0]))<1e-6
        np.testing.assert_allclose(values[15],after[c]-after[rival],atol=2e-5)
        records.append(dict(**case,obs_id=int(f.iloc[j].obs_id),true_class=c,rival=rival,old_pred=int(before.argmax()),new_pred=int(after.argmax()),
            old_margin=float(margin(before[None],y[j:j+1])[0]),new_margin=float(margin(after[None],y[j:j+1])[0]),
            raw_gap=(local[j,c,...,0]-local[j,rival,...,0]).tolist(),reference_obs=table.iloc[rr].obs_id.astype(int).tolist(),
            channel_shapley_fixed_margin=attrib,context_only_margin=values[0],full_fixed_margin=values[15],provenance=provenance,path=str(path)))
    (out/'cases.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(pd.DataFrame(paired).to_string(index=False));print(pd.DataFrame(uncertainty).to_string(index=False))
    print(pd.DataFrame(records)[['case','seed','obs_id','true_class','rival','old_margin','new_margin','channel_shapley_fixed_margin']].to_string(index=False))
    print('COHORT',len(te),f.iloc[te].mmsi.nunique(),f.iloc[te].pass_group.nunique())

if __name__=='__main__':main()
