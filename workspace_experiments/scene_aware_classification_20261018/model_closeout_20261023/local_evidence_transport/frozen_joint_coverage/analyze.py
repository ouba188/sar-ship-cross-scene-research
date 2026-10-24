"""Paired coverage statistics and native query/reference case attribution."""
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

ev=load('coverage_eval',HERE/'evaluate.py')
ex=load('coverage_extract',HERE/'extract.py')

def margin(s,y):
    return s[np.arange(len(y)),y]-np.where(np.arange(8)[None]==y[:,None],-np.inf,s).max(1)

def main():
    torch.set_num_threads(2)
    metrics=pd.read_csv(HERE/'metrics.csv')
    summary=metrics.groupby(['dataset','plan','scope','arm'])[['OA','BA','NLL','Brier']].agg(['mean','std'])
    summary.to_csv(HERE/'summary.csv')
    paired=[];predictions=[];intervals=[];candidates=[];coverage=[]
    for dataset in ('development','expanded'):
        f=pd.read_csv(HERE/f'{dataset}_table.csv',dtype={'mmsi':str});y=f.class_id.to_numpy(int)
        pg=f.pass_group.to_numpy();unique=np.unique(pg);groups={g:pg==g for g in unique}
        counts=np.array([[((pg==g)&(y==c)).sum() for c in range(8)] for g in unique])
        rng=np.random.default_rng(1024);draws=rng.integers(len(unique),size=(2000,len(unique)))
        ct=counts[draws].sum(1);ok=(ct>0).all(1)
        for cls,g in f.groupby(['class_id','class_name']):
            coverage.append(dict(dataset=dataset,class_id=cls[0],class_name=cls[1],n=len(g),ships=g.mmsi.nunique(),passes=g.pass_group.nunique()))
        for plan in metrics.plan.unique():
            deltas=[]
            for seed in ev.SEEDS:
                z=np.load(HERE/'results'/f'{dataset}_{plan}_{seed}.npz')
                b=z['strong'].argmax(1)==y;a=z['metric_only'].argmax(1)==y;d=a.astype(float)-b
                deltas.append(d)
                paired.append(dict(dataset=dataset,plan=plan,seed=seed,n=len(y),rescued=int((a&~b).sum()),harmed=int((b&~a).sum()),OA_pp=100*d.mean(),BA_pp=100*np.mean([d[y==c].mean() for c in range(8)])))
                for arm in ev.ARMS:
                    part=f[['obs_id','mmsi','class_id','class_name','pass_group','port']].copy()
                    part['dataset']=dataset;part['plan']=plan;part['seed']=seed;part['arm']=arm
                    part['prediction']=z[arm].argmax(1);part['correct']=part.prediction.eq(part.class_id)
                    part['margin']=margin(z[arm],y);predictions.append(part)
                if dataset=='expanded' and plan=='nearest2' and seed!=17:
                    changes=margin(z['metric_only'],y)-margin(z['strong'],y)
                    for j in range(len(y)):
                        status='CC' if a[j] and b[j] else ('WW' if not a[j] and not b[j] else ('rescue' if a[j] else 'harm'))
                        candidates.append(dict(index=j,seed=seed,status=status,change=float(changes[j])))
            d=np.mean(deltas,0);sums=np.array([[d[(pg==g)&(y==c)].sum() for c in range(8)] for g in unique])
            sm=sums[draws].sum(1);oa=100*sm.sum(1)/ct.sum(1);ba=100*(sm[ok]/ct[ok]).mean(1)
            perpass=np.array([d[groups[g]].mean() for g in unique])
            intervals.append(dict(dataset=dataset,plan=plan,OA_low=np.quantile(oa,.025),OA_high=np.quantile(oa,.975),BA_low=np.quantile(ba,.025),BA_high=np.quantile(ba,.975),BA_draws=int(ok.sum()),positive_passes=int((perpass>1e-12).sum()),negative_passes=int((perpass < -1e-12).sum()),equal_passes=int((abs(perpass)<=1e-12).sum())))
    pd.DataFrame(paired).to_csv(HERE/'paired.csv',index=False)
    pd.DataFrame(intervals).to_csv(HERE/'uncertainty.csv',index=False)
    pd.DataFrame(coverage).to_csv(HERE/'coverage.csv',index=False)
    detail=pd.concat(predictions,ignore_index=True);detail.to_csv(HERE/'predictions.csv.gz',index=False,compression='gzip')
    primary=detail[detail.plan.eq('nearest2')]
    for field in ('class_name','pass_group','port'):
        primary.groupby(['dataset','seed','arm',field]).agg(n=('correct','size'),accuracy=('correct','mean'),margin=('margin','mean')).to_csv(HERE/f'by_{field}.csv')
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    comparison=pd.DataFrame(paired).query("dataset=='expanded'").groupby('plan')[['OA_pp','BA_pp']].mean()
    order=list(metrics.plan.unique());comparison=comparison.loc[order]
    xx=np.arange(len(order))
    ax[0].bar(xx-.18,comparison.OA_pp,width=.36,label='OA',color='#287c8e')
    ax[0].bar(xx+.18,comparison.BA_pp,width=.36,label='BA',color='#b54b70')
    ax[0].set_xticks(xx,order,rotation=35,ha='right');ax[0].axhline(0,color='black',linewidth=.7)
    ax[0].set(ylabel='Fixed-T minus same-reference strong (pp)',title='Expanded queries: frozen reference sensitivity');ax[0].legend()
    cls=primary[primary.dataset.eq('expanded')].groupby(['arm','class_id','class_name']).correct.mean().unstack('arm')
    gain=100*(cls.metric_only-cls.strong)
    labels=[v[1].replace('_',' ') for v in gain.index]
    ax[1].barh(np.arange(len(gain)),gain,color=['#287c8e' if v>=0 else '#b54b70' for v in gain])
    ax[1].set_yticks(np.arange(len(gain)),labels);ax[1].axvline(0,color='black',linewidth=.7)
    ax[1].set(xlabel='Recall change (pp)',title='Original nearest2: class-specific change')
    fig.suptitle('3,075 historical isolated queries; three-configuration mean; no new training')
    fig.tight_layout();fig.savefig(HERE/'coverage_summary.png',dpi=160);plt.close(fig)
    # Each case is selected after prediction for explanation, never for model choice.
    chosen=[]
    for status in ('rescue','harm','CC','WW'):
        pool=[v for v in candidates if v['status']==status]
        if pool:chosen.append(max(pool,key=lambda v:abs(v['change'])))
    source=pd.read_csv(ev.UP/'table.csv',dtype={'mmsi':str})
    f=pd.read_csv(HERE/'expanded_table.csv',dtype={'mmsi':str});y=f.class_id.to_numpy(int)
    refs=np.load(HERE/'expanded_references.npz');bank=refs['bank'];picks=refs['nearest2']
    snapshots={int(s['obs_id']):s for s in json.loads((ex.joint.geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    target_tok=np.load(ev.EXT/'tokens.npz');source_tok=np.load(ev.ROOT/'tokens.npz')
    out=HERE/'analysis';out.mkdir(exist_ok=True);records=[]
    for case in chosen:
        j=int(case['index']);seed=int(case['seed']);c=int(y[j]);z=np.load(HERE/'results'/f'expanded_nearest2_{seed}.npz')
        before=z['strong'][j];after=z['metric_only'][j];rival=int(np.where(np.arange(8)==c,-np.inf,before).argmax())
        rr=np.array([bank[k,picks[j,k]] for k in (c,rival)]).ravel()
        fig,ax=plt.subplots(2,5,figsize=(16,7));provenance=[]
        for p in range(2):
            for k,i in enumerate([j,*rr]):
                table=f if k==0 else source;tok=target_tok if k==0 else source_tok
                pix,fp,g,placement=ex.joint.geo.native(table,snapshots,int(i),p)
                ax[p,k].imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                valid=int(tok['valid'][i,p])
                if valid and g is not None:
                    mask=ex.joint.geo.soft_regions(g)[0]*fp
                    if mask.max()>.5:ax[p,k].contour(mask,levels=[.5],colors=['#20a780'],linewidths=.6)
                label='QUERY' if k==0 else ('TRUE' if k<3 else 'RIVAL')
                ax[p,k].set_title(f'{("VV","VH")[p]} {label} {int(table.iloc[i].obs_id)} valid={valid}\n{table.iloc[i].class_name}',fontsize=8)
                ax[p,k].set(xticks=[],yticks=[])
                provenance.append(dict(obs_id=int(table.iloc[i].obs_id),pol=p,path=table.iloc[i]['image_'+('vv','vh')[p]],valid=valid))
        oldmargin=float(margin(before[None],y[j:j+1])[0]);newmargin=float(margin(after[None],y[j:j+1])[0])
        fig.suptitle(f"Expanded {case['status']} seed{seed}: strong {before.argmax()} -> fixed-T {after.argmax()} (true {c})\nmargin {oldmargin:+.4f} -> {newmargin:+.4f}")
        fig.text(.02,.02,'Native160 crop/pad; no resize. Green=body. Same selected source references. Appearance is not physical causality.',fontsize=9)
        fig.subplots_adjust(top=.83,bottom=.09,wspace=.12,hspace=.15)
        path=out/f"{case['status']}_{int(f.iloc[j].obs_id)}_{seed}.png";fig.savefig(path,dpi=120);plt.close(fig)
        head=torch.load(ev.ROOT/'models'/f'layer1_independent_{seed}.pt',map_location='cpu',weights_only=False)
        base=np.load(ev.EXT/f'strategies_seed{seed}.npz')['logits'][j:j+1,2].astype(np.float32)
        oldloc=z['strong_local'][j:j+1];newloc=z['metric_only_local'][j:j+1];values=[]
        for subset in range(64):
            loc=oldloc.copy()
            for k in range(6):
                if subset&(1<<k):loc[:,:,k//3,k%3]=newloc[:,:,k//3,k%3]
            s=ev.classify(loc,base,head)[0];values.append(float(s[c]-s[rival]))
        attribution=[]
        for k in range(6):
            val=0.
            for subset in range(64):
                if subset&(1<<k):continue
                n=subset.bit_count();val+=math.factorial(n)*math.factorial(5-n)/math.factorial(6)*(values[subset|(1<<k)]-values[subset])
            attribution.append(val)
        assert abs(sum(attribution)-(values[-1]-values[0]))<1e-6
        np.testing.assert_allclose([values[0],values[-1]],[before[c]-before[rival],after[c]-after[rival]],atol=3e-5)
        records.append(dict(**case,obs_id=int(f.iloc[j].obs_id),true_class=c,rival=rival,old_pred=int(before.argmax()),new_pred=int(after.argmax()),old_margin=oldmargin,new_margin=newmargin,region_shapley=attribution,reference_obs=source.iloc[rr].obs_id.astype(int).tolist(),true_local_change=(newloc[0,c]-oldloc[0,c]).tolist(),rival_local_change=(newloc[0,rival]-oldloc[0,rival]).tolist(),provenance=provenance,path=str(path)))
    (out/'cases.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (HERE/'analysis_receipt.json').write_text(json.dumps(dict(cases=len(records),bootstrap=2000,unit='pass-group after mean across3 configurations; BA conditional all8classes',new_fits=0),indent=2),encoding='utf-8')
    print(pd.DataFrame(paired).groupby(['dataset','plan'])[['OA_pp','BA_pp','rescued','harmed']].mean().to_string(),flush=True)
    print(pd.DataFrame(records)[['status','seed','obs_id','old_margin','new_margin','region_shapley']].to_string(index=False),flush=True)

if __name__=='__main__':main()
