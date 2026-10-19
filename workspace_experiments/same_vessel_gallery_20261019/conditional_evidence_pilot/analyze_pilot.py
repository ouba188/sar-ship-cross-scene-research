"""Paired task outcomes and raw-chip review selection for the fixed probe."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def aggregate(g):
    return dict(n=len(g),ships=g.mmsi.nunique(),oa=g.correct.mean(),ba=g.groupby('class_id').correct.mean().mean(),ship_ba=g.groupby(['class_id','mmsi']).correct.mean().groupby('class_id').mean().mean(),brier=g.brier.mean())


def bootstrap(base,other):
    merged=base.merge(other,on=['obs_id','mmsi','class_id'],suffixes=('_base','_new'),validate='one_to_one')
    merged['delta']=merged.correct_new.astype(float)-merged.correct_base.astype(float)
    ship=merged.groupby(['class_id','mmsi']).agg(n=('delta','size'),d=('delta','sum')).reset_index()
    groups=[g[['n','d']].to_numpy() for _,g in ship.groupby('class_id')]
    rng=np.random.default_rng(20261019); oa=[]; ba=[]; sb=[]
    for _ in range(500):
        sampled=[g[rng.integers(len(g),size=len(g))] for g in groups]
        n=sum(g[:,0].sum() for g in sampled); d=sum(g[:,1].sum() for g in sampled)
        oa.append(d/n); ba.append(np.mean([g[:,1].sum()/g[:,0].sum() for g in sampled])); sb.append(np.mean([np.mean(g[:,1]/g[:,0]) for g in sampled]))
    result={}
    for name,values in (('oa',oa),('ba',ba),('ship_ba',sb)):
        result[name+'_ci_low'],result[name+'_ci_high']=np.quantile(values,[.025,.975])
    result.update(rescues=int(((~merged.correct_base)&merged.correct_new).sum()),harms=int((merged.correct_base&(~merged.correct_new)).sum()))
    return result


def render(ids,reason,f,p,classes):
    fig,axes=plt.subplots(len(ids),2,figsize=(9,4.1*len(ids)),squeeze=False)
    details=[]
    for r,oid in enumerate(ids):
        o=f.loc[oid]; scores=p[p.obs_id.eq(oid)].set_index('arm')
        for col,pol in enumerate(('vv','vh')):
            im=np.asarray(Image.open(o['image_'+pol])); ax=axes[r,col]
            ax.imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest'); ax.set_aspect('equal')
            ax.set_title(f"obs {oid} | {pol.upper()} | {im.shape[1]}x{im.shape[0]} | {o['port']}\n{o['time']} | true {o.class_name}\nimage: {classes[int(scores.loc['image','prediction'])]} ({scores.loc['image','margin']:+.3f})\ninteraction: {classes[int(scores.loc['interaction','prediction'])]} ({scores.loc['interaction','margin']:+.3f})",fontsize=9)
            details.append(dict(obs_id=oid,reason=reason,pol=pol,path=o['image_'+pol],mmsi=o.mmsi,incidence=o.incidence,axis_deg=o.axis_deg,los_axis_deg=o.los_axis_deg,elongation=o.elongation,match_status=o.match_status))
    fig.suptitle(reason+' | source-fitted fixed-feature probe, not full model',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(ROOT/('case_'+str(ids[0])+'.png'),dpi=130); plt.close(fig)
    return details


def main():
    p=pd.read_csv(ROOT/'predictions.csv',dtype={'mmsi':str}); f=pd.read_csv(ROOT/'cohort.csv',dtype={'mmsi':str}).set_index('obs_id')
    summary=[]
    for arm,g in p.groupby('arm'): summary.append(dict(arm=arm,**aggregate(g)))
    s=pd.DataFrame(summary).set_index('arm'); s.to_csv(ROOT/'summary.csv')
    comparisons=[]
    for other,base in [('interaction','image'),('concat','image'),('interaction','concat'),('interaction','content_interaction'),('interaction','mismatched_interaction')]:
        comparisons.append(dict(candidate=other,baseline=base,**{key+'_delta':s.loc[other,key]-s.loc[base,key] for key in ('oa','ba','ship_ba','brier')},**bootstrap(p[p.arm.eq(base)],p[p.arm.eq(other)])))
    pd.DataFrame(comparisons).to_csv(ROOT/'paired_results.csv',index=False)
    for field in ('port','class_id','fold'):
        rows=[]
        for (arm,key),g in p.groupby(['arm',field]): rows.append(dict(arm=arm,group=key,**aggregate(g)))
        pd.DataFrame(rows).to_csv(ROOT/f'by_{field}.csv',index=False)
    # Geometry confidence is an observed proxy, not physical ground truth.
    pp=p.merge(f[['elongation','match_status']],left_on='obs_id',right_index=True,validate='many_to_one')
    pp['axis_group']=np.where(pp.elongation.ge(1.5)&pp.match_status.eq('inside_unique_box'),'elongated_unique','other')
    rows=[]
    for (arm,key),g in pp.groupby(['arm','axis_group']): rows.append(dict(arm=arm,group=key,**aggregate(g)))
    pd.DataFrame(rows).to_csv(ROOT/'by_axis_group.csv',index=False)
    a=p[p.arm.eq('image')].set_index('obs_id'); b=p[p.arm.eq('interaction')].set_index('obs_id')
    joined=a[['mmsi','correct','margin']].join(b[['correct','margin']],rsuffix='_new')
    joined['gain']=joined.margin_new-joined.margin
    selections=[]; used=set(); classes=f[['class_id','class_name']].drop_duplicates().set_index('class_id').class_name.to_dict()
    for reason,mask,ascending in [('rescue',~joined.correct&joined.correct_new,False),('harm',joined.correct&~joined.correct_new,True),('persistent_correct',joined.correct&joined.correct_new,False),('persistent_wrong',~joined.correct&~joined.correct_new,True)]:
        candidates=joined[mask&~joined.mmsi.isin(used)].sort_values('gain',ascending=ascending)
        if candidates.empty: continue
        oid=int(candidates.index[0]); used.add(f.loc[oid,'mmsi']); ids=[oid]
        mates=joined[joined.mmsi.eq(f.loc[oid,'mmsi'])&joined.index.to_series().ne(oid)]
        # Prefer the same ship with a different base outcome, when available.
        opposite=mates[mates.correct.ne(joined.loc[oid,'correct'])]
        if len(opposite): mates=opposite
        if len(mates): ids.append(int(mates.index[0]))
        selections+=render(ids,reason,f,p,classes)
    pd.DataFrame(selections).to_csv(ROOT/'review_cases.csv',index=False)
    print(s.to_string())
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__=='__main__': main()
