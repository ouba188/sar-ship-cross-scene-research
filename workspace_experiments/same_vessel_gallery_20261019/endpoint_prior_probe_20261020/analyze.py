"""Inspect strong-end evidence without anatomical labels or model reselection."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from probe_endpoints import ROOT,PREV,old,endpoint_strength


def main():
    p=pd.read_csv(ROOT/'predictions.csv',dtype={'mmsi':str})
    a=p[p.arm.eq('hard_strong')].set_index('obs_id'); b=p[p.arm.eq('axis')].set_index('obs_id').loc[a.index]
    f0=pd.read_csv(old.prior.ROOT/'cohort.csv',dtype={'mmsi':str}); f=f0.set_index('obs_id')
    position={oid:i for i,oid in enumerate(f0.obs_id)}
    views={s['obs_id']:s for s in json.loads((old.LOCAL/'view_snapshot.json').read_text(encoding='utf-8'))}
    names=f.drop_duplicates('class_id').set_index('class_id').class_name.to_dict()
    previous=np.load(PREV/'descriptors.npz'); extra=np.load(ROOT/'endpoint_descriptors.npz')
    strength=pd.read_csv(ROOT/'endpoint_strength.csv')
    scope=strength[strength.obs_id.isin(a.index)]
    quality=scope.groupby('pol').agg(n=('obs_id','size'),peak_tie_rate=('peak_tie','mean'),
        mean_absolute_contrast=('contrast',lambda s:s.abs().mean()),
        zero_contrast_rate=('contrast',lambda s:(s==0).mean()),
        saturated_end0_rate=('end0_saturation',lambda s:(s>0).mean()),
        saturated_end2_rate=('end2_saturation',lambda s:(s>0).mean())).reset_index()
    quality.to_csv(ROOT/'test_strength_quality.csv',index=False)
    ships=sorted(a.mmsi.unique()); sid={s:i for i,s in enumerate(ships)}; rows=[]
    for arm in ('hard_strong','soft_strong'):
        alt=p[p.arm.eq(arm)].set_index('obs_id').loc[b.index]
        count=np.zeros((len(ships),8)); delta=np.zeros_like(count)
        for oid,r in alt.iterrows():
            i=sid[r.mmsi]; c=int(r.class_id); count[i,c]+=1; delta[i,c]+=int(r.correct)-int(b.loc[oid,'correct'])
        rng=np.random.default_rng(20261020); sample=[]
        for _ in range(500):
            ii=rng.integers(len(ships),size=len(ships)); n=count[ii].sum(0); d=delta[ii].sum(0)
            sample.append((d.sum()/n.sum(),np.mean(d[n>0]/n[n>0])))
        for j,key in enumerate(('oa','ba')):
            values=np.array(sample)[:,j]; rows.append(dict(arm=arm,metric=key,lo=np.quantile(values,.025),hi=np.quantile(values,.975)))
    pd.DataFrame(rows).to_csv(ROOT/'conditional_ci.csv',index=False)
    rows=[]
    for arm in ('hard_strong','soft_strong'):
        for intervention in ('reverse','zero'):
            data=[np.load(ROOT/f'intervention_f{k}_{arm}_{intervention}.npz') for k in range(3)]
            ids=np.concatenate([d['obs_id'] for d in data]); scores=np.concatenate([d['scores'] for d in data])
            yy=f.loc[ids].class_id.to_numpy(); original=p[p.arm.eq(arm)].set_index('obs_id').loc[ids]
            alt=scores.argmax(1)
            rows.append(dict(arm=arm,intervention=intervention,flips=int(np.sum(alt!=original.prediction.to_numpy())),
                rescue=int(np.sum((alt==yy)&~original.correct.to_numpy())),harm=int(np.sum((alt!=yy)&original.correct.to_numpy())),
                **old.prior.metrics(yy,scores)))
    pd.DataFrame(rows).to_csv(ROOT/'intervention_summary.csv',index=False)
    cases=[]
    for name,mask,ascending in [('rescue',a.correct&~b.correct,False),('harm',~a.correct&b.correct,True),
            ('persistent_correct',a.correct&b.correct,False),('persistent_wrong',~a.correct&~b.correct,True)]:
        gain=a.margin-b.margin
        for oid in gain[mask].sort_values(ascending=ascending).index[:2 if name in ('rescue','harm') else 1]: cases.append((name,int(oid)))
    if 30317 in a.index and 30317 not in [oid for _,oid in cases]: cases.append(('cross_response_anchor',30317))
    records=[]; contributions=[]
    for outcome,oid in cases:
        r=f.loc[oid]; aa=a.loc[oid]; bb=b.loc[oid]; i=position[oid]
        head=np.load(ROOT/f'head_f{int(aa.fold)}_hard_strong.npz')
        raw=np.concatenate((previous['base'][i],previous['axis'][i],extra['hard_strong'][i]))
        x=(raw-head['mean'])/head['std']/np.sqrt(len(raw)); scores=head['coef']@x+head['intercept']
        c=int(r.class_id); rival=int(bb.prediction) if int(bb.prediction)!=c else int(np.argmax(np.where(np.arange(8)==c,-np.inf,scores)))
        terms=x*(head['coef'][c]-head['coef'][rival]); bias=float(head['intercept'][c]-head['intercept'][rival])
        rec=dict(obs_id=oid,true_class=c,rival=rival,intercept=bias,shared_descriptor=float(terms[:-128].sum()),
            vv_ordered=float(terms[-128:-64].sum()),vh_ordered=float(terms[-64:].sum()),total_margin=float(scores[c]-scores[rival]))
        assert np.isclose(rec['intercept']+rec['shared_descriptor']+rec['vv_ordered']+rec['vh_ordered'],rec['total_margin'])
        contributions.append(rec)
        fig,axs=plt.subplots(2,2,figsize=(10,9))
        fig.suptitle(f'{outcome} obs {oid} | {r.class_name} | {r.port}\naxis {names[int(bb.prediction)]} ({bb.margin:+.3f}) -> strong end {names[int(aa.prediction)]} ({aa.margin:+.3f})')
        for j,pol in enumerate(('vv','vh')):
            image=np.asarray(Image.open(r['image_'+pol]).convert('L')); v=views[oid]['images'][pol]
            pix,m,_,place=old.regions(image,v,oid); contrast,stats=endpoint_strength(pix,m[3:6])
            axs[0,j].imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axs[0,j].set_title(f'{pol.upper()} native {image.shape[1]}x{image.shape[0]} | contrast {contrast:+.3f}')
            sx,sy=place['crop_x'],place['crop_y']; dx,dy=place['pad_left'],place['pad_top']; cw,ch=place['crop_width'],place['crop_height']
            slots=m[3:6,dy:dy+ch,dx:dx+cw]
            label=np.zeros(image.shape); canonical=np.sign(contrast)*(slots[2]-slots[0])
            label[sy:sy+ch,sx:sx+cw]=canonical
            axs[1,j].imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axs[1,j].imshow(np.ma.masked_where(label==0,label),cmap='coolwarm',vmin=-1,vmax=1,alpha=.65,interpolation='nearest')
            axs[1,j].set_title('Red: stronger end; blue: weaker end; NOT bow/stern')
            records.append(dict(outcome=outcome,obs_id=oid,pol=pol,path=r['image_'+pol],
                axis_prediction=int(bb.prediction),new_prediction=int(aa.prediction),axis_margin=bb.margin,new_margin=aa.margin,**stats))
        fig.tight_layout(); fig.savefig(ROOT/f'case_{outcome}_{oid}.png',dpi=130); plt.close(fig)
    pd.DataFrame(records).to_csv(ROOT/'case_review.csv',index=False)
    pd.DataFrame(contributions).to_csv(ROOT/'case_contributions.csv',index=False)
    print('cases',cases)
    for name in ('summary.csv','paired.csv','intervention_summary.csv','test_strength_quality.csv'):
        print(name); print(pd.read_csv(ROOT/name).to_string(index=False))


if __name__=='__main__': main()
