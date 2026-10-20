"""Paired uncertainty and actual native-image case panels; no model selection."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from probe import ROOT,LOCAL,prior,regions


def main():
    p=pd.read_csv(ROOT/'predictions.csv',dtype={'mmsi':str})
    a=p[p.arm.eq('axis')].set_index('obs_id')
    b=p[p.arm.eq('whole')].set_index('obs_id').loc[a.index]
    classes=sorted(a.class_id.unique()); ships=sorted(a.mmsi.unique())
    sid={s:i for i,s in enumerate(ships)}
    count=np.zeros((len(ships),8)); diff=np.zeros_like(count)
    for oid,r in a.iterrows():
        i=sid[r.mmsi]; c=int(r.class_id); count[i,c]+=1
        diff[i,c]+=int(r.correct)-int(b.loc[oid,'correct'])
    rng=np.random.default_rng(20261020); deltas=[]
    for _ in range(500):
        ii=rng.integers(len(ships),size=len(ships)); n=count[ii].sum(0); d=diff[ii].sum(0)
        deltas.append((d.sum()/n.sum(),np.mean(d[n>0]/n[n>0])))
    pd.DataFrame([dict(metric=k,lo=float(np.quantile(np.array(deltas)[:,j],.025)),hi=float(np.quantile(np.array(deltas)[:,j],.975))) for j,k in enumerate(('oa','ba'))]).to_csv(ROOT/'conditional_ci.csv',index=False)
    f=pd.read_csv(prior.ROOT/'cohort.csv',dtype={'mmsi':str}).set_index('obs_id')
    views={s['obs_id']:s for s in json.loads((LOCAL/'view_snapshot.json').read_text(encoding='utf-8'))}
    names=f.drop_duplicates('class_id').set_index('class_id').class_name.to_dict()
    chosen=[]
    for name,mask,ascending in [('rescue',a.correct&~b.correct,False),('harm',~a.correct&b.correct,True),
                                ('persistent_correct',a.correct&b.correct,False),('persistent_wrong',~a.correct&~b.correct,True)]:
        gain=a.margin-b.margin; candidates=gain[mask].sort_values(ascending=ascending)
        for oid in candidates.index[:2 if name in ('rescue','harm') else 1]: chosen.append((name,int(oid)))
    records=[]
    contributions=[]
    descriptors=np.load(ROOT/'descriptors.npz')
    order=pd.read_csv(prior.ROOT/'cohort.csv').obs_id.tolist()
    position={oid:i for i,oid in enumerate(order)}
    for outcome,oid in chosen:
        fig,axs=plt.subplots(2,2,figsize=(10,9))
        row=f.loc[oid]; rr=a.loc[oid]; bb=b.loc[oid]
        head=np.load(ROOT/f'head_f{int(rr.fold)}_axis.npz')
        i=position[oid]; raw=np.concatenate((descriptors['base'][i],descriptors['axis'][i]))
        x=(raw-head['mean'])/head['std']/np.sqrt(len(raw))
        scores=head['coef']@x+head['intercept']
        rival=int(bb.prediction) if int(bb.prediction)!=int(row.class_id) else int(np.argsort(np.where(np.arange(8)==int(row.class_id),-np.inf,scores))[-1])
        terms=x*(head['coef'][int(row.class_id)]-head['coef'][rival])
        intercept=float(head['intercept'][int(row.class_id)]-head['intercept'][rival])
        record=dict(obs_id=oid,true_class=int(row.class_id),rival=rival,intercept=intercept,
            shared_descriptor=float(terms[:descriptors['base'].shape[1]].sum()),
            total_margin=float(scores[int(row.class_id)]-scores[rival]))
        offset=descriptors['base'].shape[1]
        for pj,pol in enumerate(('vv','vh')):
            for gj,label in enumerate(('middle','ends_mean','ends_difference')):
                begin=offset+pj*192+gj*64
                record[pol+'_'+label]=float(terms[begin:begin+64].sum())
        assert np.isclose(sum(v for k,v in record.items() if k not in ('obs_id','true_class','rival','total_margin')),record['total_margin'])
        contributions.append(record)
        fig.suptitle(f'{outcome} obs {oid} | {row.class_name} | {row.port}\nwhole {names[int(bb.prediction)]} ({bb.margin:+.3f}) -> axis {names[int(rr.prediction)]} ({rr.margin:+.3f})')
        for j,pol in enumerate(('vv','vh')):
            raw=np.asarray(Image.open(row['image_'+pol]).convert('L'))
            v=views[oid]['images'][pol]; _,m,_,place=regions(raw,v,oid)
            axs[0,j].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axs[0,j].set_title(f'{pol.upper()} original {raw.shape[1]}x{raw.shape[0]}')
            sx,sy=place['crop_x'],place['crop_y']; dx,dy=place['pad_left'],place['pad_top']
            cw,ch=place['crop_width'],place['crop_height']
            label=np.zeros(raw.shape)
            slots=m[3:6,dy:dy+ch,dx:dx+cw]
            label[sy:sy+ch,sx:sx+cw]=sum((k+1)*slots[k] for k in range(3))
            axs[1,j].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axs[1,j].imshow(np.ma.masked_where(label==0,label),cmap='coolwarm',vmin=1,vmax=3,alpha=.6,interpolation='nearest')
            axs[1,j].set_title('Axial thirds: interchangeable ends, middle')
            axis=v['priors'].get('ship_axis')
            records.append(dict(outcome=outcome,obs_id=oid,pol=pol,path=row['image_'+pol],width=raw.shape[1],height=raw.shape[0],
                target_length_px=axis['length_px'] if axis else None,target_width_px=axis['width_px'] if axis else None,
                whole_prediction=int(bb.prediction),axis_prediction=int(rr.prediction),whole_margin=bb.margin,axis_margin=rr.margin,
                axis_minus_whole_margin=rr.margin-bb.margin,**place))
        fig.tight_layout(); fig.savefig(ROOT/f'case_{outcome}_{oid}.png',dpi=130); plt.close(fig)
    pd.DataFrame(records).to_csv(ROOT/'case_review.csv',index=False)
    pd.DataFrame(contributions).to_csv(ROOT/'case_contributions.csv',index=False)
    print('cases',chosen)
    print(pd.read_csv(ROOT/'summary.csv').to_string(index=False))
    print(pd.read_csv(ROOT/'paired.csv').to_string(index=False))


if __name__=='__main__': main()
