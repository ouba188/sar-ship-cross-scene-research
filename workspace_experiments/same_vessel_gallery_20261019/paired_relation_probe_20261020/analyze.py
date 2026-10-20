"""Matched outcomes and raw-chip case panels for the local relation probe."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from run import ROOT, ARMS, LOCAL, STRIP, load_data, matrices, prior


def read_scores(f, arm, seed):
    score=[]; base=[]; rows=[]
    for fold in range(3):
        data=np.load(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz')
        rows.extend(data['rows']); score.append(data['scores']); base.append(data['base'])
    return np.asarray(rows),np.concatenate(score),np.concatenate(base)


def main():
    f,splits=load_data(); names=f.groupby('class_id').class_name.first().to_dict()
    result=[]; grouped=[]; comparisons=[]; constants=[]
    for seed in range(3):
        stored={}
        for arm in ('baseline',)+ARMS:
            rows,scores,base=read_scores(f,arm,seed); yy=f.iloc[rows].class_id.to_numpy()
            correct=scores.argmax(1)==yy; bc=base.argmax(1)==yy
            stored[arm]=(rows,scores,base)
            result.append(dict(arm=arm,seed=seed,rescued=int((correct&~bc).sum()),harmed=int((~correct&bc).sum()),
                net=int(correct.sum()-bc.sum()),**prior.metrics(yy,scores)))
            for field in ('class_id','port'):
                for value in f.iloc[rows][field].unique():
                    sel=f.iloc[rows][field].to_numpy()==value
                    met=prior.metrics(yy[sel],scores[sel]); bm=prior.metrics(yy[sel],base[sel])
                    grouped.append(dict(arm=arm,seed=seed,field=field,group=value,delta_oa=met['oa']-bm['oa'],
                                        delta_ba=met['ba']-bm['ba'],**met))
            if arm!='baseline':
                cs=[]
                for fold in range(3):
                    d=np.load(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz')
                    cs.append(d['base']+d['constant'])
                constants.append(dict(arm=arm,seed=seed,**prior.metrics(yy,np.concatenate(cs))))
        for a,b in (('joint','factorized'),('joint','marginal'),('joint_condition','joint'),
                    ('joint_condition','factorized_condition'),('joint_condition','joint_mismatched')):
            rows,sa,base=stored[a]; rb,sb,_=stored[b]; np.testing.assert_array_equal(rows,rb)
            yy=f.iloc[rows].class_id.to_numpy(); ca=sa.argmax(1)==yy; cb=sb.argmax(1)==yy
            ma=prior.metrics(yy,sa); mb=prior.metrics(yy,sb)
            comparisons.append(dict(seed=seed,arm=a,reference=b,rescued=int((ca&~cb).sum()),
                harmed=int((~ca&cb).sum()),delta_oa=ma['oa']-mb['oa'],delta_ba=ma['ba']-mb['ba']))
    per=pd.DataFrame(result); per.to_csv(ROOT/'per_seed.csv',index=False)
    summary=per.groupby('arm')[['oa','ba','brier','net']].agg(['mean','std'])
    summary.to_csv(ROOT/'summary.csv')
    pd.DataFrame(grouped).to_csv(ROOT/'groups.csv',index=False)
    pd.DataFrame(comparisons).to_csv(ROOT/'comparisons.csv',index=False)
    pd.DataFrame(constants).to_csv(ROOT/'source_constant_summary.csv',index=False)
    # Predeclared primary arm; do not choose a model by test performance.
    rows,scores,base=read_scores(f,'joint_condition',0); yy=f.iloc[rows].class_id.to_numpy()
    rivals=np.argmax(np.where(np.arange(8)[None,:]==yy[:,None],-np.inf,base),1)
    margins=scores[np.arange(len(rows)),yy]-scores[np.arange(len(rows)),rivals]
    bm=base[np.arange(len(rows)),yy]-base[np.arange(len(rows)),rivals]
    delta=margins-bm; correct=scores.argmax(1)==yy; bc=base.argmax(1)==yy
    chosen=[]
    for label,mask in (('rescue',correct&~bc),('harm',~correct&bc),
                        ('persistent_correct',correct&bc),('persistent_wrong',~correct&~bc)):
        js=np.flatnonzero(mask); js=js[np.argsort(-np.abs(delta[js]))]
        for j in js[:2 if label in ('rescue','harm') else 1]:
            chosen.append((label,int(j)))
    views={v['obs_id']:v for v in json.loads((LOCAL/'view_snapshot.json').read_text(encoding='utf-8'))}
    lookup={int(i):j for j,i in enumerate(rows)}; records=[]; terms=[]
    features={}; heads={}
    for label,j in chosen:
        i=int(rows[j]); row=f.iloc[i]; fold=int(row.fold); oid=int(row.obs_id)
        if fold not in features:
            z,_=matrices(np.load(ROOT/f'features_f{fold}.npz'),f,splits[fold],fold)
            features[fold]=z['joint_condition']
            heads[fold]=torch.load(ROOT/f'head_f{fold}_joint_condition_s0.pt',map_location='cpu',weights_only=False)
        weight=heads[fold]['state']['weight'].numpy()
        contribution=(features[fold][i]*(weight[int(yy[j])]-weight[int(rivals[j])])).reshape(2,2,-1).sum(-1)
        assert abs(contribution.sum()-delta[j])<2e-5
        for p,pol in enumerate(('vv','vh')):
            for ref in range(2):
                terms.append(dict(obs_id=oid,pol=pol,reference=('ship','radar_or_fallback')[ref],margin_term=float(contribution[p,ref])))
        others=f.iloc[rows][f.iloc[rows].mmsi.eq(row.mmsi)&f.iloc[rows].class_id.eq(row.class_id)&f.iloc[rows]['product'].ne(row['product'])].copy()
        # Same-ship panel partner chosen by time separation, not prediction.
        if len(others):
            others['gap']=(pd.to_datetime(others.time)-pd.Timestamp(row.time)).abs()
            partner=others.sort_values(['gap','obs_id'],ascending=[False,True]).iloc[0]
            panel=[row,partner]
        else:
            panel=[row]
        fig,axs=plt.subplots(len(panel),2,figsize=(10,4.5*len(panel)),squeeze=False)
        fig.suptitle(f'{label}: {oid}, {row.class_name}; baseline -> joint_condition, seed0')
        for rr,item in enumerate(panel):
            pi=int(item.name); jj=lookup[pi]; v=views[int(item.obs_id)]
            for pp,pol in enumerate(('vv','vh')):
                raw=np.asarray(Image.open(item['image_'+pol]).convert('L'))
                ax=axs[rr,pp]; ax.imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axis=v['images'][pol]['priors'].get('ship_axis')
                if axis:
                    cx,cy=axis['center']; angle=axis['angle']; half=axis['length_px']/2
                    ax.plot([cx-half*np.cos(angle),cx+half*np.cos(angle)],
                            [cy-half*np.sin(angle),cy+half*np.sin(angle)],color='cyan',linewidth=.8)
                ax.set_title(f'{pol.upper()} obs{int(item.obs_id)} {raw.shape[1]}x{raw.shape[0]}\n'
                    f'{str(item.time)[:10]} | {names[int(base[jj].argmax())]} -> {names[int(scores[jj].argmax())]}\n'
                    f'true-vs-fixed-rival margin {bm[jj]:+.3f} -> {margins[jj]:+.3f}',fontsize=9)
                ax.set_xlabel('Native pixels; cyan = detected unsigned ship axis')
        fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(ROOT/f'case_{label}_{oid}.png',dpi=125); plt.close(fig)
        records.append(dict(case=label,obs_id=oid,mmsi=row.mmsi,class_id=int(yy[j]),fold=fold,seed=0,
            base_pred=int(base[j].argmax()),prediction=int(scores[j].argmax()),base_margin=float(bm[j]),margin=float(margins[j]),
            partner_obs=int(partner.obs_id) if len(others) else -1,
            image_vv=row.image_vv,image_vh=row.image_vh,step=heads[fold]['step']))
    pd.DataFrame(records).to_csv(ROOT/'case_review.csv',index=False)
    pd.DataFrame(terms).to_csv(ROOT/'case_reference_terms.csv',index=False)
    print(summary.to_string(),flush=True)


if __name__=='__main__':
    main()
