"""Paired outcomes, gate diagnostics, and native-image cases."""
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw,ImageFont
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
import run as exp

ROOT=exp.ROOT


def met(g):
    return dict(oa=accuracy_score(g.class_id,g.prediction),ba=g.groupby('class_id').correct.mean().mean(),
                macro_f1=f1_score(g.class_id,g.prediction,average='macro',zero_division=0))


def summarize(p):
    result=[];class_rows=[];group_rows=[]
    for (seed,arm),a in p.groupby(['seed','arm']):
        ref=p[(p.seed==seed)&(p.arm=='reference')].set_index('obs_id').loc[a.obs_id]
        assert len(a)==4744 and a.obs_id.nunique()==4744
        result.append(dict(seed=seed,arm=arm,**met(a),
            rescue=int((a.correct.to_numpy()&~ref.correct.to_numpy()).sum()),
            harm=int((~a.correct.to_numpy()&ref.correct.to_numpy()).sum()),
            mean_weight=a.weight.mean()))
        for c,b in a.groupby('class_id'):class_rows.append(dict(seed=seed,arm=arm,class_id=c,n=len(b),recall=b.correct.mean()))
        for key in ('port','geometry_both','r_nonzero'):
            for value,b in a.groupby(key):group_rows.append(dict(seed=seed,arm=arm,group=key,value=value,n=len(b),**met(b)))
    result=pd.DataFrame(result);result.to_csv(ROOT/'summary_by_seed.csv',index=False,lineterminator='\n')
    result.groupby('arm')[['oa','ba','macro_f1','rescue','harm','mean_weight']].agg(['mean','std']).to_csv(ROOT/'summary_mean_std.csv',lineterminator='\n')
    pd.DataFrame(class_rows).to_csv(ROOT/'class_recall.csv',index=False,lineterminator='\n')
    pd.DataFrame(group_rows).to_csv(ROOT/'group_metrics.csv',index=False,lineterminator='\n')
    print(result.groupby('arm')[['oa','ba','macro_f1','rescue','harm','mean_weight']].mean().to_string())


def paired(p):
    avg=p.groupby(['arm','obs_id']).agg(correct=('correct','mean'),mmsi=('mmsi','first'),class_id=('class_id','first')).reset_index()
    pairs=[(a,'reference') for a in exp.ARMS]+[('joint',a) for a in ('content','constant','geometry','sham','joint_source_mean_weight')]
    rows=[]
    for arm,ref in pairs:
        a=avg[avg.arm==arm].set_index('obs_id').copy();b=avg[avg.arm==ref].set_index('obs_id').loc[a.index]
        a['delta']=a.correct-b.correct;a['n']=1
        ships=a.groupby(['mmsi','class_id'])[['delta','n']].sum().reset_index();assert ships.mmsi.nunique()==len(ships)
        rng=np.random.default_rng(20261021);ds=[];ns=[]
        for c in range(8):
            g=ships[ships.class_id==c];ii=rng.integers(len(g),size=(1500,len(g)))
            ds.append(g.delta.to_numpy()[ii].sum(1));ns.append(g.n.to_numpy()[ii].sum(1))
        ds,ns=np.array(ds).T,np.array(ns).T
        for metric,point,boot in (('oa',a.delta.mean(),ds.sum(1)/ns.sum(1)),
                                  ('ba',a.groupby('class_id').delta.mean().mean(),(ds/ns).mean(1))):
            lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(arm=arm,reference=ref,metric=metric,
                difference=point,ci_low=lo,ci_high=hi,replicates=1500,scope='fixed models, run-averaged correctness, class-stratified MMSI'))
    pd.DataFrame(rows).to_csv(ROOT/'paired_uncertainty.csv',index=False,lineterminator='\n')


def verify_selection():
    val=pd.read_csv(ROOT/'validation_selection.csv');rows=[]
    for (fold,seed,arm),g in val.groupby(['fold','seed','arm']):
        chosen=g.sort_values(['ba','brier','step'],ascending=[False,True,True]).iloc[0]
        saved=torch.load(ROOT/f'head_f{fold}_{arm}_s{seed}.pt',map_location='cpu',weights_only=False)
        assert saved['step']==chosen.step
        rows.append(dict(fold=fold,seed=seed,arm=arm,step=int(chosen.step),val_ba_gain=float(chosen.ba-g[g.step.eq(0)].ba.iloc[0])))
    pd.DataFrame(rows).to_csv(ROOT/'selected_steps.csv',index=False,lineterminator='\n')
    (ROOT/'VERIFICATION.json').write_text(json.dumps(dict(status='passed',source_selected_heads=len(rows),
        replay_checks=len(pd.read_csv(ROOT/'checks.csv')),test_rows_per_run=4744),indent=2),encoding='utf-8')


def cases(p):
    f,views,disabled=exp.old.control.load_data();index=dict(zip(f.obs_id,f.index));g,v=exp.old.relation.conditions(f)
    q=p[p.seed.eq(0)].set_index(['arm','obs_id']);a=q.loc['joint'];r=q.loc['reference'].loc[a.index]
    selected=[('anchor',oid) for oid in (4002,10461,30237) if oid in a.index];used={i for _,i in selected}
    for name,mask in (('rescue',a.correct&~r.correct),('harm',~a.correct&r.correct),
                       ('correct',a.correct&r.correct),('wrong',~a.correct&~r.correct)):
        sub=a[mask&~a.index.isin(used)].copy();sub['dw']=abs(sub.weight-1);sub=sub.sort_values('dw')
        if len(sub):oid=int(sub.index[len(sub)//2]);selected.append((name,oid));used.add(oid)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12);records=[]
    for kind,oid in selected:
        i=index[oid];row=f.iloc[i];canvas=Image.new('RGB',(1130,470),'white');d=ImageDraw.Draw(canvas)
        d.text((10,8),f'{kind} | obs {oid}, {row.class_name}, {row.port} | run 0, fold {int(a.loc[oid,"fold"])}',fill='black',font=font)
        for poln,pol in enumerate(('vv','vh')):
            pix,masks,_,_,_=exp.old.control.prepare_view(row,views[i]['images'][pol],pol)
            x=pix[0];rr=masks[2].astype(bool);mu=float(x[rr].mean());z=.5+.5*(x-mu)*rr
            y0=58+195*poln
            canvas.paste(Image.fromarray(np.rint(x*255).astype(np.uint8)).convert('RGB'),(10,y0))
            canvas.paste(Image.fromarray(np.rint(z*255).astype(np.uint8)).convert('RGB'),(185,y0))
            d.text((10,y0-19),f'{pol.upper()} original; native pixels',fill='black',font=small)
            d.text((185,y0-19),'Frozen neighborhood input',fill='black',font=small)
            theta=float(np.degrees(np.arctan2(g[i,poln,1],g[i,poln,0]))/2)
            text=f'geometry valid={bool(v[i,poln])}; relative axis={theta:.1f}, incidence={g[i,poln,2]:.1f}'
            d.text((10,y0+163),text,fill='black',font=small)
        for j,arm in enumerate(('baseline','reference','constant','content','geometry','joint','sham',
                                 'joint_zero_geometry','joint_shuffled_geometry','joint_source_mean_weight')):
            z=q.loc[(arm,oid)];text=f'{arm:26} pred {int(z.prediction)}; margin {z.margin:+.4f}; weight {z.weight:.4f}'
            d.text((380,55+30*j),text,fill='black',font=small)
            records.append(dict(case=kind,obs_id=oid,seed=0,fold=int(z.fold),arm=arm,class_id=int(row.class_id),
                prediction=int(z.prediction),margin=z.margin,weight=z.weight,step=int(z.step),
                image_vv=row.image_vv,image_vh=row.image_vh))
        d.text((380,375),'No pixel/scale changes. Only the weight on frozen R scores changes.',fill='black',font=small)
        d.text((380,395),'Class IDs: 0 bulk, 1 fishing, 2 cargo, 3 chemical/product,',fill='black',font=small)
        d.text((380,413),'4 container, 5 crude tanker, 6 tug, 7 offshore.',fill='black',font=small)
        d.text((380,433),'Displayed geometry availability is raw metadata; gating also requires usable R.',fill='black',font=small)
        canvas.save(ROOT/f'case_{kind}_{oid}.png')
    pd.DataFrame(records).to_csv(ROOT/'case_index.csv',index=False,lineterminator='\n')


if __name__=='__main__':
    p=pd.read_csv(ROOT/'predictions.csv',dtype={'mmsi':str});summarize(p);paired(p);verify_selection();cases(p)
