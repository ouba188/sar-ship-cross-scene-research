"""Paired full-model effects, branch interventions, and native query/reference panels."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_native as tr

def margin(s,y):
    return s[np.arange(len(y)),y]-np.where(np.arange(8)[None]==y[:,None],-np.inf,s).max(1)

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=tr.Data();f=data.table;y=f.class_id.to_numpy(int);te=data.te
    arms=('old_full','score_only',*tr.ARMS)
    scores={(seed,arm):np.load(HERE/'results'/f'{arm}_{seed}.npz')['score'] for seed in tr.SEEDS for arm in arms}
    pd.concat((pd.read_csv(HERE/'metrics.csv'),pd.read_csv(HERE/'score_control_metrics.csv'))).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].agg(['mean','std']).to_csv(HERE/'summary.csv')
    effects=[];detail=[];candidates=[]
    pairs=[(a,'old_full') for a in ('score_only',*tr.ARMS)]+[('moment','mean'),('grid_moment','moment'),('axis_moment','moment'),('axis_moment','grid_moment'),('axis_moment','score_only')]
    for seed in tr.SEEDS:
        for arm in arms:
            s=scores[seed,arm]
            row=f.iloc[te][['obs_id','class_id','class_name','mmsi','pass_group','port']].copy()
            row['seed']=seed;row['arm']=arm;row['prediction']=s[te].argmax(1);row['correct']=row.prediction.eq(row.class_id);row['margin']=margin(s[te],y[te]);detail.append(row)
        for arm,comp in pairs:
            a=scores[seed,arm][te].argmax(1)==y[te];b=scores[seed,comp][te].argmax(1)==y[te];d=a.astype(float)-b
            effects.append(dict(seed=seed,arm=arm,comparator=comp,rescued=int((a&~b).sum()),harmed=int((b&~a).sum()),net=int(d.sum()),OA_pp=100*d.mean(),BA_pp=100*np.mean([d[y[te]==c].mean() for c in range(8)])))
        for arm in ('axis_moment','moment','grid_moment'):
            a=scores[seed,arm][te];b=scores[seed,'old_full'][te]
            changes=margin(a,y[te])-margin(b,y[te])
            for k,j in enumerate(te):
                ac=a[k].argmax()==y[j];bc=b[k].argmax()==y[j]
                case='CC' if ac and bc else 'WW' if not ac and not bc else 'rescue' if ac else 'harm'
                candidates.append(dict(seed=seed,arm=arm,index=int(j),case=case,change=float(changes[k])))
    pd.DataFrame(effects).to_csv(HERE/'paired.csv',index=False)
    detail=pd.concat(detail);detail.to_csv(HERE/'predictions.csv.gz',index=False,compression='gzip')
    detail.groupby(['seed','arm','class_id','class_name']).agg(n=('correct','size'),accuracy=('correct','mean'),margin=('margin','mean')).to_csv(HERE/'classes.csv')
    detail.groupby(['seed','arm','pass_group']).agg(n=('correct','size'),accuracy=('correct','mean')).to_csv(HERE/'passes.csv')
    groups=f.iloc[te].pass_group.to_numpy();unique=np.unique(groups);rng=np.random.default_rng(1024)
    draws=rng.integers(len(unique),size=(2000,len(unique)))
    ct0=np.array([[((groups==g)&(y[te]==c)).sum() for c in range(8)] for g in unique]);ct=ct0[draws].sum(1);valid=(ct>0).all(1)
    intervals=[]
    for arm,comp in pairs:
        d=np.mean([(scores[s,arm][te].argmax(1)==y[te]).astype(float)-(scores[s,comp][te].argmax(1)==y[te]) for s in tr.SEEDS],axis=0)
        total=np.array([[d[(groups==g)&(y[te]==c)].sum() for c in range(8)] for g in unique])[draws].sum(1)
        oa=100*total.sum(1)/ct.sum(1);ba=100*(total[valid]/ct[valid]).mean(1)
        intervals.append(dict(arm=arm,comparator=comp,OA_low=np.quantile(oa,.025),OA_high=np.quantile(oa,.975),BA_low=np.quantile(ba,.025),BA_high=np.quantile(ba,.975),valid_BA_draws=int(valid.sum())))
    pd.DataFrame(intervals).to_csv(HERE/'uncertainty.csv',index=False)
    cf=pd.DataFrame(candidates);chosen=[]
    for arm in ('axis_moment','moment'):
        for status in ('rescue','harm','CC','WW'):
            pool=cf[cf.arm.eq(arm)&cf.case.eq(status)]
            if len(pool):
                pool=pool[pool.seed.eq(pool.seed.min())]
                chosen.append(pool.loc[pool.change.abs().idxmax()].to_dict())
    geo=tr.load('native_analysis_geo',tr.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    uptable=pd.read_csv(tr.UP/'table.csv');refs=np.load(tr.LOCAL/'scores_layer1.npz')['refs'];validpol=np.load(tr.UP/'summaries.npz')['valid']
    outdir=HERE/'analysis';outdir.mkdir(exist_ok=True)
    reader=dict(np.load(HERE/'reader.npz'));records=[]
    for case in chosen:
        seed,j,arm=int(case['seed']),int(case['index']),case['arm'];i=int(f.iloc[j].up_index);c=int(y[j]);obs=int(f.iloc[j].obs_id)
        before=scores[seed,'old_full'][j];after=scores[seed,arm][j]
        rival=int(after.argmax()) if case['case']=='harm' else int(np.where(np.arange(8)==c,-np.inf,before).argmax())
        rr=refs[i,[c,rival]].ravel()
        fig,axes=plt.subplots(2,5,figsize=(16,7));provenance=[]
        for p in range(2):
            for col,ri in enumerate([i,*rr]):
                if validpol[ri,p]:
                    pix,fp,g,_=geo.native(uptable,snapshots,int(ri),p)
                    axes[p,col].imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    for region,color in zip(geo.soft_regions(g)*fp,('#20a780','#d77329','#3287d0')):
                        if region.max()>.5:axes[p,col].contour(region,levels=[.5],colors=[color],linewidths=.55)
                else:axes[p,col].text(.5,.5,'MISSING',ha='center')
                label='QUERY' if col==0 else 'TRUE' if col<3 else 'RIVAL'
                axes[p,col].set_title(f'{("VV","VH")[p]} {label} {int(uptable.iloc[ri].obs_id)}\n{uptable.iloc[ri].class_name}',fontsize=8)
                axes[p,col].set(xticks=[],yticks=[])
                provenance.append(dict(up_index=int(ri),polarization=('VV','VH')[p],path=str(uptable.iloc[ri]['image_'+('vv','vh')[p]])))
        fig.suptitle(f'{arm} {case["case"]} seed{seed} true{c}: {before.argmax()} -> {after.argmax()}\nMargin {margin(before[None],y[j:j+1])[0]:+.5f} -> {margin(after[None],y[j:j+1])[0]:+.5f}')
        fig.text(.02,.02,'Native160 crop/pad; no image resize. Green body, orange ends, blue sides. M3 references UNCHANGED.',fontsize=9)
        fig.subplots_adjust(top=.84,bottom=.09,wspace=.12,hspace=.15)
        path=outdir/f'{arm}_{case["case"]}_{obs}_{seed}.png';fig.savefig(path,dpi=120);plt.close(fig)
        ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
        model=tr.Model(arm,reader).cuda();model.load_state_dict(ck['state'])
        replay=tr.predict(model,data,np.array([j]),data.baselines[seed])[0]
        np.testing.assert_allclose(replay,after,atol=3e-5)
        interventions={}
        if model.conv is not None:
            weight=model.conv.output if arm=='axis_moment' else model.conv.out
            saved=weight.detach().clone()
            with torch.no_grad():weight.zero_()
            no_conv=tr.predict(model,data,np.array([j]),data.baselines[seed])[0]
            with torch.no_grad():weight.copy_(saved)
            interventions['conv_zero']=dict(pred=int(no_conv.argmax()),fixed_margin=float(no_conv[c]-no_conv[rival]),margin_contribution=float((after[c]-after[rival])-(no_conv[c]-no_conv[rival])))
        # Replace evidence by its source-normalized mean, retaining availability/context.
        for p in range(2):
            ss=[]
            with torch.no_grad():
                for k in range(4):
                    b=data.batch(np.array([j]),k);base=torch.tensor(before[None],device='cuda')
                    d=model.describe(b);raw=model.raw(d,base,b['flags']);z=(raw-model.raw_mu)/model.raw_scale
                    cols=list(range(p*27,(p+1)*27))+list(range(54+p*27,54+(p+1)*27))
                    z[:,:,cols]=0
                    delta=model.head(z).squeeze(-1);delta-=delta.mean(1,keepdim=True)
                    ss.append((base+delta)[0].cpu().numpy())
            s=np.mean(ss,axis=0)
            interventions[('VV','VH')[p]+'_source_mean']=dict(pred=int(s.argmax()),fixed_margin=float(s[c]-s[rival]),margin_contribution=float((after[c]-after[rival])-(s[c]-s[rival])))
        records.append(dict(**case,obs_id=obs,true_class=c,rival=rival,old_pred=int(before.argmax()),new_pred=int(after.argmax()),old_margin=float(margin(before[None],y[j:j+1])[0]),new_margin=float(margin(after[None],y[j:j+1])[0]),selected_step=ck['receipt']['selected_step'],interventions=interventions,reference_obs=uptable.iloc[rr].obs_id.astype(int).tolist(),provenance=provenance,path=str(path)))
        del model
    (outdir/'cases.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (HERE/'analysis_receipt.json').write_text(json.dumps(dict(queries=len(te),ships=int(f.iloc[te].mmsi.nunique()),passes=len(unique),cases=len(records),bootstrap='2000 acquisition clusters; seeds averaged; BA conditioned on all classes',interventions='model diagnostic, not physical interventions; same head with conv zero or polarized evidence replaced by source mean'),indent=2))
    print(pd.DataFrame(effects).to_string(index=False),flush=True)
    print(pd.DataFrame(records)[['arm','case','seed','obs_id','interventions']].to_string(index=False),flush=True)

if __name__=='__main__':main()
