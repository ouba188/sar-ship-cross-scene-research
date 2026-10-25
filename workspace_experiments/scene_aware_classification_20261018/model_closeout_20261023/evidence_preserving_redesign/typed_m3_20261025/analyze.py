"""Paired outcomes, matched-reference diagnostics and native VV/VH case panels."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import train as run
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp

PAIRS=[(a,'old_full') for a in run.ARMS]+[('typed_both',c) for c in ('typed_fixed','content_both','score_only')]+[('typed_fixed','score_only')]

def margin(s,y):
    return s[np.arange(len(y)),y]-np.where(np.arange(8)[None]==y[:,None],-np.inf,s).max(1)

def effect(a,b,y):
    ap,bp=a.argmax(1)==y,b.argmax(1)==y
    d=ap.astype(float)-bp
    return dict(n=len(y),rescued=int((ap&~bp).sum()),harmed=int((~ap&bp).sum()),net=int(d.sum()),
                OA_pp=100*d.mean(),BA_pp=100*np.mean([d[y==c].mean() for c in np.unique(y)]))

def statistics(data,scores):
    f=data.table.iloc[data.te].reset_index(drop=True);y=f.class_id.to_numpy(int)
    pairs=[];strata=[];pred=[]
    for seed in run.SEEDS:
        for arm in ('old_full',*run.ARMS):
            s=scores[seed,arm][data.te];p=f[['obs_id','up_index','class_id','class_name','mmsi','pass_group','port']].copy()
            p['seed']=seed;p['arm']=arm;p['prediction']=s.argmax(1);p['margin']=margin(s,y)
            for c in range(8):p[f'score_{c}']=s[:,c]
            pred.append(p)
        for arm,comp in PAIRS:
            a,b=scores[seed,arm][data.te],scores[seed,comp][data.te]
            pairs.append(dict(seed=seed,arm=arm,comparator=comp,**effect(a,b,y)))
            for kind,col in (('class','class_id'),('acquisition','pass_group')):
                for group,ix in f.groupby(col).indices.items():
                    strata.append(dict(seed=seed,arm=arm,comparator=comp,kind=kind,group=group,ships=f.iloc[ix].mmsi.nunique(),**effect(a[ix],b[ix],y[ix])))
    pd.DataFrame(pairs).to_csv(HERE/'paired.csv',index=False)
    pd.DataFrame(strata).to_csv(HERE/'strata.csv',index=False)
    pd.concat(pred).to_csv(HERE/'predictions.csv.gz',index=False,compression='gzip')
    pd.read_csv(HERE/'metrics.csv').query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].agg(['mean','std']).to_csv(HERE/'summary.csv')
    clusters,names=pd.factorize(f.pass_group,sort=True);counts=np.zeros((len(names),8));np.add.at(counts,(clusters,y),1)
    draws=np.random.default_rng(1025).integers(len(names),size=(2000,len(names)));samplecounts=counts[draws].sum(1)
    has8=(samplecounts>0).all(1);intervals=[]
    for arm,comp in PAIRS:
        d=np.mean([(scores[s,arm][data.te].argmax(1)==y).astype(float)-(scores[s,comp][data.te].argmax(1)==y) for s in run.SEEDS],0)
        total=np.zeros((len(names),8));np.add.at(total,(clusters,y),d);sample=total[draws].sum(1)
        for metric,dist in (('OA_pp',100*sample.sum(1)/samplecounts.sum(1)),('BA_pp',100*(sample[has8]/samplecounts[has8]).mean(1))):
            lo,hi=np.quantile(dist,[.025,.975]);intervals.append(dict(arm=arm,comparator=comp,metric=metric,low=lo,high=hi,draws=len(dist),clusters=len(names)))
    pd.DataFrame(intervals).to_csv(HERE/'uncertainty.csv',index=False)
    return dict(n=len(f),ships=f.mmsi.nunique(),passes=len(names),ports=f.port.nunique(),class_counts=np.bincount(y,minlength=8))

def model_for(seed,arm):
    model=run.Model(arm).cuda();ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state']);model.eval();return model,ck['receipt']

@torch.inference_mode()
def switches(data,scores):
    rows=[];cache={};y=data.table.iloc[data.te].class_id.to_numpy(int)
    for seed in run.SEEDS:
        model,receipt=model_for(seed,'typed_both');base=torch.tensor(data.baselines[seed],device='cuda')
        replay=run.predict(model,data,data.te,base)
        error=float(abs(replay-scores[seed,'typed_both'][data.te]).max());assert error<3e-5,error
        for mode in ('initial','score_only','transport_only'):
            value=run.predict(model,data,data.te,base,mode);cache[seed,mode]=value
            rows.append(dict(seed=seed,intervention=mode,selected_step=receipt['selected_step'],replay=error,
                             changed=int((value.argmax(1)!=replay.argmax(1)).sum()),max_score_change=float(abs(replay-value).max()),**effect(replay,value,y)))
        del model
    pd.DataFrame(rows).to_csv(HERE/'fixed_head_switch.csv',index=False)
    np.savez_compressed(HERE/'results/switches.npz',**{f'{s}_{m}':v for (s,m),v in cache.items()})
    return cache

def cases(data,scores,switch):
    ids=data.te;y=data.table.iloc[ids].class_id.to_numpy(int);out=[];used=set()
    for name in ('rescue','harm','CC','WW'):
        candidates=[]
        for seed in run.SEEDS:
            a=scores[seed,'typed_both'][ids].argmax(1)==y;b=scores[seed,'old_full'][ids].argmax(1)==y
            take={'rescue':a&~b,'harm':~a&b,'CC':a&b,'WW':~a&~b}[name]
            for k in np.flatnonzero(take):
                obs=int(data.table.iloc[ids[k]].obs_id)
                if obs not in used:candidates.append((seed,obs,int(ids[k]),int(k)))
        candidates.sort()
        if candidates:
            seed,obs,j,k=candidates[0] if name in ('rescue','harm') else candidates[(len(candidates)-1)//2]
            out.append(dict(seed=seed,obs_id=obs,index=j,eval_index=k,case=name,comparison='old_full'));used.add(obs)
    # Separately inspect an actual metric-caused flip when it differs from system flips.
    for name in ('metric_rescue','metric_harm'):
        candidates=[]
        for seed in run.SEEDS:
            a=scores[seed,'typed_both'][ids].argmax(1)==y;b=switch[seed,'initial'].argmax(1)==y
            take=a&~b if name=='metric_rescue' else ~a&b
            for k in np.flatnonzero(take):
                obs=int(data.table.iloc[ids[k]].obs_id)
                if obs not in used:candidates.append((seed,obs,int(ids[k]),int(k)))
        if candidates:
            seed,obs,j,k=sorted(candidates)[0];used.add(obs)
            out.append(dict(seed=seed,obs_id=obs,index=j,eval_index=k,case=name,comparison='fixed_initial_metric'))
    return out

def panel(case,data,before,after,geo,up,valid,snapshots,refs):
    i=int(data.up[case['index']]);true=int(up.iloc[i].class_id)
    anchor=after if 'harm' in case['case'] else before
    rival=int(np.where(np.arange(8)==true,-np.inf,anchor).argmax())
    views=[(i,'QUERY')]+[(int(refs[i,c,k]),label+str(k+1)) for c,label in ((true,'TRUE '),(rival,'RIVAL ')) for k in range(2)]
    fig,axs=plt.subplots(2,5,figsize=(16,7));provenance=[]
    for p,pol in enumerate(('vv','vh')):
        for col,(ri,label) in enumerate(views):
            row=up.iloc[ri];ax=axs[p,col];obs=int(row.obs_id);entry=dict(up_index=ri,obs_id=obs,polarization=pol,path=str(row['image_'+pol]))
            if valid[ri,p] and obs in snapshots and pol in snapshots[obs]['images']:
                pix,fp,g,place=geo.native(up,snapshots,ri,p)
                ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                for region,color in zip(geo.soft_regions(g)*fp,('#20a780','#d77329','#3287d0')):
                    if region.min()<.5<region.max():ax.contour(region,levels=[.5],colors=[color],linewidths=.55)
                entry.update(status='rendered',geometry=g,placement=place)
            else:ax.text(.5,.5,'MISSING VIEW',ha='center',transform=ax.transAxes);entry['status']='missing'
            ax.set_title(f'{pol.upper()} {label} {obs}\n{row.class_name}',fontsize=8);ax.set(xticks=[],yticks=[]);provenance.append(entry)
    fig.suptitle(f"typed_both vs {case['comparison']} | {case['case']} | seed{case['seed']} | obs{case['obs_id']}\ntrue={true},rival={rival}; pred {before.argmax()} -> {after.argmax()}; fixed margin {before[true]-before[rival]:+.5f} -> {after[true]-after[rival]:+.5f}",fontsize=11)
    fig.text(.02,.02,'Native160 crop/pad; no ship resize. Green body / orange ends / blue sides. Actual frozen references.',fontsize=9)
    fig.subplots_adjust(top=.84,bottom=.09,wspace=.12,hspace=.18);path=HERE/'analysis'/f"{case['case']}_{case['obs_id']}_{case['seed']}.png"
    fig.savefig(path,dpi=120);plt.close(fig)
    return path,provenance,true,rival

@torch.inference_mode()
def case_diagnostics(data,case,true,rival):
    seed=case['seed'];j=case['index'];ids=[j]
    model,_=model_for(seed,'typed_both');base=torch.tensor(data.baselines[seed][ids],device='cuda')
    local,diag=data.evidence(ids,model,diagnostic=True);initial=data.evidence(ids,model,mode='initial')
    cover=data.cover[ids];score=model.read(local,base,cover)[0];fm=score[true]-score[rival]
    rows=[]
    for t in range(18):
        changed=local.clone();changed[:,:,t]=initial[:,:,t]
        s=model.read(changed,base,cover)[0]
        rows.append(dict(type=t,polarization=('VV','VH')[t//9],region=('body','ends','sides')[(t%9)//3],band=('fine','middle','low')[t%3],
                         learned_metric_margin_effect=float(fm-(s[true]-s[rival])),
                         true_evidence=local[0,true,t].cpu().numpy(),rival_evidence=local[0,rival,t].cpu().numpy(),
                         true_initial=initial[0,true,t].cpu().numpy(),rival_initial=initial[0,rival,t].cpu().numpy(),
                         metric_diagnostics_true=diag[0,true,t].cpu().numpy(),metric_diagnostics_rival=diag[0,rival,t].cpu().numpy()))
    participants=torch.unique(torch.cat((torch.tensor(ids,device='cuda'),data.refs[j].flatten())))
    original=data.x[participants].clone();blocks=[]
    for p in range(2):
        for label,sl in (('class_summary',slice(32,56)),('native_band_and_covariance',slice(56,91))):
            data.x[participants,p*9:(p+1)*9,:,sl]=0
            alt=model.read(data.evidence(ids,model),base,cover)[0]
            blocks.append(dict(polarization=('VV','VH')[p],block=label,fixed_margin_effect=float(fm-(alt[true]-alt[rival])),prediction=int(alt.argmax())))
            data.x[participants]=original
    return dict(types=rows,source_mean_block_replacements=blocks,local=local.cpu().numpy(),initial=initial.cpu().numpy(),
                interpretation='Fixed-head conditional replacements are not additive, trained ablations or physical causation. Query and actually used references both changed.')

def main():
    assert (HERE/'complete.json').is_file();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=run.Data();run.prepare_fixed(data);scores={}
    for seed in run.SEEDS:
        for arm in ('old_full',*run.ARMS):
            with np.load(HERE/'results'/f'{arm}_{seed}.npz') as z:
                np.testing.assert_array_equal(z['up_index'],data.up[:data.n]);scores[seed,arm]=z['score'].copy()
    receipt=statistics(data,scores);switch=switches(data,scores);chosen=cases(data,scores,switch)
    (HERE/'analysis').mkdir(exist_ok=True)
    geo=run.load('typed_case_geo',run.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up=pd.read_csv(run.BASE/'regional_utility_pilot/upstream/table.csv')
    valid=np.load(run.BASE/'regional_utility_pilot/upstream/summaries.npz')['valid']
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs=np.load(run.LOCAL/'scores_layer1.npz')['refs'];records=[]
    for case in chosen:
        seed,j,k=case['seed'],case['index'],case['eval_index'];after=scores[seed,'typed_both'][j]
        before=scores[seed,'old_full'][j] if case['comparison']=='old_full' else switch[seed,'initial'][k]
        path,prov,true,rival=panel(case,data,before,after,geo,up,valid,snapshots,refs)
        records.append(dict(**case,true=true,rival=rival,before=before,after=after,
                            comparisons={a:scores[seed,a][j] for a in ('old_full',*run.ARMS)},
                            diagnostics=case_diagnostics(data,case,true,rival),figure=str(path),provenance=prov))
        print('CASE',case,flush=True)
    run.write_json(HERE/'analysis/cases.json',records)
    receipt.update(cases=len(records),case_selection='deterministic sorted seed/obs for flips, median for CC/WW; optional distinct fixed-metric flips',
                   uncertainty='2000 acquisition-cluster bootstrap draws of seed-averaged effects; BA only all8classes draws')
    run.write_json(HERE/'analysis_receipt.json',receipt)
    print(pd.read_csv(HERE/'fixed_head_switch.csv').to_string(index=False),flush=True)

if __name__=='__main__':main()
