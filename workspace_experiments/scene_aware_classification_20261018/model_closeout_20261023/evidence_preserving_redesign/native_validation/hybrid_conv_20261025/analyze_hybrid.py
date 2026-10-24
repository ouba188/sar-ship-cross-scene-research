"""Paired outcomes, fixed-head interventions, and original-chip case panels."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import train_hybrid as run
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
tr=run.tr
ARMS=('old_full',*run.ARMS)
PAIRS=[(a,'old_full') for a in run.ARMS]+[(a,'moment') for a in run.ARMS if a!='moment']+[
    ('axis_role','grid_role'),('axis_role','axis_difference'),('axis_role','axis_early')]


def write_json(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,indent=2,default=convert,allow_nan=False),encoding='utf-8')


def margin(score,y):
    return score[np.arange(len(y)),y]-np.where(np.arange(8)[None]==y[:,None],-np.inf,score).max(1)


def values(score,y):
    lp=score.astype(float)-logsumexp(score.astype(float),axis=1,keepdims=True)
    return np.column_stack((score.argmax(1)==y,-lp[np.arange(len(y)),y],((np.exp(lp)-np.eye(8)[y])**2).sum(1)))


def effect(a,b,y):
    av,bv=values(a,y),values(b,y);d=av-bv
    return dict(n=len(y),rescued=int(((av[:,0]==1)&(bv[:,0]==0)).sum()),
        harmed=int(((av[:,0]==0)&(bv[:,0]==1)).sum()),net=int(d[:,0].sum()),
        OA_pp=100*d[:,0].mean(),BA_pp=100*np.mean([d[y==c,0].mean() for c in np.unique(y)]),
        NLL_delta=d[:,1].mean(),Brier_delta=d[:,2].mean())


def statistics(data,scores):
    f=data.table.iloc[data.te].reset_index(drop=True);y=f.class_id.to_numpy(int)
    metrics=[];pairs=[];strata=[];predictions=[]
    for seed in tr.SEEDS:
        for arm in ARMS:
            s=scores[seed,arm][data.te]
            metrics.append(dict(seed=seed,arm=arm,**tr.metrics(y,s)))
            p=f[['obs_id','up_index','class_id','class_name','mmsi','pass_group','port']].copy()
            p['seed']=seed;p['arm']=arm;p['prediction']=s.argmax(1);p['margin']=margin(s,y)
            for c in range(8):p[f'score_{c}']=s[:,c]
            predictions.append(p)
        for arm,comp in PAIRS:
            a,b=scores[seed,arm][data.te],scores[seed,comp][data.te]
            pairs.append(dict(seed=seed,arm=arm,comparator=comp,**effect(a,b,y)))
            for kind,col in [('class','class_id'),('acquisition','pass_group')]:
                for group,ids in f.groupby(col).indices.items():
                    strata.append(dict(seed=seed,arm=arm,comparator=comp,kind=kind,group=group,
                        ships=f.iloc[ids].mmsi.nunique(),**effect(a[ids],b[ids],y[ids])))
    for name,rows in [('evaluation_metrics',metrics),('paired',pairs),('strata',strata)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    pd.DataFrame(metrics).groupby('arm')[['OA','BA','NLL','Brier']].agg(['mean','std']).to_csv(HERE/'summary.csv')
    pd.DataFrame(pairs).groupby(['arm','comparator'])[['rescued','harmed','net','OA_pp','BA_pp']].agg(['mean','std']).to_csv(HERE/'paired_summary.csv')
    pd.concat(predictions).to_csv(HERE/'predictions.csv.gz',index=False,compression='gzip')
    clusters,names=pd.factorize(f.pass_group,sort=True)
    draws=np.random.default_rng(1025).integers(len(names),size=(2000,len(names)))
    counts=np.zeros((len(names),8));np.add.at(counts,(clusters,y),1)
    sampled=counts[draws].sum(1);has8=(sampled>0).all(1);intervals=[]
    for arm,comp in PAIRS:
        d=np.mean([values(scores[s,arm][data.te],y)-values(scores[s,comp][data.te],y) for s in tr.SEEDS],axis=0)
        total=np.zeros((len(names),8,3));np.add.at(total,(clusters,y),d)
        sample=total[draws].sum(1)
        for name,dist in [('OA_pp',100*sample[:,:,0].sum(1)/sampled.sum(1)),
                          ('BA_pp',100*(sample[has8,:,0]/sampled[has8]).mean(1))]:
            low,high=np.quantile(dist,[.025,.975])
            intervals.append(dict(arm=arm,comparator=comp,metric=name,low=low,high=high,valid_draws=len(dist),clusters=len(names)))
    pd.DataFrame(intervals).to_csv(HERE/'uncertainty.csv',index=False)
    return dict(n=len(f),ships=f.mmsi.nunique(),passes=len(names),ports=f.port.nunique(),class_counts=np.bincount(y,minlength=8))


def load_model(arm,seed,reader):
    ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
    model=run.Model(arm,reader).cuda();model.load_state_dict(ck['state']);model.eval()
    return model,ck['receipt']


@torch.inference_mode()
def switches(data,reader,scores):
    rows=[];interventions={};y=data.table.iloc[data.te].class_id.to_numpy(int)
    for seed in tr.SEEDS:
        for arm in run.ARMS[1:]:
            model,receipt=load_model(arm,seed,reader)
            replay=tr.predict(model,data,data.te,data.baselines[seed])
            error=float(np.abs(replay-scores[seed,arm][data.te]).max())
            assert error < 3e-5
            saved=model.conv.output.clone();model.conv.output.zero_()
            off=tr.predict(model,data,data.te,data.baselines[seed])
            interventions[seed,arm,'off']=off
            rows.append(dict(seed=seed,arm=arm,intervention='conv_off',selected_step=receipt['selected_step'],
                replay_error=error,max_score_change=float(np.abs(replay-off).max()),
                changed=int((replay.argmax(1)!=off.argmax(1)).sum()),**effect(replay,off,y)))
            model.conv.output.copy_(saved)
            if arm=='axis_role':
                model.conv.early_mean=True
                early=tr.predict(model,data,data.te,data.baselines[seed])
                interventions[seed,arm,'early']=early
                rows.append(dict(seed=seed,arm=arm,intervention='early_mean',selected_step=receipt['selected_step'],
                    replay_error=error,max_score_change=float(np.abs(replay-early).max()),
                    changed=int((replay.argmax(1)!=early.argmax(1)).sum()),**effect(replay,early,y)))
            del model
    pd.DataFrame(rows).to_csv(HERE/'fixed_head_switch.csv',index=False)
    np.savez_compressed(HERE/'results/switches.npz',**{f'{s}_{a}_{v}':x for (s,a,v),x in interventions.items()})
    return interventions


def choose_cases(data,scores,interventions):
    ids=data.te;y=data.table.iloc[ids].class_id.to_numpy(int);chosen=[];used=set()
    for status in ('rescue','harm','CC','WW'):
        candidates=[]
        for seed in tr.SEEDS:
            on=scores[seed,'axis_role'][ids];off=interventions[seed,'axis_role','off']
            a,b=on.argmax(1)==y,off.argmax(1)==y
            mask={'rescue':a&~b,'harm':~a&b,'CC':a&b,'WW':~a&~b}[status]
            for k in np.flatnonzero(mask):
                j=int(ids[k]);obs=int(data.table.iloc[j].obs_id)
                if obs not in used:candidates.append((seed,obs,j,k,'conv_off'))
        # If the operator has no direct flip, inspect complete trained-system flips separately.
        if not candidates and status in ('rescue','harm'):
            for seed in tr.SEEDS:
                a=scores[seed,'axis_role'][ids].argmax(1)==y
                b=scores[seed,'moment'][ids].argmax(1)==y
                mask=a&~b if status=='rescue' else ~a&b
                for k in np.flatnonzero(mask):
                    j=int(ids[k]);obs=int(data.table.iloc[j].obs_id)
                    if obs not in used:candidates.append((seed,obs,j,k,'trained_moment'))
        candidates.sort()
        if candidates:
            item=candidates[0] if status in ('rescue','harm') else candidates[(len(candidates)-1)//2]
            seed,obs,j,k,comp=item;used.add(obs)
            chosen.append(dict(seed=seed,obs_id=obs,index=j,eval_index=k,case=status,comparison=comp,
                               candidate_exposures=len(candidates),selection='sorted seed,obs first for flips; lower median for CC/WW'))
    return chosen


def render(case,data,before,after,geo,up,snapshots,valid,refs):
    i=int(data.table.iloc[case['index']].up_index);true=int(up.iloc[i].class_id)
    anchor=after if case['case']=='harm' else before
    rival=int(np.where(np.arange(8)==true,-np.inf,anchor).argmax())
    views=[(i,'QUERY',true)]+[(int(refs[i,c,s]),label+str(s+1),c) for c,label in ((true,'TRUE '),(rival,'RIVAL ')) for s in range(2)]
    fig,axes=plt.subplots(2,5,figsize=(16,7));provenance=[]
    for p,pol in enumerate(('VV','VH')):
        for col,(ri,label,cls) in enumerate(views):
            ax=axes[p,col];entry=dict(up_index=ri,polarization=pol,label=label)
            if not 0<=ri<len(up):
                ax.text(.5,.5,'MISSING REFERENCE',ha='center');title=label;entry['status']='missing_reference'
            else:
                row=up.iloc[ri];obs=int(row.obs_id);path=str(row['image_'+pol.lower()])
                if col>0:assert int(row.class_id)==cls
                title=f'{pol} {label} {obs}\n{row.class_name}'
                entry.update(obs_id=obs,path=path,class_id=int(row.class_id))
                if valid[ri,p] and obs in snapshots and pol.lower() in snapshots[obs]['images'] and Path(path).is_file():
                    pix,foot,g,placement=geo.native(up,snapshots,ri,p)
                    ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    for region,color in zip(geo.soft_regions(g)*foot,('#20a780','#d77329','#3287d0')):
                        if region.min()<.5<region.max():ax.contour(region,levels=[.5],colors=[color],linewidths=.55)
                    entry.update(status='rendered',geometry=g,placement=placement)
                else:
                    ax.text(.5,.5,'MISSING VIEW',ha='center',transform=ax.transAxes);entry['status']='missing_view'
            ax.set_title(title,fontsize=8);ax.set(xticks=[],yticks=[]);provenance.append(entry)
    fig.suptitle(f"axis_role vs {case['comparison']} | {case['case']} | seed{case['seed']} | obs{case['obs_id']}\n"
        f"true={true}, rival={rival}; prediction {before.argmax()} -> {after.argmax()}; fixed margin {before[true]-before[rival]:+.5f} -> {after[true]-after[rival]:+.5f}",fontsize=11)
    fig.text(.02,.02,'Native160 crop/pad, no resize. Green body / orange ends / blue sides. Actual frozen M3 references.',fontsize=9)
    fig.subplots_adjust(top=.84,bottom=.09,wspace=.12,hspace=.18)
    path=HERE/'analysis'/f"{case['case']}_{case['obs_id']}_{case['seed']}.png"
    fig.savefig(path,dpi=120);plt.close(fig)
    return str(path),provenance,true,rival


@torch.inference_mode()
def case_diagnostics(case,data,reader,rival):
    model,_=load_model('axis_role',case['seed'],reader)
    j=case['index'];true=int(data.table.iloc[j].class_id);base=torch.tensor(data.baselines[case['seed']][[j]],device='cuda')
    saved=model.conv.output.clone();sensitivity=[];energy=[]
    for rotation in range(4):
        b=data.batch([j],rotation);desc=model.describe(b)
        model.conv.output.zero_();off=model.describe(b);model.conv.output.copy_(saved)
        def score(d):
            raw=model.raw(d,base,b['flags']);delta=model.head((raw-model.raw_mu)/model.raw_scale).squeeze(-1)
            return (base+delta-delta.mean(1,keepdim=True))[0]
        full=score(desc)
        for p,pol in enumerate(('VV','VH')):
            for r,name in enumerate(('body','ends','sides')):
                altered=desc.clone();altered[:,p,:,r]=off[:,p,:,r];ss=score(altered)
                sensitivity.append(dict(rotation=rotation,polarization=pol,region=name,
                    fixed_margin_effect=float((full[true]-full[rival])-(ss[true]-ss[rival])),
                    descriptor_rms=float((desc[:,p,:,r]-off[:,p,:,r]).square().mean().sqrt())))
        energy.append(float((desc-off).square().mean().sqrt()))
    del model
    return dict(regional_conditional_sensitivity=sensitivity,descriptor_rms_C4=energy,
                interpretation='Replace one region descriptor by fixed-head conv-off descriptor; nonlinear head, effects not additive; not physical causation.')


def main():
    assert (HERE/'complete.json').is_file()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=tr.Data();reader=dict(np.load(tr.HERE/'reader.npz'));scores={}
    for seed in tr.SEEDS:
        for arm in ARMS:
            with np.load(HERE/'results'/f'{arm}_{seed}.npz') as z:
                np.testing.assert_array_equal(z['up_index'],data.up);scores[seed,arm]=z['score'].copy()
    receipt=statistics(data,scores)
    interventions=switches(data,reader,scores)
    chosen=choose_cases(data,scores,interventions);(HERE/'analysis').mkdir(exist_ok=True)
    geo=tr.load('hybrid_case_geo',tr.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    up=pd.read_csv(tr.UP/'table.csv');valid=np.load(tr.UP/'summaries.npz')['valid'];refs=np.load(tr.LOCAL/'scores_layer1.npz')['refs']
    records=[]
    for case in chosen:
        seed,j,k=case['seed'],case['index'],case['eval_index'];after=scores[seed,'axis_role'][j]
        before=interventions[seed,'axis_role','off'][k] if case['comparison']=='conv_off' else scores[seed,'moment'][j]
        path,provenance,true,rival=render(case,data,before,after,geo,up,snapshots,valid,refs)
        records.append(dict(**case,true=true,rival=rival,before=before,after=after,
            old_full=scores[seed,'old_full'][j],moment=scores[seed,'moment'][j],
            conv_off=interventions[seed,'axis_role','off'][k],early=interventions[seed,'axis_role','early'][k],
            diagnostics=case_diagnostics(case,data,reader,rival),figure=path,provenance=provenance))
        print('CASE',case,flush=True)
    write_json(HERE/'analysis/cases.json',records)
    receipt.update(seeds=list(tr.SEEDS),cases=len(records),bootstrap='2000 acquisition-cluster draws after averaging seed effects; BA requires all8classes',
                   interpretation='Repeated development cohort; no target tuning, new supports, backbone, or M3 training.')
    write_json(HERE/'analysis_receipt.json',receipt)
    print(pd.read_csv(HERE/'fixed_head_switch.csv').to_string(index=False),flush=True)


if __name__=='__main__':main()
