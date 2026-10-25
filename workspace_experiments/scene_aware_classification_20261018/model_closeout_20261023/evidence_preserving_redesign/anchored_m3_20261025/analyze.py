"""Paired development outcomes and native-image explanations for anchored M3."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import train as run
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

previous=run.load('anchored_existing_analysis',HERE.parent/'typed_m3_20261025/analyze.py')
# Reuse the established paired-statistics implementation, with this round's arms/output.
previous.HERE=HERE
previous.PAIRS=[(a,'old_full') for a in run.ARMS]+[
    ('typed_both','old_continue'),('typed_both','content_both'),('typed_transport','typed_both')]
effect,margin=previous.effect,previous.margin


def summary_figure():
    frame=pd.read_csv(HERE/'metrics.csv').query("role=='evaluation'")
    arms=('old_full',)+run.ARMS
    labels=('Old both','Continue old','Content both','Typed T only','Typed both')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axs,('OA','BA')):
        for seed,color in zip(run.SEEDS,('#167d8d','#ce6541','#875797')):
            values=[100*float(frame.loc[frame.arm.eq(a)&frame.seed.eq(seed),metric].iloc[0]) for a in arms]
            ax.plot(np.arange(5),values,'o-',color=color,label=f'Seed {seed}',lw=1,ms=4,alpha=.75)
        means=[100*frame.loc[frame.arm.eq(a),metric].mean() for a in arms]
        ax.plot(np.arange(5),means,'ks--',label='Three-seed mean',lw=1.5,ms=5)
        ax.set_xticks(np.arange(5),labels,rotation=15,ha='right')
        ax.set_ylabel(f'{metric} (%)');ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axs[1].legend(fontsize=8)
    fig.suptitle('Same 485 development queries, frozen references and classification head',fontsize=11)
    fig.tight_layout();fig.savefig(HERE/'analysis/paired_accuracy.png',dpi=150);plt.close(fig)


def source_and_development(data,scores):
    rows=[]
    for seed in run.SEEDS:
        for arm in run.ARMS:
            for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
                y=data.table.iloc[ids].class_id.to_numpy(int)
                a,b=scores[seed,arm][ids],scores[seed,'old_full'][ids]
                ma,mb=run.metrics(y,a),run.metrics(y,b)
                rows.append(dict(seed=seed,arm=arm,role=role,**effect(a,b,y),
                    NLL_change=ma['NLL']-mb['NLL'],Brier_change=ma['Brier']-mb['Brier'],
                    margin_change_mean=float((margin(a,y)-margin(b,y)).mean()),
                    max_score_change=float(abs(a-b).max())))
    pd.DataFrame(rows).to_csv(HERE/'source_vs_development.csv',index=False)


def restored(reader,arm,seed):
    model=run.make_model(reader,arm,seed)
    ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state']);model.eval()
    return model,ck['receipt']


@torch.inference_mode()
def switches(data,scores):
    rows=[];cache={}
    y=data.table.iloc[data.te].class_id.to_numpy(int)
    for seed in run.SEEDS:
        reader=run.Reader(data,seed)
        model,receipt=restored(reader,'typed_both',seed)
        for mode in ('initial','transport','score'):
            value=reader.predict(data.te,model,mode=mode,batch=64)
            if mode=='initial':
                error=float(abs(value-scores[seed,'old_full'][data.te]).max())
                assert error<3e-5,error
            cache[f'{seed}_{mode}']=value
            rows.append(dict(seed=seed,mode=mode,selected_step=receipt['selected_step']) |
                effect(value,scores[seed,'old_full'][data.te],y) | run.metrics(y,value))
    pd.DataFrame(rows).to_csv(HERE/'fixed_model_switches.csv',index=False)
    np.savez_compressed(HERE/'results/switches.npz',**cache)


@torch.inference_mode()
def align_diagnostics(data):
    # Training launched before review found the legacy diagnostic-reference mismatch.
    # Preserve its original diagnostic while correcting only that field, never predictions.
    errors=[]
    for seed in run.SEEDS:
        reader=run.Reader(data,seed);model,_=restored(reader,'old_continue',seed)
        score,local,diag=reader.predict(np.arange(data.n),model,diagnostic=True,batch=64)
        path=HERE/'results'/f'old_continue_{seed}.npz'
        with np.load(path) as z: saved={k:z[k].copy() for k in z.files}
        error=float(abs(score-saved['score']).max())
        assert error<3e-5 and np.array_equal(score.argmax(1),saved['score'].argmax(1)),error
        assert float(abs(local-saved['local']).max())<2e-5
        saved.setdefault('historical_diagnostic',saved['diagnostic'])
        saved['diagnostic']=diag
        np.savez_compressed(path,**saved)
        errors.append(dict(seed=seed,score_replay=error))
    run.write_json(HERE/'diagnostic_alignment.json',dict(
        reference='Frozen retained learned metric K0 and T0 for all four arms',
        fields=['transport_L1','mean_abs_K_change','matched_E_change'],replay=errors,
        scope='Diagnostic-only correction; original diagnostic preserved; no fitting or checkpoint reselection.'))


@torch.inference_mode()
def inference_cost(data):
    rows=[]
    for seed in run.SEEDS:
        reader=run.Reader(data,seed)
        for arm in ('old_full','typed_transport','typed_both'):
            if arm=='old_full':
                model=run.make_model(reader,'old_continue',seed);step=None
            else:
                model,receipt=restored(reader,arm,seed);step=receipt['selected_step']
            reader.predict(data.tr[:8],model)
            torch.cuda.synchronize();tick=time.perf_counter()
            score=reader.predict(np.arange(data.n),model,batch=64)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-tick
            with np.load(HERE/'results'/f'{arm}_{seed}.npz') as z: old=z['score']
            error=float(abs(score-old).max())
            assert error<3e-5 and np.array_equal(score.argmax(1),old.argmax(1)),(arm,seed,error)
            rows.append(dict(seed=seed,arm=arm,selected_step=step,n=data.n,
                             seconds=seconds,diagnostic=False,batch=64,score_replay=error))
    pd.DataFrame(rows).to_csv(HERE/'cached_inference_cost.csv',index=False)


def choose_cases(data,scores):
    result=[];used=set();ids=data.te;y=data.table.iloc[ids].class_id.to_numpy(int)
    receipts=pd.read_csv(HERE/'training_receipts.csv')
    for arm in ('typed_both','typed_transport'):
        active=receipts.query('arm==@arm and selected_step>0').seed.tolist()
        if not active:
            active=list(run.SEEDS)
        for case in ('rescue','harm','CC','WW'):
            choices=[]
            for seed in active:
                new=scores[seed,arm][ids].argmax(1)==y
                old=scores[seed,'old_full'][ids].argmax(1)==y
                take={'rescue':new&~old,'harm':~new&old,'CC':new&old,'WW':~new&~old}[case]
                for k in np.flatnonzero(take):
                    obs=int(data.table.iloc[ids[k]].obs_id)
                    if (arm,obs) not in used:choices.append((seed,obs,int(ids[k])))
            choices.sort()
            if choices:
                seed,obs,j=choices[0] if case in ('rescue','harm') else choices[(len(choices)-1)//2]
                result.append(dict(arm=arm,seed=seed,obs_id=obs,index=j,case=case))
                used.add((arm,obs))
    return result


def panel(case,data,before,after,geo,up,valid,snapshots,refs):
    i=int(data.up[case['index']]);true=int(up.iloc[i].class_id)
    anchor=after if case['case']=='harm' else before
    rival=int(np.where(np.arange(8)==true,-np.inf,anchor).argmax())
    views=[(i,'QUERY')]+[(int(refs[i,c,k]),label+str(k+1))
        for c,label in ((true,'TRUE '),(rival,'RIVAL ')) for k in range(2)]
    fig,axs=plt.subplots(2,5,figsize=(16,7));provenance=[]
    for p,pol in enumerate(('vv','vh')):
        for col,(ri,label) in enumerate(views):
            row=up.iloc[ri];ax=axs[p,col];obs=int(row.obs_id)
            entry=dict(up_index=ri,obs_id=obs,polarization=pol,path=str(row['image_'+pol]))
            if valid[ri,p] and obs in snapshots and pol in snapshots[obs]['images']:
                pix,fp,g,place=geo.native(up,snapshots,ri,p)
                ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                for region,color in zip(geo.soft_regions(g)*fp,('#20a780','#d77329','#3287d0')):
                    if region.min()<.5<region.max():
                        ax.contour(region,levels=[.5],colors=[color],linewidths=.55)
                entry.update(status='rendered',geometry=g,placement=place)
            else:
                ax.text(.5,.5,'MISSING VIEW',ha='center',transform=ax.transAxes)
                entry['status']='missing'
            ax.set_title(f'{pol.upper()} {label} {obs}\n{row.class_name}',fontsize=8)
            ax.set(xticks=[],yticks=[]);provenance.append(entry)
    fig.suptitle(f"{case['arm']} vs old both | {case['case']} | seed{case['seed']} | obs{case['obs_id']}\n"
        f"true={true},rival={rival}; prediction {before.argmax()} -> {after.argmax()}; "
        f"fixed margin {before[true]-before[rival]:+.5f} -> {after[true]-after[rival]:+.5f}",fontsize=11)
    fig.text(.02,.02,'Native160 crop/pad; no ship resize. Green body / orange ends / blue sides. Actual fixed references.',fontsize=9)
    fig.subplots_adjust(top=.84,bottom=.09,wspace=.12,hspace=.18)
    path=HERE/'analysis'/f"{case['arm']}_{case['case']}_{case['obs_id']}_{case['seed']}.png"
    fig.savefig(path,dpi=120);plt.close(fig)
    return path,provenance,true,rival


@torch.inference_mode()
def diagnose_case(data,reader,model,case,true,rival):
    ids=[case['index']]
    score,local,diag=reader.forward(ids,model,diagnostic=True)
    score0,local0,_=reader.forward(ids,model,mode='initial',diagnostic=True)
    m=score[0,true]-score[0,rival]
    idx=torch.tensor(reader.ix[ids],device='cuda')
    rows=[]
    for p in range(2):
        for r in range(3):
            replaced=local.clone();replaced[:,:,p,r]=local0[:,:,p,r]
            alt=reader.old.anchor[idx]+reader.old.delta(replaced,idx)-reader.old.old_delta[idx]
            rows.append(dict(polarization=('VV','VH')[p],region=('body','sides','ends')[r],
                fixed_margin_effect=float(m-(alt[0,true]-alt[0,rival])),
                prediction=int(alt[0].argmax()),true_local=local[0,true,p,r].cpu().numpy(),
                rival_local=local[0,rival,p,r].cpu().numpy(),
                true_old=local0[0,true,p,r].cpu().numpy(),rival_old=local0[0,rival,p,r].cpu().numpy(),
                true_change=diag[0,true,p,r].cpu().numpy(),rival_change=diag[0,rival,p,r].cpu().numpy()))
    participants=torch.unique(torch.cat((torch.tensor(ids,device='cuda'),reader.refs[ids].flatten())))
    original=data.desc[participants].clone();blocks=[]
    for p in range(2):
        for label,sl in (('class_moments',slice(32,56)),('band_and_covariance',slice(56,91))):
            data.desc[participants,p,:,:,:,sl]=0.
            alt=reader.forward(ids,model)[0]
            blocks.append(dict(polarization=('VV','VH')[p],block=label,
                fixed_margin_effect=float(m-(alt[true]-alt[rival])),prediction=int(alt.argmax())))
            data.desc[participants]=original
    return dict(regions=rows,source_mean_block_replacements=blocks,
        interpretation='Fixed-model interventions on query and actually used references; not additive causal contributions or retrained ablations.')


def main():
    assert (HERE/'complete.json').exists()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=run.Data();scores={}
    for seed in run.SEEDS:
        for arm in ('old_full',)+run.ARMS:
            with np.load(HERE/'results'/f'{arm}_{seed}.npz') as z:
                np.testing.assert_array_equal(z['up_index'],data.up[:data.n])
                scores[seed,arm]=z['score'].copy()
    receipt=previous.statistics(data,scores)
    source_and_development(data,scores)
    align_diagnostics(data)
    switches(data,scores)
    inference_cost(data)
    chosen=choose_cases(data,scores);(HERE/'analysis').mkdir(exist_ok=True)
    summary_figure()
    geo=run.load('anchored_native_geo',run.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up=pd.read_csv(run.BASE/'regional_utility_pilot/upstream/table.csv')
    valid=np.load(run.BASE/'regional_utility_pilot/upstream/summaries.npz')['valid']
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs=np.load(run.LOCAL/'scores_layer1.npz')['refs'];records=[]
    for case in chosen:
        seed,j,arm=case['seed'],case['index'],case['arm']
        before,after=scores[seed,'old_full'][j],scores[seed,arm][j]
        path,prov,true,rival=panel(case,data,before,after,geo,up,valid,snapshots,refs)
        reader=run.Reader(data,seed);model,_=restored(reader,arm,seed)
        records.append(dict(**case,true=true,rival=rival,before=before,after=after,
            comparisons={a:scores[seed,a][j] for a in ('old_full',)+run.ARMS},
            diagnostics=diagnose_case(data,reader,model,case,true,rival),figure=str(path),provenance=prov))
        print('CASE',case,flush=True)
    run.write_json(HERE/'analysis/cases.json',records)
    receipt.update(cases=len(records),case_selection='Prefer active checkpoints; if none, retained baseline. Sorted first rescue/harm, median CC/WW; illustrative not population prevalence')
    run.write_json(HERE/'analysis_receipt.json',receipt)
    print(pd.read_csv(HERE/'paired.csv').to_string(index=False),flush=True)


if __name__=='__main__':main()
