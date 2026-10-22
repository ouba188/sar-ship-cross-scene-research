"""Replay decoupled scores, compare frozen references, and inspect saturation."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from run_score import HERE,PARENT,EXP,ARM,SEEDS,ScoreDecoupled,load_inputs,metrics,probabilistic_metrics
sys.path.insert(0,str(PARENT/'common_gain_training'))
from analyze_gates import references
sys.path.insert(0,str(PARENT/'partition_replication'))
from analyze_replication import intervals


def render(t,p):
    rows,panels,missing=[],[],[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for cohort in ['old','new']:
        anchor=p[p.cohort.eq(cohort)&p.arm.eq(ARM)&p.seed.eq(29)]
        for outcome in ['rescued','harmed','CC','WW']:
            selected=anchor[anchor.outcome.eq(outcome)&anchor.neighbors.eq(0)].sort_values('instance_id')
            if selected.empty:selected=anchor[anchor.outcome.eq(outcome)].sort_values('instance_id')
            if selected.empty:
                missing.append(dict(cohort=cohort,outcome=outcome));continue
            case=selected.iloc[0];iid=case.instance_id;truth=int(case.class_id)
            meta=t.set_index('instance_id').loc[iid]
            part=p[p.cohort.eq(cohort)&p.instance_id.eq(iid)&(p.seed.eq(29)|p.arm.eq('A'))].set_index('arm')
            ims=[Image.open(EXP/'readout/native_targets'/f'{iid}_{pol}.png').convert('RGB') for pol in ['vv','vh']]
            canvas=Image.new('RGB',(1140,300+max(x.height for x in ims)),'white');ink=ImageDraw.Draw(canvas)
            ink.text((10,8),f'{cohort} {iid} {outcome}; seed29 fold{case.fold}; truth={truth} {meta.class_name}; {meta.port}',font=font,fill='black')
            ink.text((10,32),'Native VV/VH, no resize/enhancement. Outcomes: score-decoupled vs frozen A.',font=font,fill='black')
            arms=['A','original_reference','contrast_only_gate','range_matched_unrestricted',ARM,'unscaled_core_diagnostic']
            for j,arm in enumerate(arms):
                rr=part.loc[arm];s=rr[['s0','s1','s2']].to_numpy(float)
                margin=float(s[truth]-np.delete(s,truth).max())
                ink.text((10,57+23*j),f'{arm}: pred={int(rr.prediction)}, true margin={margin:+.6f}',font=font,fill='black')
                rows.append(dict(cohort=cohort,instance_id=iid,outcome=outcome,seed=29,fold=int(case.fold),arm=arm,
                    class_id=truth,prediction=int(rr.prediction),margin=margin,port=meta.port,pass_group=meta.pass_group,
                    native_vv=str(EXP/'readout/native_targets'/f'{iid}_vv.png'),native_vh=str(EXP/'readout/native_targets'/f'{iid}_vh.png'),
                    window_vv=meta.target_window_vv,window_vh=meta.target_window_vh,
                    selection='seed29 first ID per score-decoupled outcome, no external neighbors preferred'))
            for x,pol,im in zip([10,580],['VV','VH'],ims):
                ink.text((x,207),f'{pol} {im.width}x{im.height}',font=font,fill='black');canvas.paste(im,(x,231))
            path=HERE/f'case_{cohort}_{iid}.png';canvas.save(path);panels.append(str(path))
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    return panels,missing


def main():
    torch.set_num_threads(2)
    t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    p=pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    checks=[]
    for cohort in ['old','new']:
        for fold in range(6):
            d,kind,_,_=load_inputs(cohort,fold);test=d['test']
            env=torch.tensor(d['env'][test],dtype=torch.float32)
            q=torch.tensor(d['q'][test],dtype=torch.float32)
            base=torch.tensor(d['base'][test],dtype=torch.float64)
            for seed in SEEDS:
                model=ScoreDecoupled(kind).eval()
                model.load_state_dict(torch.load(HERE/'models'/f'{cohort}_fold{fold}_{seed}.pt',weights_only=True,map_location='cpu'))
                with torch.inference_mode():
                    a,m,coeff=model.parts(env)
                    core=base+(coeff[:,:,None]*q).sum(1)
                    score=model(env,q,base)
                assert torch.equal(score.argmax(1),core.argmax(1))
                saved=p[p.cohort.eq(cohort)&p.fold.eq(fold)&p.seed.eq(seed)].set_index('instance_id').loc[t.iloc[test].instance_id]
                error=float(abs(score.numpy()-saved[['s0','s1','s2']].to_numpy()).max())
                assert error<2e-5
                np.testing.assert_array_equal(score.argmax(1).numpy(),saved.prediction)
                np.testing.assert_allclose(base.numpy(),saved[['b0','b1','b2']].to_numpy(),atol=2e-14,rtol=0)
                checks.append(dict(cohort=cohort,fold=fold,seed=seed,error=error,top1_changes=0,
                    active_sum_error=float(coeff.sum(1).abs().max()),scale_min=float((1+m).min()),scale_max=float((1+m).max())))
    pd.DataFrame(checks).to_csv(HERE/'independent_replay.csv',index=False)
    cols=['cohort','fold','seed','arm','instance_id','class_id','prediction','base_prediction','s0','s1','s2']
    core=p.copy();core['arm']='unscaled_core_diagnostic'
    for c in range(3):core[f's{c}']=core[f'u{c}']
    core['prediction']=core[['s0','s1','s2']].to_numpy().argmax(1)
    np.testing.assert_array_equal(core.prediction,p.prediction)
    trained=pd.read_csv(PARENT/'common_gain_training/predictions.csv')
    original=references().query('arm == "original_reference"')
    base=pd.read_csv(PARENT/'common_gain_training/baseline_predictions.csv')
    allp=pd.concat([p[cols],core[cols],trained[cols],original[cols],base[cols]],ignore_index=True)
    allp=allp.merge(t[['instance_id','mmsi','port','pass_group','neighbors']],on='instance_id',validate='many_to_one')
    allp['correct']=allp.prediction.eq(allp.class_id)
    allp['net']=allp.correct.astype(int)-allp.base_prediction.eq(allp.class_id).astype(int)
    allp['outcome']=np.select([allp.net.eq(1),allp.net.eq(-1),allp.correct],['rescued','harmed','CC'],default='WW')
    pooled=[]
    for (cohort,arm,seed),part in allp.groupby(['cohort','arm','seed']):
        score=part[['s0','s1','s2']].to_numpy();y=part.class_id.to_numpy()
        pooled.append(dict(cohort=cohort,arm=arm,seed=seed,net=int(part.net.sum()),rescued=int(part.net.eq(1).sum()),
            harmed=int(part.net.eq(-1).sum()),**metrics(y,score),**probabilistic_metrics(y,score)))
    pd.DataFrame(pooled).to_csv(HERE/'comparison_pooled.csv',index=False)
    allp.groupby(['cohort','arm','seed','class_id']).agg(n=('instance_id','size'),net=('net','sum'),
        rescued=('net',lambda x:int(x.eq(1).sum())),harmed=('net',lambda x:int(x.eq(-1).sum()))).reset_index().to_csv(HERE/'class_effects.csv',index=False)
    allp[allp.arm.eq(ARM)].groupby(['cohort','seed','port','pass_group']).agg(n=('instance_id','size'),net=('net','sum')).reset_index().to_csv(HERE/'acquisition_effects.csv',index=False)
    paired,ci,pairframes=[],[],[]
    for cohort in ['old','new']:
        for reference in ['A','original_reference','contrast_only_gate','range_matched_unrestricted']:
            differences=[]
            for seed in SEEDS:
                aa=allp[allp.cohort.eq(cohort)&allp.arm.eq(ARM)&allp.seed.eq(seed)].set_index('instance_id').loc[t.instance_id]
                bb=allp[allp.cohort.eq(cohort)&allp.arm.eq(reference)&allp.seed.eq(-1 if reference=='A' else seed)].set_index('instance_id').loc[t.instance_id]
                np.testing.assert_array_equal(aa.base_prediction,bb.base_prediction)
                np.testing.assert_array_equal(aa.fold,bb.fold)
                diff=aa.correct.to_numpy().astype(int)-bb.correct.to_numpy().astype(int)
                differences.append(diff)
                pm=probabilistic_metrics(t.class_id.to_numpy(),aa[['s0','s1','s2']].to_numpy())
                rm=probabilistic_metrics(t.class_id.to_numpy(),bb[['s0','s1','s2']].to_numpy())
                paired.append(dict(cohort=cohort,reference=reference,seed=seed,net=int(diff.sum()),OA_delta=float(diff.mean()*100),
                    BA_delta=float(np.mean([diff[t.class_id.eq(c)].mean() for c in range(3)])*100),
                    **{k+'_delta':pm[k]-rm[k] for k in pm}))
                pairframes.append(pd.DataFrame(dict(cohort=cohort,reference=reference,seed=seed,instance_id=t.instance_id,
                    class_id=t.class_id,prediction=aa.prediction.to_numpy(),reference_prediction=bb.prediction.to_numpy(),net=diff)))
            for block in ['pass_group','mmsi']:
                ci.append(dict(cohort=cohort,reference=reference,**intervals(t,np.mean(differences,axis=0),block)))
    pd.DataFrame(paired).to_csv(HERE/'paired_metrics.csv',index=False)
    pd.DataFrame(ci).to_csv(HERE/'paired_intervals.csv',index=False)
    pd.concat(pairframes,ignore_index=True).to_csv(HERE/'paired_predictions.csv',index=False)
    gates=pd.read_csv(HERE/'sample_gates.csv')
    summaries=[]
    for (cohort,seed),part in gates.groupby(['cohort','seed']):
        summaries.append(dict(cohort=cohort,seed=seed,n=len(part),m_mean=float(part.m.mean()),
            **{f'm_p{x:02d}':float(part.m.quantile(x/100)) for x in [0,5,25,50,75,95,100]},
            d_sd_mean=float(part.d_sd.mean()),d_sd_p05=float(part.d_sd.quantile(.05)),d_sd_p95=float(part.d_sd.quantile(.95)),
            saturation_fraction=float(part.saturation_fraction.mean()),all_active_saturated_fraction=float(part.saturation_fraction.eq(1).mean()),
            m_gt_049_fraction=float(part.m.gt(.49).mean()),correlation_m_d_sd=float(part.m.corr(part.d_sd))))
    pd.DataFrame(summaries).to_csv(HERE/'gate_pooled.csv',index=False)
    gates['m_bin']=pd.cut(gates.m,[-.500001,0,.25,.45,.49,.500001],include_lowest=True)
    gates.groupby(['cohort','seed','m_bin'],observed=True).agg(n=('instance_id','size'),m_mean=('m','mean'),
        d_sd_mean=('d_sd','mean'),saturation_fraction=('saturation_fraction','mean')).reset_index().to_csv(HERE/'saturation_strata.csv',index=False)
    panels,missing=render(t,allp)
    summary=dict(replayed_heads=len(checks),max_cpu_replay_error=max(x['error'] for x in checks),
        baseline_scores_unchanged=True,full_core_argmax_same=True,panels=panels,missing_outcomes=missing,
        note='Same-capacity different operators; fixed-core temperature diagnostic is not another trained arm. No new variants.')
    (HERE/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).groupby(['cohort','arm'])[['OA','BA','NLL','balanced_NLL','Brier','net']].mean().to_string(),flush=True)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
