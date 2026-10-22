"""Saved-model replay and paired analysis of one matched gate-training round."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from run_gates import HERE, PARENT, NEW, EXP, ARMS, SEEDS, GroupGate, load_inputs, metrics
sys.path.insert(0,str(NEW))
from analyze_replication import intervals


def with_outcomes(p):
    p = p.copy()
    p['correct'] = p.prediction.eq(p.class_id)
    p['net'] = p.correct.astype(int)-p.base_prediction.eq(p.class_id).astype(int)
    p['outcome'] = np.select([p.net.eq(1),p.net.eq(-1),p.correct],['rescued','harmed','CC'],default='WW')
    return p


def references():
    columns = ['cohort','fold','seed','arm','instance_id','class_id','prediction','base_prediction','s0','s1','s2']
    frames = []
    for cohort in ['old','new']:
        path = PARENT/'control_predictions.csv' if cohort=='old' else NEW/'predictions.csv'
        arm = 'own_regions_full' if cohort=='old' else 'centered_own_full'
        p = pd.read_csv(path)
        p = p[p.arm.eq(arm)].copy()
        p['cohort'],p['arm'] = cohort,'original_reference'
        frames.append(p[columns])
    p = pd.read_csv(PARENT/'common_gain_diagnosis/predictions.csv')
    p = p[p.variant.eq('contrast_only')].copy()
    p['arm'] = 'fixed_contrast_diagnostic'
    frames.append(p[columns])
    return pd.concat(frames,ignore_index=True)


def render(t,p,b):
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    rows,panels,missing = [],[],[]
    for cohort in ['old','new']:
        anchor = p[p.cohort.eq(cohort)&p.arm.eq('contrast_only_gate')&p.seed.eq(29)]
        for outcome in ['rescued','harmed','CC','WW']:
            chosen = anchor[anchor.outcome.eq(outcome)&anchor.neighbors.eq(0)].sort_values('instance_id')
            if chosen.empty:
                chosen = anchor[anchor.outcome.eq(outcome)].sort_values('instance_id')
            if chosen.empty:
                missing.append(dict(cohort=cohort,outcome=outcome))
                continue
            case = chosen.iloc[0]
            iid = case.instance_id
            meta = t.set_index('instance_id').loc[iid]
            truth = int(meta.class_id)
            base = b[b.cohort.eq(cohort)&b.instance_id.eq(iid)].iloc[0]
            part = p[p.cohort.eq(cohort)&p.seed.eq(29)&p.instance_id.eq(iid)].set_index('arm')
            scores = [('A',base)]+[(arm,part.loc[arm]) for arm in ['original_reference','fixed_contrast_diagnostic']+ARMS]
            ims = [Image.open(EXP/'readout/native_targets'/f'{iid}_{pol}.png').convert('RGB') for pol in ['vv','vh']]
            canvas = Image.new('RGB',(1100,280+max(im.height for im in ims)),'white')
            ink = ImageDraw.Draw(canvas)
            ink.text((10,8),f'{cohort} {iid} {outcome}; seed29 fold{case.fold}; truth={truth} {meta.class_name}; {meta.port}',font=font,fill='black')
            ink.text((10,32),'Native pixels, no resize/enhancement. Outcomes defined by trained contrast gate vs A.',font=font,fill='black')
            for j,(arm,rr) in enumerate(scores):
                s = rr[['s0','s1','s2']].to_numpy(float)
                margin = float(s[truth]-np.delete(s,truth).max())
                ink.text((10,57+24*j),f'{arm}: pred={int(rr.prediction)}, true margin={margin:+.6f}',font=font,fill='black')
                rows.append(dict(cohort=cohort,instance_id=iid,outcome=outcome,fold=int(case.fold),seed=29,arm=arm,
                    class_id=truth,prediction=int(rr.prediction),margin=margin,port=meta.port,pass_group=meta.pass_group,
                    native_vv=str(EXP/'readout/native_targets'/f'{iid}_vv.png'),native_vh=str(EXP/'readout/native_targets'/f'{iid}_vh.png'),
                    window_vv=meta.target_window_vv,window_vh=meta.target_window_vh,
                    selection='seed29 trained-contrast outcome first ID, prefer no external neighbors'))
            for x,pol,im in zip([10,560],['VV','VH'],ims):
                ink.text((x,190),f'{pol} {im.width}x{im.height}',font=font,fill='black')
                canvas.paste(im,(x,214))
            path = HERE/f'case_{cohort}_{iid}.png'
            canvas.save(path)
            panels.append(str(path))
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    return panels,missing


def main():
    torch.set_num_threads(2)
    t = pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    p = pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    b = pd.read_csv(HERE/'baseline_predictions.csv',dtype={'mmsi':str})
    replay,gain_rows = [],[]
    for cohort in ['old','new']:
        for fold in range(6):
            d,kind,_,_ = load_inputs(cohort,fold)
            test = d['test']
            q = torch.tensor(d['q'][test],dtype=torch.float32)
            env = torch.tensor(d['env'][test],dtype=torch.float32)
            for arm in ARMS:
                for seed in SEEDS:
                    model = GroupGate(kind,arm).eval()
                    model.load_state_dict(torch.load(HERE/'models'/f'{cohort}_fold{fold}_{arm}_{seed}.pt',weights_only=True,map_location='cpu'))
                    with torch.inference_mode():
                        coeff = model.coefficients(env).numpy()
                        score = d['base'][test]+model(env,q).numpy()
                    active = model.active.numpy().astype(bool)
                    zero_sum_error = float(abs(coeff[:,active].sum(1)).max())
                    if arm=='contrast_only_gate':
                        assert zero_sum_error < 4e-6
                    saved = p[p.cohort.eq(cohort)&p.fold.eq(fold)&p.arm.eq(arm)&p.seed.eq(seed)].set_index('instance_id').loc[t.iloc[test].instance_id]
                    error = float(abs(score-saved[['s0','s1','s2']].to_numpy()).max())
                    assert error < 2e-5
                    np.testing.assert_array_equal(score.argmax(1),saved.prediction)
                    np.testing.assert_allclose(d['base'][test],saved[['b0','b1','b2']].to_numpy(),atol=2e-14,rtol=0)
                    replay.append(dict(cohort=cohort,fold=fold,arm=arm,seed=seed,error=error,top1_changes=0,active_sum_error=zero_sum_error))
                    gain_rows.append(dict(cohort=cohort,fold=fold,arm=arm,seed=seed,
                        gain_mean=float((1+coeff[:,active]).mean()),sample_gain_sd_mean=float(coeff[:,active].std(1).mean()),
                        gain_min=float((1+coeff[:,active]).min()),gain_max=float((1+coeff[:,active]).max())))
    pd.DataFrame(replay).to_csv(HERE/'independent_replay.csv',index=False)
    pd.DataFrame(gain_rows).to_csv(HERE/'gain_summary.csv',index=False)
    columns = ['cohort','fold','seed','arm','instance_id','class_id','prediction','base_prediction','s0','s1','s2']
    allp = pd.concat([p[columns],references()],ignore_index=True)
    baseline_lookup = b.set_index(['cohort','instance_id'])
    for (cohort,arm,seed),part in allp.groupby(['cohort','arm','seed']):
        ref = baseline_lookup.loc[[(cohort,i) for i in part.instance_id]]
        np.testing.assert_array_equal(part.base_prediction,ref.prediction)
        np.testing.assert_array_equal(part.fold,ref.fold)
    allp = with_outcomes(allp.merge(t[['instance_id','mmsi','pass_group','port','neighbors']],on='instance_id',validate='many_to_one'))
    assert not allp.duplicated(['cohort','arm','seed','instance_id']).any()
    pooled = []
    for (cohort,arm,seed),part in allp.groupby(['cohort','arm','seed']):
        pooled.append(dict(cohort=cohort,arm=arm,seed=seed,net=int(part.net.sum()),rescued=int(part.net.eq(1).sum()),
            harmed=int(part.net.eq(-1).sum()),**metrics(part.class_id.to_numpy(),part[['s0','s1','s2']].to_numpy())))
    pd.DataFrame(pooled).to_csv(HERE/'comparison_pooled.csv',index=False)
    allp.groupby(['cohort','arm','seed','class_id']).agg(n=('instance_id','size'),net=('net','sum'),
        rescued=('net',lambda x:int(x.eq(1).sum())),harmed=('net',lambda x:int(x.eq(-1).sum()))).reset_index().to_csv(HERE/'class_effects.csv',index=False)
    allp.groupby(['cohort','arm','seed','port','pass_group']).agg(n=('instance_id','size'),net=('net','sum')).reset_index().to_csv(HERE/'acquisition_effects.csv',index=False)
    scopes=[]
    for (cohort,arm,seed),part in allp.groupby(['cohort','arm','seed']):
        for label,sub in [('no_external',part[part.neighbors.eq(0)]),('external_available',part[part.neighbors.gt(0)])]:
            scopes.append(dict(cohort=cohort,arm=arm,seed=seed,scope=label,n=len(sub),net=int(sub.net.sum()),
                **metrics(sub.class_id.to_numpy(),sub[['s0','s1','s2']].to_numpy())))
    pd.DataFrame(scopes).to_csv(HERE/'scope_effects.csv',index=False)
    paired,interval_rows,pairframes = [],[],[]
    comparisons = [(arm,'A') for arm in ARMS]+[(arm,'original_reference') for arm in ARMS]+[
        ('contrast_only_gate','range_matched_unrestricted'),('contrast_only_gate','fixed_contrast_diagnostic')]
    for cohort in ['old','new']:
        for arm,reference in comparisons:
            differences=[]
            for seed in SEEDS:
                aa = allp[allp.cohort.eq(cohort)&allp.arm.eq(arm)&allp.seed.eq(seed)].set_index('instance_id').loc[t.instance_id]
                if reference=='A':
                    rp = aa.base_prediction.to_numpy()
                else:
                    rp = allp[allp.cohort.eq(cohort)&allp.arm.eq(reference)&allp.seed.eq(seed)].set_index('instance_id').loc[t.instance_id].prediction.to_numpy()
                diff = aa.correct.to_numpy().astype(int)-(rp==t.class_id.to_numpy()).astype(int)
                differences.append(diff)
                paired.append(dict(cohort=cohort,arm=arm,reference=reference,seed=seed,net=int(diff.sum()),
                    OA_delta=float(diff.mean()*100),BA_delta=float(np.mean([diff[t.class_id.eq(c)].mean() for c in range(3)])*100),
                    rescued=int((diff==1).sum()),harmed=int((diff==-1).sum())))
                pairframes.append(pd.DataFrame(dict(cohort=cohort,arm=arm,reference=reference,seed=seed,instance_id=t.instance_id,
                    class_id=t.class_id,prediction=aa.prediction.to_numpy(),reference_prediction=rp,net=diff)))
            for block in ['pass_group','mmsi']:
                interval_rows.append(dict(cohort=cohort,arm=arm,reference=reference,**intervals(t,np.mean(differences,axis=0),block)))
    pd.DataFrame(paired).to_csv(HERE/'paired_metrics.csv',index=False)
    pd.DataFrame(interval_rows).to_csv(HERE/'paired_intervals.csv',index=False)
    pd.concat(pairframes,ignore_index=True).to_csv(HERE/'paired_predictions.csv',index=False)
    panels,missing = render(t,allp,b)
    summary = dict(replayed_heads=len(replay),max_cpu_replay_error=max(x['error'] for x in replay),
        max_cpu_contrast_sum_error=max(x['active_sum_error'] for x in replay if x['arm']=='contrast_only_gate'),
        baseline_scores_unchanged=True,panels=panels,missing_outcomes=missing,
        boundary='Fixed two development partitions. Diagnostic deletion and separately trained gate are different experiments.')
    (HERE/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    print(pd.DataFrame(paired).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
