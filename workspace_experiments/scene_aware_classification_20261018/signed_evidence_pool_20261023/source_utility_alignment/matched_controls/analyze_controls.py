"""Compare fixed primary candidate with primary controls; no new fits."""
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from run_controls import HERE, ALIGNMENT, parent, assessment

ARMS = ['candidate_signed_only','score_only','conditional_std','linear_distribution']


def block_intervals(y, difference, labels, draws=1000):
    codes, names = pd.factorize(labels, sort=True)
    denominator, numerator = np.zeros((len(names),3)), np.zeros((len(names),3))
    np.add.at(denominator,(codes,y),1)
    np.add.at(numerator,(codes,y),difference)
    rng = np.random.default_rng(20261023)
    all_oa, all_ba, missing = [], [], 0
    for start in range(0,draws,100):
        ix = rng.integers(0,len(names),size=(min(100,draws-start),len(names)))
        dd, nn = denominator[ix].sum(1), numerator[ix].sum(1)
        valid = (dd>0).all(1)
        missing += int((~valid).sum())
        dd, nn = dd[valid], nn[valid]
        all_oa.append(nn.sum(1)/dd.sum(1))
        all_ba.append((nn/dd).mean(1))
    oa, ba = np.concatenate(all_oa), np.concatenate(all_ba)
    jj = .5*(oa+ba)
    return dict(blocks=len(names),draws=draws,missing_class_draws=missing,
        OA_low_pp=100*np.quantile(oa,.025),OA_high_pp=100*np.quantile(oa,.975),
        BA_low_pp=100*np.quantile(ba,.025),BA_high_pp=100*np.quantile(ba,.975),
        J_low_pp=100*np.quantile(jj,.025),J_high_pp=100*np.quantile(jj,.975))


def candidate(role):
    name = 'predictions.csv' if role=='test' else 'validation_predictions.csv'
    g = pd.read_csv(ALIGNMENT/name,dtype={'mmsi':str})
    g = g[g.cell.eq('jointCE__J')].copy()
    g['arm'] = 'candidate_signed_only'
    g['selector'] = 'J'
    return g


def render(table, primary, baseline_label='A', subtitle='Primary J only. MLP seed17; linear deterministic. Green=recorded OBB; native pixels, nearest display.'):
    selections, missing = [], []
    show = primary[(primary.arm.eq('linear_distribution') & primary.seed.eq(-1))|
                   (~primary.arm.eq('linear_distribution') & primary.seed.eq(17))]
    for arm in ARMS:
        arm_rows = show[show.arm.eq(arm)]
        for outcome in ['rescued','harmed','CC','WW']:
            hit = arm_rows[arm_rows.outcome.eq(outcome)].sort_values('instance_id')
            if hit.empty:
                missing.append(dict(arm=arm,outcome=outcome))
            else:
                selections.append(dict(instance_id=hit.iloc[0].instance_id,selection=f'{arm}:{outcome}:firstID'))
    table = table.set_index('instance_id')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    rows, panels = [], []
    for ident in dict.fromkeys(row['instance_id'] for row in selections):
        record = table.loc[ident]
        g = show[show.instance_id.eq(ident)].set_index('arm').loc[ARMS]
        base = g.iloc[0][['b0','b1','b2']].to_numpy(float)
        truth = int(record.class_id)
        wrong = max((c for c in range(3) if c!=truth),key=lambda c:base[c])
        canvas = Image.new('RGB',(1180,1040),'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((18,12),f'{ident} | {record.port} | fold {int(g.iloc[0].fold)} | true={truth}, fixed wrong={wrong}',font=font,fill='black')
        draw.text((18,43),subtitle,font=small,fill='black')
        for pol,x0 in [('vv',35),('vh',625)]:
            path = parent.EXP/'readout/native_targets'/f'{ident}_{pol}.png'
            with Image.open(path) as im:
                native = im.convert('RGB')
            w,h = native.size
            scale = max(1,min(430//w,430//h))
            canvas.paste(native.resize((w*scale,h*scale),Image.Resampling.NEAREST),(x0,115))
            points = np.array(json.loads(record.target_points))-np.array(json.loads(record[f'target_window_{pol}']))[:2]
            points = points*scale+[x0,115]
            draw.line([tuple(v) for v in np.r_[points,points[:1]]],fill=(0,220,80),width=2)
            draw.text((x0,83),f'{pol.upper()} native {w}x{h}px',font=font,fill='black')
        draw.text((18,565),f'{baseline_label} prediction={base.argmax()}, true-versus-fixed-wrong margin={base[truth]-base[wrong]:+.6f}',font=font,fill='black')
        selection = ';'.join(row['selection'] for row in selections if row['instance_id']==ident)
        for j,arm in enumerate(ARMS):
            r = g.loc[arm]
            score = r[['s0','s1','s2']].to_numpy(float)
            margin = score[truth]-score[wrong]
            yy = 610+85*j
            draw.text((18,yy),f'{arm} | epoch={int(r.epoch)} | pred={int(r.prediction)} | {r.outcome}',font=font,fill='black')
            draw.text((18,yy+29),f'margin={margin:+.6f}; pair increment={margin-base[truth]+base[wrong]:+.6f}',font=small,fill='black')
            rows.append(dict(instance_id=ident,selection=selection,arm=arm,seed=int(r.seed),epoch=int(r.epoch),
                fold=int(r.fold),class_id=truth,port=record.port,outcome=r.outcome,
                base_prediction=int(base.argmax()),prediction=int(r.prediction),
                base_margin=float(base[truth]-base[wrong]),margin=float(margin),
                native_vv=str(parent.EXP/'readout/native_targets'/f'{ident}_vv.png'),
                native_vh=str(parent.EXP/'readout/native_targets'/f'{ident}_vh.png')))
        draw.text((18,985),'0=bulk; 1=fishing; 2=general. Visual facts do not establish a physical mechanism.',font=small,fill='black')
        path = HERE/f'case_{ident}.png'
        canvas.save(path)
        panels.append(str(path))
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    return panels,missing


def main():
    started = time.perf_counter()
    table = pd.read_csv(parent.EXP/'readout/targets.csv',dtype={'mmsi':str})
    controls = pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    primary = pd.concat([candidate('test'),controls[controls.selector.eq('J')]],ignore_index=True)
    primary.to_csv(HERE/'primary_predictions.csv',index=False)
    metrics_rows, pairs, classes, intervals, pair_samples = [], [], [], [], []
    for role in ['validation','test']:
        cand = candidate(role)
        path = HERE/('predictions.csv' if role=='test' else 'validation_predictions.csv')
        ct = pd.read_csv(path,dtype={'mmsi':str})
        joined = pd.concat([cand,ct],ignore_index=True)
        for (arm,selector,seed),g in joined.groupby(['arm','selector','seed']):
            original = assessment(g.class_id.to_numpy(),g[['b0','b1','b2']].to_numpy())
            current = assessment(g.class_id.to_numpy(),g[['s0','s1','s2']].to_numpy())
            metrics_rows.append(dict(role=role,arm=arm,selector=selector,seed=seed,n=len(g),net=int(g.net.sum()),
                **current,**{f'delta_{k}':current[k]-original[k] for k in original}))
        for seed in [17,29,43]:
            left = cand[cand.seed.eq(seed)].set_index('instance_id').sort_index()
            for arm in ARMS[1:]:
                right_seed = -1 if arm=='linear_distribution' else seed
                right = ct[ct.arm.eq(arm)&ct.selector.eq('J')&ct.seed.eq(right_seed)].set_index('instance_id').loc[left.index]
                np.testing.assert_array_equal(left.class_id,right.class_id)
                np.testing.assert_allclose(left[['b0','b1','b2']],right[['b0','b1','b2']],atol=1e-12,rtol=0)
                lc,rc = left.prediction.eq(left.class_id),right.prediction.eq(right.class_id)
                diff = lc.astype(int)-rc.astype(int)
                lm = assessment(left.class_id.to_numpy(),left[['s0','s1','s2']].to_numpy())
                rm = assessment(right.class_id.to_numpy(),right[['s0','s1','s2']].to_numpy())
                meta = dict(role=role,reference=arm,candidate_seed=seed,control_seed=right_seed)
                pairs.append(dict(**meta,candidate_only_correct=int((lc&~rc).sum()),
                    control_only_correct=int((~lc&rc).sum()),net=int(diff.sum()),
                    prediction_disagreements=int((left.prediction!=right.prediction).sum()),
                    **{f'delta_{k}':lm[k]-rm[k] for k in lm}))
                if role=='test':
                    for c in range(3):
                        mask = left.class_id.eq(c)
                        classes.append(dict(**meta,class_id=c,n=int(mask.sum()),net=int(diff[mask].sum())))
                    for block in ['pass_group','mmsi']:
                        intervals.append(dict(**meta,block=block,**block_intervals(left.class_id.to_numpy(),diff.to_numpy(),left[block])))
                    gg = left[['class_id','fold','port','pass_group','mmsi']].copy()
                    gg['reference'],gg['candidate_seed'],gg['control_seed'] = arm,seed,right_seed
                    gg['candidate_prediction'],gg['control_prediction'],gg['paired_net'] = left.prediction,right.prediction,diff
                    pair_samples.append(gg.reset_index())
    pd.DataFrame(metrics_rows).to_csv(HERE/'role_metrics.csv',index=False)
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.groupby(['role','arm','selector'])[['OA','BA','J','NLL','balancedNLL','Brier','net']].mean().reset_index().to_csv(HERE/'arm_means.csv',index=False)
    pd.DataFrame(pairs).to_csv(HERE/'candidate_comparison.csv',index=False)
    pd.DataFrame(classes).to_csv(HERE/'candidate_class_comparison.csv',index=False)
    pd.DataFrame(intervals).to_csv(HERE/'candidate_paired_intervals.csv',index=False)
    pd.concat(pair_samples,ignore_index=True).to_csv(HERE/'candidate_pairs.csv',index=False)
    sel = pd.read_csv(HERE/'selections.csv')
    sel.groupby(['arm','selector']).agg(heads=('epoch','size'),epoch0=('epoch',lambda x:int((x==0).sum())),
        epoch_min=('epoch','min'),epoch_median=('epoch','median'),epoch_max=('epoch','max'),
        validation_J_gain_mean=('J_gain','mean')).reset_index().to_csv(HERE/'selection_summary.csv',index=False)
    reconstructed = controls[['b0','b1','b2']].to_numpy()+controls[['delta0','delta1','delta2']].to_numpy()
    error = float(abs(reconstructed-controls[['s0','s1','s2']].to_numpy()).max())
    assert error<1e-12 and np.array_equal(reconstructed.argmax(1),controls.prediction)
    panels,missing = render(table,primary)
    receipt = dict(panels=panels,missing_outcomes=missing,score_reconstruction_error=error,
        seconds=time.perf_counter()-started,intervals='fixed primary predictions, separate pass/MMSI blocks, 1000 draws; not selection uncertainty')
    (HERE/'analysis_summary.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(pd.DataFrame(pairs).query('role == "test"').to_string(index=False),flush=True)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
