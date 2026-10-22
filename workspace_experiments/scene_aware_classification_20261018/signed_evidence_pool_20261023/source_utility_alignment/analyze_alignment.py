"""Fixed predictions, factorial comparisons, validation uncertainty, native panels."""
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from run_alignment import HERE, parent, assessment

CELLS = ['balancedCE__balancedNLL','balancedCE__J','jointCE__balancedNLL','jointCE__J']


def paired_blocks(g, blocks, draws=1000):
    y = g.class_id.to_numpy()
    difference = g.prediction.eq(g.class_id).to_numpy().astype(int)-g.base_prediction.eq(g.class_id).to_numpy().astype(int)
    codes, names = pd.factorize(g[blocks], sort=True)
    denom, numer = np.zeros((len(names),3)), np.zeros((len(names),3))
    np.add.at(denom, (codes,y), 1)
    np.add.at(numer, (codes,y), difference)
    rng = np.random.default_rng(20261023)
    sampled = rng.integers(0,len(names),size=(draws,len(names)))
    dd, nn = denom[sampled].sum(1), numer[sampled].sum(1)
    valid = (dd>0).all(1)
    dd, nn = dd[valid], nn[valid]
    oa, ba = nn.sum(1)/dd.sum(1), (nn/dd).mean(1)
    j = .5*(oa+ba)
    return dict(block=blocks, blocks=len(names), draws=draws, missing_class_draws=int((~valid).sum()),
        J_low_pp=100*np.quantile(j,.025), J_high_pp=100*np.quantile(j,.975),
        OA_low_pp=100*np.quantile(oa,.025), OA_high_pp=100*np.quantile(oa,.975),
        BA_low_pp=100*np.quantile(ba,.025), BA_high_pp=100*np.quantile(ba,.975))


def render(table, predictions):
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    selections, missing = [], []
    p = predictions[predictions.seed.eq(17)]
    for cell in CELLS:
        g = p[p.cell.eq(cell)]
        for outcome in ['rescued','harmed','CC','WW']:
            hit = g[g.outcome.eq(outcome)].sort_values('instance_id')
            if hit.empty:
                missing.append(dict(cell=cell,outcome=outcome))
            else:
                selections.append(dict(instance_id=hit.iloc[0].instance_id,selection=f'{cell}:{outcome}:firstID'))
    rows, panels = [], []
    table = table.set_index('instance_id')
    for ident in dict.fromkeys(s['instance_id'] for s in selections):
        target = table.loc[ident]
        g = p[p.instance_id.eq(ident)].set_index('cell').loc[CELLS]
        base = g.iloc[0][['b0','b1','b2']].to_numpy(float)
        truth = int(target.class_id)
        wrong = max((c for c in range(3) if c!=truth),key=lambda c:base[c])
        canvas = Image.new('RGB',(1180,1040),'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((18,12),f'{ident} | {target.port} | fold {int(g.iloc[0].fold)} | true={truth}, fixed wrong={wrong} | seed17',font=font,fill='black')
        draw.text((18,43),'Native pixels: nearest display only. Green: recorded OBB. No physical-mechanism labels.',font=small,fill='black')
        for pol, x0 in [('vv',35),('vh',625)]:
            path = parent.EXP/'readout/native_targets'/f'{ident}_{pol}.png'
            with Image.open(path) as im:
                native = im.convert('RGB')
            w,h = native.size
            scale = max(1,min(430//w,430//h))
            canvas.paste(native.resize((w*scale,h*scale),Image.Resampling.NEAREST),(x0,115))
            points = np.array(json.loads(target.target_points))-np.array(json.loads(target[f'target_window_{pol}']))[:2]
            points = points*scale+[x0,115]
            draw.line([tuple(v) for v in np.r_[points,points[:1]]],fill=(0,220,80),width=2)
            draw.text((x0,83),f'{pol.upper()} native {w}x{h}px',font=font,fill='black')
        draw.text((18,565),f'A: prediction={base.argmax()}, true-versus-fixed-wrong margin={base[truth]-base[wrong]:+.6f}',font=font,fill='black')
        selection = ';'.join(s['selection'] for s in selections if s['instance_id']==ident)
        for j, cell in enumerate(CELLS):
            r = g.loc[cell]
            score = r[['s0','s1','s2']].to_numpy(float)
            margin = score[truth]-score[wrong]
            yy = 610+85*j
            draw.text((18,yy),f'{cell} | epoch={int(r.epoch)} | pred={int(r.prediction)} | {r.outcome}',font=font,fill='black')
            draw.text((18,yy+29),f'margin={margin:+.6f}; pair increment={margin-base[truth]+base[wrong]:+.6f}',font=small,fill='black')
            rows.append(dict(instance_id=ident,selection=selection,cell=cell,seed=17,epoch=int(r.epoch),
                fold=int(r.fold),class_id=truth,port=target.port,outcome=r.outcome,
                base_prediction=int(base.argmax()),prediction=int(r.prediction),
                base_margin=float(base[truth]-base[wrong]),margin=float(margin),
                native_vv=str(parent.EXP/'readout/native_targets'/f'{ident}_vv.png'),
                native_vh=str(parent.EXP/'readout/native_targets'/f'{ident}_vh.png')))
        draw.text((18,985),'0=bulk; 1=fishing; 2=general. Same architecture and inputs in all four cells.',font=small,fill='black')
        path = HERE/f'case_{ident}.png'
        canvas.save(path)
        panels.append(str(path))
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    return panels, missing


def main():
    start = time.perf_counter()
    p = pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    v = pd.read_csv(HERE/'validation_predictions.csv',dtype={'mmsi':str})
    selected = pd.read_csv(HERE/'selections.csv')
    table = pd.read_csv(parent.EXP/'readout/targets.csv',dtype={'mmsi':str})
    role_rows, intervals, paired = [], [], []
    for role, all_rows in [('validation',v),('test',p)]:
        for (cell,seed), g in all_rows.groupby(['cell','seed']):
            base = assessment(g.class_id.to_numpy(),g[['b0','b1','b2']].to_numpy())
            score = assessment(g.class_id.to_numpy(),g[['s0','s1','s2']].to_numpy())
            role_rows.append(dict(role=role,cell=cell,seed=seed,n=len(g),net=int(g.net.sum()),
                **{f'base_{k}':val for k,val in base.items()},**score,
                **{f'delta_{k}':score[k]-base[k] for k in base}))
        for seed in [17,29,43]:
            arrays = {cell:all_rows[(all_rows.cell==cell)&(all_rows.seed==seed)].set_index('instance_id').sort_index() for cell in CELLS}
            for cell,ref in [(CELLS[1],CELLS[0]),(CELLS[3],CELLS[2]),(CELLS[2],CELLS[0]),(CELLS[3],CELLS[1])]:
                left,right=arrays[cell],arrays[ref]
                np.testing.assert_array_equal(left.index,right.index)
                lc=left.prediction.eq(left.class_id);rc=right.prediction.eq(right.class_id)
                lm=assessment(left.class_id.to_numpy(),left[['s0','s1','s2']].to_numpy())
                rm=assessment(right.class_id.to_numpy(),right[['s0','s1','s2']].to_numpy())
                paired.append(dict(role=role,seed=seed,cell=cell,reference=ref,net=int(lc.sum()-rc.sum()),
                    newly_correct=int((lc&~rc).sum()),newly_wrong=int((~lc&rc).sum()),
                    **{f'delta_{key}':lm[key]-rm[key] for key in lm}))
    for (fold,cell,seed),g in v.groupby(['fold','cell','seed']):
        for block in ['pass_group','mmsi']:
            intervals.append(dict(fold=fold,cell=cell,seed=seed,**paired_blocks(g,block)))
    pd.DataFrame(role_rows).to_csv(HERE/'role_pooled.csv',index=False)
    pd.DataFrame(paired).to_csv(HERE/'factor_pairing.csv',index=False)
    pd.DataFrame(intervals).to_csv(HERE/'validation_intervals.csv',index=False)
    role_frame=pd.DataFrame(role_rows)
    cols=['OA','BA','NLL','balancedNLL','J','Brier','delta_OA','delta_BA','delta_NLL','delta_balancedNLL','delta_J','net']
    role_frame.groupby(['role','cell'])[cols].mean().reset_index().to_csv(HERE/'cell_means.csv',index=False)
    summaries=selected.groupby('cell').agg(heads=('epoch','size'),epoch0=('epoch',lambda a:int((a==0).sum())),
        epoch_min=('epoch','min'),epoch_median=('epoch','median'),epoch_max=('epoch','max'),
        J_gain_min=('J_gain','min'),J_gain_max=('J_gain','max'),
        J_next_gap_median=('J_best_minus_next_distinct','median'),
        J_near_best_checkpoints_median=('J_within_0_1pp_of_best','median')).reset_index()
    summaries.to_csv(HERE/'selection_summary.csv',index=False)
    reconstruction=p[['b0','b1','b2']].to_numpy()+p[['delta0','delta1','delta2']].to_numpy()
    error=float(abs(reconstruction-p[['s0','s1','s2']].to_numpy()).max())
    assert error<1e-12 and np.array_equal(reconstruction.argmax(1),p.prediction)
    panels,missing=render(table,p)
    summary=dict(panels=panels,missing_outcomes=missing,prediction_replay_error=error,
        seconds=time.perf_counter()-start,validation_intervals='fixed selected heads; no correction for selection optimism')
    (HERE/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(role_frame.groupby(['role','cell'])[cols].mean().to_string(),flush=True)
    print(summaries.to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
