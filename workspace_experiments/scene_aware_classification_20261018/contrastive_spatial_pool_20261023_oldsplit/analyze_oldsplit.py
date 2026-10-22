"""Oldsplit std paired accounting and four native-chip outcome cases."""
import sys
sys.dont_write_bytecode=True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from run_oldsplit import HERE,EXP,ARMS,metrics
REVIEW_IDS=[]


def intervals(t,p):
    indexed=p.set_index(['arm','instance_id'])
    truth=t.class_id.to_numpy()
    base=indexed.loc[ARMS[0]].loc[t.instance_id].base_prediction.to_numpy()
    predictions={a:indexed.loc[a].loc[t.instance_id].prediction.to_numpy() for a in ARMS}
    predictions['A']=base
    pairs=[(a,'A') for a in ARMS]
    rows=[]
    for block in ['pass_group','mmsi']:
        codes,names=pd.factorize(t[block],sort=True)
        denom=np.zeros((len(names),3))
        np.add.at(denom,(codes,truth),1)
        rng=np.random.default_rng(20261023)
        draws=rng.integers(0,len(names),size=(2000,len(names)),dtype=np.int32)
        for arm,reference in pairs:
            diff=(predictions[arm]==truth).astype(int)-(predictions[reference]==truth).astype(int)
            numer=np.zeros_like(denom)
            np.add.at(numer,(codes,truth),diff)
            # Reuse the same block draws for every comparison, never pick a favorable unit.
            total_denom=denom[draws].sum(1)
            total_numer=numer[draws].sum(1)
            oa=100*total_numer.sum(1)/total_denom.sum(1)
            ba=100*(total_numer/total_denom).mean(1)
            rows.append(dict(arm=arm,reference=reference,block=block,blocks=len(names),draws=2000,seed=20261023,
                OA_delta=100*diff.mean(),BA_delta=100*np.mean([diff[truth==c].mean() for c in range(3)]),
                OA_low=float(np.quantile(oa,.025)),OA_high=float(np.quantile(oa,.975)),
                BA_low=float(np.quantile(ba,.025)),BA_high=float(np.quantile(ba,.975))))
    pd.DataFrame(rows).to_csv(HERE/'paired_intervals.csv',index=False)


def render(t,p):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    selections=[];missing=[]
    for arm in ARMS:
        g=p[p.arm==arm]
        for outcome in ['rescued','harmed','CC','WW']:
            hit=g[g.outcome==outcome].sort_values('instance_id')
            if len(hit):
                selections.append(dict(instance_id=hit.iloc[0].instance_id,selection=f'{arm}:{outcome}:firstID'))
            else:
                missing.append(dict(arm=arm,outcome=outcome))
    selections.extend(dict(instance_id=i,selection='previous_fixed_spatial_case') for i in REVIEW_IDS)
    table=t.set_index('instance_id')
    rows=[];panels=[]
    for ident in dict.fromkeys(s['instance_id'] for s in selections):
        row=table.loc[ident]
        g=p[p.instance_id==ident].set_index('arm').loc[ARMS]
        base=g.iloc[0][['b0','b1','b2']].to_numpy(float)
        truth=int(row.class_id)
        wrong=max((c for c in range(3) if c!=truth),key=lambda c:base[c])
        selection=';'.join(s['selection'] for s in selections if s['instance_id']==ident)
        canvas=Image.new('RGB',(1100,820),'white')
        draw=ImageDraw.Draw(canvas)
        draw.text((15,10),f'{ident} | {row.port} | fold {int(g.iloc[0].fold)} | true={truth}, fixed wrong={wrong}',font=font,fill='black')
        draw.text((15,40),'Native pixels displayed nearest only; green = recorded detection OBB. No physical mechanism labels.',font=small,fill='black')
        for pol,x0 in [('vv',35),('vh',585)]:
            path=EXP/'readout/native_targets'/f'{ident}_{pol}.png'
            with Image.open(path) as im:
                native=im.convert('RGB')
            w,h=native.size
            scale=max(1,min(430//w,430//h))
            display=native.resize((w*scale,h*scale),Image.Resampling.NEAREST)
            canvas.paste(display,(x0,100))
            points=np.array(json.loads(row.target_points))-np.array(json.loads(row[f'target_window_{pol}']))[:2]
            points=points*scale+[x0,100]
            draw.line([tuple(x) for x in np.r_[points,points[:1]]],fill=(0,220,80),width=2)
            draw.text((x0,72),f'{pol.upper()} native {w}x{h}px',font=font,fill='black')
        draw.text((15,550),f'A: pred={base.argmax()}, fixed-pair margin={base[truth]-base[wrong]:+.6f}',font=font,fill='black')
        for j,arm in enumerate(ARMS):
            rec=g.loc[arm]
            score=rec[['s0','s1','s2']].to_numpy(float)
            dd=rec[['delta0','delta1','delta2']].to_numpy(float)
            draw.text((15,585+j*46),f'{arm}: lambda={rec.selected_lambda:g}, pred={int(rec.prediction)}, {rec.outcome}, margin={score[truth]-score[wrong]:+.6f}',font=font,fill='black')
            draw.text((15,608+j*46),'delta=['+', '.join(f'{v:.5f}' for v in dd)+f']; weighted pair increment={rec.selected_lambda*(dd[truth]-dd[wrong]):+.6f}',font=small,fill='black')
            rows.append(dict(instance_id=ident,selection=selection,arm=arm,outcome=rec.outcome,fold=int(rec.fold),port=row.port,
                class_id=truth,wrong_class=wrong,base_prediction=int(base.argmax()),prediction=int(rec.prediction),
                selected_lambda=rec.selected_lambda,base_margin=float(base[truth]-base[wrong]),margin=float(score[truth]-score[wrong]),
                delta_margin=float(dd[truth]-dd[wrong]),weighted_delta_margin=float(rec.selected_lambda*(dd[truth]-dd[wrong])),
                native_vv=str(EXP/'readout/native_targets'/f'{ident}_vv.png'),native_vh=str(EXP/'readout/native_targets'/f'{ident}_vh.png')))
        draw.text((15,752),'Class 0=bulk, 1=fishing, 2=general. Pair contrast fixed to baseline strongest wrong class.',font=small,fill='black')
        path=HERE/f'case_{ident}.png';canvas.save(path);panels.append(str(path))
    pd.DataFrame(rows).to_csv(HERE/'review_cases.csv',index=False)
    return panels,missing


def main():
    start=time.perf_counter()
    t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    p=pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    classes=[];acquisition=[];paired=[]
    for arm,g in p.groupby('arm',sort=False):
        for c,q in g.groupby('class_id'):
            classes.append(dict(arm=arm,class_id=c,n=len(q),baseline_correct=int(q.base_prediction.eq(c).sum()),
                correct=int(q.prediction.eq(c).sum()),rescued=int(q.net.eq(1).sum()),harmed=int(q.net.eq(-1).sum()),net=int(q.net.sum())))
        for unit in ['port','pass_group','fold']:
            for name,q in g.groupby(unit):
                acquisition.append(dict(arm=arm,unit=unit,name=name,n=len(q),mmsi=q.mmsi.nunique(),
                    class_0=int(q.class_id.eq(0).sum()),class_1=int(q.class_id.eq(1).sum()),class_2=int(q.class_id.eq(2).sum()),
                    rescued=int(q.net.eq(1).sum()),harmed=int(q.net.eq(-1).sum()),net=int(q.net.sum()),
                    **metrics(q.class_id.to_numpy(),q[['s0','s1','s2']].to_numpy())))
        replay=g[['b0','b1','b2']].to_numpy()+g.selected_lambda.to_numpy()[:,None]*g[['delta0','delta1','delta2']].to_numpy()
        assert np.max(abs(replay-g[['s0','s1','s2']].to_numpy()))<1e-12
        assert np.array_equal(replay.argmax(1),g.prediction)
    pd.DataFrame(classes).to_csv(HERE/'class_effects.csv',index=False)
    pd.DataFrame(acquisition).to_csv(HERE/'acquisition_effects.csv',index=False)
    pd.DataFrame(paired).to_csv(HERE/'paired_arms.csv',index=False)
    intervals(t,p)
    panels,missing=render(t,p)
    summary=dict(prediction_replay=True,panels=panels,missing_outcomes=missing,seconds=time.perf_counter()-start,
                 bootstrap_scope='fixed predictions, separate pass and MMSI blocks, not training or selection uncertainty')
    (HERE/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
