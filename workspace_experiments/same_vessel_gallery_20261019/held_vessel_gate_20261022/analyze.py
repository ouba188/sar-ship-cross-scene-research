"""Paired results, supervision exposure, and native-image evidence."""
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import f1_score
import run as exp

ROOT = exp.ROOT


def summarize(p):
    rows, classes, ports = [], [], []
    for (seed, arm), g in p.groupby(['seed','arm']):
        ref = p[(p.seed==seed)&p.arm.eq('reference')].set_index('obs_id').loc[g.obs_id]
        assert len(g)==4744 and g.obs_id.nunique()==4744
        rows.append(dict(seed=seed,arm=arm,oa=g.correct.mean(),
            ba=g.groupby('class_id').correct.mean().mean(),
            macro_f1=f1_score(g.class_id,g.prediction,average='macro',zero_division=0),
            rescue=int((g.correct.to_numpy()&~ref.correct.to_numpy()).sum()),
            harm=int((~g.correct.to_numpy()&ref.correct.to_numpy()).sum()),mean_weight=g.weight.mean()))
        for c,b in g.groupby('class_id'):
            classes.append(dict(seed=seed,arm=arm,class_id=c,n=len(b),recall=b.correct.mean()))
        for port,b in g.groupby('port'):
            ports.append(dict(seed=seed,arm=arm,port=port,n=len(b),oa=b.correct.mean(),
                              ba=b.groupby('class_id').correct.mean().mean()))
    rows=pd.DataFrame(rows)
    rows.to_csv(ROOT/'summary_by_seed.csv',index=False)
    rows.groupby('arm').agg({k:['mean','std'] for k in ('oa','ba','macro_f1','rescue','harm','mean_weight')}).to_csv(ROOT/'summary_mean_std.csv')
    pd.DataFrame(classes).to_csv(ROOT/'class_recall.csv',index=False)
    pd.DataFrame(ports).to_csv(ROOT/'port_results.csv',index=False)
    print(rows.groupby('arm').mean(numeric_only=True).to_string())
    avg=p.groupby(['arm','obs_id']).agg(correct=('correct','mean'),mmsi=('mmsi','first'),class_id=('class_id','first')).reset_index()
    pairs=[(a,'reference') for a in exp.ARMS]+[('oof_score','in_score'),('oof_score','oof_constant'),
                                              ('in_score','in_constant'),('oof_score','oof_score_source_mean')]
    intervals=[]
    for arm,ref in pairs:
        a=avg[avg.arm==arm].set_index('obs_id').copy()
        b=avg[avg.arm==ref].set_index('obs_id').loc[a.index]
        a['delta']=a.correct-b.correct;a['n']=1
        ships=a.groupby(['class_id','mmsi'])[['delta','n']].sum().reset_index()
        assert ships.mmsi.nunique()==len(ships)
        rng=np.random.default_rng(20261022);dd=[];nn=[]
        for c in range(8):
            block=ships[ships.class_id==c];ii=rng.integers(len(block),size=(1500,len(block)))
            dd.append(block.delta.to_numpy()[ii].sum(1));nn.append(block.n.to_numpy()[ii].sum(1))
        dd,nn=np.array(dd).T,np.array(nn).T
        for met,point,boot in (('oa',a.delta.mean(),dd.sum(1)/nn.sum(1)),
                              ('ba',a.groupby('class_id').delta.mean().mean(),(dd/nn).mean(1))):
            lo,hi=np.quantile(boot,[.025,.975])
            intervals.append(dict(arm=arm,reference=ref,metric=met,difference=point,ci_low=lo,ci_high=hi,
                                  scope='fixed-model run-averaged class-stratified MMSI bootstrap'))
    pd.DataFrame(intervals).to_csv(ROOT/'paired_uncertainty.csv',index=False)


def training_signal():
    f,splits=exp.old.relation.load_data();y=f.class_id.to_numpy();rows=[]
    for fold,(tr,val,test) in enumerate(splits):
        oof=np.load(ROOT/f'oof_f{fold}.npz');np.testing.assert_array_equal(oof['rows'],tr)
        for seed in range(3):
            full=np.load(exp.PREVIOUS/f'inputs_f{fold}_s{seed}.npz')
            for part,ii,a,d in (('in_sample',tr,full['base'][tr],full['residual'][tr]),
                                ('held_source',tr,oof['base'][seed],oof['residual'][seed]),
                                ('validation',val,full['base'][val],full['residual'][val]),
                                ('test_posthoc',test,full['base'][test],full['residual'][test])):
                truth=y[ii];target=2*np.eye(8)[truth]-1
                before=a.argmax(1)==truth;after=(a+d).argmax(1)==truth
                sw=exp.old.prior.weights(f,ii)
                numerator=np.sum(sw[:,None]*d*(target-a))
                denominator=np.sum(sw[:,None]*d*d)
                derivative=2*np.mean((a+d-target)*d,axis=1)
                active=np.abs(d).max(1)>1e-8
                rows.append(dict(fold=fold,seed=seed,partition=part,n=len(ii),
                    base_oa=before.mean(),reference_oa=after.mean(),rescue=int((after&~before).sum()),
                    harm=int((~after&before).sum()),active=int(active.sum()),
                    weighted_mse_derivative=float(np.average(derivative,weights=sw)),
                    increase_fraction=float((derivative[active]<0).mean()) if active.any() else np.nan,
                    posthoc_optimal_constant=float(np.clip(numerator/denominator,0,2)) if denominator>1e-12 else np.nan))
    pd.DataFrame(rows).to_csv(ROOT/'training_signal.csv',index=False)


def continuous_scores():
    f,splits=exp.old.relation.load_data();rows=[]
    for seed in range(3):
        for arm in (*exp.ARMS,'baseline','reference'):
            yy=[];ss=[]
            for fold,(_,_,test) in enumerate(splits):
                yy.append(f.iloc[test].class_id.to_numpy())
                if arm in exp.ARMS:
                    ss.append(np.load(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz')['scores'])
                else:
                    z=np.load(exp.PREVIOUS/f'inputs_f{fold}_s{seed}.npz')
                    ss.append(z['base'][test]+(z['residual'][test] if arm=='reference' else 0))
            rows.append(dict(seed=seed,arm=arm,**exp.old.control.metrics(np.concatenate(yy),np.concatenate(ss))))
    table=pd.DataFrame(rows);table.to_csv(ROOT/'pooled_score_metrics.csv',index=False)
    print(table.groupby('arm')[['brier','nll']].mean().to_string())


def verification():
    selected=pd.read_csv(ROOT/'validation_selection.csv');checked=[]
    f,splits=exp.old.relation.load_data()
    for (fold,seed,arm),g in selected.groupby(['fold','seed','arm']):
        expected=g.sort_values(['ba','brier','step'],ascending=[False,True,True]).iloc[0]
        h=torch.load(ROOT/f'head_f{fold}_{arm}_s{seed}.pt',weights_only=False,map_location='cpu')
        assert h['step']==expected.step
        frozen=np.load(exp.PREVIOUS/f'inputs_f{fold}_s{seed}.npz');test=splits[fold][2]
        model=exp.Gate(arm.endswith('constant')).eval();model.load_state_dict(h['state'])
        with torch.no_grad():
            w=model(torch.tensor(frozen['base'][test]),torch.tensor(frozen['residual'][test])).numpy()
        cached=np.load(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz')
        error=float(np.abs(w-cached['weight']).max());assert error<1e-6
        scores=frozen['base'][test]+w[:,None]*frozen['residual'][test]
        assert np.array_equal(scores.argmax(1),cached['scores'].argmax(1))
        checked.append(dict(fold=fold,seed=seed,arm=arm,step=h['step'],cpu_weight_error=error))
    pd.DataFrame(checked).to_csv(ROOT/'selected_steps.csv',index=False)
    stack=pd.read_csv(ROOT/'stack_selection.csv');stack_checks=0
    for (fold,inner,seed,stage),g in stack.groupby(['fold','inner','seed','stage']):
        expected=g.sort_values(['ba','brier','step'],ascending=[False,True,True]).iloc[0]
        if stage=='ridge':
            h=np.load(ROOT/f'inner_f{fold}_k{inner}_normalization.npz')
            assert float(h['ridge_alpha'])==expected.step
        else:
            h=torch.load(ROOT/f'inner_f{fold}_k{inner}_{stage}_s{seed}.pt',weights_only=False,map_location='cpu')
            assert h['step']==expected.step
        stack_checks+=1
    (ROOT/'VERIFICATION.json').write_text(json.dumps(dict(status='passed',gate_selection_and_cpu_replays=len(checked),
        inner_stack_selections=stack_checks,scope='held identities checked in training; no independent data claim'),indent=2),encoding='utf-8')


def cases(p):
    f,views,disabled=exp.old.control.load_data();idx=dict(zip(f.obs_id,f.index))
    selected=[('anchor',10461,0),('anchor',30237,0)]
    used={10461,30237}
    # Select median weight-change examples, not only the largest improvements.
    for name in ('rescue','harm','persistent_correct','persistent_wrong'):
        options=[]
        for seed in range(3):
            a=p[(p.seed==seed)&p.arm.eq('oof_score')].set_index('obs_id')
            b=p[(p.seed==seed)&p.arm.eq('reference')].set_index('obs_id').loc[a.index]
            masks={'rescue':a.correct&~b.correct,'harm':~a.correct&b.correct,
                   'persistent_correct':a.correct&b.correct,'persistent_wrong':~a.correct&~b.correct}
            sub=a[masks[name]&~a.index.isin(used)].copy()
            sub['change']=abs(sub.weight-1);sub=sub[sub.change>1e-6]
            options.extend((float(row.change),int(oid),seed) for oid,row in sub.iterrows())
        if options:
            options.sort();_,oid,seed=options[len(options)//2]
            selected.append((name,oid,seed));used.add(oid)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12);records=[]
    for kind,oid,seed in selected:
        i=idx[oid];row=f.iloc[i];q=p[(p.obs_id==oid)&(p.seed==seed)].set_index('arm')
        canvas=Image.new('RGB',(1110,470),'white');draw=ImageDraw.Draw(canvas)
        draw.text((10,8),f'{kind} | obs {oid} | {row.class_name} | {row.port} | seed {seed}',font=font,fill='black')
        for poln,pol in enumerate(('vv','vh')):
            pix,m,_,_,_=exp.old.control.prepare_view(row,views[i]['images'][pol],pol)
            x=pix[0];r=m[2].astype(bool);mu=float(x[r].mean()) if r.any() else 0
            z=.5+.5*(x-mu)*r;y0=60+195*poln
            canvas.paste(Image.fromarray(np.rint(x*255).astype(np.uint8)).convert('RGB'),(10,y0))
            canvas.paste(Image.fromarray(np.rint(z*255).astype(np.uint8)).convert('RGB'),(185,y0))
            draw.text((10,y0-18),f'{pol.upper()} native, no resize',font=small,fill='black')
            draw.text((185,y0-18),'Frozen R input',font=small,fill='black')
            draw.text((10,y0+163),f'R mean={mu:.4f}; VH disabled={bool(disabled[i])}' if pol=='vh' else f'R mean={mu:.4f}',font=small,fill='black')
        for j,arm in enumerate(('baseline','reference','in_constant','oof_constant','in_score','oof_score','oof_score_source_mean')):
            z=q.loc[arm]
            draw.text((380,60+34*j),f'{arm:24} pred {int(z.prediction)}; margin {z.margin:+.4f}; w {z.weight:.4f}',font=small,fill='black')
            records.append(dict(case=kind,obs_id=oid,seed=seed,fold=int(z.fold),arm=arm,class_id=int(row.class_id),
                prediction=int(z.prediction),margin=z.margin,weight=z.weight,image_vv=row.image_vv,image_vh=row.image_vh))
        draw.text((380,335),'Same frozen image responses; only A+wD weighting changes.',font=small,fill='black')
        draw.text((380,360),'0 bulk, 1 fishing, 2 cargo, 3 chemical/product,',font=small,fill='black')
        draw.text((380,382),'4 container, 5 crude tanker, 6 tug, 7 offshore.',font=small,fill='black')
        draw.text((380,417),'No target support bank in this diagnostic. Not a new physical readout.',font=small,fill='black')
        canvas.save(ROOT/f'case_{kind}_{oid}_s{seed}.png')
    pd.DataFrame(records).to_csv(ROOT/'case_index.csv',index=False)


if __name__=='__main__':
    p=pd.read_csv(ROOT/'predictions.csv',dtype={'mmsi':str})
    summarize(p);training_signal();continuous_scores();verification();cases(p)
