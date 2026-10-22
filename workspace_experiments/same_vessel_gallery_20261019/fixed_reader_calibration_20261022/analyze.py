"""Repeated-split paired evaluation; no test-time model ensembling."""
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw,ImageFont
from sklearn.metrics import f1_score
import run as exp

ROOT=exp.ROOT


def met(g):
    return dict(oa=g.correct.mean(),ba=g.groupby('class_id').correct.mean().mean(),
                macro_f1=f1_score(g.class_id,g.prediction,average='macro',zero_division=0))


def write_repeat_summary(rows):
    keys=['oa','ba','macro_f1','rescue','harm','weight']
    # Old full-source scores repeat across inner splits for pairing, not new fits.
    summary=rows.groupby('arm')[keys].agg(['mean','std'])
    old=rows[rows.arm.eq('old_full_reference')&rows.inner.eq(0)]
    for key in ('oa','ba','macro_f1','weight'):
        summary.loc['old_full_reference',(key,'mean')]=old[key].mean()
        summary.loc['old_full_reference',(key,'std')]=old[key].std()
    summary.to_csv(ROOT/'summary_mean_std.csv')
    counts=rows.groupby('arm').size().rename('paired_repetitions').to_frame()
    counts['performance_repetitions']=counts.paired_repetitions
    counts.loc['old_full_reference','performance_repetitions']=3
    counts.to_csv(ROOT/'summary_counts.csv')


def summarize(p):
    rows=[];classes=[];groups=[]
    for (inner,seed,arm),a in p.groupby(['inner','seed','arm']):
        b=p[(p.inner==inner)&(p.seed==seed)&p.arm.eq('reference')].set_index('obs_id').loc[a.obs_id]
        assert len(a)==4744 and a.obs_id.nunique()==4744
        rows.append(dict(inner=inner,seed=seed,arm=arm,**met(a),
            rescue=int((a.correct.to_numpy()&~b.correct.to_numpy()).sum()),
            harm=int((~a.correct.to_numpy()&b.correct.to_numpy()).sum()),weight=a.weight.mean()))
        for c,g in a.groupby('class_id'):
            classes.append(dict(inner=inner,seed=seed,arm=arm,class_id=c,n=len(g),recall=g.correct.mean()))
        for key in ('port','active'):
            for value,g in a.groupby(key):
                groups.append(dict(inner=inner,seed=seed,arm=arm,group=key,value=value,n=len(g),**met(g)))
    rows=pd.DataFrame(rows);rows.to_csv(ROOT/'summary_repetitions.csv',index=False)
    keys=['oa','ba','macro_f1','rescue','harm','weight']
    write_repeat_summary(rows)
    rows.groupby(['seed','arm'])[keys].mean().reset_index().to_csv(ROOT/'seed_means.csv',index=False)
    rows.groupby(['inner','arm'])[keys].mean().reset_index().to_csv(ROOT/'split_means.csv',index=False)
    pd.DataFrame(classes).to_csv(ROOT/'class_recall.csv',index=False)
    pd.DataFrame(groups).to_csv(ROOT/'group_results.csv',index=False)
    print(rows.groupby('arm')[keys].mean().to_string())
    avg=p.groupby(['arm','obs_id']).agg(correct=('correct','mean'),mmsi=('mmsi','first'),class_id=('class_id','first')).reset_index()
    pairs=[(a,'reference') for a in exp.ARMS]+[('calibration_score','seen_score'),
        ('calibration_score','calibration_constant'),('calibration_score','calibration_score_cal_mean'),
        ('calibration_score','old_full_reference'),('reference','old_full_reference')]
    ci=[]
    for arm,ref in pairs:
        a=avg[avg.arm==arm].set_index('obs_id').copy();b=avg[avg.arm==ref].set_index('obs_id').loc[a.index]
        a['delta']=a.correct-b.correct;a['n']=1
        ships=a.groupby(['class_id','mmsi'])[['delta','n']].sum().reset_index()
        assert ships.mmsi.nunique()==len(ships)
        rng=np.random.default_rng(20261022);dd=[];nn=[]
        for c in range(8):
            g=ships[ships.class_id==c];ii=rng.integers(len(g),size=(1500,len(g)))
            dd.append(g.delta.to_numpy()[ii].sum(1));nn.append(g.n.to_numpy()[ii].sum(1))
        dd,nn=np.array(dd).T,np.array(nn).T
        for name,point,samples in (('oa',a.delta.mean(),dd.sum(1)/nn.sum(1)),
                                  ('ba',a.groupby('class_id').delta.mean().mean(),(dd/nn).mean(1))):
            lo,hi=np.quantile(samples,[.025,.975])
            ci.append(dict(arm=arm,reference=ref,metric=name,delta=point,lo=lo,hi=hi,
                scope='class-stratified MMSI; averaged repetitions; fixed fitted models'))
    pd.DataFrame(ci).to_csv(ROOT/'paired_uncertainty.csv',index=False)


def verify_and_scores():
    f,splits=exp.old.relation.load_data();y=f.class_id.to_numpy()
    selections=pd.read_csv(ROOT/'validation_selection.csv');checked=[];score_metrics=[]
    for inner in range(3):
        for seed in range(3):
            pooled={a:[] for a in (*exp.ARMS,'reference','baseline','old_full_reference')};truth=[]
            for fold,(_,_,test) in enumerate(splits):
                frozen=np.load(ROOT/f'inputs_f{fold}_k{inner}_s{seed}.npz')
                a,d=frozen['base'][test],frozen['residual'][test];truth.append(y[test])
                pooled['reference'].append(a+d);pooled['baseline'].append(a)
                oldfull=np.load(exp.held.PREVIOUS/f'inputs_f{fold}_s{seed}.npz')
                pooled['old_full_reference'].append(oldfull['base'][test]+oldfull['residual'][test])
                for arm in exp.ARMS:
                    save=torch.load(ROOT/f'head_f{fold}_k{inner}_{arm}_s{seed}.pt',map_location='cpu',weights_only=False)
                    group=selections[(selections.fold==fold)&(selections.inner==inner)&(selections.seed==seed)&selections.arm.eq(arm)]
                    expected=group.sort_values(['ba','brier','step'],ascending=[False,True,True]).iloc[0]
                    assert save['step']==expected.step
                    gate=exp.held.Gate(arm.endswith('constant')).eval();gate.load_state_dict(save['state'])
                    with torch.no_grad():w=gate(torch.tensor(a),torch.tensor(d)).numpy()
                    cached=np.load(ROOT/f'scores_f{fold}_k{inner}_{arm}_s{seed}.npz')
                    err=float(np.abs(w-cached['weight']).max());assert err<1e-6
                    ss=a+w[:,None]*d
                    assert np.array_equal(ss.argmax(1),cached['scores'].argmax(1))
                    checked.append(dict(fold=fold,inner=inner,seed=seed,arm=arm,step=save['step'],cpu_weight_error=err))
                    pooled[arm].append(cached['scores'])
            for arm,ss in pooled.items():
                score_metrics.append(dict(inner=inner,seed=seed,arm=arm,
                    **exp.old.control.metrics(np.concatenate(truth),np.concatenate(ss))))
    pd.DataFrame(checked).to_csv(ROOT/'selected_steps.csv',index=False)
    pd.DataFrame(score_metrics).to_csv(ROOT/'pooled_score_metrics.csv',index=False)
    (ROOT/'VERIFICATION.json').write_text(json.dumps(dict(status='passed',saved_gate_replays=len(checked),
        max_cpu_weight_error=max(r['cpu_weight_error'] for r in checked),same_test_predictions=True,
        reader_replays=len(pd.read_csv(ROOT/'checks.csv'))),indent=2),encoding='utf-8')


def cases(p):
    f,views,disabled=exp.old.control.load_data();idx=dict(zip(f.obs_id,f.index))
    chosen=[('anchor',10461,0,0),('anchor',30237,0,0)];used={10461,30237}
    for name in ('rescue','harm','persistent_correct','persistent_wrong'):
        options=[]
        for (inner,seed),block in p.groupby(['inner','seed']):
            a=block[block.arm.eq('calibration_score')].set_index('obs_id')
            b=block[block.arm.eq('reference')].set_index('obs_id').loc[a.index]
            masks=dict(rescue=a.correct&~b.correct,harm=~a.correct&b.correct,
                       persistent_correct=a.correct&b.correct,persistent_wrong=~a.correct&~b.correct)
            sub=a[masks[name]&~a.index.isin(used)&a.active].copy()
            sub['change']=abs(sub.weight-1);sub=sub[sub.change>1e-6]
            options.extend((float(row.change),int(oid),int(inner),int(seed)) for oid,row in sub.iterrows())
        if options:
            options.sort();_,oid,inner,seed=options[len(options)//2]
            chosen.append((name,oid,inner,seed));used.add(oid)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    rows=[]
    for kind,oid,inner,seed in chosen:
        i=idx[oid];row=f.iloc[i]
        q=p[(p.obs_id==oid)&(p.inner==inner)&(p.seed==seed)].set_index('arm')
        canvas=Image.new('RGB',(1140,470),'white');draw=ImageDraw.Draw(canvas)
        draw.text((10,8),f'{kind} | obs {oid} | {row.class_name} | {row.port} | split {inner}, seed {seed}',font=font,fill='black')
        for ip,pol in enumerate(('vv','vh')):
            pix,m,_,_,_=exp.old.control.prepare_view(row,views[i]['images'][pol],pol)
            x=pix[0];r=m[2].astype(bool);mu=float(x[r].mean()) if r.any() else 0
            z=.5+.5*(x-mu)*r;y0=60+195*ip
            canvas.paste(Image.fromarray(np.rint(x*255).astype(np.uint8)).convert('RGB'),(10,y0))
            canvas.paste(Image.fromarray(np.rint(z*255).astype(np.uint8)).convert('RGB'),(185,y0))
            draw.text((10,y0-18),f'{pol.upper()} native, no resize',font=small,fill='black')
            draw.text((185,y0-18),'Frozen R input',font=small,fill='black')
            draw.text((10,y0+163),f'R mean {mu:.4f}; disabled={bool(disabled[i]) if pol=="vh" else False}',font=small,fill='black')
        arms=('baseline','reference','seen_constant','calibration_constant','seen_score','calibration_score','calibration_score_cal_mean','old_full_reference')
        for j,arm in enumerate(arms):
            z=q.loc[arm]
            draw.text((380,58+32*j),f'{arm:27} pred {int(z.prediction)}; margin {z.margin:+.4f}; w {z.weight:.4f}',font=small,fill='black')
            rows.append(dict(case=kind,obs_id=oid,inner=inner,seed=seed,fold=int(z.fold),arm=arm,
                class_id=int(row.class_id),prediction=int(z.prediction),margin=z.margin,weight=z.weight,
                active=bool(z.active),image_vv=row.image_vv,image_vh=row.image_vh))
        draw.text((380,338),'Same fixed reader for calibration and test. No inference ensemble.',font=small,fill='black')
        draw.text((380,365),'0 bulk, 1 fishing, 2 cargo, 3 chemical/product,',font=small,fill='black')
        draw.text((380,388),'4 container, 5 crude, 6 tug, 7 offshore.',font=small,fill='black')
        draw.text((380,418),'Old full reference uses more upstream training data; separate comparator.',font=small,fill='black')
        canvas.save(ROOT/f'case_{kind}_{oid}_k{inner}_s{seed}.png')
    pd.DataFrame(rows).to_csv(ROOT/'case_index.csv',index=False)


if __name__=='__main__':
    p=pd.read_csv(ROOT/'predictions.csv',dtype={'mmsi':str})
    summarize(p);verify_and_scores();cases(p)
