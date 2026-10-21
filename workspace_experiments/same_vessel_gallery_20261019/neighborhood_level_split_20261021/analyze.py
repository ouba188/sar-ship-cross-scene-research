"""Paired reporting and native-pixel cases for the neighborhood pilot."""
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr
import run as exp

ROOT = exp.ROOT


def evaluate():
    f, splits = exp.relation.load_data()
    y = f.class_id.to_numpy()
    records, predictions, classes, fixed = [], [], [], []
    for seed in range(3):
        for arm in ('baseline', *exp.ARMS):
            files = [np.load(ROOT/f'scores_f{k}_{arm}_s{seed}.npz') for k in range(3)]
            rows = np.concatenate([a['rows'] for a in files])
            scores = np.concatenate([a['scores'] for a in files])
            base = np.concatenate([np.load(ROOT/f'scores_f{k}_baseline_s{seed}.npz')['scores'] for k in range(3)])
            correct = scores.argmax(1) == y[rows]; old = base.argmax(1) == y[rows]
            records.append(dict(arm=arm,seed=seed,rescue=int((correct & ~old).sum()),
                                harm=int((old & ~correct).sum()), **exp.control.metrics(y[rows],scores)))
            other=scores.copy(); other[np.arange(len(rows)),y[rows]]=-np.inf
            for j,i in enumerate(rows):
                predictions.append(dict(arm=arm,seed=seed,obs_id=int(f.iloc[i].obs_id),mmsi=f.iloc[i].mmsi,
                    class_id=int(y[i]),correct=bool(correct[j]),prediction=int(scores[j].argmax()),
                    margin=float(scores[j,y[i]]-other[j].max()),port=f.iloc[i].port))
            for c in range(8):
                take=y[rows]==c
                classes.append(dict(arm=arm,seed=seed,class_id=c,n=int(take.sum()),
                                    recall=float(correct[take].mean())))
            if arm!='baseline':
                for key in ('source_mean_level','source_constant'):
                    ss=np.concatenate([a[key] for a in files])
                    fixed.append(dict(arm=arm,seed=seed,control=key,
                        flips=int((ss.argmax(1)!=scores.argmax(1)).sum()),**exp.control.metrics(y[rows],ss)))
    summary=pd.DataFrame(records); p=pd.DataFrame(predictions)
    summary.to_csv(ROOT/'summary_by_seed.csv',index=False)
    summary.groupby('arm').agg({v:['mean','std'] for v in ('oa','ba','macro_f1','brier','nll','rescue','harm')}).to_csv(ROOT/'summary_mean_std.csv')
    pd.DataFrame(classes).to_csv(ROOT/'class_recall.csv',index=False)
    pd.DataFrame(fixed).to_csv(ROOT/'pooled_fixed_controls.csv',index=False)
    p.to_csv(ROOT/'all_predictions.csv',index=False)
    return f, p


def uncertainty(p):
    averaged=p.groupby(['arm','obs_id']).agg(correct=('correct','mean'),mmsi=('mmsi','first'),
                                            class_id=('class_id','first')).reset_index()
    results=[]
    for candidate, reference in (('raw','baseline'),('centered','baseline'),('separated','baseline'),
                                 ('centered','raw'),('separated','centered'),('separated','raw')):
        a=averaged[averaged.arm.eq(candidate)].set_index('obs_id')
        b=averaged[averaged.arm.eq(reference)].set_index('obs_id').loc[a.index]
        a['delta']=a.correct-b.correct; a['n']=1
        ships=a.groupby(['mmsi','class_id'])[['delta','n']].sum().reset_index()
        assert ships.mmsi.nunique()==len(ships)
        rng=np.random.default_rng(20261021); ds=[]; ns=[]
        for c in range(8):
            group=ships[ships.class_id.eq(c)]
            ix=rng.integers(len(group),size=(1500,len(group)))
            ds.append(group.delta.to_numpy()[ix].sum(1)); ns.append(group.n.to_numpy()[ix].sum(1))
        ds,ns=np.array(ds).T,np.array(ns).T
        for metric,boots,value in (('oa',ds.sum(1)/ns.sum(1),a.delta.mean()),
                                   ('ba',(ds/ns).mean(1),a.groupby('class_id').delta.mean().mean())):
            low,high=np.quantile(boots,[.025,.975])
            results.append(dict(candidate=candidate,reference=reference,metric=metric,
                difference=value,ci_low=low,ci_high=high,replicates=1500,
                unit='class-stratified MMSI; seed-averaged correctness; fixed models'))
    pd.DataFrame(results).to_csv(ROOT/'paired_uncertainty.csv',index=False)


def pairs(p):
    source=pd.read_csv(ROOT.parent/'condition_outcomes_20261021/crude_pair_details.csv')
    data=[]; summary=[]
    for (arm,seed),group in p.groupby(['arm','seed']):
        q=group.set_index('obs_id')
        for pair in source.itertuples():
            a,b=q.loc[pair.obs_id_a],q.loc[pair.obs_id_b]
            data.append(dict(arm=arm,seed=seed,pair_id=pair.pair_id,obs_id_a=pair.obs_id_a,
                obs_id_b=pair.obs_id_b,both_correct=bool(a.correct and b.correct),
                both_wrong=bool(not a.correct and not b.correct),delta_margin=b.margin-a.margin,
                vv_contrast=pair.vv_contrast,vh_contrast=pair.vh_contrast))
    table=pd.DataFrame(data)
    for (arm,seed),group in table.groupby(['arm','seed']):
        summary.append(dict(arm=arm,seed=seed,pairs=len(group),both_correct=int(group.both_correct.sum()),
            both_wrong=int(group.both_wrong.sum()),vv_rho=spearmanr(group.vv_contrast,group.delta_margin).statistic,
            vh_rho=spearmanr(group.vh_contrast,group.delta_margin).statistic))
    table.to_csv(ROOT/'crude_pairs.csv',index=False)
    pd.DataFrame(summary).to_csv(ROOT/'crude_pair_summary.csv',index=False)


def cases(p):
    frame,views,disabled=exp.control.load_data()
    index=dict(zip(frame.obs_id,frame.index))
    data=p[p.seed.eq(0)].set_index(['arm','obs_id'])
    avg=p.groupby(['arm','obs_id']).agg(correct=('correct','mean'),margin=('margin','mean'),
                                      class_id=('class_id','first')).reset_index()
    a=avg[avg.arm.eq('separated')].set_index('obs_id')
    b=avg[avg.arm.eq('baseline')].set_index('obs_id').loc[a.index]
    selected=[('prior_example_4002',4002),('prior_example_22543',22543)]
    used={4002,22543}
    # All-seed outcome criteria; median margin-change example, not the strongest.
    for label,mask in (('rescue',(a.correct==1)&(b.correct==0)),
                        ('harm',(a.correct==0)&(b.correct==1)),
                        ('persistent_correct',(a.correct==1)&(b.correct==1)),
                        ('persistent_wrong',(a.correct==0)&(b.correct==0))):
        g=a[mask & ~a.index.isin(used)].copy()
        if len(g):
            g['delta']=g.margin-b.loc[g.index].margin
            g=g.sort_values('delta'); oid=int(g.index[len(g)//2])
            selected.append((label,oid));used.add(oid)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    records=[]
    for label,oid in selected:
        i=index[oid]; row=frame.iloc[i]
        canvas=Image.new('RGB',(1010,300),'white'); draw=ImageDraw.Draw(canvas)
        draw.text((10,5),f'{label} | obs {oid} | true class {row.class_id} | {row.port} | native 1:1 pixels',font=font,fill='black')
        for j,title in enumerate(('VV original','VH original','VV R raw encoded','VH R raw encoded','VV R centered','VH R centered')):
            draw.text((10+166*j,28),title,font=small,fill='black')
        for pp,pol in enumerate(('vv','vh')):
            pix,mask,_,_,_=exp.control.prepare_view(row,views[i]['images'][pol],pol)
            x=torch.from_numpy(pix[None]); r=torch.from_numpy(mask[None,2:3])
            raw,mu=exp.encode(x,r,False); centered,_=exp.encode(x,r,True)
            for k,v in enumerate((pix[0],raw[0,0].numpy(),centered[0,0].numpy())):
                im=Image.fromarray(np.rint(v*255).astype(np.uint8)).convert('RGB')
                canvas.paste(im,(10+166*(2*k+pp),48))
            draw.text((10+166*pp,212),f'{pol.upper()} R mean={float(mu[0,0]):.4f}',font=small,fill='black')
        for k,arm in enumerate(('baseline',*exp.ARMS)):
            v=data.loc[(arm,oid)]
            draw.text((10+250*k,235),f'{arm}: {int(v.prediction)} margin {v.margin:+.4f}',font=small,fill='black')
            records.append(dict(case=label,obs_id=oid,seed=0,arm=arm,prediction=int(v.prediction),
                margin=float(v.margin),true_class=int(row.class_id),image_vv=row.image_vv,image_vh=row.image_vh,
                all_seed_correct=avg[(avg.arm==arm)&(avg.obs_id==oid)].correct.iloc[0],vh_disabled=bool(disabled[i])))
        draw.text((10,263),'R excludes accepted target/other OBBs. Grey = neutral fill, NOT sea. PNG level is not calibrated backscatter.',font=small,fill='black')
        canvas.save(ROOT/f'case_{label}.png')
    pd.DataFrame(records).to_csv(ROOT/'case_index.csv',index=False)


def verify():
    f,splits=exp.relation.load_data(); cache=np.load(ROOT/'features.npz')
    rows=[]
    for fold,(tr,val,test) in enumerate(splits):
        matrices,_=exp.standardize(cache,tr)
        for seed in range(3):
            for arm in exp.ARMS:
                saved=torch.load(ROOT/f'head_f{fold}_{arm}_s{seed}.pt',map_location='cpu',weights_only=False)
                model=exp.Residual().eval();model.load_state_dict(saved['state'])
                score=np.load(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz')
                np.testing.assert_array_equal(test,score['rows'])
                with torch.no_grad():
                    replay=score['base']+model(torch.from_numpy(matrices[arm][test])).numpy()
                error=float(np.abs(replay-score['scores']).max())
                assert error<2e-5,(fold,seed,arm,error)
                rows.append(dict(test='chosen_head_replay',fold=fold,seed=seed,arm=arm,max_abs=error))
        assert not set(f.iloc[tr].mmsi)&set(f.iloc[test].mmsi)
        assert not set(f.iloc[val].mmsi)&set(f.iloc[test].mmsi)
        assert not set(f.iloc[tr]['product'])&set(f.iloc[test]['product'])
        assert not set(f.iloc[val]['product'])&set(f.iloc[test]['product'])
    coverage=pd.read_csv(ROOT/'input_coverage.csv')
    assert (coverage.native_target_pixels==coverage.retained_target_pixels).all()
    assert (coverage.input_scale==1).all()
    pd.DataFrame(rows).to_csv(ROOT/'head_replay.csv',index=False)
    (ROOT/'VERIFICATION.json').write_text(json.dumps(dict(status='passed',head_replays=len(rows),
        max_error=max(r['max_abs'] for r in rows),native_scale_preserved=True,
        disjoint_source_target_identities_and_products=True),indent=2),encoding='utf-8')


if __name__=='__main__':
    f,p=evaluate(); uncertainty(p); pairs(p); cases(p); verify()
    print(pd.read_csv(ROOT/'summary_by_seed.csv').to_string(index=False))
