"""Fixed paired summaries, cluster intervals and native-chip readouts."""
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from run_transfer import HERE, PARENT, BASELINE, ARMS, SEEDS, metrics, per_row, load_frame

ROOT=PARENT.parents[1]
sys.path.insert(0,str(ROOT/'same_vessel_gallery_20261019/local_condition_module'))
from native_input import crop_pad
from prepare import masks_for


def paired_intervals(p,b):
    ids=b.instance_id.to_numpy()
    y=b.class_id.to_numpy()
    truth={}
    truth['A']=np.stack(per_row(y,b[[f's{c}' for c in range(8)]].to_numpy()),axis=1)
    for arm in ARMS:
        values=[]
        for seed in SEEDS:
            part=p[p.arm.eq(arm)&p.seed.eq(seed)].set_index('instance_id').loc[ids]
            values.append(np.stack(per_row(y,part[[f's{c}' for c in range(8)]].to_numpy()),axis=1))
        truth[arm]=np.mean(values,axis=0)
    pairs=[(arm,'A') for arm in ARMS]+[('score_scale_decoupled','original_centered'),
        ('original_centered','additive'),('score_scale_decoupled','additive')]
    records=[]
    for block in ('mmsi','pass_group'):
        groups,inverse=np.unique(b[block].to_numpy(),return_inverse=True)
        counts=np.zeros((len(groups),8))
        np.add.at(counts,(inverse,y),1)
        rng=np.random.default_rng(20261023)
        # Identical resamples for all comparisons; seeds are averaged per row.
        samples=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=2000)
        denominators=samples@counts
        valid=(denominators>0).all(1)
        assert valid.any()
        for arm,reference in pairs:
            diff=truth[arm]-truth[reference]
            for j,ordinary,balanced,mult in [(0,'OA','BA',100.),(1,'NLL','balanced_NLL',1.),(2,'Brier','balanced_Brier',1.)]:
                sums=np.zeros_like(counts)
                np.add.at(sums,(inverse,y),diff[:,j])
                numerators=samples@sums
                dist=(numerators.sum(1)/denominators.sum(1))*mult
                bdist=(numerators[valid]/denominators[valid]).mean(1)*mult
                for metric,values,point in [(ordinary,dist,diff[:,j].mean()*mult),
                    (balanced,bdist,np.mean([diff[y==c,j].mean() for c in range(8)])*mult)]:
                    lo,hi=np.quantile(values,[.025,.975])
                    records.append(dict(arm=arm,reference=reference,block=block,metric=metric,delta=point,
                        low=lo,high=hi,resamples=len(values),clusters=len(groups)))
    pd.DataFrame(records).to_csv(HERE/'paired_intervals.csv',index=False)


def tables(p,b):
    rows=[]
    for (arm,seed,c),part in p.groupby(['arm','seed','class_id']):
        y=part.class_id.to_numpy()
        new=metrics(y,part[[f's{k}' for k in range(8)]].to_numpy())
        old=metrics(y,part[[f'b{k}' for k in range(8)]].to_numpy())
        rows.append(dict(arm=arm,seed=seed,class_id=c,class_name=part.class_name.iloc[0],
            rescued=int(part.net.eq(1).sum()),harmed=int(part.net.eq(-1).sum()),net=int(part.net.sum()),
            **new,**{'base_'+k:v for k,v in old.items()},**{'delta_'+k:new[k]-old[k] for k in ('OA','NLL','Brier')}))
    pd.DataFrame(rows).to_csv(HERE/'class_metrics.csv',index=False)
    for kind,keys in [('acquisition',['port','pass_group']),('vh',['vh_valid']),('month',['month'])]:
        rows=[]
        for group,part in p.groupby(['arm','seed']+keys):
            y=part.class_id.to_numpy()
            new=metrics(y,part[[f's{k}' for k in range(8)]].to_numpy())
            old=metrics(y,part[[f'b{k}' for k in range(8)]].to_numpy())
            rows.append(dict(zip(['arm','seed']+keys,group))|dict(net=int(part.net.sum()),
                rescued=int(part.net.eq(1).sum()),harmed=int(part.net.eq(-1).sum()),**new,
                **{'base_'+k:v for k,v in old.items()},**{'delta_'+k:new[k]-old[k] for k in ('OA','BA','NLL','Brier')}))
        pd.DataFrame(rows).to_csv(HERE/(kind+'_metrics.csv'),index=False)
    pooled=pd.read_csv(HERE/'pooled.csv')
    numeric=['OA','BA','NLL','balanced_NLL','Brier','balanced_Brier','net','rescued','harmed']
    pooled.groupby('arm')[numeric].mean().reset_index().to_csv(HERE/'mean_metrics.csv',index=False)
    paired_intervals(p,b)


def controls(f):
    rows=[]
    for fold in range(3):
        with np.load(BASELINE/f'fold{fold}_inputs.npz') as d:
            test=d['test'];q=d['q'][test];base=d['base'][test]
        for arm in ARMS:
            for seed in SEEDS:
                with np.load(HERE/'readouts'/f'fold{fold}_{arm}_{seed}.npz') as r:
                    a=r['control'][test].astype(np.float64)
                if arm=='additive':
                    rows.append(dict(fold=fold,arm=arm,seed=seed,group=-1,mean=a.mean(),std=a.std(),
                        min=a.min(),max=a.max(),common_mean=np.nan,contrast_rms=np.nan))
                else:
                    m=a.mean(1,keepdims=True)
                    for g in range(16):
                        rows.append(dict(fold=fold,arm=arm,seed=seed,group=g,mean=a[:,g].mean(),std=a[:,g].std(),
                            min=a[:,g].min(),max=a[:,g].max(),common_mean=m.mean(),
                            contrast_rms=np.sqrt(np.mean((a-m)**2))))
    pd.DataFrame(rows).to_csv(HERE/'control_statistics.csv',index=False)


def select_cases(p):
    cases=[]
    for arm in ARMS:
        for outcome in ('rescued','harmed','CC','WW'):
            chosen=None
            for seed in SEEDS:
                part=p[p.arm.eq(arm)&p.seed.eq(seed)&p.outcome.eq(outcome)]
                if len(part):
                    chosen=part.sort_values('obs_id').iloc[0].to_dict()
                    chosen['requested_arm']=arm
                    break
            if chosen is None:
                for fallback in ARMS:
                    for seed in SEEDS:
                        part=p[p.arm.eq(fallback)&p.seed.eq(seed)&p.outcome.eq(outcome)]
                        if len(part):
                            chosen=part.sort_values('obs_id').iloc[0].to_dict()
                            chosen['requested_arm']=arm
                            break
                    if chosen is not None:
                        break
            if chosen is not None:
                cases.append(chosen)
    return pd.DataFrame(cases)


def review(f,p):
    cases=select_cases(p)
    snapshots={s['obs_id']:s for s in json.loads((ROOT/'same_vessel_gallery_20261019/local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    names=f.groupby('class_id').class_name.first().to_dict()
    (HERE/'review').mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    records=[];groups=[]
    for row in cases.itertuples():
        fold=int(row.fold);i=int(row.row_index);truth=int(row.class_id)
        with np.load(BASELINE/f'fold{fold}_inputs.npz') as d:
            base=d['base'][i];q=d['q'][i];valid=d['valid'][i];whole=d['whole64'][i]
        with np.load(HERE/'readouts'/f'fold{fold}_{row.arm}_{row.seed}.npz') as r:
            a=r['control'][i].astype(np.float64);score=r['score'][i]
        opponent=int(row.base_prediction) if row.outcome=='rescued' else int(row.prediction)
        if opponent==truth:
            s=score.copy();s[truth]=-np.inf;opponent=int(s.argmax())
        before=float(base[truth]-base[opponent]);after=float(score[truth]-score[opponent])
        if row.arm=='additive':
            common=0.;contrast=float(a[truth]-a[opponent]);m=0.
            np.testing.assert_allclose(before+contrast,after,atol=1e-10)
        else:
            m=float(a.mean());dq=q[:,truth]-q[:,opponent]
            if row.arm=='original_centered':
                terms=a*dq;common=float(m*dq.sum());contrast=float(((a-m)*dq).sum())
            else:
                terms=(1+m)*(a-m)*dq;common=m*before;contrast=float(terms.sum())
            np.testing.assert_allclose(before+common+contrast,after,atol=1e-10)
            for g in range(16):
                groups.append(dict(arm=row.arm,seed=row.seed,fold=fold,instance_id=row.instance_id,outcome=row.outcome,
                    truth=truth,opponent=opponent,group=g,pol='VV' if g<8 else 'VH',pol_valid=bool(valid[g//8]),
                    q_margin=float(dq[g]),a=float(a[g]),m=m,margin_increment=float(terms[g])))
        name=f'{row.requested_arm}_{row.outcome}_{row.instance_id}.png'
        panel=HERE/'review'/name
        canvas=Image.new('RGB',(748,316),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),f'{row.arm} | seed {row.seed} | {row.outcome} | fold {fold}',font=font,fill='black')
        draw.text((8,23),f'{row.instance_id} | obs {row.obs_id} | {row.port}',font=font,fill='black')
        draw.text((8,41),f'Truth {names[truth]} | A {names[int(row.base_prediction)]} -> new {names[int(row.prediction)]}',font=font,fill='black')
        draw.text((8,59),f'Margin vs {names[opponent]}: {before:+.4f} -> {after:+.4f}; m={m:+.4f}',font=font,fill='black')
        for pol,key in enumerate(('vv','vh')):
            src=f.iloc[i]['image_'+key]
            original=np.asarray(Image.open(src).convert('L'))
            pixels,masks,place=crop_pad(original,masks_for(snapshots[int(row.obs_id)]['images'][key]),160)
            support=masks.sum(0)>0
            rgb=np.repeat(pixels[...,None],3,-1).astype(float)
            tint=np.zeros_like(rgb);tint[support]=[40,220,255]
            opacity=.25*support[...,None]
            overlay=np.uint8(rgb*(1-opacity)+tint*opacity)
            x=8+pol*370
            draw.text((x,81),key.upper()+(' used' if valid[pol] else ' NOT USED'),font=font,fill='black' if valid[pol] else '#b91c1c')
            draw.text((x+178,81),'Whole valid footprint',font=font,fill='black')
            canvas.paste(Image.fromarray(pixels).convert('RGB'),(x,101))
            canvas.paste(Image.fromarray(overlay),(x+178,101))
            records.append(dict(requested_arm=row.requested_arm,arm=row.arm,seed=row.seed,fold=fold,
                instance_id=row.instance_id,obs_id=int(row.obs_id),outcome=row.outcome,truth=truth,
                truth_name=names[truth],base_prediction=int(row.base_prediction),prediction=int(row.prediction),
                opponent=opponent,margin_before=before,margin_after=after,common_increment=common,
                contrast_increment=contrast,m=m,whole64_norm=float(np.linalg.norm(whole)),pol=key,
                pol_valid=bool(valid[pol]),image_path=src,panel_path=str(panel),**place))
        draw.text((8,271),'Cyan: actual whole pooling area; black padding excluded only at pooling.',font=font,fill='black')
        draw.text((8,289),'Original pixels; no resize. Conv receptive fields can mix target, context and padding.',font=font,fill='black')
        canvas.save(panel)
    pd.DataFrame(records).to_csv(HERE/'review_cases.csv',index=False)
    pd.DataFrame(groups).to_csv(HERE/'review_group_contributions.csv',index=False)
    return len(cases)


def main():
    started=time.perf_counter()
    f=load_frame()
    p=pd.read_csv(HERE/'predictions.csv',dtype={'mmsi':str})
    p['month']=p.time.str[:7]
    b=pd.read_csv(HERE/'baseline_predictions.csv',dtype={'mmsi':str}).sort_values('row_index')
    assert b.instance_id.is_unique and len(b)==4744
    tables(p,b)
    controls(f)
    cases=review(f,p)
    summary=dict(status='analysis_tables_complete',cases=cases,bootstrap_resamples=2000,
        bootstrap_blocks=['mmsi','pass_group'],seconds=time.perf_counter()-started)
    (HERE/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.read_csv(HERE/'mean_metrics.csv').to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
