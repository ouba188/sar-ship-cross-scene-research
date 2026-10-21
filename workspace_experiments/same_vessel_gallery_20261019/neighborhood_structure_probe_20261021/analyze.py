"""Matched and fixed-head summaries, paired differences and native case inputs."""
import json
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
import run as exp

ROOT=exp.ROOT
BODY=False


def evaluate():
    f,splits=exp.old.relation.load_data();y=f.class_id.to_numpy()
    results=[];predictions=[];classes=[];constants=[]
    variants=[('baseline','reference'),('reference','reference')]+[(a,m) for m in ('fixed','matched') for a in exp.ARMS]
    for seed in range(3):
        for arm,mode in variants:
            suffix='reference' if mode=='reference' else arm+'_'+mode
            files=[np.load(ROOT/f'scores_f{k}_{suffix}_s{seed}.npz') for k in range(3)]
            rows=np.concatenate([z['rows'] for z in files])
            assert len(np.unique(rows))==4744
            scores=np.concatenate([z['base' if arm=='baseline' else 'scores'] for z in files])
            base=np.concatenate([z['base'] for z in files])
            reference=np.concatenate([np.load(ROOT/f'scores_f{k}_reference_s{seed}.npz')['scores'] for k in range(3)])
            correct=scores.argmax(1)==y[rows];refcorrect=reference.argmax(1)==y[rows]
            results.append(dict(arm=arm,mode=mode,seed=seed,rescue_vs_D=int((correct&~refcorrect).sum()),
                harm_vs_D=int((~correct&refcorrect).sum()),**exp.old.control.metrics(y[rows],scores)))
            wrong=scores.copy();wrong[np.arange(len(rows)),y[rows]]=-np.inf
            for j,i in enumerate(rows):
                predictions.append(dict(arm=arm,mode=mode,seed=seed,obs_id=int(f.iloc[i].obs_id),
                    mmsi=f.iloc[i].mmsi,class_id=int(y[i]),port=f.iloc[i].port,prediction=int(scores[j].argmax()),
                    correct=bool(correct[j]),margin=float(scores[j,y[i]]-wrong[j].max())))
            for c in range(8):
                take=y[rows]==c
                classes.append(dict(arm=arm,mode=mode,seed=seed,class_id=c,n=int(take.sum()),recall=float(correct[take].mean())))
            if mode=='matched':
                const=np.concatenate([z['source_constant'] for z in files])
                constants.append(dict(arm=arm,seed=seed,**exp.old.control.metrics(y[rows],const)))
    result=pd.DataFrame(results);p=pd.DataFrame(predictions)
    result.to_csv(ROOT/'summary_by_seed.csv',index=False,lineterminator='\n')
    result.groupby(['mode','arm']).agg({k:['mean','std'] for k in ('oa','ba','macro_f1','brier','nll','rescue_vs_D','harm_vs_D')}).to_csv(ROOT/'summary_mean_std.csv',lineterminator='\n')
    pd.DataFrame(classes).to_csv(ROOT/'class_recall.csv',index=False,lineterminator='\n')
    pd.DataFrame(constants).to_csv(ROOT/'source_constant.csv',index=False,lineterminator='\n')
    p.to_csv(ROOT/'predictions.csv',index=False,lineterminator='\n')
    return p


def differences(p):
    avg=p.groupby(['mode','arm','obs_id']).agg(correct=('correct','mean'),mmsi=('mmsi','first'),
                                             class_id=('class_id','first')).reset_index()
    pairs=[('reference','reference','reference','baseline')]
    for mode in ('fixed','matched'):
        for arm in exp.ARMS:
            pairs.append((mode,arm,'reference','reference'))
            if mode=='matched':pairs.append((mode,arm,'reference','baseline'))
        pairs.append((mode,'tile_shuffle',mode,'global_shuffle'))
    results=[]
    for mode,arm,refmode,refarm in pairs:
        a=avg[(avg['mode']==mode)&(avg.arm==arm)].set_index('obs_id').copy()
        b=avg[(avg['mode']==refmode)&(avg.arm==refarm)].set_index('obs_id').loc[a.index]
        a['delta']=a.correct-b.correct;a['n']=1
        ships=a.groupby(['mmsi','class_id'])[['delta','n']].sum().reset_index()
        assert ships.mmsi.nunique()==len(ships)
        rng=np.random.default_rng(20261021);ds=[];ns=[]
        for c in range(8):
            g=ships[ships.class_id==c];ii=rng.integers(len(g),size=(1500,len(g)))
            ds.append(g.delta.to_numpy()[ii].sum(1));ns.append(g.n.to_numpy()[ii].sum(1))
        ds,ns=np.array(ds).T,np.array(ns).T
        for name,boot,point in (('oa',ds.sum(1)/ns.sum(1),a.delta.mean()),
                                ('ba',(ds/ns).mean(1),a.groupby('class_id').delta.mean().mean())):
            low,high=np.quantile(boot,[.025,.975])
            results.append(dict(mode=mode,arm=arm,reference_mode=refmode,reference=refarm,metric=name,
                difference=point,ci_low=low,ci_high=high,replicates=1500,
                scope='fixed trained models; paired-run averaged correctness; class-stratified MMSI'))
    pd.DataFrame(results).to_csv(ROOT/'paired_uncertainty.csv',index=False,lineterminator='\n')


def cases(p):
    frame,views,disabled=exp.old.control.load_data();index=dict(zip(frame.obs_id,frame.index))
    usable=np.load(exp.ROOT/'features.npz')['usable']
    data=p[p.seed.eq(0)].set_index(['mode','arm','obs_id'])
    avg=p.groupby(['mode','arm','obs_id']).agg(correct=('correct','mean'),margin=('margin','mean')).reset_index()
    ref=avg[(avg['mode']=='reference')&(avg.arm=='reference')].set_index('obs_id')
    candidate=avg[(avg['mode']=='matched')&(avg.arm=='tile_shuffle')].set_index('obs_id').loc[ref.index]
    selected=[('prior_4002',4002),('prior_22543',22543),('prior_2952',2952),('prior_10461',10461)]
    used={v for _,v in selected}
    for label,condition in (('tile_rescue',(candidate.correct==1)&(ref.correct==0)),
                            ('tile_harm',(candidate.correct==0)&(ref.correct==1)),
                            ('persistent_correct',(candidate.correct==1)&(ref.correct==1)),
                            ('persistent_wrong',(candidate.correct==0)&(ref.correct==0))):
        g=candidate[condition & ~candidate.index.isin(used)].copy()
        if len(g):
            g['delta']=g.margin-ref.loc[g.index].margin;g=g.sort_values('delta')
            oid=int(g.index[len(g)//2]);selected.append((label,oid));used.add(oid)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    records=[]
    for label,oid in selected:
        i=index[oid];row=frame.iloc[i]
        canvas=Image.new('RGB',(1176 if BODY else 1010,505),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),f'{label} | obs {oid}, true class {row.class_id}, {row.port} | native 1:1 pixels, paired run 0',font=font,fill='black')
        labels=('Original view','Centered R','Half amplitude','Tile shuffle 16px','Global shuffle','Flat + original mu')
        if BODY:labels=('Body baseline input',)+labels
        for j,title in enumerate(labels):draw.text((8+166*j,26),title,font=font,fill='black')
        for pp,pol in enumerate(('vv','vh')):
            pix,masks,_,_,_=exp.old.control.prepare_view(row,views[i]['images'][pol],pol)
            x=pix[0];r=masks[2].astype(bool);mu=float(x[r].mean())
            variants=[x,.5+.5*(x-mu)*r]+[exp.transform(x,r,arm,oid,pp,0) for arm in exp.ARMS]
            if BODY:variants=[x*masks[0]]+variants
            y0=48+185*pp
            for j,z in enumerate(variants):
                canvas.paste(Image.fromarray(np.rint(z*255).astype(np.uint8)).convert('RGB'),(8+166*j,y0))
            draw.text((8,y0+162),f'{pol.upper()} R mean={mu:.4f}, std={float(x[r].std()):.4f}; used={bool(usable[i,pp])}',font=font,fill='black')
        names=[('reference','baseline'),('reference','reference')]+[('matched',a) for a in exp.ARMS]
        for j,(mode,arm) in enumerate(names):
            v=data.loc[(mode,arm,oid)]
            short=('A base','R ref','Half','Tile','Global','Flat')[j]
            draw.text((8+166*j,420),f'{short}: {int(v.prediction)}, {v.margin:+.3f}',font=font,fill='black')
            if mode=='matched':
                q=data.loc[('fixed',arm,oid)]
                draw.text((8+166*j,440),f'fixed R: {int(q.prediction)}, {q.margin:+.3f}',font=font,fill='black')
            records.append(dict(case=label,obs_id=oid,mode=mode,arm=arm,seed=0,prediction=int(v.prediction),
                margin=float(v.margin),class_id=int(row.class_id),image_vv=row.image_vv,image_vh=row.image_vh))
        note=('Body baseline sees x * target BEFORE convolution; original view is for reference only.' if BODY else
              'Whole-chip baseline remains unchanged; these interventions affect only the added R channel.')
        draw.text((8,465),note,font=font,fill='black')
        draw.text((8,485),'Prediction labels below are separate model outputs, not column-aligned inputs. Grey is artificial fill, not water.',font=font,fill='black')
        canvas.save(ROOT/f'case_{label}.png')
    pd.DataFrame(records).to_csv(ROOT/'case_index.csv',index=False,lineterminator='\n')


def verify():
    f,splits=exp.old.relation.load_data();cache=np.load(exp.ROOT/'features.npz');records=[];extremes=[]
    selection=pd.read_csv(ROOT/'validation_selection.csv')
    for fold,(tr,val,test) in enumerate(splits):
        for seed in range(3):
            for arm in (('original',*exp.ARMS) if BODY else exp.ARMS):
                if arm=='original':
                    material,stats=exp.old.standardize(np.load(exp.old.ROOT/'features.npz'),tr)
                    data=material['separated']
                else:
                    data,stats=exp.matrices(cache,exp.feature_key(arm,seed),tr)
                saved=torch.load(ROOT/f'head_f{fold}_{arm}_s{seed}.pt',map_location='cpu',weights_only=False)
                source=selection[(selection.fold==fold)&(selection.seed==seed)&(selection.arm==arm)]
                expected=source.sort_values(['ba','brier','step'],ascending=[False,True,True]).iloc[0]
                assert int(expected.step)==saved['step']
                head=exp.old.Residual().eval();head.load_state_dict(saved['state'])
                scores=np.load(ROOT/f'scores_f{fold}_{arm}_matched_s{seed}.npz')
                np.testing.assert_array_equal(test,scores['rows'])
                with torch.no_grad():
                    cpu=scores['base']+head(torch.from_numpy(data[test])).numpy()
                    # Replay the original all-row GPU evaluation; CPU cancellation is diagnostic only.
                    head=head.cuda()
                    replay=scores['base']+head(torch.tensor(data,device='cuda')).cpu().numpy()[test]
                error=float(np.abs(replay-scores['scores']).max())
                flips=int((replay.argmax(1)!=scores['scores'].argmax(1)).sum())
                np.testing.assert_allclose(replay,scores['scores'],rtol=0,atol=2e-5)
                assert flips==0,(fold,seed,arm,flips)
                assert np.array_equal(cpu.argmax(1),scores['scores'].argmax(1)),(fold,seed,arm,'CPU predictions')
                records.append(dict(fold=fold,seed=seed,arm=arm,source_selection=True,replay_error=error,
                    scaled_error=float((np.abs(replay-scores['scores'])/(1+np.abs(scores['scores']))).max()),
                    prediction_changes=flips,cpu_max_abs=float(np.abs(cpu-scores['scores']).max())))
                for j in np.flatnonzero(np.abs(scores['scores']).max(1)>10):
                    i=test[j];k=int(np.abs(data[i,:256]).argmax());pol,channel=divmod(k,128)
                    extremes.append(dict(fold=fold,seed=seed,arm=arm,obs_id=int(f.iloc[i].obs_id),
                        max_score=float(np.abs(scores['scores'][j]).max()),max_standardized=float(data[i,k]),
                        pol=pol,channel=channel,source_mean=float(stats['centered_mean'][pol,channel]),
                        source_std=float(stats['centered_std'][pol,channel])))
    pd.DataFrame(records).to_csv(ROOT/'verification.csv',index=False,lineterminator='\n')
    pd.DataFrame(extremes).to_csv(ROOT/'extreme_score_diagnostic.csv',index=False,lineterminator='\n')
    (ROOT/'VERIFICATION.json').write_text(json.dumps(dict(status='passed',heads=len(records),
        max_replay_error=max(r['replay_error'] for r in records),
        max_scaled_error=max(r['scaled_error'] for r in records),prediction_changes=0,
        source_selection_checked=True,extreme_rows=len(extremes)),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--body',action='store_true');args=parser.parse_args()
    BODY=args.body
    if BODY:ROOT=exp.ROOT/'body_only'
    p=evaluate();differences(p);cases(p);verify()
    print(pd.read_csv(ROOT/'summary_by_seed.csv').groupby(['mode','arm'])[['oa','ba','macro_f1']].mean().to_string())
