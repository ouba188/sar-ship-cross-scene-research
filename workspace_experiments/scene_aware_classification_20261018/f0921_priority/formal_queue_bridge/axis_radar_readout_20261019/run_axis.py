"""Frozen full-decision-path evaluation; no target selection or neural fitting."""
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location('roi_controls',PARENT/'roi_controls_20261019/controls.py')
roi=importlib.util.module_from_spec(spec)
spec.loader.exec_module(roi)
bridge=roi.bridge
KEYS=roi.KEYS
COMPARISONS=[('ship_soft','image_soft'),('ship_soft','fg_bg_global'),
             ('ship_radar','ship_soft'),('ship_radar','ship_image'),
             ('ship_radar','fg_bg_global'),('ship_radar','global_only'),
             ('ship_radar','global_grid'),('ship_radar','target_only')]


def kernels(features,directions,axes,support,return_descriptors=False):
    old=roi.local_kernels(features,support)
    n=len(axes)
    merged,valid=roi.merged_regions(features['prepool'],features['mass12'])
    fg,_=bridge.token_descriptor(merged,valid,support)
    repeat=np.tile(fg,(1,1,4,1))/2
    descriptors={}
    no_axis=axes.axis_reliability.to_numpy()<1e-8
    for ref in ['image','ship']:
        z,_=bridge.token_descriptor(directions[ref],directions[ref+'_mass']>1e-8,support)
        z[no_axis]=repeat[no_axis]
        descriptors[ref]=z
    available=directions['radar_valid']
    for ref,name in [('radar','radar'),('image','image_matched')]:
        valid=(directions[ref+'_mass']>1e-8)&available[:,None]&~no_axis[:,None]
        z,_=bridge.token_descriptor(directions[ref],valid,support)
        # If the support bank has no observed radar geometry, there is no fitted
        # radar coordinate reference; fall back for the complete task.
        if not np.any(valid[support]):
            z=descriptors['ship'].copy()
        else:
            z[~available|no_axis]=descriptors['ship'][~available|no_axis]
        descriptors[name]=z
    k={name:bridge.kernel(z) for name,z in descriptors.items()}
    local=dict(fg_bg_global=old['fg_bg_global'],target_only=old['target_only'],
        image_soft=.5*old['fg_bg_global']+.5*k['image'],
        ship_soft=.5*old['fg_bg_global']+.5*k['ship'],
        ship_image=.5*old['fg_bg_global']+.25*k['ship']+.25*k['image_matched'],
        ship_radar=.5*old['fg_bg_global']+.25*k['ship']+.25*k['radar'])
    return (local,descriptors) if return_descriptors else local


def paired(pred,met):
    events,draws,summary=[],[],[]
    for arm,comparator in COMPARISONS:
        a=pred[pred.arm.eq(arm)].set_index(KEYS).sort_index()
        b=pred[pred.arm.eq(comparator)].set_index(KEYS).sort_index()
        assert a.index.equals(b.index)
        event=a[['truth','axis_reliability','radar_valid','same_class_support']].copy()
        event['rescue']=a.correct&~b.correct
        event['harm']=~a.correct&b.correct
        for name in ['margin','ap','brier']:
            event['delta_'+name]=a[name]-b[name]
        event['arm'],event['comparator']=arm,comparator
        events.append(event.reset_index())
        index=['fold','budget','draw']
        ma=met[met.arm.eq(arm)].set_index(index)
        mb=met[met.arm.eq(comparator)].set_index(index)
        delta=(ma[['oa','ba','brier','nll','map_r']]-mb[['oa','ba','brier','nll','map_r']])
        delta=delta.groupby(['budget','draw']).mean().reset_index()
        delta['arm'],delta['comparator']=arm,comparator
        draws.append(delta)
        for budget,g in delta.groupby('budget'):
            e=event.reset_index().query('budget==@budget')
            summary.append(dict(arm=arm,comparator=comparator,budget=int(budget),
                delta_oa_pp=100*g.oa.mean(),delta_ba_pp=100*g.ba.mean(),
                std_oa_pp=100*g.oa.std(),std_ba_pp=100*g.ba.std(),
                positive_draws=int((g.oa>1e-12).sum()),negative_draws=int((g.oa < -1e-12).sum()),
                rescues=int(e.rescue.sum()),harms=int(e.harm.sum()),
                delta_brier=g.brier.mean(),delta_map=g.map_r.mean()))
    ev=pd.concat(events)
    ev.to_csv(HERE/'paired_events.csv',index=False)
    pd.concat(draws).to_csv(HERE/'paired_draws.csv',index=False)
    table=pd.DataFrame(summary)
    table.to_csv(HERE/'paired_summary.csv',index=False)
    ev.groupby(['arm','comparator','budget','radar_valid'])[['rescue','harm','delta_margin','delta_brier']].agg(['sum','mean','size']).to_csv(HERE/'geometry_strata.csv')
    ev.groupby(['arm','comparator','budget','truth'])[['rescue','harm']].sum().to_csv(HERE/'class_flips.csv')
    return table


def main():
    started=time.perf_counter()
    meta=pd.read_csv(PARENT/'cohort.csv',dtype={'date':str,'mmsi':str})
    f=dict(np.load(PARENT/'features.npz'))
    directions=dict(np.load(HERE/'direction_features.npz'))
    axes=pd.read_csv(HERE/'axes.csv')
    np.testing.assert_array_equal(directions['clean_rows'],meta.clean_row)
    tasks=json.loads((PARENT/'episodes.json').read_text())
    y=meta.label.to_numpy()
    records,metrics,trace,timing=[],[],[],[]
    cache={}
    with threadpool_limits(limits=1):
        for task in tasks:
            src,q=np.array(task['source']),np.array(task['query'])
            fold,budget=task['fold'],task['budget']
            if fold not in cache:
                g=bridge.global_descriptor(f['global'],src)
                b=bridge.global_descriptor(f['r18'],src)
                kg,kb=g@g.T,b@b.T
                cache[fold]=kg,kb,bridge.prior(kg,src,y,7),bridge.prior(kb,src,y,7)
            kg,kb,pa,pb=cache[fold]
            for draw,rows in enumerate(task['supports']):
                s=np.array(rows)
                tic=time.perf_counter()
                grid=bridge.cosine(f['grid2'].reshape(len(meta),2,-1),s).astype(np.float64)
                ks=dict(global_only=kg,global_grid=.5*kg+.5*grid)
                ks.update({name:.5*kg+.25*grid+.25*k for name,k in kernels(f,directions,axes,s).items()})
                expert,_=bridge.adapted(kb,pb,s,y)
                scores,alphas={},{}
                for arm,k in ks.items():
                    score,alpha=bridge.adapted(k,pa,s,y)
                    scores[arm],alphas[arm]=score,alpha
                    prob=bridge.late(score[q],expert[q])
                    sim=k[np.ix_(q,s)]
                    ap=np.array([bridge.ap_at_r(v,y[s]==c) for v,c in zip(sim,y[q])])
                    metrics.append(dict(fold=fold,budget=budget,draw=draw,arm=arm,queries=len(q),**bridge.metrics(prob,y[q],ap)))
                    for j,i in enumerate(q):
                        order=np.argsort(-sim[j],kind='stable')
                        def top(c,same=True):
                            return next((int(s[t]) for t in order if (y[s[t]]==c)==same),-1)
                        p,truth=prob[j],int(y[i])
                        pred=int(p.argmax())
                        record=dict(fold=fold,budget=budget,draw=draw,arm=arm,row=int(i),clean_row=int(meta.iloc[i].clean_row),
                            truth=truth,prediction=pred,correct=pred==truth,margin=float(p[truth]-np.delete(p,truth).max()),
                            ap=float(ap[j]),brier=float(np.square(p-np.eye(7)[truth]).sum()),nll=float(-np.log(max(p[truth],1e-12))),
                            top_same_row=top(truth),top_wrong_row=top(truth,False),top_predicted_row=top(pred),
                            same_class_support=int((y[s]==truth).sum()),radar_valid=bool(directions['radar_valid'][i]),
                            axis_reliability=float(axes.iloc[i].axis_reliability))
                        record.update({f'p{c}':float(p[c]) for c in range(7)})
                        records.append(record)
                for arm,comp in COMPARISONS[:5]:
                    rival_score=scores[comp][q].copy()
                    rival_score[np.arange(len(q)),y[q]]=-np.inf
                    rival=rival_score.argmax(1)
                    direct,refit=bridge.margin_parts(ks[comp],ks[arm],alphas[comp],alphas[arm],q,s,y[q],rival)
                    diff=scores[arm][q]-scores[comp][q]
                    observed=diff[np.arange(len(q)),y[q]]-diff[np.arange(len(q)),rival]
                    np.testing.assert_allclose(direct+refit,observed,atol=1e-9)
                    trace.extend(dict(fold=fold,budget=budget,draw=draw,row=int(i),arm=arm,comparator=comp,
                        fixed_rival=int(rival[j]),direct_kernel=float(direct[j]),support_refit=float(refit[j]),
                        r50_margin_change=float(observed[j])) for j,i in enumerate(q))
                timing.append(dict(fold=fold,budget=budget,draw=draw,seconds=time.perf_counter()-tic))
            print('DONE',fold,budget,flush=True)
    pred,met=pd.DataFrame(records),pd.DataFrame(metrics)
    previous=pd.read_csv(PARENT/'roi_controls_20261019/predictions.csv',dtype={'fold':str})
    errors={}
    for arm in ['global_only','global_grid','target_only','fg_bg_global']:
        a=pred[pred.arm.eq(arm)].set_index(KEYS).sort_index()[roi.PROBS]
        b=previous[previous.arm.eq(arm)].set_index(KEYS).sort_index()[roi.PROBS]
        assert a.index.equals(b.index)
        np.testing.assert_allclose(a,b,atol=2e-12,rtol=0)
        errors[arm]=float(np.abs(a.to_numpy()-b.to_numpy()).max())
    pred.to_csv(HERE/'predictions.csv',index=False)
    met.to_csv(HERE/'metrics.csv',index=False)
    cols=['oa','ba','brier','nll','map_r']
    met.groupby(['budget','arm'])[cols].mean().to_csv(HERE/'summary_equal_date.csv')
    met.groupby(['fold','budget','arm'])[cols].mean().to_csv(HERE/'by_date.csv')
    pred.groupby(['budget','arm','truth'])[['correct','brier','ap']].mean().to_csv(HERE/'by_class.csv')
    weighted=pred.groupby(['budget','arm'])[['correct','brier','ap']].mean()
    weighted['ba']=pred.groupby(['budget','arm','truth']).correct.mean().groupby(['budget','arm']).mean()
    weighted.to_csv(HERE/'summary_query_weighted.csv')
    pd.DataFrame(trace).to_csv(HERE/'margin_decomposition.csv',index=False)
    pd.DataFrame(timing).to_csv(HERE/'timing.csv',index=False)
    paired_table=paired(pred,met)
    (HERE/'receipt.json').write_text(json.dumps(dict(finished=True,seconds=time.perf_counter()-started,
        observations=len(meta),classes=7,tasks=len(tasks),support_draws=5,
        new_neural_training=False,source_priors_refit=True,old_probability_replay=errors),indent=2))
    print(met.groupby(['budget','arm'])[cols].mean().to_string())
    print(paired_table.to_string(index=False))


if __name__=='__main__':
    main()
