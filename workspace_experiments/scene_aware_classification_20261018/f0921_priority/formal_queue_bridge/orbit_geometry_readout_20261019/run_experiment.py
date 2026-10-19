"""Fixed additive readout controls on the unchanged complete decision path."""
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
COMPARISONS=[('add_orbit','base'),('add_orbit','add_image'),('add_orbit','add_pooled'),
             ('add_image','base'),('add_pooled','base')]


def kernels(features,spatial,support,kg):
    merged,valid=roi.merged_regions(features['prepool'],features['mass12'])
    z,_=bridge.token_descriptor(merged,valid,support)
    grid=bridge.cosine(features['grid2'].reshape(len(merged),2,-1),support).astype(np.float64)
    base=.5*kg+.25*grid+.25*bridge.kernel(z)
    available=spatial['valid']
    output={'base':base}
    for ref in ['pooled','image','orbit']:
        if ref=='pooled':
            tokens=np.tile(merged,(1,1,4,1))
            ok=np.tile(valid,(1,4))&available[:,None]
        else:
            tokens=spatial[ref]
            ok=(spatial[ref+'_mass']>1e-8)&available[:,None]
        z,_=bridge.token_descriptor(tokens,ok,support)
        extra=bridge.kernel(z)
        # Unknown rows contribute no new QS or SS links; old links stay intact.
        np.testing.assert_allclose(extra[~available],0.,atol=0.)
        output['add_'+ref]=base+.0625*extra
    return output


def paired(pred,metrics):
    events,draws,summary=[],[],[]
    for arm,comp in COMPARISONS:
        a=pred[pred.arm.eq(arm)].set_index(KEYS).sort_index()
        b=pred[pred.arm.eq(comp)].set_index(KEYS).sort_index()
        assert a.index.equals(b.index)
        e=a[['truth','radar_valid','margin','same_class_support']].copy()
        e['rescue']=a.correct&~b.correct
        e['harm']=~a.correct&b.correct
        e['delta_margin']=a.margin-b.margin
        e['delta_brier']=a.brier-b.brier
        e['arm'],e['comparator']=arm,comp
        events.append(e.reset_index())
        index=['fold','budget','draw']
        cols=['oa','ba','brier','nll','map_r']
        d=(metrics[metrics.arm.eq(arm)].set_index(index)[cols]
           -metrics[metrics.arm.eq(comp)].set_index(index)[cols]).groupby(['budget','draw']).mean().reset_index()
        d['arm'],d['comparator']=arm,comp
        draws.append(d)
        for budget,g in d.groupby('budget'):
            part=e.reset_index().query('budget==@budget')
            summary.append(dict(arm=arm,comparator=comp,budget=int(budget),
                delta_oa_pp=100*g.oa.mean(),delta_ba_pp=100*g.ba.mean(),
                std_oa_pp=100*g.oa.std(),std_ba_pp=100*g.ba.std(),
                positive_draws=int((g.oa>1e-12).sum()),negative_draws=int((g.oa< -1e-12).sum()),
                rescues=int(part.rescue.sum()),harms=int(part.harm.sum()),
                delta_brier=g.brier.mean(),delta_map=g.map_r.mean()))
    ev=pd.concat(events)
    ev.to_csv(HERE/'paired_events.csv',index=False)
    pd.concat(draws).to_csv(HERE/'paired_draws.csv',index=False)
    result=pd.DataFrame(summary)
    result.to_csv(HERE/'paired_summary.csv',index=False)
    ev.groupby(['arm','comparator','budget','radar_valid'])[['rescue','harm','delta_margin','delta_brier']].agg(['sum','mean','size']).to_csv(HERE/'geometry_strata.csv')
    return result


def main():
    started=time.perf_counter()
    meta=pd.read_csv(PARENT/'cohort.csv',dtype={'date':str,'mmsi':str})
    features=dict(np.load(PARENT/'features.npz'))
    spatial=dict(np.load(HERE/'spatial_features.npz'))
    np.testing.assert_array_equal(spatial['clean_rows'],meta.clean_row)
    tasks=json.loads((PARENT/'episodes.json').read_text())
    y=meta.label.to_numpy()
    records,metrics,traces,timings,cache=[],[],[],[],{}
    with threadpool_limits(limits=1):
        for task in tasks:
            src,q=np.array(task['source']),np.array(task['query'])
            fold,budget=task['fold'],task['budget']
            if fold not in cache:
                g=bridge.global_descriptor(features['global'],src)
                b=bridge.global_descriptor(features['r18'],src)
                kg,kb=g@g.T,b@b.T
                cache[fold]=kg,kb,bridge.prior(kg,src,y,7),bridge.prior(kb,src,y,7)
            kg,kb,pa,pb=cache[fold]
            for draw,rows in enumerate(task['supports']):
                s=np.array(rows)
                tic=time.perf_counter()
                ks=kernels(features,spatial,s,kg)
                expert,_=bridge.adapted(kb,pb,s,y)
                scores,alphas={},{}
                for arm,k in ks.items():
                    score,alpha=bridge.adapted(k,pa,s,y)
                    scores[arm],alphas[arm]=score,alpha
                    p=bridge.late(score[q],expert[q])
                    sim=k[np.ix_(q,s)]
                    ap=np.array([bridge.ap_at_r(v,y[s]==c) for v,c in zip(sim,y[q])])
                    metrics.append(dict(fold=fold,budget=budget,draw=draw,arm=arm,queries=len(q),
                                        **bridge.metrics(p,y[q],ap)))
                    for j,i in enumerate(q):
                        order=np.argsort(-sim[j],kind='stable')
                        def top(c,same=True):
                            return next((int(s[t]) for t in order if (y[s[t]]==c)==same),-1)
                        truth=int(y[i]);pred=int(p[j].argmax())
                        record=dict(fold=fold,budget=budget,draw=draw,arm=arm,row=int(i),
                            clean_row=int(meta.iloc[i].clean_row),truth=truth,prediction=pred,correct=pred==truth,
                            margin=float(p[j,truth]-np.delete(p[j],truth).max()),ap=float(ap[j]),
                            brier=float(np.square(p[j]-np.eye(7)[truth]).sum()),nll=float(-np.log(max(p[j,truth],1e-12))),
                            same_class_support=int((y[s]==truth).sum()),radar_valid=bool(spatial['valid'][i]),
                            top_same_row=top(truth),top_wrong_row=top(truth,False),top_predicted_row=top(pred))
                        record.update({f'p{c}':float(p[j,c]) for c in range(7)})
                        records.append(record)
                for arm,comp in COMPARISONS:
                    old=scores[comp][q].copy();old[np.arange(len(q)),y[q]]=-np.inf
                    rival=old.argmax(1)
                    direct,refit=bridge.margin_parts(ks[comp],ks[arm],alphas[comp],alphas[arm],q,s,y[q],rival)
                    delta=scores[arm][q]-scores[comp][q]
                    observed=delta[np.arange(len(q)),y[q]]-delta[np.arange(len(q)),rival]
                    np.testing.assert_allclose(direct+refit,observed,atol=1e-9)
                    traces.extend(dict(fold=fold,budget=budget,draw=draw,row=int(i),arm=arm,comparator=comp,
                        fixed_rival=int(rival[j]),direct=float(direct[j]),refit=float(refit[j]),
                        r50_margin_change=float(observed[j])) for j,i in enumerate(q))
                timings.append(dict(fold=fold,budget=budget,draw=draw,seconds=time.perf_counter()-tic))
            print('DONE',fold,budget,flush=True)
    pred,met=pd.DataFrame(records),pd.DataFrame(metrics)
    old=pd.read_csv(PARENT/'roi_controls_20261019/predictions.csv',dtype={'fold':str})
    a=pred[pred.arm.eq('base')].set_index(KEYS).sort_index()[roi.PROBS]
    b=old[old.arm.eq('fg_bg_global')].set_index(KEYS).sort_index()[roi.PROBS]
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a,b,atol=2e-12,rtol=0)
    pred.to_csv(HERE/'predictions.csv',index=False)
    met.to_csv(HERE/'metrics.csv',index=False)
    cols=['oa','ba','brier','nll','map_r']
    met.groupby(['budget','arm'])[cols].mean().to_csv(HERE/'summary_equal_date.csv')
    met.groupby(['fold','budget','arm'])[cols].mean().to_csv(HERE/'by_date.csv')
    pred.groupby(['budget','arm','truth'])[['correct','brier','ap']].mean().to_csv(HERE/'by_class.csv')
    pd.DataFrame(traces).to_csv(HERE/'margin_decomposition.csv',index=False)
    pd.DataFrame(timings).to_csv(HERE/'timing.csv',index=False)
    result=paired(pred,met)
    receipt=dict(finished=True,seconds=time.perf_counter()-started,observations=len(meta),classes=7,
                 tasks=len(tasks),support_draws=5,neural_training=False,
                 baseline_probability_error=float(np.abs(a.to_numpy()-b.to_numpy()).max()))
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(met.groupby(['budget','arm'])[cols].mean().to_string())
    print(result.to_string(index=False))


if __name__=='__main__':
    main()
