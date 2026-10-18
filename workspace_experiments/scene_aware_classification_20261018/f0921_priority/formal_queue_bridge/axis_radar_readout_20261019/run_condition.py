"""One posthoc conditional relation test, preserving the first frozen outputs."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_axis as ev

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent


def geometry_kernels(axes,geo):
    valid=geo.geometry_valid.to_numpy(bool)&(axes.axis_reliability.to_numpy()>1e-8)
    ix=np.flatnonzero(valid)
    missing=np.flatnonzero(~valid)
    availability=(valid[:,None]==valid[None,:]).astype(float)
    angle=np.zeros_like(availability)
    joint=np.zeros_like(availability)
    angle[np.ix_(missing,missing)]=1.
    joint[np.ix_(missing,missing)]=1.
    relative=(axes.ship_axis_rad.to_numpy()-geo.radar_angle_image_rad.to_numpy())[ix]
    incidence=geo.incidence_deg.to_numpy()[ix]
    angular=np.exp(.5*(np.cos(2*(relative[:,None]-relative[None,:]))-1))
    angle[np.ix_(ix,ix)]=angular
    joint[np.ix_(ix,ix)]=angular*np.exp(-.5*((incidence[:,None]-incidence[None,:])/10)**2)
    return availability,angle,joint,valid


def conditional_local(features,directions,axes,support,geometry):
    availability,angle,joint,_=geometry
    local,z=ev.kernels(features,directions,axes,support,return_descriptors=True)
    ks=ev.bridge.kernel(z['ship'])
    base=.5*local['fg_bg_global']+.25*ks
    local.update(availability_control=base+.25*ks*availability,
                 conditioned_ship=base+.25*ks*joint,
                 relative_angle=base+.25*ks*angle,
                 geometry_only=base+.25*joint)
    return local


def main():
    started=time.perf_counter()
    meta=pd.read_csv(PARENT/'cohort.csv',dtype={'date':str,'mmsi':str})
    f=dict(np.load(PARENT/'features.npz'))
    d=dict(np.load(HERE/'direction_features.npz'))
    axes=pd.read_csv(HERE/'axes.csv')
    geo=pd.read_csv(HERE/'geometry.csv')
    availability,angle,joint,valid=geometry_kernels(axes,geo)
    tasks=json.loads((PARENT/'episodes.json').read_text())
    y=meta.label.to_numpy()
    records,metrics,cache=[],[],{}
    eigen={name:float(np.linalg.eigvalsh(k).min()) for name,k in [('availability',availability),('angle',angle),('joint',joint)]}
    assert min(eigen.values())>-1e-9
    with threadpool_limits(limits=1):
        for task in tasks:
            src,q=np.array(task['source']),np.array(task['query'])
            fold,budget=task['fold'],task['budget']
            if fold not in cache:
                g=ev.bridge.global_descriptor(f['global'],src)
                b=ev.bridge.global_descriptor(f['r18'],src)
                kg,kb=g@g.T,b@b.T
                cache[fold]=kg,kb,ev.bridge.prior(kg,src,y,7),ev.bridge.prior(kb,src,y,7)
            kg,kb,pa,pb=cache[fold]
            for draw,rows in enumerate(task['supports']):
                s=np.array(rows)
                local=conditional_local(f,d,axes,s,(availability,angle,joint,valid))
                new={name:local[name] for name in ['availability_control','conditioned_ship','relative_angle','geometry_only']}
                grid=ev.bridge.cosine(f['grid2'].reshape(len(meta),2,-1),s).astype(np.float64)
                expert,_=ev.bridge.adapted(kb,pb,s,y)
                for arm,k in new.items():
                    k=.5*kg+.25*grid+.25*k
                    score,_=ev.bridge.adapted(k,pa,s,y)
                    prob=ev.bridge.late(score[q],expert[q])
                    sim=k[np.ix_(q,s)]
                    ap=np.array([ev.bridge.ap_at_r(v,y[s]==c) for v,c in zip(sim,y[q])])
                    metrics.append(dict(fold=fold,budget=budget,draw=draw,arm=arm,queries=len(q),**ev.bridge.metrics(prob,y[q],ap)))
                    for j,i in enumerate(q):
                        order=np.argsort(-sim[j],kind='stable')
                        def top(c,same=True):
                            return next((int(s[t]) for t in order if (y[s[t]]==c)==same),-1)
                        p,truth=prob[j],int(y[i]);prediction=int(p.argmax())
                        record=dict(fold=fold,budget=budget,draw=draw,arm=arm,row=int(i),clean_row=int(meta.iloc[i].clean_row),
                            truth=truth,prediction=prediction,correct=prediction==truth,
                            margin=float(p[truth]-np.delete(p,truth).max()),ap=float(ap[j]),
                            brier=float(np.square(p-np.eye(7)[truth]).sum()),nll=float(-np.log(max(p[truth],1e-12))),
                            top_same_row=top(truth),top_wrong_row=top(truth,False),top_predicted_row=top(prediction),
                            same_class_support=int((y[s]==truth).sum()),radar_valid=bool(valid[i]),
                            axis_reliability=float(axes.iloc[i].axis_reliability))
                        record.update({f'p{c}':float(p[c]) for c in range(7)})
                        records.append(record)
            print('CONDITION',fold,budget,flush=True)
    pred,met=pd.DataFrame(records),pd.DataFrame(metrics)
    pred.to_csv(HERE/'condition_predictions.csv',index=False)
    met.to_csv(HERE/'condition_metrics.csv',index=False)
    cols=['oa','ba','brier','nll','map_r']
    met.groupby(['budget','arm'])[cols].mean().to_csv(HERE/'condition_summary.csv')
    met.groupby(['fold','budget','arm'])[cols].mean().to_csv(HERE/'condition_by_date.csv')
    pred.groupby(['budget','arm','truth'])[['correct','brier','ap']].mean().to_csv(HERE/'condition_by_class.csv')
    allpred=pd.concat([pd.read_csv(HERE/'predictions.csv',dtype={'fold':str}),pred])
    allmet=pd.concat([pd.read_csv(HERE/'metrics.csv',dtype={'fold':str}),met])
    events,draws,summaries=[],[],[]
    comparisons=[('conditioned_ship',c) for c in ['availability_control','ship_soft','fg_bg_global','relative_angle','geometry_only']]
    comparisons += [('relative_angle','availability_control'),('geometry_only','fg_bg_global')]
    for arm,comp in comparisons:
        a=allpred[allpred.arm.eq(arm)].set_index(ev.KEYS).sort_index()
        b=allpred[allpred.arm.eq(comp)].set_index(ev.KEYS).sort_index()
        event=a[['truth','radar_valid']].copy()
        event['rescue']=a.correct&~b.correct;event['harm']=~a.correct&b.correct
        event['delta_margin']=a.margin-b.margin;event['delta_brier']=a.brier-b.brier
        event['arm'],event['comparator']=arm,comp;events.append(event.reset_index())
        index=['fold','budget','draw']
        a=allmet[allmet.arm.eq(arm)].set_index(index)[cols]
        b=allmet[allmet.arm.eq(comp)].set_index(index)[cols]
        delta=(a-b).groupby(['budget','draw']).mean().reset_index()
        delta['arm'],delta['comparator']=arm,comp;draws.append(delta)
        for budget,g in delta.groupby('budget'):
            e=event.reset_index().query('budget==@budget')
            summaries.append(dict(arm=arm,comparator=comp,budget=int(budget),delta_oa_pp=100*g.oa.mean(),delta_ba_pp=100*g.ba.mean(),
                std_oa_pp=100*g.oa.std(),std_ba_pp=100*g.ba.std(),positive_draws=int((g.oa>1e-12).sum()),negative_draws=int((g.oa < -1e-12).sum()),
                rescues=int(e.rescue.sum()),harms=int(e.harm.sum()),delta_brier=g.brier.mean(),delta_map=g.map_r.mean()))
    pd.concat(events).to_csv(HERE/'condition_events.csv',index=False)
    pd.concat(draws).to_csv(HERE/'condition_draws.csv',index=False)
    pd.DataFrame(summaries).to_csv(HERE/'condition_paired.csv',index=False)
    (HERE/'condition_receipt.json').write_text(json.dumps(dict(finished=True,posthoc=True,
        seconds=time.perf_counter()-started,geometry_valid=int(valid.sum()),kernel_min_eigenvalues=eigen,
        new_training=False,incidence_scale_degrees=10,weight_search=False),indent=2))
    print(met.groupby(['budget','arm'])[cols].mean().to_string())
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__=='__main__':
    main()
