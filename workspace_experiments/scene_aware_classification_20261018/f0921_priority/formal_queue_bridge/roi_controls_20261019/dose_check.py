"""Posthoc diagnostic: remove only the background term, keep foreground dose."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from controls import HERE,PARENT,bridge,merged_regions,KEYS,PROBS


def main():
    f=dict(np.load(PARENT/'features.npz'));meta=pd.read_csv(PARENT/'cohort.csv')
    y=meta.label.to_numpy();tasks=json.loads((PARENT/'episodes.json').read_text())
    pooled,valid=merged_regions(f['prepool'],f['mass12'])
    original=pd.read_csv(HERE/'predictions.csv',dtype={'fold':str})
    output=[];metrics=[];cache={};error=0.
    with threadpool_limits(limits=1):
        for task in tasks:
            fold,budget=task['fold'],task['budget'];sour=np.array(task['source']);q=np.array(task['query'])
            if fold not in cache:
                a=bridge.global_descriptor(f['global'],sour);b=bridge.global_descriptor(f['r18'],sour)
                ka,kb=a@a.T,b@b.T
                cache[fold]=ka,kb,bridge.prior(ka,sour,y,7),bridge.prior(kb,sour,y,7)
            ka,kb,pa,pb=cache[fold]
            for draw,rows in enumerate(task['supports']):
                s=np.array(rows);grid=bridge.cosine(f['grid2'].reshape(len(meta),2,-1),s)
                z,_=bridge.token_descriptor(pooled,valid,s)
                kfg=bridge.kernel(z[:,:,:1]);kbg=bridge.kernel(z[:,:,1:])
                np.testing.assert_allclose(kfg+kbg,bridge.kernel(z),atol=1e-12)
                expert,_=bridge.adapted(kb,pb,s,y)
                for arm,k in [('fg_dose_only',.5*ka+.25*grid+.25*kfg),
                              ('fg_bg_global',.5*ka+.25*grid+.25*(kfg+kbg))]:
                    value,_=bridge.adapted(k,pa,s,y);prob=bridge.late(value[q],expert[q])
                    sim=k[np.ix_(q,s)]
                    ap=np.array([bridge.ap_at_r(v,y[s]==label) for v,label in zip(sim,y[q])])
                    metrics.append(dict(fold=fold,budget=budget,draw=draw,arm=arm,**bridge.metrics(prob,y[q],ap)))
                    if arm=='fg_bg_global':
                        ref=original[(original.fold==fold)&original.budget.eq(budget)&original.draw.eq(draw)&original.arm.eq(arm)].set_index('row').loc[q,PROBS]
                        error=max(error,float(np.max(np.abs(prob-ref.to_numpy()))))
                        np.testing.assert_allclose(prob,ref,atol=1e-12,rtol=0)
                    for j,i in enumerate(q):
                        p=prob[j];truth=int(y[i]);order=np.argsort(-sim[j],kind='stable')
                        def top(c,same=True):return next((int(s[t]) for t in order if (y[s[t]]==c)==same),-1)
                        row=dict(fold=fold,budget=budget,draw=draw,arm=arm,row=int(i),truth=truth,
                            prediction=int(p.argmax()),correct=int(p.argmax())==truth,
                            margin=float(p[truth]-np.delete(p,truth).max()),ap=float(ap[j]),
                            brier=float(np.square(p-np.eye(7)[truth]).sum()),top_same_row=top(truth),
                            top_wrong_row=top(truth,False),top_predicted_row=top(int(p.argmax())))
                        row.update({f'p{c}':float(p[c]) for c in range(7)});output.append(row)
    pred=pd.DataFrame(output);met=pd.DataFrame(metrics)
    pred.to_csv(HERE/'dose_predictions.csv',index=False);met.to_csv(HERE/'dose_metrics.csv',index=False)
    met.groupby(['budget','arm'])[['oa','ba','brier','nll','map_r']].mean().to_csv(HERE/'dose_summary.csv')
    a=pred[pred.arm.eq('fg_bg_global')].set_index(KEYS).sort_index()
    b=pred[pred.arm.eq('fg_dose_only')].set_index(KEYS).sort_index()
    events=a[['truth','correct','margin','brier']].copy()
    events['rescue']=a.correct&~b.correct;events['harm']=~a.correct&b.correct
    events['delta_margin']=a.margin-b.margin;events['delta_brier']=a.brier-b.brier
    events.reset_index().to_csv(HERE/'dose_events.csv',index=False)
    (HERE/'dose_receipt.json').write_text(json.dumps(dict(posthoc=True,fg_bg_replay_error=error,
        interpretation='Adding exact background kernel at fixed foreground term; not a physical-causality test'),indent=2))
    print(met.groupby(['budget','arm'])[['oa','ba','brier','map_r']].mean().to_string())
    print(events[events.rescue|events.harm].to_string())


if __name__=='__main__':main()
