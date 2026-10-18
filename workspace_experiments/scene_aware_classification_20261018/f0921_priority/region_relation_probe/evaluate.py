"""Fixed cross-acquisition support draws; no target model selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax

HERE=Path(__file__).resolve().parent


def cosine(x,support):
    z=x-x[support].mean(0,keepdims=True)
    z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
    return np.einsum('npd,mpd->nm',z,z)/z.shape[1]


def observed_region_cosine(x,area,support):
    """Diagnostic: missing blocks remain zero after the same support centering."""
    z=(x-x[support].mean(0,keepdims=True)).reshape(len(x),2,4,-1)
    z=z*(area>0)[...,None]
    z=z.reshape(len(x),2,-1)
    z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
    return np.einsum('npd,mpd->nm',z,z)/z.shape[1]


def ap_at_r(scores,truth):
    r=int(truth.sum())
    if not r:return np.nan
    correct=truth[np.argsort(-scores,kind='stable')[:r]]
    return float(np.sum(np.cumsum(correct)/np.arange(1,r+1)*correct)/r)


def choose_support(source,labels,seed,budget=16):
    rng=np.random.default_rng(seed);selected=[]
    for label in sorted(np.unique(labels[source])):
        pool=source[labels[source]==label]
        selected.extend(rng.choice(pool,min(2,len(pool)),replace=False).tolist())
    if len(selected)>budget:raise ValueError('Budget too small for declared stratified supports')
    rest=np.setdiff1d(source,selected)
    selected.extend(rng.choice(rest,min(budget-len(selected),len(rest)),replace=False).tolist())
    return np.array(selected,dtype=int)


def main(diagnostic=False):
    meta=pd.read_csv(HERE/'cohort.csv');feat=dict(np.load(HERE/'features.npz'))
    labels=meta.label.to_numpy();products=meta['product'].to_numpy();nclass=labels.max()+1
    records=[];metrics=[];support_rows=[]
    for product in sorted(set(products)):
        query=np.flatnonzero(products==product);source=np.flatnonzero(products!=product)
        for seed in range(5):
            s=choose_support(source,labels,seed)
            assert set(meta.iloc[s].mmsi).isdisjoint(set(meta.iloc[query].mmsi))
            for i in s:support_rows.append(dict(fold=product[-4:],draw=seed,support_row=int(i),object_id=meta.iloc[i].object_id,label=int(labels[i])))
            kernels={k:cosine(x,s) for k,x in feat.items()}
            arms=[('global',None),('global_grid','grid'),('global_region','region'),('global_area','area')]
            if diagnostic:
                kernels['observed_region']=observed_region_cosine(feat['region'],feat['area'],s)
                arms=[('global_observed_region','observed_region')]
            for arm,extra in arms:
                matrix=kernels['global'] if extra is None else .5*kernels['global']+.5*kernels[extra]
                score=matrix[np.ix_(query,s)]
                logits=np.stack([score[:,labels[s]==k].mean(1) for k in range(nclass)],axis=1)
                prob=softmax(logits/.1,axis=1);pred=prob.argmax(1);truth=labels[query]
                ap=np.array([ap_at_r(row,labels[s]==y) for row,y in zip(score,truth)])
                gaps=np.array([row[labels[s]==y].mean()-max(row[labels[s]==k].mean() for k in range(nclass) if k!=y) for row,y in zip(score,truth)])
                brier=np.square(prob-np.eye(nclass)[truth]).sum(1)
                nll=-np.log(np.maximum(prob[np.arange(len(query)),truth],1e-12))
                metrics.append(dict(fold=product[-4:],draw=seed,arm=arm,queries=len(query),support=len(s),query_classes=len(set(truth)),oa=float(np.mean(pred==truth)),ba=float(np.mean([np.mean(pred[truth==k]==k) for k in sorted(set(truth))])),map_r=float(np.nanmean(ap)),relation_margin=float(np.mean(gaps)),brier=float(brier.mean()),nll=float(nll.mean())))
                for j,q in enumerate(query):
                    order=np.argsort(-score[j],kind='stable')
                    same=[i for i in order if labels[s[i]]==truth[j]]
                    wrong=[i for i in order if labels[s[i]]!=truth[j]]
                    row=dict(fold=product[-4:],draw=seed,arm=arm,row=int(q),object_id=meta.iloc[q].object_id,mmsi=int(meta.iloc[q].mmsi),truth=int(truth[j]),prediction=int(pred[j]),correct=bool(pred[j]==truth[j]),ap=float(ap[j]),relation_margin=float(gaps[j]),brier=float(brier[j]),nll=float(nll[j]),top_support_row=int(s[order[0]]),top_same_row=int(s[same[0]]),top_wrong_row=int(s[wrong[0]]))
                    row.update({f'p{k}':float(prob[j,k]) for k in range(nclass)})
                    records.append(row)
    met=pd.DataFrame(metrics);pred=pd.DataFrame(records)
    if diagnostic:
        met.to_csv(HERE/'diagnostic_metrics.csv',index=False)
        pred.to_csv(HERE/'diagnostic_predictions.csv',index=False)
        print(met.groupby('arm')[['oa','ba','map_r','brier']].mean().to_string())
        return
    met.to_csv(HERE/'metrics.csv',index=False);pred.to_csv(HERE/'predictions.csv',index=False)
    pd.DataFrame(support_rows).to_csv(HERE/'supports.csv',index=False)
    mean=met.groupby('arm')[['oa','ba','map_r','relation_margin','brier','nll']].mean()
    mean.to_csv(HERE/'summary_equal_fold.csv')
    effects=[]
    for comparator in ['global','global_grid','global_area']:
        a=pred[pred.arm.eq('global_region')].set_index(['fold','draw','row'])
        b=pred[pred.arm.eq(comparator)].set_index(['fold','draw','row'])
        for key,row in a.iterrows():
            old=b.loc[key]
            effects.append(dict(fold=key[0],draw=key[1],row=key[2],object_id=row.object_id,comparator=comparator,rescue=bool(row.correct and not old.correct),harm=bool(old.correct and not row.correct),delta_ap=row.ap-old.ap,delta_margin=row.relation_margin-old.relation_margin,delta_brier=row.brier-old.brier))
    pd.DataFrame(effects).to_csv(HERE/'paired_effects.csv',index=False)
    print(mean.to_string(),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostic',action='store_true')
    main(parser.parse_args().diagnostic)
