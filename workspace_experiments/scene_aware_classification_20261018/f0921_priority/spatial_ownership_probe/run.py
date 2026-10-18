"""Frozen spatial descriptors with detection-ownership feature lifting."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'region_relation_probe'
sys.path.insert(0,str(PREV))
from evaluate import choose_support,ap_at_r,cosine


def cell_fractions(regions):
    # Match extract.py's TL, TR, BL, BR flattening exactly. No map distinction.
    ownership=np.stack([regions[:,0],regions[:,1],regions[:,2]+regions[:,3]],axis=1)
    cells=[]
    for r,c in [(0,0),(0,1),(1,0),(1,1)]:
        cells.append(ownership[:,:,r*64:(r+1)*64,c*64:(c+1)*64].mean((-2,-1)))
    p=np.stack(cells,axis=1)
    np.testing.assert_allclose(p.sum(-1),1,atol=1e-7)
    return p.astype(np.float32)


def normalize_grid(grid,support):
    z=grid-grid[support].mean(0,keepdims=True)
    z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
    return z.reshape(len(grid),grid.shape[1],4,-1)


def lift(z,proportions):
    return (z[:,:,:,None,:]*np.sqrt(proportions)[:,None,:,:,None]).reshape(len(z),z.shape[1],-1)


def inner(z):
    z=z.reshape(len(z),z.shape[1],-1)
    return np.einsum('npd,mpd->nm',z,z)/z.shape[1]


def build_kernels(features,p,support):
    z=normalize_grid(features['grid'],support)
    g=cosine(features['global'],support);local=inner(z)
    p2=np.stack([p[:,:,0],p[:,:,1:].sum(-1)],axis=-1)
    o2=inner(lift(z,p2));o3=inner(lift(z,p))
    geom=np.sqrt(p).reshape(len(p),-1)/2
    geometry=geom@geom.T
    return dict(global_=g,global_grid=.5*g+.5*local,
                spatial_ownership2=.5*g+.25*local+.25*o2,
                spatial_ownership3=.5*g+.25*local+.25*o3,
                ownership_only=.5*g+.25*local+.25*geometry),z


def main():
    started=time.perf_counter()
    meta=pd.read_csv(PREV/'cohort.csv');features=dict(np.load(PREV/'features.npz'))
    regions=np.load(PREV/'inputs.npz')['regions'];p=cell_fractions(regions)
    labels=meta.label.to_numpy();products=meta['product'].to_numpy();classes=np.unique(labels)
    metrics=[];records=[];supports=[];decomp=[]
    for product in sorted(set(products)):
        query=np.flatnonzero(products==product);source=np.flatnonzero(products!=product)
        for seed in range(5):
            s=choose_support(source,labels,seed)
            assert set(meta.iloc[s].mmsi).isdisjoint(set(meta.iloc[query].mmsi))
            supports.extend(dict(fold=product[-4:],draw=seed,support_row=int(i),label=int(labels[i])) for i in s)
            kernels,z=build_kernels(features,p,s)
            kernels['global']=kernels.pop('global_')
            logits_by_arm={}
            for arm,kernel in kernels.items():
                sim=kernel[np.ix_(query,s)]
                logits=np.stack([sim[:,labels[s]==k].mean(1) for k in classes],axis=1)
                logits_by_arm[arm]=logits
                prob=softmax(logits/.1,axis=1);pred=prob.argmax(1);truth=labels[query]
                ap=np.array([ap_at_r(row,labels[s]==y) for row,y in zip(sim,truth)])
                margin=np.array([row[y]-max(row[k] for k in classes if k!=y) for row,y in zip(logits,truth)])
                brier=np.square(prob-np.eye(len(classes))[truth]).sum(1)
                nll=-np.log(np.maximum(prob[np.arange(len(query)),truth],1e-12))
                metrics.append(dict(fold=product[-4:],draw=seed,arm=arm,queries=len(query),support=len(s),
                                    oa=float(np.mean(pred==truth)),ba=float(np.mean([np.mean(pred[truth==k]==k) for k in np.unique(truth)])),
                                    map_r=float(ap.mean()),relation_margin=float(margin.mean()),brier=float(brier.mean()),nll=float(nll.mean())))
                for j,q in enumerate(query):
                    rank=np.argsort(-sim[j],kind='stable')
                    same=[i for i in rank if labels[s[i]]==truth[j]]
                    wrong=[i for i in rank if labels[s[i]]!=truth[j]]
                    predicted=[i for i in rank if labels[s[i]]==pred[j]]
                    item=dict(fold=product[-4:],draw=seed,arm=arm,row=int(q),object_id=meta.iloc[q].object_id,
                              mmsi=int(meta.iloc[q].mmsi),truth=int(truth[j]),prediction=int(pred[j]),correct=bool(pred[j]==truth[j]),
                              ap=float(ap[j]),relation_margin=float(margin[j]),brier=float(brier[j]),nll=float(nll[j]),
                              top_support_row=int(s[rank[0]]),top_same_row=int(s[same[0]]),top_wrong_row=int(s[wrong[0]]),
                              top_predicted_row=int(s[predicted[0]]))
                    item.update({f'p{k}':float(prob[j,k]) for k in classes});records.append(item)
            # Exact fixed-rival margin decomposition, not a causal physical attribution.
            compatibility=np.einsum('qcr,scr->qsc',np.sqrt(p[query]),np.sqrt(p[s]))
            cell_dot=np.einsum('qpcd,spcd->qsc',z[query],z[s])/z.shape[1]
            delta=.25*cell_dot*(compatibility-1)
            for j,q in enumerate(query):
                y=labels[q];rival=max((k for k in classes if k!=y),key=lambda k:logits_by_arm['global_grid'][j,k])
                contribution=delta[j,labels[s]==y].mean(0)-delta[j,labels[s]==rival].mean(0)
                expected=(logits_by_arm['spatial_ownership3'][j,y]-logits_by_arm['spatial_ownership3'][j,rival])-(logits_by_arm['global_grid'][j,y]-logits_by_arm['global_grid'][j,rival])
                np.testing.assert_allclose(contribution.sum(),expected,atol=3e-7)
                row=dict(fold=product[-4:],draw=seed,row=int(q),truth=int(y),fixed_rival=int(rival),delta_fixed_margin=float(expected),
                         same_compatibility=float(compatibility[j,labels[s]==y].mean()),
                         rival_compatibility=float(compatibility[j,labels[s]==rival].mean()))
                row.update({f'cell{k}_contribution':float(v) for k,v in enumerate(contribution)})
                decomp.append(row)
    met=pd.DataFrame(metrics);pred=pd.DataFrame(records)
    old=pd.read_csv(PREV/'predictions.csv')
    keys=['fold','draw','arm','row'];columns=[f'p{k}' for k in classes]
    base=pred[pred.arm.isin(['global','global_grid'])].set_index(keys).sort_index()
    prior=old[old.arm.isin(['global','global_grid'])].set_index(keys).sort_index()
    np.testing.assert_allclose(base[columns],prior[columns],atol=2e-6,rtol=0)
    np.testing.assert_array_equal(base.prediction,prior.prediction)
    met.to_csv(HERE/'metrics.csv',index=False);pred.to_csv(HERE/'predictions.csv',index=False)
    pd.DataFrame(supports).to_csv(HERE/'supports.csv',index=False)
    pd.DataFrame(decomp).to_csv(HERE/'margin_decomposition.csv',index=False)
    mean=met.groupby('arm')[['oa','ba','map_r','relation_margin','brier','nll']].mean()
    mean.to_csv(HERE/'summary_equal_fold.csv')
    met.groupby(['fold','arm'])[['oa','ba','map_r','brier']].mean().to_csv(HERE/'by_acquisition.csv')
    pred.groupby(['arm','truth'])[['correct','ap','relation_margin','brier']].mean().to_csv(HERE/'by_class.csv')
    weighted=pred.groupby('arm')[['correct','ap','brier','nll']].mean()
    weighted['ba']=pred.groupby(['arm','truth']).correct.mean().groupby('arm').mean()
    weighted.to_csv(HERE/'summary_query_weighted.csv')
    paired=[]
    comparator=pred[pred.arm.eq('global_grid')].set_index(['fold','draw','row'])
    for arm in ['spatial_ownership2','spatial_ownership3','ownership_only']:
        method=pred[pred.arm.eq(arm)].set_index(['fold','draw','row'])
        d=method[['correct','ap','relation_margin','brier']].join(comparator[['correct','ap','relation_margin','brier']],rsuffix='_base')
        d['rescue']=d.correct&~d.correct_base;d['harm']=~d.correct&d.correct_base
        for metric in ['ap','relation_margin','brier']:d['delta_'+metric]=d[metric]-d[metric+'_base']
        d['arm']=arm;paired.append(d.reset_index())
    pd.concat(paired).to_csv(HERE/'paired_effects.csv',index=False)
    receipt=dict(seconds=time.perf_counter()-started,neural_training=False,new_backbone_inference=False,
                 baseline_probability_replay=True,feature_source=str(PREV/'features.npz'),
                 support_budget=16,support_draws=5,query_observations=84,products=5,
                 ownership_source='detector masks only; water/rest map distinction merged',
                 claims='Exploratory relation probe with provisional identity associations; not complete system gain.')
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(mean.to_string());print(weighted.to_string())


if __name__=='__main__':main()
