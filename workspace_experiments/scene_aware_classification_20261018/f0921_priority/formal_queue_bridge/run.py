"""Held-date source priors, target support residual refits, and dual-head fusion."""
import json
import importlib.util
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check')
from unified_head_core import coefficients,late
spec=importlib.util.spec_from_file_location('prepool_readout',HERE.parent/'prepool_local_probe/run.py')
prepool=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepool)
token_descriptor,kernel=prepool.token_descriptor,prepool.kernel
sys.path.insert(0,str(HERE.parent/'region_relation_probe'))
from evaluate import cosine,ap_at_r


def global_descriptor(x,source):
    z=np.asarray(x,dtype=np.float64)-x[source].mean(0,keepdims=True)
    z/=np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
    if z.ndim==3:return z.reshape(len(z),-1)/np.sqrt(z.shape[1])
    return z


def prior(kernel,source,labels,nclass):
    alpha=coefficients(kernel[np.ix_(source,source)],np.zeros((len(source),nclass)),labels[source],1.)
    return kernel[:,source]@alpha


def adapted(kernel,source_prior,support,labels):
    alpha=coefficients(kernel[np.ix_(support,support)],source_prior[support],labels[support],8.)
    return source_prior+kernel[:,support]@alpha,alpha


def metrics(prob,truth,ap):
    pred=prob.argmax(1)
    return dict(oa=float((pred==truth).mean()),
        ba=float(np.mean([(pred[truth==c]==c).mean() for c in np.unique(truth)])),
        brier=float(np.square(prob-np.eye(prob.shape[1])[truth]).sum(1).mean()),
        nll=float(-np.log(np.maximum(prob[np.arange(len(truth)),truth],1e-12)).mean()),
        map_r=float(np.nanmean(ap)),retrieval_defined=int(np.isfinite(ap).sum()))


def margin_parts(old_k,new_k,old_a,new_a,query,support,truth,rivals):
    delta_k=(new_k-old_k)[np.ix_(query,support)]
    sum_k=(new_k+old_k)[np.ix_(query,support)]
    direct=.5*delta_k@(new_a+old_a)
    refit=.5*sum_k@(new_a-old_a)
    idx=np.arange(len(query))
    return direct[idx,truth]-direct[idx,rivals],refit[idx,truth]-refit[idx,rivals]


def main():
    start=time.perf_counter();meta=pd.read_csv(HERE/'cohort.csv',dtype={'mmsi':str,'date':str})
    features=np.load(HERE/'features.npz');np.testing.assert_array_equal(features['clean_rows'],meta.clean_row)
    episodes=json.loads((HERE/'episodes.json').read_text());y=meta.label.to_numpy();nc=int(y.max()+1)
    availability=pd.read_csv(HERE/'availability.csv').set_index('row')
    records=[];summaries=[];supports=[];decompositions=[];fold_cache={};timings=[]
    with threadpool_limits(limits=1):
        for task in episodes:
            src=np.array(task['source']);q=np.array(task['query']);fold=task['fold'];budget=task['budget']
            assert not set(meta.iloc[src].mmsi)&set(meta[meta.date.eq(fold)].mmsi)
            assert not meta.iloc[src].date.eq(fold).any()
            if fold not in fold_cache:
                g=global_descriptor(features['global'],src);b=global_descriptor(features['r18'],src)
                kg=g@g.T;kb=b@b.T
                pa=prior(kg,src,y,nc);pb=prior(kb,src,y,nc)
                fold_cache[fold]=(kg,kb,pa,pb)
            kg,kb,pa,pb=fold_cache[fold]
            for draw,rows in enumerate(task['supports']):
                s=np.array(rows);tic=time.perf_counter()
                assert len(s)==budget and len(set(meta.iloc[s].mmsi))==budget
                assert not set(meta.iloc[s].mmsi)&set(meta.iloc[q].mmsi)
                supports.extend(dict(fold=fold,budget=budget,draw=draw,support_row=int(i),label=int(y[i]),clean_row=int(meta.iloc[i].clean_row)) for i in s)
                old=cosine(features['grid2'].reshape(len(meta),2,-1),s).astype(np.float64)
                kernels=dict(global_only=kg,global_grid=.5*kg+.5*old,strength_control=.5*kg+.25*old)
                descriptors={}
                for name in ['grid12','postpool','prepool']:
                    valid=np.ones_like(features['valid12']) if name=='grid12' else features['valid12']
                    z,_=token_descriptor(features[name],valid,s);descriptors[name]=z
                    kernels[name]=.5*kg+.25*old+.25*kernel(z)
                hybrid=descriptors['postpool'].copy();hybrid[:,:,::3,:]=descriptors['prepool'][:,:,::3,:]
                np.testing.assert_allclose(np.square(hybrid).sum((-2,-1)),np.square(descriptors['postpool']).sum((-2,-1)),atol=1e-6)
                kernels['target_only']=.5*kg+.25*old+.25*kernel(hybrid)
                expert,_=adapted(kb,pb,s,y)
                scores={};alphas={}
                for arm,k in kernels.items():
                    value,alpha=adapted(k,pa,s,y);scores[arm]=value;alphas[arm]=alpha
                    prob=late(value[q],expert[q]);r50prob=softmax(value[q],axis=1);r18prob=softmax(expert[q],axis=1)
                    sim=k[np.ix_(q,s)];ap=np.array([ap_at_r(v,y[s]==label) for v,label in zip(sim,y[q])])
                    summaries.append(dict(fold=fold,budget=budget,draw=draw,arm=arm,queries=len(q),source=len(src),
                        support_classes=len(np.unique(y[s])),**metrics(prob,y[q],ap),
                        r50_oa=metrics(r50prob,y[q],ap)['oa'],r50_ba=metrics(r50prob,y[q],ap)['ba'],
                        r18_oa=metrics(r18prob,y[q],ap)['oa'],r18_ba=metrics(r18prob,y[q],ap)['ba']))
                    for j,i in enumerate(q):
                        order=np.argsort(-sim[j],kind='stable')
                        def top(c,equal=True):
                            found=[int(s[t]) for t in order if (y[s[t]]==c)==equal]
                            return found[0] if found else -1
                        p=prob[j];true=int(y[i]);pred=int(p.argmax());rivals=np.delete(np.arange(nc),true)
                        row=dict(fold=fold,budget=budget,draw=draw,arm=arm,row=int(i),clean_row=int(meta.iloc[i].clean_row),
                            truth=true,prediction=pred,correct=bool(pred==true),margin=float(p[true]-p[rivals].max()),
                            ap=float(ap[j]),brier=float(np.square(p-np.eye(nc)[true]).sum()),nll=float(-np.log(max(p[true],1e-12))),
                            top_same_row=top(true),top_wrong_row=top(true,False),top_predicted_row=top(pred),
                            same_class_support=int((y[s]==true).sum()),target_polarizations=int(availability.loc[i,'target_polarizations']),
                            r50_prediction=int(value[i].argmax()),r18_prediction=int(expert[i].argmax()),
                            prior_r50_prediction=int(pa[i].argmax()),prior_r18_prediction=int(pb[i].argmax()))
                        row.update({f'p{c}':float(p[c]) for c in range(nc)});records.append(row)
                for arm in ['prepool','target_only']:
                    old_score=scores['global_grid'][q].copy();old_score[np.arange(len(q)),y[q]]=-np.inf
                    rival=old_score.argmax(1)
                    direct,refit=margin_parts(kernels['global_grid'],kernels[arm],alphas['global_grid'],alphas[arm],q,s,y[q],rival)
                    change=scores[arm][q]-scores['global_grid'][q]
                    observed=change[np.arange(len(q)),y[q]]-change[np.arange(len(q)),rival]
                    np.testing.assert_allclose(direct+refit,observed,atol=1e-8)
                    decompositions.extend(dict(fold=fold,budget=budget,draw=draw,arm=arm,row=int(i),fixed_rival=int(rival[j]),
                        direct_kernel=float(direct[j]),support_refit=float(refit[j]),r50_fixed_margin_change=float(observed[j])) for j,i in enumerate(q))
                timings.append(dict(fold=fold,budget=budget,draw=draw,seconds=time.perf_counter()-tic))
            print('DONE',fold,'B',budget,flush=True)
    met=pd.DataFrame(summaries);pred=pd.DataFrame(records)
    met.to_csv(HERE/'metrics.csv',index=False);pred.to_csv(HERE/'predictions.csv',index=False)
    pd.DataFrame(supports).to_csv(HERE/'supports.csv',index=False)
    pd.DataFrame(decompositions).to_csv(HERE/'margin_decomposition.csv',index=False)
    pd.DataFrame(timings).to_csv(HERE/'timings.csv',index=False)
    cols=['oa','ba','brier','nll','map_r','r50_oa','r50_ba','r18_oa','r18_ba']
    summary=met.groupby(['budget','arm'])[cols].mean();summary.to_csv(HERE/'summary_equal_date.csv')
    met.groupby(['fold','budget','arm'])[cols].mean().to_csv(HERE/'by_date.csv')
    pred.groupby(['budget','arm','truth'])[['correct','brier','ap']].mean().to_csv(HERE/'by_class.csv')
    weighted=pred.groupby(['budget','arm'])[['correct','brier','ap']].mean()
    weighted['ba']=pred.groupby(['budget','arm','truth']).correct.mean().groupby(['budget','arm']).mean()
    weighted.to_csv(HERE/'summary_query_weighted.csv')
    (HERE/'run_receipt.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,episodes=len(episodes),
        draws=5,arms=list(kernels),source_prior_fits=2*len(fold_cache),backbone_training=False,complete_decision_path=True,
        historical_full_model_reproduced=False,classification_head_classes=nc,finished=True),indent=2))
    print(summary.to_string())


if __name__=='__main__':main()
