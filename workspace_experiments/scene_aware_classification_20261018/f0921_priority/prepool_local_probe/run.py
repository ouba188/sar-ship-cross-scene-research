"""Paired relation evaluation of equal-sized local token readouts."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax

HERE=Path(__file__).resolve().parent;PREV=HERE.parent/'region_relation_probe'
sys.path.insert(0,str(PREV))
from evaluate import choose_support,ap_at_r,cosine


def token_descriptor(tokens,valid,support):
    count=valid[support].sum(0)
    center=(tokens[support]*valid[support,None,:,None]).sum(0)/np.maximum(count[None,:,None],1)
    z=(tokens-center[None])*valid[:,None,:,None]*(count>0)[None,None,:,None]
    norm=np.linalg.norm(z,axis=-1,keepdims=True)
    z=z/np.maximum(norm,1e-12)
    denom=np.sqrt(np.square(z).sum((-2,-1),keepdims=True))
    return z/np.maximum(denom,1e-12),count


def kernel(z):
    return np.einsum('nptd,mptd->nm',z,z)/z.shape[1]


def main():
    meta=pd.read_csv(PREV/'cohort.csv');base=dict(np.load(PREV/'features.npz'))
    local=dict(np.load(HERE/'local_features.npz'))
    y=meta.label.to_numpy();products=meta['product'].to_numpy();classes=np.unique(y)
    metrics=[];predictions=[];supports=[];coverage=[];decomposition=[]
    for product in sorted(set(products)):
        q=np.flatnonzero(products==product);source=np.flatnonzero(products!=product)
        for seed in range(5):
            s=choose_support(source,y,seed)
            assert set(meta.iloc[s].mmsi).isdisjoint(set(meta.iloc[q].mmsi))
            supports.extend(dict(fold=product[-4:],draw=seed,support_row=int(i),label=int(y[i])) for i in s)
            g=cosine(base['global'],s);old=cosine(base['grid'],s)
            matrices={'global_grid':.5*g+.5*old};descriptors={}
            for name in ['grid12','postpool','prepool']:
                valid=np.ones_like(local['valid12']) if name=='grid12' else local['valid12']
                z,count=token_descriptor(local[name],valid,s);descriptors[name]=z
                matrices[name]=.5*g+.25*old+.25*kernel(z)
                for t,c in enumerate(count):coverage.append(dict(fold=product[-4:],draw=seed,arm=name,token=t,support_count=int(c),query_available=int(valid[q,t].sum())))
            logits_by_arm={}
            for arm,mat in matrices.items():
                sim=mat[np.ix_(q,s)]
                logits=np.stack([sim[:,y[s]==c].mean(1) for c in classes],axis=1)
                logits_by_arm[arm]=logits
                probs=softmax(logits/.1,axis=1);pred=probs.argmax(1);truth=y[q]
                ap=np.array([ap_at_r(row,y[s]==label) for row,label in zip(sim,truth)])
                margin=np.array([row[label]-max(row[c] for c in classes if c!=label) for row,label in zip(logits,truth)])
                brier=np.square(probs-np.eye(len(classes))[truth]).sum(1)
                nll=-np.log(np.maximum(probs[np.arange(len(q)),truth],1e-12))
                metrics.append(dict(fold=product[-4:],draw=seed,arm=arm,queries=len(q),support=len(s),
                    oa=float((pred==truth).mean()),ba=float(np.mean([(pred[truth==c]==c).mean() for c in np.unique(truth)])),
                    map_r=float(ap.mean()),relation_margin=float(margin.mean()),brier=float(brier.mean()),nll=float(nll.mean())))
                for j,i in enumerate(q):
                    order=np.argsort(-sim[j],kind='stable')
                    def top(label,equal=True):
                        return int(next(s[k] for k in order if (y[s[k]]==label)==equal))
                    item=dict(fold=product[-4:],draw=seed,arm=arm,row=int(i),object_id=meta.iloc[i].object_id,mmsi=int(meta.iloc[i].mmsi),
                        truth=int(truth[j]),prediction=int(pred[j]),correct=bool(pred[j]==truth[j]),ap=float(ap[j]),relation_margin=float(margin[j]),
                        brier=float(brier[j]),nll=float(nll[j]),top_support_row=int(s[order[0]]),top_same_row=top(truth[j]),top_wrong_row=top(truth[j],False),top_predicted_row=top(pred[j]))
                    item.update({f'p{c}':float(probs[j,c]) for c in classes});predictions.append(item)
            # Pre/post have aligned tokens, so their margin difference decomposes exactly.
            parts={name:np.einsum('qptd,sptd->qst',descriptors[name][q],descriptors[name][s])/2 for name in ['prepool','postpool']}
            for j,i in enumerate(q):
                rival=max((c for c in classes if c!=y[i]),key=lambda c:logits_by_arm['postpool'][j,c])
                delta=.25*(parts['prepool'][j]-parts['postpool'][j])
                contribution=delta[y[s]==y[i]].mean(0)-delta[y[s]==rival].mean(0)
                observed=(logits_by_arm['prepool'][j,y[i]]-logits_by_arm['prepool'][j,rival])-(logits_by_arm['postpool'][j,y[i]]-logits_by_arm['postpool'][j,rival])
                np.testing.assert_allclose(contribution.sum(),observed,atol=5e-7)
                roles=contribution.reshape(4,3).sum(0)
                decomposition.append(dict(fold=product[-4:],draw=seed,row=int(i),truth=int(y[i]),fixed_rival=int(rival),
                    delta_fixed_margin=float(observed),target=float(roles[0]),other=float(roles[1]),background=float(roles[2])))
    met=pd.DataFrame(metrics);pred=pd.DataFrame(predictions)
    prior=pd.read_csv(PREV/'predictions.csv');keys=['fold','draw','row'];cols=[f'p{c}' for c in classes]
    oldpred=prior[prior.arm.eq('global_grid')].set_index(keys).sort_index()
    newpred=pred[pred.arm.eq('global_grid')].set_index(keys).sort_index()
    np.testing.assert_allclose(newpred[cols],oldpred[cols],atol=1e-7)
    met.to_csv(HERE/'metrics.csv',index=False);pred.to_csv(HERE/'predictions.csv',index=False)
    pd.DataFrame(supports).to_csv(HERE/'supports.csv',index=False)
    pd.DataFrame(coverage).to_csv(HERE/'token_support_coverage.csv',index=False)
    pd.DataFrame(decomposition).to_csv(HERE/'pre_post_margin_decomposition.csv',index=False)
    summary=met.groupby('arm')[['oa','ba','map_r','relation_margin','brier','nll']].mean()
    summary.to_csv(HERE/'summary_equal_fold.csv')
    met.groupby(['fold','arm'])[['oa','ba','map_r','brier']].mean().to_csv(HERE/'by_acquisition.csv')
    pred.groupby(['arm','truth'])[['correct','ap','relation_margin','brier']].mean().to_csv(HERE/'by_class.csv')
    weighted=pred.groupby('arm')[['correct','ap','brier']].mean()
    weighted['ba']=pred.groupby(['arm','truth']).correct.mean().groupby('arm').mean()
    weighted.to_csv(HERE/'summary_query_weighted.csv')
    print(summary.to_string());print(weighted.to_string())
    (HERE/'receipt.json').write_text(json.dumps(dict(baseline_replay=True,n_observations=len(meta),support_budget=16,support_draws=5,
        neural_training=False,feature_tokens=12,token_channels=256,all_run_complete=True),indent=2),encoding='utf-8')


if __name__=='__main__':main()
