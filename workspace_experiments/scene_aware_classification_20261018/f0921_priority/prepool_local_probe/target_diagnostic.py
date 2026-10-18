"""Posthoc: replace only target slots, keep postpool context slots unchanged."""
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from run import HERE,PREV,choose_support,ap_at_r,cosine,token_descriptor,kernel


def main():
    meta=pd.read_csv(PREV/'cohort.csv');features=np.load(HERE/'local_features.npz');old=np.load(PREV/'features.npz')
    y=meta.label.to_numpy();products=meta['product'].to_numpy();classes=np.unique(y)
    metrics=[];records=[];norm_errors=[]
    for product in sorted(set(products)):
        q=np.flatnonzero(products==product);source=np.flatnonzero(products!=product)
        for seed in range(5):
            s=choose_support(source,y,seed);valid=features['valid12']
            pre,_=token_descriptor(features['prepool'],valid,s)
            post,_=token_descriptor(features['postpool'],valid,s)
            hybrid=post.copy();hybrid[:,:,::3,:]=pre[:,:,::3,:]
            norm_errors.append(float(np.abs(np.square(hybrid).sum((-2,-1))-np.square(post).sum((-2,-1))).max()))
            # Unit token norms and identical availability make target replacement norm preserving.
            np.testing.assert_allclose(np.square(hybrid).sum((-2,-1)),np.square(post).sum((-2,-1)),atol=1e-6)
            mat=.5*cosine(old['global'],s)+.25*cosine(old['grid'],s)+.25*kernel(hybrid)
            score=mat[np.ix_(q,s)]
            logits=np.stack([score[:,y[s]==c].mean(1) for c in classes],1)
            prob=softmax(logits/.1,axis=1);pred=prob.argmax(1);truth=y[q]
            ap=np.array([ap_at_r(row,y[s]==label) for row,label in zip(score,truth)])
            margins=np.array([row[label]-max(row[c] for c in classes if c!=label) for row,label in zip(logits,truth)])
            brier=np.square(prob-np.eye(7)[truth]).sum(1);nll=-np.log(np.maximum(prob[np.arange(len(q)),truth],1e-12))
            metrics.append(dict(fold=product[-4:],draw=seed,arm='target_only',queries=len(q),support=len(s),
                oa=float((pred==truth).mean()),ba=float(np.mean([(pred[truth==c]==c).mean() for c in np.unique(truth)])),
                map_r=float(ap.mean()),relation_margin=float(margins.mean()),brier=float(brier.mean()),nll=float(nll.mean())))
            for j,i in enumerate(q):
                order=np.argsort(-score[j],kind='stable')
                def top(c,equal=True):return int(next(s[k] for k in order if (y[s[k]]==c)==equal))
                row=dict(fold=product[-4:],draw=seed,arm='target_only',row=int(i),object_id=meta.iloc[i].object_id,mmsi=int(meta.iloc[i].mmsi),
                    truth=int(truth[j]),prediction=int(pred[j]),correct=bool(pred[j]==truth[j]),ap=float(ap[j]),relation_margin=float(margins[j]),
                    brier=float(brier[j]),nll=float(nll[j]),top_support_row=int(s[order[0]]),top_same_row=top(truth[j]),top_wrong_row=top(truth[j],False),top_predicted_row=top(pred[j]))
                row.update({f'p{c}':float(prob[j,c]) for c in classes});records.append(row)
    met=pd.DataFrame(metrics);pred=pd.DataFrame(records)
    met.to_csv(HERE/'target_diagnostic_metrics.csv',index=False);pred.to_csv(HERE/'target_diagnostic_predictions.csv',index=False)
    summary=met.groupby('arm')[['oa','ba','map_r','relation_margin','brier','nll']].mean()
    summary.to_csv(HERE/'target_diagnostic_summary.csv')
    original=pd.read_csv(HERE/'predictions.csv');keys=['fold','draw','row']
    new=pred.set_index(keys);pairs=[]
    for name in ['global_grid','postpool','prepool']:
        base=original[original.arm.eq(name)].set_index(keys)
        d=new[['correct','ap','relation_margin','brier']].join(base[['correct','ap','relation_margin','brier']],rsuffix='_base')
        d['comparator']=name;d['rescue']=d.correct&~d.correct_base;d['harm']=~d.correct&d.correct_base
        d['delta_margin']=d.relation_margin-d.relation_margin_base;pairs.append(d.reset_index())
    pd.concat(pairs).to_csv(HERE/'target_diagnostic_pairs.csv',index=False)
    (HERE/'target_diagnostic_receipt.json').write_text(json.dumps(dict(posthoc=True,changed_slots='target only; other/background exactly postpool',
        max_norm_difference=max(norm_errors),no_weight_search=True,reason='Target slots dominated fixed-rival margin gains in original pre/post comparison'),indent=2),encoding='utf-8')
    print(summary.to_string());print(pd.concat(pairs).groupby('comparator')[['rescue','harm']].sum().to_string())


if __name__=='__main__':main()
