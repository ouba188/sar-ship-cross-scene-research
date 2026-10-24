"""Source-only matched reference organization; no new training or target use."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('scope_probe',HERE/'probe.py')
probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)


def main():
    table,x,mass,bank,idx,tr,va,te=probe.prev.exp.source_data()
    z=np.load(HERE/'source_profiles.npz'); ids=z['ids']; per=z['reference']; covariance=z['cov']; valid=per>-9999
    assert set(table.iloc[ids].role)=={'utility_train','selection'}
    body=(x[:,:,:8]*mass[:,:,:8,None]).sum(2)/np.maximum(mass[:,:,:8].sum(2)[...,None],1e-8)
    body=body/np.maximum(np.linalg.norm(body,axis=-1,keepdims=True),1e-8)
    affinity=np.einsum('nd,crd->ncr',body.reshape(len(x),-1)[ids],body.reshape(len(x),-1)[bank])
    rng=np.random.default_rng(711); random=rng.random(affinity.shape)
    masks={}
    for name,rank in [('nearest2',affinity),('random2',random)]:
        rank=np.broadcast_to(rank[:,None],(len(ids),6,8,8)).copy()
        rv=valid[:,:,0]
        order=np.argsort(-np.where(rv,rank,-np.inf),axis=-1)[...,:2]
        choose=np.zeros_like(rv)
        np.put_along_axis(choose,order,True,axis=-1)
        masks[name]=choose&rv
    masks['all8']=valid[:,:,0]
    results={}; rows=[]; sample=[]
    yy=table.iloc[ids].class_id.to_numpy(int); roles=table.iloc[ids].role.to_numpy()
    querymask=probe.prev.organize(x,mass)[1][ids]
    weights=querymask/np.maximum(querymask.sum(-1,keepdims=True),1)
    for name,mask in masks.items():
        local=logsumexp(np.where(mask[:,:,None],per,-np.inf),axis=-1)-np.log(np.maximum(mask.sum(-1),1))[:,:,None]
        local=np.where(np.isfinite(local),local,0.)
        score=(local*weights[...,None]).sum(2); results[name]=score
        for role in ('utility_train','selection'):
            for g,group in enumerate(probe.prev.GROUPS):
                use=(roles==role)&querymask[:,g].any(-1)&mask[:,g].any(-1).all(-1)
                y=yy[use]; s=score[use,g]; pred=s.argmax(-1)
                gap=s[np.arange(len(y)),y]-np.where(np.arange(8)[None]==y[:,None],-np.inf,s).max(-1)
                rows.append(dict(arm=name,role=role,group=group,n=int(use.sum()),OA=float((pred==y).mean()),BA=float(np.mean([(pred[y==c]==c).mean() for c in np.unique(y)])),macro_gap=float(np.mean([gap[y==c].mean() for c in np.unique(y)]))))
    for i in range(len(ids)):
        for g,group in enumerate(probe.prev.GROUPS):
            if not querymask[i,g].any(): continue
            c=int(yy[i]); h=np.exp(per[i,g,:,c]-per[i,g,:,c].max(-1,keepdims=True))*valid[i,g,:,c]
            r=h/np.maximum(h.sum(-1,keepdims=True),1e-20)
            effective=(1/np.maximum((r*r).sum(-1),1e-20)*weights[i,g]).sum()
            compatibility=(r*affinity[i,c][None]).sum(-1)@weights[i,g]
            cov=covariance[i,g,c]; variance=float(weights[i,g]@cov@weights[i,g])
            sample.append(dict(role=roles[i],obs_id=int(table.iloc[ids[i]].obs_id),class_id=c,group=group,effective_references=float(effective),responsibility_body_affinity=float(compatibility),variance=variance,
                all_correct=int(results['all8'][i,g].argmax()==c),near_correct=int(results['nearest2'][i,g].argmax()==c),random_correct=int(results['random2'][i,g].argmax()==c)))
    pd.DataFrame(rows).to_csv(HERE/'reference_scope_metrics.csv',index=False)
    pd.DataFrame(sample).to_csv(HERE/'reference_scope_samples.csv',index=False)
    np.savez_compressed(HERE/'reference_scope_scores.npz',ids=ids,**results)
    summaries=[]
    for (role,g),f in pd.DataFrame(sample).groupby(['role','group']):
        summaries.append(dict(role=role,group=g,variance_neff_spearman=float(spearmanr(f.variance,f.effective_references).statistic),variance_affinity_spearman=float(spearmanr(f.variance,f.responsibility_body_affinity).statistic)))
    pd.DataFrame(summaries).to_csv(HERE/'reference_concentration.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(summaries).to_string(index=False))
    (HERE/'reference_scope_receipt.json').write_text(json.dumps(dict(source_only=True,fits=0,nearest='two highest shared body affinities among valid references in each group',random='fixed711 query-class ranks matched count',historical_baseline='used two shared body-nearest references; not identical operator/head to current'),indent=2),encoding='utf-8')


if __name__=='__main__': main()
