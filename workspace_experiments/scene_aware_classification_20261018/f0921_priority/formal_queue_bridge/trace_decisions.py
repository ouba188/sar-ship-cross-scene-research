"""Exact saved-event replay and support contributions; no new fitted candidate."""
import json
import argparse
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run import HERE,global_descriptor,prior,adapted,late,cosine,token_descriptor,kernel


def local_kernels(feat,kg,s,comparator='global_grid'):
    old=cosine(feat['grid2'].reshape(len(kg),2,-1),s)
    pre,_=token_descriptor(feat['prepool'],feat['valid12'],s)
    post,_=token_descriptor(feat['postpool'],feat['valid12'],s)
    hybrid=post.copy();hybrid[:,:,::3,:]=pre[:,:,::3,:]
    base=.5*kg+.5*old if comparator=='global_grid' else .5*kg+.25*old+.25*kernel(post)
    return base,.5*kg+.25*old+.25*kernel(hybrid)


def main(comparator='global_grid'):
    meta=pd.read_csv(HERE/'cohort.csv',keep_default_na=False)
    feat=np.load(HERE/'features.npz');episodes=json.loads((HERE/'episodes.json').read_text());y=meta.label.to_numpy()
    saved=pd.read_csv(HERE/'predictions.csv');flips=pd.read_csv(HERE/'flips.csv')
    rows=[];events=[];origin=[];max_error=0.
    with threadpool_limits(limits=1):
        for task in episodes:
            q=np.array(task['query']);src=np.array(task['source']);fold=int(task['fold']);budget=task['budget']
            ga=global_descriptor(feat['global'],src);gb=global_descriptor(feat['r18'],src)
            kg=ga@ga.T;kb=gb@gb.T;pa=prior(kg,src,y,7);pb=prior(kb,src,y,7)
            for draw,sup in enumerate(task['supports']):
                s=np.array(sup)
                for i in q:
                    acq=meta.iloc[i].acquisition
                    sa=meta.iloc[s].acquisition
                    known=(sa!='') & bool(acq)
                    origin.append(dict(fold=fold,budget=budget,draw=draw,row=int(i),
                        same_known_acquisition=int((known&sa.eq(acq)).sum()),
                        different_known_acquisition=int((known&sa.ne(acq)).sum()),unknown=int((~known).sum())))
                cases=flips[(flips.fold==fold)&(flips.budget==budget)&(flips.draw==draw)]
                if not len(cases):continue
                old,new=local_kernels(feat,kg,s,comparator)
                old_score,ao=adapted(old,pa,s,y);new_score,an=adapted(new,pa,s,y)
                expert,_=adapted(kb,pb,s,y)
                op=late(old_score[q],expert[q]);npred=late(new_score[q],expert[q])
                for arm,p in [(comparator,op),('target_only',npred)]:
                    ref=saved[(saved.fold==fold)&(saved.budget==budget)&(saved.draw==draw)&saved.arm.eq(arm)].set_index('row').loc[q]
                    error=float(np.max(np.abs(p-ref[[f'p{c}' for c in range(7)]].to_numpy())))
                    max_error=max(max_error,error);np.testing.assert_allclose(p,ref[[f'p{c}' for c in range(7)]],atol=1e-12)
                for case in cases.itertuples():
                    i=int(case.row);j=int(np.flatnonzero(q==i)[0]);true=int(y[i]);rivals=np.delete(np.arange(7),true)
                    rival=int(rivals[np.argmax(op[j,rivals])])
                    direct=.5*(new[i,s]-old[i,s])*((an+ao)[:,true]-(an+ao)[:,rival])
                    refit=.5*(new[i,s]+old[i,s])*((an-ao)[:,true]-(an-ao)[:,rival])
                    observed=(new_score[i,true]-new_score[i,rival])-(old_score[i,true]-old_score[i,rival])
                    np.testing.assert_allclose((direct+refit).sum(),observed,atol=1e-10)
                    for k,sr in enumerate(s):
                        rows.append(dict(fold=fold,budget=budget,draw=draw,row=i,support_row=int(sr),
                            true=true,fixed_final_rival=rival,support_label=int(y[sr]),
                            old_similarity=float(old[i,sr]),new_similarity=float(new[i,sr]),
                            kernel_term=float(direct[k]),refit_term=float(refit[k]),total_term=float(direct[k]+refit[k]),
                            support_acquisition=meta.iloc[sr].acquisition))
                    events.append(dict(fold=fold,budget=budget,draw=draw,row=i,true=true,fixed_final_rival=rival,
                        baseline_prediction=int(op[j].argmax()),candidate_prediction=int(npred[j].argmax()),
                        old_probability_margin=float(op[j,true]-op[j,rival]),new_probability_margin=float(npred[j,true]-npred[j,rival]),
                        old_r50_margin=float(old_score[i,true]-old_score[i,rival]),new_r50_margin=float(new_score[i,true]-new_score[i,rival]),
                        r18_fixed_margin=float(expert[i,true]-expert[i,rival]),direct_kernel=float(direct.sum()),
                        support_refit=float(refit.sum()),r50_margin_change=float(observed),
                        strongest_helping_support=int(s[np.argmax(direct+refit)]),strongest_harming_support=int(s[np.argmin(direct+refit)])))
    prefix='' if comparator=='global_grid' else comparator+'_'
    pd.DataFrame(rows).to_csv(HERE/(prefix+'case_support_contributions.csv'),index=False)
    pd.DataFrame(events).to_csv(HERE/(prefix+'decision_trace.csv'),index=False)
    if not prefix:pd.DataFrame(origin).to_csv(HERE/'support_acquisition_composition.csv',index=False)
    (HERE/(prefix+'trace_receipt.json')).write_text(json.dumps(dict(max_saved_probability_error=max_error,
        events=len(events),new_candidates_fitted=False,meaning='Exact numeric path, not physical causal identification'),indent=2))
    print(pd.DataFrame(events).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--comparator',choices=['global_grid','postpool'],default='global_grid')
    main(parser.parse_args().comparator)
