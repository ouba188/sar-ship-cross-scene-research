"""Attach exact signed support terms to prespecified representative cases."""

from run_paths import OUT,ROOT,read,write,arrays,Geometry,BATCH
from analyze_paths import manifest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def main():
    tasks=read(ROOT/'paper_core_freeze_20261013/FROZEN_REFERENCE.json')['tasks']
    cases=pd.read_csv(OUT/'cases.csv')
    terms=[];checks=[]
    for task in tasks:
        suffix=task['task'];dataset,name=suffix.split('/')
        selected=cases[(cases.dataset==dataset)&(cases.task==name)]
        if selected.empty:continue
        folder=OUT/'tasks'/suffix
        data={k:np.load(folder/'input'/(k+'.npy'),mmap_mode='r') for k in ('rows','ids','support')}
        coeff=arrays(task['coefficients'])
        geo=Geometry(task,data,coeff,folder)
        union=np.unique(data['support'])
        support_views=geo.encode(union)
        path,idcol=manifest(task)
        columns=pd.read_csv(path,nrows=0).columns
        used=[k for k in (idcol,'class_id','class_name','mmsi','mmsi_or_ship_id') if k in columns]
        frame=pd.read_csv(path,usecols=used,keep_default_na=False,
                          dtype={k:str for k in used if k!='class_id'}).iloc[data['rows']].reset_index(drop=True)
        for case in selected.itertuples():
            draw,q=int(case.draw),int(case.query_index)
            s=data['support'][draw]
            # Match original PCA float32 casting batches and support-union projection.
            lo=(q//BATCH)*BATCH
            fv=geo.encode(np.arange(lo,min(lo+BATCH,len(data['rows']))))
            kr_all,kb_all=geo.kernels(fv,support_views)
            columns=np.searchsorted(union,s)
            kr,kb=kr_all[q-lo,columns],kb_all[q-lo,columns]
            ra,ta=coeff[f'{draw}_r50_expert'],coeff[f'{draw}_dual_refit_expert']
            y,r=int(case.true_class),int(case.comparison_class)
            da=ta-ra
            old=kr*(da[:,y]-da[:,r])
            extra=kb*(ta[:,y]-ta[:,r])
            np.testing.assert_allclose(old.sum(),case.reweight_margin_delta,atol=1e-8,rtol=1e-8)
            np.testing.assert_allclose(extra.sum(),case.extra_relation_margin_delta,atol=1e-8,rtol=1e-8)
            checks.append(dict(task=suffix,kind=case.kind,query_index=q,
                old_margin_error=float(abs(old.sum()-case.reweight_margin_delta)),
                extra_margin_error=float(abs(extra.sum()-case.extra_relation_margin_delta))))
            for j,index in enumerate(s):
                terms.append(dict(dataset=dataset,task=name,kind=case.kind,query_index=q,
                    query_sample_id=case.sample_id,query_class=y,comparison_class=r,
                    support_slot=j,support_index=int(index),support_sample_id=str(data['ids'][index]),
                    support_class=int(frame.iloc[index].class_id),support_class_name=frame.iloc[index].class_name,
                    extra_similarity=float(kb[j]),extra_margin_term=float(extra[j]),
                    reweight_margin_term=float(old[j]),total_support_margin_change=float(old[j]+extra[j])))
        print('CASES',suffix,len(selected),flush=True)
    frame=pd.DataFrame(terms)
    frame.to_csv(OUT/'support_case_terms.csv',index=False)
    top=[]
    for _,group in frame.groupby(['dataset','task','kind','query_index']):
        for side in ('positive','negative'):
            part=group[group.extra_margin_term>0] if side=='positive' else group[group.extra_margin_term<0]
            part=part.sort_values('extra_margin_term',ascending=side=='negative').head(3).copy()
            part['side']=side;part['rank']=np.arange(1,len(part)+1);top.append(part)
    pd.concat(top,ignore_index=True).to_csv(OUT/'top_support_case_terms.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'case_sum_checks.csv',index=False)
    write(OUT/'CASE_EXPLANATIONS_COMPLETE.json',dict(cases=len(checks),support_terms=len(frame),
        checks=2*len(checks),interpretation='Fixed-coefficient margin accounting, not leave-support-out causal influence'))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
