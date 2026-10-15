"""Bounded post-result diagnostics: scale transfer, redundancy, coverage, utility."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from run import OUT, METHODS, core
from analyze import weights


def main():
    rows=[]
    source=pd.read_csv(OUT/'source_contexts.csv')
    target=pd.read_csv(OUT/'target_contexts.csv')
    feature_names=['ova','source_distance','support_distance','log_predicted_class_count']
    for kind, contexts in [('source',source),('target',target)]:
        for row in contexts.itertuples():
            z=np.load(OUT/row.path); y=z['labels']; unknown=int(z['unknown'])
            known,w,u=weights(y,unknown)
            counts=np.bincount(z['support_labels'],minlength=8)
            scope=row.role if kind=='source' else row.task
            for i,name in enumerate(feature_names):
                x=z['inputs'][:,i]
                groups={'known':known,'unknown':~known,'unsupported_known':known&(counts[y]==0),
                        'supported_known':known&(counts[y]>0)}
                auc=100*roc_auc_score(~known,x,sample_weight=w+u)
                for group,mask in groups.items():
                    if not mask.any():continue
                    ww=(w+u)*mask;ww/=ww.sum()
                    rows.append(dict(kind=kind,scope=scope,outer=row.outer,arm=row.arm,draw=row.draw,
                                     feature=name,group=group,mean=float(ww@x),auroc=auc,n=int(mask.sum())))
    d=pd.DataFrame(rows)
    d.to_csv(OUT/'feature_diagnostics.csv',index=False)
    ds=d.groupby(['kind','scope','arm','feature','group'])[['mean','auroc','n']].mean().reset_index()
    ds.to_csv(OUT/'feature_summary.csv',index=False)
    m=pd.read_csv(OUT/'metrics.csv')
    # A random rejection rule at matched macro known FRR tests whether U alone implies separation.
    m['matched_frr_random_u20']=.2*m.known_false_reject-.8*m.known_ba_before*m.known_false_reject/100
    m['u20_above_random']=m.utility_20-m.matched_frr_random_u20
    m.groupby(['task','arm','calibration','method'])[['matched_frr_random_u20','u20_above_random']].mean().reset_index().to_csv(OUT/'random_rejection_diagnostic.csv',index=False)
    m['v20']=.8*m.known_ba_after+.2*m.unknown_tpr
    keys=['task','unknown','draw','calibration','method']
    paired=m[m.arm=='cover1'].merge(m[m.arm=='random'],on=keys,suffixes=('_cover','_random'),validate='one_to_one')
    for col in ['v20','known_false_reject','unknown_tpr','known_ba_after']:
        paired['delta_'+col]=paired[col+'_cover']-paired[col+'_random']
    paired.groupby(['task','calibration','method'])[['delta_'+c for c in ['v20','known_false_reject','unknown_tpr','known_ba_after']]].mean().reset_index().to_csv(OUT/'coverage_effect.csv',index=False)
    thresholds=[]
    for row in target[target.arm=='random'].itertuples():
        z=np.load(OUT/row.path)
        for i,method in enumerate(METHODS):
            thresholds.append(dict(task=row.task,outer=row.outer,draw=row.draw,method=method,
                                   source=float(z['source_thresholds'][i]),support=float(z['support_thresholds'][i])))
    pd.DataFrame(thresholds).groupby(['task','method'])[['source','support']].mean().reset_index().to_csv(OUT/'threshold_transfer.csv',index=False)
    print('FEATURE MEANS\n'+ds[(ds.arm=='random')&ds.feature.isin(['source_distance','support_distance'])&ds.group.isin(['known','unknown'])].to_string(index=False))
    print('COVERAGE\n'+pd.read_csv(OUT/'coverage_effect.csv').query("method in ['joint','fixed_union','support_knn']").to_string(index=False))
    core.write(OUT/'DIAGNOSIS_COMPLETE.json',dict(no_new_model_fits=True,target_labels_diagnostic_only=True,
        random_control='expected rejection at same macro known FRR; not a deployable threshold'))


if __name__=='__main__':main()
