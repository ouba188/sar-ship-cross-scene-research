"""Post-fit bounded cache diagnostics, without new fits or threshold selection."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
import pandas as pd
import run as edge

HERE=Path(__file__).resolve().parent


def stats(v):
    v=np.asarray(v)
    v=v[np.isfinite(v)]
    if not len(v):
        return dict(n_valid=0,mean=np.nan,q10=np.nan,q25=np.nan,median=np.nan,q75=np.nan,q90=np.nan)
    q=np.quantile(v,[.1,.25,.5,.75,.9])
    return dict(n_valid=len(v),mean=float(v.mean()),q10=q[0],q25=q[1],median=q[2],q75=q[3],q90=q[4])


def main():
    table,x,mass,bank=edge.support.load()
    cache=np.load(HERE/'localcache.npz')
    features=np.load(HERE/'features.npz')
    idx=features['idx']
    role=table.iloc[idx].role.to_numpy()
    y=table.iloc[idx].class_id.to_numpy(dtype=int)
    rows,details,norms=[],[],[]
    for seed in edge.SEEDS:
        saved=np.load(HERE/f'predictions_{seed}.npz')
        anchor=saved['anchor'][idx]
        competing=anchor.copy()
        competing[np.arange(len(idx)),y]=-np.inf
        rival=competing.argmax(-1)
        for arm in ('body','additive','mixed'):
            raw=cache[f'{arm}_raw'][idx]
            valid=cache[f'{arm}_valid'][idx]
            standardized=features[arm][idx]
            for g,name in enumerate(edge.GROUPS):
                t=raw[np.arange(len(idx)),y,g]
                r=raw[np.arange(len(idx)),rival,g]
                ok=valid[np.arange(len(idx)),y,g]&valid[np.arange(len(idx)),rival,g]
                z=standardized[np.arange(len(idx)),y,g]-standardized[np.arange(len(idx)),rival,g]
                for stage in ('utility_train','selection','evaluation'):
                    for c in [-1]+list(range(8)):
                        select=(role==stage)&(True if c<0 else y==c)
                        for metric,value in [('raw_true',t),('raw_rival',r),('raw_gap',t-r),('standardized_gap',z)]:
                            used=select&ok
                            rows.append(dict(seed=seed,arm=arm,role=stage,class_id=c,group=name,metric=metric,
                                             n=int(select.sum()),positive_fraction=float((value[used]>0).mean()) if used.any() else np.nan,
                                             **stats(value[used])))
                for j in range(len(idx)):
                    details.append(dict(seed=seed,arm=arm,table_index=int(idx[j]),obs_id=int(table.iloc[idx[j]].obs_id),
                                        role=role[j],true_class=int(y[j]),rival=int(rival[j]),group=name,
                                        both_valid=bool(ok[j]),raw_true=float(t[j]),raw_rival=float(r[j]),
                                        raw_gap=float(t[j]-r[j]),standardized_gap=float(z[j])))
    qb,qn,legal,_=edge.edges(x,mass)
    difference=np.linalg.norm(qb-qn,axis=-1)
    body=np.linalg.norm(qb,axis=-1)
    for stage in ['all3387','bank64','utility_train','selection','evaluation']:
        chosen=np.ones(len(table),bool)
        if stage=='bank64':
            chosen=np.zeros(len(table),bool)
            chosen[bank.ravel()]=True
        elif stage!='all3387':
            chosen=table.role.to_numpy()==stage
        for c in [-1]+list(range(8)):
            select=chosen&(True if c<0 else table.class_id.to_numpy()==c)
            for g,name in enumerate(edge.GROUPS):
                keep=legal[select,g]
                values=difference[select,g][keep]
                for metric,value in [('difference_norm',values),('body_norm',body[select,g][keep]),
                                     ('mixed_self_similarity',.5*values**2)]:
                    norms.append(dict(role=stage,class_id=c,group=name,metric=metric,observations=int(select.sum()),
                                      exact_zero=int((value==0).sum()),minimum=float(value.min()) if len(value) else np.nan,
                                      maximum=float(value.max()) if len(value) else np.nan,**stats(value)))
    curves=pd.read_csv(HERE/'selection_curve.csv')
    out=[]
    for (seed,arm),frame in curves.groupby(['seed','arm']):
        first=frame[frame.step==0].iloc[0]
        for _,r in frame.iterrows():
            selection_loss=.5*(r.NLL+r.balanced_NLL)
            first_loss=.5*(first.NLL+first.balanced_NLL)
            out.append(dict(seed=int(seed),arm=arm,step=int(r.step),train_joint_CE=r.train_CE,
                            train_objective=r.objective,delta_train_CE=r.train_CE-first.train_CE,
                            delta_train_objective=r.objective-first.objective,
                            selection_joint_CE=selection_loss,delta_selection_joint_CE=selection_loss-first_loss,
                            selection_OA=r.OA,selection_BA=r.BA,selection_J=r.J,
                            delta_selection_J=r.J-first.J,delta_selection_OA=r.OA-first.OA,
                            delta_selection_BA=r.BA-first.BA))
    for name,records in [('raw_margin_diagnosis',rows),('sample_raw_margins',details),
                         ('difference_norms',norms),('checkpoint_diagnosis',out)]:
        pd.DataFrame(records).to_csv(HERE/f'{name}.csv',index=False)
    receipt=dict(status='complete',new_fits=0,new_thresholds=0,cache_only=True,
                 source_only=True,no_git=True,no_images=True,
                 missing='raw-margin summaries require both candidate groups valid; norm summaries use only legal edges',
                 norms='original unit-token difference without renormalization; overlapping edges are not independent samples')
    (HERE/'diagnosis_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('CHECKPOINT DIAGNOSIS')
    print(pd.DataFrame(out)[['seed','arm','step','delta_train_CE','delta_selection_joint_CE','delta_selection_J']].to_string(index=False))
    print('EVALUATION NORMS')
    print(pd.DataFrame(norms).query("role=='evaluation' and class_id==-1")[[
        'group','metric','n_valid','mean','median','q10','q90','exact_zero']].to_string(index=False))
    print('SEED17 EVALUATION GAPS')
    print(pd.DataFrame(rows).query("role=='evaluation' and class_id==-1 and seed==17 and metric=='raw_gap'")[[
        'arm','group','n_valid','mean','median','q10','q90','positive_fraction']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
