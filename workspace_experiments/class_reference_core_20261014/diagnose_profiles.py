"""Check finite reference responses on the actual queried/support cohorts."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from model import reference_map

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent


def main():
    records=[]
    for dataset in ['ports','f0921','oss2']:
        where=ROOT/'scene_applicability_expansion_20261014/encoded.npz' if dataset=='ports' else ROOT/f'external_scene_transfer_20261013/{dataset}_encoded.npz'
        with np.load(where) as f:
            z=np.c_[f['r'],f['t']]; rows=f['rows']
        with (OUT/('ports_banks.pkl' if dataset=='ports' else 'external_banks.pkl')).open('rb') as h:
            banks=pickle.load(h)
        archives=list((OUT/'predictions').glob(dataset+'_*.npz'))
        query=[];support=[]
        for path in archives:
            with np.load(path) as f:
                query.extend(f['query_rows'].tolist());support.extend(f['support_rows'].tolist())
        for role, used in [('query',np.unique(query)),('support',np.unique(support))]:
            selected=z[np.searchsorted(rows,used)]
            for arm in ['class_mean','cluster_0','cluster_1','cluster_2']:
                bank=banks[arm]
                a=-cdist(selected,bank['centers'],'sqeuclidean')/bank['tau2']
                r=np.exp(a); norm=np.linalg.norm(r,axis=1)
                old=r/np.maximum(norm[:,None],1e-15)
                stable=np.exp(a-a.max(axis=1,keepdims=True))
                stable/=np.linalg.norm(stable,axis=1,keepdims=True)
                change=np.linalg.norm(old-stable,axis=1)
                records.append(dict(dataset=dataset,role=role,arm=arm,n=len(used),
                    below_floor=int((norm<1e-15).sum()),all_zero=int((norm==0).sum()),
                    changed_gt1e8=int((change>1e-8).sum()),max_change=float(change.max()),
                    old_trace=float(np.var(old,axis=0).sum()),stable_trace=float(np.var(stable,axis=0).sum()),
                    min_log_response=float(a.min()),max_log_response=float(a.max())))
    pd.DataFrame(records).to_csv(OUT/'reference_activity.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
