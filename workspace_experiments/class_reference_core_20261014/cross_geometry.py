"""Locate pruning loss: cross fixed original-kernel and reference geometries."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
import importlib.util
from pathlib import Path
import pickle
import json
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from model import reference_map

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'geometry_cross'
spec=importlib.util.spec_from_file_location('core_experiment',ROOT/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
KEEP=np.r_[0:128,2176:2240]


def main():
    OUT.mkdir(exist_ok=True);ev=run.old.operator();start=time.perf_counter()
    conditions=pd.read_csv(ROOT/'conditions.csv')
    conditions=conditions[conditions.budget==64]
    metrics=[]; events=[]; replay=[]
    for dataset,g in conditions.groupby('dataset'):
        where=run.EXPAND/'encoded.npz' if dataset=='ports' else run.old.OLD/f'{dataset}_encoded.npz'
        f=run.load_npz(where);z=np.c_[f['r'],f['t']];zl=z[:,KEEP]*np.sqrt(2.)
        metadata=run.old.source_metadata() if dataset=='ports' else pd.read_csv(
            run.old.OLD/f'{dataset}_metadata.csv',dtype={'identity':str},keep_default_na=False)
        bn='ports' if dataset=='ports' else 'external'
        with (ROOT/'normalization_repair'/f'{bn}_banks.pkl').open('rb') as h:
            full_bank=pickle.load(h)['class_mean']
        with (ROOT/'lean192'/f'{bn}_banks.pkl').open('rb') as h:
            lean_bank=pickle.load(h)['class_mean']
        r=reference_map(z,full_bank);rl=reference_map(zl,lean_bank)
        for _,row in g.iterrows():
            key={k:row[k] for k in ['dataset','episode','port','cohort','budget','draw','cell','study']}
            tag=f"{dataset}_{row.episode:03d}_{row.budget}_{row.draw}_{row.cell}.npz"
            full=run.load_npz(ROOT/'normalization_repair/predictions'/tag)
            lean=run.load_npz(ROOT/'lean192/predictions'/tag)
            q=full['query_rows'];s=full['support_rows'];qi=np.searchsorted(f['rows'],q);si=np.searchsorted(f['rows'],s)
            y=metadata.iloc[q].class_id.to_numpy(int);sy=metadata.iloc[s].class_id.to_numpy(int)
            base=2*full['full'][0]-full['expert'][0]
            np.testing.assert_allclose(base,2*lean['full'][0]-lean['expert'][0],atol=1e-10)
            arms={
                'kernel2752_ref2752':full['full'][2],
                'kernel192_ref192':lean['full'][2],
                'original2752':full['full'][0],
                'original192':lean['full'][0]}
            for name,zz,rr in [('kernel2752_ref192',z,rl),('kernel192_ref2752',zl,r)]:
                ss=.75*(zz[si]@zz[si].T+1)+.25*(rr[si]@rr[si].T)
                qs=.75*(zz[qi]@zz[si].T+1)+.25*(rr[qi]@rr[si].T)
                coef=ev.coefficients(ss,f['pd'][si],sy,8.)
                arms[name]=.5*(base+softmax(f['pd'][qi]+qs@coef,axis=1))
            for arm,p in arms.items():
                metrics.append(dict(**key,arm=arm,**run.old.scores(y,p)))
                pred=p.argmax(1);old=full['full'][2].argmax(1)
                for c in np.unique(y):
                    mask=y==c
                    events.append(dict(**key,arm=arm,class_id=int(c),n=int(mask.sum()),
                        rescued=int(((old!=y)&(pred==y)&mask).sum()),
                        harmed=int(((old==y)&(pred!=y)&mask).sum())))
        print('CROSS_DONE',dataset,len(g),flush=True)
    data=pd.DataFrame(metrics);data.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'class_events.csv',index=False)
    ports=data[data.dataset=='ports'].copy();ports['cohort']='all48'
    x=pd.concat([data,ports]);keys=['dataset','study','cohort','budget','cell','arm']
    values=['oa','ba','macro_brier','macro_nll']
    draws=x.groupby(keys+['draw','port'])[values].mean().groupby(keys+['draw']).mean().reset_index()
    draws.to_csv(OUT/'draw_metrics.csv',index=False)
    summary=draws.groupby(keys)[values].mean().reset_index();summary.to_csv(OUT/'summary.csv',index=False)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(
        cells=len(conditions),new_expert_solves=2*len(conditions),seconds=time.perf_counter()-start,
        new_source_fits=0,protocol='Posthoc fixed2x2 path diagnosis, not target-selected deployment',
        variables='Original support kernel dimension vs source-class reference dimension; priors/base/labels fixed'),indent=2),encoding='utf-8')
    print(summary[summary.cell=='other_acquisition'].to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
