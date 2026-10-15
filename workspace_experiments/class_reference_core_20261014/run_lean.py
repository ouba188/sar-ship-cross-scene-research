"""Actually remove PCA-complement coordinates, retaining a complete reader."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
import importlib.util
from pathlib import Path
import pickle
import time
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from model import ReferenceSupportModel,unit_response,cluster_bank

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'lean192'
KEEP=np.r_[0:128,2176:2240]
spec=importlib.util.spec_from_file_location('core_experiment',ROOT/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
spec=importlib.util.spec_from_file_location('core_analysis',ROOT/'analyze.py')
analyze=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyze)


def transform(z):
    # Existing retained blocks carry sqrt(.5) from the retained/complement mixture.
    assert z.shape[1]==2752
    return z[:,KEEP]*np.sqrt(2.)


def lean_banks(full_z, labels, template, name):
    started=time.perf_counter();z=transform(full_z)
    rng=np.random.default_rng(20261014)
    left=rng.choice(len(z),2048,replace=True);right=rng.choice(len(z),2048,replace=True)
    d=np.sum((z[left]-z[right])**2,axis=1);tau2=float(np.median(d[d>0]))
    centers=np.stack([z[labels==c].mean(0) for c in range(8)])
    r=unit_response(z,centers,tau2);center=r.mean(0)
    scale=np.sqrt(np.mean(np.sum(z*z,axis=1)+1)/np.mean(np.sum((r-center)**2,axis=1)))
    template=dict(centers=centers,tau2=tau2,center=center,scale=scale)
    banks={'original':None,'shrink':None,'class_mean':template};counts=[]
    for i,seed in enumerate(run.SEEDS):
        banks[f'cluster_{i}'],assignment=cluster_bank(z,template,seed)
        for k in range(8):
            for c in range(8):
                counts.append(dict(domain=name,seed=seed,cluster=k,class_id=c,
                                   n=int(np.sum((assignment==k)&(labels==c)))))
    with (OUT/f'{name}_banks.pkl').open('wb') as h:
        pickle.dump(banks,h)
    pd.DataFrame(counts).to_csv(OUT/f'{name}_cluster_composition.csv',index=False)
    return banks,dict(domain=name,n=len(z),dim=192,tau2=tau2,seconds=time.perf_counter()-started)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'PROTOCOL.json').write_text(json.dumps(dict(
        status='Frozen before lean192 outcomes; separate from numeric normalization repair',
        purpose='Remove actual PCA-complement blocks rather than renaming their presence',
        expert_features='Source-normalized retained R50 128 + SSL64, each divided by sqrt192',
        removed='2048 R50 discarded-PCA coordinates and512 SSL discarded-PCA coordinates',
        fixed='Same C4base, source scores/heads, support/query identities, rho.25,ridge8,fusion.5',
        source_fit='New192d reference centers, common source bandwidth and normalization;3KMeans seeds',
        fair_primary='Lean192 class reference versus lean192 original and3equal-capacity cluster controls',
        secondary='Absolute performance versus retained+complement2752 reader',
        not_done='No backbone retraining, no legacy main replacement, no target tuning'),indent=2),encoding='utf-8')
    t=time.perf_counter();run.OUT=OUT;run.banks=lean_banks;ev=run.old.operator()
    contexts,info=run.prepare_external(ev);ports,pinfo=run.prepare_ports();contexts.append(ports)
    for _,f,_,_,_ in contexts:
        f['z']=transform(f['z'])
    run.write('BANK_FITS.json',[info,pinfo])
    run.evaluate(contexts,ev,t,model_type=ReferenceSupportModel,replay_arms=())
    analyze.OUT=OUT;analyze.main()
    before=pd.read_csv(ROOT/'normalization_repair/absolute_summary.csv')
    after=pd.read_csv(OUT/'absolute_summary.csv');keys=analyze.KEY+['arm']
    pair=before.merge(after,on=keys,suffixes=('_2752','_192'),validate='one_to_one')
    for metric in analyze.VALUES:
        pair[metric+'_lean_delta']=pair[metric+'_mean_192']-pair[metric+'_mean_2752']
    pair.to_csv(OUT/'lean_absolute_effects.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
