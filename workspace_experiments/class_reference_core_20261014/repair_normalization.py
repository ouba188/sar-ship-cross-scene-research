"""One bounded numerical repair; frozen centers, data and model choices."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import pickle
import json
import time
import importlib.util
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from model import ReferenceSupportModel, unit_response

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'normalization_repair'

spec=importlib.util.spec_from_file_location('core_experiment',ROOT/'run.py')
run=importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
spec=importlib.util.spec_from_file_location('core_analysis',ROOT/'analyze.py')
analyze=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def recalibrate(z, labels, template, name):
    t=time.perf_counter()
    with (ROOT/f'{name}_banks.pkl').open('rb') as h:
        banks=pickle.load(h)
    records=[]
    for arm, bank in banks.items():
        if bank is None:
            continue
        r=unit_response(z,bank['centers'],bank['tau2'])
        center=r.mean(0)
        scale=np.sqrt(np.mean(np.sum(z*z,axis=1)+1)/np.mean(np.sum((r-center)**2,axis=1)))
        records.append(dict(domain=name,arm=arm,
            center_max_change=float(np.max(abs(center-bank['center']))),
            scale_old=float(bank['scale']),scale_new=float(scale),
            max_unit_norm_error=float(np.max(abs(np.linalg.norm(r,axis=1)-1)))))
        bank['center']=center;bank['scale']=scale
    with (OUT/f'{name}_banks.pkl').open('wb') as h:
        pickle.dump(banks,h)
    pd.DataFrame(records).to_csv(OUT/f'{name}_recalibration.csv',index=False)
    return banks,dict(domain=name,n=len(z),seconds=time.perf_counter()-t,
                      fitting='Recalibrated only source center and energy; frozen centers/bandwidths')


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'PROTOCOL.json').write_text(json.dumps(dict(
        status='One posthoc implementation correction, not independent confirmation',
        trigger='F reference norm floor changes intended L2 response; 552/4812 unique query observations affected',
        fix='Subtract row maximum from negative squared-distance logits before exp and L2; no epsilon floor',
        changed='Same numerical fix for class and all3cluster arms; source response center/energy recomputed',
        fixed='All source centers, source rows, bandwidths, queries/supports, rho.25, ridge8, fusion.5',
        preserves='Parent directory contains initial experiment, legacy_model.py reproduces it',
        no_hyperparameter_search=True),indent=2),encoding='utf-8')
    started=time.perf_counter();run.OUT=OUT;run.banks=recalibrate
    ev=run.old.operator()
    contexts,info=run.prepare_external(ev)
    ports,pinfo=run.prepare_ports();contexts.append(ports)
    run.write('BANK_FITS.json',[info,pinfo])
    run.evaluate(contexts,ev,started,model_type=ReferenceSupportModel,
                 replay_arms=(('original','old_original'),))
    analyze.OUT=OUT
    # Identical cluster assignments: copy their counts as an analysis input.
    for domain in ['ports','external']:
        frame=pd.read_csv(ROOT/f'{domain}_cluster_composition.csv')
        frame.to_csv(OUT/f'{domain}_cluster_composition.csv',index=False)
    analyze.main()
    keys=analyze.KEY+['arm']
    before=pd.read_csv(ROOT/'absolute_summary.csv');after=pd.read_csv(OUT/'absolute_summary.csv')
    pair=before.merge(after,on=keys,suffixes=('_before','_after'),validate='one_to_one')
    for metric in analyze.VALUES:
        pair[metric+'_repair_delta']=pair[metric+'_mean_after']-pair[metric+'_mean_before']
    pair.to_csv(OUT/'repair_effects.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
