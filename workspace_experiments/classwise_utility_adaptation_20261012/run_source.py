"""Frozen source-fold selection followed by disjoint-vessel source readout."""

import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import time
import argparse
import numpy as np
import pandas as pd
from scipy.special import softmax

OUT=Path(__file__).resolve().parent
STUDY=OUT.parent/'same_vessel_condition_study_20261010'
BUDGET=OUT.parent/'frozen_candidate_coverage_20261012/budget'
sys.path.insert(0,str(BUDGET))
sys.path.insert(0,str(OUT.parent/'prior_protection_20261012'))
import run_coverage as c
from core import TAUS,MODES,fit_gate,support_oof,loss_grad
from prior_rules import penalties,solve_weights

CONTEXTS=('Los Angeles','Port Klang','Santos')
REFERENCE=STUDY/'support_budget_fusion_select_20261012'


def arm_name(mode,tau):
    return mode+'_t'+str(tau).replace('.','p')


def subset(path,keys):
    with np.load(path) as z:
        return {k:z[k] for k in keys}


def probabilities(base,expert):
    return .5*(softmax(base,axis=1)+softmax(expert,axis=1))


def metrics(y,p):
    correct=p.argmax(1)==y
    loss=np.square(p-np.eye(8)[y]).sum(1)
    return dict(oa=100*correct.mean(),ba=100*np.mean([correct[y==k].mean() for k in np.unique(y)]),
        macro_brier=np.mean([loss[y==k].mean() for k in np.unique(y)]),brier=loss.mean())


def freeze():
    started=time.perf_counter()
    meta=pd.read_csv(STUDY/'observations.csv',dtype={'mmsi':str})
    folds=c.archive(STUDY/'crossfit_evidence_20261011/source_folds.npz')
    logs,gates=[],[]
    for context in CONTEXTS:
        maps=subset(STUDY/'case_relation_headroom_20261012'/context/'maps.npz',('obs_ids','single','prior'))
        bd=c.archive(REFERENCE/'base_cache'/context/'base.npz')
        index=pd.Index(maps['obs_ids'])
        np.testing.assert_array_equal(bd['obs_ids'],maps['obs_ids'])
        nfit=len(folds['fit_obs_ids'])
        np.testing.assert_array_equal(maps['obs_ids'][:nfit],folds['fit_obs_ids'])
        fit_prior=subset(STUDY/'case_relation_student_20261012'/context/'source_readers.npz',('fit_prior',))['fit_prior']
        for family in ('selection','source'):
            for path in sorted((REFERENCE/family/context).glob('*.npz')):
                old=c.archive(path)
                s,q=index.get_indexer(old['support_ids']),index.get_indexer(old['query_ids'])
                budget=len(s)
                assert (s>=0).all() and (q>=0).all()
                sm=meta.iloc[old['support_ids']]
                sy=sm.class_id.to_numpy(int)
                groups=sm.mmsi.to_numpy(str)
                assert not set(groups)&set(meta.iloc[old['query_ids']].mmsi)
                prior=fit_prior if family=='selection' else maps['prior']
                bp=bd['prior_oof'] if family=='selection' else bd['prior_held']
                if family=='selection':
                    assert max(s.max(),q.max())<nfit
                else:
                    assert min(s.min(),q.min())>=nfit
                xs,xq=maps['single'][s],maps['single'][q]
                ke=xs@xs.T
                parts={k:bd[k][s] for k in ('phi','mean','conditions')}
                kb=c.base_kernel(parts,bd['embedding'][s].astype(float),np.arange(budget),float(bd['eta']))
                oof,oy,parts,notes=support_oof(ke,prior[s],kb,bp[s],sy,groups,c.coefficients)
                ae=c.coefficients(ke,prior[s],sy,budget/8.)
                residual=(xq@xs.T)@ae
                raw=prior[q]+residual
                expected=old['expert'][0] if family=='selection' else old['expert_raw']
                np.testing.assert_allclose(raw,expected,atol=1e-9,rtol=0)
                np.testing.assert_array_equal(raw.argmax(1),expected.argmax(1))
                # The same base operator supplies validation and query scores.
                joined=np.r_[s,q]
                bk=c.base_kernel({k:bd[k][joined] for k in ('phi','mean','conditions')},
                    bd['embedding'][joined].astype(float),np.arange(budget),float(bd['eta']))
                base=bp[q]+bk[budget:]@c.coefficients(kb,bp[s],sy,budget/8.)
                np.testing.assert_allclose(base,old['base'],atol=1e-9,rtol=0)
                values=dict(raw=raw,base=base,prior=prior[q],residual=residual,
                            support_ids=old['support_ids'],query_ids=old['query_ids'])
                count_w=solve_weights(xs,prior[s],sy,penalties(sy,prior[s],budget/8.,'count'))
                values['count']=prior[q]+xq@count_w
                values['fixed_scaling']=prior[q]+(.25 if budget==16 else 1.)*residual
                fit_rows=[]
                for mode in MODES:
                    for tau in TAUS:
                        arm=arm_name(mode,tau)
                        gate,optim=fit_gate(*oof,oy,tau,mode)
                        values[arm]=prior[q]+gate*residual
                        values['gate_'+arm]=gate
                        valrisk=loss_grad(gate,*oof,oy,0.)[0]
                        rawrisk=loss_grad(np.ones(8),*oof,oy,0.)[0]
                        for cls in range(8):
                            gates.append(dict(family=family,context=context,unit=path.stem,budget=budget,
                                mode=mode,tau=tau,class_id=cls,support_n=int((sy==cls).sum()),gamma=gate[cls],
                                oof_risk_gain=rawrisk-valrisk))
                        fit_rows.append(dict(arm=arm,optimizers=optim))
                dest=OUT/family/context
                dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/path.name,**values)
                np.savez_compressed(dest/(path.stem+'_oof.npz'),prior=oof[0],residual=oof[1],
                    base_probability=oof[2],labels=oy,support_labels=sy,groups=groups)
                c.write(dest/(path.stem+'_fit.json'),dict(folds=parts,warnings=notes,optimizers=fit_rows))
                logs.append(dict(family=family,context=context,unit=path.stem,budget=budget,
                    raw_error=float(abs(raw-expected).max()),base_error=float(abs(base-old['base']).max())))
            print('FROZEN',context,family,flush=True)
    pd.DataFrame(logs).to_csv(OUT/'replays.csv',index=False)
    pd.DataFrame(gates).to_csv(OUT/'gates.csv',index=False)
    c.write(OUT/'SOURCE_PREDICTIONS_FROZEN.json',dict(units=len(logs),
        selection_units=sum(z['family']=='selection' for z in logs),held_units=sum(z['family']=='source' for z in logs),
        taus=TAUS,modes=MODES,new_neural_training=0,query_labels_in_gate_fit=False,
        source_selection_labels_in_prior='Original source vessel-OOF priors',
        seconds=time.perf_counter()-started,runtime=sys.executable))


def evaluate(family):
    assert c.read_json(OUT/'SOURCE_PREDICTIONS_FROZEN.json')['units']==108
    meta=pd.read_csv(STUDY/'observations.csv')
    records=[]
    for context in CONTEXTS:
        for path in sorted((OUT/family/context).glob('*.npz')):
            if path.stem.endswith('_oof'):
                continue
            z=c.archive(path)
            y=meta.iloc[z['query_ids']].class_id.to_numpy(int)
            baseline=metrics(y,probabilities(z['base'],z['raw']))
            for arm in ['raw','count','fixed_scaling']+[arm_name(m,t) for m in MODES for t in TAUS]:
                score=metrics(y,probabilities(z['base'],z[arm]))
                records.append(dict(family=family,context=context,unit=path.stem,budget=len(z['support_ids']),arm=arm,
                    **score,**{'delta_'+k:score[k]-baseline[k] for k in score}))
    f=pd.DataFrame(records)
    f.to_csv(OUT/(family+'_metrics.csv'),index=False)
    f.groupby(['budget','arm']).mean(numeric_only=True).reset_index().to_csv(OUT/(family+'_summary.csv'),index=False)
    if family=='selection':
        selected={}
        for mode in MODES:
            vals=[]
            for tau in TAUS:
                sub=f[f.arm==arm_name(mode,tau)]
                vals.append((float(sub.macro_brier.mean()),-tau,tau))
            selected[mode]=min(vals)[2]
        c.write(OUT/'FROZEN_CHOICE.json',dict(taus=selected,criterion='Source FIT equal context/fold/draw/budget macro Brier',
            source_held_used=False,target_used=False))
    choice=c.read_json(OUT/'FROZEN_CHOICE.json')['taus']
    arms=['raw','count','fixed_scaling']+[arm_name(m,t) for m,t in choice.items()]
    selected=f[f.arm.isin(arms)]
    selected.to_csv(OUT/(family+'_selected_metrics.csv'),index=False)
    print(family,selected.groupby(['budget','arm'])[['oa','ba','delta_oa','delta_ba','delta_macro_brier']].mean().to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','evaluate','all'])
    args=parser.parse_args()
    with c.threadpool_limits(limits=2):
        if args.stage in ('freeze','all'):
            freeze()
        if args.stage in ('evaluate','all'):
            evaluate('selection')
            evaluate('source')
