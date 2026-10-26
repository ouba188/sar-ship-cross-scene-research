"""Source-only scalar shrinkage of frozen matched M3 endpoints."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'paper_evidence_completion_20261026'))
import run_classical as base
OLD=ROOT/'raw_region_bypass_20261026/downstream/raw_summary'
NEWBASE=ROOT/'raw_region_bypass_20261026'
BETAS=(0.,.25,.5,.75,1.)
ENDPOINTS=('B_fixed','frequency')
ARMS=('bare','B_fixed','frequency','B_fixed_scaled','frequency_scaled')


def shrink(bare, endpoint, beta):
    if beta==0:
        return bare.copy()
    if beta==1:
        return endpoint.copy()
    return bare+np.float32(beta)*(endpoint-bare)


def fit():
    choices, curves, checks=[] ,[],[]
    for source,other in (('oss1','oss2'),('oss2','oss1')):
        direction=f'{source}-to-{other}'
        table=pd.read_csv(ROOT/f'{source}_m123_replication_20261026/stack_table.csv',dtype={'mmsi':str})
        target=base.tr.read_table(base.TRANSFER/direction/'target_table.csv')
        assert 'class_id' not in target.columns
        selection=np.flatnonzero(table.role.eq('selection'))
        for field in ('mmsi','pass_group'):
            assert not set(table.loc[table.role.ne('evaluation'),field]) & set(target[field])
            assert not set(table.iloc[selection][field]) & set(table.loc[table.role.isin(['fit_train','utility_train']),field])
        banks={side:base.tr.load_npz(NEWBASE/source/f'{side}_predictions.npz') for side in ('source','cross')}
        for seed in base.SEEDS:
            folder=OLD/direction/f'seed{seed}'
            caches={side:base.tr.load_npz(folder/f'{side}_predictions.npz') for side in ('source','cross')}
            for side,tab in (('source',table),('cross',target)):
                z=caches[side]
                np.testing.assert_array_equal(z['obs_id'],tab.obs_id)
                np.testing.assert_array_equal(banks[side]['obs_id'],tab.obs_id)
                np.testing.assert_array_equal(z['bare'],banks[side][f'raw_summary_seed{seed}'])
                for endpoint in ENDPOINTS:
                    w=z[f'weight_{endpoint}']
                    score=z['bare']+w[:,:1]*z['delta_vv']+w[:,1:]*z['delta_vh']
                    np.testing.assert_array_equal(score,z[endpoint])
                    np.testing.assert_array_equal(shrink(z['bare'],z[endpoint],0),z['bare'])
                    np.testing.assert_array_equal(shrink(z['bare'],z[endpoint],1),z[endpoint])
                checks.append(dict(source=source,seed=seed,side=side,base_endpoint_identity=True,weight_closure=True))
            selected={}
            z=caches['source']
            labels=table.iloc[selection].class_id.to_numpy(int)
            for endpoint in ENDPOINTS:
                candidates=[]
                for beta in BETAS:
                    score=shrink(z['bare'][selection],z[endpoint][selection],beta)
                    j=base.ranking(labels,score)
                    candidates.append((j,beta))
                    curves.append(dict(source=source,seed=seed,endpoint=endpoint,beta=beta,
                        **base.classification_metrics(labels,score)))
                best,beta=max(candidates,key=lambda v:(v[0],-v[1]))
                selected[endpoint]=beta
                choices.append(dict(source=source,seed=seed,endpoint=endpoint,beta=beta,selection_n=len(selection),J=float(best)))
            out=HERE/direction/f'seed{seed}'
            out.mkdir(parents=True,exist_ok=True)
            for side,z in caches.items():
                scores={arm:z[arm] for arm in ('bare',*ENDPOINTS)}
                for endpoint,beta in selected.items():
                    scores[endpoint+'_scaled']=shrink(z['bare'],z[endpoint],beta)
                np.savez_compressed(out/f'{side}_predictions.npz',obs_id=z['obs_id'],**scores)
            base.write_json(out/'selection.json',selected)
    pd.DataFrame(choices).to_csv(HERE/'selections.csv',index=False)
    pd.DataFrame(curves).to_csv(HERE/'source_curves.csv',index=False)
    base.write_json(HERE/'checks.json',checks)
    base.write_json(HERE/'predictions_saved.json',dict(configurations=6,beta_selections=12,
        new_neural_training=False,target_selection=False,beta_grid=BETAS,
        all_saved_before_target_evaluation=True,endpoint_inputs_frozen=True))
    print(pd.DataFrame(choices).to_string(index=False),flush=True)


def evaluate():
    assert (HERE/'predictions_saved.json').exists()
    rows=[]
    for source,other in (('oss1','oss2'),('oss2','oss1')):
        direction=f'{source}-to-{other}'
        tables={'source':pd.read_csv(ROOT/f'{source}_m123_replication_20261026/stack_table.csv',dtype={'mmsi':str}),
                'cross':base.tr.read_table(base.TRANSFER/direction/'target_table.csv',labels=True)}
        for side,table in tables.items():
            control=base.tr.load_npz(base.HERE/f'classical_fit_only/{source}/{side}_predictions.npz')
            np.testing.assert_array_equal(control['obs_id'],table.obs_id)
            for seed in base.SEEDS:
                z=base.tr.load_npz(HERE/direction/f'seed{seed}/{side}_predictions.npz')
                np.testing.assert_array_equal(z['obs_id'],table.obs_id)
                scores={arm:z[arm] for arm in ARMS}
                scores['raw_mlp']=control[f'regional_mlp_seed{seed}']
                if seed==base.SEEDS[0]:
                    scores['raw_ridge']=control['regional_ridge']
                for role,ix in table.groupby('role').indices.items():
                    y=table.iloc[ix].class_id.to_numpy(int)
                    for arm,score in scores.items():
                        rows.append(dict(source=source,side=side,role=role,seed=seed if arm!='raw_ridge' else -1,
                            arm=arm,**base.classification_metrics(y,score[ix])))
    frame=pd.DataFrame(rows)
    frame.to_csv(HERE/'metrics.csv',index=False)
    summary=frame.groupby(['source','side','role','arm'])[['OA','BA','NLL','Brier']].agg(['mean','std'])
    summary.to_csv(HERE/'summary.csv')
    base.write_json(HERE/'complete.json',dict(calibration_finished=True,all_six_configs_evaluated=True,raw_review_pending=True))
    print(frame.query("side=='cross'").groupby(['source','arm'])[['OA','BA']].mean().to_string(),flush=True)


if __name__=='__main__':
    fit()
    evaluate()
