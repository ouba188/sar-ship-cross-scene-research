"""Persist objective and protocol diagnostics without tuning on target outcomes."""

import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='2'
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from run_directional_fusion_trial import OUT, SOURCE, archive, write_json


def main():
    labels=pd.read_csv(SOURCE/'manifest.csv',usecols=['class_id']).class_id.to_numpy()
    records=[]
    for folder in sorted((OUT/'ports').iterdir()):
        p=archive(folder/'policies.npz'); y=labels[p['query_rows']]
        for draw,ev in enumerate(p['eval_mask']):
            for i,name in enumerate(p['modes']):
                s=p['base_scores'][draw,i,ev]; truth=y[ev]
                loss=logsumexp(s,axis=1)-s[np.arange(len(truth)),truth]
                records.append(dict(port=folder.name,draw=draw,mode=name,balanced_ce=float(np.mean([loss[truth==c].mean() for c in np.unique(truth)]))))
    d=pd.DataFrame(records); d.to_csv(OUT/'target_base_loss.csv',index=False)
    d.groupby(['port','mode']).balanced_ce.mean().groupby('mode').mean().to_csv(OUT/'target_base_loss_means.csv')
    branches=pd.concat([pd.read_csv(p) for p in (OUT/'ports').glob('*/source_episode_metrics.csv')])
    branches.groupby(['outer_port','mode'])[['oa','ba']].mean().groupby('mode').mean().to_csv(OUT/'shared_geometry_source_branch_means.csv')
    a=archive(OUT/'nested_geometry_probe/ports/Callao/source_predictions.npz')
    b=archive(OUT/'nested_geometry_gpu/ports/Callao/source_predictions.npz')
    for key in ['rows','labels','episode']: np.testing.assert_array_equal(a[key],b[key])
    parity={k:float(np.max(np.abs(a[k]-b[k]))) for k in ['original','c4','context']}
    if max(parity.values())>3e-4: raise ValueError('GPU projection changes nested predictions excessively')
    write_json(OUT/'nested_geometry_gpu/CPU_PARITY.json',dict(scope='Callao outer fold, all22source episodes',max_score_error=parity,
        mismatched_argmax={k:int(np.count_nonzero(a[k].argmax(1)!=b[k].argmax(1))) for k in ['original','c4']},
        cpu='float64 numpy projection, scipy CPU environment',gpu='float32 highest-precision Torch matmul; float64 downstream fits'))
    nested=[]
    for path in (OUT/'nested_geometry_gpu/ports').glob('*/fusion_models.json'):
        models=json.loads(path.read_text()); shared=json.loads((OUT/'ports'/path.parent.name/'fusion_models.json').read_text())
        for c in range(8):
            nested.append(dict(port=path.parent.name,class_id=c,shared_alpha=shared['class_mix']['alpha'][c],nested_alpha=models['class_mix']['alpha'][c],
                shared_global=shared['global_mix']['alpha'],nested_global=models['global_mix']['alpha']))
    pd.DataFrame(nested).to_csv(OUT/'shared_vs_nested_weights.csv',index=False)
    transitions=[]
    corrected=OUT/'nested_corrected_target'
    for folder in sorted((corrected/'ports').iterdir()):
        p=archive(folder/'policies.npz'); y=labels[p['query_rows']]
        cslot=list(p['modes']).index('class_mix')
        denom=p['eval_mask'].sum(0)
        for stage,pred in [('base',p['base_scores'].argmax(3)),('full',p['predictions'])]:
            for draw,ev in enumerate(p['eval_mask']):
                before,after=pred[draw,0],pred[draw,cslot]
                for cl in [3,5,6]:
                    mask=ev&(y==cl)&(before!=after)
                    for a,b in np.unique(np.c_[before[mask],after[mask]],axis=0):
                        chosen=mask&(before==a)&(after==b)
                        transitions.append(dict(port=folder.name,draw=draw,stage=stage,class_id=cl,
                            from_prediction=int(a),to_prediction=int(b),query_events=int(chosen.sum()),
                            image_weighted_events=float((1./denom[chosen]).sum())))
    pd.DataFrame(transitions).to_csv(corrected/'class_competition_transitions.csv',index=False)


if __name__=='__main__': main()
