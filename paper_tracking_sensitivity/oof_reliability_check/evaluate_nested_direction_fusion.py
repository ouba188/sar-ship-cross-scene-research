"""Apply inner-geometry-held source fusion rules to untouched target policies."""

import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='2'
import json
import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from run_directional_fusion_trial import OUT as TRIAL, PARENT, REFERENCE, C4, SOURCE, LEARNED, inputs, archive, write_json
from direction_score_fusion import predict_fusion
from evidence_aware_handoff import predict_gate, route_scores
from run_protected_drift_activation import gate_features
from run_generic_transfer import metrics

OUT=TRIAL/'nested_corrected_target'
NESTED=TRIAL/'nested_geometry_gpu'
MODELS=NESTED
MODES=('c4','half_mix',*LEARNED)


def freeze(frame,context):
    for parent in sorted((TRIAL/'ports').iterdir()):
        port=parent.name
        output=OUT/'ports'/port; output.mkdir(parents=True,exist_ok=True)
        if (output/'FROZEN.json').exists(): continue
        source=NESTED/'ports'/port
        if not (source/'COMPLETE.json').exists(): raise ValueError(f'Missing corrected source rules {port}')
        models=json.loads((MODELS/'ports'/port/'fusion_models.json').read_text())
        if not all(m['success'] for m in models.values()): raise ValueError(f'Unconverged rule {port}')
        original=archive(REFERENCE/port/'policies.npz'); reference=archive(C4/'ports'/port/'policies.npz')
        for key in ['query_rows','support_rows','eval_mask']:
            np.testing.assert_array_equal(original[key],reference[key])
        oldslot=list(original['mode_names']).index('direct_condition'); cslot=list(reference['base_modes']).index('matched_c4')
        with (PARENT/port/'source_models.pkl').open('rb') as f: gate=pickle.load(f)['gates']['joint']
        rows=reference['query_rows']; bases=[]; predictions=[]; selected=[]; weights=[]
        for draw,ev in enumerate(reference['eval_mask']):
            old=original['base_scores'][draw,oldslot]; c4=reference['base_scores'][draw,cslot]; expert=original['expert_scores'][draw,oldslot]
            support_y=frame.loc[reference['support_rows'][draw],'class_id'].to_numpy()
            if np.intersect1d(reference['support_rows'][draw],rows[ev]).size: raise ValueError('Support/query overlap')
            scores={'c4':c4,'half_mix':.5*(old+c4)}
            for name in LEARNED:
                scores[name],alpha=predict_fusion(models[name],old,c4,context[rows])
                if alpha.size:
                    weights.append(dict(port=port,draw=draw,mode=name,**{f'class_{c}':float(alpha[ev,c].mean()) for c in range(8)}))
            bp=[]; pp=[]; ss=[]
            for name in MODES:
                base=scores[name]; routed=route_scores(base,expert,predict_gate(gate,gate_features(base,expert)),support_y,ev,rows)
                if routed['selected'][~ev].any() or routed['selected'].sum()>int(.05*ev.sum()): raise ValueError('Budget mismatch')
                if name=='c4': np.testing.assert_array_equal(routed['predictions'],reference['full_predictions'][draw,1])
                bp.append(base); pp.append(routed['predictions']); ss.append(routed['selected'])
            bases.append(np.stack(bp)); predictions.append(np.stack(pp)); selected.append(np.stack(ss))
        np.savez_compressed(output/'policies.npz',query_rows=rows,support_rows=reference['support_rows'],eval_mask=reference['eval_mask'],
            modes=np.asarray(MODES),base_scores=np.stack(bases),predictions=np.stack(predictions),selected=np.stack(selected))
        pd.DataFrame(weights).to_csv(output/'alpha.csv',index=False)
        write_json(output/'FROZEN.json',dict(port=port,source_geometry_inner_held=True,target_query_labels_used=0))
    write_json(OUT/'ALL_FROZEN.json',dict(ports=23,correction='Inner held-port geometry during source rule fitting only; target classifiers unchanged'))


def report(frame):
    records=[]; class_rows=[]; fits=[]; diagnostics=[]; alphas=[]
    for folder in sorted((OUT/'ports').iterdir()):
        p=archive(folder/'policies.npz'); y=frame.loc[p['query_rows'],'class_id'].to_numpy()
        alphas.append(pd.read_csv(folder/'alpha.csv'))
        fitted=json.loads((MODELS/'ports'/folder.name/'fusion_models.json').read_text())
        for name,model in fitted.items():
            diagnostics.append(dict(port=folder.name,mode=name,success=model['success'],nit=model['nit'],loss=model['training_logloss']))
        fits.append(pd.read_csv(NESTED/'ports'/folder.name/'branch_metrics.csv').assign(outer_port=folder.name))
        for draw,ev in enumerate(p['eval_mask']):
            for mi,name in enumerate(MODES):
                score=p['base_scores'][draw,mi,ev]; yy=y[ev]
                loss=logsumexp(score,axis=1)-score[np.arange(len(yy)),yy]
                ce=np.mean([loss[yy==cl].mean() for cl in np.unique(yy)])
                for stage,all_pred in [('base',p['base_scores'][draw].argmax(2)),('full',p['predictions'][draw])]:
                    pred=all_pred[mi]; ref=all_pred[0]; m=metrics(yy,pred[ev],8)
                    records.append(dict(port=folder.name,draw=draw,mode=name,stage=stage,n=int(ev.sum()),oa=m['acc'],ba=m['ba'],base_balanced_ce=ce,
                        rescues=int((ev&(ref!=y)&(pred==y)).sum()),harms=int((ev&(ref==y)&(pred!=y)).sum())))
        denom=p['eval_mask'].sum(0); valid=denom>0
        for stage,all_pred in [('base',p['base_scores'].argmax(3)),('full',p['predictions'])]:
            ref=all_pred[:,0]
            for mi,name in enumerate(MODES):
                pred=all_pred[:,mi]
                for cl in np.unique(y[valid]):
                    mask=valid&(y==cl)
                    correct=((pred==y)*p['eval_mask']).sum(0)[mask]/denom[mask]
                    rescue=(((ref!=y)&(pred==y))*p['eval_mask']).sum(0)[mask]/denom[mask]
                    harm=(((ref==y)&(pred!=y))*p['eval_mask']).sum(0)[mask]/denom[mask]
                    class_rows.append(dict(port=folder.name,stage=stage,mode=name,class_id=int(cl),n=int(mask.sum()),
                        correct_sum=float(correct.sum()),rescue_sum=float(rescue.sum()),harm_sum=float(harm.sum())))
    df=pd.DataFrame(records); df.to_csv(OUT/'metrics.csv',index=False)
    pm=df.groupby(['port','stage','mode']).mean(numeric_only=True).reset_index(); pm.to_csv(OUT/'port_means.csv',index=False)
    nclass=frame.groupby('port').class_id.nunique(); summaries=[]
    for group,mask in [('all23',np.ones(len(pm),dtype=bool)),('strict8_12',pm.port.map(nclass).eq(8))]:
        s=pm[mask].groupby(['stage','mode']).agg(ports=('port','nunique'),oa=('oa','mean'),ba=('ba','mean'),base_balanced_ce=('base_balanced_ce','mean')).reset_index()
        s.insert(0,'group',group); summaries.append(s)
    summary=pd.concat(summaries); summary.to_csv(OUT/'summary.csv',index=False)
    pd.concat(alphas).to_csv(OUT/'alpha.csv',index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'port_class_sums.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'solver_diagnostics.csv',index=False)
    fit=pd.concat(fits); fit.to_csv(OUT/'source_branch_metrics.csv',index=False)
    fit.groupby(['outer_port','mode'])[['oa','ba']].mean().groupby('mode').mean().to_csv(OUT/'source_branch_means.csv')
    write_json(OUT/'COMPLETE.json',dict(ports=23,draws=5,correction='Nested source geometry instead of shared geometry',target_queries_used_in_learning=False,
        experts_and_gate_unchanged=True,independent_confirmation=False,original_negative_preserved=True))
    print(summary[summary.stage=='full'].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-root',type=Path,default=MODELS)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args(); MODELS=args.model_root; OUT=args.output
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETE.json').exists(): raise ValueError('Preserve corrected experiment')
    frame,context=inputs(); freeze(frame,context); report(frame)
