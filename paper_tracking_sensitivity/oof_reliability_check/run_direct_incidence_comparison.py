"""Fixed 2x2: temporal response correction versus direct incidence information."""

import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='2'

import json
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from direct_incidence_conditioning import condition_features, extend_transforms, predict_conditioned
from run_conditioned_vessel_drift import OUT as PROBE, SIDECAR
from run_incidence_correction_trial import OUT as CORRECTION, predict_joint
from run_same_vessel_temporal_probe import jacobians
from run_protected_drift_activation import make_views, make_transforms, gate_features
from protected_drift_activation import fit_source_heads
from evidence_aware_handoff import predict_gate, route_scores
from task_stable_sensitivity import SOURCE, PARENT, ROOT
from sar_joint_completion import write_json
from run_generic_transfer import metrics


OUT=ROOT/'direct_incidence_comparison_20261009'
MODES=('reference','same_vessel_correction','direct_condition','correction_plus_condition')


def freeze_port(folder,frame,raw):
    out=OUT/'ports'/folder.name
    out.mkdir(parents=True,exist_ok=True)
    with (folder/'geometry.pkl').open('rb') as f: g=pickle.load(f)
    with (folder/'source_models.pkl').open('rb') as f: fitted=pickle.load(f)
    with np.load(folder/'policies.npz') as f: old={k:f[k] for k in f.files}
    with np.load(CORRECTION/'ports'/folder.name/'policies.npz') as f: cached={k:f[k] for k in f.files}
    with np.load(PROBE/'ports'/folder.name/'response_models.npz') as f: response=f['stable_drift_incidence_position']
    sr,tr=old['source_rows'],old['query_rows']
    for key in ('query_rows','support_rows','eval_mask'):
        np.testing.assert_array_equal(old[key],cached[key])
    source=np.load(folder/'source_bank.npy',mmap_mode='r')
    target=np.load(folder/'target_bank.npy',mmap_mode='r')
    su=condition_features(frame.loc[sr,'incidence_deg'].to_numpy())
    tu=condition_features(frame.loc[tr,'incidence_deg'].to_numpy())
    source_shift=su @ response[:2]; target_shift=tu @ response[:2]
    j=jacobians(g)
    transforms=extend_transforms({k:v for k,v in make_transforms(g).items() if k.startswith('joint_')})
    extra=np.load(ROOT/'module_completion_20261008/ports_projection'/folder.name/'matched_joint/joint_extra.npy',mmap_mode='r')
    all_base=[cached['base_scores'].copy()]
    all_expert=[cached['expert_scores'].copy()]
    all_pred=[cached['predictions'].copy()]
    all_selected=[cached['selected'].copy()]
    replay=0.
    for corrected in (False,True):
        bank=np.asarray(target)-target_shift @ j['bank'] if corrected else np.asarray(target)
        rv=np.asarray(raw[tr],dtype=float)-target_shift @ j['raw'] if corrected else raw[tr]
        sb=np.asarray(source)-source_shift @ j['bank'] if corrected else np.asarray(source)
        heads=fit_source_heads(np.column_stack((sb,su)),frame.loc[sr,'class_id'].to_numpy(),transforms)
        with (out/('corrected_direct_heads.pkl' if corrected else 'direct_heads.pkl')).open('wb') as f:
            pickle.dump(heads,f)
        views=make_views(bank,rv,extra,g)
        if corrected:
            with (CORRECTION/'ports'/folder.name/'corrected_heads.pkl').open('rb') as f: old_heads=pickle.load(f)
        else:
            old_heads=fitted['heads']
        first=np.searchsorted(tr,old['support_rows'][0])
        a,b=predict_joint(bank,views,old_heads,first,frame.loc[old['support_rows'][0],'class_id'].to_numpy())
        slot=int(corrected)
        error=max(float(np.max(np.abs(a-cached['base_scores'][0,slot]))),float(np.max(np.abs(b-cached['expert_scores'][0,slot]))))
        replay=max(replay,error)
        if error>3e-6: raise ValueError(f'Parent replay failed {folder.name}/{corrected}: {error}')
        bases,experts,preds,selected=[],[],[],[]
        for draw,ev in enumerate(old['eval_mask']):
            support=np.searchsorted(tr,old['support_rows'][draw])
            np.testing.assert_array_equal(tr[support],old['support_rows'][draw])
            if ev[support].any(): raise ValueError('Support/query overlap')
            y=frame.loc[old['support_rows'][draw],'class_id'].to_numpy()
            base,expert=predict_conditioned(bank,views,heads,support,y,tu)
            gate=predict_gate(fitted['gates']['joint'],gate_features(base,expert))
            routed=route_scores(base,expert,gate,y,ev,tr)
            if routed['selected'][~ev].any(): raise ValueError('Support entered routing budget')
            bases.append(base); experts.append(expert); preds.append(routed['predictions']); selected.append(routed['selected'])
        all_base.append(np.stack(bases)[:,None]); all_expert.append(np.stack(experts)[:,None])
        all_pred.append(np.stack(preds)[:,None]); all_selected.append(np.stack(selected)[:,None])
    np.savez_compressed(out/'policies.npz',query_rows=tr,support_rows=old['support_rows'],eval_mask=old['eval_mask'],
        mode_names=np.asarray(MODES),base_scores=np.concatenate(all_base,axis=1),expert_scores=np.concatenate(all_expert,axis=1),
        predictions=np.concatenate(all_pred,axis=1),selected=np.concatenate(all_selected,axis=1))
    check=dict(port=folder.name,max_parent_replay_error=replay,target_labels_per_draw=64,budget=.05,
        coverage=float(np.isfinite(frame.loc[tr,'incidence_deg']).mean()),source_heads_refit=True,support_refit=True,gate_refit=False)
    write_json(out/'FROZEN.json',check)
    print(json.dumps(check),flush=True)
    return check


def evaluate(frame):
    records=[]
    for folder in sorted((OUT/'ports').iterdir()):
        with np.load(folder/'policies.npz') as f: p={k:f[k] for k in f.files}
        truth=frame.loc[p['query_rows'],'class_id'].to_numpy()
        for draw,ev in enumerate(p['eval_mask']):
            for m,mode in enumerate(MODES):
                row=dict(port=folder.name,draw=draw,mode=mode,query_n=int(ev.sum()),selected=int(p['selected'][draw,m].sum()))
                for stage,pred in [('full',p['predictions'][draw,m]),('base',p['base_scores'][draw,m].argmax(1)),('expert',p['expert_scores'][draw,m].argmax(1))]:
                    score=metrics(truth[ev],pred[ev],8)
                    row[stage+'_oa'],row[stage+'_ba']=score['acc'],score['ba']
                records.append(row)
    df=pd.DataFrame(records); df.to_csv(OUT/'metrics.csv',index=False)
    pm=df.groupby(['port','mode']).mean(numeric_only=True).drop(columns='draw').reset_index()
    ref=pm[pm['mode']=='reference'].set_index('port')
    for metric in ('full_oa','full_ba','base_oa','base_ba','expert_oa','expert_ba'):
        pm[metric+'_delta_pp']=[100*(r[metric]-ref.loc[r.port,metric]) for _,r in pm.iterrows()]
    pm.to_csv(OUT/'port_means.csv',index=False)
    full8=frame.groupby('port').class_id.nunique(); summaries=[]
    for name,mask in [('all23',np.ones(len(pm),dtype=bool)),('all8_target12',pm.port.map(full8).eq(8))]:
        mean=pm[mask].groupby('mode').mean(numeric_only=True).reset_index(); mean.insert(0,'group',name); summaries.append(mean)
    summary=pd.concat(summaries,ignore_index=True); summary.to_csv(OUT/'summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)


def main():
    if not (CORRECTION/'COMPLETE.json').exists(): raise ValueError('Missing frozen parent experiment')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETE.json').exists(): raise ValueError('Preserve completed experiment')
    write_json(OUT/'PROTOCOL.json',dict(date='2026-10-09',question='Does same-vessel response add value beyond the same incidence information?',
        modes=list(MODES),parent=str(CORRECTION),features='t=(angle-38)/5,t^2; zero for missing; no new missingness flag',
        source_head='Append two independent condition coefficients per class/head; balanced ridge1',
        target_base='Append u/sqrt2 before existing bias, original residual ridge8',
        target_expert='Original kernel plus .25*(u u_supportT)/2, original residual ridge8',
        correction='Frozen source same-vessel response and strength1 from previous experiment',
        fixed='23targets,5same64labeldraws,5percenthandoff,originalgate/backbones/geometry/scales',
        parameter_search=False,parameter_matched=False,same_information=True,
        different_inductive_bias='Correction changes visual metric cross-terms; direct conditioning appends independent condition kernel',
        main_model_unchanged=True,independent_confirmation=False,all_policies_frozen_before_query_evaluation=True))
    frame=pd.read_csv(SOURCE/'manifest.csv',usecols=['instance_id','product','port','class_id'])
    side=pd.read_csv(SIDECAR,usecols=['instance_id','product','status','incidence_deg'])
    frame=frame.merge(side,on=['instance_id','product'],how='left',sort=False,validate='one_to_one')
    frame.loc[frame.status.ne('ok'),'incidence_deg']=np.nan
    raw=np.load(SOURCE/'pooled_features.npy',mmap_mode='r'); checks=[]
    with threadpool_limits(limits=2):
        for folder in sorted((PARENT/'ports').iterdir()):
            if (folder/'policies.npz').exists(): checks.append(freeze_port(folder,frame,raw))
    pd.DataFrame(checks).to_csv(OUT/'verification.csv',index=False)
    write_json(OUT/'ALL_FROZEN.json',dict(ports=len(checks),target_query_labels_used=0))
    evaluate(frame)
    write_json(OUT/'COMPLETE.json',dict(ports=len(checks),draws=5,modes=list(MODES),max_parent_replay_error=max(c['max_parent_replay_error'] for c in checks)))


if __name__=='__main__': main()
