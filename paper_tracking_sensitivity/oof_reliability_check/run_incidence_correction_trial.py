"""One fixed incidence-response correction with coordinated head/support fits."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import json
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from conditioned_vessel_drift import incidence_shift
from run_conditioned_vessel_drift import OUT as PROBE, SIDECAR
from run_same_vessel_temporal_probe import jacobians
from run_protected_drift_activation import make_views, make_transforms, gate_features
from protected_drift_activation import fit_source_heads
from residual_expert import fit_correction
from projection_residual_expert import fit_kernel_correction
from evidence_aware_handoff import predict_gate, route_scores
from task_stable_sensitivity import SOURCE, PARENT, ROOT
from sar_joint_completion import write_json
from run_generic_transfer import metrics


OUT = ROOT/'incidence_correction_trial_20261009'
MODES = ('reference','incidence_corrected')


def predict_joint(bank, views, heads, support, labels):
    bp = bank @ heads['joint_base']['coef'] + heads['joint_base']['intercept']
    ep = bank @ heads['joint_expert']['coef'] + heads['joint_expert']['intercept']
    phi = views['phis'][True,True]
    bcoef = fit_correction(phi[support],labels,bp[support])
    common = views['common'][True]
    k = .5*(common['retained'] @ common['retained'][support].T
             + common['complement'] @ common['complement'][support].T)+1.
    k += .25*views['stable'] @ views['stable'][support].T
    ecoef = fit_kernel_correction(k[support],labels,ep[support])
    return bp+phi @ bcoef, ep+k @ ecoef


def freeze_port(folder, frame, raw):
    out = OUT/'ports'/folder.name
    out.mkdir(parents=True,exist_ok=True)
    with (folder/'geometry.pkl').open('rb') as f:
        g = pickle.load(f)
    with (folder/'source_models.pkl').open('rb') as f:
        fitted = pickle.load(f)
    with np.load(folder/'policies.npz') as f:
        old = {k:f[k] for k in f.files}
    with np.load(PROBE/'ports'/folder.name/'response_models.npz') as f:
        coef = f['stable_drift_incidence_position']
    sr, tr = old['source_rows'], old['query_rows']
    np.testing.assert_array_equal(sr,np.load(folder/'source_rows.npy'))
    np.testing.assert_array_equal(tr,np.load(folder/'target_rows.npy'))
    sb = np.load(folder/'source_bank.npy',mmap_mode='r')
    tb = np.load(folder/'target_bank.npy',mmap_mode='r')
    j = jacobians(g)
    source_shift = incidence_shift(frame.loc[sr,'incidence_deg'].to_numpy(),coef)
    target_shift = incidence_shift(frame.loc[tr,'incidence_deg'].to_numpy(),coef)
    source_changed = np.asarray(sb)-source_shift @ j['bank']
    target_changed = np.asarray(tb)-target_shift @ j['bank']
    target_raw = np.asarray(raw[tr],dtype=float)-target_shift @ j['raw']
    transforms = {k:v for k,v in make_transforms(g).items() if k.startswith('joint_')}
    heads = fit_source_heads(source_changed,frame.loc[sr,'class_id'].to_numpy(),transforms)
    with (out/'corrected_heads.pkl').open('wb') as f:
        pickle.dump(heads,f)
    del source_changed, sb
    extra = np.load(ROOT/'module_completion_20261008/ports_projection'/folder.name/
                    'matched_joint/joint_extra.npy',mmap_mode='r')
    views = make_views(target_changed,target_raw,extra,g)
    # The original implementation must replay on exactly the same support draw.
    orig_views = make_views(tb,raw[tr],extra,g)
    first_support = np.searchsorted(tr,old['support_rows'][0])
    a,b = predict_joint(tb,orig_views,fitted['heads'],first_support,
                        frame.loc[old['support_rows'][0],'class_id'].to_numpy())
    index = list(old['arm_names']).index('joint')
    replay = max(float(np.max(np.abs(a-old['base_scores'][0,index]))),
                 float(np.max(np.abs(b-old['expert_scores'][0,index]))))
    if replay > 3e-6:
        raise ValueError(f'Original joint replay failed: {replay}')
    del orig_views
    bases,experts,predictions,selected=[],[],[],[]
    for draw,ev in enumerate(old['eval_mask']):
        support=np.searchsorted(tr,old['support_rows'][draw])
        np.testing.assert_array_equal(tr[support],old['support_rows'][draw])
        if ev[support].any():
            raise ValueError('Support/query overlap')
        labels=frame.loc[old['support_rows'][draw],'class_id'].to_numpy()
        base,expert=predict_joint(target_changed,views,heads,support,labels)
        gate=predict_gate(fitted['gates']['joint'],gate_features(base,expert))
        route=route_scores(base,expert,gate,labels,ev,tr)
        if route['selected'][~ev].any():
            raise ValueError('Support routed into query budget')
        bases.append(np.stack((old['base_scores'][draw,index],base)))
        experts.append(np.stack((old['expert_scores'][draw,index],expert)))
        predictions.append(np.stack((old['predictions'][draw,index],route['predictions'])))
        selected.append(np.stack((old['selected'][draw,index],route['selected'])))
    np.savez_compressed(out/'policies.npz',query_rows=tr,support_rows=old['support_rows'],
        eval_mask=old['eval_mask'],base_scores=np.stack(bases),expert_scores=np.stack(experts),
        predictions=np.stack(predictions),selected=np.stack(selected))
    check=dict(port=folder.name,max_replay_error=replay,
        source_condition_coverage=float(np.isfinite(frame.loc[sr,'incidence_deg']).mean()),
        target_condition_coverage=float(np.isfinite(frame.loc[tr,'incidence_deg']).mean()),
        angle_response_rank=int(np.linalg.matrix_rank(coef[:2])),source_heads_refit=True,
        support_adaptation_refit=True,gate_parameters_refit=False)
    write_json(out/'FROZEN.json',check)
    print(json.dumps(check),flush=True)
    return check


def evaluate(frame):
    records=[]
    for folder in sorted((OUT/'ports').iterdir()):
        with np.load(folder/'policies.npz') as f:
            p={k:f[k] for k in f.files}
        truth=frame.loc[p['query_rows'],'class_id'].to_numpy()
        for draw,ev in enumerate(p['eval_mask']):
            for m,mode in enumerate(MODES):
                row=dict(port=folder.name,draw=draw,mode=mode,query_n=int(ev.sum()),
                    selected=int(p['selected'][draw,m].sum()))
                for stage,pred in [('full',p['predictions'][draw,m]),
                    ('base',p['base_scores'][draw,m].argmax(1)),
                    ('expert',p['expert_scores'][draw,m].argmax(1))]:
                    score=metrics(truth[ev],pred[ev],8)
                    row[stage+'_oa'],row[stage+'_ba']=score['acc'],score['ba']
                records.append(row)
    df=pd.DataFrame(records)
    df.to_csv(OUT/'metrics.csv',index=False)
    pm=df.groupby(['port','mode']).mean(numeric_only=True).drop(columns='draw').reset_index()
    ref=pm[pm['mode']=='reference'].set_index('port')
    for metric in ('full_oa','full_ba','base_oa','base_ba','expert_oa','expert_ba'):
        pm[metric+'_delta_pp']=[100*(r[metric]-ref.loc[r.port,metric]) for _,r in pm.iterrows()]
    pm.to_csv(OUT/'port_means.csv',index=False)
    full8=frame.groupby('port').class_id.nunique()
    rows=[]
    for name,mask in [('all23',np.ones(len(pm),dtype=bool)),('all8_target12',pm.port.map(full8).eq(8))]:
        mean=pm[mask].groupby('mode').mean(numeric_only=True).reset_index()
        mean.insert(0,'group',name); rows.append(mean)
    summary=pd.concat(rows,ignore_index=True)
    summary.to_csv(OUT/'summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)


def main():
    if not (PROBE/'COMPLETE.json').exists():
        raise ValueError('Complete source condition replication before correction')
    scores=pd.read_csv(PROBE/'response_metrics.csv')
    held=scores[(scores.space=='stable_drift')&(scores['split']=='new_ship_month')]
    p=held.pivot(index='target_port',columns='model',values='skill')
    if p.incidence_position.mean()<=0 or (p.incidence_position-p.position).mean()<=0:
        raise ValueError('No average held source condition replication; do not proceed')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed correction trial')
    write_json(OUT/'PROTOCOL.json',dict(date='2026-10-09',condition_probe=str(PROBE),
        method='Subtract only incidence component of joint incidence-position source response',
        reference_incidence_deg=38,strength=1,space='existing Qs/Qd span',
        fit='Source response from fixed source train identities/months; no target outcomes',
        coordinated='Source classification heads and all64-label support fits use corrected features',
        unchanged='Backbones/geometry/scales/R18/gate parameters',
        missing='No direct feature shift; predictions can still change through common refitted heads',
        budget=.05,draws=5,labels_per_draw=64,arms=list(MODES),
        interpretation='Observational response correction, not identified causal intervention',
        fixed_gate_limitation='Original joint gate receives updated confidence features; gate not refit',
        all_policies_frozen_before_query_evaluation=True,main_model_unchanged=True))
    frame=pd.read_csv(SOURCE/'manifest.csv',usecols=['instance_id','product','port','class_id'])
    side=pd.read_csv(SIDECAR,usecols=['instance_id','product','status','incidence_deg'])
    frame=frame.merge(side,on=['instance_id','product'],how='left',sort=False,validate='one_to_one')
    frame.loc[frame.status.ne('ok'),'incidence_deg']=np.nan
    raw=np.load(SOURCE/'pooled_features.npy',mmap_mode='r')
    checks=[]
    with threadpool_limits(limits=2):
        for folder in sorted((PARENT/'ports').iterdir()):
            if (folder/'policies.npz').exists():
                checks.append(freeze_port(folder,frame,raw))
    pd.DataFrame(checks).to_csv(OUT/'verification.csv',index=False)
    write_json(OUT/'ALL_FROZEN.json',dict(ports=len(checks),target_queries_read=0))
    evaluate(frame)
    write_json(OUT/'COMPLETE.json',dict(ports=len(checks),draws=5,
        max_replay_error=max(c['max_replay_error'] for c in checks),main_model_unchanged=True))


if __name__=='__main__':
    main()
