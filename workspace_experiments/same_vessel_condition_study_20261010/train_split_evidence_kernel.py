"""Train score evidence through a fixed task-space support residual readout."""

import copy
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import train_explicit_candidate as previous
from train_directional_candidate import OUT as DIRECTIONAL, make_drift
from train_explicit_candidate import (
    BASE, LOCAL, RELIABLE, SEEDS, SUPPORTS, load_data, sample_weights, metrics,
    save_json, vessel_targets, centered, gradient_scale, tensors, build_targets,
    CandidateProjector, graph_data, ridge_probe, embedding_diagnostics, phi, residual_head,
)

OUT = BASE/'split_evidence_kernel_20261011'
ARMS = ('task','relation','selected_directions')


def main(out=OUT,arms=ARMS,stability_weights=None):
    torch.set_num_threads(1)
    started=time.perf_counter(); out.mkdir(exist_ok=False)
    meta,split,roles,source,raw,norm,fixed,h=load_data()
    fit,held=split['factor_fit'],split['test_source']
    fm,hm=[meta.iloc[ids].reset_index(drop=True) for ids in (fit,held)]
    x=torch.from_numpy(norm); y=torch.tensor(meta.class_id.to_numpy(copy=True))
    probability=sample_weights(fm); probability/=probability.sum()
    matches=pd.read_pickle(LOCAL/'source_matches.pkl')
    lookup={int(row):i for i,row in enumerate(fit)}
    validation={}
    with np.load(previous.OUT/'validation_episodes.npz') as f:
        for role in ('val_new_ship','val_new_time'):
            reverse={int(row):i for i,row in enumerate(roles[role])}
            validation[role]=[(np.array([reverse[int(v)] for v in f[f'{role}_{j}_support']]),
                               np.array([reverse[int(v)] for v in f[f'{role}_{j}_query']])) for j in range(3)]
    initial=CandidateProjector(x[fit],y[fit],probability)
    initial.load_state_dict(torch.load(previous.OUT/'pca_initial.pt',map_location='cpu',weights_only=True))
    save_json(out/'PROTOCOL.json',dict(
        arms=arms,seeds=SEEDS,steps=400,trainable_score_parameters=16648,
        stability_weights=stability_weights or {str(seed):.1 for seed in SEEDS},
        calibration='Default.1; optional source-only INITIAL frame-gradient norm cap: .1/max(1, ratio_to_task_gradient). Same first8 old source episodes plus full source-fit/graph expectation; fixed after initialization, no target-selected coefficient.',
        hypothesis='Preserve original task case-matching geometry and learn selectively stable prior evidence through that exact fixed support readout.',
        capacity='One frozen R18; separate32D score and case projections. Score projection+8class head trainable16648, case projection frozen16384 parameters. Not two neuralbackbones. Additional source-trained casebranch shared by all newarms.',
        initialization='Score uses identical OLD PCA32 state in all newarms, NOT target-chosen frozen swap or warm-started directional head. Case projection uses corresponding OLD source-selected task checkpoint. Two-stage sourcevalidated design; same old sourcevalidation reused, not independent confirmation.',
        fixed='Exact old source1296/525ships,64support96query training episodes, teachergraph, sourcevalidation episodes and3portsx3drawsx64targetlabels. Scalar weights and400steps unchanged. Case features detached in both training and validation.',
        losses='task: supervised+.5fullteachercenteredMSE+.5adaptedquerybalancedMSE. relation adds.25 oldgradientscaled reliableinterval. selected_directions adds initialvalue-normalized reliable directional stability with the recorded per-seed stability_weights. Same batches/episodes acrossarms.',
        differentiable='Support residual solve remains differentiable through support priors and query priors. Case geometry fixed; this is joint evidence/readout optimization, NOT joint training of two feature branches.',
        selection='Original earliestbest source adaptedBA every20steps including0; pooledsource episodes, not strictleaveportout. Last source snapshot diagnostic only, no target reselect.',
        evaluation='Predictions exported before reading query metrics. Previous task/shared-directional/frozenprior-only swap references replayed exactly. Frozen swap included as diagnostic reference, not extra trained model.',
        boundary='FrozenR18 sideprototype, reuseddevelopment targets, not full mainmodel or independent confirmation. Primary comparison within new three arms isolates reliable direction effect from adding fixed casebranch.'))
    kernels,zs,ps,models={},{},{},{}
    traces,choices=[],[]
    for seed in SEEDS:
        for arm,folder in [('task',previous.OUT),('selected_directions',DIRECTIONAL)]:
            with np.load(folder/f'{arm}_{seed}_vectors.npz') as f:
                z=torch.from_numpy(f['z'])
                if arm=='task':
                    kernels[seed]=z
                    np.savez_compressed(out/f'case_projection_{seed}.npz',
                        basis=f['basis'],scale=f['scale'],input_center=f['input_center'],
                        raw_center=split['center'],raw_scale=split['scale'])
            state=torch.load(folder/f'{arm}_{seed}.pt',map_location='cpu',weights_only=True)['selected']
            name='old_'+arm;zs[name,seed]=z
            ps[name,seed]=F.linear(z,state['head.weight'],state['head.bias'])
        with np.load(RELIABLE/'source_scores.npz') as f:
            np.testing.assert_array_equal(f['fit_obs_ids'],fit)
            prior=f[f'fit_baseline_{seed}']
        _,teacher,valid=vessel_targets(prior,fm)
        teacher,valid=torch.from_numpy(teacher),torch.from_numpy(valid)
        targets,_=build_targets(prior,fm,matches);target=tensors(targets['joint'])
        graph=graph_data(x[fit],fm,matches,target,probability)
        with np.load(previous.OUT/f'training_episodes_{seed}.npz') as f:
            episodes=[(np.array([lookup[int(v)] for v in s]),np.array([lookup[int(v)] for v in q]))
                      for s,q in zip(f['support_obs_ids'],f['query_obs_ids'])]
        with torch.no_grad(): _,p=initial(x[fit])
        scale,_=gradient_scale(p,teacher,y[fit],target,probability)
        for arm in arms:
            drift=make_drift(arm,y[fit],target) if arm=='selected_directions' else None
            model,trace,choice=previous.train(copy.deepcopy(initial),x[fit],y[fit],teacher,valid,target,
                probability,graph,episodes,validation,x,y,roles,seed,arm,scale,
                drift_fn=drift,output_dir=out,kernel_fit=kernels[seed][fit],kernel_all=kernels[seed],
                stability_weight=.1 if stability_weights is None else stability_weights[seed])
            traces.extend(trace);choices.append(choice);models['split_'+arm,seed]=model
    pd.DataFrame(traces).to_csv(out/'training.csv',index=False)
    pd.DataFrame(choices).to_csv(out/'selection.csv',index=False)
    source_rows,geometry=[],[]
    for (name,seed),model in models.items():
        with torch.no_grad():
            z,p=model(x);basis,scale=model.geometry()
            probe=ridge_probe(z[fit],y[fit],probability)
            probe_ba=metrics((phi(z[held])@probe).argmax(1).numpy(),y[held].numpy())['BA']
        zs[name,seed],ps[name,seed]=z,p
        np.savez_compressed(out/f'{name}_{seed}_vectors.npz',obs_ids=np.arange(len(meta)),
            manifest_rows=meta.manifest_row.to_numpy(),z=z.numpy(),basis=basis.numpy(),scale=scale.numpy(),
            input_center=model.center.numpy(),raw_center=split['center'],raw_scale=split['scale'])
        source_rows.append(dict(arm=name,seed=seed,snapshot='selected',
            **metrics(p[held].argmax(1).numpy(),y[held].numpy()),probe_BA=probe_ba))
        state=torch.load(out/f'{name.removeprefix("split_")}_{seed}.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['last'])
        with torch.no_grad(): _,last_p=model(x[held])
        source_rows.append(dict(arm=name,seed=seed,snapshot='last',
            **metrics(last_p.argmax(1).numpy(),y[held].numpy())))
        model.load_state_dict(state['selected'])
    for (name,seed),z in zs.items():
        for scope,values in [('score_space',z[held]),('decision',centered(ps[name,seed][held]))]:
            geometry.append(dict(arm=name,seed=seed,scope=scope,
                **embedding_diagnostics(values.numpy(),hm)))
    pd.DataFrame(source_rows).to_csv(out/'source_results.csv',index=False)
    pd.DataFrame(geometry).to_csv(out/'held_geometry.csv',index=False)
    tasks=[];replays=0
    names=('old_task','old_selected_directions','frozen_split')+tuple('split_'+a for a in arms)
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task=path.stem.removesuffix('_split');tasks.append(task)
        with np.load(path) as f: s,q,sy=[f[k] for k in ('support_obs_ids','query_obs_ids','support_labels')]
        archive=dict(support_obs_ids=s,query_obs_ids=q)
        with np.load(DIRECTIONAL/f'{task}_predictions.npz') as old, np.load(DIRECTIONAL/'mechanism'/f'{task}.npz') as swapped:
            for name in names:
                for seed in SEEDS:
                    p=ps['old_selected_directions' if name=='frozen_split' else name,seed]
                    z=zs[name,seed] if name.startswith('old_') else kernels[seed]
                    beta,_,_=residual_head(phi(z[s]),p[s],torch.tensor(sy),balanced=True)
                    score=(p[q]+phi(z[q])@beta).numpy()
                    if name.startswith('old_'):
                        np.testing.assert_array_equal(score,old[f'{name.removeprefix("old_")}__{seed}']);replays+=1
                    elif name=='frozen_split':
                        np.testing.assert_array_equal(score,swapped[f'task__prior_only__{seed}']);replays+=1
                    archive[f'{name}__{seed}']=score;archive[f'{name}_raw__{seed}']=p[q].numpy()
        np.savez_compressed(out/f'{task}_predictions.npz',**archive)
    save_json(out/'PREDICTIONS_SAVED.json',dict(tasks=len(tasks),query_metrics_read=False,exact_reference_replays=replays))
    rows,classes=[] ,[]
    for task in tasks:
        with np.load(out/f'{task}_predictions.npz') as f:
            mm=meta.iloc[f['query_obs_ids']];truth=mm.class_id.to_numpy()
            for name in names:
                for seed in SEEDS:
                    pred=f[f'{name}__{seed}'].argmax(1)
                    raw_result=metrics(f[f'{name}_raw__{seed}'].argmax(1),truth)
                    row=dict(arm=name,seed=seed,task=task,port=mm.port.iloc[0])
                    rows.append(dict(**row,**metrics(pred,truth),raw_OA=raw_result['OA'],raw_BA=raw_result['BA']))
                    for c in np.unique(truth):
                        classes.append(dict(**row,class_id=int(c),recall=float(100*(pred[truth==c]==c).mean())))
    df=pd.DataFrame(rows);df.to_csv(out/'target_results.csv',index=False)
    pd.DataFrame(classes).to_csv(out/'target_classes.csv',index=False)
    for name,keys in [('summary',['arm']),('seed_summary',['arm','seed']),('port_summary',['arm','port'])]:
        df.groupby(keys)[['OA','BA','raw_OA','raw_BA']].mean().reset_index().to_csv(out/f'target_{name}.csv',index=False)
    save_json(out/'COMPLETE.json',dict(seconds=time.perf_counter()-started,new_fits=len(arms)*len(SEEDS),
        exact_reference_replays=replays,main_model_modified=False))
    print(df.groupby('arm')[['OA','BA']].mean().to_string(),flush=True)


if __name__=='__main__':
    main()
