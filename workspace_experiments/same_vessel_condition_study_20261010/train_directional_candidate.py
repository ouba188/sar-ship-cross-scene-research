"""Fixed matched trial: full-vector versus logit versus class-direction stability."""

import copy
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import train_explicit_candidate as previous
from train_explicit_candidate import (
    BASE, LOCAL, RELIABLE, SEEDS, SUPPORTS, load_data,sample_weights,metrics,save_json,
    vessel_targets,gradient_scale,tensors,build_targets,CandidateProjector,graph_data,
    ridge_probe,embedding_diagnostics,incidence_probe,phi,residual_head,
)
from directional_candidate_loss import directional_rows

OUT = BASE/'directional_candidate_space_20261011'
ARMS = ('logit','all_directions','selected_directions')
REFERENCES = ('pca','task','relation','uniform','reliable')


def make_drift(arm,labels,target):
    def drift(model,anchors,donors,indices):
        if arm == 'logit':
            differences = F.linear(anchors-donors,model.head.weight)
            return (differences-differences.mean(1,keepdim=True)).square().mean(1)
        mask = target['mask'][indices] if arm == 'selected_directions' else torch.ones((len(indices),8),dtype=torch.bool)
        return directional_rows(anchors,donors,model.head.weight,labels[indices],mask)
    return drift


def main():
    torch.set_num_threads(1)
    started = time.perf_counter(); OUT.mkdir(exist_ok=False)
    meta,split,roles,source,raw,norm,fixed,h = load_data()
    fit,held = split['factor_fit'],split['test_source']
    fm,hm = [meta.iloc[ids].reset_index(drop=True) for ids in (fit,held)]
    x = torch.from_numpy(norm); y = torch.tensor(meta.class_id.to_numpy(copy=True))
    probability = sample_weights(fm); probability /= probability.sum()
    matches = pd.read_pickle(LOCAL/'source_matches.pkl')
    lookup = {int(row):i for i,row in enumerate(fit)}
    validation = {}
    with np.load(previous.OUT/'validation_episodes.npz') as z:
        for role in ('val_new_ship','val_new_time'):
            reverse = {int(row):i for i,row in enumerate(roles[role])}
            validation[role] = [(np.array([reverse[int(v)] for v in z[f'{role}_{j}_support']]),
                                 np.array([reverse[int(v)] for v in z[f'{role}_{j}_query']])) for j in range(3)]
    initial = CandidateProjector(x[fit],y[fit],probability)
    initial.load_state_dict(torch.load(previous.OUT/'pca_initial.pt',map_location='cpu',weights_only=True))
    save_json(OUT/'PROTOCOL.json',dict(
        seeds=SEEDS,arms=ARMS,reference_arms=REFERENCES,steps=400,dimension=32,
        purpose='Test whether restricting invariance to task-discriminative directions preserves class usefulness versus full-vector contraction.',
        controls='All unchanged source1296/PCA32init/16648parameters/fullMSE/reliableinterval/.5adapted64supportqueryloss/.1stability/sourceval adaptedBAselection. Reuse exact previous train/validation episodes and teachergraph. Only stability metric changes; initial expected metric each normalized to1.',
        logit='CoRe-inspired centered logit pair consistency on same graph and reliabilityweights; head receives gradients. This is an adapted compact reference, not official CoRe reproduction.',
        all_directions='Mean squared same-vessel latent difference projected on normalized current W_y-W_k for all7rivals. Directions stop-gradient; latent gradients remain. Same scalar graph reliability as selected/fullvector.',
        selected_directions='Same as all_directions but mask restricts to original same-vessel plus condition-matched different-vessel jointlypositive reliable class relations. Zero selected rows contributezero. Direction loss counts each relation; it is a PSD directional metric, not an orthogonal span projector.',
        mathematical_scope='8linearclassrows span atmost7 difference directions in32D. Orthogonal complement is unpenalized by directional stability alone, not provennoise orcausal. Normalization avoids direct headweight shrink reducingthe loss; detached directions still change through supervised headupdates.',
        no_search='No weights/ranks/masks selected on target queries. Predictions frozen before metrics. Initial source-only loss normalization matches value not exactgradient magnitude.',
        boundary='FrozenR18sideprototype; reused3portsx3drawsx64targetlabels,2optimizer/teacherseeds. Not independentconfirmation/fullmainmodel/officialliteraturebenchmark.'))
    histories,choices = [],[]
    students = {}
    with np.load(RELIABLE/'source_scores.npz') as old:
        np.testing.assert_array_equal(old['fit_obs_ids'],fit)
        for seed in SEEDS:
            prior = old[f'fit_baseline_{seed}']
            _,teacher,valid = vessel_targets(prior,fm)
            teacher,valid = torch.from_numpy(teacher),torch.from_numpy(valid)
            targets,_ = build_targets(prior,fm,matches); target=tensors(targets['joint'])
            graph=graph_data(x[fit],fm,matches,target,probability)
            with np.load(previous.OUT/f'graph_{seed}.npz') as g:
                np.testing.assert_array_equal(fit[graph['anchor']],g['anchor_obs_ids'])
                np.testing.assert_array_equal(graph['probability'],g['probability'])
                np.testing.assert_array_equal(graph['reliability'].numpy(),g['reliability'])
            with np.load(previous.OUT/f'training_episodes_{seed}.npz') as z:
                episodes=[(np.array([lookup[int(v)] for v in s]),np.array([lookup[int(v)] for v in q]))
                          for s,q in zip(z['support_obs_ids'],z['query_obs_ids'])]
            with torch.no_grad():
                _,p=initial(x[fit])
            scale,_=gradient_scale(p,teacher,y[fit],target,probability)
            for arm in ARMS:
                model,trace,choice=previous.train(copy.deepcopy(initial),x[fit],y[fit],teacher,valid,target,
                    probability,graph,episodes,validation,x,y,roles,seed,arm,scale,
                    drift_fn=make_drift(arm,y[fit],target),output_dir=OUT)
                students[arm,seed]=model; histories.extend(trace);choices.append(choice)
    pd.DataFrame(histories).to_csv(OUT/'training.csv',index=False)
    pd.DataFrame(choices).to_csv(OUT/'selection.csv',index=False)
    zs,ps={},{}
    geometry,source_rows=[] ,[]
    for arm in REFERENCES:
        for seed in SEEDS:
            with np.load(previous.OUT/f'{arm}_{seed}_vectors.npz') as z:
                zs[arm,seed]=torch.from_numpy(z['z'])
            state=torch.load(previous.OUT/'pca_initial.pt' if arm=='pca' else previous.OUT/f'{arm}_{seed}.pt',
                             map_location='cpu',weights_only=True)
            if arm!='pca': state=state['selected']
            ps[arm,seed]=F.linear(zs[arm,seed],state['head.weight'],state['head.bias'])
    for (arm,seed),model in students.items():
        with torch.no_grad():
            z,p=model(x);q,scale=model.geometry()
            probe=ridge_probe(z[fit],y[fit],probability)
            probe_scores=phi(z[held])@probe
        zs[arm,seed],ps[arm,seed]=z,p
        np.savez_compressed(OUT/f'{arm}_{seed}_vectors.npz',obs_ids=np.arange(len(meta)),
            manifest_rows=meta.manifest_row.to_numpy(),z=z.numpy(),basis=q.numpy(),scale=scale.numpy(),
            input_center=model.center.numpy(),raw_center=split['center'],raw_scale=split['scale'])
        source_rows.append(dict(arm=arm,seed=seed,snapshot='selected',
            **metrics(p[held].argmax(1).numpy(),y[held].numpy()),
            probe_BA=metrics(probe_scores.argmax(1).numpy(),y[held].numpy())['BA'],
            incidence_skill=incidence_probe(z[fit].numpy(),z[held].numpy(),fm,hm),
            head_norm=float(model.head.weight.norm())))
        for role,ids,mm in [('fit',fit,fm),('held',held,hm)]:
            geometry.append(dict(arm=arm,seed=seed,role=role,**embedding_diagnostics(z[ids].numpy(),mm)))
        selected=copy.deepcopy(model.state_dict())
        model.load_state_dict(torch.load(OUT/f'{arm}_{seed}.pt',map_location='cpu',weights_only=True)['last'])
        with torch.no_grad(): last_z,last_p=model(x[held])
        source_rows.append(dict(arm=arm,seed=seed,snapshot='last',
            **metrics(last_p.argmax(1).numpy(),y[held].numpy()),head_norm=float(model.head.weight.norm())))
        geometry.append(dict(arm=arm,seed=seed,role='held_last',**embedding_diagnostics(last_z.numpy(),hm)))
        model.load_state_dict(selected)
    pd.DataFrame(geometry).to_csv(OUT/'geometry.csv',index=False)
    pd.DataFrame(source_rows).to_csv(OUT/'source_results.csv',index=False)
    tasks=[]
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task=path.stem.removesuffix('_split');tasks.append(task)
        with np.load(path) as f:
            s,q,sy=[f[k] for k in ('support_obs_ids','query_obs_ids','support_labels')]
        archive=dict(support_obs_ids=s,query_obs_ids=q)
        with np.load(previous.OUT/f'{task}_predictions.npz') as old:
            np.testing.assert_array_equal(s,old['support_obs_ids']);np.testing.assert_array_equal(q,old['query_obs_ids'])
            for arm in REFERENCES+ARMS:
                for seed in SEEDS:
                    z,p=zs[arm,seed],ps[arm,seed]
                    beta,_,_=residual_head(phi(z[s]),p[s],torch.tensor(sy),balanced=True)
                    score=(p[q]+phi(z[q])@beta).numpy()
                    if arm in REFERENCES:
                        np.testing.assert_array_equal(score,old[f'{arm}__{seed}'])
                    archive[f'{arm}__{seed}']=score;archive[f'{arm}_raw__{seed}']=p[q].numpy()
        np.savez_compressed(OUT/f'{task}_predictions.npz',**archive)
    save_json(OUT/'PREDICTIONS_SAVED.json',dict(tasks=len(tasks),query_metrics_read=False))
    rows,classes=[],[]
    for task in tasks:
        with np.load(OUT/f'{task}_predictions.npz') as f:
            mm=meta.iloc[f['query_obs_ids']];truth=mm.class_id.to_numpy()
            for arm in REFERENCES+ARMS:
                for seed in SEEDS:
                    pred=f[f'{arm}__{seed}'].argmax(1)
                    raw_result=metrics(f[f'{arm}_raw__{seed}'].argmax(1),truth)
                    row=dict(arm=arm,seed=seed,task=task,port=mm.port.iloc[0])
                    rows.append(dict(**row,**metrics(pred,truth),raw_BA=raw_result['BA'],raw_OA=raw_result['OA']))
                    for c in np.unique(truth):
                        classes.append(dict(**row,class_id=int(c),recall=float(100*(pred[truth==c]==c).mean())))
    df=pd.DataFrame(rows);df.to_csv(OUT/'target_results.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'target_classes.csv',index=False)
    for name,keys in [('summary',['arm']),('seed_summary',['arm','seed']),('port_summary',['arm','port'])]:
        df.groupby(keys)[['OA','BA','raw_OA','raw_BA']].mean().reset_index().to_csv(OUT/f'target_{name}.csv',index=False)
    save_json(OUT/'COMPLETE.json',dict(seconds=time.perf_counter()-started,new_fits=6,
        exact_reference_replays=90,main_model_modified=False))
    print(df.groupby('arm')[['OA','BA']].mean().to_string(),flush=True)


if __name__=='__main__':
    main()
