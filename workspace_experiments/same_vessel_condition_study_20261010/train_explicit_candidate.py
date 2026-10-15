"""Directly extract a 32D candidate using reliable relations and support episodes."""

import copy
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from reliable_relation_distillation import (
    BASE, LOCAL, SEEDS, SUPPORTS, load_data, sample_weights, metrics,
    save_json, vessel_targets, centered, gradient_scale, tensors,
)
from reliable_relation_targets import build_targets, interval_rows
from explicit_candidate_projection import CandidateProjector, ridge_probe, identity_episode, balanced_mse
from candidate_embedding_diagnostics import embedding_diagnostics
from target_response_adapter import residual_head

OUT = BASE/'explicit_candidate_space_20261011'
RELIABLE = BASE/'reliable_relation_distillation_20261011'
ARMS = ('task','relation','uniform','reliable')
STEPS = 400


def phi(z):
    return torch.cat([z,torch.ones((len(z),1),dtype=z.dtype)],1)


def graph_data(x, meta, matches, target, probability):
    anchors, donor_means, weights, reliability = [], [], [], []
    for a, group in matches.groupby('anchor',sort=True):
        if not target['eligible'][a]:
            continue
        donors = sorted(set(tuple(sorted(ids)) for ids in group.same_indices))
        for ids in donors:
            anchors.append(a); donor_means.append(x[list(ids)].mean(0))
            weights.append(probability[a]/len(donors))
            reliability.append(float(target['mask'][a].float().mean()*8/7))
    weights = np.array(weights); weights /= weights.sum()
    reliability = np.array(reliability)
    reliability /= weights@reliability
    return dict(anchor=np.array(anchors), donor=torch.stack(donor_means),
                probability=weights,reliability=torch.tensor(reliability,dtype=x.dtype))


def train(model,x,y,teacher,valid,target,probability,graph,episodes,validation,all_x,all_y,roles,seed,arm,scale,
          drift_fn=None,output_dir=None,kernel_fit=None,kernel_all=None,stability_weight=.1):
    opt = torch.optim.Adam(model.parameters(),lr=.001)
    if kernel_fit is not None:
        kernel_fit = kernel_fit.detach()
    if kernel_all is not None:
        kernel_all = kernel_all.detach()
    with torch.no_grad():
        anchor_z, donor_z = model.encode(x[graph['anchor']]), model.encode(graph['donor'])
        pair_distance = ((anchor_z-donor_z).square().sum(1) if drift_fn is None else
                         drift_fn(model,anchor_z,donor_z,graph['anchor']))
        gw = torch.tensor(graph['probability'],dtype=x.dtype)
        gfactor = graph['reliability'] if arm == 'reliable' or drift_fn is not None else torch.ones_like(gw)
        initial_drift = float((gw*gfactor*pair_distance).sum())
    best, best_ba, best_step = copy.deepcopy(model.state_dict()),-np.inf,0
    trace = []

    def validate(step, losses=None):
        nonlocal best,best_ba,best_step
        row = dict(arm=arm,seed=seed,step=step,**(losses or {}))
        bas = []
        with torch.no_grad():
            for role in ('val_new_ship','val_new_time'):
                ids = roles[role]; z,p = model(all_x[ids])
                design = phi(z if kernel_all is None else kernel_all[ids])
                row[role+'_raw_BA'] = metrics(p.argmax(1).numpy(),all_y[ids].numpy())['BA']
                local = []
                for s,q in validation[role]:
                    beta,_,_ = residual_head(design[s],p[s],all_y[ids][s],balanced=True)
                    score = p[q]+design[q]@beta
                    local.append(metrics(score.argmax(1).numpy(),all_y[ids][q].numpy())['BA'])
                row[role+'_adapt_BA'] = float(np.mean(local)); bas.append(row[role+'_adapt_BA'])
        row['selection_BA'] = float(np.mean(bas)); trace.append(row)
        if row['selection_BA'] > best_ba+1e-10:
            best,best_ba,best_step = copy.deepcopy(model.state_dict()),row['selection_BA'],step

    validate(0)
    for step in range(1,STEPS+1):
        rng = np.random.default_rng(seed+step)
        ids = rng.choice(len(x),128,replace=True,p=probability)
        edges = rng.choice(len(graph['anchor']),64,replace=True,p=graph['probability'])
        s,q = episodes[step-1]
        inputs = torch.cat([x[ids],x[s],x[q],x[graph['anchor'][edges]],graph['donor'][edges]])
        z,score = model(inputs)
        zs,zq,za,zd = torch.split(z[128:],[64,96,64,64])
        ps,pq = score[128:192],score[192:288]
        ks,kq = (zs,zq) if kernel_fit is None else (kernel_fit[s],kernel_fit[q])
        beta,_,_ = residual_head(phi(ks),ps,y[s],balanced=True)
        adapted = pq+phi(kq)@beta
        supervised = F.mse_loss(score[:128],2*F.one_hot(y[ids],8).float()-1)
        distillation = (centered(score[:128][valid[ids]])-centered(teacher[ids][valid[ids]])).square().mean()
        episode = balanced_mse(adapted,y[q])
        eligible = target['eligible'][ids]
        rows = interval_rows(score[:128],y[ids],target['lower'][ids],target['upper'][ids],target['mask'][ids])
        relation = rows[eligible].mean() if eligible.any() else score.sum()*0
        distance = ((za-zd).square().sum(1) if drift_fn is None else
                    drift_fn(model,za,zd,graph['anchor'][edges]))
        gweight = graph['reliability'][edges] if arm == 'reliable' or drift_fn is not None else torch.ones_like(distance)
        stability = (gweight*distance).mean()/initial_drift
        loss = supervised+.5*distillation+.5*episode
        if arm != 'task':
            loss = loss+.25*scale*relation
        if arm in ('uniform','reliable') or drift_fn is not None:
            loss = loss+stability_weight*stability
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        if step%20 == 0:
            validate(step,dict(loss=float(loss.detach()),supervised=float(supervised.detach()),
                full=float(distillation.detach()),episode=float(episode.detach()),
                relation=float(relation.detach()),stability=float(stability.detach())))
    state = dict(selected=best,last=copy.deepcopy(model.state_dict()))
    torch.save(state,(OUT if output_dir is None else output_dir)/f'{arm}_{seed}.pt')
    model.load_state_dict(best)
    choice = dict(arm=arm,seed=seed,step=best_step,selection_BA=best_ba,
                  relation_scale=scale,initial_graph_drift=initial_drift,stability_weight=stability_weight)
    print('SELECTED',choice,flush=True)
    return model.eval(),trace,choice


def incidence_probe(zfit,zheld,fm,hm):
    good = np.isfinite(fm.incidence_deg.to_numpy(float))
    test = np.isfinite(hm.incidence_deg.to_numpy(float))
    a = np.c_[zfit[good],np.ones(good.sum())]
    b = np.c_[zheld[test],np.ones(test.sum())]
    y = fm.incidence_deg.to_numpy(float)[good]; truth = hm.incidence_deg.to_numpy(float)[test]
    penalty = np.eye(a.shape[1])*10; penalty[-1,-1] = 0
    coef = np.linalg.solve(a.T@a+penalty,a.T@y)
    return float(100*(1-np.mean((b@coef-truth)**2)/np.mean((y.mean()-truth)**2)))


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    meta,split,roles,source,raw,norm,fixed,h = load_data()
    fit,held = split['factor_fit'],split['test_source']
    fm,hm = [meta.iloc[ids].reset_index(drop=True) for ids in (fit,held)]
    x = torch.from_numpy(norm); y = torch.tensor(meta.class_id.to_numpy(copy=True))
    probability = sample_weights(fm); probability /= probability.sum()
    matches = pd.read_pickle(LOCAL/'source_matches.pkl')
    validation = {role:[identity_episode(meta.iloc[roles[role]].reset_index(drop=True),
                         np.random.default_rng(91011+j),query_size=None) for j in range(3)]
                  for role in ('val_new_ship','val_new_time')}
    episode_dump = {}
    for role,eps in validation.items():
        for j,(s,q) in enumerate(eps):
            episode_dump[f'{role}_{j}_support'] = roles[role][s]
            episode_dump[f'{role}_{j}_query'] = roles[role][q]
    np.savez_compressed(OUT/'validation_episodes.npz',**episode_dump)
    save_json(OUT/'PROTOCOL.json',dict(
        arms=ARMS,seeds=SEEDS,steps=STEPS,dimension=32,
        objective='Operational target: relatively condition-insensitive class-useful explicit coordinates. No absolute invariance, physical axis names or universal port gains required.',
        representation='Learn512x32 frame, QR-orthogonalized, per-coordinate population variance from SOURCE FIT covariance normalized to1/32. Export frame, scales, center, vectors and exact orthogonal input-space residual decomposition. Original R18 frozen; no prior residual shortcut around z.',
        capacity='All4 arms identical32D frame and8-class linearhead; sourcePCA32 initialization and balancedridge1 initial head. All predictions and support kernels use z. PCA reference untrained. Not apples-to-apples with old full512 score-residual pipeline.',
        training='Same source1296,525ships;128 supervised rows;64support identities+96query samples from disjoint remaining identities in POOLED source-fit ports. Episodic train is not strictleaveportout. Same batches/episodes acrossarms.',
        losses='All: supervisedMSE+.5 originalfullteachercenteredMSE+.5 post64support classbalancedMSE. relation adds.25 gradientRMSscaled jointinterval. uniform adds.1 same-vessel latentdrift to relation. reliable instead weights samegraph by fractionjointpositiveedges. Both drift terms divided by own sourceinitialexpected drift, no target tuning.',
        graph='Same metadataeligible612anchors, other-acquisition same-vessel scene-mean features. Reliability from prior jointmask over7rivals, includes cross-vessel donor agreement. Normalize expected weight; zero-reliability edges retained withzero contribution. Other-vessel feature vectors are not forced equal.',
        selection='Source validation3fixed64support identity-disjoint episodes each in originalnewship/newtime roles; earliestbest mean adaptedBA every20steps including0. Newtime may sharefitidentities by originaldesign. Heldsource823 never selects.',
        target='Original3ports x3draws x64labels; singlechip input, no target identity/condition information at inference. Allvectors/predictions frozen beforequerymetrics. Previouslyreuseddevelopment notindependentconfirmation.',
        diagnostics='Heldsource explicitvectors: effective rank, energy-normalized withinvessel drift, sameclass other-vessel ratio, independent fixedridge classprobe, selectedhead raw/adapted OA/BA, observational incidenceprobe. Lower readability not proofnoncausal factors absent.',
        no_search='Fixed32dimension400steps4arms2seeds; no target-selected dimension, weight, or checkpoint. Mainmodel notmodified.'))
    torch.manual_seed(31415)
    initial = CandidateProjector(x[fit],y[fit],probability)
    initial_state = copy.deepcopy(initial.state_dict())
    torch.save(initial_state,OUT/'pca_initial.pt')
    students = {('pca',seed):copy.deepcopy(initial).eval() for seed in SEEDS}
    histories,choices,scales = [],[],[]
    with np.load(RELIABLE/'source_scores.npz') as old:
        np.testing.assert_array_equal(old['fit_obs_ids'],fit)
        for seed in SEEDS:
            prior = old[f'fit_baseline_{seed}']
            _,teacher,valid = vessel_targets(prior,fm)
            teacher = torch.from_numpy(teacher); valid = torch.from_numpy(valid)
            targets,audit = build_targets(prior,fm,matches)
            audit.to_csv(OUT/f'graph_coverage_{seed}.csv',index=False)
            target = tensors(targets['joint'])
            graph = graph_data(x[fit],fm,matches,target,probability)
            np.savez_compressed(OUT/f'graph_{seed}.npz',anchor_obs_ids=fit[graph['anchor']],
                                probability=graph['probability'],reliability=graph['reliability'].numpy())
            with torch.no_grad():
                _,initial_prior = initial(x[fit])
            scale, norms = gradient_scale(initial_prior,teacher,y[fit],target,probability)
            scales.append(dict(seed=seed,scale=scale,**norms))
            episodes = [identity_episode(fm,np.random.default_rng(seed+30000+step)) for step in range(1,STEPS+1)]
            np.savez_compressed(OUT/f'training_episodes_{seed}.npz',
                                support_obs_ids=np.stack([fit[s] for s,q in episodes]),
                                query_obs_ids=np.stack([fit[q] for s,q in episodes]))
            for arm in ARMS:
                model = copy.deepcopy(initial)
                model,trace,choice = train(model,x[fit],y[fit],teacher,valid,target,probability,graph,
                                          episodes,validation,x,y,roles,seed,arm,scale)
                students[arm,seed] = model
                histories.extend(trace); choices.append(choice)
    pd.DataFrame(histories).to_csv(OUT/'training.csv',index=False)
    pd.DataFrame(choices).to_csv(OUT/'selection.csv',index=False)
    pd.DataFrame(scales).to_csv(OUT/'relation_scales.csv',index=False)
    source_rows,geometry = [],[]
    vectors,scores = {},{}
    for (arm,seed),model in students.items():
        with torch.no_grad():
            z,p = model(x); q,scale = model.geometry()
            probe = ridge_probe(z[fit],y[fit],probability)
            probed = phi(z[held])@probe
        vectors[arm,seed],scores[arm,seed] = z,p
        np.savez_compressed(OUT/f'{arm}_{seed}_vectors.npz',obs_ids=np.arange(len(meta)),
            manifest_rows=meta.manifest_row.to_numpy(),z=z.numpy(),basis=q.numpy(),scale=scale.numpy(),
            input_center=model.center.numpy())
        source_rows.append(dict(arm=arm,seed=seed,role='held',snapshot='selected',
            **metrics(p[held].argmax(1).numpy(),y[held].numpy()),
            probe_BA=metrics(probed.argmax(1).numpy(),y[held].numpy())['BA'],
            incidence_skill=incidence_probe(z[fit].numpy(),z[held].numpy(),fm,hm)))
        for role,ids,mm in [('fit',fit,fm),('held',held,hm)]:
            geometry.append(dict(arm=arm,seed=seed,role=role,**embedding_diagnostics(z[ids].numpy(),mm)))
        if arm != 'pca':
            selected = copy.deepcopy(model.state_dict())
            state = torch.load(OUT/f'{arm}_{seed}.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(state['last'])
            with torch.no_grad():
                last_z,last_p = model(x[held])
            source_rows.append(dict(arm=arm,seed=seed,role='held',snapshot='last',
                **metrics(last_p.argmax(1).numpy(),y[held].numpy())))
            geometry.append(dict(arm=arm,seed=seed,role='held_last',**embedding_diagnostics(last_z.numpy(),hm)))
            model.load_state_dict(selected)
    pd.DataFrame(source_rows).to_csv(OUT/'source_results.csv',index=False)
    pd.DataFrame(geometry).to_csv(OUT/'geometry.csv',index=False)
    tasks = []
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task = path.stem.removesuffix('_split'); tasks.append(task)
        with np.load(path) as z:
            support,query,sy = [z[k] for k in ('support_obs_ids','query_obs_ids','support_labels')]
        assert not set(meta.iloc[support].mmsi)&set(meta.iloc[query].mmsi)
        assert not set(meta.iloc[np.r_[support,query]].mmsi)&set(meta.iloc[source].mmsi)
        archive = dict(support_obs_ids=support,query_obs_ids=query)
        for arm in ('pca',)+ARMS:
            for seed in SEEDS:
                z,p = vectors[arm,seed],scores[arm,seed]
                beta,_,_ = residual_head(phi(z[support]),p[support],torch.tensor(sy),balanced=True)
                archive[f'{arm}__{seed}'] = (p[query]+phi(z[query])@beta).numpy()
                archive[f'{arm}_raw__{seed}'] = p[query].numpy()
        np.savez_compressed(OUT/f'{task}_predictions.npz',**archive)
    save_json(OUT/'PREDICTIONS_SAVED.json',dict(tasks=len(tasks),query_metrics_read=False))
    rows,classes = [],[]
    for task in tasks:
        with np.load(OUT/f'{task}_predictions.npz') as z:
            mm = meta.iloc[z['query_obs_ids']]; truth = mm.class_id.to_numpy()
            for arm in ('pca',)+ARMS:
                for seed in SEEDS:
                    pred = z[f'{arm}__{seed}'].argmax(1)
                    raw_result = metrics(z[f'{arm}_raw__{seed}'].argmax(1),truth)
                    row = dict(arm=arm,seed=seed,task=task,port=mm.port.iloc[0])
                    rows.append(dict(**row,**metrics(pred,truth),raw_OA=raw_result['OA'],raw_BA=raw_result['BA']))
                    for c in np.unique(truth):
                        classes.append(dict(**row,class_id=int(c),recall=float(100*(pred[truth==c]==c).mean())))
    df = pd.DataFrame(rows); df.to_csv(OUT/'target_results.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'target_classes.csv',index=False)
    for name,keys in [('summary',['arm']),('seed_summary',['arm','seed']),('port_summary',['arm','port'])]:
        df.groupby(keys)[['OA','BA','raw_OA','raw_BA']].mean().reset_index().to_csv(OUT/f'target_{name}.csv',index=False)
    save_json(OUT/'COMPLETE.json',dict(seconds=time.perf_counter()-started,new_fits=8,
        dimensions=32,main_model_modified=False,projection_exported=True))
    print(df.groupby('arm')[['OA','BA','raw_BA']].mean().to_string(),flush=True)


if __name__ == '__main__':
    main()
