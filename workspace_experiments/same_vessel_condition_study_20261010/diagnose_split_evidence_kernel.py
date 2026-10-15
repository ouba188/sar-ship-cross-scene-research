"""Source-only objective/selection diagnosis for fixed-kernel evidence training."""

import copy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from train_split_evidence_kernel import OUT, ARMS, previous
from train_explicit_candidate import (
    LOCAL, RELIABLE, SEEDS, load_data, sample_weights, vessel_targets,
    build_targets, tensors, CandidateProjector, graph_data, centered,
    gradient_scale, phi, residual_head, metrics, embedding_diagnostics, save_json,
)
from explicit_candidate_projection import balanced_mse
from reliable_relation_targets import interval_rows
from train_directional_candidate import make_drift


def objective_parts(model,x,y,teacher,valid,target,weights,graph,kernel,episodes,scale,normalizer):
    z,p=model(x)
    supervised=(weights*(p-(2*F.one_hot(y,8).float()-1)).square().mean(1)).sum()
    full=(weights[valid]*(centered(p[valid])-centered(teacher[valid])).square().mean(1)).sum()/weights[valid].sum()
    eligible=target['eligible']
    rows=interval_rows(p,y,target['lower'],target['upper'],target['mask'])
    relation=(weights[eligible]*rows[eligible]).sum()/weights[eligible].sum()
    episode=[]
    for s,q in episodes:
        beta,_,_=residual_head(phi(kernel[s]),p[s],y[s],balanced=True)
        episode.append(balanced_mse(p[q]+phi(kernel[q])@beta,y[q]))
    episode=torch.stack(episode).mean()
    za=z[graph['anchor']]; zd=model.encode(graph['donor'])
    drift=make_drift('selected_directions',y,target)(model,za,zd,graph['anchor'])
    weight=torch.as_tensor(graph['probability'],dtype=x.dtype)*graph['reliability']
    stability=(weight*drift).sum()/normalizer
    objective=supervised+.5*full+.25*scale*relation+.5*episode
    return dict(supervised=supervised,full=full,relation=relation,episode=episode,
                task_objective=objective,stability=stability)


def main(run_dir=OUT,arms=ARMS,stability_weights=None):
    torch.set_num_threads(1)
    out=run_dir/'mechanism';out.mkdir(exist_ok=False)
    meta,split,roles,source,raw,norm,*_=load_data()
    fit,held=split['factor_fit'],split['test_source']
    fm,hm=[meta.iloc[ids].reset_index(drop=True) for ids in (fit,held)]
    x=torch.from_numpy(norm);y=torch.tensor(meta.class_id.to_numpy(copy=True))
    probability=sample_weights(fm);probability/=probability.sum()
    weights=torch.tensor(probability,dtype=x.dtype)
    matches=pd.read_pickle(LOCAL/'source_matches.pkl');lookup={int(row):i for i,row in enumerate(fit)}
    initial=CandidateProjector(x[fit],y[fit],probability)
    initial.load_state_dict(torch.load(previous.OUT/'pca_initial.pt',map_location='cpu',weights_only=True))
    rows,geometry=[],[]
    for seed in SEEDS:
        with np.load(RELIABLE/'source_scores.npz') as f: prior=f[f'fit_baseline_{seed}']
        _,teacher,valid=vessel_targets(prior,fm)
        teacher,valid=torch.from_numpy(teacher),torch.from_numpy(valid)
        targets,_=build_targets(prior,fm,matches);target=tensors(targets['joint'])
        graph=graph_data(x[fit],fm,matches,target,probability)
        with np.load(previous.OUT/f'task_{seed}_vectors.npz') as f: kernel=torch.from_numpy(f['z'][fit])
        with np.load(previous.OUT/f'training_episodes_{seed}.npz') as f:
            episodes=[(np.array([lookup[int(v)] for v in s]),np.array([lookup[int(v)] for v in q]))
                      for s,q in zip(f['support_obs_ids'][:8],f['query_obs_ids'][:8])]
        with torch.no_grad():
            _,p0=initial(x[fit])
            za,zd=initial.encode(x[fit][graph['anchor']]),initial.encode(graph['donor'])
            dist=make_drift('selected_directions',y[fit],target)(initial,za,zd,graph['anchor'])
            normalizer=float((torch.tensor(graph['probability'],dtype=x.dtype)*graph['reliability']*dist).sum())
        scale,_=gradient_scale(p0,teacher,y[fit],target,probability)
        states=[('pca','initial',copy.deepcopy(initial.state_dict()))]
        for arm in arms:
            state=torch.load(run_dir/f'{arm}_{seed}.pt',map_location='cpu',weights_only=True)
            states.extend((arm,snapshot,state[snapshot]) for snapshot in ('selected','last'))
        for arm,snapshot,state in states:
            model=copy.deepcopy(initial);model.load_state_dict(state)
            losses=objective_parts(model,x[fit],y[fit],teacher,valid,target,weights,graph,kernel,episodes,scale,normalizer)
            gt=torch.autograd.grad(losses['task_objective'],model.frame,retain_graph=True)[0].flatten()
            coefficient=.1 if stability_weights is None else stability_weights[seed]
            gs=torch.autograd.grad(coefficient*losses['stability'],model.frame)[0].flatten()
            cosine=float((gt@gs)/(gt.norm()*gs.norm()).clamp_min(1e-12))
            row=dict(arm=arm,seed=seed,snapshot=snapshot,stability_weight=coefficient,
                **{k:float(v.detach()) for k,v in losses.items()},gradient_cosine=cosine,
                stability_to_task_gradient_norm=float(gs.norm()/gt.norm().clamp_min(1e-12)))
            with torch.no_grad():
                z,p=model(x)
                row.update({'held_'+k:v for k,v in metrics(p[held].argmax(1).numpy(),y[held].numpy()).items()})
                g=embedding_diagnostics(centered(p[held]).numpy(),hm)
                geometry.append(dict(arm=arm,seed=seed,snapshot=snapshot,**g))
                anchor=graph['anchor'];pa=p[fit][anchor];pd_=model.head(model.encode(graph['donor']))
                directions=model.head.weight[y[fit][anchor],None,:]-model.head.weight[None,:,:]
                directions/=directions.norm(dim=2,keepdim=True).clamp_min(1e-12)
                mask=target['mask'][anchor]
                delta=z[fit][anchor]-model.encode(graph['donor'])
                energy=torch.einsum('nd,ncd->nc',delta,directions).square()*mask
                gw=torch.tensor(graph['probability'],dtype=x.dtype)*graph['reliability']/mask.sum(1).clamp_min(1)
                energy*=gw[:,None]
                ya=y[fit][anchor];ma=pa.gather(1,ya[:,None])-pa;md=pd_.gather(1,ya[:,None])-pd_
                total=energy.sum().clamp_min(1e-12)
                row['drift_energy_both_relation_positive_pct']=float(100*(energy*((ma>0)&(md>0))).sum()/total)
                both_correct=(pa.argmax(1)==ya)&(pd_.argmax(1)==ya)
                row['drift_energy_both_top1_correct_pct']=float(100*(energy*both_correct[:,None]).sum()/total)
                lower,upper=target['lower'][anchor],target['upper'][anchor]
                within=(ma>=lower)&(ma<=upper)&(md>=lower)&(md<=upper)
                row['drift_energy_both_within_interval_pct']=float(100*(energy*within).sum()/total)
            rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(out/'source_objectives.csv',index=False)
    pd.DataFrame(geometry).to_csv(out/'held_decision_geometry.csv',index=False)
    save_json(out/'COMPLETE.json',dict(new_training=0,target_queries_used=False,
        diagnostic='Full source-fit expectation for supervision and graph; first8 original source episodes for adapted task term. Gradient cosine on frame ONLY. Local conflict not unique rootcause or causal proof. Same-source positive-margin drift energy not necessarily useless. Selected/last heldsource diagnostics do not reselect target checkpoint.'))
    cols=['task_objective','episode','stability','gradient_cosine','stability_to_task_gradient_norm',
          'held_BA','drift_energy_both_relation_positive_pct','drift_energy_both_within_interval_pct']
    print(df.groupby(['arm','snapshot'])[cols].mean().round(6).to_string(),flush=True)


if __name__=='__main__':
    main()
