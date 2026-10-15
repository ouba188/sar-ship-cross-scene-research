"""Frozen attribution of the joint-reliability increment; no new training."""

import numpy as np
import pandas as pd
import torch
from reliable_relation_distillation import (
    OUT, LOCAL, PREVIOUS, SEEDS, SUPPORTS, EvidenceStudent, centered, previous,
    FactorNet, BoundedOperator, encode_two, standardize, load_data, sample_weights,
    residual_head, metrics, save_json, build_targets,
)


def main():
    torch.set_num_threads(1)
    out = OUT/'mechanism'; out.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    fit, held = split['factor_fit'], split['test_source']
    x = torch.from_numpy(norm)
    matches = dict(fit=pd.read_pickle(LOCAL/'source_matches.pkl'),
                   held=pd.read_pickle(OUT/'held_matches.pkl'))
    constants, models, frozen, losses = {}, {}, {}, []
    with np.load(OUT/'source_scores.npz') as z:
        for seed in SEEDS:
            fm = meta.iloc[fit].reset_index(drop=True)
            p = sample_weights(fm); p /= p.sum()
            constants[seed] = np.sum(p[:,None]*(z[f'fit_joint_{seed}']-z[f'fit_full_{seed}']),axis=0).astype(np.float32)
            for role, ids in [('fit',fit),('held',held)]:
                mm = meta.iloc[ids].reset_index(drop=True)
                prior = z[f'{role}_baseline_{seed}']
                targets, _ = build_targets(prior,mm,matches[role])
                y = mm.class_id.to_numpy()
                p = sample_weights(mm)
                correct = prior.argmax(1) == y
                for mode, target in targets.items():
                    eligible = target['eligible']; mask = target['mask']
                    w = p.copy(); w[~eligible] = 0; w /= w.sum()
                    denominator = np.maximum(mask.sum(1),1)
                    for arm in ('baseline','full','joint'):
                        scores = z[f'{role}_{arm}_{seed}']
                        margins = scores[np.arange(len(mm)),y,None]-scores
                        lower = np.maximum(target['lower']-margins,0)**2*mask
                        upper = np.maximum(margins-target['upper'],0)**2*mask
                        lo = lower.sum(1)/denominator
                        up = upper.sum(1)/denominator
                        losses.append(dict(seed=seed,role=role,mode=mode,arm=arm,
                            lower_loss=float(w@lo),upper_loss=float(w@up),
                            baseline_correct_loss=float(w@((lo+up)*correct)),
                            baseline_wrong_loss=float(w@((lo+up)*(~correct)))))
            factor = FactorNet()
            factor.load_state_dict(torch.load(previous.old.OUT/f'anchor_{seed}_factor.pt',map_location='cpu',weights_only=True)['model'])
            factor.eval()
            _, condition = encode_two(factor,norm)
            context = torch.from_numpy(standardize(condition,roles['fit']).astype(np.float32))
            operator = BoundedOperator(fixed['basis'],(h[roles['fit']]@fixed['basis']).square().mean(0).sqrt(),'residual')
            operator.load_state_dict(torch.load(previous.OUT/f'anchor_residual_{seed}_operator.pt',map_location='cpu',weights_only=True))
            frozen[seed] = operator, context
            for arm in ('full','joint'):
                model = EvidenceStudent(centered(torch.from_numpy(z[f'fit_baseline_{seed}'])).square().mean(0).sqrt())
                path = PREVIOUS/f'other_acquisition_teacher_{seed}.pt' if arm == 'full' else OUT/f'joint_{seed}.pt'
                model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['selected'])
                models[arm,seed] = model.eval()
    pd.DataFrame(losses).to_csv(out/'interval_loss_paths.csv',index=False)
    save_json(out/'CONSTANTS.json',{str(k):v.tolist() for k,v in constants.items()})
    tasks = []
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task = path.stem.removesuffix('_split'); tasks.append(task)
        with np.load(path) as z:
            support,query,sy = [z[k] for k in ('support_obs_ids','query_obs_ids','support_labels')]
        ids = np.r_[support,query]; archive = dict(query_obs_ids=query)
        with np.load(OUT/f'{task}_predictions.npz') as old:
            for seed in SEEDS:
                operator,context = frozen[seed]
                with torch.no_grad():
                    changed = operator(h[ids],context[ids])
                    base = changed@fixed['weight']+fixed['bias']
                    full = models['full',seed](x[ids],base)
                    joint = models['joint',seed](x[ids],base)
                phi = torch.cat([changed,torch.ones((len(ids),1))],1)
                bf,_,_ = residual_head(phi[:64],full[:64],torch.tensor(sy),balanced=True)
                bj,_,_ = residual_head(phi[:64],joint[:64],torch.tensor(sy),balanced=True)
                constant = full+torch.from_numpy(constants[seed])
                bc,_,_ = residual_head(phi[:64],constant[:64],torch.tensor(sy),balanced=True)
                variants = dict(full=full[64:]+phi[64:]@bf,joint=joint[64:]+phi[64:]@bj,
                    query_only=joint[64:]+phi[64:]@bf,support_only=full[64:]+phi[64:]@bj,
                    constant=constant[64:]+phi[64:]@bc)
                for arm,scores in variants.items():
                    if arm in ('full','joint'):
                        np.testing.assert_array_equal(scores.numpy(),old[f'{arm}__{seed}'])
                    archive[f'{arm}__{seed}'] = scores.numpy()
        np.savez_compressed(out/f'{task}.npz',**archive)
    save_json(out/'PREDICTIONS_SAVED.json',dict(tasks=len(tasks),query_metrics_read=False))
    rows = []
    for task in tasks:
        with np.load(out/f'{task}.npz') as z:
            mm = meta.iloc[z['query_obs_ids']]; y = mm.class_id.to_numpy()
            for seed in SEEDS:
                for arm in ('full','joint','query_only','support_only','constant'):
                    rows.append(dict(task=task,port=mm.port.iloc[0],seed=seed,arm=arm,
                        **metrics(z[f'{arm}__{seed}'].argmax(1),y)))
    df = pd.DataFrame(rows); df.to_csv(out/'target_results.csv',index=False)
    for name,keys in [('summary',['arm']),('seed_summary',['arm','seed']),('port_summary',['arm','port'])]:
        df.groupby(keys)[['OA','BA']].mean().reset_index().to_csv(out/f'target_{name}.csv',index=False)
    save_json(out/'COMPLETE.json',dict(new_training=0,exact_diagonal_replays=36,
        scope='Fixed query/support prior swaps and one source-weighted mean score increment per seed. No target parameter selection. Crossed paths are diagnostics, not deployment.'))
    print(df.groupby('arm')[['OA','BA']].mean().to_string(),flush=True)


if __name__ == '__main__':
    main()
