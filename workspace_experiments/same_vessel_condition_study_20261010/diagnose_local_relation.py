"""Frozen path and source-scale diagnostics after the fixed local-teaching trial."""

import numpy as np
import pandas as pd
import torch
from local_relation_distillation import (
    OUT, PREVIOUS, SEEDS, SUPPORTS, EvidenceStudent, centered, previous, FactorNet,
    BoundedOperator, encode_two, standardize, load_data, sample_weights,
    vessel_targets, residual_head, metrics, save_json, auxiliary_rows,
)


def main():
    torch.set_num_threads(1)
    out = OUT/'mechanism'
    out.mkdir(exist_ok=False)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    fit, test = split['factor_fit'], split['test_source']
    x = torch.from_numpy(norm)
    labels = torch.tensor(meta.class_id.to_numpy(copy=True))
    pairs = [('all', 'full_all', 'local_all'), ('matched', 'full_matched', 'local_matched')]
    models, frozen, scales, source_rows = {}, {}, {}, []
    probability = torch.tensor(sample_weights(meta.iloc[fit].reset_index(drop=True)), dtype=torch.float32)
    probability /= probability.sum()
    for seed in SEEDS:
        factor = FactorNet()
        factor.load_state_dict(torch.load(previous.old.OUT/f'anchor_{seed}_factor.pt',map_location='cpu',weights_only=True)['model'])
        factor.eval()
        _, condition = encode_two(factor,norm)
        context = torch.from_numpy(standardize(condition,roles['fit']).astype(np.float32))
        operator = BoundedOperator(fixed['basis'],(h[roles['fit']]@fixed['basis']).square().mean(0).sqrt(),'residual')
        operator.load_state_dict(torch.load(previous.OUT/f'anchor_residual_{seed}_operator.pt',map_location='cpu',weights_only=True))
        with torch.no_grad():
            prior = operator(h,context)@fixed['weight']+fixed['bias']
        frozen[seed]=(operator,context)
        scores = dict(baseline=prior)
        for arm in ('full_all','local_all','full_matched','local_matched','other_matched'):
            path = PREVIOUS/f'other_acquisition_teacher_{seed}.pt' if arm == 'full_all' else OUT/f'{arm}_{seed}.pt'
            model = EvidenceStudent(centered(prior[fit]).square().mean(0).sqrt())
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['selected'])
            models[arm,seed]=model.eval()
            with torch.no_grad():
                scores[arm]=model(x,prior)
        for name,a,b in pairs:
            rms_a=(probability*centered(scores[a][fit]).square().mean(1)).sum().sqrt()
            rms_b=(probability*centered(scores[b][fit]).square().mean(1)).sum().sqrt()
            scales[name,seed]=float(rms_a/rms_b)
        for role,ids in [('fit',fit),('held',test)]:
            mm=meta.iloc[ids].reset_index(drop=True)
            _,teacher,valid=vessel_targets(prior[ids].numpy(),mm)
            teacher=torch.from_numpy(teacher)
            w=torch.tensor(sample_weights(mm),dtype=torch.float32)[valid];w/=w.sum()
            for arm,score in scores.items():
                s=score[ids]
                local=auxiliary_rows(s[valid],teacher[valid],labels[ids][valid],'local')
                full=auxiliary_rows(s[valid],teacher[valid],labels[ids][valid],'full')
                source_rows.append(dict(role=role,arm=arm,seed=seed,
                    **metrics(s.argmax(1).numpy(),labels[ids].numpy()),
                    local_shortfall=float((w*local).sum()),full_teacher_mse=float((w*full).sum())))
    pd.DataFrame(source_rows).to_csv(out/'source_objectives.csv',index=False)
    save_json(out/'SCALES.json',{f'{n}_{s}':v for (n,s),v in scales.items()})
    tasks=[]
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task=path.stem.removesuffix('_split');tasks.append(task)
        with np.load(path) as z:
            support,query,sy=[z[k] for k in ('support_obs_ids','query_obs_ids','support_labels')]
        ids=np.r_[support,query]
        archive=dict(query_obs_ids=query)
        with np.load(OUT/f'{task}_predictions.npz') as original:
            for seed in SEEDS:
                operator,context=frozen[seed]
                with torch.no_grad():
                    changed=operator(h[ids],context[ids])
                    base=changed@fixed['weight']+fixed['bias']
                phi=torch.cat([changed,torch.ones((len(ids),1))],1)
                for name,a,b in pairs:
                    with torch.no_grad():
                        ap=models[a,seed](x[ids],base)
                        bp=models[b,seed](x[ids],base)
                    beta_a,_,_=residual_head(phi[:64],ap[:64],torch.tensor(sy),balanced=True)
                    beta_b,_,_=residual_head(phi[:64],bp[:64],torch.tensor(sy),balanced=True)
                    score_variants=dict(full=ap[64:]+phi[64:]@beta_a,
                        query_only=bp[64:]+phi[64:]@beta_a,
                        support_only=ap[64:]+phi[64:]@beta_b,
                        local=bp[64:]+phi[64:]@beta_b)
                    np.testing.assert_array_equal(score_variants['full'].numpy(),original[f'{a}__{seed}'])
                    np.testing.assert_array_equal(score_variants['local'].numpy(),original[f'{b}__{seed}'])
                    scaled=bp.mean(1,keepdim=True)+scales[name,seed]*centered(bp)
                    np.testing.assert_array_equal(scaled.argmax(1).numpy(),bp.argmax(1).numpy())
                    beta_scaled,_,_=residual_head(phi[:64],scaled[:64],torch.tensor(sy),balanced=True)
                    score_variants['source_rms_scaled']=scaled[64:]+phi[64:]@beta_scaled
                    for mode,score in score_variants.items():
                        archive[f'{name}__{mode}__{seed}']=score.numpy()
        np.savez_compressed(out/f'{task}.npz',**archive)
    rows=[]
    for task in tasks:
        with np.load(out/f'{task}.npz') as z:
            qm=meta.iloc[z['query_obs_ids']]
            y=qm.class_id.to_numpy()
            for key in z.files:
                if '__' not in key:
                    continue
                pair,mode,seed=key.split('__')
                pred=z[key].argmax(1)
                rows.append(dict(pair=pair,mode=mode,seed=int(seed),task=task,port=qm.port.iloc[0],**metrics(pred,y)))
    data=pd.DataFrame(rows)
    data.to_csv(out/'target_results.csv',index=False)
    data.groupby(['pair','mode'])[['OA','BA']].mean().to_csv(out/'target_summary.csv')
    save_json(out/'COMPLETE.json',dict(new_training=0,diagonal_exact_replays=len(tasks)*len(SEEDS)*4,
        scope='Posthoc fixed two-pair query/support prior interventions. No target selection of method/scale.',
        amplitude='One global positive scalar per seed/pair fit from source weighted centered-score RMS; preserves raw prior argmax, not support-adapted predictions.',
        boundary='Crossed query/support cells require two trained functions; diagnostic only. All local positive teaching still lacks a purified content encoder.'))
    print(data.groupby(['pair','mode'])[['OA','BA']].mean().to_string(),flush=True)


if __name__=='__main__':
    main()
