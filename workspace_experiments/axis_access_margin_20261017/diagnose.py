"""No-training diagnostics: restored query evidence, missing classes and boundaries."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from model import prior, predict

HERE = Path(__file__).resolve().parent
KEYS = ['fold','seed','origin','draw','global_row']


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    p = pd.read_csv(HERE/'predictions.csv')
    columns = [f'p{k}' for k in range(8)]
    baseline = p[p.arm.eq('baseline')].set_index(KEYS)
    geometry = pd.read_csv(prior.OLD/'geometry.csv').set_index('global_row')
    metadata = geometry[['sample_id','class_name','chip_path_vv','chip_path_vh','raw_patch_path']]
    changed = pd.read_csv(HERE/'changed_queries.csv')
    changed = changed[changed.arm.isin(['access','margin','joint'])]
    changed = changed.join(metadata,on='global_row')
    labels = geometry.drop_duplicates('class_id').set_index('class_id').class_name.to_dict()
    changed['reference_class_name'] = changed.reference_prediction.map(labels)
    changed['predicted_class_name'] = changed.prediction.map(labels)
    changed.to_csv(HERE/'changed_native_cases.csv',index=False)
    manifest = pd.read_csv(prior.old.MANIFEST,dtype={'mmsi':str})
    quality = dict(np.load(prior.HERE/'quality.npz'))
    rows, replay, controls, control_cases = [], [], [], []
    with threadpool_limits(limits=1):
        for args in prior.old.folds():
            data,features,_,_,tasks,_,_ = prior.prepare_fold(args,manifest,quality)
            for seed in range(3):
                bc = np.load(HERE/data['fold']/f'baseline_{seed}'/'readouts.npz')
                baseline_raw = torch.tensor(bc['axis_raw'])
                for arm in ['access','joint']:
                    cache = np.load(HERE/data['fold']/f'{arm}_{seed}'/'readouts.npz')
                    np.testing.assert_array_equal(cache['global_rows'],data['rows'])
                    axis,common = torch.tensor(cache['axis_raw']),torch.tensor(cache['common_raw'])
                    av,cv = torch.tensor(cache['axis_valid']),torch.tensor(cache['common_valid'])
                    saved = p[p.arm.eq(arm)].set_index(KEYS)
                    for task in tasks:
                        ids,ns,q = task['ids'],task['ns'],task['q']
                        prob,h,_ = predict(axis[ids],av[ids],task,common[ids],cv[ids])
                        frozen = [(arm+'_axis_only', prior.old.predict(axis[ids],av[ids],task,.5)[0])]
                        if arm == 'access':
                            frozen.append(('baseline_half_axis',prior.old.predict(baseline_raw[ids],av[ids],task,.5)[0]))
                        for name, counterfactual in frozen:
                            truth = data['labels'][q]
                            score = prior.old.scores(truth,counterfactual.numpy())
                            controls.append(dict(fold=data['fold'],seed=seed,origin=task['origin'],draw=task['draw'],arm=name,**score))
                            for j,index in enumerate(q):
                                key = (data['fold'],seed,task['origin'],task['draw'],int(data['rows'][index]))
                                before = int(baseline.loc[key,'prediction'])
                                after = int(counterfactual[j].argmax())
                                change = int(after==truth[j])-int(before==truth[j])
                                if change:
                                    control_cases.append(dict(fold=data['fold'],seed=seed,origin=task['origin'],draw=task['draw'],
                                        arm=name,global_row=int(data['rows'][index]),truth=int(truth[j]),reference_prediction=before,
                                        prediction=after,change=change))
                        ss = task['ss']+h[:ns]@h[:ns].T
                        root = task['root']
                        coef = root[:,None]*torch.linalg.solve(root[:,None]*ss*root[None,:]+8*torch.eye(ns,dtype=ss.dtype),
                                  root[:,None]*(task['desired']-task['prior_s']))
                        split = axis.shape[1]
                        common_qs = h[ns:,split:]@h[:ns,split:].T
                        logits = task['prior_q']+(task['qs']+h[ns:]@h[:ns].T)@coef
                        direct_logits = common_qs@coef
                        without = .5*(task['base']+(logits-direct_logits).softmax(-1))
                        direct_logits = direct_logits.numpy()
                        for j,index in enumerate(q):
                            global_row = int(data['rows'][index])
                            key = (data['fold'],seed,task['origin'],task['draw'],global_row)
                            actual = prob[j].numpy()
                            error = float(np.max(abs(actual-saved.loc[key,columns].to_numpy(float))))
                            if error > 1e-10:
                                raise AssertionError(f'Cached combined replay {error}')
                            replay.append(error)
                            if bool(av[index]):
                                continue
                            ref = baseline.loc[key,columns].to_numpy(float)
                            y = int(data['labels'][index])
                            rival = int(np.argmax(np.where(np.arange(8)==y,-np.inf,ref)))
                            before_margin = ref[y]-ref[rival]
                            after_margin = actual[y]-actual[rival]
                            rows.append(dict(fold=data['fold'],arm=arm,seed=seed,origin=task['origin'],draw=task['draw'],
                                global_row=global_row,truth=y,class_name=labels[y],same_class_support=int((data['labels'][task['s']]==y).sum()),
                                reference_correct=int(ref.argmax()==y),prediction_correct=int(actual.argmax()==y),
                                max_probability_change=float(abs(actual-ref).max()),
                                true_probability_change=float(actual[y]-ref[y]),
                                baseline_fixed_rival_margin=float(before_margin),new_fixed_rival_margin=float(after_margin),
                                margin_change=float(after_margin-before_margin),
                                common_direct_expert_margin=float(direct_logits[j,y]-direct_logits[j,rival]),
                                direct_common_max_probability_effect=float(abs(actual-without[j].numpy()).max()),
                                brier_gain=float(((ref-np.eye(8)[y])**2).sum()-((actual-np.eye(8)[y])**2).sum()),
                                ap_before=float(baseline.loc[key,'ap']),ap_after=float(saved.loc[key,'ap'])))
    result = pd.DataFrame(rows)
    result.to_csv(HERE/'recovered_query_diagnostics.csv',index=False)
    summaries = []
    for (origin,arm),group in result.groupby(['origin','arm']):
        wrong = group[group.reference_correct.eq(0)]
        gap = -wrong.baseline_fixed_rival_margin
        summaries.append(dict(origin=origin,arm=arm,queries=group.global_row.nunique(),contexts=len(group),
            changed_probability_contexts=int((group.max_probability_change>1e-12).sum()),
            max_probability_change=group.max_probability_change.max(),mean_max_probability_change=group.max_probability_change.mean(),
            direct_common_effect_mean=group.direct_common_max_probability_effect.mean(),
            positive_true_probability_contexts=int((group.true_probability_change>0).sum()),
            brier_gain=group.brier_gain.mean(),sameclass_missing=int(group.same_class_support.eq(0).sum()),
            wrong_contexts=len(wrong),wrong_margin_deficit_median=gap.median(),wrong_margin_deficit_min=gap.min(),
            wrong_margin_change_median=wrong.margin_change.median(),
            wrong_closure_fraction_median=(wrong.margin_change/gap).median(),
            wrong_common_direct_help_fraction=wrong.common_direct_expert_margin.gt(0).mean(),
            eligible_map_before=group.ap_before.mean(),eligible_map_after=group.ap_after.mean()))
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE/'recovered_query_summary.csv',index=False)
    pd.DataFrame(controls).to_csv(HERE/'frozen_kernel_controls.csv',index=False)
    pd.DataFrame(control_cases).to_csv(HERE/'frozen_control_cases.csv',index=False)
    (HERE/'diagnostic_replay.json').write_text(json.dumps(dict(probability_vectors=len(replay),max_error=max(replay)),indent=2),encoding='utf-8')
    print(summary.to_string(index=False))
    print(changed[changed.origin.eq('cross')&changed.arm.eq('margin')][
        ['fold','seed','draw','global_row','class_name','reference_class_name','predicted_class_name','change','ba_contribution_pp']].to_string(index=False))
    print('CLASS_MAP',labels)
    print(pd.DataFrame(controls).groupby(['origin','arm'])[['oa','ba']].mean().to_string())


if __name__=='__main__':
    main()
