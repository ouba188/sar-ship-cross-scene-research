"""Post-hoc mechanism probe: remove only a source-estimated class-wide score shift.

No training or target tuning. This is a diagnostic, not a new retained module.
"""
import analyze as a
import json
import numpy as np
import pandas as pd


def main():
    run=a.run;here=a.HERE
    table=pd.read_csv(run.NATIVE/'table.csv');upids=table.up_index.to_numpy(int)
    tr=np.flatnonzero(table.role.eq('utility_train'))
    te=np.flatnonzero(table.role.eq('evaluation'));y=table.iloc[te].class_id.to_numpy(int)
    sourcey=table.iloc[tr].class_id.to_numpy(int)
    scores={};rows=[];offsets=[];flips=[]
    for seed in run.SEEDS:
        old=np.load(here/'results'/f'old_full_{seed}.npz')['score']
        new=np.load(here/'results'/f'typed_both_{seed}.npz')['score']
        delta=new[tr]-old[tr]
        offset=np.stack([delta[sourcey==c].mean(0) for c in range(8)]).mean(0)
        corrected=new-offset[None]
        scores[seed]=(old,new,corrected)
        for c,v in enumerate(offset):offsets.append(dict(seed=seed,class_id=c,source_macro_shift=float(v)))
        for comp,value in (('old_full',old),('typed_both',new)):
            rows.append(dict(seed=seed,comparator=comp) | a.effect(corrected[te],value[te],y)
                        | run.metrics(y,corrected[te]))
        for k in np.flatnonzero(corrected[te].argmax(1)!=old[te].argmax(1)):
            row=table.iloc[te[k]]
            flips.append(dict(seed=seed,obs_id=int(row.obs_id),class_id=int(y[k]),
                old_prediction=int(old[te[k]].argmax()),typed_prediction=int(new[te[k]].argmax()),
                diagnostic_prediction=int(corrected[te[k]].argmax()),pass_group=row.pass_group))
        np.savez_compressed(here/'results'/f'bias_diagnostic_{seed}.npz',score=corrected,
                            offset=offset,up_index=upids[:1741])
    pd.DataFrame(rows).to_csv(here/'bias_diagnostic.csv',index=False)
    pd.DataFrame(offsets).to_csv(here/'source_class_offsets.csv',index=False)
    pd.DataFrame(flips).to_csv(here/'bias_diagnostic_flips.csv',index=False)
    # Only additional top-1 rescues/harms vs old require new example panels.
    known={int(x['obs_id']) for name in ('cases.json','error_cases.json')
           for x in json.loads((here/'analysis'/name).read_text(encoding='utf-8'))}
    geo=run.load('anchored_bias_geo',run.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up=pd.read_csv(run.BASE/'regional_utility_pilot/upstream/table.csv')
    valid=np.load(run.BASE/'regional_utility_pilot/upstream/summaries.npz')['valid']
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs=np.load(run.LOCAL/'scores_layer1.npz')['refs']
    class Metadata:
        pass
    data=Metadata();data.up=upids
    records=[]
    for kind in ('rescue','harm'):
        candidates=[]
        for seed,(old,new,corrected) in scores.items():
            ap,bp=corrected[te].argmax(1)==y,old[te].argmax(1)==y
            keep=ap&~bp if kind=='rescue' else ~ap&bp
            for k in np.flatnonzero(keep):
                obs=int(table.iloc[te[k]].obs_id)
                if obs not in known:candidates.append((seed,obs,int(te[k])))
        if candidates:
            seed,obs,j=sorted(candidates)[0];old,new,corrected=scores[seed]
            case=dict(arm='source_centered_diagnostic',seed=seed,obs_id=obs,index=j,case=kind)
            path,prov,true,rival=a.panel(case,data,old[j],corrected[j],geo,up,valid,snapshots,refs)
            records.append(dict(**case,true=true,rival=rival,before=old[j],after=corrected[j],
                                typed=new[j],figure=str(path),provenance=prov))
    run.write_json(here/'analysis/bias_cases.json',records)
    run.write_json(here/'bias_diagnostic_protocol.json',dict(
        scope='Post-hoc explanation, not a new trained or retained component',
        fitting='Source utility_train 644 only; macro-average over its eight true classes',
        correction='s_diagnostic = s_typed - mean_source_class_balanced(s_typed-s_old)',
        coefficient=1,no_target_fit=True,no_selection=True,no_extra_training=True,
        limitation='Removing a class-wide score shift does not establish physical causality or isolate all query-specific information.'))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(flips).to_string(index=False),flush=True)


if __name__=='__main__':main()
