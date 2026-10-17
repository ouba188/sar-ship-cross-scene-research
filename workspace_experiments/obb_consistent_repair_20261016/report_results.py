"""Paired complete-classifier outcomes and exact class contribution accounting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
MET=['oa','ba','macro_nll','macro_brier']
TASK=['seed','origin','acquisition','draw','task']
KEY=TASK+['global_row']
P=[f'p{i}' for i in range(8)]


def hierarchy(table,keys=(),values=MET):
    return table.groupby([*keys,'seed','origin','acquisition'])[values].mean().groupby([*keys,'seed','origin']).mean().reset_index()


def main():
    out=HERE/'analysis'; out.mkdir(exist_ok=True)
    records=[]
    for arm in ('historical','wrong','fixed'):
        for model in ('B','strict','optional'):
            for seed in range(3):
                path=HERE/arm/'evaluation'/f'{model}_seed{seed}_predictions.csv'
                f=pd.read_csv(path,dtype={'mmsi':str})
                f['arm']=arm; f['model']=model; f['seed']=seed
                assert f.budget.eq(32).all()
                records.append(f)
    pred=pd.concat(records,ignore_index=True)
    assert not pred.duplicated(['arm','model',*KEY]).any()
    metrics=[]
    for (arm,model,*task),g in pred.groupby(['arm','model',*TASK]):
        y=g.truth.to_numpy(int); p=g[P].to_numpy(float)
        np.testing.assert_allclose(p.sum(1),1.,atol=1e-10,rtol=0)
        correct=p.argmax(1)==y
        nll=-np.log(p[np.arange(len(y)),y].clip(1e-12))
        brier=((p-np.eye(8)[y])**2).sum(1)
        classes=np.unique(y)
        metrics.append(dict(arm=arm,model=model,**dict(zip(TASK,task)),n=len(y),
            oa=100*correct.mean(),ba=100*np.mean([correct[y==c].mean() for c in classes]),
            macro_nll=np.mean([nll[y==c].mean() for c in classes]),
            macro_brier=np.mean([brier[y==c].mean() for c in classes])))
    task=pd.DataFrame(metrics); task.to_csv(out/'task_metrics.csv',index=False)
    seed=hierarchy(task,['arm','model'])
    seed.to_csv(out/'seed_means.csv',index=False)
    summary=seed.groupby(['origin','arm','model'])[MET].agg(['mean','std'])
    summary.to_csv(out/'summary.csv')
    hierarchy(task,['arm','model','draw']).to_csv(out/'support_draw_means.csv',index=False)
    task.groupby(['arm','model','origin','acquisition'])[MET].mean().to_csv(out/'per_acquisition.csv')
    deltas=[]; all_events=[]; all_contributions=[]
    for model in ('B','strict','optional'):
        for ref in ('wrong','historical'):
            a=pred.loc[(pred.arm=='fixed')&(pred.model==model)]
            b=pred.loc[(pred.arm==ref)&(pred.model==model)]
            joined=a.merge(b[KEY+['truth','pred',*P]],on=KEY,suffixes=('','_ref'),validate='one_to_one')
            assert len(joined)==len(a)==len(b)
            np.testing.assert_array_equal(joined.truth,joined.truth_ref)
            joined['comparison']='fixed-'+ref
            joined['delta_correct']=joined.pred.eq(joined.truth).astype(int)-joined.pred_ref.eq(joined.truth).astype(int)
            joined['rescue']=joined.delta_correct.eq(1); joined['harm']=joined.delta_correct.eq(-1)
            all_events.append(joined)
            left=seed.loc[(seed.arm=='fixed')&(seed.model==model)]
            right=seed.loc[(seed.arm==ref)&(seed.model==model)]
            d=left.merge(right,on=['seed','origin'],suffixes=('','_ref'),validate='one_to_one')
            for row in d.itertuples():
                deltas.append(dict(model=model,comparison='fixed-'+ref,seed=row.seed,origin=row.origin,
                                   **{'delta_'+m:getattr(row,m)-getattr(row,m+'_ref') for m in MET}))
            for group,g in joined.groupby(TASK):
                counts=g.truth.value_counts(); nclasses=len(counts)
                for cls in range(8):
                    part=g.loc[g.truth==cls]; net=int(part.delta_correct.sum())
                    all_contributions.append(dict(model=model,comparison='fixed-'+ref,
                        **dict(zip(TASK,group)),truth=cls,n=len(part),
                        rescue=int(part.rescue.sum()),harm=int(part.harm.sum()),
                        delta_oa=100*net/len(g),
                        delta_ba=100*net/(nclasses*counts[cls]) if cls in counts else 0.))
    deltas=pd.DataFrame(deltas); deltas.to_csv(out/'paired_deltas.csv',index=False)
    events=pd.concat(all_events,ignore_index=True)
    changed=events.loc[events.rescue|events.harm].copy()
    changed.to_csv(out/'changed_cases.csv',index=False)
    con=pd.DataFrame(all_contributions)
    con.to_csv(out/'task_class_contributions.csv',index=False)
    hierarchy(con,['model','comparison','truth'],['delta_oa','delta_ba']).to_csv(out/'class_contributions.csv',index=False)
    accounted=con.groupby(['model','comparison',*TASK])[['delta_oa','delta_ba']].sum().reset_index()
    accounted=hierarchy(accounted,['model','comparison'],['delta_oa','delta_ba'])
    checks=accounted.merge(deltas,on=['model','comparison','seed','origin'],suffixes=('','_measured'))
    np.testing.assert_allclose(checks[['delta_oa','delta_ba']],checks[['delta_oa_measured','delta_ba_measured']],atol=1e-10,rtol=0)
    counts=[]
    for keys,g in events.groupby(['model','comparison','seed','origin']):
        c=g.loc[g.rescue|g.harm]
        counts.append(dict(zip(['model','comparison','seed','origin'],keys))|dict(
            rescue_events=int(g.rescue.sum()),harm_events=int(g.harm.sum()),
            changed_queries=int(c.global_row.nunique()),changed_ships=int(c.mmsi.nunique())))
    pd.DataFrame(counts).to_csv(out/'event_counts.csv',index=False)
    result=dict(prediction_rows=len(pred),task_metrics=len(task),
                unique_query_images=int(pred.global_row.nunique()),
                unique_query_ships=int(pred.mmsi.nunique()),
                unique_acquisitions=int(pred.acquisition.nunique()),
                class_contributions_match_full_deltas=True,
                task_counts=task.groupby(['arm','model','seed']).size().unique().tolist())
    (out/'RESULT_CHECK.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(seed.groupby(['origin','arm','model'])[MET].mean().to_string())
    print(deltas.groupby(['origin','model','comparison'])[['delta_'+m for m in MET]].mean().to_string())
    print(json.dumps(result))


if __name__=='__main__':
    main()
