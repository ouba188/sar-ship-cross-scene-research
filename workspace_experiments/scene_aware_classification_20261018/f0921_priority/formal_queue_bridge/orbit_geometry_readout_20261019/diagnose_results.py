"""Bounded similarity diagnostics and strongest-contribution raw support views."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_experiment as ev
import review_results as review

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent


def main():
    meta=pd.read_csv(PARENT/'cohort.csv',dtype={'date':str,'mmsi':str}).set_index('row',drop=False)
    features=dict(np.load(PARENT/'features.npz'))
    spatial=dict(np.load(HERE/'spatial_features.npz'))
    geometry=pd.read_csv(HERE/'orbit_geometry.csv')
    axes=pd.read_csv(HERE/'axes.csv')
    records=[]
    with threadpool_limits(limits=1):
        for task in json.loads((PARENT/'episodes.json').read_text()):
            q=np.array(task['query'])
            for draw,rows in enumerate(task['supports']):
                s=np.array(rows)
                ks=ev.kernels(features,spatial,s,np.zeros((len(meta),len(meta))))
                known=spatial['valid'][q,None]&spatial['valid'][None,s]
                for comp in ['add_image','add_pooled']:
                    a=(ks['add_orbit']-ks['base'])[np.ix_(q,s)]/.0625
                    b=(ks[comp]-ks['base'])[np.ix_(q,s)]/.0625
                    records.append(dict(fold=task['fold'],budget=task['budget'],draw=draw,comparator=comp,
                        known_pairs=int(known.sum()),mean_abs_delta=float(np.abs(a-b).mean()),
                        known_mean_abs_delta=float(np.abs(a-b)[known].mean()) if known.any() else np.nan,
                        max_abs_delta=float(np.abs(a-b).max()),
                        known_correlation=float(np.corrcoef(a[known],b[known])[0,1]) if known.sum()>1 else np.nan))
    pd.DataFrame(records).to_csv(HERE/'kernel_diagnostics.csv',index=False)
    pred=pd.read_csv(HERE/'predictions.csv',dtype={'fold':str})
    comparisons={}
    for comp in ['add_image','add_pooled','base']:
        a=pred[pred.arm.eq('add_orbit')].set_index(ev.KEYS).sort_index()
        b=pred[pred.arm.eq(comp)].set_index(ev.KEYS).sort_index()
        delta=a[ev.roi.PROBS].to_numpy()-b[ev.roi.PROBS].to_numpy()
        comparisons[comp]=dict(mean_abs_probability_change=float(np.abs(delta).mean()),
            max_abs_probability_change=float(np.abs(delta).max()),top1_changes=int((a.prediction!=b.prediction).sum()))
    contribution=pd.read_csv(HERE/'case_support_contributions.csv')
    selected=pd.read_csv(HERE/'selected_cases.csv',dtype={'fold':str})
    regions=np.load(PARENT/'masks.npz')['regions']
    target=HERE/'case_review/signed_supports'
    target.mkdir(exist_ok=True)
    traced=[]
    for row in [73,236]:
        e=selected[selected.row.eq(row)&selected.comparator.eq('base')].iloc[0]
        group=contribution[contribution.row.eq(row)&contribution.comparator.eq('base')].copy()
        ids=[row]+group.iloc[np.argsort(-np.abs(group.total.to_numpy()))[:2]].support_row.astype(int).tolist()
        review.panel(ids,meta,axes,geometry,regions,target/f'row{row}_signed.png')
        for i in ids[1:]:
            path=target/f'row{row}_support{i}_native.png'
            review.review.native_query(meta.loc[i],path)
            traced.append(dict(query=row,support=i,fold=e.fold,budget=int(e.budget),draw=int(e.draw),native=str(path)))
    pd.DataFrame(traced).to_csv(HERE/'signed_native_paths.csv',index=False)
    receipt=dict(probabilities=comparisons,valid_rows=int(spatial['valid'].sum()),
        kernel_group_means=pd.DataFrame(records).groupby('comparator')[['known_mean_abs_delta','known_correlation']].mean().to_dict('index'))
    (HERE/'diagnostic_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
