"""Exact class contribution accounting for the observed OA/BA tradeoff."""

import numpy as np
import pandas as pd
from train_split_evidence_kernel import OUT, SEEDS, load_data


def main():
    meta,*_=load_data();rows=[]
    for path in sorted(OUT.glob('*_predictions.npz')):
        with np.load(path) as f:
            mm=meta.iloc[f['query_obs_ids']];y=mm.class_id.to_numpy();classes=np.unique(y)
            for seed in SEEDS:
                for reference in ('old_task','split_task'):
                    old=f[f'{reference}__{seed}'].argmax(1);new=f[f'split_relation__{seed}'].argmax(1)
                    for c in classes:
                        mask=y==c;rescue=int((mask&(old!=y)&(new==y)).sum());harm=int((mask&(old==y)&(new!=y)).sum())
                        net=rescue-harm
                        rows.append(dict(task=path.stem.removesuffix('_predictions'),port=mm.port.iloc[0],
                            seed=seed,reference=reference,class_id=int(c),samples=int(mask.sum()),
                            rescue=rescue,harm=harm,net=net,oa_contribution=100*net/len(y),
                            ba_contribution=100*net/(mask.sum()*len(classes))))
    df=pd.DataFrame(rows);df.to_csv(OUT/'mechanism/target_class_accounting.csv',index=False)
    result=df.groupby(['reference','class_id'])[['samples','rescue','harm','net','oa_contribution','ba_contribution']].mean()
    result.to_csv(OUT/'mechanism/target_class_accounting_summary.csv')
    observed=pd.read_csv(OUT/'target_results.csv')
    for reference in ('old_task','split_task'):
        units=df[df.reference.eq(reference)].groupby(['task','seed'])[['oa_contribution','ba_contribution']].sum()
        for metric,column in [('OA','oa_contribution'),('BA','ba_contribution')]:
            scores=observed.pivot(index=['task','seed'],columns='arm',values=metric)
            np.testing.assert_allclose(units[column],scores.loc[units.index,'split_relation']-scores.loc[units.index,reference],atol=1e-12)
    print(result.round(6).to_string(),flush=True)


if __name__=='__main__':
    main()
