"""Keep old complete systems and strong simple controls in the within-scene table."""
from pathlib import Path
import numpy as np
import pandas as pd
import analyze_readout as analysis

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def main():
    records=[]
    for dataset in ('oss1','oss2'):
        for regime in ('classical_fit_only','classical'):
            path=ROOT/'paper_evidence_completion_20261026'/regime/dataset/'metrics.csv'
            frame=pd.read_csv(path).query("cohort=='within'").copy()
            frame['method']=frame.method.str.replace(r'_seed\d+$','',regex=True)
            for method,g in frame.groupby('method'):
                records.append(dict(dataset=dataset,version=regime,method=method,runs=len(g),
                    **{k:float(g[k].mean()) for k in ('OA','BA','NLL','Brier')}))
        oldroot=ROOT/f'{dataset}_m123_replication_20261026'/('m3_results_grid01' if dataset=='oss1' else 'm3_results')
        old=pd.read_csv(oldroot/'metrics.csv')
        for method,g in old.groupby('arm'):
            records.append(dict(dataset=dataset,version='old_formal_M123',method=method,runs=len(g),
                **{k:float(g[k].mean()) for k in ('OA','BA','NLL','Brier')}))
    new=pd.read_csv(HERE/'readout/metrics.csv').query("arm=='full'")
    for (dataset,method),g in new.groupby(['dataset','endpoint']):
        records.append(dict(dataset=dataset,version='new_raw_summary_paired',method=method,runs=len(g),
            **{k:float(g[k].mean()) for k in ('OA','BA','NLL','Brier')}))
    frame=pd.DataFrame(records)
    frame.to_csv(HERE/'complete_system_comparisons.csv',index=False)
    pairs=[]
    for dataset in ('oss1','oss2'):
        root=ROOT/f'{dataset}_m123_replication_20261026'
        oldroot=root/('m3_results_grid01' if dataset=='oss1' else 'm3_results')
        table=pd.read_csv(root/'stack_table.csv',dtype={'mmsi':str})
        oldtable=pd.read_csv(oldroot/'table.csv',dtype={'mmsi':str})
        np.testing.assert_array_equal(table.obs_id,oldtable.obs_id)
        ids=np.flatnonzero(table.role.eq('evaluation'))
        target=table.iloc[ids].reset_index(drop=True)
        y=target.class_id.to_numpy(int)
        for baseline in ('old_both','B_fixed','frequency'):
            effects=[]
            for seed in (17,29,43):
                old=analysis.run.tr.load_npz(oldroot/f'seed{seed}/predictions.npz')[baseline][ids]
                new=analysis.run.tr.load_npz(analysis.run.DEST/dataset/'full'/f'seed{seed}/source_predictions.npz')['frequency'][ids]
                d=(new.argmax(1)==y).astype(float)-(old.argmax(1)==y)
                effects.append(d)
                pairs.append(dict(dataset=dataset,before=baseline,after='new_full',seed=seed,
                    rescue=int((d>0).sum()),harm=int((d<0).sum()),OA_pp=d.mean()*100,
                    BA_pp=np.mean([d[y==c].mean() for c in range(4)])*100))
            pairs.append(dict(dataset=dataset,before=baseline,after='new_full',seed='mean',**analysis.interval(target,effects)))
    pd.DataFrame(pairs).to_csv(HERE/'version_paired.csv',index=False)
    # This check catches accidental selection/cross rows or a changed metric denominator.
    a=pd.read_csv(HERE/'readout/metrics.csv').rename(columns={'dataset':'source'})
    b=pd.read_csv(ROOT/'new_version_ablations_20261026/readout/metrics.csv').query("side=='source' and role=='evaluation'")
    merged=a.merge(b,on=['source','arm','seed','endpoint'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(merged)==126
    for key in ('OA','BA','NLL','Brier'):
        np.testing.assert_allclose(merged[f'{key}_new'],merged[f'{key}_old'],rtol=0,atol=1e-12)
    print(frame.to_string(index=False))
    print(pd.DataFrame(pairs).query("seed=='mean'").to_string(index=False))


if __name__=='__main__':main()
