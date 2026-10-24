"""Export per-observation fixed-head evidence without shipping model caches."""
import train_hybrid as run
import numpy as np
import pandas as pd
tr=run.tr


def main():
    table=pd.read_csv(tr.OLD/'table.csv',dtype={'mmsi':str})
    ids=np.flatnonzero(table.role.eq('evaluation'));f=table.iloc[ids].reset_index(drop=True)
    y=f.class_id.to_numpy(int);cache=np.load(run.HERE/'results/switches.npz');parts=[]
    for seed in tr.SEEDS:
        for arm in run.ARMS[1:]:
            on=np.load(run.HERE/'results'/f'{arm}_{seed}.npz')['score'][ids]
            for variant in ('off','early') if arm=='axis_role' else ('off',):
                off=cache[f'{seed}_{arm}_{variant}']
                row=f[['obs_id','class_id','class_name','mmsi','pass_group','port']].copy()
                a,b=on.argmax(1),off.argmax(1)
                row['seed']=seed;row['arm']=arm;row['intervention']=variant
                row['prediction_on']=a;row['prediction_intervened']=b
                row['rescued']=(a==y)&(b!=y);row['harmed']=(a!=y)&(b==y)
                for c in range(8):row[f'on_{c}']=on[:,c];row[f'intervened_{c}']=off[:,c]
                parts.append(row)
    allrows=pd.concat(parts,ignore_index=True)
    allrows.to_csv(run.HERE/'switch_predictions.csv.gz',index=False,compression='gzip')
    flips=allrows[allrows.rescued|allrows.harmed]
    flips.to_csv(run.HERE/'switch_flips.csv',index=False)
    print(flips.query("arm=='axis_role'")[['seed','intervention','obs_id','class_name','prediction_intervened','prediction_on','rescued','harmed']].to_string(index=False))
    comparisons=[]
    for seed in tr.SEEDS:
        for left,right in [('axis_role','grid_role'),('axis_difference','moment')]:
            a=np.load(run.HERE/'results'/f'{left}_{seed}.npz')['score'][ids]
            b=np.load(run.HERE/'results'/f'{right}_{seed}.npz')['score'][ids]
            comparisons.append(dict(seed=seed,left=left,right=right,
                changed=int((a.argmax(1)!=b.argmax(1)).sum()),max_score_difference=float(np.abs(a-b).max())))
    pd.DataFrame(comparisons).to_csv(run.HERE/'prediction_identity.csv',index=False)
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__=='__main__':main()
