"""Frozen path swaps and held-object decision-direction stability."""

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import train_explicit_candidate as previous
from train_directional_candidate import OUT, ARMS, REFERENCES
from train_explicit_candidate import SEEDS, SUPPORTS, load_data, phi, residual_head, metrics, save_json


def main():
    torch.set_num_threads(1)
    out = OUT/'mechanism'; out.mkdir(exist_ok=False)
    meta,split,*_ = load_data()
    held = split['test_source']; mm = meta.iloc[held].reset_index(drop=True)
    zs,ps,weights = {},{},{}
    for arm in REFERENCES+ARMS:
        folder = OUT if arm in ARMS else previous.OUT
        for seed in SEEDS:
            with np.load(folder/f'{arm}_{seed}_vectors.npz') as z:
                zs[arm,seed] = torch.from_numpy(z['z'])
            state = torch.load(previous.OUT/'pca_initial.pt' if arm=='pca' else folder/f'{arm}_{seed}.pt',
                               map_location='cpu',weights_only=True)
            if arm!='pca': state=state['selected']
            weights[arm,seed] = state['head.weight'].numpy()
            ps[arm,seed] = F.linear(zs[arm,seed],state['head.weight'],state['head.bias'])
    # Equal-vessel estimates on real held cross-acquisition pairs. The metric
    # uses each model's own head; it is not a common physically identified axis.
    records=[]
    products,ports=mm['product'].to_numpy(),mm.port.to_numpy()
    for arm in REFERENCES+ARMS:
        for seed in SEEDS:
            z=zs[arm,seed][held].numpy().astype(np.float64)
            w=weights[arm,seed].astype(np.float64)
            d=w[:,None,:]-w[None,:,:]
            d/=np.maximum(np.linalg.norm(d,axis=2,keepdims=True),1e-12)
            variance=np.square(z-z.mean(0)).mean()
            for ship,group in mm.groupby('mmsi',sort=True):
                ii,jj=np.triu_indices(len(group),1)
                i,j=group.index.to_numpy()[ii],group.index.to_numpy()[jj]
                eligible=products[i]!=products[j]
                label=int(group.class_id.iloc[0]); directions=d[label,np.arange(8)!=label]
                margin_variance=float(np.var(z@directions.T,axis=0).mean())
                for mode,mask in [('all',eligible),('cross_port',eligible&(ports[i]!=ports[j])),
                                  ('same_port',eligible&(ports[i]==ports[j]))]:
                    if not mask.any(): continue
                    delta=z[i[mask]]-z[j[mask]]
                    drift=float(np.square(delta@directions.T).mean())
                    records.append(dict(arm=arm,seed=seed,mmsi=ship,class_id=label,mode=mode,pairs=int(mask.sum()),
                        directional_drift=drift,normalized_directional_drift=drift/margin_variance,
                        normalized_vector_drift=float(np.square(delta).mean()/variance)))
    df=pd.DataFrame(records);df.to_csv(out/'held_direction_pairs.csv',index=False)
    keys=['directional_drift','normalized_directional_drift','normalized_vector_drift']
    df.groupby(['arm','seed','mode'])[keys].mean().reset_index().to_csv(out/'held_direction_summary.csv',index=False)
    comparisons=[('task','selected_directions'),('reliable','selected_directions')]
    tasks=[]; replay_count=0
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task=path.stem.removesuffix('_split');tasks.append(task)
        with np.load(path) as f:
            s,q,sy=[f[k] for k in ('support_obs_ids','query_obs_ids','support_labels')]
        archive=dict(query_obs_ids=q)
        with np.load(OUT/f'{task}_predictions.npz') as old:
            for a,b in comparisons:
                for seed in SEEDS:
                    for mode,pa,ka in [('reference',a,a),('prior_only',b,a),('kernel_only',a,b),('joint',b,b)]:
                        p=ps[pa,seed];design=phi(zs[ka,seed])
                        beta,_,_=residual_head(design[s],p[s],torch.tensor(sy),balanced=True)
                        score=(p[q]+design[q]@beta).numpy()
                        if mode in ('reference','joint'):
                            np.testing.assert_array_equal(score,old[f'{pa}__{seed}']);replay_count+=1
                        archive[f'{a}__{mode}__{seed}']=score
        np.savez_compressed(out/f'{task}.npz',**archive)
    rows=[]
    for task in tasks:
        with np.load(out/f'{task}.npz') as f:
            m=meta.iloc[f['query_obs_ids']];y=m.class_id.to_numpy()
            for a,b in comparisons:
                for seed in SEEDS:
                    for mode in ('reference','prior_only','kernel_only','joint'):
                        rows.append(dict(reference=a,candidate=b,task=task,port=m.port.iloc[0],seed=seed,mode=mode,
                            **metrics(f[f'{a}__{mode}__{seed}'].argmax(1),y)))
    df=pd.DataFrame(rows);df.to_csv(out/'target_results.csv',index=False)
    df.groupby(['reference','mode'])[['OA','BA']].mean().reset_index().to_csv(out/'target_summary.csv',index=False)
    df.groupby(['reference','mode','seed'])[['OA','BA']].mean().reset_index().to_csv(out/'target_seed_summary.csv',index=False)
    save_json(out/'COMPLETE.json',dict(new_training=0,exact_diagonal_replays=replay_count,
        scope='Frozen prior/kernel swap; own-head all7rival sourceheld pairs, not target tuning or causal effect identification.'))
    print(df.groupby(['reference','mode'])[['OA','BA']].mean().to_string(),flush=True)


if __name__=='__main__':
    main()
