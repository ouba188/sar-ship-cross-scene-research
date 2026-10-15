"""Frozen prior/kernel attribution and deployable normalization export."""

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from train_explicit_candidate import OUT, SEEDS, ARMS, SUPPORTS, load_data, phi, residual_head, metrics, save_json


def crossport_drift():
    meta,split,*_ = load_data()
    ids = split['test_source']; mm = meta.iloc[ids].reset_index(drop=True)
    records = []
    for arm in ('pca',)+ARMS:
        for seed in SEEDS:
            with np.load(OUT/f'{arm}_{seed}_vectors.npz') as f:
                values = f['z'][ids].astype(np.float64)
            units = values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)
            variance = np.square(values-values.mean(0)).mean()
            for ship,group in mm.groupby('mmsi',sort=True):
                ii,jj = np.triu_indices(len(group),1)
                i,j = group.index.to_numpy()[ii],group.index.to_numpy()[jj]
                product = mm['product'].to_numpy(); ports = mm.port.to_numpy()
                different = product[i] != product[j]
                for mode,mask in [('same_port',different&(ports[i]==ports[j])),
                                  ('cross_port',different&(ports[i]!=ports[j]))]:
                    if mask.any():
                        a,b = i[mask],j[mask]
                        records.append(dict(arm=arm,seed=seed,mmsi=ship,mode=mode,pairs=len(a),
                            normalized_drift=float(np.square(values[a]-values[b]).mean()/variance),
                            direction_distance=float(np.mean(1-(units[a]*units[b]).sum(1)))))
    df = pd.DataFrame(records)
    df.to_csv(OUT/'mechanism/held_crossport_drift.csv',index=False)
    result = df.groupby(['arm','seed','mode']).agg(vessels=('mmsi','nunique'),pairs=('pairs','sum'),
        normalized_drift=('normalized_drift','mean'),direction_distance=('direction_distance','mean')).reset_index()
    result.to_csv(OUT/'mechanism/held_crossport_summary.csv',index=False)
    print(result.groupby(['arm','mode'])[['vessels','pairs','normalized_drift','direction_distance']].mean().to_string())


def main():
    torch.set_num_threads(1)
    out = OUT/'mechanism'; out.mkdir(exist_ok=False)
    meta,split,roles,source,raw,norm,fixed,h = load_data()
    fit,held = split['factor_fit'],split['test_source']
    np.savez_compressed(OUT/'INPUT_NORMALIZATION.npz',raw_center=split['center'],raw_scale=split['scale'])
    zs,ps,bases,rows = {},{},{},[]
    for arm in ('pca',)+ARMS:
        for seed in SEEDS:
            with np.load(OUT/f'{arm}_{seed}_vectors.npz') as z:
                values = torch.from_numpy(z['z'])
                basis,scale,center = [z[k] for k in ('basis','scale','input_center')]
                # A small mixed-role replay catches a wrong normalization/basis export.
                ids = np.r_[fit[:4],held[:4],roles['val_new_ship'][:4]]
                rebuilt = ((norm[ids]-center)@basis)/scale/np.sqrt(32)
                np.testing.assert_allclose(rebuilt,values[ids].numpy(),rtol=2e-5,atol=2e-6)
                np.testing.assert_allclose(basis.T@basis,np.eye(32),atol=2e-6,rtol=2e-5)
                bases[arm,seed] = basis
            state = torch.load(OUT/'pca_initial.pt' if arm == 'pca' else OUT/f'{arm}_{seed}.pt',
                               map_location='cpu',weights_only=True)
            if arm != 'pca':
                state = state['selected']
            priors = F.linear(values,state['head.weight'],state['head.bias'])
            zs[arm,seed],ps[arm,seed] = values,priors
            for c in sorted(meta.iloc[held].class_id.unique()):
                y = meta.iloc[held].class_id.to_numpy(); valid = y==c
                rows.append(dict(arm=arm,seed=seed,class_id=int(c),count=int(valid.sum()),
                    recall=float(100*(priors[held].argmax(1).numpy()[valid]==c).mean())))
    pd.DataFrame(rows).to_csv(out/'source_classes.csv',index=False)
    comparisons = [('task','relation'),('task','uniform'),('task','reliable')]
    overlaps = []
    for a,b in comparisons:
        for seed in SEEDS:
            overlap = float(np.square(bases[a,seed].T@bases[b,seed]).sum()/32)
            overlaps.append(dict(reference=a,candidate=b,seed=seed,mean_squared_subspace_cosine=overlap))
    pd.DataFrame(overlaps).to_csv(out/'subspace_overlap.csv',index=False)
    tasks = []
    for path in sorted(SUPPORTS.glob('*_split.npz')):
        task = path.stem.removesuffix('_split'); tasks.append(task)
        with np.load(path) as z:
            s,q,sy = [z[k] for k in ('support_obs_ids','query_obs_ids','support_labels')]
        archive = dict(query_obs_ids=q)
        with np.load(OUT/f'{task}_predictions.npz') as old:
            for a,b in comparisons:
                name = a+'_'+b
                for seed in SEEDS:
                    for mode,pa,ka in [('reference',a,a),('prior_only',b,a),('kernel_only',a,b),('joint',b,b)]:
                        prior = ps[pa,seed]; design = phi(zs[ka,seed])
                        beta,_,_ = residual_head(design[s],prior[s],torch.tensor(sy),balanced=True)
                        score = (prior[q]+design[q]@beta).numpy()
                        if mode in ('reference','joint'):
                            np.testing.assert_array_equal(score,old[f'{pa}__{seed}'])
                        archive[f'{name}__{mode}__{seed}'] = score
        np.savez_compressed(out/f'{task}.npz',**archive)
    save_json(out/'PREDICTIONS_SAVED.json',dict(tasks=len(tasks),query_metrics_read=False))
    rows = []
    for task in tasks:
        with np.load(out/f'{task}.npz') as z:
            mm = meta.iloc[z['query_obs_ids']]; y = mm.class_id.to_numpy()
            for a,b in comparisons:
                for seed in SEEDS:
                    for mode in ('reference','prior_only','kernel_only','joint'):
                        score = z[f'{a}_{b}__{mode}__{seed}']
                        rows.append(dict(task=task,port=mm.port.iloc[0],candidate=b,seed=seed,mode=mode,
                                         **metrics(score.argmax(1),y)))
    df = pd.DataFrame(rows); df.to_csv(out/'target_results.csv',index=False)
    for name,keys in [('summary',['candidate','mode']),('seed_summary',['candidate','mode','seed']),
                      ('port_summary',['candidate','mode','port'])]:
        df.groupby(keys)[['OA','BA']].mean().reset_index().to_csv(out/f'target_{name}.csv',index=False)
    save_json(out/'COMPLETE.json',dict(new_training=0,exact_diagonal_replays=108,
        exported_geometry_replays=10,scope='Frozen prior versus32D support kernel replacements; no target parameter selection. Diagonals exact, geometry replays numerically close with different GEMM batch shapes.'))
    print(df.groupby(['candidate','mode'])[['OA','BA']].mean().to_string(),flush=True)


if __name__ == '__main__':
    main()
