"""Equivalent compact coordinates check for the low-rank reader projection."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import probe_compression as probe

base=probe.base
HERE=probe.HERE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',choices=('oss1','oss2'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    root,cross,table,target,raw,ext,local,elocal=base.dataset(args.source)
    fit=np.flatnonzero(table.role.eq('fit_train'))
    val=np.flatnonzero(table.role.eq('fit_validation'))
    xs=base.representations(raw,local)['regional']
    xe=base.representations(ext,elocal)['regional']
    bs=np.load(HERE/'compression'/args.source/'bases.npz')
    out=HERE/'compact_coordinates'/args.source
    out.mkdir(parents=True,exist_ok=True)
    sources,externals,receipts={},{},[]
    for arm,prefix in [('reader_coords','rowspace'),('pca_coords','pca')]:
        packed=[]
        for values,flags in ((xs,raw['valid']),(xe,ext['valid'])):
            region=values[:,:-2].reshape(-1,2,3,256).astype(float)
            parts=[(region[:,p]@bs[f'{prefix}_{p}'].T).reshape(len(region),-1) for p in range(2)]
            packed.append(np.c_[*parts,flags].astype(np.float32))
        x,e=packed
        dest=out/arm
        dest.mkdir(exist_ok=True)
        sc=base.standardize(x,fit)
        np.savez_compressed(dest/'scaler.npz',**sc)
        x,e=base.transform(x,sc),base.transform(e,sc)
        s,t,record=base.train_ridge(x,e,table,fit,val,dest)
        sources[arm+'_ridge'],externals[arm+'_ridge']=s,t
        receipts.append(dict(arm=arm,head='ridge',seed=0,**record))
        for seed in base.SEEDS:
            s,t,record=base.train_mlp(x,e,table,fit,val,seed,dest)
            sources[f'{arm}_mlp_seed{seed}'],externals[f'{arm}_mlp_seed{seed}']=s,t
            receipts.append(dict(arm=arm,head='mlp',seed=seed,**record))
    np.savez_compressed(out/'source_predictions.npz',obs_id=table.obs_id.to_numpy(),**sources)
    np.savez_compressed(out/'cross_predictions.npz',obs_id=target.obs_id.to_numpy(),**externals)
    pd.DataFrame(receipts).to_csv(out/'training_receipts.csv',index=False)
    label=base.tr.read_table(cross/'target_table.csv',labels=True)
    np.testing.assert_array_equal(label.obs_id,target.obs_id)
    rows=[]
    for arm,s in sources.items():
        for role,ix in table.groupby('role').indices.items():
            rows.append(dict(source=args.source,cohort=role,method=arm,
                **base.classification_metrics(table.iloc[ix].class_id.to_numpy(int),s[ix])))
        rows.append(dict(source=args.source,cohort='cross',method=arm,
            **base.classification_metrics(label.class_id.to_numpy(int),externals[arm])))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    base.write_json(out/'complete.json',dict(fits=8,source_basis_reused=True,target_selection=False,
        purpose='same retained information represented by independent low-rank coordinates, not 256 repeated coordinates',
        scope='dimension/conditioning diagnostic; nominal MLP parameter count differs from reconstructed controls'))
    print(pd.DataFrame(rows).query("cohort=='cross'").to_string(index=False),flush=True)


if __name__=='__main__':main()
