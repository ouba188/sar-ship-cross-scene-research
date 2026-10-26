"""One capacity control: 137-wide raw MLP is slightly larger than 128-wide summary MLP."""
import numpy as np
import pandas as pd
import torch
import run


def main():
    torch.set_num_threads(2)
    for source in ('oss1','oss2'):
        root,cross,table,target,raw,eraw,local,elocal=run.base.dataset(source)
        fit=np.flatnonzero(table.role.eq('fit_train'))
        val=np.flatnonzero(table.role.eq('fit_validation'))
        x=run.base.representations(raw,local)['regional']
        ext=run.base.representations(eraw,elocal)['regional']
        scale=run.base.standardize(x,fit)
        x,ext=run.base.transform(x,scale),run.base.transform(ext,scale)
        out=run.HERE/'wide_control'/source
        out.mkdir(parents=True,exist_ok=True)
        src,dst,receipts={},{},[]
        for seed in run.base.SEEDS:
            src[f'raw_wide_seed{seed}'],dst[f'raw_wide_seed{seed}'],rec=run.train(x,ext,None,None,None,table,fit,val,seed,'raw_only',out,hidden=137)
            receipts.append(rec)
        np.savez_compressed(out/'source_predictions.npz',obs_id=table.obs_id.to_numpy(),**src)
        np.savez_compressed(out/'cross_predictions.npz',obs_id=target.obs_id.to_numpy(),**dst)
        pd.DataFrame(receipts).to_csv(out/'training.csv',index=False)
        truth=run.base.tr.read_table(cross/'target_table.csv',labels=True)
        np.testing.assert_array_equal(target.obs_id,truth.obs_id)
        rows=[]
        for key in src:
            for role,ids in table.groupby('role').indices.items():
                rows.append(dict(source=source,cohort=role,method=key,**run.base.classification_metrics(table.iloc[ids].class_id.to_numpy(int),src[key][ids])))
            rows.append(dict(source=source,cohort='cross',method=key,**run.base.classification_metrics(truth.class_id.to_numpy(int),dst[key])))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        run.base.write_json(out/'PROTOCOL.json',dict(hidden=137,parameters=211395,summary_candidate_parameters=210820,steps=200,seeds=run.base.SEEDS,source_selection_only=True,development='Added after positive raw_summary result to distinguish increased parameter capacity. No width sweep.'))
        print(pd.DataFrame(rows).query("cohort=='cross'").to_string(index=False),flush=True)


if __name__=='__main__':main()
