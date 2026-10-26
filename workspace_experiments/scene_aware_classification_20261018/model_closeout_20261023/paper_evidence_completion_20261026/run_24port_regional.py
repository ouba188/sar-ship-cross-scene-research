"""Same regional mean controls on the original 24-port-origin 8-class cohort."""
import argparse
import numpy as np
import pandas as pd
import torch
from run_classical import (HERE,ROOT,SEEDS,tr,representations,standardize,transform,
                          train_ridge,train_mlp,classification_metrics,write_json)


def region_features(path,table):
    z=tr.load_npz(path)
    ix=pd.Index(z['instance_ids'].astype(str)).get_indexer(table.instance_id.astype(str))
    assert (ix>=0).all()
    local={k:z[k][ix] for k in ('layer1','mass')}
    raw=dict(valid=z['valid'][ix],global1=np.zeros((len(table),2,256)),global3=np.zeros((len(table),2,1024)))
    return representations(raw,local)['regional']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training-role',choices=('pooled','fit_only'),default='pooled')
    args=p.parse_args()
    torch.set_num_threads(2)
    upstream=ROOT/'regional_utility_pilot/upstream'
    local=ROOT/'local_evidence_transport'
    external=local/'external_confirmation'
    table=pd.read_csv(upstream/'table.csv',dtype={'mmsi':str,'instance_id':str})
    target=pd.read_csv(external/'table.csv',dtype={'mmsi':str,'instance_id':str})
    roles=tr.load_npz(upstream/'indices.npz')
    fit=roles['fit_train']
    val=roles['fit_validation']
    if args.training_role=='pooled':
        fit=np.r_[fit,np.flatnonzero(table.role.eq('utility_train'))]
        val=np.r_[val,np.flatnonzero(table.role.eq('selection'))]
    test=np.flatnonzero(table.role.eq('evaluation'))
    for field in ('mmsi','pass_group'):
        assert not set(table.iloc[fit][field]) & set(table.iloc[val][field])
        assert not set(table.iloc[np.r_[fit,val]][field]) & set(target[field])
        assert not set(table.iloc[np.r_[fit,val]][field]) & set(table.iloc[test][field])
    out=HERE/'regional_24port'/args.training_role
    out.mkdir(parents=True,exist_ok=True)
    x,xe=region_features(local/'tokens.npz',table),region_features(external/'tokens.npz',target)
    scale=standardize(x,fit)
    np.savez_compressed(out/'scaler.npz',**scale)
    x,xe=transform(x,scale),transform(xe,scale)
    s,e,receipt=train_ridge(x,xe,table,fit,val,out)
    predictions={'regional_ridge':(s,e)}
    receipts=[dict(method='regional_ridge',**receipt)]
    for seed in SEEDS:
        s,e,receipt=train_mlp(x,xe,table,fit,val,seed,out)
        predictions[f'regional_mlp_seed{seed}']=(s,e)
        receipts.append(dict(method='regional_mlp',seed=seed,**receipt))
    np.savez_compressed(out/'source_predictions.npz',obs_id=table.obs_id.to_numpy(),**{k:v[0] for k,v in predictions.items()})
    np.savez_compressed(out/'expanded_predictions.npz',obs_id=target.obs_id.to_numpy(),**{k:v[1] for k,v in predictions.items()})
    rows=[]
    for method,(s,e) in predictions.items():
        rows.append(dict(cohort='dev485',method=method,**classification_metrics(table.iloc[test].class_id.to_numpy(int),s[test])))
        rows.append(dict(cohort='expanded3075',method=method,**classification_metrics(target.class_id.to_numpy(int),e)))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(receipts).to_csv(out/'training_receipts.csv',index=False)
    write_json(out/'complete.json',dict(training_role=args.training_role,fit=len(fit),selection=len(val),
        development=len(test),expanded=len(target),classes=8,source_only=True,
        representation='same native C4 layer1 OBB body/side/end region means; no class-response projection'))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
