"""Fixed-input source-reader rowspace/complement diagnostic, not causal discovery."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import argparse
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'paper_evidence_completion_20261026'
sys.path.insert(0,str(PRIOR))
import run_classical as base


def projected_inputs(region, external, valid, evalid, fit, W):
    x=region.astype(np.float64)
    xe=external.astype(np.float64)
    row, erow, principal, eprincipal=np.zeros_like(x),np.zeros_like(xe),np.zeros_like(x),np.zeros_like(xe)
    receipt=[]
    bases={}
    for p in range(2):
        _,sing,vh=np.linalg.svd(W[p],full_matrices=False)
        rank=int((sing>np.finfo(float).eps*max(W[p].shape)*sing[0]).sum())
        V=vh[:rank]
        row[:,p]=(x[:,p]@V.T)@V
        erow[:,p]=(xe[:,p]@V.T)@V
        samples=x[fit,p].reshape(-1,256)
        keep=np.repeat(valid[fit,p],3)
        P=PCA(n_components=rank,svd_solver='full').fit(samples[keep]).components_
        principal[:,p]=(x[:,p]@P.T)@P
        eprincipal[:,p]=(xe[:,p]@P.T)@P
        null=x[:,p]-row[:,p]
        enull=xe[:,p]-erow[:,p]
        preserve=float(np.max(abs(row[:,p]@W[p].T-x[:,p]@W[p].T)))
        remove=float(np.max(abs(null@W[p].T)))
        extremove=float(np.max(abs(enull@W[p].T)))
        assert max(preserve,remove,extremove)<1e-8
        assert np.max(abs((row[:,p]@V.T)@V-row[:,p]))<1e-8
        bases[f'rowspace_{p}']=V
        bases[f'pca_{p}']=P
        receipt.append(dict(polarization=('VV','VH')[p],rank=rank,fit_regions=int(keep.sum()),
            response_preservation_error=preserve,null_response_error=remove,cross_null_error=extremove))
    def pack(a,flags):
        return np.c_[a.reshape(len(a),-1),flags].astype(np.float32)
    values={
        'direct':(pack(x,valid),pack(xe,evalid)),
        'reader_span':(pack(row,valid),pack(erow,evalid)),
        'reader_complement':(pack(x-row,valid),pack(xe-erow,evalid)),
        'pca_same_rank':(pack(principal,valid),pack(eprincipal,evalid)),
    }
    np.testing.assert_allclose(row+(x-row),x,rtol=1e-12,atol=1e-12)
    return values,bases,receipt


def summary_inputs(root,cross,table,target):
    keys=('summary1','quality','geometry')
    src=base.load_selected(root/'frontend/summaries.npz',table,keys)
    dst=base.load_selected(cross/'summaries/summaries.npz',target,keys)
    # Geometry and quality have the same transformations used by the original M2.
    def flatten(z):
        fields=[]
        for p in range(2):
            g,q=z['geometry'][:,p],z['quality'][:,p]
            fields.extend([np.log1p(g[:,0]),np.log1p(g[:,1]),g[:,2],g[:,3]])
            for r in range(3):
                fields.extend([np.log1p(q[:,r,0]),q[:,r,1],np.log1p(q[:,r,2]),q[:,r,3]])
        return np.c_[z['summary1'].reshape(len(z['summary1']),-1),np.stack(fields,-1)].astype(np.float32)
    return flatten(src),flatten(dst)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',choices=('oss1','oss2'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    root,cross,table,target,raw,eraw,local,elocal=base.dataset(args.source)
    fit=np.flatnonzero(table.role.eq('fit_train'))
    val=np.flatnonzero(table.role.eq('fit_validation'))
    test=np.flatnonzero(table.role.eq('evaluation'))
    for key in ('mmsi','pass_group'):
        assert not set(table.iloc[fit][key])&set(table.iloc[val][key])
        assert not set(table.loc[table.role.ne('evaluation'),key])&set(target[key])
    src=base.representations(raw,local)['regional']
    dst=base.representations(eraw,elocal)['regional']
    xr=src[:,:-2].reshape(-1,2,3,256)
    xe=dst[:,:-2].reshape(-1,2,3,256)
    W=np.load(root/'frontend/models/reader_layer1.npz')['W'].reshape(4,2,256).transpose(1,0,2)
    values,bases,receipt=projected_inputs(xr,xe,raw['valid'],eraw['valid'],fit,W)
    np.testing.assert_array_equal(values['direct'][0],src)
    np.testing.assert_array_equal(values['direct'][1],dst)
    values['actual_m2_summary']=summary_inputs(root,cross,table,target)
    out=HERE/'compression'/args.source
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'bases.npz',**bases)
    base.write_json(out/'projection_checks.json',receipt)
    base.write_json(out/'protocol.json',dict(source=args.source,seeds=base.SEEDS,fit=len(fit),validation=len(val),
        arms=list(values),steps=200,hidden=128,training_roles='fit_train / fit_validation',
        representations='raw region means projected into existing per-pol class-reader rowspace/complement or equal-rank source PCA',
        summary_control='actual M2 mean/high-low summaries +32 quality/geometry inputs; different dimension, no upstream score',
        scaler='each arm source fit only, constant columns masked; no target fit',
        same_nominal_dimension=['direct','reader_span','reader_complement','pca_same_rank'],
        scope='information availability under fixed readout family, not complete pipeline ablation'))
    outputs,training=[],[]
    sources,externals={},{}
    for arm,(x,ext) in values.items():
        path=out/arm
        path.mkdir(exist_ok=True)
        sc=base.standardize(x,fit)
        np.savez_compressed(path/'scaler.npz',**sc)
        x,ext=base.transform(x,sc),base.transform(ext,sc)
        s,e,record=base.train_ridge(x,ext,table,fit,val,path)
        sources[arm+'_ridge'],externals[arm+'_ridge']=s,e
        training.append(dict(arm=arm,head='ridge',seed=0,**record))
        for seed in base.SEEDS:
            s,e,record=base.train_mlp(x,ext,table,fit,val,seed,path)
            sources[f'{arm}_mlp_seed{seed}'],externals[f'{arm}_mlp_seed{seed}']=s,e
            training.append(dict(arm=arm,head='mlp',seed=seed,**record))
        print('FROZEN',args.source,arm,flush=True)
    np.savez_compressed(out/'source_predictions.npz',obs_id=table.obs_id.to_numpy(),**sources)
    np.savez_compressed(out/'cross_predictions.npz',obs_id=target.obs_id.to_numpy(),**externals)
    pd.DataFrame(training).to_csv(out/'training_receipts.csv',index=False)
    # Freeze every arm before any target-label evaluation.
    target_labels=base.tr.read_table(cross/'target_table.csv',labels=True)
    np.testing.assert_array_equal(target.obs_id,target_labels.obs_id)
    rows=[]
    for arm,s in sources.items():
        for role,ix in table.groupby('role').indices.items():
            rows.append(dict(source=args.source,cohort=role,method=arm,
                **base.classification_metrics(table.iloc[ix].class_id.to_numpy(int),s[ix])))
        rows.append(dict(source=args.source,cohort='cross',method=arm,
            **base.classification_metrics(target_labels.class_id.to_numpy(int),externals[arm])))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    # Direct arm reuses the prior pipeline exactly; changes here would confound diagnosis.
    replay=[]
    for side,data in [('source',sources),('cross',externals)]:
        old=np.load(PRIOR/'classical_fit_only'/args.source/f'{side}_predictions.npz')
        for name,score in data.items():
            if name.startswith('direct_'):
                oldscore=old[name.replace('direct_','regional_',1)]
                error=float(abs(score-oldscore).max())
                np.testing.assert_allclose(score,oldscore,rtol=1e-5,atol=2e-5)
                np.testing.assert_array_equal(score.argmax(1),oldscore.argmax(1))
                replay.append(dict(side=side,method=name,max_error=error))
    base.write_json(out/'complete.json',dict(fits=20,target_selection=False,direct_replay=replay,raw_review_pending=True))
    print(pd.DataFrame(rows).query("cohort in ['evaluation','cross']").to_string(index=False),flush=True)


if __name__=='__main__':
    main()
