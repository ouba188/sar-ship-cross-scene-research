"""No retraining: source/test shifts and exact conditional-block removal."""
import numpy as np
import pandas as pd
import torch
from run import ROOT, ARMS, load_data, matrices, prior


def main():
    f,splits=load_data(); rows=[]; removals=[]; cases=[]
    review=pd.read_csv(ROOT/'case_review.csv')
    case_ids=set(review.obs_id)|set(review.partner_obs[review.partner_obs.ge(0)])
    for fold,(tr,val,test) in enumerate(splits):
        raw=np.load(ROOT/f'features_f{fold}.npz')
        z,_=matrices(raw,f,splits[fold],fold)
        covariance=raw['joint']-raw['factorized']
        for part,ii in (('train',tr),('val',val),('test',test)):
            norm=np.linalg.norm(raw['joint'][ii].reshape(len(ii),-1),axis=1)
            covnorm=np.linalg.norm(covariance[ii].reshape(len(ii),-1),axis=1)
            for arm in ARMS:
                fnorm=np.linalg.norm(z[arm][ii],axis=1)
                rows.append(dict(fold=fold,partition=part,arm=arm,n=len(ii),
                    feature_norm_p50=np.quantile(fnorm,.5),feature_norm_p95=np.quantile(fnorm,.95),
                    feature_norm_max=fnorm.max(),covariance_fraction_p50=np.median(covnorm/np.maximum(norm,1e-8))))
        for seed in range(3):
            head=torch.load(ROOT/f'head_f{fold}_joint_condition_s{seed}.pt',map_location='cpu',weights_only=False)
            d=np.load(ROOT/f'scores_f{fold}_joint_condition_s{seed}.npz')
            weight=head['state']['weight'].numpy(); full=z['joint_condition'][test]
            replay=d['base']+full@weight.T
            np.testing.assert_allclose(replay,d['scores'],rtol=1e-5,atol=2e-5)
            blocks=full.reshape(len(test),2,2,256).copy()
            blocks[:,:,:,64:]=0
            removed=d['base']+blocks.reshape(len(test),-1)@weight.T
            if seed==0:
                for j,i in enumerate(test):
                    if int(f.iloc[i].obs_id) not in case_ids:
                        continue
                    truth=int(f.iloc[i].class_id)
                    rival=int(np.argmax(np.where(np.arange(8)==truth,-np.inf,d['base'][j])))
                    term=(full[j]*(weight[truth]-weight[rival])).reshape(2,2,256)
                    cases.append(dict(obs_id=int(f.iloc[i].obs_id),fold=fold,seed=seed,
                        incidence=f.iloc[i].incidence,cos2=f.iloc[i].cos2,sin2=f.iloc[i].sin2,
                        static_relation_margin=float(term[:,:,:64].sum()),
                        conditional_relation_margin=float(term[:,:,64:].sum()),
                        full_prediction=int(d['scores'][j].argmax()),removed_prediction=int(removed[j].argmax())))
            removals.append(dict(fold=fold,seed=seed,step=head['step'],
                flips=int(np.sum(removed.argmax(1)!=d['scores'].argmax(1))),
                **prior.metrics(f.iloc[test].class_id.to_numpy(),removed)))
            np.savez_compressed(ROOT/f'condition_removed_f{fold}_s{seed}.npz',rows=test,scores=removed)
    pd.DataFrame(rows).to_csv(ROOT/'feature_shift.csv',index=False)
    pd.DataFrame(removals).to_csv(ROOT/'condition_removal.csv',index=False)
    pd.DataFrame(cases).to_csv(ROOT/'case_condition_terms.csv',index=False)
    sums=[]
    for seed in range(3):
        scores=[]; yy=[]
        for fold in range(3):
            d=np.load(ROOT/f'condition_removed_f{fold}_s{seed}.npz')
            scores.append(d['scores']); yy.extend(f.iloc[d['rows']].class_id.tolist())
        sums.append(dict(seed=seed,**prior.metrics(np.asarray(yy),np.concatenate(scores))))
    pd.DataFrame(sums).to_csv(ROOT/'condition_removal_summary.csv',index=False)
    metrics=pd.read_csv(ROOT/'metrics.csv')
    metrics.groupby(['arm','partition'])[['oa','ba','brier']].mean().to_csv(ROOT/'fit_generalization.csv')
    metrics[metrics.partition.eq('test')].groupby('arm').step.agg(['count',lambda x:int((x>0).sum())]).to_csv(ROOT/'selected_nonzero.csv')


if __name__=='__main__':
    main()
