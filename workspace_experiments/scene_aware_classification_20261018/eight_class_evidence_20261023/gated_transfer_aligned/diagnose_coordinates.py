"""Quantify reachable inner/outer controller-coordinate mismatches, no fitting."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BASELINE=HERE.parent/'baseline'


def main():
    started=time.perf_counter()
    samples,dimensions,axes,rows=[],[],[],[]
    for fold in range(3):
        with np.load(BASELINE/f'fold{fold}_inputs.npz') as d:
            outer=d['whole64'];oof=d['oof_whole64'];ids=d['instance_ids']
            source=d['train'];bins=d['inner_bin']
        with np.load(BASELINE/'models'/f'fold{fold}_outer.npz') as m:
            oc=m['cond_pca_components'].astype(float);os=m['cond_scale_scale']
        for inner in range(4):
            with np.load(BASELINE/'models'/f'fold{fold}_inner{inner}.npz') as m:
                held=m['held'];ic=m['cond_pca_components'].astype(float);isc=m['cond_scale_scale']
            np.testing.assert_array_equal(held,source[bins[source]==inner])
            x=oof[held].astype(float);z=outer[held].astype(float)
            cosine=np.sum(x*z,axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(z,axis=1))
            centered_x=x-x.mean(0);centered_z=z-z.mean(0)
            denom=np.linalg.norm(centered_x,axis=0)*np.linalg.norm(centered_z,axis=0)
            corr=np.divide(np.sum(centered_x*centered_z,axis=0),denom,out=np.full(64,np.nan),where=denom>0)
            for i,cos in zip(held,cosine):
                samples.append(dict(fold=fold,inner=inner,row_index=int(i),instance_id=ids[i],cosine=float(cos)))
            dimensions.extend(dict(fold=fold,inner=inner,dimension=g,correlation=corr[g],defined=bool(denom[g]>0)) for g in range(64))
            diagcos=None;bestcos=None;same=None
            for space,a,b in [('standardized_stem',ic,oc),('original_stem_effective',ic/isc,oc/os)]:
                a=a/np.linalg.norm(a,axis=1,keepdims=True);b=b/np.linalg.norm(b,axis=1,keepdims=True)
                sim=a@b.T
                diag=np.diag(sim);best=np.argmax(abs(sim),axis=1)
                for g in range(64):
                    axes.append(dict(fold=fold,inner=inner,space=space,inner_axis=g,same_signed_cosine=diag[g],
                        same_absolute_cosine=abs(diag[g]),best_outer_axis=int(best[g]),best_signed_cosine=sim[g,best[g]],
                        best_absolute_cosine=abs(sim[g,best[g]]),best_is_same=bool(best[g]==g)))
                if space=='standardized_stem':
                    diagcos=diag;bestcos=abs(sim[np.arange(64),best]);same=best==np.arange(64)
            rows.append(dict(fold=fold,inner=inner,n=len(held),cosine_mean=cosine.mean(),cosine_median=np.median(cosine),
                cosine_p05=np.quantile(cosine,.05),cosine_p95=np.quantile(cosine,.95),negative_sample_fraction=(cosine<0).mean(),
                dimension_correlation_mean=np.nanmean(corr),dimension_absolute_correlation_mean=np.nanmean(abs(corr)),
                negative_dimension_correlations=int((corr<0).sum()),undefined_dimensions=int((denom==0).sum()),
                same_axis_signed_mean=diagcos.mean(),same_axis_abs_mean=abs(diagcos).mean(),
                same_axis_negative=int((diagcos<0).sum()),best_axis_abs_mean=bestcos.mean(),same_best_axes=int(same.sum())))
    pd.DataFrame(samples).to_csv(HERE/'coordinate_samples.csv',index=False)
    pd.DataFrame(dimensions).to_csv(HERE/'coordinate_dimensions.csv',index=False)
    pd.DataFrame(axes).to_csv(HERE/'coordinate_axes.csv',index=False)
    pd.DataFrame(rows).to_csv(HERE/'coordinate_summary.csv',index=False)
    s=pd.DataFrame(samples);a=pd.DataFrame(axes);a=a[a.space.eq('standardized_stem')]
    summary=dict(source_fold_positions=len(s),inner_fits=12,mean_sample_cosine=float(s.cosine.mean()),
        median_sample_cosine=float(s.cosine.median()),negative_sample_fraction=float(s.cosine.lt(0).mean()),
        same_axis_abs_mean=float(a.same_absolute_cosine.mean()),same_axis_signed_mean=float(a.same_signed_cosine.mean()),
        negative_same_axes=int(a.same_signed_cosine.lt(0).sum()),axis_pairs=len(a),
        best_axis_abs_mean=float(a.best_absolute_cosine.mean()),same_best_axis_fraction=float(a.best_is_same.mean()),
        seconds=time.perf_counter()-started,refits=0,
        conclusion='Reachable shared-MLP coordinate mismatch; does not attribute all original loss to this issue.')
    (HERE/'coordinate_diagnosis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
