"""Read-only fixed-reader/map diagnostics, no fitting or feature extraction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
NEW=ROOT/'target_anchored_context_20261023/partition_replication'
SPATIAL=ROOT/'target_anchored_context_20261023/spatial_evidence_review'


def pool(x):
    return np.sum(softmax(x,axis=-1)*x,axis=-1)


def main():
    weights=[]
    for fold in range(6):
        with np.load(NEW/'origins'/f'fold{fold}_outer.npz') as d:
            w=d['W'];fit=d['fit'];intercept=d['raw_intercept']
        with np.load(NEW/f'fold{fold}_inputs.npz') as inp:
            np.testing.assert_array_equal(fit,inp['train'])
        weights.append(dict(fold=fold,classes=w.shape[0],channels=w.shape[1],
            mean_weight_relative_norm=float(np.linalg.norm(w.mean(0))/np.linalg.norm(w)),
            class_sum_weight_max=float(abs(w.sum(0)).max()),intercept_sum=float(intercept.sum())))
    pd.DataFrame(weights).to_csv(HERE/'ridge_common_field.csv',index=False)
    rows=[];totals=[]
    cases=pd.read_csv(SPATIAL/'cases.csv')
    rng=np.random.default_rng(20261023)
    for case in cases.itertuples():
        with np.load(SPATIAL/f'{case.instance_id}_decomposition.npz') as d:
            w=d['W'];cl=[];co=[];st=[]
            with np.load(NEW/'origins'/f'fold{case.fold}_outer.npz') as source:
                np.testing.assert_array_equal(w,source['W'])
            for pol,key in enumerate(('vv','vh')):
                maps=d[key+'_maps_c4'].astype(float)
                for rot in range(4):
                    features=maps[rot].reshape(1024,-1)
                    centered=features-features.mean(1,keepdims=True)
                    v=w[:,pol*1024:(pol+1)*1024]@centered
                    r=v-v.mean(0,keepdims=True)
                    a,b,s=pool(v),pool(r),r.std(axis=-1,ddof=0)
                    cl.append(a/4);co.append(b/4);st.append(s/4)
                    # These checks catch class/spatial-axis mixups and a misplaced centering.
                    spatial_mean=float(abs(v.mean(1)).max())
                    assert spatial_mean<1e-10
                    assert a.min()>-1e-10 and b.min()>-1e-10
                    permutation=rng.permutation(v.shape[-1])
                    np.testing.assert_allclose(pool(r[:,permutation]),b,atol=1e-12)
                    common=rng.normal(size=(1,v.shape[-1]));common-=common.mean()
                    perturbed=v+common
                    rp=perturbed-perturbed.mean(0,keepdims=True)
                    np.testing.assert_allclose(pool(rp),b,atol=1e-12)
                    rows.append(dict(instance_id=case.instance_id,fold=case.fold,pol=key,rotation=rot,
                        height=maps.shape[-2],width=maps.shape[-1],spatial_mean_error=spatial_mean,
                        common_field_max=float(abs(v.mean(0)).max()),v_minus_r_max=float(abs(v-r).max()),
                        class_minus_contrast_max=float(abs(a-b).max()),
                        common_invariance_error=float(abs(pool(rp)-b).max()),
                        class_response_to_common=float(abs(pool(perturbed)-a).max())))
            csum=np.sum(cl,axis=0);rsum=np.sum(co,axis=0);ssum=np.sum(st,axis=0)
            totals.append(dict(instance_id=case.instance_id,fold=case.fold,
                class_minus_contrast_max=float(abs(csum-rsum).max()),
                **{f'class_delta_{c}':float(csum[c]) for c in range(3)},
                **{f'contrast_delta_{c}':float(rsum[c]) for c in range(3)},
                **{f'std_delta_{c}':float(ssum[c]) for c in range(3)}))
    pd.DataFrame(rows).to_csv(HERE/'cached_map_checks.csv',index=False)
    pd.DataFrame(totals).to_csv(HERE/'cached_case_deltas.csv',index=False)
    # Explicit counterexample: same mean and std do not determine softmax expectation.
    x=np.array([-1.,-1.,1.,1.])
    z=np.array([-np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)])
    np.testing.assert_allclose([x.mean(),x.std()],[z.mean(),z.std()],atol=1e-12)
    summary=dict(folds=6,cached_cases=len(cases),pol_rotation_cases=len(rows),fitting=False,extraction=False,
        mean_weight_relative_norm_max=max(r['mean_weight_relative_norm'] for r in weights),
        class_minus_contrast_delta_max=max(r['class_minus_contrast_max'] for r in totals),
        common_invariance_max=max(r['common_invariance_error'] for r in rows),
        equal_std_example=dict(x=x.tolist(),z=z.tolist(),std=float(x.std()),pool_x=float(pool(x)),pool_z=float(pool(z))),
        conclusion='Current onehot Ridge class sum almost zero: class/contrast pooling nearly coincide in four real cached cases. Not a full-cohort claim.')
    (HERE/'formula_checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(weights).to_string(index=False))
    print(pd.DataFrame(totals).to_string(index=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
