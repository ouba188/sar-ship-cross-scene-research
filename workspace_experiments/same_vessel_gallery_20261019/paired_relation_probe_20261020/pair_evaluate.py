"""Evaluate all frozen arms on independently selected one-pair-per-ship cohort."""
import numpy as np
import pandas as pd
from run import ROOT, ARMS, load_data


def main():
    f,_=load_data()
    pairs=pd.read_csv(ROOT/'data_review/main_pairs.csv',dtype={'mmsi':str})
    assert pairs.mmsi.is_unique
    results=[]; details=[]
    for seed in range(3):
        for arm in ('baseline',)+ARMS:
            predictions={}; bases={}; margins={}
            for fold in range(3):
                d=np.load(ROOT/f'scores_f{fold}_{arm}_s{seed}.npz')
                rr, ss, bs = d['rows'], d['scores'], d['base']
                for j,i in enumerate(rr):
                    row=f.iloc[i]; truth=int(row.class_id); oid=int(row.obs_id)
                    predictions[oid]=int(ss[j].argmax()); bases[oid]=int(bs[j].argmax())
                    scores=ss[j].copy(); scores[truth]=-np.inf
                    margins[oid]=float(ss[j,truth]-scores.max())
            local=[]
            for r in pairs.itertuples():
                pa,pb=predictions[r.obs_id_a],predictions[r.obs_id_b]
                ba,bb=bases[r.obs_id_a],bases[r.obs_id_b]
                a,b=pa==r.class_id,pb==r.class_id; ca,cb=ba==r.class_id,bb==r.class_id
                outcome=('C' if a else 'W')+('C' if b else 'W')
                old=('C' if ca else 'W')+('C' if cb else 'W')
                item=dict(pair_id=r.pair_id,mmsi=r.mmsi,class_id=r.class_id,fold=r.fold,seed=seed,arm=arm,
                    obs_id_a=r.obs_id_a,obs_id_b=r.obs_id_b,prediction_a=pa,prediction_b=pb,
                    outcome=outcome,baseline_outcome=old,same_wrong=bool(not a and not b and pa==pb),
                    margin_a=margins[r.obs_id_a],margin_b=margins[r.obs_id_b],
                    gained_correct=int(a)+int(b)-int(ca)-int(cb))
                local.append(item); details.append(item)
            z=pd.DataFrame(local)
            results.append(dict(arm=arm,seed=seed,ships=len(z),classes=z.class_id.nunique(),
                CC=int(z.outcome.eq('CC').sum()),CW=int(z.outcome.eq('CW').sum()),WC=int(z.outcome.eq('WC').sum()),
                WW=int(z.outcome.eq('WW').sum()),same_wrong=int(z.same_wrong.sum()),
                single_to_CC=int((z.baseline_outcome.isin(['CW','WC'])&z.outcome.eq('CC')).sum()),
                single_to_WW=int((z.baseline_outcome.isin(['CW','WC'])&z.outcome.eq('WW')).sum()),
                lost_CC=int((z.baseline_outcome.eq('CC')&z.outcome.ne('CC')).sum()),
                net_correct_observations=int(z.gained_correct.sum())))
    result=pd.DataFrame(results); result.to_csv(ROOT/'pair_metrics.csv',index=False)
    pd.DataFrame(details).to_csv(ROOT/'pair_predictions.csv',index=False)
    result.groupby('arm').mean(numeric_only=True).to_csv(ROOT/'pair_summary.csv')
    expected=pd.read_csv(ROOT/'data_review/main_pairs_seed0.csv',dtype={'mmsi':str}).set_index('pair_id')
    actual=pd.DataFrame(details)
    actual=actual[actual.arm.eq('baseline')&actual.seed.eq(0)].set_index('pair_id').loc[expected.index]
    np.testing.assert_array_equal(actual.outcome.to_numpy(),expected.outcome.to_numpy())
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
