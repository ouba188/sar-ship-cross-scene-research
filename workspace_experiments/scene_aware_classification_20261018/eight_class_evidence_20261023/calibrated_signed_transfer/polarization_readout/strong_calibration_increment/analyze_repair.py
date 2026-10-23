"""Separate exact repairs of T-induced errors from other saved-prediction flips."""
import numpy as np
import pandas as pd
from run_increment import HERE,previous


def main():
    p=pd.read_csv(HERE/'predictions.csv')
    old=pd.read_csv(previous.HERE/'predictions.csv')
    key=['fold','seed','role','instance_id']
    old=old[old.arm.eq('score_only')].set_index(key)
    rows=[]
    for (role,arm,seed),g in p.groupby(['role','arm','seed']):
        g=g.set_index(key)
        before=old.loc[g.index]
        np.testing.assert_array_equal(g.class_id,before.class_id)
        np.testing.assert_allclose(g[[f'b{c}' for c in range(8)]],before[[f's{c}' for c in range(8)]],atol=1e-12,rtol=0)
        ad=before.base_prediction.eq(g.class_id)
        b=g.base_prediction.eq(g.class_id)
        after=g.prediction.eq(g.class_id)
        for c in range(8):
            hit=g.class_id.eq(c)
            rows.append(dict(role=role,arm=arm,seed=seed,class_id=c,n=int(hit.sum()),
                T_net=int((b.astype(int)-ad.astype(int))[hit].sum()),
                T_harmed=int((hit&ad&~b).sum()),T_rescued=int((hit&~ad&b).sum()),
                recovered_T_harm=int((hit&ad&~b&after).sum()),
                rescued_both_AD_and_B_wrong=int((hit&~ad&~b&after).sum()),
                newly_harmed=int((hit&b&~after).sum()),
                lost_T_rescue=int((hit&~ad&b&~after).sum()),
                increment_net=int((after.astype(int)-b.astype(int))[hit].sum()),
                final_vs_AD_net=int((after.astype(int)-ad.astype(int))[hit].sum())))
    frame=pd.DataFrame(rows)
    frame.to_csv(HERE/'calibration_repair_counts.csv',index=False)
    print(frame[frame.arm.eq('local')&frame.class_id.isin([2,4,5])].to_string(index=False))


if __name__=='__main__':main()
