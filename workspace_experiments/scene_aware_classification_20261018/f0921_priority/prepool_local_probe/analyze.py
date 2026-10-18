"""Paired changes, confidence control, and region-readout diagnostics."""
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from run import HERE,PREV


def main():
    pred=pd.read_csv(HERE/'predictions.csv');met=pd.read_csv(HERE/'metrics.csv')
    meta=pd.read_csv(PREV/'cohort.csv');features=np.load(HERE/'local_features.npz')
    keys=['fold','draw','row'];new=pred[pred.arm.eq('prepool')].set_index(keys).sort_index()
    cols=[f'p{k}' for k in range(7)];paired=[];summary=[];confidence=[]
    for comparator in ['global_grid','grid12','postpool']:
        old=pred[pred.arm.eq(comparator)].set_index(keys).sort_index()
        d=new[['truth','prediction','correct','ap','relation_margin','brier','nll']].join(old[['prediction','correct','ap','relation_margin','brier','nll']],rsuffix='_old')
        d['rescue']=d.correct&~d.correct_old;d['harm']=~d.correct&d.correct_old
        for field in ['ap','relation_margin','brier','nll']:d['delta_'+field]=d[field]-d[field+'_old']
        d['comparator']=comparator
        before=.1*np.log(old[cols].to_numpy());after=.1*np.log(new[cols].to_numpy())
        before-=before.mean(1,keepdims=True);after-=after.mean(1,keepdims=True)
        bn=np.linalg.norm(before,axis=1);an=np.linalg.norm(after,axis=1)
        ratios=an/np.maximum(bn,1e-12)
        matched=softmax(before*ratios[:,None]/.1,axis=1)
        truth=new.truth.to_numpy(dtype=int)
        d['norm_ratio']=ratios;d['logit_direction_cosine']=(before*after).sum(1)/np.maximum(an*bn,1e-12)
        d['matched_brier']=np.square(matched-np.eye(7)[truth]).sum(1)
        d['matched_nll']=-np.log(np.maximum(matched[np.arange(len(truth)),truth],1e-12))
        d['neighbor_present']=meta.iloc[d.index.get_level_values('row')].image_other_fraction.to_numpy()>0
        d=d.reset_index();paired.append(d)
        changes=met[met.arm.eq('prepool')].set_index(['fold','draw'])[['oa','ba','map_r','brier']]-met[met.arm.eq(comparator)].set_index(['fold','draw'])[['oa','ba','map_r','brier']]
        draws=changes.groupby('draw').mean();draws.to_csv(HERE/f'paired_draws_vs_{comparator}.csv')
        summary.append(dict(comparator=comparator,rescues=int(d.rescue.sum()),harms=int(d.harm.sum()),
            unique_rescued_queries=int(d[d.rescue].row.nunique()),unique_harmed_queries=int(d[d.harm].row.nunique()),
            wrong_to_wrong=int(((d.prediction!=d.prediction_old)&~d.correct&~d.correct_old).sum()),
            improved_ap=int(d.delta_ap.gt(1e-10).sum()),worsened_ap=int(d.delta_ap.lt(-1e-10).sum()),
            oa_delta_pp=float(100*draws.oa.mean()),oa_draw_sd_pp=float(100*draws.oa.std()),
            ba_delta_pp=float(100*draws.ba.mean()),ba_draw_sd_pp=float(100*draws.ba.std()),
            oa_positive_draws=int(draws.oa.gt(1e-10).sum()),oa_negative_draws=int(draws.oa.lt(-1e-10).sum()),
            brier_better_draws=int(draws.brier.lt(0).sum()),
            mean_logit_norm_ratio=float(ratios.mean()),mean_direction_cosine=float(d.logit_direction_cosine.mean())))
        group=d.groupby(['fold','draw'])[['brier_old','brier','matched_brier','nll_old','nll','matched_nll']].mean().mean()
        confidence.append(dict(comparator=comparator,**group.to_dict()))
    all_pairs=pd.concat(paired);all_pairs.to_csv(HERE/'paired_effects.csv',index=False)
    sm=pd.DataFrame(summary);sm.to_csv(HERE/'paired_summary.csv',index=False)
    cf=pd.DataFrame(confidence);cf.to_csv(HERE/'confidence_controls.csv',index=False)
    all_pairs.groupby(['comparator','neighbor_present'])[['rescue','harm','delta_ap','delta_brier']].agg(['mean','sum','count']).to_csv(HERE/'neighbor_strata.csv')
    cov=pd.read_csv(HERE/'token_support_coverage.csv');co=cov[cov.arm.eq('prepool')].copy();co['role']=co.token%3
    co.groupby('role').support_count.agg(['min','median','mean','max']).to_csv(HERE/'role_support_coverage.csv')
    mass=features['mass12'];active=mass>0
    area=pd.DataFrame(dict(row=np.repeat(np.arange(len(mass)),12),token=np.tile(np.arange(12),len(mass)),mass=mass.ravel(),available=active.ravel()))
    area['role']=area.token%3;area.to_csv(HERE/'token_areas.csv',index=False)
    dec=pd.read_csv(HERE/'pre_post_margin_decomposition.csv')
    dec['class_raw']=meta.iloc[dec.row].class_raw.to_numpy()
    dec.groupby('class_raw')[['target','other','background','delta_fixed_margin']].mean().to_csv(HERE/'role_margin_by_class.csv')
    stat=dict(available_tokens=int(active.sum()),total_tokens=int(active.size),
        active_tokens_under_one_feature_cell=int(((mass>0)&(mass<1)).sum()),
        active_tokens_under_four_feature_cells=int(((mass>0)&(mass<4)).sum()),
        absent_support_token_episodes=int(co.support_count.eq(0).sum()),
        single_support_token_episodes=int(co.support_count.eq(1).sum()),
        query_tokens_present_but_no_support=int(co.loc[co.support_count.eq(0),'query_available'].sum()),
        role_margin_means=dec[['target','other','background','delta_fixed_margin']].mean().to_dict(),
        note='Fractional cells describe pooled mask mass, not independent receptive fields. Pre/post margin decomposition is numeric, not causal physics.')
    (HERE/'diagnostics.json').write_text(json.dumps(stat,indent=2),encoding='utf-8')
    print(sm.to_string(index=False));print(cf.to_string(index=False));print(json.dumps(stat,indent=2))


if __name__=='__main__':main()
