"""Paired support variation, fixed-cohort budgets, and targeted margin evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
KEY=['fold','budget','draw','row']


def main():
    p=pd.read_csv(HERE/'predictions.csv');m=pd.read_csv(HERE/'metrics.csv')
    meta=pd.read_csv(HERE/'cohort.csv');records=[];pairs=[];draws=[]
    for candidate in ['prepool','target_only']:
        a=p[p.arm.eq(candidate)].set_index(KEY)
        for baseline in ['global_grid','global_only','grid12','postpool','strength_control']:
            b=p[p.arm.eq(baseline)].set_index(KEY)
            d=a[['truth','prediction','correct','margin','brier','ap','same_class_support','target_polarizations']].join(
                b[['prediction','correct','margin','brier','ap']],rsuffix='_base')
            d['rescue']=d.correct&~d.correct_base;d['harm']=~d.correct&d.correct_base
            d['candidate']=candidate;d['baseline']=baseline
            for col in ['margin','brier','ap']:d['delta_'+col]=d[col]-d[col+'_base']
            d=d.reset_index();pairs.append(d)
            am=m[m.arm.eq(candidate)].set_index(['fold','budget','draw'])
            bm=m[m.arm.eq(baseline)].set_index(['fold','budget','draw'])
            change=am[['oa','ba','brier','map_r']]-bm[['oa','ba','brier','map_r']]
            mean_draw=change.groupby(['budget','draw']).mean().reset_index()
            mean_draw['candidate']=candidate;mean_draw['baseline']=baseline;draws.append(mean_draw)
            for budget,g in d.groupby('budget'):
                v=mean_draw[mean_draw.budget.eq(budget)]
                records.append(dict(candidate=candidate,baseline=baseline,budget=int(budget),
                    query_events=len(g),unique_queries=g.row.nunique(),folds=g.fold.nunique(),
                    oa_pp=float(v.oa.mean()*100),oa_std_pp=float(v.oa.std()*100),
                    ba_pp=float(v.ba.mean()*100),ba_std_pp=float(v.ba.std()*100),
                    oa_draw_wins=int(v.oa.gt(1e-12).sum()),oa_draw_losses=int(v.oa.lt(-1e-12).sum()),
                    ba_draw_wins=int(v.ba.gt(1e-12).sum()),ba_draw_losses=int(v.ba.lt(-1e-12).sum()),
                    rescues=int(g.rescue.sum()),harms=int(g.harm.sum()),
                    rescued_ships=meta.iloc[g.loc[g.rescue,'row']].mmsi.nunique(),
                    harmed_ships=meta.iloc[g.loc[g.harm,'row']].mmsi.nunique(),
                    brier_delta=float(v.brier.mean()),map_delta=float(v.map_r.mean())))
    table=pd.DataFrame(records);pair=pd.concat(pairs);draw=pd.concat(draws)
    table.to_csv(HERE/'paired_summary.csv',index=False)
    pair.to_csv(HERE/'paired_events.csv',index=False);draw.to_csv(HERE/'paired_draws.csv',index=False)
    # Budget curves only on identical query dates; the headline B summaries differ in scope.
    fixed=[]
    scopes={'B16_B32':['20250315','20250401'],'B16_B32_B64':['20250315']}
    for scope,folds in scopes.items():
        subset=m[m.fold.astype(str).isin(folds)]
        for (budget,arm),g in subset.groupby(['budget','arm']):
            if scope=='B16_B32' and budget==64:continue
            fixed.append(dict(scope=scope,budget=int(budget),arm=arm,folds=len(folds),
                              **g[['oa','ba','brier','map_r']].mean().to_dict()))
    pd.DataFrame(fixed).to_csv(HERE/'fixed_query_budget_comparison.csv',index=False)
    d=pair[(pair.candidate=='target_only')&(pair.baseline=='global_grid')].copy()
    d=d.merge(meta[['row','mmsi','class_raw','code','date']],on='row')
    d.groupby(['budget','fold','class_raw']).agg(query_events=('row','size'),queries=('row','nunique'),
        rescues=('rescue','sum'),harms=('harm','sum'),margin_change=('delta_margin','mean'),
        brier_change=('delta_brier','mean'),ap_change=('delta_ap','mean')).to_csv(HERE/'class_flip_breakdown.csv')
    d['coverage']=np.where(d.same_class_support.eq(0),'missing',np.where(d.same_class_support.le(2),'one_or_two','three_or_more'))
    d.groupby(['budget','coverage']).agg(events=('row','size'),rescues=('rescue','sum'),harms=('harm','sum'),
        margin_change=('delta_margin','mean'),brier_change=('delta_brier','mean')).to_csv(HERE/'support_coverage_strata.csv')
    d.groupby(['budget','target_polarizations']).agg(events=('row','size'),rescues=('rescue','sum'),harms=('harm','sum'),
        margin_change=('delta_margin','mean'),brier_change=('delta_brier','mean')).to_csv(HERE/'mask_strata.csv')
    d[d.rescue|d.harm].to_csv(HERE/'flips.csv',index=False)
    decomposition=pd.read_csv(HERE/'margin_decomposition.csv')
    decomposition.groupby(['budget','arm'])[['direct_kernel','support_refit','r50_fixed_margin_change']].mean().to_csv(HERE/'decomposition_summary.csv')
    report=dict(final_pre_target_same_predictions=bool(np.array_equal(
        p[p.arm.eq('prepool')].set_index(KEY).sort_index().prediction,
        p[p.arm.eq('target_only')].set_index(KEY).sort_index().prediction)),
        source_prior_fit_unchanged_across_readouts=True,query_labels_used_for_selection=False,
        interpretation='Conditional budget scopes, not monotonic budget evidence; case-selected boundaries exploratory.')
    (HERE/'analysis_receipt.json').write_text(json.dumps(report,indent=2))
    print(table[(table.candidate=='target_only')&table.baseline.isin(['global_grid','grid12','postpool'])].to_string(index=False))
    print(d[d.rescue|d.harm][['fold','budget','draw','row','mmsi','class_raw','rescue','harm','same_class_support','delta_margin']].to_string(index=False))


if __name__=='__main__':main()
