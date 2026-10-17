"""Independent outcomes, paired training controls and per-query accounting."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('previous_summary',HERE.parent/'axis_preserving_addition_20261017'/'summarize.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
KEYS = previous.KEYS
PKEYS = previous.PKEYS
VALUES = previous.VALUES


def main():
    m,p = pd.read_csv(HERE/'metrics.csv'),pd.read_csv(HERE/'predictions.csv')
    assert not p.duplicated(['arm']+PKEYS).any()
    calculated = previous.recompute(p).set_index(['arm']+KEYS)
    original = m.set_index(['arm']+KEYS).loc[calculated.index]
    error = float(abs(calculated[VALUES]-original[VALUES]).to_numpy().max())
    assert error<1e-10
    calculated.to_csv(HERE/'independent_metrics.csv')
    measures = VALUES+['support_map_r']
    m.groupby(['origin','arm'])[measures].mean().to_csv(HERE/'summary.csv')
    effects = []
    for reference_arm in ['margin','initial_common','axis_continue']:
        ref = m[m.arm.eq(reference_arm)].set_index(KEYS)
        for arm in m.arm.unique():
            current = m[m.arm.eq(arm)].set_index(KEYS)
            delta = current[measures]-ref.loc[current.index,measures]
            delta = delta.rename(columns={c:'delta_'+c for c in measures})
            delta['arm'],delta['reference'] = arm,reference_arm
            effects.append(delta.reset_index())
    effects = pd.concat(effects,ignore_index=True)
    effects.to_csv(HERE/'paired_effects.csv',index=False)
    delta_columns = ['delta_'+c for c in measures]
    for name,keys in [('effect_summary',['origin','arm','reference']),('effect_seeds',['origin','arm','reference','seed']),
                      ('effect_folds',['origin','arm','reference','fold']),('effect_draws',['origin','arm','reference','draw'])]:
        effects.groupby(keys)[delta_columns].mean().to_csv(HERE/(name+'.csv'))
    effects.groupby(['origin','arm','reference','seed'])[delta_columns].mean().groupby(['origin','arm','reference']).agg(['mean','std']).to_csv(HERE/'reader_seed_uncertainty.csv')
    reference = p[p.arm.eq('margin')].set_index(PKEYS)
    matched = reference.loc[pd.MultiIndex.from_frame(p[PKEYS])]
    columns = [f'p{k}' for k in range(8)]
    before,after = matched[columns].to_numpy(),p[columns].to_numpy()
    y = p.truth.to_numpy(int)
    p['reference_prediction'] = matched.prediction.to_numpy()
    p['change'] = p.prediction.eq(p.truth).astype(int)-p.reference_prediction.eq(p.truth).astype(int)
    p['max_probability_change'] = abs(after-before).max(1)
    p['true_probability_change'] = (after-before)[np.arange(len(p)),y]
    p['brier_gain'] = ((before-np.eye(8)[y])**2).sum(1)-((after-np.eye(8)[y])**2).sum(1)
    cohort = reference.reset_index().drop_duplicates(['fold','global_row'])
    class_counts = cohort.groupby(['fold','truth']).size().rename('class_queries')
    fold_counts = cohort.groupby('fold').agg(fold_queries=('global_row','size'),fold_classes=('truth','nunique'))
    p = p.join(class_counts,on=['fold','truth']).join(fold_counts,on='fold')
    ncontexts = m[m.arm.eq('margin')].groupby('origin').size()
    p['oa_contribution_pp'] = 100*p.change/(p.origin.map(ncontexts)*p.fold_queries)
    p['ba_contribution_pp'] = 100*p.change/(p.origin.map(ncontexts)*p.fold_classes*p.class_queries)
    p.to_csv(HERE/'paired_predictions.csv',index=False)
    for name,keys in [('subgroup_contributions',['origin','arm','axis_valid']),('class_contributions',['origin','arm','truth']),
                      ('query_contributions',['origin','arm','global_row','truth'])]:
        p.groupby(keys).agg(rescues=('change',lambda s:int((s>0).sum())),harms=('change',lambda s:int((s<0).sum())),
            delta_oa=('oa_contribution_pp','sum'),delta_ba=('ba_contribution_pp','sum'),
            contexts=('global_row','size'),unique_queries=('global_row','nunique'),
            brier_gain=('brier_gain','mean'),max_probability_change=('max_probability_change','mean')).to_csv(HERE/(name+'.csv'))
    sums = p.groupby(['origin','arm'])[['oa_contribution_pp','ba_contribution_pp']].sum()
    actual = effects[effects.reference.eq('margin')].groupby(['origin','arm'])[['delta_oa','delta_ba']].mean().loc[sums.index]
    np.testing.assert_allclose(sums.to_numpy(),actual.to_numpy(),atol=1e-10)
    changed = p[p.change.ne(0)].copy()
    geometry = pd.read_csv(HERE.parent/'oss1_joint_geometry_20261017'/'geometry.csv',dtype={'mmsi':str})
    changed.merge(geometry,on='global_row',validate='many_to_one',suffixes=('','_metadata')).to_csv(HERE/'changed_native_cases.csv',index=False)
    histories = []
    for fold in m.fold.unique():
        for seed in range(3):
            for arm in ['frozen_common','axis_continue']:
                frame = pd.read_csv(HERE/fold/f'{arm}_{seed}'/'learning.csv')
                frame['fold'],frame['seed'],frame['arm'] = fold,seed,arm
                histories.append(frame)
    history = pd.concat(histories,ignore_index=True)
    history.to_csv(HERE/'learning_curves.csv',index=False)
    history.groupby(['arm','evaluation','step'])[['brier','retrieval_ce','margin_loss','oa','ba']].mean().to_csv(HERE/'learning_summary.csv')
    pd.read_csv(HERE/'cached_latency.csv').groupby(['fold','arm']).seconds.agg(['median','mean']).to_csv(HERE/'latency_summary.csv')
    checks = pd.read_csv(HERE/'freeze_and_replay_checks.csv')
    details = dict(metric_rows=len(m),prediction_rows=len(p),unique_queries=int(p.global_row.nunique()),
                   metric_max_error=error,old_probability_max_error=float(checks.probability_replay_max_error.max()),
                   frozen_state_max_error=float(checks.old_state_max_error.max()),
                   frozen_readout_max_error=float(checks.old_readout_max_error.max()),
                   new_fits=int(pd.read_csv(HERE/'timings.csv').trained.sum()))
    (HERE/'verification.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(m.groupby(['origin','arm'])[measures].mean().to_string())
    print(effects[effects.arm.eq('frozen_common')].groupby(['origin','reference'])[['delta_oa','delta_ba','delta_macro_brier']].mean().to_string())
    print(p[p.arm.eq('frozen_common')&p.origin.eq('cross')].groupby('axis_valid')[['change','oa_contribution_pp','ba_contribution_pp']].sum().to_string())
    print(details)


if __name__=='__main__':
    main()
