"""Recompute outcomes, paired subgroup effects, and source coverage changes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
KEYS = ['fold','seed','origin','draw']
PKEYS = KEYS+['global_row']
MAIN = ['baseline','access','margin','joint']


def independent_scores(group):
    correct = group.prediction.eq(group.truth)
    p = group[[f'p{k}' for k in range(8)]].to_numpy()
    y = group.truth.to_numpy(int)
    brier = ((p-np.eye(8)[y])**2).sum(1)
    return dict(oa=100*correct.mean(), ba=100*correct.groupby(group.truth).mean().mean(),
        brier=brier.mean(), macro_brier=pd.Series(brier,index=group.index).groupby(group.truth).mean().mean(),
        nll=-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean(),
        query=len(group), classes=group.truth.nunique())


def grouped_scores(p, keys):
    rows = []
    for values, group in p.groupby(keys):
        rows.append(dict(zip(keys,values), **independent_scores(group)))
    return pd.DataFrame(rows)


def main():
    metrics = pd.read_csv(HERE/'metrics.csv')
    p = pd.read_csv(HERE/'predictions.csv')
    assert not p.duplicated(['arm']+PKEYS).any()
    calculated = grouped_scores(p,['arm']+KEYS).set_index(['arm']+KEYS)
    original = metrics.set_index(['arm']+KEYS).loc[calculated.index]
    columns = ['oa','ba','brier','macro_brier','nll']
    errors = abs(calculated[columns]-original[columns]).max()
    assert float(errors.max()) < 1e-10
    errors.rename('max_error').to_csv(HERE/'metric_replay.csv')
    calculated.reset_index().to_csv(HERE/'independent_metrics.csv',index=False)
    score_cols = ['oa','ba','delta_oa','delta_ba','brier','macro_brier','nll','kernel_risk_gain','support_map_r']
    for name, keys in [('summary',['origin','arm']), ('fold_summary',['origin','arm','fold']),
                       ('seed_summary',['origin','arm','seed']), ('draw_summary',['origin','arm','draw'])]:
        metrics.groupby(keys)[score_cols].mean().to_csv(HERE/(name+'.csv'))
    baseline = metrics[metrics.arm.eq('baseline')].set_index(KEYS)
    effects = []
    for arm in ['access','margin','joint','initial','initial_access']:
        current = metrics[metrics.arm.eq(arm)].set_index(KEYS)
        delta = current[score_cols]-baseline.loc[current.index,score_cols]
        delta = delta.rename(columns={k:'change_'+k for k in score_cols})
        delta['arm'] = arm
        effects.append(delta.reset_index())
    effects = pd.concat(effects,ignore_index=True)
    effects.to_csv(HERE/'paired_effects.csv',index=False)
    effects.groupby(['origin','arm']).mean(numeric_only=True).to_csv(HERE/'effect_summary.csv')
    seed_effects = effects.groupby(['origin','arm','seed'])[['change_oa','change_ba','change_macro_brier','change_support_map_r']].mean()
    seed_effects.to_csv(HERE/'effect_seeds.csv')
    seed_effects.groupby(['origin','arm']).agg(['mean','std']).to_csv(HERE/'reader_seed_uncertainty.csv')

    reference = p[p.arm.eq('baseline')].set_index(PKEYS)
    lookup = pd.MultiIndex.from_frame(p[PKEYS])
    p['reference_prediction'] = reference.loc[lookup,'prediction'].to_numpy()
    p['change'] = p.prediction.eq(p.truth).astype(int)-p.reference_prediction.eq(p.truth).astype(int)
    p['true_probability_change'] = p.true_probability.to_numpy()-reference.loc[lookup,'true_probability'].to_numpy()
    cohort = reference.reset_index().drop_duplicates(['fold','global_row'])
    class_counts = cohort.groupby(['fold','truth']).size().rename('class_query_count').reset_index()
    fold_counts = cohort.groupby('fold').agg(fold_queries=('global_row','size'),fold_classes=('truth','nunique'))
    p = p.merge(class_counts,on=['fold','truth'],validate='many_to_one').join(fold_counts,on='fold')
    contexts = len(metrics[metrics.arm.eq('baseline')&metrics.origin.eq('cross')])
    p['oa_contribution_pp'] = 100*p.change/(contexts*p.fold_queries)
    p['ba_contribution_pp'] = 100*p.change/(contexts*p.fold_classes*p.class_query_count)
    p.to_csv(HERE/'paired_predictions.csv',index=False)
    p[p.change.ne(0)].to_csv(HERE/'changed_queries.csv',index=False)
    events = p.groupby(['origin','arm','axis_valid']).agg(
        rescue_events=('change',lambda s:(s>0).sum()),harm_events=('change',lambda s:(s<0).sum()),
        global_delta_oa=('oa_contribution_pp','sum'),global_delta_ba=('ba_contribution_pp','sum'))
    events.to_csv(HERE/'subgroup_contributions.csv')
    # Group deltas use the original full-cohort denominator; therefore these sum.
    summed = events.groupby(['origin','arm'])[['global_delta_oa','global_delta_ba']].sum()
    for (origin,arm),row in summed.iterrows():
        if arm=='baseline':
            continue
        expected = effects[effects.origin.eq(origin)&effects.arm.eq(arm)][['change_oa','change_ba']].mean()
        np.testing.assert_allclose(row.to_numpy(),expected.to_numpy(),atol=1e-10)
    subgroup = grouped_scores(p,['arm']+KEYS+['axis_valid'])
    subgroup.to_csv(HERE/'subgroup_metrics.csv',index=False)
    subgroup.groupby(['origin','arm','axis_valid'])[['oa','ba','brier','macro_brier','nll','query','classes']].mean().to_csv(HERE/'subgroup_summary.csv')
    classes = p.groupby(['origin','arm','truth']).agg(
        correct=('prediction',lambda s:int((s==p.loc[s.index,'truth']).sum())),
        events=('global_row','size'),distinct_queries=('global_row','nunique'),
        rescue_events=('change',lambda s:(s>0).sum()),harm_events=('change',lambda s:(s<0).sum()),
        global_delta_oa=('oa_contribution_pp','sum'),global_delta_ba=('ba_contribution_pp','sum'))
    classes['recall'] = 100*classes.correct/classes.events
    classes.to_csv(HERE/'class_summary.csv')
    p.groupby(['origin','arm','fold','global_row','truth','axis_valid']).agg(
        rescue_events=('change',lambda s:(s>0).sum()),harm_events=('change',lambda s:(s<0).sum()),
        delta_oa=('oa_contribution_pp','sum'),delta_ba=('ba_contribution_pp','sum'),
        probability_change=('true_probability_change','mean')).to_csv(HERE/'query_contributions.csv')
    invalid_margin = p[p.arm.eq('margin')&~p.axis_valid]
    np.testing.assert_array_equal(invalid_margin.prediction,invalid_margin.reference_prediction)

    access = pd.read_csv(HERE/'source_access.csv')
    access.groupby(['kind','origin','truth'])[['axis_valid','common_valid','old_aux_eligible','new_aux_eligible','newly_aux_eligible']].agg(['count','sum']).to_csv(HERE/'source_access_contexts.csv')
    distinct = access.groupby(['kind','origin','global_row','truth'])[
        ['axis_valid','common_valid','old_aux_eligible','new_aux_eligible','newly_aux_eligible']].max().reset_index()
    distinct.groupby(['kind','origin','truth']).agg(
        distinct_queries=('global_row','nunique'),old_local=('axis_valid','sum'),
        common_local=('common_valid','sum'),old_aux=('old_aux_eligible','sum'),
        new_aux=('new_aux_eligible','sum'),new_in_some_context=('newly_aux_eligible','sum')).to_csv(HERE/'source_access_distinct.csv')
    histories = []
    for fold in metrics.fold.unique():
        for arm in MAIN:
            for seed in range(3):
                frame = pd.read_csv(HERE/fold/f'{arm}_{seed}'/'learning.csv')
                frame['fold'],frame['arm'],frame['seed'] = fold,arm,seed
                histories.append(frame)
    history = pd.concat(histories,ignore_index=True)
    history.to_csv(HERE/'learning_curves.csv',index=False)
    history.groupby(['arm','evaluation','step'])[['brier','retrieval_ce','margin_loss','oa','ba']].mean().to_csv(HERE/'learning_summary.csv')

    figures = HERE/'figures'
    figures.mkdir(exist_ok=True)
    fig,axes = plt.subplots(1,2,figsize=(10,3.8))
    names = ['Common access','Decision margin','Combined']
    for ax,column in zip(axes,['change_oa','change_ba']):
        for i,arm in enumerate(['access','margin','joint']):
            values = seed_effects.loc[('cross',arm),column].to_numpy()
            ax.scatter(i+np.linspace(-.09,.09,3),values,color='#167d8d',s=28)
            ax.plot([i-.2,i+.2],[values.mean()]*2,color='#b04940',lw=2)
        ax.axhline(0,color='.5',lw=1)
        ax.set_xticks(range(3),names)
        ax.set_ylabel(('OA' if column=='change_oa' else 'BA')+' change vs retained baseline (pp)')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('OSS1 frozen 4-fold cross-acquisition | dots: 3 reader seeds')
    fig.tight_layout()
    fig.savefig(figures/'paired_outcomes.png',dpi=170)
    plt.close(fig)
    details = dict(metric_rows=len(metrics),prediction_rows=len(p),unique_queries=int(p.global_row.nunique()),
        metric_max_error=float(errors.max()),source_contexts=len(access),
        old_invalid_queries=int(cohort[~cohort.axis_valid].global_row.nunique()),
        old_valid_queries=int(cohort[cohort.axis_valid].global_row.nunique()))
    (HERE/'verification.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(metrics.groupby(['origin','arm'])[['oa','ba','kernel_risk_gain']].mean().to_string())
    print(events.loc['cross'].to_string())
    print(details)


if __name__=='__main__':
    main()
