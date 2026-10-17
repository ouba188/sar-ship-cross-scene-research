"""Paired factorial effects, learning curves and support-context exposure."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
KEYS = ['fold','seed','origin','draw']
ARMS = ['initial','fixed_aux','fixed_task','diverse_aux','diverse_task']


def main():
    metrics = pd.read_csv(HERE/'metrics.csv')
    columns = ['oa','ba','delta_oa','delta_ba','kernel_risk_gain','delta_brier','delta_nll','support_map_r']
    metrics.groupby(['origin','arm'])[columns].mean().to_csv(HERE/'summary.csv')
    metrics.groupby(['origin','arm','fold'])[columns].mean().to_csv(HERE/'fold_summary.csv')
    seeds = metrics.groupby(['origin','arm','seed'])[columns].mean()
    seeds.to_csv(HERE/'seed_summary.csv')
    seeds.groupby(['origin','arm']).agg(['mean','std']).to_csv(HERE/'seed_uncertainty.csv')
    metrics.groupby(['origin','arm','draw'])[columns].mean().to_csv(HERE/'support_draw_summary.csv')
    pairs = [('remove_aux_fixed','fixed_task','fixed_aux'),
             ('diversify_with_aux','diverse_aux','fixed_aux'),
             ('diversify_task_only','diverse_task','fixed_task'),
             ('remove_aux_diverse','diverse_task','diverse_aux'),
             ('combined_vs_previous','diverse_task','fixed_aux')]
    pairs += [(name+'_vs_initial',name,'initial') for name in ARMS[1:]]
    effects = []
    for name, new, old in pairs:
        a = metrics[metrics.arm.eq(new)].set_index(KEYS)
        b = metrics[metrics.arm.eq(old)].set_index(KEYS)
        np.testing.assert_array_equal(a.index.to_numpy(), b.index.to_numpy())
        result = a[['oa','ba','kernel_risk_gain','support_map_r']]-b[['oa','ba','kernel_risk_gain','support_map_r']]
        result = result.rename(columns={'oa':'delta_oa','ba':'delta_ba','support_map_r':'delta_map_r'})
        result['comparison'] = name
        effects.append(result.reset_index())
    effects = pd.concat(effects, ignore_index=True)
    effects.to_csv(HERE/'paired_effects.csv', index=False)
    effect_columns = ['delta_oa','delta_ba','kernel_risk_gain','delta_map_r']
    effects.groupby(['origin','comparison'])[effect_columns].mean().to_csv(HERE/'effect_summary.csv')
    effects.groupby(['origin','comparison','seed'])[effect_columns].mean().to_csv(HERE/'effect_seeds.csv')
    intervals = []
    for (origin, comparison), group in effects.groupby(['origin','comparison']):
        fold = group.groupby('fold')[effect_columns].mean()
        rng = np.random.default_rng(20261017)
        boot = fold.to_numpy()[rng.integers(0,len(fold),(5000,len(fold)))].mean(1)
        for i, col in enumerate(effect_columns):
            intervals.append(dict(origin=origin,comparison=comparison,metric=col,mean=fold[col].mean(),
                low=np.quantile(boot[:,i],.025),high=np.quantile(boot[:,i],.975),clusters=len(fold)))
    pd.DataFrame(intervals).to_csv(HERE/'exploratory_fold_intervals.csv',index=False)

    histories, exposures = [], []
    for fold in metrics.fold.unique():
        banks = json.loads((HERE/fold/'support_bank.json').read_text(encoding='utf-8'))
        members = {(row['episode'],row['draw']):tuple(sorted(row['support_global_rows']))
                   for row in banks if row['kind']=='fit'}
        for arm in ARMS[1:]:
            for seed in range(3):
                folder = HERE/fold/f'{arm}_{seed}'
                h = pd.read_csv(folder/'learning.csv')
                h['fold'],h['arm'],h['seed'] = fold,arm,seed
                histories.append(h)
                schedule = np.load(folder/'schedule.npz')
                uses = pd.DataFrame({'episode':schedule['episode'].ravel(),'draw':schedule['draw'].ravel()})
                for ep, rows in uses.groupby('episode'):
                    count = len({members[(int(ep),int(draw))] for draw in rows.draw})
                    exposures.append(dict(fold=fold,arm=arm,seed=seed,episode=ep,visits=len(rows),unique_support_sets=count))
    h = pd.concat(histories,ignore_index=True)
    start = h[h.step.eq(0)].set_index(['fold','arm','seed','evaluation'])
    for name in ('risk','retrieval_ce','oa','ba'):
        ref = pd.MultiIndex.from_frame(h[['fold','arm','seed','evaluation']])
        h[name+'_change_from_initial'] = h[name].to_numpy()-start.loc[ref,name].to_numpy()
    h.to_csv(HERE/'learning_curves.csv',index=False)
    h.groupby(['arm','evaluation','step']).mean(numeric_only=True).to_csv(HERE/'learning_summary.csv')
    pd.DataFrame(exposures).to_csv(HERE/'training_support_exposure.csv',index=False)

    p = pd.read_csv(HERE/'predictions.csv')
    query_keys = KEYS+['global_row']
    initial = p[p.arm.eq('initial')].set_index(query_keys)
    lookup = pd.MultiIndex.from_frame(p[query_keys])
    p['initial_prediction'] = initial.loc[lookup,'prediction'].to_numpy()
    p['net_vs_old'] = p.prediction.eq(p.truth).astype(int)-p.baseline.eq(p.truth).astype(int)
    p['net_vs_initial'] = p.prediction.eq(p.truth).astype(int)-p.initial_prediction.eq(p.truth).astype(int)
    p['true_probability_gain_vs_old'] = p.true_probability-p.baseline_true_probability
    p['true_probability_gain_vs_initial'] = p.true_probability.to_numpy()-initial.loc[lookup,'true_probability'].to_numpy()
    probability_columns = [f'p{k}' for k in range(8)]
    onehot = np.eye(8)[p.truth]
    p['brier_gain_vs_initial'] = ((initial.loc[lookup,probability_columns].to_numpy()-onehot)**2).sum(1)-((p[probability_columns].to_numpy()-onehot)**2).sum(1)
    p.to_csv(HERE/'prediction_diagnosis.csv',index=False)
    p.groupby(['origin','arm','truth'])[['net_vs_old','net_vs_initial','true_probability_gain_vs_old',
            'true_probability_gain_vs_initial','brier_gain_vs_initial']].mean().to_csv(HERE/'class_summary.csv')
    p[(p.net_vs_old!=0)|(p.net_vs_initial!=0)].to_csv(HERE/'changed_queries.csv',index=False)

    figure = HERE/'figures'
    figure.mkdir(exist_ok=True)
    names = ['Initial','Fixed + aux','Fixed task only','Diverse + aux','Diverse task only']
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for ax,col in zip(axes,['delta_oa','delta_ba']):
        for i,arm in enumerate(ARMS):
            values = seeds.loc[('cross',arm),col]
            ax.scatter(i+np.linspace(-.12,.12,len(values)),values,color='#087e8b',s=25)
            ax.plot([i-.2,i+.2],[values.mean()]*2,color='#b2403b',lw=2)
        ax.set_xticks(range(5),names,rotation=25)
        ax.set_ylabel(col+' vs old model (percentage points)')
        ax.axhline(0,color='.5',lw=1)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same frozen 4-fold target | dots: reader seeds; line: mean')
    fig.tight_layout()
    fig.savefig(figure/'target_seeds.png',dpi=160)
    plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(12,3.7))
    palette = ['#087e8b','#b2403b','#7161a8','#5c8f3c']
    for ax,evaluation in zip(axes,['fit_fixed','selection_fixed','selection_diverse']):
        for arm,color in zip(ARMS[1:],palette):
            values = h[h.arm.eq(arm)&h.evaluation.eq(evaluation)].groupby('step').risk_change_from_initial.mean()
            ax.plot(values.index,-values.values,marker='o',label=arm,color=color)
        ax.axhline(0,color='.5',lw=1)
        ax.set_title(evaluation,fontsize=11)
        ax.set_xlabel('Training step')
        ax.set_ylabel('Macro Brier improvement vs initialization')
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure/'source_learning.png',dpi=160)
    plt.close(fig)
    print(pd.read_csv(HERE/'summary.csv').to_string(index=False))
    print(pd.read_csv(HERE/'effect_summary.csv').query('origin == "cross"').to_string(index=False))


if __name__ == '__main__':
    main()
