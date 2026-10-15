"""Summarize scope and descriptive frozen-weight curves without target selection."""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import OUT, OLD, source_metadata, write


def main():
    protocol = json.loads((OUT/'PROTOCOL.json').read_text(encoding='utf-8'))
    protocol['memory_identity'] = 'Known consistent-label source ships only; exclude unknown identities rather than treat them as one ship'
    protocol['selection_vs_deployment'] = ('Held source ports excluded during weight-selection fitting; deployment reuses full allowed source including validation ports. '
        'This is source-selected hyperparameter transfer, not a permanently held-source model. No external labels used for selection.')
    write(OUT/'PROTOCOL.json', protocol)
    d = source_metadata(); fit = np.load(OUT/'validation_fit_rows.npy')
    source = np.load(OLD/'source_rows.npy')
    counts = {}
    for name in ['validation', 'deployment']:
        with (OUT/f'{name}_memory.pkl').open('rb') as h: bank = pickle.load(h)
        rows = bank['source_rows']; m = d.iloc[rows]
        assert m.known_ship.all() and m.identity.nunique() == len(m)
        counts[name] = dict(reference_ships=len(m), reference_classes=np.bincount(m.class_id, minlength=8).tolist(), tau2=bank['tau2'])
    assert d.iloc[fit].known_ship.all()
    counts['deployment_fit'] = dict(rows=len(source), unknown_identity_rows=int((~d.iloc[source].known_ship).sum()))
    write(OUT/'MEMORY_SCOPE.json', counts)
    oldscope = json.loads((OLD/'SCOPE.json').read_text(encoding='utf-8'))
    write(OUT/'SCOPE.json', [s for s in oldscope if s['study'] != 'support_exchange'])
    vals = pd.read_csv(OUT/'source_grid_metrics.csv')
    selections = json.loads((OUT/'SELECTED_WEIGHTS.json').read_text(encoding='utf-8'))
    selected_check = []
    for (budget, arm), group in vals.groupby(['budget', 'arm']):
        means = group.groupby(['port', 'weight']).ba.mean().unstack('weight').mean(axis=0)
        best = sorted(means.index, key=lambda w: (-means[w], abs(w-.5), w))[0]
        assert float(best) == selections[f'{budget}:{arm}']['weight']
        selected_check.append(f'{budget}:{arm}')
    curves = pd.read_csv(OUT/'target_grid_summary.csv')
    ef = pd.read_csv(OUT/'effect_summary.csv')
    ev = pd.read_csv(OUT/'changed_cases.csv', dtype={'ship': str})
    ev.groupby(['study', 'dataset', 'budget', 'cell', 'comparison', 'class_id', 'outcome']).agg(
        events=('row', 'size'), observations=('row', 'nunique'), ships=('ship', 'nunique')).to_csv(OUT/'event_scope.csv')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'original': '#267e79', 'class_mean': '#b04e65', 'shrink_control': '#777777'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), constrained_layout=True)
    for row, b in enumerate([16, 64]):
        for col, dataset in enumerate(['f0921', 'oss2']):
            ax = axes[row, col]
            for arm in colors:
                sub = curves[(curves.study == 'dataset_adaptation')&(curves.dataset == dataset)&(curves.budget == b)&(curves.arm == arm)]
                ax.plot(sub.weight, sub.ba, color=colors[arm], label=arm.replace('_', ' '))
                w = selections[f'{b}:{arm}']['weight']
                ax.scatter([w], sub[sub.weight == w].ba, color=colors[arm], s=48, zorder=3)
            ax.axvline(.5, color='#aaa', linewidth=1, linestyle=':')
            ax.set(title=f'{dataset.upper()}, B={b}', xlabel='Expert probability weight', ylabel='BA (%)')
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Global tasks: dots are weights selected on SOURCE validation', fontsize=14)
    fig.savefig(OUT/'fusion_weight_curves.png', dpi=160)
    fig.savefig(OUT/'fusion_weight_curves.svg'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for ax, metric in zip(axes, ['delta_oa', 'delta_ba']):
        rows = []
        for ds in ['f0921', 'oss2']:
            for study, cell in [('dataset_adaptation', 'target_support'), ('acquisition_transfer', 'other_acquisition')]:
                rows.append(ef[(ef.dataset == ds)&(ef.study == study)&(ef.budget == 64)&(ef.cell == cell)].set_index('comparison'))
        pos = np.arange(4)
        for offset, comparison, color, label in [(-.18, 'equal_gain', '#267e79', 'Both weight .5'), (.18, 'selected_gain', '#b04e65', 'Each source-selected')]:
            ax.bar(pos+offset, [r.loc[comparison, metric] for r in rows], width=.35, color=color, label=label)
        ax.axhline(0, color='#333', linewidth=1)
        ax.set_xticks(pos, ['F0921\nglobal', 'F0921\ncross-acq', 'OSS2\nglobal', 'OSS2\ncross-acq'])
        ax.set_ylabel(metric.removeprefix('delta_').upper()+' change (percentage points)')
        ax.set_title('Class reference minus original, B=64')
    axes[0].legend(fontsize=9)
    fig.savefig(OUT/'external_effects.png', dpi=160)
    fig.savefig(OUT/'external_effects.svg'); plt.close(fig)
    scope = {'date': '2026-10-14', 'status': 'computed_and_analyzed', 'selected_weight_checks': selected_check,
        'source': json.loads((OUT/'SOURCE_COMPLETE.json').read_text(encoding='utf-8')),
        'external': json.loads((OUT/'EXTERNAL_COMPLETE.json').read_text(encoding='utf-8')),
        'analysis': json.loads((OUT/'ANALYSIS_COMPLETE.json').read_text(encoding='utf-8')),
        'main_model_changed': False, 'next_experiment_started': False}
    write(OUT/'COMPUTATION_COMPLETE.json', scope)
    print('SCOPE', counts, flush=True)
    print('SOURCE_WEIGHT_CHECKS', selected_check, flush=True)


if __name__ == '__main__':
    main()
