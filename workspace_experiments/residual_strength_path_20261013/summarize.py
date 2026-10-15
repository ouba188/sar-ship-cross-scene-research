"""Summarize the frozen path experiment without fitting or selecting again."""

from run_path import OUT, MODES, VALUES, read_json, np, pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    pair_rows, selected_rows = [], []
    for family in ('selection', 'source', 'external'):
        selected = pd.read_csv(OUT / (family + '_selected.csv'))
        all_rows = pd.read_csv(OUT / (family + '_metrics.csv'))
        keys = ['family', 'context', 'unit', 'budget']
        joint = selected[selected.policy == 'joint_selected']
        references = {'joint_full': all_rows[(all_rows['mode'] == 'joint') & (all_rows.alpha == 1.)]}
        references.update({name: selected[selected.policy == name] for name in
                           ('raw', 'global_selected', 'count')})
        for name, reference in references.items():
            paired = joint.merge(reference, on=keys, suffixes=('', '_ref'), validate='one_to_one')
            for _, r in paired.iterrows():
                pair_rows.append(dict(**{k: r[k] for k in keys}, reference=name,
                    delta_oa=r.oa-r.oa_ref, delta_ba=r.ba-r.ba_ref,
                    delta_macro_brier=r.macro_brier-r.macro_brier_ref))
        summary = selected.groupby(['family', 'budget', 'policy'])[list(VALUES)].mean().reset_index()
        selected_rows.append(summary)
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(OUT / 'paired.csv', index=False)
    summaries = []
    for (family, budget, ref), g in pair.groupby(['family', 'budget', 'reference']):
        delta = g.delta_ba.to_numpy()
        summaries.append(dict(family=family, budget=budget, reference=ref, units=len(g),
            delta_oa=g.delta_oa.mean(), delta_ba=delta.mean(),
            descriptive_sd_ba=delta.std(ddof=1), delta_macro_brier=g.delta_macro_brier.mean(),
            positive=int((delta > 1e-10).sum()), negative=int((delta < -1e-10).sum()),
            ties=int((abs(delta) <= 1e-10).sum()),
            leave_best_out_ba=(delta.sum()-delta.max())/(len(delta)-1)))
    pd.DataFrame(summaries).to_csv(OUT / 'paired_summary.csv', index=False)
    pd.concat(selected_rows).to_csv(OUT / 'selected_summary.csv', index=False)
    f = pd.read_csv(OUT / 'external_summary.csv')
    diagnostic = f[f['mode'].isin(MODES)].copy()
    total_rescue = diagnostic.endpoint_rescue_kept_ba + diagnostic.endpoint_rescue_lost_ba
    total_harm = diagnostic.endpoint_harm_kept_ba + diagnostic.endpoint_harm_avoided_ba
    diagnostic['endpoint_rescue_retained_fraction'] = diagnostic.endpoint_rescue_kept_ba / total_rescue
    diagnostic['endpoint_harm_avoided_fraction'] = diagnostic.endpoint_harm_avoided_ba / total_harm
    risk_gain = -diagnostic.delta_macro_brier
    diagnostic['unchanged_prediction_risk_share'] = diagnostic.risk_gain_unchanged / risk_gain.replace(0, np.nan)
    diagnostic.to_csv(OUT / 'mechanism_summary.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    choice = read_json(OUT / 'FROZEN_ALPHA.json')['alphas']
    for column, budget in enumerate((16, 128)):
        for mode, color in [('joint', '#197b86'), ('global', '#b25a3e')]:
            sub = f[(f.budget == budget) & (f['mode'] == mode)].sort_values('alpha')
            axes[0, column].plot(sub.alpha, sub.delta_ba, marker='o', color=color, label=mode)
            point = sub[sub.alpha == choice[mode]].iloc[0]
            axes[0, column].scatter([point.alpha], [point.delta_ba], s=130, facecolors='none', edgecolors=color)
        count = float(f[(f.budget == budget) & (f.arm == 'count')].delta_ba.iloc[0])
        axes[0, column].axhline(count, linestyle='--', color='#775695', label='count reference')
        sub = f[(f.budget == budget) & (f['mode'] == 'joint')].sort_values('alpha')
        for field, color, label in [('rescue_ba', '#197b86', 'repairs'), ('harm_ba', '#b25a3e', 'harms')]:
            axes[1, column].plot(sub.alpha, sub[field], marker='o', color=color, label=label)
        axes[1, column].axvline(choice['joint'], linestyle=':', color='#666666')
        axes[0, column].set_title(f'OpenSARShip 2.0: {budget} labels')
        for row in range(2):
            axes[row, column].set_xticks([0, .25, .5, .75, 1])
            axes[row, column].set_xlabel('Fraction of learned update')
            axes[row, column].grid(axis='y', alpha=.2)
            axes[row, column].spines[['top', 'right']].set_visible(False)
            axes[row, column].legend(frameon=False, fontsize=9)
        axes[0, column].axhline(0, color='#555555', linewidth=.7)
    axes[0, 0].set_ylabel('BA change from raw (pp)')
    axes[1, 0].set_ylabel('Class-balanced contribution (pp)')
    fig.savefig(OUT / 'strength_path.png', dpi=160)
    plt.close(fig)
    print(pd.DataFrame(summaries).query("family == 'external'").to_string(index=False))
    print(diagnostic[(diagnostic['mode'] == 'joint') & (diagnostic.alpha.isin([.25, 1.]))][
        ['budget', 'alpha', 'endpoint_rescue_retained_fraction', 'endpoint_harm_avoided_fraction',
         'novel_rescue_ba', 'novel_harm_ba', 'unchanged_prediction_risk_share']].to_string(index=False))


if __name__ == '__main__':
    main()
