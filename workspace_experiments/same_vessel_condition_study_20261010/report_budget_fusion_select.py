"""Bounded post-freeze accounting for selection-objective alignment."""

from run_budget_fusion_select import OUT, PORTS, BUDGETS, POLICIES, CANDIDATES, cached
from full_teacher_data import BASE, write
from evaluate_probability_layers import probability_rows
import json
import numpy as np
import pandas as pd


def main():
    choices = json.loads((OUT/'FROZEN_CHOICE.json').read_text())['choices']
    sel = pd.read_csv(OUT/'source_selection.csv')
    selected_rows = []
    for budget in BUDGETS:
        subset = sel[sel.budget == budget]
        original = subset[subset.arm == 'raw'].set_index(['context', 'fold', 'draw'])
        for policy in POLICIES:
            choice = choices[str(budget)][policy]
            matched = subset[(subset.arm == choice['arm']) & (subset.intervention == choice['intervention'])]
            for row in matched.itertuples():
                raw = original.loc[(row.context, row.fold, row.draw)]
                selected_rows.append(dict(budget=budget, policy=policy, context=row.context, fold=row.fold,
                    draw=row.draw, selected_arm=choice['arm'], intervention=choice['intervention'],
                    source_expert_delta_ba=row.expert_ba-raw.expert_ba,
                    source_fusion_delta_ba=row.fusion_ba-raw.fusion_ba))
    selected_rows = pd.DataFrame(selected_rows)
    selected_rows.to_csv(OUT/'selected_source_episodes.csv', index=False)
    selected_rows.groupby(['budget', 'policy', 'fold'])[
        ['source_expert_delta_ba', 'source_fusion_delta_ba']].mean().reset_index().to_csv(
        OUT/'selected_source_folds.csv', index=False)

    # Diagnose potential task-loss alignment on source only, without selecting another model.
    meta = pd.read_csv(BASE/'observations.csv', usecols=['class_id'])
    risk = []
    for context in PORTS:
        for fold in range(3):
            for draw in range(3):
                for budget in BUDGETS:
                    z = cached(OUT/'selection'/context/f'{fold}_{draw}_{budget}.npz')
                    y = meta.iloc[z['query_ids']].class_id.to_numpy()
                    for i, (arm, t) in enumerate(CANDIDATES):
                        p = probability_rows(z['fusion'][i], y, 'probabilities')
                        risk.append(dict(context=context, fold=fold, draw=draw, budget=budget,
                            arm=arm, intervention=t, row_nll=p['nll'].mean(), row_brier=p['brier'].mean(),
                            class_nll=np.mean([p['nll'][y == c].mean() for c in np.unique(y)]),
                            class_brier=np.mean([p['brier'][y == c].mean() for c in np.unique(y)])))
    pd.DataFrame(risk).to_csv(OUT/'source_probability_diagnostic.csv', index=False)

    frame = pd.read_csv(OUT/'metrics.csv')
    avg = pd.read_csv(OUT/'budget_summary.csv')
    compact = avg[avg.arm.isin(['fusion_raw', 'fusion_expert_selected', 'fusion_fusion_selected',
                              'fusion_repair_fusion_selected', 'fusion_simple_fusion_selected'])]
    compact.to_csv(OUT/'compact_results.csv', index=False)
    comparisons = pd.read_csv(OUT/'selection_objective_effect.csv')
    cmp = comparisons.groupby(['family', 'budget', 'family_scope'])[
        ['delta_ba', 'delta_oa', 'portclass_risk_gain']].mean().reset_index()
    cmp.to_csv(OUT/'selection_objective_summary.csv', index=False)
    write(OUT/'DIAGNOSTIC.json', dict(source_only_probability_analysis=True, no_new_model_selected=True,
        comparisons=cmp.to_dict(orient='records'),
        boundaries='Shared historical source representations; held labels excluded from new readers, not all historical encoders'))
    print('SOURCE FULL-SYSTEM GAINS')
    print(selected_rows.groupby(['budget', 'policy']).source_fusion_delta_ba.mean().to_string())
    print('HELD/TARGET RESULTS')
    print(compact[['family', 'budget', 'arm', 'oa_mean', 'ba_mean', 'delta_ba_mean',
                   'portclass_risk_gain_mean']].to_string(index=False))
    print('CHANGE DUE TO SELECTION OBJECTIVE')
    print(cmp.to_string(index=False))


if __name__ == '__main__':
    main()
