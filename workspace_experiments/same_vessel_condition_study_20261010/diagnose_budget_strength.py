"""Explain source-selected strengths without any target strength search."""

from full_teacher_data import SOURCE, PORTS, write
from run_support_budget_strength import OUT, BUDGETS, cached
import numpy as np
import pandas as pd


def main():
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id'])
    transitions = []
    for context in PORTS:
        for budget in BUDGETS:
            for draw in range(5):
                z = cached(OUT/'target'/context/f'{draw}_{budget}.npz')
                y = manifest.iloc[z['query_ids']].class_id.to_numpy()
                classes, counts = np.unique(y, return_counts=True)
                weights = np.array([100/(len(classes)*counts[np.flatnonzero(classes == c)[0]]) for c in y])
                for arm in ['corrected', 'simple', 'fixed_corrected']:
                    e0, e1 = z['expert_raw'].argmax(1), z['expert_'+arm].argmax(1)
                    f0, f1 = z['fusion_raw'].argmax(1), z['fusion_'+arm].argmax(1)
                    groups = {'expert': (e0, e1), 'fusion': (f0, f1)}
                    for layer, (old, new) in groups.items():
                        total = 0.
                        for changed in [False, True]:
                            category = (e0 != e1) == changed
                            for event, mask, sign in [('rescue', (old != y) & (new == y), 1),
                                                      ('harm', (old == y) & (new != y), -1)]:
                                keep = category & mask
                                contribution = float(sign*weights[keep].sum())
                                total += contribution
                                transitions.append(dict(context=context, budget=budget, draw=draw, arm=arm,
                                    layer=layer, expert_argmax_changed=changed, event=event,
                                    n=int(keep.sum()), ba_contribution=contribution,
                                    oa_contribution=float(sign*100*keep.mean())))
                        actual = weights[(new == y)].sum()-weights[(old == y)].sum()
                        np.testing.assert_allclose(total, actual, atol=1e-10, rtol=0)
    frame = pd.DataFrame(transitions)
    frame.to_csv(OUT/'decision_transitions.csv', index=False)
    avg = frame.groupby(['budget', 'arm', 'layer', 'expert_argmax_changed', 'event'])[
        ['n', 'ba_contribution', 'oa_contribution']].mean().reset_index()
    avg.to_csv(OUT/'decision_transition_summary.csv', index=False)
    sel = pd.read_csv(OUT/'source_selection.csv')
    choices = pd.read_csv(OUT/'source_choices.csv')
    zero = sel[sel.intervention == 0].drop(columns='intervention').rename(columns={'ba': 'raw_ba'})
    chosen = sel.merge(choices[['budget', 'arm', 'intervention']], on=['budget', 'arm', 'intervention'])
    chosen = chosen.merge(zero[['context', 'fold', 'draw', 'budget', 'arm', 'raw_ba']],
                          on=['context', 'fold', 'draw', 'budget', 'arm'])
    chosen['delta_ba'] = chosen.ba-chosen.raw_ba
    chosen.groupby(['budget', 'arm', 'fold'])[['ba', 'delta_ba']].mean().reset_index().to_csv(
        OUT/'source_fold_gains.csv', index=False)
    metrics = pd.read_csv(OUT/'metrics.csv')
    records = []
    for budget in BUDGETS:
        target = metrics[(metrics.family == 'target') & (metrics.budget == budget)]
        for arm in ['corrected', 'simple']:
            expert = target[target.arm == 'expert_'+arm].set_index(['context', 'draw'])
            fusion = target[target.arm == 'fusion_'+arm].set_index(['context', 'draw'])
            records.append(dict(budget=budget, arm=arm, expert_delta_ba=float(expert.delta_ba.mean()),
                fusion_delta_ba=float(fusion.delta_ba.mean()),
                expert_positive_fusion_negative=int(((expert.delta_ba > 0) & (fusion.delta_ba < 0)).sum()),
                n_episodes=len(expert),
                fusion_row_brier_gain=float(fusion.delta_brier.mean()),
                fusion_portclass_brier_gain=float(fusion.portclass_risk_gain.mean()),
                expert_portclass_brier_gain=float(expert.portclass_risk_gain.mean())))
    write(OUT/'DIAGNOSTIC.json', dict(target_strength_search=False, decision_accounting_verified=True, results=records))
    print(pd.DataFrame(records).to_string(index=False))
    print(avg[(avg.budget == 128) & (avg.arm == 'corrected')].to_string(index=False))
    print(chosen.groupby(['budget', 'arm', 'fold']).delta_ba.mean().to_string())


if __name__ == '__main__':
    main()
