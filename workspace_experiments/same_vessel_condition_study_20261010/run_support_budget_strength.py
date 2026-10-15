"""Matched source-only, per-budget strength selection on existing SAR episodes."""

from full_teacher_data import BASE, SOURCE, PORTS, write
from case_support_completion import COLS, predict, adapt_support
from unified_head_core import coefficients, late
from evaluate_probability_layers import cached
import run_support_budget as previous

import numpy as np
import pandas as pd


OUT = BASE/'support_budget_strength_20261012'
OLD = BASE/'support_budget_20261012'
GRID = (0., .25, .5, .75, 1.)
BUDGETS = previous.BUDGETS


def corrected_score(phi, prior, s, q, sy, delta, dose):
    ps = adapt_support(phi[s], delta, dose)
    alpha = coefficients(ps@ps.T, prior[s], sy, previous.strength(len(s)))
    return prior[q]+(phi[q]@ps.T)@alpha


def source_episodes(meta, fit_ids, assignment):
    for fold in range(3):
        ids = np.flatnonzero(assignment == fold)
        frame = meta.iloc[fit_ids[ids]]
        ships = np.random.default_rng(121000+fold).permutation(np.sort(frame.mmsi.unique()))
        query_ship = ships[:len(ships)//5]
        q = ids[frame.mmsi.isin(query_ship).to_numpy()]
        pool = ids[~frame.mmsi.isin(query_ship).to_numpy()]
        pool_identity = meta.iloc[fit_ids[pool]].mmsi.to_numpy()
        pool_ships = np.sort(np.unique(pool_identity))
        for draw in range(3):
            rng = np.random.default_rng(122000+fold*10+draw)
            order = rng.permutation(pool_ships)[:max(BUDGETS)]
            s = np.array([rng.choice(pool[pool_identity == ship]) for ship in order])
            assert len(s) == max(BUDGETS)
            assert not set(meta.iloc[fit_ids[q]].mmsi) & set(meta.iloc[fit_ids[s]].mmsi)
            yield fold, draw, s, q


def select(meta):
    folds = cached(BASE/'crossfit_evidence_20261011/source_folds.npz')
    fit_ids = folds['fit_obs_ids']
    episodes = list(source_episodes(meta, fit_ids, folds['assignment']))
    labels = meta.iloc[fit_ids].class_id.to_numpy()
    records = []
    for context in PORTS:
        maps = cached(previous.TEACHER/context/'maps.npz')
        np.testing.assert_array_equal(maps['obs_ids'][:len(fit_ids)], fit_ids)
        phi = maps['single'][:len(fit_ids)]
        prior = cached(BASE/'case_relation_student_20261012'/context/'source_readers.npz')['fit_prior']
        models = {f: cached(previous.CONTROL/context/f'fold_{f}_model.npz') for f in range(3)}
        for fold, draw, full_s, q in episodes:
            full_delta = predict(models[fold], phi[full_s][:, COLS], np.zeros(len(full_s), int), 'image_history')
            for budget in BUDGETS:
                s, delta = full_s[:budget], full_delta[:budget]
                raw = corrected_score(phi, prior, s, q, labels[s], delta, 0.)
                for t in GRID:
                    predictions = {
                        'corrected': raw if t == 0 else corrected_score(phi, prior, s, q, labels[s], delta, t),
                        'simple': prior[q]+(1-t)*(raw-prior[q]),
                    }
                    for arm, pred in predictions.items():
                        records.append(dict(context=context, fold=fold, draw=draw, budget=budget,
                            arm=arm, intervention=t, ba=previous.ba(labels[q], pred), n_query=len(q)))
        print('SOURCE SELECTION', context, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'source_selection.csv', index=False)
    summary = frame.groupby(['budget', 'arm', 'intervention']).ba.mean().reset_index()
    summary.to_csv(OUT/'source_selection_summary.csv', index=False)
    # Both arms choose the smallest intervention in an exact tie.
    chosen = summary.sort_values(['ba', 'intervention'], ascending=[False, True], kind='stable')
    chosen = chosen.groupby(['budget', 'arm'], sort=True).head(1).sort_values(['budget', 'arm'])
    chosen.to_csv(OUT/'source_choices.csv', index=False)
    choices = {str(b): {a: float(chosen[(chosen.budget == b) & (chosen.arm == a)].intervention.iloc[0])
                       for a in ('corrected', 'simple')} for b in BUDGETS}
    write(OUT/'FROZEN_CHOICE.json', dict(choices=choices, budgets=BUDGETS, grid=GRID,
        parameterization='corrected dose=t; simple residual multiplier=1-t; t=0 is unchanged',
        criterion='Expert BA, equally averaged over source geometry/fold/draw, separately per budget',
        tie_break='smallest intervention', source_folds=3, draws_per_fold=3,
        predictor='existing fold-excluded class-common model; final source model on held/target only',
        selection_query_labels='source only', target_selection=False,
        ridge_per_example=8/64, probability_temperature=1.,
        history='Follow-up hypothesis motivated by previously observed budget results, not independent confirmation'))
    print('FROZEN CHOICES', choices, flush=True)
    return choices


def freeze(choices, meta, manifest):
    checks = []
    for family, table, draws in [('source', meta, 3), ('target', manifest, 5)]:
        for context in PORTS:
            if family == 'source':
                maps = cached(previous.TEACHER/context/'maps.npz')
                ids, phi, prior = maps['obs_ids'], maps['single'], maps['prior']
            else:
                maps = cached(BASE/'full_teacher_integration_20261012'/context/'no_training/predictions.npz')
                ids, phi, prior = maps['rows'], maps['phi'], maps['prior']
            index = pd.Index(ids)
            model = cached(previous.CONTROL/context/'final_model.npz')
            dest = OUT/family/context
            dest.mkdir(parents=True, exist_ok=True)
            for draw in range(draws):
                for budget in BUDGETS:
                    old = cached(OLD/family/context/f'{draw}_{budget}.npz')
                    s, q = index.get_indexer(old['support_ids']), index.get_indexer(old['query_ids'])
                    assert (s >= 0).all() and (q >= 0).all()
                    sy = table.iloc[old['support_ids']].class_id.to_numpy()
                    delta = predict(model, phi[s][:, COLS], np.zeros(len(s), int), 'image_history')
                    for dose, reference in [(0., 'expert_raw'), (1., 'expert_corrected')]:
                        replay = corrected_score(phi, prior, s, q, sy, delta, dose)
                        np.testing.assert_allclose(replay, old[reference], atol=1e-10, rtol=0)
                        checks.append(dict(family=family, context=context, draw=draw, budget=budget,
                                           arm=reference, max_error=float(abs(replay-old[reference]).max())))
                    chosen = choices[str(budget)]
                    repair = corrected_score(phi, prior, s, q, sy, delta, chosen['corrected'])
                    simple = prior[q]+(1-chosen['simple'])*(old['expert_raw']-prior[q])
                    variants = dict(query_ids=old['query_ids'], support_ids=old['support_ids'],
                        expert_raw=old['expert_raw'], expert_corrected=repair, expert_simple=simple,
                        expert_fixed_corrected=old['expert_corrected'], expert_fixed_simple=old['expert_simple'])
                    if family == 'target':
                        variants.update(base=old['base'], fusion_raw=old['fusion_raw'],
                            fusion_corrected=late(old['base'], repair), fusion_simple=late(old['base'], simple),
                            fusion_fixed_corrected=old['fusion_corrected'], fusion_fixed_simple=old['fusion_simple'])
                    np.savez_compressed(dest/f'{draw}_{budget}.npz', **variants)
            print('PREDICTIONS FROZEN', family, context, flush=True)
    write(OUT/'REPLAY.json', checks)
    write(OUT/'PREDICTIONS_FROZEN.json', dict(target_episodes=45, held_episodes=27,
        target_query_labels_read_by_evaluator_only=True, replay_count=len(checks)))
    return checks


def summarize(frame, summary, avg):
    # Label kernel interventions precisely; simple attenuation changes the readout only.
    frame['kernel_intervention'] = frame.arm.isin(['expert_corrected', 'expert_fixed_corrected'])
    frame.to_csv(OUT/'metrics.csv', index=False)
    comparisons = []
    for family, layer in [('source', 'expert'), ('target', 'expert'), ('target', 'fusion')]:
        for budget in BUDGETS:
            f = frame[(frame.family == family) & (frame.budget == budget)].set_index(['context', 'draw', 'arm'])
            for arm, reference in [('corrected', 'fixed_corrected'), ('simple', 'fixed_simple'), ('corrected', 'simple')]:
                a = f.xs(layer+'_'+arm, level='arm')
                b = f.xs(layer+'_'+reference, level='arm')
                for ix in a.index:
                    comparisons.append(dict(family=family, layer=layer, budget=budget, context=ix[0], draw=ix[1],
                        comparison=arm+'_vs_'+reference, delta_ba=a.loc[ix, 'ba']-b.loc[ix, 'ba'],
                        delta_oa=a.loc[ix, 'oa']-b.loc[ix, 'oa'],
                        portclass_risk_gain=b.loc[ix, 'portclass_brier']-a.loc[ix, 'portclass_brier']))
    pd.DataFrame(comparisons).to_csv(OUT/'paired_comparisons.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    styles = [('raw', 'Original', '#555555', '-'),
              ('fixed_corrected', 'Repair: fixed dose 1', '#b86126', '--'),
              ('corrected', 'Repair: source-selected by B', '#007b83', '-'),
              ('fixed_simple', 'Attenuation: fixed a=.75', '#986287', '--'),
              ('simple', 'Attenuation: source-selected by B', '#4477aa', '-')]
    for arm, label, color, linestyle in styles:
        selected = avg[(avg.family == 'target') & (avg.arm == 'fusion_'+arm)].sort_values('budget')
        for ax, metric in zip(axes, ['ba_mean', 'oa_mean', 'portclass_risk_gain_mean']):
            ax.plot(selected.budget, selected[metric], marker='o', label=label, color=color, linestyle=linestyle)
    for ax, label in zip(axes, ['Full-system BA (%)', 'Full-system OA (%)', 'Port/class Brier gain vs original']):
        ax.set(xlabel='Labeled support observations B', ylabel=label, xticks=BUDGETS)
        ax.grid(alpha=.2)
    axes[2].axhline(0, color='black', linewidth=.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .16, 1, 1))
    fig.savefig(OUT/'STRENGTH_EVIDENCE.png', dpi=160)
    fig.savefig(OUT/'STRENGTH_EVIDENCE.pdf')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    meta = pd.read_csv(BASE/'observations.csv', usecols=['class_id', 'class_name', 'port', 'mmsi', 'product'])
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'class_name', 'port', 'mmsi', 'product'])
    choices = select(meta)
    checks = freeze(choices, meta, manifest)
    previous.OUT = OUT
    frame, summary, avg = previous.evaluate(meta, manifest)
    summarize(frame, summary, avg)
    write(OUT/'COMPLETE.json', dict(status='complete', new_neural_fits=0, new_predictor_fits=0,
        selected_strengths=choices, target_selection=False, target_episodes=45, source_held_episodes=27,
        source_candidate_scores=810, endpoint_replay_checks=len(checks),
        scope='historical development, source geometry versions are not independent ports',
        kernel_coefficients_resolved=True, expert_selection_not_fusion_selection=True))


if __name__ == '__main__':
    main()
