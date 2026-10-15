"""Select the same frozen candidates by expert versus complete-system source BA."""

from full_teacher_data import BASE, SOURCE, PORTS, write
from case_support_completion import COLS, predict
from unified_head_core import score, late
from unified_head_run import base_kernel
from evaluate_probability_layers import cached
from run_support_budget_strength import source_episodes, corrected_score, GRID, BUDGETS
import run_support_budget as previous

import argparse
import json
import numpy as np
import pandas as pd


OUT = BASE/'support_budget_fusion_select_20261012'
OLD = BASE/'support_budget_20261012'
EXPERT_CHOICE = BASE/'support_budget_strength_20261012'
CANDIDATES = [('raw', 0.)]+[(a, t) for a in ('simple', 'corrected') for t in GRID[1:]]
POLICIES = ('expert_selected', 'fusion_selected', 'repair_expert_selected',
            'repair_fusion_selected', 'simple_expert_selected', 'simple_fusion_selected')


def base_scores(data, prior, s, q, sy):
    components = {k: data[k] for k in ('phi', 'mean', 'conditions')}
    kernel = base_kernel(components, data['embedding'].astype(float), s, float(data['eta']))
    return score(kernel, prior, s, sy, previous.strength(len(s)))[0][q]


def choose(frame, criterion, family=None):
    subset = frame if family is None else frame[frame.arm.isin(['raw', family])]
    means = subset.groupby(['arm', 'intervention'])[criterion].mean().reset_index()
    means['priority'] = means.arm.map({'raw': 0, 'simple': 1, 'corrected': 2})
    winner = means.sort_values([criterion, 'intervention', 'priority'],
                              ascending=[False, True, True], kind='stable').iloc[0]
    return dict(arm=str(winner.arm), intervention=float(winner.intervention),
                source_ba=float(winner[criterion]))


def check_selector():
    tied = pd.DataFrame([dict(arm=a, intervention=t, ba=40.) for a, t in CANDIDATES])
    assert choose(tied, 'ba')['arm'] == 'raw'
    tied.loc[tied.arm == 'corrected', 'ba'] = 41.
    assert choose(tied, 'ba')['intervention'] == .25
    assert choose(tied, 'ba', 'simple')['arm'] == 'raw'
    opposing = pd.DataFrame([
        dict(arm='raw', intervention=0., expert_ba=40., fusion_ba=40.),
        dict(arm='simple', intervention=.25, expert_ba=45., fusion_ba=39.),
        dict(arm='corrected', intervention=.25, expert_ba=44., fusion_ba=43.),
    ])
    assert choose(opposing, 'expert_ba')['arm'] == 'simple'
    assert choose(opposing, 'fusion_ba')['arm'] == 'corrected'
    return ['ties_prefer_raw_then_least_intervention', 'family_selection_excludes_other_family',
            'criterion_change_can_reverse_selection_with_identical_candidates']


def selection(meta):
    from budget_fusion_source_base import prepare
    folds = cached(BASE/'crossfit_evidence_20261011/source_folds.npz')
    fit_ids = folds['fit_obs_ids']
    episodes = list(source_episodes(meta, fit_ids, folds['assignment']))
    y = meta.iloc[fit_ids].class_id.to_numpy()
    records = []
    for context in PORTS:
        bd = prepare(context)
        maps = cached(previous.TEACHER/context/'maps.npz')
        np.testing.assert_array_equal(maps['obs_ids'], bd['obs_ids'])
        np.testing.assert_array_equal(maps['obs_ids'][:len(fit_ids)], fit_ids)
        phi = maps['single'][:len(fit_ids)]
        prior = cached(BASE/'case_relation_student_20261012'/context/'source_readers.npz')['fit_prior']
        base_data = {k: bd[k][:len(fit_ids)] for k in ('phi', 'mean', 'conditions', 'embedding')}
        base_data['eta'] = bd['eta']
        models = {f: cached(previous.CONTROL/context/f'fold_{f}_model.npz') for f in range(3)}
        dest = OUT/'selection'/context
        dest.mkdir(parents=True, exist_ok=True)
        for fold, draw, full_s, q in episodes:
            full_delta = predict(models[fold], phi[full_s][:, COLS], np.zeros(len(full_s), int), 'image_history')
            for budget in BUDGETS:
                s, delta = full_s[:budget], full_delta[:budget]
                base = base_scores(base_data, bd['prior_oof'], s, q, y[s])
                raw = corrected_score(phi, prior, s, q, y[s], delta, 0.)
                experts, fusions = [], []
                for arm, t in CANDIDATES:
                    if arm == 'raw':
                        expert = raw
                    elif arm == 'simple':
                        expert = prior[q]+(1-t)*(raw-prior[q])
                    else:
                        expert = corrected_score(phi, prior, s, q, y[s], delta, t)
                    fusion = late(base, expert)
                    experts.append(expert)
                    fusions.append(fusion)
                    records.append(dict(context=context, fold=fold, draw=draw, budget=budget, arm=arm,
                        intervention=t, expert_ba=previous.ba(y[q], expert), fusion_ba=previous.ba(y[q], fusion),
                        expert_oa=100*np.mean(expert.argmax(1) == y[q]),
                        fusion_oa=100*np.mean(fusion.argmax(1) == y[q]), n_query=len(q)))
                np.savez_compressed(dest/f'{fold}_{draw}_{budget}.npz', query_ids=fit_ids[q],
                    support_ids=fit_ids[s], base=base, expert=np.stack(experts), fusion=np.stack(fusions))
        print('SOURCE CANDIDATES', context, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'source_selection.csv', index=False)
    summary = frame.groupby(['budget', 'arm', 'intervention'])[
        ['expert_ba', 'fusion_ba', 'expert_oa', 'fusion_oa']].mean().reset_index()
    summary.to_csv(OUT/'source_selection_summary.csv', index=False)
    # Source expert scores must reproduce the previous experiment; only selection changes.
    old = pd.read_csv(EXPERT_CHOICE/'source_selection.csv')
    compare = frame[frame.arm != 'raw'].merge(old, on=['context', 'fold', 'draw', 'budget', 'arm', 'intervention'])
    assert len(compare) == 648
    np.testing.assert_allclose(compare.expert_ba, compare.ba, atol=1e-10, rtol=0)
    choices = {}
    for budget in BUDGETS:
        f = frame[frame.budget == budget]
        choices[str(budget)] = {
            'expert_selected': choose(f, 'expert_ba'), 'fusion_selected': choose(f, 'fusion_ba'),
            'repair_expert_selected': choose(f, 'expert_ba', 'corrected'),
            'repair_fusion_selected': choose(f, 'fusion_ba', 'corrected'),
            'simple_expert_selected': choose(f, 'expert_ba', 'simple'),
            'simple_fusion_selected': choose(f, 'fusion_ba', 'simple'),
        }
    write(OUT/'FROZEN_CHOICE.json', dict(choices=choices, candidates=CANDIDATES,
        selection='Equal source episode expert or fusion BA on identical candidate outputs',
        ties='Least intervention, then raw/simple/corrected', target_selection=False,
        source_expert_replay_count=len(compare), source_prior='OOF by vessel, exclude held vessels',
        historical_encoder_geometry_exposure=True, loss_changed=False))
    print('CHOICES', json.dumps(choices), flush=True)
    return choices


def chosen_scores(choices, phi, prior, s, q, sy, model, raw):
    delta = predict(model, phi[s][:, COLS], np.zeros(len(s), int), 'image_history')
    scores = {('raw', 0.): raw}
    answer = {}
    for policy, choice in choices.items():
        arm, t = choice['arm'], choice['intervention']
        key = (arm, t)
        if key not in scores:
            if arm == 'simple':
                scores[key] = prior[q]+(1-t)*(raw-prior[q])
            else:
                scores[key] = corrected_score(phi, prior, s, q, sy, delta, t)
        answer[policy] = scores[key]
    return answer


def freeze(choices, meta, manifest):
    from budget_fusion_source_base import prepare
    replays = []
    for family, table, draws in [('source', meta, 3), ('target', manifest, 5)]:
        for context in PORTS:
            if family == 'source':
                maps = cached(previous.TEACHER/context/'maps.npz')
                ids, phi, prior = maps['obs_ids'], maps['single'], maps['prior']
                bd = prepare(context)
                np.testing.assert_array_equal(bd['obs_ids'], ids)
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
                    base = base_scores(bd, bd['prior_held'], s, q, sy) if family == 'source' else old['base']
                    raw = old['expert_raw']
                    outputs = chosen_scores(choices[str(budget)], phi, prior, s, q, sy, model, raw)
                    variants = dict(query_ids=old['query_ids'], support_ids=old['support_ids'], base=base,
                                    expert_raw=raw, fusion_raw=late(base, raw))
                    for policy, expert in outputs.items():
                        variants['expert_'+policy] = expert
                        variants['fusion_'+policy] = late(base, expert)
                    previous_selected = cached(EXPERT_CHOICE/family/context/f'{draw}_{budget}.npz')
                    for new, prev in [('repair_expert_selected', 'corrected'), ('simple_expert_selected', 'simple')]:
                        error = float(abs(outputs[new]-previous_selected['expert_'+prev]).max())
                        np.testing.assert_allclose(outputs[new], previous_selected['expert_'+prev], atol=1e-10, rtol=0)
                        replays.append(dict(family=family, context=context, budget=budget, draw=draw, arm=new, max_error=error))
                    if family == 'target':
                        np.testing.assert_allclose(variants['fusion_raw'], old['fusion_raw'], atol=1e-10, rtol=0)
                    np.savez_compressed(dest/f'{draw}_{budget}.npz', **variants)
            print('FROZEN', family, context, flush=True)
        write(OUT/(family.upper()+'_PREDICTIONS_FROZEN.json'), dict(episodes=draws*len(PORTS)*len(BUDGETS)))
    write(OUT/'REPLAY.json', replays)
    write(OUT/'PREDICTIONS_FROZEN.json', dict(target_episodes=45, source_held_episodes=27,
        target_query_labels_evaluated_after_freeze=True))


def evaluate(choices, meta, manifest):
    previous.OUT = OUT
    frame, summary, avg = previous.evaluate(meta, manifest)
    def is_kernel(row):
        if not row.arm.startswith('expert_') or row.arm == 'expert_raw':
            return False
        return choices[str(row.budget)][row.arm[len('expert_'):]]['arm'] == 'corrected'
    frame['kernel_intervention'] = frame.apply(is_kernel, axis=1)
    frame.to_csv(OUT/'metrics.csv', index=False)
    differences = []
    for key, g in frame.groupby(['family', 'context', 'budget', 'draw']):
        indexed = g.set_index('arm')
        for prefix in ('', 'repair_', 'simple_'):
            a = indexed.loc['fusion_'+prefix+'fusion_selected']
            b = indexed.loc['fusion_'+prefix+'expert_selected']
            differences.append(dict(zip(['family', 'context', 'budget', 'draw'], key), family_scope=prefix or 'union',
                delta_ba=a.ba-b.ba, delta_oa=a.oa-b.oa,
                portclass_risk_gain=b.portclass_brier-a.portclass_brier))
    pd.DataFrame(differences).to_csv(OUT/'selection_objective_effect.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm, label, color in [('raw', 'Original', '#555555'),
        ('expert_selected', 'Chosen by expert BA', '#aa6677'),
        ('fusion_selected', 'Chosen by full-system BA', '#007f88'),
        ('repair_fusion_selected', 'Repair: full-system selection', '#997700')]:
        z = avg[(avg.family == 'target') & (avg.arm == 'fusion_'+arm)].sort_values('budget')
        for ax, metric in zip(axes, ['ba_mean', 'oa_mean']):
            ax.plot(z.budget, z[metric], 'o-', label=label, color=color)
    for ax, label in zip(axes, ['Full-system BA (%)', 'Full-system OA (%)']):
        ax.set(xlabel='Labeled support observations B', ylabel=label, xticks=BUDGETS)
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, .17, 1, 1))
    fig.savefig(OUT/'FUSION_SELECTION.png', dpi=160)
    fig.savefig(OUT/'FUSION_SELECTION.pdf')
    plt.close(fig)
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-selection', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    tests = check_selector()
    meta = pd.read_csv(BASE/'observations.csv')
    manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'class_name', 'port', 'mmsi', 'product'])
    if args.reuse_selection:
        choices = json.loads((OUT/'FROZEN_CHOICE.json').read_text())['choices']
    else:
        choices = selection(meta)
    freeze(choices, meta, manifest)
    avg = evaluate(choices, meta, manifest)
    write(OUT/'COMPLETE.json', dict(status='complete', new_neural_fits=0, new_repair_fits=0,
        base_reader_fits=12, deployment_replay_reader_fits=3,
        selection_objective='Full fusion versus expert BA',
        budgets=BUDGETS, source_held_episodes=27, target_episodes=45, tests=tests,
        learned_loss_unchanged=True, main_model_unchanged=True, target_selection=False))


if __name__ == '__main__':
    main()
