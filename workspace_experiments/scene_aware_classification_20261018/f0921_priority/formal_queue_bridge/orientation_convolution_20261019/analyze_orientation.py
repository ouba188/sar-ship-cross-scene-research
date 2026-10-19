"""Paired effects, angle-conditioned evidence, and actual support contributions."""
import json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from run_orientation import HERE, Study, OrientationReader

KEY = ['task', 'seed', 'draw', 'row']
METKEY = ['task', 'line', 'budget', 'seed', 'draw']
COLS = ['oa', 'ba', 'brier', 'nll', 'map_r']
COMPARISONS = [('A', 'base'), ('B', 'base'), ('C', 'base'), ('D', 'base'),
               ('B', 'A'), ('C', 'B'), ('C', 'D')]


def main():
    p = pd.read_csv(HERE / 'predictions.csv')
    m = pd.read_csv(HERE / 'metrics.csv')
    events, deltas, selected = [], [], []
    for arm, comp in COMPARISONS:
        a = p[p.arm.eq(arm)].set_index(KEY).sort_index()
        b = p[p.arm.eq(comp)].set_index(KEY).sort_index()
        assert a.index.equals(b.index)
        e = a[['line', 'budget', 'truth', 'prediction', 'margin', 'top_same', 'top_wrong', 'same_support']].copy()
        e['arm'], e['comparator'], e['old_prediction'] = arm, comp, b.prediction
        e['delta_margin'], e['delta_brier'] = a.margin-b.margin, a.brier-b.brier
        e['rescue'], e['harm'] = a.correct & ~b.correct, ~a.correct & b.correct
        e['persistent_correct'], e['persistent_wrong'] = a.correct & b.correct, ~a.correct & ~b.correct
        e = e.reset_index()
        events.append(e)
        d = m[m.arm.eq(arm)].set_index(METKEY)[COLS]-m[m.arm.eq(comp)].set_index(METKEY)[COLS]
        d = d.reset_index()
        d['arm'], d['comparator'] = arm, comp
        deltas.append(d)
        if (arm, comp) in [('B', 'A'), ('C', 'B'), ('C', 'D'), ('C', 'base')]:
            for category in ['rescue', 'harm', 'persistent_correct', 'persistent_wrong']:
                group = e[e.line.eq('cross_pass') & e.budget.eq(16) & e[category]]
                group = group.sort_values(['delta_margin', 'row', 'seed', 'draw'])
                if len(group):
                    case = group.iloc[(len(group)-1)//2].to_dict()
                    case['category'] = category
                    selected.append(case)
            # Inspect the positive B32 scope too; it cannot change trained models.
            if comp != 'base':
                group = e[e.line.eq('cross_pass') & e.budget.eq(32) & e.rescue].sort_values(['delta_margin', 'row', 'seed', 'draw'])
                if len(group):
                    case = group.iloc[(len(group)-1)//2].to_dict()
                    case['category'] = 'rescue_B32'
                    selected.append(case)
    events, delta = pd.concat(events), pd.concat(deltas)
    events.to_csv(HERE / 'paired_events.csv', index=False)
    delta.to_csv(HERE / 'paired_metrics.csv', index=False)
    seeds = delta.groupby(['arm', 'comparator', 'line', 'budget', 'seed'])[COLS].mean().reset_index()
    seeds.to_csv(HERE / 'paired_seed.csv', index=False)
    summary = []
    for (arm, comp, line, budget), g in seeds.groupby(['arm', 'comparator', 'line', 'budget']):
        ev = events[events.arm.eq(arm) & events.comparator.eq(comp) & events.line.eq(line) & events.budget.eq(budget)]
        summary.append(dict(arm=arm, comparator=comp, line=line, budget=budget,
                            oa_pp=100*g.oa.mean(), ba_pp=100*g.ba.mean(),
                            oa_seed_sd_pp=100*g.oa.std(), ba_seed_sd_pp=100*g.ba.std(),
                            positive_oa_seeds=int((g.oa > 1e-10).sum()), positive_ba_seeds=int((g.ba > 1e-10).sum()),
                            brier_delta=g.brier.mean(), map_delta=g.map_r.mean(),
                            rescue_events=int(ev.rescue.sum()), harm_events=int(ev.harm.sum()),
                            rescued_rows=ev[ev.rescue].row.nunique(), harmed_rows=ev[ev.harm].row.nunique()))
    summary = pd.DataFrame(summary)
    summary.to_csv(HERE / 'paired_summary.csv', index=False)
    events.groupby(['arm', 'comparator', 'line', 'budget', 'truth'])[['rescue', 'harm', 'delta_margin']].agg(['sum', 'mean']).to_csv(HERE / 'class_effects.csv')
    cases = pd.DataFrame(selected)
    cases.to_csv(HERE / 'selected_cases.csv', index=False)
    print(summary.to_string(index=False))

    relations = pd.read_csv(HERE / 'direction_relations.csv')
    relkey = KEY+['kind']
    outputs = []
    for arm, comp in [('B', 'A'), ('C', 'B'), ('C', 'D')]:
        a = relations[relations.arm.eq(arm)].set_index(relkey).sort_index()
        b = relations[relations.arm.eq(comp)].set_index(relkey).sort_index()
        assert a.index.equals(b.index)
        z = a[['line', 'budget', 'n_same_far', 'n_other_near']].copy()
        z['delta_gap'] = a.hard_relation_gap-b.hard_relation_gap
        z['arm'], z['comparator'] = arm, comp
        outputs.append(z.reset_index())
    rel = pd.concat(outputs)
    rel.to_csv(HERE / 'paired_direction_relations.csv', index=False)
    reports = []
    for key, group in rel.groupby(['arm', 'comparator', 'line', 'budget', 'kind']):
        ok = group.dropna(subset=['delta_gap'])
        by_query = ok.groupby('row').delta_gap.mean()
        reports.append(dict(zip(['arm', 'comparator', 'line', 'budget', 'kind'], key),
                            eligible_queries=len(by_query), repeated_query_events=len(ok),
                            mean_query_delta=float(by_query.mean()),
                            positive_queries=int((by_query > 0).sum())))
    pd.DataFrame(reports).to_csv(HERE / 'direction_summary.csv', index=False)

    study = Study()
    cache, paths, contributions = {}, [], []
    for case in cases.itertuples():
        task = next(t for t in study.tasks['tasks'] if t['name'] == case.task)
        ct = study.context(task['source'], task['supports'][case.draw], task['query'])
        j, ns = list(ct['q']).index(case.row), ct['ns']
        out = {}
        for arm in [case.arm, case.comparator]:
            if arm == 'base':
                out[arm] = [x.cpu().numpy() for x in (ct['base_prob'], ct['base'], ct['base_alpha'], ct['base_score'])]
                continue
            key = (arm, case.seed)
            if key not in cache:
                reader = OrientationReader(arm).cuda().eval()
                reader.load_state_dict(torch.load(HERE / f'{arm}_seed{case.seed}.pt', weights_only=True))
                cache[key] = reader
            with torch.no_grad():
                out[arm] = [x.cpu().numpy() for x in study.predict(cache[key], ct)[:4]]
        pp, k, alpha, score = out[case.arm]
        p0, k0, a0, s0 = out[case.comparator]
        recorded = p[p.arm.eq(case.arm) & p.task.eq(case.task) & p.seed.eq(case.seed) & p.draw.eq(case.draw) & p.row.eq(case.row)]
        np.testing.assert_allclose(pp[j], recorded[[f'p{i}' for i in range(7)]].to_numpy()[0], atol=2e-8)
        truth = case.truth
        rivals = p0[j].copy()
        rivals[truth] = -np.inf
        rival = int(rivals.argmax())
        aa, ab = alpha[:, truth]-alpha[:, rival], a0[:, truth]-a0[:, rival]
        ka, kb = k[ns+j, :ns], k0[ns+j, :ns]
        direct = .5*(ka-kb)*(aa+ab)
        refit = .5*(ka+kb)*(aa-ab)
        total = direct+refit
        observed = (score[ns+j, truth]-score[ns+j, rival])-(s0[ns+j, truth]-s0[ns+j, rival])
        np.testing.assert_allclose(total.sum(), observed, atol=1e-10)
        tags = dict(task=case.task, seed=case.seed, draw=case.draw, row=case.row,
                    arm=case.arm, comparator=case.comparator, category=case.category)
        paths.append(dict(**tags, truth=truth, rival=rival, direct=float(direct.sum()), refit=float(refit.sum()),
                          branch_margin_change=float(observed),
                          same_class_sum=float(total[study.y[ct['s']] == truth].sum()),
                          old_final_margin=float(p0[j, truth]-np.delete(p0[j], truth).max()),
                          new_final_margin=float(pp[j, truth]-np.delete(pp[j], truth).max())))
        for t, s in enumerate(ct['s']):
            contributions.append(dict(**tags, support_row=int(s), support_label=int(study.y[s]),
                                      old_similarity=float(kb[t]), new_similarity=float(ka[t]),
                                      direct=float(direct[t]), refit=float(refit[t]), total=float(total[t])))
    pd.DataFrame(paths).to_csv(HERE / 'case_paths.csv', index=False)
    pd.DataFrame(contributions).to_csv(HERE / 'case_support_contributions.csv', index=False)
    print('CASE PATHS', pd.DataFrame(paths).to_string(index=False))


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=1):
        main()
