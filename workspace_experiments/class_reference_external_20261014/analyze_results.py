"""Paired external classification, risk, fusion and support-composition effects."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from experiment import OUT, OLD, ARMS, GRID, scores, write

VALUES = ['oa', 'ba', 'brier', 'macro_brier', 'nll', 'macro_nll']
GROUP = ['study', 'dataset', 'budget', 'cell']


def main():
    plans = json.loads((OUT/'EXTERNAL_ENROLLMENT.json').read_text(encoding='utf-8'))
    selected = json.loads((OUT/'SELECTED_WEIGHTS.json').read_text(encoding='utf-8'))
    frames = {n: pd.read_csv(OLD/f'{n}_metadata.csv', dtype={'identity': str}, keep_default_na=False) for n in ['f0921', 'oss2']}
    metrics = []; classes = []; curves = []; events = []; checks = 0; decomposition = []; composition = []
    comparisons = [('expert_gain', 'original:expert', 'class_mean:expert'),
        ('equal_gain', 'original:equal', 'class_mean:equal'),
        ('selected_gain', 'original:selected', 'class_mean:selected'),
        ('kernel_at_original_weight', 'original:selected', 'class_mean:original_weight'),
        ('weight_on_class_mean', 'class_mean:original_weight', 'class_mean:selected'),
        ('original_weight_only', 'original:selected', 'original:mean_weight'),
        ('kernel_at_mean_weight', 'original:mean_weight', 'class_mean:selected'),
        ('mean_vs_shrink_equal', 'shrink_control:equal', 'class_mean:equal'),
        ('mean_vs_shrink_selected', 'shrink_control:selected', 'class_mean:selected')]
    for e in plans:
        d = frames[e['target']]
        with np.load(OUT/'predictions'/f"episode_{e['episode']:03d}.npz") as f:
            base, expert, q, supp = [f[k] for k in ['base', 'expert', 'query_rows', 'support_rows']]
        y = d.iloc[q].class_id.to_numpy(int); present = np.unique(y)
        wo = selected[f"{e['budget']}:original"]['weight']; wm = selected[f"{e['budget']}:class_mean"]['weight']
        for draw in range(5):
            for ci, cell in enumerate(e['arms']):
                info = dict(episode=e['episode'], study=e['study'], dataset=e['target'], budget=e['budget'],
                            acquisition=e['acquisition'], cell=cell, draw=draw)
                sy = d.iloc[supp[draw, ci]].class_id.to_numpy(int)
                sc = np.bincount(sy, minlength=8); results = {}
                bb = base[draw, ci]
                for ai, arm in enumerate(ARMS):
                    ee = expert[draw, ci, ai]; ws = selected[f"{e['budget']}:{arm}"]['weight']
                    modes = dict(expert=1., equal=.5, selected=ws, original_weight=wo, mean_weight=wm)
                    for mode, w in modes.items():
                        pp = (1-w)*bb+w*ee; results[f'{arm}:{mode}'] = pp
                        m = scores(y, pp)
                        np.testing.assert_allclose([m['oa'], m['ba']],
                            [100*accuracy_score(y, pp.argmax(1)), 100*recall_score(y, pp.argmax(1), labels=present, average='macro', zero_division=0)], atol=1e-10)
                        assert np.isfinite(pp).all()
                        np.testing.assert_allclose(pp.sum(1), 1, atol=1e-12)
                        checks += 1
                        metrics.append(dict(**info, arm=arm, mode=mode, weight=w, n=len(y), **m))
                    for w in GRID:
                        curves.append(dict(**info, arm=arm, weight=w, **scores(y, (1-w)*bb+w*ee)))
                for name, oldkey, newkey in comparisons:
                    p0, p1 = results[oldkey], results[newkey]
                    old, new = scores(y, p0), scores(y, p1)
                    co, cn = p0.argmax(1) == y, p1.argmax(1) == y
                    row = dict(**info, comparison=name, n=len(y), rescued=int(np.sum(~co&cn)), harmed=int(np.sum(co&~cn)),
                               **{'delta_'+k: new[k]-old[k] for k in VALUES})
                    decomposition.append(row)
                    contributions = []
                    for c in present:
                        mask = y == c; net = np.sum(cn[mask])-np.sum(co[mask])
                        rr = dict(**info, comparison=name, class_id=int(c), n=int(mask.sum()), support_n=int(sc[c]),
                            rescued=int(np.sum(~co&cn&mask)), harmed=int(np.sum(co&~cn&mask)),
                            delta_oa=100*net/len(y), delta_ba=100*net/mask.sum()/len(present),
                            old_recall=100*co[mask].mean(), new_recall=100*cn[mask].mean())
                        classes.append(rr); contributions.append([rr['delta_oa'], rr['delta_ba']])
                    np.testing.assert_allclose(np.sum(contributions, axis=0), [row['delta_oa'], row['delta_ba']], atol=1e-10)
                    if name in ['equal_gain', 'selected_gain']:
                        for status in ['support_present', 'support_absent']:
                            mask = (sc[y] > 0) if status == 'support_present' else (sc[y] == 0)
                            if mask.any():
                                a, b = scores(y[mask], p0[mask]), scores(y[mask], p1[mask])
                                composition.append(dict(**info, comparison=name, status=status, n=int(mask.sum()),
                                    delta_oa=a['oa']*-1+b['oa'], delta_ba=a['ba']*-1+b['ba']))
                        for j in np.flatnonzero(co != cn):
                            events.append(dict(**info, comparison=name, row=int(q[j]), ship=d.iloc[q[j]].identity,
                                class_id=int(y[j]), support_n=int(sc[y[j]]), outcome='rescue' if cn[j] else 'harm',
                                old_prediction=int(p0[j].argmax()), new_prediction=int(p1[j].argmax()),
                                old_true_probability=float(p0[j, y[j]]), new_true_probability=float(p1[j, y[j]])))
    m = pd.DataFrame(metrics); ef = pd.DataFrame(decomposition); cl = pd.DataFrame(classes)
    m.to_csv(OUT/'metrics.csv', index=False); ef.to_csv(OUT/'effects.csv', index=False)
    cl.to_csv(OUT/'classes.csv', index=False); pd.DataFrame(events).to_csv(OUT/'changed_cases.csv', index=False)
    pd.DataFrame(composition).to_csv(OUT/'support_composition.csv', index=False)
    c = pd.DataFrame(curves); c.to_csv(OUT/'target_grid_metrics.csv', index=False)
    c.groupby(GROUP+['arm', 'weight'])[VALUES].mean().to_csv(OUT/'target_grid_summary.csv')
    m.groupby(GROUP+['arm', 'mode'])[VALUES].mean().to_csv(OUT/'absolute_summary.csv')
    delta = ['delta_'+k for k in VALUES]
    ef.groupby(GROUP+['comparison'])[delta].mean().to_csv(OUT/'effect_summary.csv')
    ef.groupby(GROUP+['comparison', 'acquisition'])[delta].mean().to_csv(OUT/'acquisition_effects.csv')
    dr = ef.groupby(GROUP+['comparison', 'draw'])[delta].mean().reset_index()
    dr.to_csv(OUT/'draw_effects.csv', index=False)
    dr.groupby(GROUP+['comparison']).agg(ba_mean=('delta_ba', 'mean'), ba_sd=('delta_ba', 'std'),
        ba_positive=('delta_ba', lambda x: int((x > 0).sum())), oa_mean=('delta_oa', 'mean'), oa_sd=('delta_oa', 'std'),
        oa_positive=('delta_oa', lambda x: int((x > 0).sum()))).to_csv(OUT/'support_variability.csv')
    # Zero-fill missing classes before averaging complete-task BA contributions.
    totals = ef.groupby(GROUP+['comparison']).size().rename('tasks')
    cs = cl.groupby(GROUP+['comparison', 'class_id'])[['delta_oa', 'delta_ba', 'rescued', 'harmed']].sum().join(totals)
    cs[delta[:2]] = cs[delta[:2]].div(cs.tasks, axis=0)
    cs.to_csv(OUT/'class_effects.csv')
    cp = cl.groupby(GROUP+['comparison', 'class_id'])[['old_recall', 'new_recall']].mean()
    cp.to_csv(OUT/'class_recall_present_tasks.csv')
    idx = ['episode', 'draw', 'cell']
    ee = ef.set_index(idx+['comparison'])
    total = ee.xs('selected_gain', level='comparison')[delta]
    pieces = ee.xs('kernel_at_original_weight', level='comparison')[delta]+ee.xs('weight_on_class_mean', level='comparison')[delta]
    np.testing.assert_allclose(total, pieces, atol=1e-10)
    write(OUT/'ANALYSIS_COMPLETE.json', dict(metric_checks=checks, class_sum_checks=len(ef),
        additive_fusion_decomposition_checks=len(total), target_oracle_selection=False))
    print('GLOBAL EFFECTS\n', ef[ef.study == 'dataset_adaptation'].groupby(['dataset', 'budget', 'comparison'])[delta[:2]].mean().to_string(), flush=True)
    print('ACQUISITION EFFECTS\n', ef[(ef.study == 'acquisition_transfer')&ef.comparison.isin(['equal_gain','selected_gain'])].groupby(['dataset', 'budget', 'cell', 'comparison'])[delta[:2]].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
