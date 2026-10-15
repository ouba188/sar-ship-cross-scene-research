"""Compact equal-port summaries, keeping training seeds and support draws apart."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VALUES = ['oa', 'ba', 'macro_nll', 'macro_brier']
KEY = ['port', 'episode', 'draw', 'cell']


def summarize(path):
    data = pd.read_csv(path/'metrics.csv')
    full = data[data['mode'].eq('full')].copy()
    per_port_seed = full.groupby(['cell', 'anchor', 'objective', 'seed', 'port'])[VALUES].mean().reset_index()
    per_seed = per_port_seed.groupby(['cell', 'anchor', 'objective', 'seed'])[VALUES].mean().reset_index()
    per_seed.to_csv(path/'absolute_by_training_seed.csv', index=False)
    per_port_seed.groupby(['cell', 'anchor', 'objective', 'port'])[VALUES].mean().reset_index().to_csv(path/'absolute_by_port.csv', index=False)
    absolute = per_seed.groupby(['cell', 'anchor', 'objective'])[VALUES].agg(['mean', 'std'])
    absolute.columns = ['_'.join(c) for c in absolute.columns]
    absolute.reset_index().to_csv(path/'absolute_summary.csv', index=False)
    comparisons = [
        ('class_task_vs_class_fixed', 'class_mean', 'task', 'class_mean', 'fixed'),
        ('class_task_vs_class_contrastive', 'class_mean', 'task', 'class_mean', 'contrastive'),
        ('class_task_vs_cluster_task', 'class_mean', 'task', 'cluster_0', 'task'),
        ('class_task_vs_original', 'class_mean', 'task', 'original', 'fixed'),
        ('cluster_task_vs_cluster_fixed', 'cluster_0', 'task', 'cluster_0', 'fixed')]
    summary, seed_rows, draw_rows, port_rows = [], [], [], []
    for name, a, objective, b, reference in comparisons:
        left = full[full.anchor.eq(a) & full.objective.eq(objective)]
        right = full[full.anchor.eq(b) & full.objective.eq(reference)]
        keys = KEY + ([] if reference == 'fixed' else ['seed'])
        pair = left.merge(right[keys+VALUES], on=keys, suffixes=('', '_reference'), validate='many_to_one' if reference == 'fixed' else 'one_to_one')
        assert len(pair) == len(left)
        for value in VALUES:
            pair[value] -= pair[value+'_reference']
        ps = pair.groupby(['cell', 'seed', 'port'])[VALUES].mean().reset_index()
        ps = ps.groupby(['cell', 'seed'])[VALUES].mean().reset_index()
        pdw = pair.groupby(['cell', 'draw', 'port'])[VALUES].mean().reset_index()
        pdw = pdw.groupby(['cell', 'draw'])[VALUES].mean().reset_index()
        pp = pair.groupby(['cell', 'port'])[VALUES].mean().reset_index()
        for frame, dest in [(ps, seed_rows), (pdw, draw_rows), (pp, port_rows)]:
            frame.insert(0, 'comparison', name)
            dest.extend(frame.to_dict('records'))
        for cell, g in ps.groupby('cell'):
            dg = pdw[pdw.cell.eq(cell)]
            row = dict(comparison=name, cell=cell, training_seeds=len(g),
                       positive_ba_seeds=int((g.ba > 1e-12).sum()), negative_ba_seeds=int((g.ba < -1e-12).sum()),
                       positive_ba_support_draws=int((dg.ba > 1e-12).sum()),
                       negative_ba_support_draws=int((dg.ba < -1e-12).sum()))
            for value in VALUES:
                row[value+'_mean'] = g[value].mean()
                row[value+'_seed_sd'] = g[value].std()
                np.testing.assert_allclose(g[value].mean(), dg[value].mean(), atol=1e-12)
            summary.append(row)
    for name, records in [('paired_summary', summary), ('paired_by_training_seed', seed_rows),
                          ('paired_by_support_draw', draw_rows), ('paired_by_port', port_rows)]:
        pd.DataFrame(records).to_csv(path/(name+'.csv'), index=False)
    events = pd.read_csv(path/'class_events.csv')
    np.testing.assert_allclose(events.recall-events.baseline_recall,
                               100*(events.rescues-events.harms)/events.query_n, atol=1e-10)
    keys = KEY+['anchor', 'objective', 'seed']
    reconstructed = events.groupby(keys).recall.mean().reset_index()
    checked = reconstructed.merge(full[keys+['ba']], on=keys, validate='one_to_one')
    np.testing.assert_allclose(checked.recall, checked.ba, atol=1e-10)
    events['ba_contribution'] = 100*(events.rescues-events.harms)/events.query_n
    present = events.groupby(keys).class_id.transform('size')
    events['ba_contribution'] /= present
    # A missing query class contributes zero; the denominator remains all tasks.
    group = ['cell', 'anchor', 'objective', 'seed', 'port']
    denominator = full.groupby(group).size().rename('tasks').reset_index()
    contributions = events.groupby(group+['class_id']).ba_contribution.sum().reset_index()
    contributions = contributions.merge(denominator, on=group, validate='many_to_one')
    contributions['ba_contribution'] /= contributions.tasks
    contributions.to_csv(path/'class_ba_contributions_by_port_seed.csv', index=False)
    (path/'SUMMARY_CHECKS.json').write_text(json.dumps(dict(
        class_event_rows=len(events), reconstructed_full_tasks=len(checked),
        max_ba_error=float(abs(checked.recall-checked.ba).max()),
        aggregation='draws/scenes within port then equal ports; seeds and support draws separately; all48',
        independent_training_replication=False), indent=2), encoding='utf-8')
    print(path.name)
    print(pd.DataFrame(summary).query("cell == 'other_acquisition'")[['comparison', 'oa_mean', 'ba_mean',
        'ba_seed_sd', 'positive_ba_seeds', 'positive_ba_support_draws']].to_string(index=False))


if __name__ == '__main__':
    for directory in [ROOT, ROOT/'matched_training']:
        summarize(directory)
