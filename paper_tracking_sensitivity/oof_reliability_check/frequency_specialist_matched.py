"""Descriptive matched differences from completed report metrics, in percentage points.

Summary averages five draws within task, then tasks equally. Positive draw counts
refer to dataset means within each draw; positive task-draw counts are also saved.
These repeated draws are not independent replications or significance tests.
"""

import json
from pathlib import Path

import pandas as pd

from frequency_specialist_report import OUT, EXP, STAGES

SUFFIXES = ('equal', 'LL', 'LH', 'HL', 'HH', 'support_static', 'support_local')
COMPARISONS = (('specialist', 'independent'), ('independent', 'continued'),
               ('specialist', 'continued'))
VALUES = ('oa', 'ba', 'macro_f1_8')
INDEX = ['dataset', 'task', 'draw', 'stage']
GROUP = ['dataset', 'stage', 'suffix', 'comparison']
DELTAS = ['delta_'+v for v in VALUES]


def report(output=OUT):
    output = Path(output)
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    infos = [json.loads((p/'PREPARED.json').read_text()) for p in folders]
    expected = {(d['dataset'], d['task']) for d in infos}
    markers = {(p.parent.parent.name, p.parent.name)
               for p in (output/'targets').glob('*/*/COMPLETE.json')}
    if len(folders) != 25 or len(expected) != 25 or markers != expected:
        raise ValueError('Freeze all25 matching target tasks before reading report metrics')
    if not (output/'REPORT_COMPLETE.json').is_file():
        raise ValueError('Finish frequency_specialist_report before matched comparisons')
    metrics = pd.read_csv(output/'metrics.csv', dtype={'dataset': str, 'task': str})
    expected_index = pd.MultiIndex.from_tuples(
        [(dataset, task, draw, stage) for dataset, task in sorted(expected)
         for draw in range(5) for stage, _ in STAGES], names=INDEX)
    frames = {}
    for family in ('continued', 'independent', 'specialist'):
        for suffix in SUFFIXES:
            mode = f'{family}_{suffix}'
            frame = metrics.loc[metrics['mode'].eq(mode), INDEX+['n', *VALUES]].set_index(
                INDEX, verify_integrity=True).sort_index()
            if (len(frame) != len(expected_index) or not expected_index.isin(frame.index).all()
                    or frame.isna().any().any()):
                raise ValueError(f'Incomplete task/draw/stage metrics for {mode}')
            frames[mode] = frame
    records = []
    for suffix in SUFFIXES:
        for left, right in COMPARISONS:
            a, b = frames[f'{left}_{suffix}'], frames[f'{right}_{suffix}']
            if not a['n'].equals(b['n']):
                raise ValueError(f'Unmatched evaluation counts: {left}-{right}/{suffix}')
            delta = (a[list(VALUES)]-b[list(VALUES)]).rename(columns=dict(zip(VALUES, DELTAS)))
            records.append(delta.reset_index().assign(suffix=suffix, comparison=f'{left}-{right}'))
    paired = pd.concat(records, ignore_index=True)
    task = paired.groupby([*GROUP, 'task'])[DELTAS].mean()
    summary = task.groupby(GROUP)[DELTAS].mean()
    summary['tasks'] = task.groupby(GROUP).size()
    summary['task_draws'] = paired.groupby(GROUP).size()
    draws = paired.groupby([*GROUP, 'draw'])[DELTAS].mean()
    draws['tasks'] = paired.groupby([*GROUP, 'draw']).size()
    for value in VALUES:
        positive = 'positive_'+value
        paired[positive] = paired['delta_'+value] > 0
        draws['positive_tasks_'+value] = paired.groupby([*GROUP, 'draw'])[positive].sum()
        summary['positive_task_draws_'+value] = paired.groupby(GROUP)[positive].sum()
        summary['positive_draws_'+value] = (draws['delta_'+value] > 0).groupby(level=GROUP).sum()
    summary['draws'] = draws.groupby(level=GROUP).size()
    summary.reset_index().to_csv(output/'matched_summary.csv', index=False)
    draws.reset_index().to_csv(output/'matched_draw_metrics.csv', index=False)
    print(f'MATCHED: {len(summary)} comparisons; {len(draws)} dataset-draw rows', flush=True)


if __name__ == '__main__':
    report()
