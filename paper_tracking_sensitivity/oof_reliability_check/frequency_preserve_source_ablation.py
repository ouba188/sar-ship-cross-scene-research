"""Allow versus disallow the new family, with both choices based on source scores only."""

import json
import pandas as pd
from frequency_preserve_experiment import OUT, configure_run


def main():
    modes = configure_run().MODES[:-1]
    metrics = pd.read_csv(OUT/'metrics.csv')
    rows = []
    for p in sorted((OUT/'source').glob('*/*/SELECTION.json')):
        s = json.loads(p.read_text())
        allowed = [m for m in modes if not m.startswith('preserved_')]
        without = max(allowed, key=lambda m: s['source_delta_ba'][m])
        table = metrics[(metrics.dataset == s['dataset']) & (metrics.task == s['task'])]
        old = table[table['mode'].eq(without)].set_index(['draw', 'stage'])
        new = table[table['mode'].eq('source_selected')].set_index(['draw', 'stage'])
        if not old.index.equals(new.index):
            raise ValueError('Source-selected ablation rows mismatch')
        for key, old_row in old.iterrows():
            new_row = new.loc[key]
            row = dict(dataset=s['dataset'], task=s['task'], draw=key[0], stage=key[1],
                rule_without=without, rule_with=s['rule'], enabled=s['rule'].startswith('preserved_'))
            for metric in ('oa', 'ba'):
                row['without_'+metric], row['with_'+metric] = old_row[metric], new_row[metric]
                row['delta_'+metric] = new_row[metric]-old_row[metric]
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'source_candidate_ablation.csv', index=False)
    fields = ['without_oa', 'with_oa', 'delta_oa', 'without_ba', 'with_ba', 'delta_ba']
    tasks = frame.groupby(['dataset', 'stage', 'task'])[fields].mean()
    summary = tasks.groupby(['dataset', 'stage']).mean()
    summary['enabled_tasks'] = frame.groupby(['dataset', 'stage', 'task']).enabled.first().groupby(['dataset', 'stage']).sum()
    summary.to_csv(OUT/'source_candidate_ablation_summary.csv')
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
