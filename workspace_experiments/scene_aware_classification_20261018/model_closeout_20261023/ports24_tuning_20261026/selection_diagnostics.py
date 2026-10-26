"""Diagnose source-selection sensitivity; never choose using target outcomes."""
import json
import numpy as np
import pandas as pd
from tune import HERE, PREV, SEEDS, CONFIGS, LAMBDAS, load, write, op


def main():
    selected = json.loads((HERE/'selected.json').read_text(encoding='utf-8'))
    table = pd.read_csv(PREV/'bundle/table.csv')
    ids = np.flatnonzero(table.role.eq('selection'))
    f = table.iloc[ids].reset_index(drop=True)
    y = f.class_id.to_numpy(int)
    configs, correctness = [], []
    for order, config in enumerate(CONFIGS):
        zs = [load(HERE/'training'/config['name']/f'seed{seed}/source.npz') for seed in SEEDS]
        for lam in LAMBDAS:
            configs.append((config['name'], lam, order))
            correctness.append(np.mean([(z['bare'][ids]+lam*(z['frequency'][ids]-z['bare'][ids])).argmax(1) == y for z in zs], 0))
    correctness = np.asarray(correctness)
    jackknife = []
    for group, removed in f.groupby('pass_group').indices.items():
        keep = np.ones(len(f), bool)
        keep[removed] = False
        oa = correctness[:, keep].mean(1)
        ba = np.mean([correctness[:, keep & (y == c)].mean(1) for c in np.unique(y[keep])], 0)
        j = .5*(oa+ba)
        best = min(range(len(configs)), key=lambda k: (-j[k], abs(configs[k][1]-1), configs[k][2], configs[k][1]))
        name, lam, _ = configs[best]
        jackknife.append(dict(removed_pass=group, removed_n=len(removed), remaining_classes=len(np.unique(y[keep])),
            selected_config=name, correction_scale=lam, source_J=j[best]))
    pd.DataFrame(jackknife).to_csv(HERE/'source_selection_jackknife.csv', index=False)
    primary = selected['primary']
    count = sum(r['selected_config'] == primary['config'] and r['correction_scale'] == primary['correction_scale'] for r in jackknife)
    source_groups = pd.read_csv(HERE/'source_by_pass.csv')
    selected_groups = source_groups[(source_groups.config == primary['config']) &
                                    (source_groups.correction_scale == primary['correction_scale'])]
    baseline = source_groups[(source_groups.config == 'default') & (source_groups.correction_scale == 1)]
    merged = selected_groups.merge(baseline, on=['pass_group', 'seed'], suffixes=('_selected', '_default'))
    merged['OA_gain_pp'] = 100*(merged.OA_selected-merged.OA_default)
    merged['BA_gain_pp'] = 100*(merged.BA_selected-merged.BA_default)
    merged.to_csv(HERE/'selected_source_pass_changes.csv', index=False)
    write(HERE/'selection_diagnostics.json', dict(primary=primary, stable_leave_one_pass=count,
        total_passes=len(jackknife), training_rerun=False, selection_changed=False,
        interpretation='Leave-one-source-selection-pass sensitivity of frozen scores, NOT nested CV or independent validation.',
        source_pass_query_counts=f.groupby('pass_group').size().to_dict()))


if __name__ == '__main__':
    main()
