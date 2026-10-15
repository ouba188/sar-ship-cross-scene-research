"""Reuse the same fixed prior/kernel intervention on every completed task."""

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from strong_single_multidata import OUT, SEEDS
from strong_single_r50_task import ROOT, SOURCE, write_json
from strong_single_r50_task_diagnose import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = parser.parse_args()
    tables, effects = [], []
    for folder in sorted((args.out/'targets').glob('*/*')):
        if not all((folder/f'seed_{s}/COMPLETE.json').exists() for s in args.seeds):
            continue
        dataset, task = folder.parent.name, folder.name
        dest = folder/'diagnosis'
        receipt = dest/'COMPLETE.json'
        if not receipt.exists() or json.loads(receipt.read_text())['seeds'] != args.seeds:
            manifest = SOURCE/'manifest.csv' if dataset == 'self24ports' else ROOT/'f0921_detector_recrop_20261007/target_manifest.csv'
            print(f'DIAGNOSE {dataset}/{task}', flush=True)
            run(folder, args.seeds,
                reference=ROOT/'backbone_ablation_20261010/targets'/dataset/task/'predictions.npz',
                target_manifest=manifest, id_column='instance_id' if dataset == 'self24ports' else 'sample_id')
            gc.collect()
        for filename, collection in [('neural_seed_means.csv', tables), ('interactions_by_seed_draw.csv', effects)]:
            frame = pd.read_csv(dest/filename)
            frame['dataset'], frame['task'] = dataset, task
            collection.append(frame)
    if not tables:
        return
    dest = args.out/'crossover'
    dest.mkdir(exist_ok=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(dest/'task_seed_means.csv', index=False)
    neural = table.groupby(['dataset', 'seed', 'mode'])[['oa', 'ba']].mean().reset_index()
    neural.to_csv(dest/'neural_seed_means.csv', index=False)
    neural.groupby(['dataset', 'mode'])[['oa', 'ba']].agg(['mean', 'std']).to_csv(dest/'summary.csv')
    effects = pd.concat(effects, ignore_index=True)
    effects.to_csv(dest/'interactions.csv', index=False)
    columns = [c for c in effects if c.endswith('_pp')]
    effects.groupby(['dataset', 'task', 'seed', 'metric'])[columns].mean().groupby(['dataset', 'metric']).mean().to_csv(dest/'effects.csv')
    write_json(dest/'STATUS.json', dict(tasks=table.groupby('dataset').task.nunique().to_dict(),
        seeds=args.seeds, all_24_tasks_complete=table.task.nunique() == 24,
        diagonal_replay_exact=True, cross_cells_diagnostic_only=True))


if __name__ == '__main__':
    main()
