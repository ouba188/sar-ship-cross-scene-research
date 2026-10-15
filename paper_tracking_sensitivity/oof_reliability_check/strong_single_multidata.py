"""Extend the frozen OSS2 task-R50 recipe to all held ports and F_0921."""

import argparse
from contextlib import redirect_stdout
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import strong_single_r50_task as task

OUT = task.ROOT/'strong_single_multidata_20261010'
TASKS = task.ROOT/'learned_frequency_residual_20261009/tasks'
SEEDS = [20261010, 20261011, 20261012]


def folders(datasets):
    result = []
    if 'self24ports' in datasets:
        result.extend(sorted((TASKS/'self24ports').glob('*/PREPARED.json')))
    if 'f0921' in datasets:
        result.append(TASKS/'f0921/PREPARED.json')
    return [p.parent for p in result]


def report(out):
    parts = []
    for receipt in sorted((out/'targets').glob('*/*/seed_*/COMPLETE.json')):
        info = json.loads(receipt.read_text())
        part = pd.read_csv(receipt.parent/'metrics.csv')
        part['dataset'], part['task'] = info['dataset'], info['task']
        parts.append(part)
    if not parts:
        return
    table = pd.concat(parts, ignore_index=True)
    table.to_csv(out/'metrics.csv', index=False)
    means = table.groupby(['dataset', 'task', 'seed', 'arm'])[['oa', 'ba']].mean().reset_index()
    means.to_csv(out/'task_seed_means.csv', index=False)
    # Equal port weight, then neural seeds; draws never become neural replicates.
    neural = means.groupby(['dataset', 'seed', 'arm'])[['oa', 'ba']].mean().reset_index()
    neural.to_csv(out/'neural_seed_means.csv', index=False)
    neural.groupby(['dataset', 'arm'])[['oa', 'ba']].agg(['mean', 'std']).to_csv(out/'summary.csv')
    task.write_json(out/'STATUS.json', dict(completed_task_seeds=len(means)//2,
        tasks_per_dataset=means.groupby('dataset').task.nunique().to_dict(),
        expected_task_seeds=72, status='complete' if len(means)//2 == 72 else 'partial',
        interpretation='Do not compare a partial-port mean against the complete 23-port reference'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--datasets', nargs='+', choices=['self24ports', 'f0921'], default=['self24ports', 'f0921'])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--only-task')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(args.out)
        return
    selected = folders(args.datasets)
    if args.only_task:
        selected = [p for p in selected if p.name == args.only_task]
        if not selected:
            parser.error('Unknown task')
    protocol = args.out/'PROTOCOL.json'
    if not protocol.exists():
        task.write_json(protocol, dict(date='2026-10-10', scope=['23 eligible held-port tasks', 'F_0921'],
            neural_seeds=SEEDS, support_draws=5, support_labels=64, epochs=6, episodes_per_epoch=32,
            recipe='Unchanged OSS2 task-R50 training objective, source selection and refit',
            source_only_selection=True, target_selected_configuration=False,
            dual_reference='Existing matched-source-refit current dual; not retrained for each neural seed',
            limitations=['Tail adaptation, not full-network optimization',
                        'Compact readout, not full-module parity',
                        'No equalized SAR pretraining; architecture necessity not established'],
            diagnostics=['Per-class and per-port rescues AND harms',
                         'Neural-seed consistency distinct from support draws',
                         'Prior/support-relation interventions if dual remains ahead']))
    for seed in args.seeds:
        for folder in selected:
            info = json.loads((folder/'PREPARED.json').read_text())
            dest = args.out/'targets'/info['dataset']/info['task']
            run = dest/f'seed_{seed}'
            if (run/'COMPLETE.json').exists():
                continue
            run.mkdir(parents=True, exist_ok=True)
            job = argparse.Namespace(out=dest, task=folder, epochs=6, episodes=32)
            started = time.perf_counter()
            print(f'START {info["dataset"]}/{info["task"]} seed={seed}', flush=True)
            with (run/'run.log').open('a', encoding='utf-8', buffering=1) as log, redirect_stdout(log):
                if not (run/'TRAINING_COMPLETE.json').exists():
                    task.train(job, seed)
                task.evaluate(job, seed)
            gc.collect()
            torch.cuda.empty_cache()
            report(args.out)
            print(f'DONE {info["dataset"]}/{info["task"]} seed={seed} seconds={time.perf_counter()-started:.1f}', flush=True)
    report(args.out)


if __name__ == '__main__':
    main()
