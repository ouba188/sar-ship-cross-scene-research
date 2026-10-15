"""Evaluate the frozen source-selected residual strength on existing tasks."""

import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
OLD = ROOT / 'classwise_utility_adaptation_20261012'
PATH = ROOT / 'residual_strength_path_20261013'
BUDGET = ROOT / 'frozen_candidate_coverage_20261012/budget'
sys.path.insert(0, str(PATH))
from run_path import fusion, measures, path_logits

KEY = ['dataset', 'task', 'draw', 'budget']
METRICS = ['oa', 'ba', 'macro_brier']
REFERENCES = ['raw', 'count', 'global', 'joint', 'simple', 'initial']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def load(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def folders():
    ports = [('self24ports', p.parent) for p in sorted(
        (BUDGET / 'tasks/self24ports').glob('*/PREDICTIONS_FROZEN.json'))]
    return ports + [(ds, BUDGET / 'strong_external' / ds / ds) for ds in ('oss2', 'f0921')]


def freeze():
    start = time.perf_counter()
    choice = read_json(PATH / 'FROZEN_ALPHA.json')
    assert choice['alphas'] == {'joint': .25, 'global': 1.}
    previous = read_json(OLD / 'TARGET_PREDICTIONS_FROZEN.json')
    cache = Path(previous['cache'])
    tasks = folders()
    assert len(tasks) == 25
    units = []
    for ds, directory in tasks:
        task = directory.name
        manifest = read_json(directory / 'PREDICTIONS_FROZEN.json')['target_manifest']
        for budget in (16, 64, 128):
            for draw in range(5):
                filename = f'{draw}_{budget}.npz'
                old_path = directory / filename
                z = load(old_path, ['base', 'expert_raw', 'support_ids', 'query_ids'])
                n = load(cache / ds / task / filename, ['joint', 'prior', 'support_ids', 'query_ids'])
                for key in ('support_ids', 'query_ids'):
                    np.testing.assert_array_equal(z[key], n[key])
                assert len(z['support_ids']) == budget
                assert np.intersect1d(z['query_ids'], z['support_ids']).size == 0
                gate = load(OLD / 'target_oof' / ds / task / filename, ['gate_joint'])['gate_joint']
                compute_start = time.perf_counter()
                score = path_logits(z['expert_raw'], n['joint'], choice['alphas']['joint'])
                p = fusion(z['base'], score)
                compute_seconds = time.perf_counter() - compute_start
                expected = n['prior'] + (1 + .25 * (gate - 1)) * (z['expert_raw'] - n['prior'])
                error = float(np.abs(score - expected).max())
                np.testing.assert_allclose(score, expected, atol=1e-10, rtol=0)
                prior_error = None
                if ds == 'oss2' and budget in (16, 128):
                    prior = load(PATH / 'predictions/external/oss2' / filename, ['joint_a0p25'])['joint_a0p25']
                    prior_error = float(np.abs(p - prior).max())
                    np.testing.assert_allclose(p, prior, atol=1e-12, rtol=0)
                    np.testing.assert_array_equal(p.argmax(1), prior.argmax(1))
                dest = OUT / 'predictions' / ds / task / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, joint25=p, support_ids=z['support_ids'], query_ids=z['query_ids'])
                units.append(dict(dataset=ds, task=task, draw=draw, budget=budget,
                    query_n=len(p), manifest=manifest, baseline_path=str(old_path),
                    prediction_path=str(dest), logit_path_error=error,
                    old_oss2_probability_error=prior_error, conversion_seconds=compute_seconds))
        print('FROZEN', ds, task, flush=True)
    write_json('PREDICTIONS_FROZEN.json', dict(units=units, alpha=choice['alphas'],
        source_choice_path=str(PATH / 'FROZEN_ALPHA.json'), query_labels_used=False,
        new_fits=0, new_training=0, total_seconds=time.perf_counter()-start,
        prediction_conversion_seconds=sum(u['conversion_seconds'] for u in units)))


def evaluate():
    start = time.perf_counter()
    frozen = read_json(OUT / 'PREDICTIONS_FROZEN.json')
    historical = pd.read_csv(OLD / 'target_metrics.csv')
    historical = historical[(historical.layer == 'fusion') & historical.arm.isin(REFERENCES)]
    assert not historical.duplicated(KEY + ['arm']).any()
    reference_index = historical.set_index(KEY + ['arm'])
    labels = {}
    rows, classes, replays = [], [], []
    for unit in frozen['units']:
        key = {k: unit[k] for k in KEY}
        if unit['manifest'] not in labels:
            labels[unit['manifest']] = pd.read_csv(unit['manifest'], usecols=['class_id']).class_id.to_numpy(int)
        z = load(unit['prediction_path'], ['joint25', 'query_ids', 'support_ids'])
        old = load(unit['baseline_path'], ['base', 'expert_raw', 'query_ids', 'support_ids', 'support_labels'])
        for k in ('query_ids', 'support_ids'):
            np.testing.assert_array_equal(z[k], old[k])
        y = labels[unit['manifest']][z['query_ids']]
        baseline = fusion(old['base'], old['expert_raw'])
        bm, _ = measures(y, baseline, baseline)
        ref = reference_index.loc[tuple(key[k] for k in KEY) + ('raw',)]
        errors = [abs(bm[m]-ref[m]) for m in METRICS]
        np.testing.assert_allclose([bm[m] for m in METRICS], ref[METRICS].to_numpy(float), atol=1e-10, rtol=0)
        replays.append(dict(**key, metric_error=max(errors)))
        mm, cc = measures(y, z['joint25'], baseline)
        rows.append(dict(**key, arm='joint25', query_n=len(y), query_classes=len(np.unique(y)), **mm))
        for c in cc:
            c['delta_ba'] = c['rescue_ba'] - c['harm_ba']
            c['support_n'] = int((old['support_labels'] == c['class_id']).sum())
            classes.append(dict(**key, arm='joint25', **c))
    current = pd.DataFrame(rows)
    current.to_csv(OUT / 'new_metrics.csv', index=False)
    class_table = pd.DataFrame(classes)
    class_table.to_csv(OUT / 'classes.csv', index=False)
    pd.DataFrame(replays).to_csv(OUT / 'baseline_replay.csv', index=False)
    np.testing.assert_allclose(class_table.groupby(KEY).delta_ba.sum().sort_index(),
                              current.set_index(KEY).delta_ba.sort_index(), atol=1e-10)
    full = pd.concat([historical[KEY+['arm']+METRICS], current[KEY+['arm']+METRICS]], ignore_index=True)
    full.to_csv(OUT / 'all_metrics.csv', index=False)
    per_task = full.groupby(['dataset', 'task', 'budget', 'arm'])[METRICS].mean().reset_index()
    per_task.to_csv(OUT / 'task_summary.csv', index=False)
    summary = per_task.groupby(['dataset', 'budget', 'arm'])[METRICS].mean().reset_index()
    summary.to_csv(OUT / 'summary.csv', index=False)

    comparisons = []
    for arm in REFERENCES:
        ref = historical[historical.arm == arm]
        paired = current[KEY+METRICS].merge(ref[KEY+METRICS], on=KEY,
            validate='one_to_one', suffixes=('_new', '_reference'))
        assert len(paired) == len(current)
        values = paired[KEY].copy()
        values['reference'] = arm
        for m in METRICS:
            values['delta_'+m] = paired[m+'_new']-paired[m+'_reference']
        comparisons.append(values)
    paired = pd.concat(comparisons, ignore_index=True)
    paired.to_csv(OUT / 'paired.csv', index=False)
    delta = ['delta_'+m for m in METRICS]
    task_pairs = paired.groupby(['dataset', 'task', 'budget', 'reference'])[delta].mean().reset_index()
    task_pairs.to_csv(OUT / 'task_paired.csv', index=False)
    details = []
    for (ds, budget, ref), g in paired.groupby(['dataset', 'budget', 'reference']):
        task_means = g.groupby('task')[delta].mean()
        row = dict(dataset=ds, budget=int(budget), reference=ref, tasks=len(task_means), units=len(g))
        row.update(task_means.mean().to_dict())
        row.update(task_ba_positive=int((task_means.delta_ba>1e-10).sum()),
            task_ba_tie=int((abs(task_means.delta_ba)<=1e-10).sum()),
            task_ba_negative=int((task_means.delta_ba < -1e-10).sum()),
            draw_ba_positive=int((g.delta_ba>1e-10).sum()),
            draw_ba_tie=int((abs(g.delta_ba)<=1e-10).sum()),
            draw_ba_negative=int((g.delta_ba < -1e-10).sum()),
            task_ba_sd=float(task_means.delta_ba.std()) if len(task_means)>1 else None,
            paired_draw_ba_sd=float(g.delta_ba.std()),
            mean_within_task_draw_sd=float(g.groupby('task').delta_ba.std().mean()))
        details.append(row)
    compared = pd.DataFrame(details)
    compared.to_csv(OUT / 'paired_summary.csv', index=False)
    task_pairs[(task_pairs.dataset == 'self24ports') & (task_pairs.task != 'Mombasa')].groupby(
        ['budget', 'reference'])[delta].mean().reset_index().to_csv(OUT / 'without_mombasa.csv', index=False)
    class_table.groupby(['dataset','task','budget','class_id'])[['recall','delta_ba','rescue_ba','harm_ba','support_n']].mean().reset_index().to_csv(OUT/'class_task_summary.csv',index=False)
    current.groupby(['dataset','task','budget'])[['rescue_ba','harm_ba','risk_gain_unchanged','risk_gain_changed']].mean().groupby(['dataset','budget']).mean().to_csv(OUT/'mechanism_summary.csv')
    write_json('EVALUATION_COMPLETE.json', dict(units=len(current), tasks=len(folders()),
        new_probability_units=len(current), previously_evaluated_joint25_units=10,
        new_joint25_evaluation_units=len(current)-10, raw_replay_rows=len(replays),
        max_raw_metric_error=max(r['metric_error'] for r in replays),
        seconds=time.perf_counter()-start, new_fits=0, new_training=0, main_model_changed=False))
    print(compared[compared.reference.isin(['raw','count','global'])].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'evaluate', 'all'])
    args = parser.parse_args()
    if args.stage in ('freeze', 'all'):
        freeze()
    if args.stage in ('evaluate', 'all'):
        evaluate()
