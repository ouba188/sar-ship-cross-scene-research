"""Matched-scope descriptive summaries; no model or target-based selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
OLD = {'Los Angeles', 'Port Klang', 'Santos'}
MEASURES = ['oa', 'ba', 'nll', 'brier', 'class_nll', 'class_brier']
GAINS = ['delta_ba', 'delta_oa', 'rescue_ba', 'harm_ba', 'brier_gain', 'class_brier_gain']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def family(arm):
    return arm.rsplit('_', 1)[0] if arm.rsplit('_', 1)[-1].isdigit() else arm


def scopes(frame, eligible):
    ports = frame.dataset.eq('self24ports')
    learned = frame.task.isin(eligible)
    return {
        'ports14_matched': ports & learned,
        'ports_old3': ports & frame.task.isin(OLD),
        'ports_new11': ports & learned & ~frame.task.isin(OLD),
        'ports23_baselines': ports & frame.family.isin(['raw', 'initial', 'simple']),
        'oss2_strong': frame.dataset.eq('oss2'),
        'f0921_strong': frame.dataset.eq('f0921'),
        'tasks15_mixed_descriptive': learned,
        'ports13_without_tiny_Mombasa': ports & learned & ~frame.task.eq('Mombasa'),
    }


def aggregate(frame, masks, measures, paired=False):
    outputs, by_seed = [], []
    keys = ['budget', 'layer', 'family'] + (['reference_family'] if paired else [])
    for scope, mask in masks.items():
        part = frame[mask & ~frame.family.eq('reconstruction')].copy()
        if part.empty:
            continue
        # Average the two training seeds within each SAME task/draw before port means.
        draws = part.groupby(['dataset', 'task', 'draw']+keys)[measures].mean().reset_index()
        tasks = draws.groupby(['dataset', 'task']+keys)[measures].mean().reset_index()
        for key, group in tasks.groupby(keys):
            row = dict(scope=scope, **dict(zip(keys, key)), tasks=len(group))
            row.update(group[measures].mean().to_dict())
            if paired:
                row.update(positive_tasks=int((group.delta_ba > 1e-10).sum()),
                           negative_tasks=int((group.delta_ba < -1e-10).sum()),
                           tie_tasks=int((abs(group.delta_ba) <= 1e-10).sum()),
                           task_sd_ba=float(group.delta_ba.std()))
            else:
                selected = draws
                for col, value in zip(keys, key):
                    selected = selected[selected[col].eq(value)]
                row['mean_within_task_draw_sd_ba'] = float(selected.groupby(['dataset', 'task']).ba.std().mean())
            outputs.append(row)
        seeded = part.groupby(['arm']+keys)[measures].mean().reset_index()
        seeded.insert(0, 'scope', scope)
        by_seed.append(seeded)
    return pd.DataFrame(outputs), pd.concat(by_seed, ignore_index=True)


def external_versions():
    from run_coverage import archive, probability_rows
    rows = []
    for version, root in [('historical_pre_tail', 'tasks'), ('matched_strong_post_tail', 'strong_external')]:
        for ds in ('oss2', 'f0921'):
            folder = OUT/root/ds/ds
            receipt = read(folder/'PREDICTIONS_FROZEN.json')
            meta = pd.read_csv(receipt['target_manifest'])
            for budget in (16, 64, 128):
                for draw in range(5):
                    z = archive(folder/f'{draw}_{budget}.npz')
                    y = meta.iloc[z['query_ids']].class_id.to_numpy(int)
                    _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
                    w = 1.0/(len(counts)*counts[inv])
                    for layer in ('expert', 'fusion'):
                        for arm in ('raw', 'initial', 'simple'):
                            p = probability_rows(z[layer+'_'+arm], y, 'logits' if layer == 'expert' else 'probabilities')
                            rows.append(dict(dataset=ds, version=version, budget=budget, draw=draw,
                                layer=layer, arm=arm, oa=100*p['correct'].mean(), ba=100*w@p['correct'],
                                class_brier=float(w@p['brier']), query_n=len(y)))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'external_version_metrics.csv', index=False)
    frame.groupby(['dataset', 'version', 'budget', 'layer', 'arm'])[['oa', 'ba', 'class_brier']].mean().reset_index().to_csv(
        OUT/'external_version_summary.csv', index=False)


def scope_and_cost():
    audit = pd.read_csv(OUT/'coverage_audit.csv')
    rows = []
    for _, r in audit.iterrows():
        ds, task = r.dataset, r.task
        baseline = OUT/('tasks' if ds == 'self24ports' else 'strong_external')/ds/task/'PREDICTIONS_FROZEN.json'
        train = OUT/'training'/ds/task
        receipt = train/'MODELS_FROZEN.json'
        pred = OUT/'trained_predictions'/ds/task/'PREDICTIONS_FROZEN.json'
        old = ds == 'self24ports' and task in OLD
        row = dict(dataset=ds, task=task, common_query_n=r.common_query_n,
            fold0_ships=r.fold0_ships, fold1_ships=r.fold1_ships, fold2_ships=r.fold2_ships,
            baselines_complete=baseline.exists(), shared_isolated_complete=old or pred.exists(),
            reconstruction='historical_diagnostic_only' if old else 'not_expanded',
            weight_policy='own_task_historical_weights' if old else ('frozen_recipe_refit' if receipt.exists() else 'not_available'),
            baseline_version='unchanged_port' if ds == 'self24ports' else 'matched_strong_post_tail',
            new_affine_fits=0, new_optimizer_steps=0, source_prep_seconds=0., gpu_encode_seconds=0., fit_seconds=0.)
        if receipt.exists():
            models = read(receipt)
            cost = pd.read_csv(train/'training_cost.csv')
            bridge = read(train/'SOURCE_COHORT_BRIDGE.json')
            prep = read(train/'SOURCE_PREPARED.json')
            row.update(new_affine_fits=models['fits'], new_optimizer_steps=int(cost.steps.sum()),
                source_original_rows=bridge['original_fit_rows'], source_retained_rows=bridge['fit_rows'],
                source_removed_rows=bridge['removed_rows'], source_cohort_changed=bool(bridge['removed_rows']),
                source_prep_seconds=prep['seconds'], source_head_fits=prep['source_heads_refitted'],
                gpu_encode_seconds=read(train/'FREQUENCY_READY.json')['seconds'], fit_seconds=float(cost.seconds.sum()))
        row['remaining_boundary'] = ('' if row['shared_isolated_complete'] else
            ('No complete frequency/base task operator' if ds == 'oss1' else 'Original fold <160 vessels; frozen sampling unchanged'))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'coverage_final.csv', index=False)
    return result


def main():
    frame, pairs = pd.read_csv(OUT/'metrics.csv'), pd.read_csv(OUT/'paired.csv')
    eligible = set(frame.loc[frame.arm.eq('task_31415'), 'task'])
    assert len(eligible) == 15, eligible
    for data in (frame, pairs):
        data['family'] = data.arm.map(family)
    pairs['reference_family'] = pairs.reference.map(family)
    # Check matched seed/B/draw coverage; never compare learned means with all23 baselines.
    expected = ['task_31415', 'task_27182', 'isolated_31415', 'isolated_27182', 'raw', 'initial', 'simple']
    complete = frame[frame.task.isin(eligible) & frame.arm.isin(expected)]
    for _, group in complete.groupby(['dataset', 'task', 'budget', 'draw', 'layer']):
        assert set(group.arm) == set(expected)
    summary, seeded = aggregate(frame, scopes(frame, eligible), MEASURES)
    paired, paired_seeded = aggregate(pairs, scopes(pairs, eligible), GAINS, paired=True)
    for name, data in [('matched_summary', summary), ('matched_by_seed', seeded),
                       ('matched_paired', paired), ('matched_paired_by_seed', paired_seeded)]:
        data.to_csv(OUT/(name+'.csv'), index=False)
    frame[frame.family.eq('reconstruction')].to_csv(OUT/'historical_reconstruction_diagnostic.csv', index=False)
    coverage = scope_and_cost()
    print(summary[summary.layer.eq('fusion') & summary.scope.isin(['ports14_matched', 'ports_new11', 'oss2_strong', 'f0921_strong'])][
        ['scope', 'budget', 'family', 'tasks', 'oa', 'ba']].to_string(index=False), flush=True)
    print(paired[paired.layer.eq('fusion') & paired.reference_family.eq('simple') & paired.scope.isin([
        'ports14_matched', 'ports_new11', 'oss2_strong'])][['scope', 'budget', 'family', 'delta_oa', 'delta_ba',
        'class_brier_gain', 'positive_tasks', 'negative_tasks']].to_string(index=False), flush=True)
    print('NEW TRAINING COST', coverage[['new_affine_fits', 'new_optimizer_steps', 'source_prep_seconds', 'gpu_encode_seconds', 'fit_seconds']].sum().to_dict(), flush=True)
    external_versions()


if __name__ == '__main__':
    main()
