"""Describe absent-class errors and support coverage; never select a model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
SOURCE = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916')
OUT = ROOT/'strong_single_multidata_20261010'


def error_kind(prediction, truth, target_classes, support_classes):
    kinds = np.full(len(truth), 'wrong_supported_class', dtype='<U40')
    kinds[~np.isin(prediction, support_classes)] = 'wrong_class_without_support'
    kinds[~np.isin(prediction, target_classes)] = 'wrong_class_absent_from_target'
    kinds[prediction == truth] = 'correct'
    return kinds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seeds', nargs='+', type=int, default=[20261010, 20261011, 20261012])
    args = parser.parse_args()
    records = []
    for folder in sorted((args.out/'targets').glob('*/*')):
        if not all((folder/f'seed_{s}/COMPLETE.json').exists() for s in args.seeds):
            continue
        dataset, name = folder.parent.name, folder.name
        with np.load(ROOT/'backbone_ablation_20261010/targets'/dataset/name/'predictions.npz') as saved:
            data = {k: saved[k] for k in ('ids', 'rows', 'support', 'eval_masks')}
            dual = saved['predictions'][:, list(saved['modes']).index('dual_refit_late')]
        manifest = SOURCE/'manifest.csv' if dataset == 'self24ports' else ROOT/'f0921_detector_recrop_20261007/target_manifest.csv'
        idcol = 'instance_id' if dataset == 'self24ports' else 'sample_id'
        frame = pd.read_csv(manifest, usecols=[idcol, 'class_id'])
        np.testing.assert_array_equal(frame[idcol].to_numpy(str)[data['rows']], data['ids'])
        y = frame.class_id.to_numpy()[data['rows']]
        target_classes = np.unique(y)
        for seed in args.seeds:
            with np.load(folder/f'seed_{seed}/predictions.npz') as saved:
                for key in data:
                    np.testing.assert_array_equal(saved[key], data[key])
                single = saved['predictions'][:, list(saved['arms']).index('task_adapted_single_same_readout')]
                support_labels = saved['support_labels']
            for draw, mask in enumerate(data['eval_masks']):
                truth = y[mask]
                support_classes = np.unique(support_labels[draw])
                old, new = single[draw, mask], dual[draw, mask]
                old_kind = error_kind(old, truth, target_classes, support_classes)
                new_kind = error_kind(new, truth, target_classes, support_classes)
                present_classes = np.unique(truth)
                for c in present_classes:
                    cmask = truth == c
                    n = int(cmask.sum())
                    support_count = int((support_labels[draw] == c).sum())
                    for event, event_mask, kinds in [
                        ('dual_rescue', (old != truth)&(new == truth), old_kind),
                        ('dual_harm', (old == truth)&(new != truth), new_kind),
                    ]:
                        for kind in ('wrong_supported_class', 'wrong_class_without_support', 'wrong_class_absent_from_target'):
                            count = int((cmask & event_mask & (kinds == kind)).sum())
                            records.append(dict(dataset=dataset, task=name, seed=seed, draw=draw,
                                class_id=int(c), query_class_count=n, support_count=support_count,
                                true_class_supported=support_count > 0, event=event, error_kind=kind,
                                count=count, oa_contribution_pp=100*count/len(truth),
                                ba_contribution_pp=100*count/(len(present_classes)*n),
                                count_unit='query/draw/seed exposures, not independent samples'))
    if not records:
        raise SystemExit('No complete tasks for requested seeds')
    dest = args.out/'error_modes'
    dest.mkdir(exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(dest/'per_class.csv', index=False)
    fields = ['count', 'oa_contribution_pp', 'ba_contribution_pp']
    by_draw = table.groupby(['dataset', 'task', 'seed', 'draw', 'event', 'error_kind'])[fields].sum().reset_index()
    by_draw.to_csv(dest/'by_draw.csv', index=False)
    by_draw.groupby(['dataset', 'task', 'seed', 'event', 'error_kind'])[fields].mean().groupby(
        ['dataset', 'event', 'error_kind'])[fields].mean().to_csv(dest/'summary.csv')
    coverage = table.groupby(['dataset', 'task', 'seed', 'draw', 'event', 'true_class_supported'])[fields].sum().reset_index()
    # Missing groups contribute zero to task-macro effects, not a smaller task denominator.
    base = table[['dataset', 'task', 'seed', 'draw']].drop_duplicates()
    groups = pd.MultiIndex.from_product([['dual_rescue', 'dual_harm'], [False, True]],
                                       names=['event', 'true_class_supported']).to_frame(index=False)
    coverage = base.merge(groups, how='cross').merge(coverage, how='left').fillna({c: 0 for c in fields})
    coverage.groupby(['dataset', 'task', 'seed', 'event', 'true_class_supported'])[fields].mean().groupby(
        ['dataset', 'event', 'true_class_supported'])[fields].mean().to_csv(dest/'support_coverage.csv')
    # Event sums must agree with the direct per-query accuracy change, not a visual estimate.
    aggregate = by_draw.groupby(['dataset', 'task', 'seed', 'draw', 'event'])[fields[1:]].sum().unstack('event')
    expected = pd.read_csv(args.out/'metrics.csv')
    ref = pd.read_csv(ROOT/'backbone_ablation_20261010/metrics.csv')
    ref = ref[ref['mode'] == 'dual_refit_late'].set_index(['dataset', 'task', 'draw'])
    expected = expected[expected.arm == 'task_adapted_single_same_readout'].set_index(['dataset', 'task', 'seed', 'draw'])
    for key, row in aggregate.iterrows():
        reference = ref.loc[(key[0], key[1], key[3])]
        for metric in ('oa', 'ba'):
            net = row[(f'{metric}_contribution_pp', 'dual_rescue')]-row[(f'{metric}_contribution_pp', 'dual_harm')]
            np.testing.assert_allclose(net, reference[metric]-expected.loc[key, metric], atol=1e-10)
    (dest/'RECEIPT.json').write_text(json.dumps(dict(tasks=table.groupby('dataset').task.nunique().to_dict(),
        seeds=args.seeds, metric_accounting_exact=True, query_labels_used_only_for_posthoc_description=True,
        target_class_set_not_given_to_classifier=True,
        interpretation='Error categories and support coverage, not identified physical causes'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
