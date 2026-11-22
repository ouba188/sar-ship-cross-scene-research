"""Summarize existing OSS2 results; never trains or evaluates a model.

Default validation collection permits missing arms/seeds and reports their absence.
Explicit --split test requires all nine result files before opening any of them.
Public summaries contain no sample IDs. Paired predictions remain local/private.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


SEEDS = (17, 29, 43)
ARMS = ('baseline', 'disentangle', 'full_h')
STAGES = {'baseline': 'baseline_base', 'disentangle': 'disentangle_base', 'full_h': 'full_h'}
COMPARISONS = (('disentangle', 'baseline'), ('full_h', 'disentangle'), ('full_h', 'baseline'))
METRICS = {'OA': 'accuracy', 'BA': 'balanced_accuracy', 'macroF1': 'macro_f1'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def result_path(root, seed, arm, split):
    directory = root / 'runs' / f'seed{seed}'
    candidates = [directory / f'{arm}_{split}.json']
    if split == 'validation' and STAGES[arm] != arm:
        candidates.append(directory / f'{STAGES[arm]}_validation.json')
    present = [path for path in candidates if path.is_file()]
    if len(present) > 1:
        raise ValueError(f'Ambiguous evaluation files for seed{seed}/{arm}/{split}')
    return present[0] if present else None


def load_result(path, classes, expected_count, label_to_class=None):
    payload = read_json(path)
    rows = payload.get('predictions')
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{path.name}: missing nonempty predictions')
    by_id = {}
    for row in rows:
        if not all(isinstance(row.get(key), str) and row[key] for key in ('id', 'label', 'prediction')):
            raise ValueError(f'{path.name}: malformed prediction row')
        if row['id'] in by_id:
            raise ValueError(f'{path.name}: duplicate sample ID')
        if label_to_class is not None:
            if row['label'] not in label_to_class or row['prediction'] not in label_to_class:
                raise ValueError(f'{path.name}: label missing from audited manifest mapping')
            row = dict(row, label=label_to_class[row['label']], prediction=label_to_class[row['prediction']])
        if row['label'] not in classes or row['prediction'] not in classes:
            raise ValueError(f'{path.name}: class outside audited label set')
        by_id[row['id']] = row
    rows = list(by_id.values())
    if len(rows) != expected_count or payload.get('count') != expected_count:
        raise ValueError(f'{path.name}: sample count differs from preflight audit')
    support = {label: sum(row['label'] == label for row in rows) for label in classes}
    predicted = {label: sum(row['prediction'] == label for row in rows) for label in classes}
    correct = {label: sum(row['label'] == row['prediction'] == label for row in rows) for label in classes}
    recall = {label: correct[label] / support[label] if support[label] else None for label in classes}
    f1 = [2 * correct[label] / (support[label] + predicted[label])
          if support[label] + predicted[label] else 0.0 for label in classes]
    metrics = dict(OA=sum(correct.values()) / len(rows),
                   BA=statistics.mean(value for value in recall.values() if value is not None),
                   macroF1=statistics.mean(f1))
    for metric, source in METRICS.items():
        recorded = payload.get(source)
        if not isinstance(recorded, (int, float)) or not math.isfinite(recorded):
            raise ValueError(f'{path.name}: missing/invalid recorded {source}')
        if not math.isclose(metrics[metric], recorded, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError(f'{path.name}: recomputed {source} disagrees with recorded metric')
    return dict(count=len(rows), metrics=metrics, per_class_recall=recall,
                per_class_support=support), by_id


def describe(values):
    values = [value for value in values if value is not None]
    return dict(n_seeds=len(values), mean=statistics.mean(values) if values else None,
                seed_sd=statistics.stdev(values) if len(values) > 1 else None)


def collect(root, split):
    paths = {(seed, arm): result_path(root, seed, arm, split) for seed in SEEDS for arm in ARMS}
    missing = [dict(seed=seed, arm=arm) for (seed, arm), path in paths.items() if path is None]
    if split == 'test' and missing:
        raise ValueError('Test collection remains closed: all nine completed result files are required')
    preflight = read_json(root / 'preflight_summary.json')
    class_counts = preflight['split_class_counts']['val' if split == 'validation' else 'test']
    classes = sorted(class_counts)
    expected_count = preflight['split_counts'][split]
    manifest = read_json(root / 'experiment_manifest_local.json')
    manifest = manifest if isinstance(manifest, list) else manifest['records']
    label_to_class = {}
    expected_labels = {}
    split_names = ('val', 'validation') if split == 'validation' else ('test',)
    for row in manifest:
        label, class_name = str(row['label']), row['class_name']
        if label in label_to_class and label_to_class[label] != class_name:
            raise ValueError('Manifest label-to-class mapping is inconsistent')
        label_to_class[label] = class_name
        if row.get('development_split') in split_names:
            if row['id'] in expected_labels:
                raise ValueError('Audited split contains duplicate sample IDs')
            expected_labels[row['id']] = class_name
    if set(label_to_class.values()) != set(classes) or len(label_to_class) != len(classes):
        raise ValueError('Manifest label-to-class mapping disagrees with preflight classes')
    if len(expected_labels) != expected_count:
        raise ValueError('Audited manifest split count disagrees with preflight')
    results, predictions = {}, {}
    anchor_labels = None
    for key, path in paths.items():
        if path is None:
            continue
        result, rows = load_result(path, classes, expected_count, label_to_class)
        labels = {sample_id: row['label'] for sample_id, row in rows.items()}
        if labels != expected_labels:
            raise ValueError('Prediction IDs or true labels differ from the audited manifest split')
        if anchor_labels is not None and labels != anchor_labels:
            raise ValueError('Prediction IDs or true labels differ across arms/seeds; no paired summary written')
        if result['per_class_support'] != class_counts:
            raise ValueError('Per-class support differs from preflight audit')
        anchor_labels = labels
        results[key], predictions[key] = result, rows
    per_seed = []
    pairs = []
    private = []
    for seed in SEEDS:
        available = {arm: results[seed, arm] for arm in ARMS if (seed, arm) in results}
        per_seed.append(dict(seed=seed, arms=available))
        for later, earlier in COMPARISONS:
            if later not in available or earlier not in available:
                continue
            left, right = predictions[seed, earlier], predictions[seed, later]
            counts = dict(both_correct=0, rescued=0, harmed=0, both_wrong=0)
            for sample_id in left:
                old = left[sample_id]['prediction'] == left[sample_id]['label']
                new = right[sample_id]['prediction'] == right[sample_id]['label']
                counts['both_correct' if old and new else 'harmed' if old else 'rescued' if new else 'both_wrong'] += 1
            delta = {metric: 100 * (available[later]['metrics'][metric] - available[earlier]['metrics'][metric])
                     for metric in METRICS}
            recall_delta = {label: 100 * (available[later]['per_class_recall'][label] - available[earlier]['per_class_recall'][label])
                            if available[later]['per_class_recall'][label] is not None else None for label in classes}
            pairs.append(dict(seed=seed, comparison=f'{later}-minus-{earlier}',
                              delta_percentage_points=delta, per_class_recall_delta_pp=recall_delta,
                              paired_outcomes=counts))
        if available:
            first = predictions[seed, next(iter(available))]
            for sample_id in sorted(first):
                private.append(dict(seed=seed, split=split, id=sample_id, label=first[sample_id]['label'],
                                    **{arm: predictions[seed, arm][sample_id]['prediction'] if arm in available else ''
                                       for arm in ARMS}))
    aggregate = {}
    for arm in ARMS:
        entries = [results[seed, arm] for seed in SEEDS if (seed, arm) in results]
        aggregate[arm] = dict(metrics={metric: describe([entry['metrics'][metric] for entry in entries]) for metric in METRICS},
                              per_class_recall={label: describe([entry['per_class_recall'][label] for entry in entries]) for label in classes})
    paired_aggregate = {}
    for later, earlier in COMPARISONS:
        comparison = f'{later}-minus-{earlier}'
        entries = [pair for pair in pairs if pair['comparison'] == comparison]
        paired_aggregate[comparison] = dict(
            delta_percentage_points={metric: describe([entry['delta_percentage_points'][metric] for entry in entries]) for metric in METRICS},
            per_class_recall_delta_pp={label: describe([entry['per_class_recall_delta_pp'][label] for entry in entries]) for label in classes})
    summary = dict(split=split, status='complete' if not missing else 'partial' if results else 'waiting',
                   expected_seeds=list(SEEDS), missing=missing, loaded_results=len(results),
                   expected_sample_count=expected_count, classes=classes, label_to_class=label_to_class,
                   pairing_verified=bool(results), per_seed=per_seed, aggregate=aggregate,
                   per_seed_paired=pairs, paired_aggregate=paired_aggregate,
                   interpretation='Metrics/recalls are fractions; deltas are percentage points. Seed SD uses n-1 and is null for n<2. Three seeds measure training variation on shared ships, not independent ship samples; no significance claim. Historical test is exposed, not a new blind test.')
    return summary, private


def csv_rows(summary):
    rows = []
    def add(kind, seed, group, metric, label, unit, value=None, stats=None):
        rows.append(dict(split=summary['split'], kind=kind, seed=seed, group=group, metric=metric,
                         class_name=label, unit=unit, value=value,
                         n_seeds=stats['n_seeds'] if stats else '', mean=stats['mean'] if stats else '',
                         seed_sd=stats['seed_sd'] if stats else ''))
    for entry in summary['per_seed']:
        for arm, result in entry['arms'].items():
            for metric, value in result['metrics'].items():
                add('seed_metric', entry['seed'], arm, metric, '', 'fraction', value)
            for label, value in result['per_class_recall'].items():
                add('seed_recall', entry['seed'], arm, 'recall', label, 'fraction', value)
    for pair in summary['per_seed_paired']:
        for metric, value in pair['delta_percentage_points'].items():
            add('seed_delta', pair['seed'], pair['comparison'], metric, '', 'percentage_points', value)
        for label, value in pair['per_class_recall_delta_pp'].items():
            add('seed_recall_delta', pair['seed'], pair['comparison'], 'recall', label, 'percentage_points', value)
        for outcome, value in pair['paired_outcomes'].items():
            add('paired_outcome', pair['seed'], pair['comparison'], outcome, '', 'ships', value)
    for key, metric_key, recall_key, unit in (('aggregate', 'metrics', 'per_class_recall', 'fraction'),
                                             ('paired_aggregate', 'delta_percentage_points', 'per_class_recall_delta_pp', 'percentage_points')):
        for group, result in summary[key].items():
            for metric, stats in result[metric_key].items():
                add(key, '', group, metric, '', unit, stats=stats)
            for label, stats in result[recall_key].items():
                add(key + '_recall', '', group, 'recall', label, unit, stats=stats)
    for entry in summary['missing']:
        add('missing', entry['seed'], entry['arm'], '', '', '')
    return rows


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    args = parser.parse_args(argv)
    try:
        summary, private = collect(args.root, args.split)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    output = args.root / 'collected' / args.split
    output.mkdir(parents=True, exist_ok=True)
    (output / 'aggregate_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    public_rows = csv_rows(summary)
    write_csv(output / 'aggregate_summary.csv', public_rows,
              ('split', 'kind', 'seed', 'group', 'metric', 'class_name', 'unit', 'value', 'n_seeds', 'mean', 'seed_sd'))
    write_csv(output / 'paired_predictions_local.csv', private, ('seed', 'split', 'id', 'label', *ARMS))
    print(json.dumps(dict(split=args.split, status=summary['status'], loaded_results=summary['loaded_results'],
                          missing_results=len(summary['missing']), output=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
