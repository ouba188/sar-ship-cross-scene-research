"""Choose one common gamma from all three completed final validation scans.

Standard library only: no Torch, checkpoints, private caches or test results.
Inputs: final_seed{17,29,43}/aggregate_validation_fusion_scan.json.
Selection is validation tuning, not independent confirmation.
"""
import argparse
import csv
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import statistics


SEEDS = (17, 29, 43)
GAMMAS = (0., .25, .5, 1., 2., 4.)
METRICS = ('balanced_accuracy', 'macro_f1', 'accuracy')
NOTE = ('One common gamma selected on the same validation split used for checkpoint selection. '
        'This is validation tuning, not independent confirmation. Seed SD describes training variation '
        'on shared validation ships; three seeds are not independent ship samples. '
        'Report original gamma=1 and gamma=0 alongside the selected configuration. No test results were read.')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def describe(values):
    return dict(n_seeds=len(values), mean=statistics.mean(values), seed_sd=statistics.stdev(values))


def exact_mean(values):
    # Compare the actual stored decimal values, without an approximate-tie tolerance.
    return sum((Fraction(str(value)) for value in values), Fraction()) / len(values)


def validate_scan(payload, seed):
    if payload.get('status') != 'completed' or payload.get('seed') != seed:
        raise ValueError(f'final_seed{seed}: incomplete scan or wrong seed')
    if payload.get('test_images_or_predictions_read') is not False or payload.get('query_labels_used_in_matching') is not False:
        raise ValueError(f'final_seed{seed}: validation-only matching scope not established')
    if payload.get('evidence_temperature') != .2 or payload.get('validation_count') != 227:
        raise ValueError(f'final_seed{seed}: temperature or validation count changed')
    if payload.get('train_count') != 718 or payload.get('reference_count') != 32:
        raise ValueError(f'final_seed{seed}: source/reference protocol changed')
    grid = payload.get('gamma_grid', [])
    if len(grid) != len(GAMMAS) or not all(finite_number(gamma) for gamma in grid) or set(grid) != set(GAMMAS):
        raise ValueError(f'final_seed{seed}: gamma grid differs from the six fixed candidates')
    classes = payload.get('classes', [])
    if len(classes) != 4 or len(set(classes)) != 4 or not all(isinstance(label, str) for label in classes):
        raise ValueError(f'final_seed{seed}: invalid class mapping')
    rows = payload.get('scan', [])
    if len(rows) != len(GAMMAS):
        raise ValueError(f'final_seed{seed}: incomplete candidate results')
    by_gamma = {}
    support = None
    for row in rows:
        gamma = row.get('gamma')
        if not finite_number(gamma) or gamma not in GAMMAS or gamma in by_gamma:
            raise ValueError(f'final_seed{seed}: unknown or duplicate candidate')
        if row.get('count') != 227 or set(row.get('per_class', {})) != set(classes):
            raise ValueError(f'final_seed{seed}: candidate sample/class coverage changed')
        for metric in METRICS:
            if not finite_number(row.get(metric)) or not 0 <= row[metric] <= 1:
                raise ValueError(f'final_seed{seed}: invalid {metric}')
        counts = {}
        for label in classes:
            item = row['per_class'][label]
            if not isinstance(item.get('count'), int) or isinstance(item['count'], bool) or item['count'] <= 0:
                raise ValueError(f'final_seed{seed}: invalid per-class support')
            if not finite_number(item.get('recall')) or not 0 <= item['recall'] <= 1:
                raise ValueError(f'final_seed{seed}: invalid per-class recall')
            counts[label] = item['count']
        if sum(counts.values()) != 227 or (support is not None and counts != support):
            raise ValueError(f'final_seed{seed}: class support differs across candidates')
        support = counts
        for key in ('rescued', 'harmed'):
            if not isinstance(row.get(key), int) or isinstance(row[key], bool) or not 0 <= row[key] <= 227:
                raise ValueError(f'final_seed{seed}: invalid rescue/harm counts')
        by_gamma[gamma] = row
    if set(by_gamma) != set(GAMMAS):
        raise ValueError(f'final_seed{seed}: missing gamma candidates')
    if any(by_gamma[0.][metric] != payload['h0'][metric] for metric in METRICS):
        raise ValueError(f'final_seed{seed}: gamma=0 does not reproduce the recorded h0')
    if by_gamma[0.]['rescued'] or by_gamma[0.]['harmed']:
        raise ValueError(f'final_seed{seed}: gamma=0 unexpectedly changes h0 predictions')
    if not isinstance(payload.get('checkpoint_selected_epoch'), int) or not 1 <= payload['checkpoint_selected_epoch'] <= 20:
        raise ValueError(f'final_seed{seed}: invalid selected checkpoint epoch')
    if not all(isinstance(payload.get(key), str) and payload[key] for key in ('manifest_sha256', 'checkpoint_sha256')):
        raise ValueError(f'final_seed{seed}: missing scan provenance')
    return by_gamma, support


def collect(root):
    paths = {seed: root / f'final_seed{seed}' / 'aggregate_validation_fusion_scan.json' for seed in SEEDS}
    missing = [seed for seed, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError('All three final scans are required; missing seeds: ' + ', '.join(map(str, missing)))
    scans, provenance = {}, []
    common_classes, common_support, manifest_hash = None, None, None
    for seed, path in paths.items():
        payload = read_json(path)
        scan, support = validate_scan(payload, seed)
        if common_classes is not None and (payload['classes'] != common_classes or support != common_support
                                           or payload['manifest_sha256'] != manifest_hash):
            raise ValueError('Final scans disagree on class mapping, validation support or manifest')
        common_classes, common_support, manifest_hash = payload['classes'], support, payload['manifest_sha256']
        scans[seed] = scan
        provenance.append(dict(seed=seed, source_file=str(path.relative_to(root)).replace('\\', '/'),
                               checkpoint_selected_epoch=payload['checkpoint_selected_epoch'],
                               checkpoint_sha256=payload['checkpoint_sha256']))
    candidates = []
    rank = {}
    for gamma in GAMMAS:
        per_seed = []
        for seed in SEEDS:
            row = scans[seed][gamma]
            per_seed.append(dict(seed=seed, metrics={metric: row[metric] for metric in METRICS},
                                 per_class_recall={label: row['per_class'][label]['recall'] for label in common_classes},
                                 rescued_vs_gamma0=row['rescued'], harmed_vs_gamma0=row['harmed'],
                                 delta_vs_gamma0_pp={metric: 100 * (row[metric] - scans[seed][0.][metric]) for metric in METRICS},
                                 delta_vs_gamma1_pp={metric: 100 * (row[metric] - scans[seed][1.][metric]) for metric in METRICS}))
        aggregate = dict(metrics={metric: describe([row['metrics'][metric] for row in per_seed]) for metric in METRICS},
                         per_class_recall={label: describe([row['per_class_recall'][label] for row in per_seed]) for label in common_classes},
                         delta_vs_gamma0_pp={metric: describe([row['delta_vs_gamma0_pp'][metric] for row in per_seed]) for metric in METRICS},
                         delta_vs_gamma1_pp={metric: describe([row['delta_vs_gamma1_pp'][metric] for row in per_seed]) for metric in METRICS},
                         rescued_vs_gamma0=describe([row['rescued_vs_gamma0'] for row in per_seed]),
                         harmed_vs_gamma0=describe([row['harmed_vs_gamma0'] for row in per_seed]))
        candidates.append(dict(gamma=gamma, per_seed=per_seed, aggregate=aggregate))
        rank[gamma] = (exact_mean([row['metrics']['balanced_accuracy'] for row in per_seed]),
                       exact_mean([row['metrics']['macro_f1'] for row in per_seed]), -gamma)
    selected_gamma = max(GAMMAS, key=lambda gamma: rank[gamma])
    best_ba = rank[selected_gamma][0]
    ba_ties = [gamma for gamma in GAMMAS if rank[gamma][0] == best_ba]
    f1_ties = [gamma for gamma in ba_ties if rank[gamma][1] == rank[selected_gamma][1]]
    selected_candidate = next(row for row in candidates if row['gamma'] == selected_gamma)
    selection = dict(status='selected_on_validation', selected_gamma=selected_gamma,
                     evidence_temperature=.2, seeds=list(SEEDS), gamma_grid=list(GAMMAS),
                     criterion='Highest mean validation BA; exact tie: highest mean macro-F1; exact tie: smaller gamma. OA is reported, not used for selection.',
                     tie_comparison='Exact arithmetic on the decimal metric values stored in the scan JSON; no approximate-tie tolerance.',
                     gamma_tied_for_mean_ba=ba_ties, gamma_tied_after_macro_f1=f1_ties,
                     selected_aggregate=selected_candidate['aggregate'],
                     source_scans=provenance, manifest_sha256=manifest_hash,
                     zero_gamma_means_h_fallback=selected_gamma == 0.,
                     interpretation=NOTE + (' Selected gamma=0: report that this scan selects the h0 fallback, not a gain from H.' if selected_gamma == 0. else ''))
    summary = dict(status='complete', split='validation', seeds=list(SEEDS), classes=common_classes,
                   per_class_support=common_support, validation_count_per_seed=227,
                   evidence_temperature=.2, gamma_grid=list(GAMMAS), selected_gamma=selected_gamma,
                   candidates=candidates, source_scans=provenance, manifest_sha256=manifest_hash,
                   units='Metrics/recalls are fractions; paired deltas are percentage points; seed_sd uses n-1.',
                   interpretation=NOTE)
    return summary, selection


def candidate_csv(summary):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=('gamma', 'scope', 'seed', 'quantity', 'class_label', 'unit', 'value', 'mean', 'seed_sd', 'n_seeds'), lineterminator='\n')
    writer.writeheader()
    for candidate in summary['candidates']:
        for row in candidate['per_seed']:
            for group, unit in (('metrics', 'fraction'), ('per_class_recall', 'fraction'),
                                ('delta_vs_gamma0_pp', 'percentage_points'), ('delta_vs_gamma1_pp', 'percentage_points')):
                for key, value in row[group].items():
                    writer.writerow(dict(gamma=candidate['gamma'], scope='per_seed', seed=row['seed'],
                                         quantity=group if group == 'per_class_recall' else f'{group}.{key}',
                                         class_label=key if group == 'per_class_recall' else '', unit=unit, value=value))
            for group in ('rescued_vs_gamma0', 'harmed_vs_gamma0'):
                writer.writerow(dict(gamma=candidate['gamma'], scope='per_seed', seed=row['seed'], quantity=group, unit='ships', value=row[group]))
        for group, values in candidate['aggregate'].items():
            entries = {'': values} if group in ('rescued_vs_gamma0', 'harmed_vs_gamma0') else values
            unit = 'ships' if group in ('rescued_vs_gamma0', 'harmed_vs_gamma0') else 'percentage_points' if group.startswith('delta_') else 'fraction'
            for key, stats in entries.items():
                writer.writerow(dict(gamma=candidate['gamma'], scope='seed_aggregate',
                                     quantity=group if group == 'per_class_recall' or not key else f'{group}.{key}',
                                     class_label=key if group == 'per_class_recall' else '', unit=unit, **stats))
    return buffer.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    try:
        summary, selection = collect(args.root)
        outputs = {'aggregate_fusion_scan_summary.json': json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                   'fusion_candidate_table.csv': candidate_csv(summary),
                   'selected_gamma.json': json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False)}
        # Preserve an earlier selection rather than silently replacing it.
        for name, content in outputs.items():
            path = args.root / name
            if path.exists() and path.read_text(encoding='utf-8-sig') != content:
                raise ValueError(f'Existing {name} differs; refusing to overwrite a prior selection')
        for name, content in outputs.items():
            path = args.root / name
            if not path.exists():
                temporary = path.with_suffix(path.suffix + '.tmp')
                temporary.write_text(content, encoding='utf-8', newline='')
                temporary.replace(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(status='selected_on_validation', selected_gamma=selection['selected_gamma'],
                          seeds=list(SEEDS), independent_confirmation=False)))


if __name__ == '__main__':
    main()
