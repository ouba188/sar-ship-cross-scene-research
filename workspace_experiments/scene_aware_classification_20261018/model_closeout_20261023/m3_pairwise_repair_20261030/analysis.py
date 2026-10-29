"""CPU-only M3 pairwise analysis; current-round development predictions only.

python -B analysis.py --dataset all --seeds 17 --gallery
Each comparison uses its own available matched seeds. No training or source NPZs.
"""
import argparse
from itertools import combinations
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'analysis'
ARMS = ('ce', 'rank', 'hybrid')
SEEDS = (17, 29, 43)
COMPARISONS = tuple((a, 'saved_baseline') for a in ARMS) + (('rank', 'ce'), ('hybrid', 'ce'))
spec = importlib.util.spec_from_file_location('pairwise_replay_helpers', ROOT / 'm3_source_replay_20261030/analysis.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
read, dump, metric = replay.read, replay.dump, replay.metric
METRICS, STATES = replay.METRICS, replay.STATES
load_pack = replay.load_pack


def load_dataset(dataset, requested):
    manifest = replay.previous.manifest_dir(dataset)
    group = replay.prior.rows(manifest / ('development.json' if dataset == 'private' else 'validation.json'))
    refs = replay.prior.rows(manifest / 'references.json')
    classes = replay.prior.PRIVATE if dataset == 'private' else replay.prior.PUBLIC
    y = np.array([r['label'] for r in group])
    scores, receipts, pending, paths = {}, {}, [], []
    for seed in requested:
        for arm in ARMS:
            # Exact names deliberately exclude {arm}_source_s{seed}.npz.
            receipt_path = HERE / dataset / 'training' / f'{arm}_s{seed}.json'
            prediction_path = HERE / dataset / 'predictions' / f'{arm}_s{seed}.npz'
            missing = [str(p) for p in (receipt_path, prediction_path) if not p.is_file()]
            if missing:
                pending.append(dict(arm=arm, seed=seed, missing_files=missing))
                continue
            report = read(receipt_path)
            pack, absent = load_pack(prediction_path, group, classes)
            if absent:
                pending.append(dict(arm=arm, seed=seed, **absent))
                continue
            for key, expected in (('dataset', dataset), ('arm', arm), ('seed', seed),
                                  ('query_count', len(group)), ('steps', 54 if dataset == 'private' else 96)):
                if report.get(key) != expected:
                    raise ValueError(f'{receipt_path}: {key} != {expected!r}')
            if report.get('frozen_unchanged') is not True:
                raise ValueError(f'{receipt_path}: frozen parameter check did not pass')
            for key in ('restore_error', 'native_error'):
                if not 0 <= report[key] < 1e-4:
                    raise ValueError(f'{receipt_path}: {key} failed')
            if report.get('evaluation_used', False):
                raise ValueError(f'{receipt_path}: final evaluation used')
            if list(map(str, report.get('reference_ids', []))) != [r['id'] for r in refs]:
                raise ValueError(f'{receipt_path}: actual references differ from gallery manifest')
            for field, phase in (('score', 'after'), ('baseline', 'baseline')):
                actual = metric(pack[field], y)
                for key in METRICS:
                    saved = report[phase][key.lower()]
                    if not np.isclose(actual[key], saved, rtol=0, atol=1e-6):
                        raise ValueError(f'{receipt_path}: {phase}/{key} mismatch')
            if ('saved_baseline', seed) in scores and not np.array_equal(scores['saved_baseline', seed], pack['baseline']):
                raise ValueError(f'{dataset} s{seed}: arms have different baselines')
            for other in ARMS:
                if (other, seed) in receipts:
                    for key in ('lr', 'steps', 'reference_ids'):
                        if report[key] != receipts[other, seed][key]:
                            raise ValueError(f'{dataset} s{seed}: unmatched {key}')
            scores[arm, seed] = pack['score']
            scores['saved_baseline', seed] = pack['baseline']
            receipts[arm, seed] = report
            paths.extend(map(str, (receipt_path, prediction_path)))
    return dict(dataset=dataset, group=group, classes=classes, y=y, refs=refs,
                manifest=str(manifest), requested=list(requested), scores=scores, receipts=receipts,
                seeds=sorted({s for _, s in receipts}), pending=pending, input_paths=paths)


def paired_seeds(d, arm, reference):
    return [s for s in d['seeds'] if (arm, s) in d['scores'] and (reference, s) in d['scores']]


def paired_tables(d, arms=ARMS, comparisons=COMPARISONS):
    tables = [[], [], [], []]
    # Reuse established ID/event/risk calculations with only the available arms.
    for seed in d['seeds']:
        replay.ARMS = tuple(a for a in arms if (a, seed) in d['scores'])
        replay.COMPARISONS = tuple((a, b) for a, b in comparisons if seed in paired_seeds(d, a, b))
        result = replay.paired_tables(dict(d, seeds=[seed]))
        for target, rows in zip(tables, result):
            target.extend(rows)
    return tables


def shared_shift(old, new, y):
    delta = new - old
    delta = delta - delta.mean(axis=1, keepdims=True)
    common = delta.mean(axis=0, keepdims=True)
    residual = delta - common
    energy = float(np.square(delta).sum())
    simple, specific = old + common, old + residual
    initial, actual, shared = old.argmax(1), new.argmax(1), simple.argmax(1)
    changed = actual != initial
    result = dict(count=len(y), centered_delta_energy=energy,
        common_energy_fraction=float(len(y) * np.square(common).sum() / energy) if energy else None,
        residual_energy=float(np.square(residual).sum()), common_class_offset=common[0].tolist(),
        actual_changed=int(changed.sum()), common_changed=int((shared != initial).sum()),
        same_prediction=int((actual == shared).sum()),
        same_among_actual_changed=int((changed & (actual == shared)).sum()))
    for name, z in (('reference', old), ('actual', new), ('common_only', simple), ('residual_only', specific)):
        result.update({name + '_' + k: v for k, v in metric(z, y).items()})
    return result


def query_pair_metrics(d):
    """Evaluation only: equal class-pair weight, uniform legal pairs within each."""
    group, y = d['group'], d['y']
    pools, coverage = [], []
    for a, b in combinations(range(len(d['classes'])), 2):
        left = [i for i, r in enumerate(group) if r['label'] == a and r.get('mmsi') and r.get('pass_group')]
        right = [j for j, r in enumerate(group) if r['label'] == b and r.get('mmsi') and r.get('pass_group')]
        pairs = [(i, j) for i in left for j in right
                 if str(group[i]['mmsi']) != str(group[j]['mmsi'])
                 and group[i]['pass_group'] != group[j]['pass_group']]
        row = dict(a=a, b=b, class_a=d['classes'][a], class_b=d['classes'][b], pairs=len(pairs),
                   missing=not bool(pairs),
                   reason=None if pairs else ('absent_query_class' if not (np.any(y == a) and np.any(y == b))
                                              else 'no_legal_pair_with_known_ship_and_pass'))
        coverage.append(row)
        if pairs:
            pools.append((row, np.asarray(pairs, dtype=int)))
    details, endpoints, paired = [], [], []
    for (arm, seed), z in sorted(d['scores'].items()):
        losses, accuracies = [], []
        for row in coverage:
            if row['missing']:
                details.append(dict(arm=arm, seed=seed, **row, pair_logistic=None, pair_rank_accuracy=None))
        for row, ij in pools:
            i, j = ij[:, 0], ij[:, 1]
            a, b = row['a'], row['b']
            gap = (z[i, a] - z[i, b]) - (z[j, a] - z[j, b])
            loss, accuracy = float(np.logaddexp(0, -gap).mean()), float((gap > 0).mean())
            losses.append(loss)
            accuracies.append(accuracy)
            details.append(dict(arm=arm, seed=seed, **row, pair_logistic=loss, pair_rank_accuracy=accuracy))
        endpoints.append(dict(arm=arm, seed=seed, evaluated_class_pairs=len(pools),
            missing_class_pairs=len(coverage)-len(pools), legal_query_pairs=sum(len(ij) for _, ij in pools),
            pair_logistic=float(np.mean(losses)) if losses else None,
            pair_rank_accuracy=float(np.mean(accuracies)) if accuracies else None))
    lookup = {(r['arm'], r['seed']): r for r in endpoints}
    for arm, reference in COMPARISONS:
        for seed in paired_seeds(d, arm, reference):
            new, old = lookup[arm, seed], lookup[reference, seed]
            paired.append(dict(arm=arm, reference=reference, seed=seed,
                **{'delta_' + k: new[k]-old[k] if new[k] is not None else None
                   for k in ('pair_logistic', 'pair_rank_accuracy')}))
    means = []
    for arm, reference in COMPARISONS:
        rows = [r for r in paired if (r['arm'], r['reference']) == (arm, reference)]
        entry = dict(arm=arm, reference=reference, seeds=[r['seed'] for r in rows])
        for key in ('pair_logistic', 'pair_rank_accuracy'):
            values = [r['delta_' + key] for r in rows if r['delta_' + key] is not None]
            entry['delta_' + key] = float(np.mean(values)) if values else None
            entry['sd_' + key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        means.append(entry)
    return dict(manifest=d['manifest'], query_count=len(group),
        queries_missing_ship_or_pass=sum(not (r.get('mmsi') and r.get('pass_group')) for r in group),
        weighting='Equal over available class pairs; uniform over legal query pairs within each class pair.',
        definition='softplus(-((z_i,a-z_i,b)-(z_j,a-z_j,b))); rank accuracy uses gap>0; ties are incorrect.',
        scope='Development diagnostic only; labels organize evaluation, never training or sample selection. Pairs share queries and are not independent evidence.',
        common_bias='Query-common class offsets cancel exactly in pair gaps; parameter updates may still introduce such offsets.',
        coverage=coverage, endpoints=endpoints, paired=paired, paired_means=means, class_pair_metrics=details)


def summarize(d):
    out = OUT / d['dataset']
    out.mkdir(parents=True, exist_ok=True)
    endpoints, paired, tradeoffs, events = paired_tables(d)
    means, shifts = [], []
    for arm, reference in COMPARISONS:
        seeds = paired_seeds(d, arm, reference)
        rows = [r for r in paired if (r['arm'], r['reference']) == (arm, reference)]
        es = [e for e in events if (e['arm'], e['reference']) == (arm, reference)]
        result = dict(arm=arm, reference=reference, orders=len(seeds), seeds=seeds,
                      requested_complete=seeds == d['requested'], three_seed_complete=set(seeds) == set(SEEDS))
        for key in METRICS:
            values = [r['delta_' + key] for r in rows]
            result['delta_' + key] = float(np.mean(values)) if values else None
            result['sd_' + key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            result['improved_' + key] = sum(v > 0 if key in ('OA', 'BA') else v < 0 for v in values)
        for state in STATES:
            subset = [e for e in es if e['state'] == state]
            result[state + '_events'] = len(subset)
            result[state + '_queries'] = len({e['id'] for e in subset})
        means.append(result)
        for seed in seeds:
            shifts.append(dict(arm=arm, reference=reference, seed=seed,
                **shared_shift(d['scores'][reference, seed], d['scores'][arm, seed], d['y'])))
    endpoint_means = []
    for arm in ('saved_baseline', *ARMS):
        rows = [r for r in endpoints if r['arm'] == arm]
        entry = dict(arm=arm, seeds=[r['seed'] for r in rows])
        for key in METRICS:
            values = [r[key] for r in rows]
            entry[key] = float(np.mean(values)) if values else None
            entry['sd_' + key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        endpoint_means.append(entry)
    summary = dict(dataset=d['dataset'], scope='development', requested=d['requested'],
        available_seeds=d['seeds'], available_by_arm={a: paired_seeds(d, a, 'saved_baseline') for a in ARMS},
        expected_query_count=len(d['y']), pending=d['pending'], input_paths=d['input_paths'],
        manifest=d['manifest'], paired_means=means, endpoint_means=endpoint_means,
        requested_complete=all(r['requested_complete'] for r in means),
        three_seed_complete=all(r['three_seed_complete'] for r in means),
        training=[dict(arm=a, seed=s, receipt=r) for (a, s), r in d['receipts'].items()],
        notes=[
            'Only exact current-round development NPZ names are loaded; source NPZs and prior results are excluded.',
            'Each comparison uses its own complete paired seeds; unavailable arms/seeds are pending, never zero gain.',
            'Single-seed results are a screening result, not three-seed confirmation. SD is null for one seed.',
            'OA/BA differences are percentage points; NLL/Brier are unscaled and lower is better.',
            'BA averages present query classes. Missing classes have null recall.',
            'Repeated query events across seeds are not independent observations or vessels.',
            'Shared shift is the class-centered new logit increment, not the whole M3 contribution.',
            'Common-only and residual-only are label-free offsets fitted on development logits: descriptive diagnostics, not deployable candidates.',
            'Source/pair objectives and gradients are reported training evidence, not proof of development benefit or physical causality.',
            'Generated galleries require visual review; selected cases do not estimate population proportions.'])
    dump(out / 'summary.json', summary)
    dump(out / 'shared_shift.json', shifts)
    pair_result = query_pair_metrics(d)
    dump(out / 'query_pairs.json', pair_result)
    for name, rows in (('query_pair_metrics', pair_result['endpoints']),
                       ('query_pair_deltas', pair_result['paired']),
                       ('query_pair_means', pair_result['paired_means']),
                       ('query_class_pairs', pair_result['class_pair_metrics'])):
        replay.previous.write_csv(out / (name + '.csv'), rows)
    for name, rows in (('metrics', endpoints), ('endpoint_means', endpoint_means), ('paired', paired), ('paired_means', means),
                       ('class_tradeoffs', tradeoffs), ('events', events), ('shared_shift', shifts),
                       ('changed_queries', [e for e in events if e['changed']])):
        replay.previous.write_csv(out / (name + '.csv'), rows)
    lines = [f"# {d['dataset']} pairwise development analysis", '',
             f"Requested seeds: {d['requested']}; available: {summary['available_by_arm']}.",
             f"Queries per paired seed: {len(d['y'])}. Three-seed complete: {summary['three_seed_complete']}."]
    replay.prior.table(lines, ['Candidate - reference', 'Seed', 'Delta OA', 'Delta BA', 'Delta NLL', 'Delta Brier', 'Rescue', 'Harm'],
        [[r['arm'] + ' - ' + r['reference'], r['seed'], *[r['delta_' + k] for k in METRICS], r['rescued'], r['harmed']] for r in paired])
    replay.prior.table(lines, ['Candidate - reference', 'Seed', 'Common energy fraction', 'Same prediction', 'Same changed'],
        [[r['arm'] + ' - ' + r['reference'], r['seed'], r['common_energy_fraction'],
          f"{r['same_prediction']}/{r['count']}", f"{r['same_among_actual_changed']}/{r['actual_changed']}"] for r in shifts])
    replay.prior.table(lines, ['Arm', 'Seed', 'Pair logistic', 'Rank accuracy', 'Class pairs', 'Missing class pairs'],
        [[r['arm'], r['seed'], r['pair_logistic'], r['pair_rank_accuracy'], r['evaluated_class_pairs'],
          r['missing_class_pairs']] for r in pair_result['endpoints']])
    lines += ['', pair_result['weighting'], pair_result['scope'], pair_result['common_bias'],
              '', *summary['notes'], '', '## Pending', '', '```json', json.dumps(d['pending'], indent=2), '```']
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def gallery(d, arm, reference):
    seeds = paired_seeds(d, arm, reference)
    if not seeds:
        return dict(arm=arm, reference=reference, seeds=[], status='pending')
    replay.OUT = OUT
    replay.ARMS = tuple(a for a in ARMS if all((a, s) in d['scores'] for s in seeds))
    # Same fixed bank is checked against every current receipt before rendering.
    selection = replay.gallery(dict(d, seeds=seeds), arm, reference)
    folder = OUT / d['dataset'] / (arm + '_vs_' + reference)
    cases = read(folder / 'cases.json')
    for case in cases:
        case['score_increment_diagnostics'] = []
        i = next(i for i, r in enumerate(d['group']) if r['id'] == case['id'])
        for seed in seeds:
            old, new = d['scores'][reference, seed], d['scores'][arm, seed]
            increment = new - old
            increment -= increment.mean(axis=1, keepdims=True)
            common = increment.mean(axis=0)
            a, b = case['label'], case['rival']
            case['score_increment_diagnostics'].append(dict(seed=seed,
                total_truth_rival_change=float(increment[i, a] - increment[i, b]),
                common_truth_rival_change=float(common[a] - common[b]),
                residual_truth_rival_change=float(increment[i, a] - increment[i, b] - common[a] + common[b])))
    dump(folder / 'cases.json', cases)
    page = folder / 'index.html'
    page.write_text(page.read_text(encoding='utf-8').replace('M3 source replay cases', 'M3 pairwise cases'), encoding='utf-8')
    return selection


def guard_analysis(d):
    """Keep supplied source-offset outputs separate from trained model results."""
    g = dict(d, scores=dict(d['scores']))
    reports, pending = [], []
    guards = tuple(a + '_guard' for a in ARMS)
    comparisons = tuple((a + '_guard', b) for a in ARMS for b in ('saved_baseline', a))
    comparisons += (('rank_guard', 'ce_guard'), ('hybrid_guard', 'ce_guard'))
    for arm in ARMS:
        for seed in paired_seeds(d, arm, 'saved_baseline'):
            candidate = arm + '_guard'
            rp = HERE / d['dataset'] / 'alignment' / f'{candidate}_s{seed}.json'
            pp = HERE / d['dataset'] / 'predictions' / f'{candidate}_s{seed}.npz'
            missing = [str(p) for p in (rp, pp) if not p.is_file()]
            if missing:
                pending.append(dict(arm=candidate, seed=seed, missing_files=missing))
                continue
            report = read(rp)
            if report.get('target_used_for_shift') is not False or report.get('evaluation_used'):
                raise ValueError(f'{rp}: source-only alignment scope disagrees')
            pack, absent = load_pack(pp, d['group'], d['classes'])
            if absent:
                pending.append(dict(arm=candidate, seed=seed, **absent))
                continue
            if not np.array_equal(pack['baseline'], d['scores']['saved_baseline', seed]):
                raise ValueError(f'{pp}: unmatched baseline')
            expected = d['scores'][arm, seed] - np.asarray(report['source_shift'])
            if not np.allclose(pack['score'], expected, rtol=0, atol=1e-10):
                raise ValueError(f'{pp}: score is not parent minus reported source offset')
            g['scores'][candidate, seed] = pack['score']
            reports.append(dict(arm=candidate, seed=seed,
                                receipt={k: v for k, v in report.items() if k != 'source_ids'}))
    out = OUT / 'guard_diagnostics' / d['dataset']
    out.mkdir(parents=True, exist_ok=True)
    endpoints, paired, tradeoffs, events = paired_tables(g, (*ARMS, *guards), comparisons)
    means = []
    for arm, reference in comparisons:
        rows = [r for r in paired if (r['arm'], r['reference']) == (arm, reference)]
        entry = dict(arm=arm, reference=reference, seeds=[r['seed'] for r in rows])
        for key in METRICS:
            values = [r['delta_' + key] for r in rows]
            entry['delta_' + key] = float(np.mean(values)) if values else None
            entry['sd_' + key] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        entry.update({state + '_events': sum(r[state] for r in rows) for state in STATES})
        means.append(entry)
    result = dict(dataset=d['dataset'], scope='separate_offline_source_offset_coordination_diagnostic',
        paired_means=means, pending=pending, reports=reports, source_predictions_read=False,
        notes=['Supplied source-class/vessel-balanced offsets, fixed multiplier 1. No target mean is used in corrected predictions.',
               'Each guard is compared to its raw parent and baseline; rank/hybrid guards also compare to CE guard.',
               'This is not a new attention mechanism or an integrated trained candidate.',
               'A query-common offset leaves every class-pair gap unchanged; guard pair logistic/rank accuracy equal the raw parent.',
               'The class-centered energy ratio is not an information fraction.'])
    dump(out / 'summary.json', result)
    for name, rows in (('metrics', endpoints), ('paired', paired), ('paired_means', means),
                       ('class_tradeoffs', tradeoffs), ('events', events)):
        replay.previous.write_csv(out / (name + '.csv'), rows)
    lines = ['# Separate source-offset diagnostic', '', *result['notes']]
    replay.prior.table(lines, ['Candidate - reference', 'Seeds', 'Delta OA', 'Delta BA', 'Delta NLL', 'Delta Brier'],
        [[r['arm'] + ' - ' + r['reference'], r['seeds'], *[r['delta_' + k] for k in METRICS]] for r in means])
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dict(available_reports=len(reports), pending=pending, output=str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('private', 'oss2', 'all'), default='all')
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=[17])
    parser.add_argument('--gallery', action='store_true')
    args = parser.parse_args()
    status = []
    for dataset in (('private', 'oss2') if args.dataset == 'all' else (args.dataset,)):
        d = load_dataset(dataset, sorted(set(args.seeds)))
        summary = summarize(d)
        item = dict(dataset=dataset, available_by_arm=summary['available_by_arm'],
                    requested_complete=summary['requested_complete'], pending=d['pending'])
        if args.gallery:
            item['galleries'] = [gallery(d, a, b) for a, b in COMPARISONS]
        item['guard_diagnostics'] = guard_analysis(d)
        status.append(item)
    assert 'torch' not in sys.modules, 'CPU analysis must not import model runtime'
    receipt = dict(status=status, cpu_only=True, torch_imported=False,
                   final_test_manifests_read=False, final_test_predictions_read=False,
                   source_predictions_read=False)
    dump(OUT / f'run_{args.dataset}_receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
