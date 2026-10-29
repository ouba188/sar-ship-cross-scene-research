"""CPU-only development analysis of adapt_only and source_mix frozen predictions.

Run: python -B analysis.py --dataset all --gallery
No prediction fallback, model import, final-test access, or training is performed.
"""
import argparse
from collections import Counter
import html
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE.parents[2]
OUT = HERE / 'analysis'
ARMS = ('adapt_only', 'source_mix')
SEEDS = (17, 29, 43)
STATES = ('rescued', 'harmed', 'CC', 'WW')
COMPARISONS = (('adapt_only', 'saved_baseline'), ('source_mix', 'saved_baseline'),
               ('source_mix', 'adapt_only'))
spec = importlib.util.spec_from_file_location('source_replay_previous', ROOT / 'm3_center_joint_20261030/analysis.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
coverage, prior = previous.coverage, previous.prior
read, dump, metric, events = previous.read, previous.dump, previous.metric, previous.events
METRICS = previous.METRICS


def load_pack(path, group, classes):
    """Align by ID; a partial cohort is pending, never a smaller silent denominator."""
    expected = [r['id'] for r in group]
    with np.load(path, allow_pickle=False) as pack:
        ids = pack['ids'].astype(str).tolist()
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate prediction IDs: {path}')
        extra = sorted(set(ids) - set(expected))
        if extra:
            raise ValueError(f'Prediction IDs outside development manifest: {path}: {extra}')
        missing = sorted(set(expected) - set(ids))
        if missing:
            return None, dict(path=str(path), missing_samples=len(missing), missing_ids=missing,
                              actual_samples=len(ids), expected_samples=len(expected))
        lookup = {key: i for i, key in enumerate(ids)}
        ix = [lookup[key] for key in expected]
        y = np.array([r['label'] for r in group])
        if not np.array_equal(pack['y'][ix], y):
            raise ValueError(f'Labels disagree with development manifest: {path}')
        result = {key: pack[key][ix].astype(float) for key in ('score', 'baseline')}
    for value in result.values():
        if value.shape != (len(group), len(classes)) or not np.isfinite(value).all():
            raise ValueError(f'Invalid logits shape or nonfinite values: {path}')
    return result, None


def load_dataset(dataset, requested):
    manifest = previous.manifest_dir(dataset)
    group = prior.rows(manifest / ('development.json' if dataset == 'private' else 'validation.json'))
    classes = prior.PRIVATE if dataset == 'private' else prior.PUBLIC
    refs = prior.rows(manifest / 'references.json')
    scores, receipts, pending, seeds, paths = {}, {}, [], [], []
    y = np.array([r['label'] for r in group])
    for seed in requested:
        required = [HERE / dataset / folder / f'{arm}_s{seed}.{ext}'
                    for arm in ARMS for folder, ext in (('training', 'json'), ('predictions', 'npz'))]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            pending.append(dict(seed=seed, missing_files=missing))
            continue
        packs, reports = {}, {}
        for arm in ARMS:
            path = HERE / dataset / 'predictions' / f'{arm}_s{seed}.npz'
            packs[arm], absent = load_pack(path, group, classes)
            if absent:
                pending.append(dict(seed=seed, arm=arm, **absent))
            report = read(HERE / dataset / 'training' / f'{arm}_s{seed}.json')
            for key, value in (('dataset', dataset), ('arm', arm), ('seed', seed), ('query_count', len(group))):
                if key in report and report[key] != value:
                    raise ValueError(f'{arm} s{seed}: training {key} disagrees: {report[key]!r}')
            if report.get('evaluation_used', False):
                raise ValueError(f'{arm} s{seed}: report declares final evaluation use')
            if 'reference_ids' in report and list(map(str, report['reference_ids'])) != [r['id'] for r in refs]:
                raise ValueError(f'{arm} s{seed}: training reference bank differs from gallery manifest')
            reports[arm] = report
        if any(pack is None for pack in packs.values()):
            continue
        if not np.array_equal(packs[ARMS[0]]['baseline'], packs[ARMS[1]]['baseline']):
            raise ValueError(f'{dataset} s{seed}: arms have different saved baselines')
        for arm in ARMS:
            report = reports[arm]
            for field, phase in (('score', 'after'), ('baseline', 'baseline')):
                actual = metric(packs[arm][field], y)
                for key in METRICS:
                    saved = report.get(phase, {}).get(key.lower())
                    if saved is not None and not np.isclose(actual[key], saved, rtol=0, atol=1e-6):
                        raise ValueError(f'{dataset} {arm} s{seed}: {phase}/{key} receipt mismatch')
            scores[arm, seed] = packs[arm]['score']
            receipts[arm, seed] = report
        scores['saved_baseline', seed] = packs[ARMS[0]]['baseline']
        seeds.append(seed)
        paths.extend(str(p) for p in required)
    return dict(dataset=dataset, group=group, classes=classes, scores=scores, receipts=receipts,
                seeds=seeds, requested=list(requested), pending=pending, input_paths=paths,
                refs=refs, manifest=str(manifest), y=y)


def paired_tables(d):
    endpoints, paired, tradeoffs, all_events = [], [], [], []
    y, scores = d['y'], d['scores']
    for seed in d['seeds']:
        for arm in ('saved_baseline', *ARMS):
            endpoints.append(dict(arm=arm, seed=seed, n=len(y), **metric(scores[arm, seed], y)))
        for arm, reference in COMPARISONS:
            old, new = scores[reference, seed], scores[arm, seed]
            es = events(old, new, d['group'], seed)
            for row, event in zip(d['group'], es):
                event.update(arm=arm, reference=reference,
                             **{k: row.get(k) for k in ('mmsi', 'pass_group', 'port')})
            all_events.extend(es)
            counts = Counter(e['state'] for e in es)
            a, b = metric(old, y), metric(new, y)
            paired.append(dict(arm=arm, reference=reference, seed=seed, n=len(y),
                **{'delta_' + k: b[k]-a[k] for k in METRICS}, **{k: counts[k] for k in STATES},
                top1_changes=sum(e['changed'] for e in es),
                wrong_to_wrong_changes=sum(e['changed'] and e['state'] == 'WW' for e in es)))
            for c, name in enumerate(d['classes']):
                ix = y == c
                subset = [e for e in es if e['label'] == c]
                sc = Counter(e['state'] for e in subset)
                a, b = (metric(old[ix], y[ix]), metric(new[ix], y[ix])) if ix.any() else ({}, {})
                tradeoffs.append(dict(arm=arm, reference=reference, seed=seed, label=c, class_name=name,
                    queries=int(ix.sum()), absent_class=not bool(ix.any()),
                    old_recall=a.get('OA'), new_recall=b.get('OA'),
                    delta_recall_pp=b['OA']-a['OA'] if ix.any() else None,
                    delta_NLL=b['NLL']-a['NLL'] if ix.any() else None,
                    delta_Brier=b['Brier']-a['Brier'] if ix.any() else None,
                    **{k: sc[k] for k in STATES},
                    top1_changes=sum(e['changed'] for e in subset)))
    return endpoints, paired, tradeoffs, all_events


def summarize(d):
    out = OUT / d['dataset']
    out.mkdir(parents=True, exist_ok=True)
    endpoints, paired, tradeoffs, es = paired_tables(d)
    means = []
    for arm, reference in COMPARISONS:
        rows = [r for r in paired if (r['arm'], r['reference']) == (arm, reference)]
        selected = [e for e in es if (e['arm'], e['reference']) == (arm, reference)]
        row = dict(arm=arm, reference=reference, orders=len(rows), seeds=d['seeds'])
        for k in METRICS:
            values = [r['delta_' + k] for r in rows]
            row['delta_' + k] = float(np.mean(values)) if values else None
            row['sd_' + k] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            row['positive_' + k] = sum(v > 0 for v in values)
            row['zero_' + k] = sum(v == 0 for v in values)
            row['negative_' + k] = sum(v < 0 for v in values)
        for state in STATES:
            subset = [e for e in selected if e['state'] == state]
            row[state + '_events'] = len(subset)
            row[state + '_queries'] = len({e['id'] for e in subset})
        row['top1_change_events'] = sum(e['changed'] for e in selected)
        row['top1_changed_queries'] = len({e['id'] for e in selected if e['changed']})
        means.append(row)
    endpoint_means = []
    for arm in ('saved_baseline', *ARMS):
        selected = [r for r in endpoints if r['arm'] == arm]
        row = dict(arm=arm, orders=len(selected))
        for k in METRICS:
            values = [r[k] for r in selected]
            row[k] = float(np.mean(values)) if values else None
            row['sd_' + k] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        endpoint_means.append(row)
    summary = dict(dataset=d['dataset'], scope='development', seeds=d['seeds'], requested=d['requested'],
        paired_three_orders_complete=set(d['seeds']) == set(SEEDS), pending=d['pending'],
        expected_query_count=len(d['group']), analyzed_query_count=len(d['group']) if d['seeds'] else 0,
        input_paths=d['input_paths'], manifest=d['manifest'], endpoint_means=endpoint_means, paired_means=means,
        training=[dict(arm=a, seed=s, receipt=d['receipts'][a, s]) for a, s in d['receipts']],
        final_test_manifests_read=False, final_test_predictions_read=False,
        notes=['Only complete paired seeds with both new prediction files and training reports are analyzed.',
               'No old prediction files or old training results are substituted. Baseline comes from these new NPZs.',
               'Scores are logits; OA/BA and differences are percent/percentage points; NLL/Brier are unscaled.',
               'BA averages present classes. Absent classes have null recall, not zero recall.',
               'Means and sample SD are over available paired seeds; incomplete results are provisional.',
               'Repeated query events across seeds are not independent samples or ships.',
               'Positive OA/BA differences improve accuracy; negative NLL/Brier differences improve risk.',
               'Training reports are retained as reported; gradients alone do not establish a causal benefit.'])
    dump(out / 'summary.json', summary)
    for name, rows in (('metrics', endpoints), ('endpoint_means', endpoint_means), ('paired', paired),
                       ('paired_means', means), ('class_tradeoffs', tradeoffs), ('events', es),
                       ('changed_queries', [e for e in es if e['changed']])):
        previous.write_csv(out / (name + '.csv'), rows)
    lines = [f"# {d['dataset']} development", '', f"Completed paired seeds: {d['seeds']}.",
             f"Three-seed complete: {summary['paired_three_orders_complete']}.",
             f"Analyzed queries: {summary['analyzed_query_count']}; expected: {len(d['group'])}."]
    prior.table(lines, ['Candidate - reference', 'Seed', 'Delta OA', 'Delta BA', 'Rescue', 'Harm', 'WW changes'],
                [[r['arm'] + ' - ' + r['reference'], r['seed'], r['delta_OA'], r['delta_BA'],
                  r['rescued'], r['harmed'], r['wrong_to_wrong_changes']] for r in paired])
    lines += ['', '## Missing Inputs', '', '```json', json.dumps(d['pending'], indent=2), '```', '', *summary['notes']]
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def select_cases(d, arm, reference):
    pool = [e for s in d['seeds'] for e in events(d['scores'][reference, s], d['scores'][arm, s], d['group'], s)]
    selected, absent, used = [], [], set()
    for state in STATES:
        rows = [e for e in pool if e['state'] == state]
        frequency = Counter(e['id'] for e in rows)
        rows.sort(key=lambda e: (-frequency[e['id']], -abs(e['margin_change']), e['id'], e['seed']))
        event = next((e for e in rows if e['id'] not in used), None)
        if event is None:
            absent.append(dict(state=state, state_events=len(rows),
                               reason=('No paired predictions available.' if not d['seeds'] else
                                       'No distinct unselected query in this state.' if rows else 'No events in this state.')))
        else:
            used.add(event['id'])
            selected.append(dict(event, arm=arm, reference=reference, state_orders=frequency[event['id']]))
    return selected, absent


def gallery(d, arm, reference):
    helper = coverage.module('source_replay_native_panel', ROOT / 'spatial_m12_refit_20261029/analysis.py')
    helper.CLASSES = tuple(d['classes'])
    native = coverage.module('source_replay_native_input', WORK / 'sar_m123_standalone/sar_m123/input.py')
    folder = OUT / d['dataset'] / (arm + '_vs_' + reference)
    folder.mkdir(parents=True, exist_ok=True)
    chosen, absent = select_cases(d, arm, reference)
    lookup = {r['id']: i for i, r in enumerate(d['group'])}
    cases, sources, blocks = [], [], []
    for selected in chosen:
        i = lookup[selected['id']]
        q = d['group'][i]
        truth, rival = q['label'], selected['rival']
        case = dict(selected, dataset=d['dataset'], visual_status='not_reviewed',
                    class_name=d['classes'][truth], rival_name=d['classes'][rival],
                    **{k: q.get(k) for k in ('mmsi', 'pass_group', 'port')}, orders=[], panels={})
        for seed in d['seeds']:
            row = dict(seed=seed)
            for name in ('saved_baseline', *ARMS):
                z = d['scores'][name, seed][i]
                row[name] = dict(prediction=int(z.argmax()), logits=z.tolist(), truth_rival_margin=float(z[truth]-z[rival]))
            row['state'] = prior.state(row[reference]['prediction'], row[arm]['prediction'], truth)
            case['orders'].append(row)
        legal = [r for r in d['refs'] if helper.eligible(q, r)]
        case['all_legal_reference_ids'] = [r['id'] for r in legal]
        for pol in ('vv', 'vh'):
            available = lambda r: bool(r.get(pol) and r[pol].get('alignment_valid', True))
            own = [r for r in legal if r['label'] == truth and available(r)]
            other = [r for r in legal if r['label'] == rival and available(r)]
            shown = own[:2] + other[:2]
            members = [q, *shown]
            roles = ['QUERY'] + ['TRUE REF'] * len(own[:2]) + ['RIVAL REF'] * len(other[:2])
            name = f"{selected['state']}_{q['id']}_{pol}.png"
            receipts = helper.panel(members, roles, pol, folder / name, native)
            for r, entry in zip(members, receipts):
                entry.update(case_id=q['id'], panel=name, label=r['label'], class_name=d['classes'][r['label']],
                             pair_legal=entry['role'] != 'QUERY',
                             usable_with_query=bool(receipts[0]['available'] and entry['available']))
            sources.extend(receipts)
            case['panels'][pol] = dict(file=name, same_class_candidates=len(own), rival_candidates=len(other),
                shown_reference_ids=[r['id'] for r in shown], query_available=receipts[0]['available'],
                same_class_usable=sum(e['available'] for e in receipts if e['role'] == 'TRUE REF'),
                rival_usable=sum(e['available'] for e in receipts if e['role'] == 'RIVAL REF'))
        cases.append(case)
        blocks += ['<h2>' + html.escape(f"{selected['state']}: {q['id']}") + '</h2>',
                   '<pre>' + html.escape(json.dumps(case, ensure_ascii=False, indent=2)) + '</pre>']
        blocks += [f'<h3>{p.upper()}</h3><img src="{case["panels"][p]["file"]}">' for p in ('vv', 'vh')]
    dump(folder / 'cases.json', cases)
    dump(folder / 'native_sources.json', sources)
    selection = dict(arm=arm, reference=reference, seeds=d['seeds'], absent=absent,
        cases=len(cases), panels=2 * len(cases), visual_status='not_reviewed',
        rule='One distinct query/state, ordered by state frequency, absolute fixed-rival margin change, ID, seed. No random sampling.',
        references='Actual fixed reference manifest; exclude identical ID/MMSI; first two same/rival class entries in bank order. No resizing.',
        limitation='Outcome-selected examples are not population proportions. Bank order is not learned attention ranking. No new per-reference attention trace or physical causal evidence.')
    dump(folder / 'selection.json', selection)
    page = '<!doctype html><meta charset="utf-8"><title>M3 source replay cases</title><style>body{font:14px Arial;margin:24px}pre{white-space:pre-wrap}img{image-rendering:pixelated;max-width:none}</style>'
    page += '<h1>' + html.escape(d['dataset'] + ': ' + arm + ' vs ' + reference) + '</h1>'
    page += '<p>VV/VH separate; original pixels and exact crop/pad; separate OBB overlay. Rendered, not visually reviewed.</p>'
    page += '<pre>' + html.escape(json.dumps(selection, indent=2)) + '</pre>'
    (folder / 'index.html').write_text(page + ''.join(blocks), encoding='utf-8')
    return selection


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
        item = dict(dataset=dataset, seeds=d['seeds'], pending=d['pending'],
                    paired_three_orders_complete=summary['paired_three_orders_complete'])
        if args.gallery:
            item['galleries'] = [gallery(d, arm, reference) for arm, reference in COMPARISONS]
        status.append(item)
    assert 'torch' not in sys.modules, 'CPU analysis must not import model runtime'
    receipt = dict(status=status, cpu_only=True, torch_imported=False,
                   final_test_manifests_read=False, final_test_predictions_read=False)
    dump(OUT / f'run_{args.dataset}_receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
