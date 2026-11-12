"""CPU-only analysis of current bidirectional fine-tuning predictions.

Historical directories supply manifests and native-panel helpers ONLY. No model,
checkpoint, training module, or full image cache is loaded. Run after training
and prediction export; --partial permits available runs and matched seed pairs.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import html
import importlib.util
import json
from string import Template

import numpy as np


ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'cross_dataset_retrained_20261111'
DIRECTIONS = ('oss2_to_ports24', 'ports24_to_oss2')
SEEDS = (17, 29, 43)
ARMS = ('plain', 'full')
METRICS = ('oa', 'ba', 'macro_f1')
KINDS = ('rescued', 'harmed', 'both_correct', 'both_wrong')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def load_visuals():
    # Reuse the retrained analyzer's exact helper binding without running main().
    spec = importlib.util.spec_from_file_location('retrained_analysis', OLD / 'analyze.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.viz.ROOT = ROOT
    return old.viz


def confusion(y, pred):
    return np.bincount(4 * y + pred, minlength=16).reshape(4, 4)


def cm_metrics(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(-1), cm.sum(-2)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp),
                   where=(support + predicted) > 0)
    return np.stack((tp.sum(-1) / cm.sum(axis=(-2, -1)),
                     recall.sum(-1) / (support > 0).sum(-1), f1.mean(-1)), axis=-1) * 100


def metrics(viz, y, score):
    return dict(viz.metrics(y, score),
                macro_f1=float(cm_metrics(confusion(y, score.argmax(1)))[2]))


def load_prediction(path, rows, expected_ids):
    expected_ids = np.asarray(expected_ids, dtype=np.int64)
    with np.load(path, allow_pickle=False) as data:
        ids = data['ids']
        labels = data['label']
        score = np.asarray(data['score'], dtype=np.float64)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu':
        raise ValueError(f'{path.name}: ids must be integer manifest indices')
    if (len(ids) != len(expected_ids) or len(np.unique(ids)) != len(ids)
            or not np.array_equal(np.sort(ids), np.sort(expected_ids))):
        raise ValueError(f'{path.name}: ids do not match the declared split')
    if labels.shape != ids.shape or score.shape != (len(ids), 4):
        raise ValueError(f'{path.name}: expected label[N] and score[N,4]')
    order = np.argsort(ids)
    aligned = order[np.searchsorted(ids[order], expected_ids)]
    labels, score = labels[aligned], score[aligned]
    y = np.array([rows[i]['label'] for i in expected_ids], dtype=np.int64)
    if not np.array_equal(labels, y) or not np.isfinite(score).all():
        raise ValueError(f'{path.name}: labels disagree with manifest or nonfinite scores')
    return y, score


def flatten(data, prefix=''):
    output = {}
    for key, value in data.items():
        key = f'{prefix}{key}'
        if isinstance(value, dict):
            output.update(flatten(value, key + '.'))
        else:
            output[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
    return output


def sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def csv_table(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns or ['status'])
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    if not rows:
        return '(No eligible current results.)'
    columns = list(rows[0])
    def cell(value):
        if value is None:
            return '--'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(columns) + ' |',
             '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines.extend('| ' + ' | '.join(cell(row.get(c)) for c in columns) + ' |' for row in rows)
    return '\n'.join(lines)


def bootstrap(y, ships, scores, seeds, repeats, rng_seed):
    """Resample vessels jointly for both arms and all fixed observed seeds."""
    _, inverse = np.unique(ships, return_inverse=True)
    nships = int(inverse.max()) + 1
    counts = np.zeros((2, len(seeds), nships, 4, 4), dtype=np.int64)
    for a, arm in enumerate(ARMS):
        for s, seed in enumerate(seeds):
            pred = scores[arm, seed].argmax(1)
            np.add.at(counts[a, s], (inverse, y, pred), 1)
    rng = np.random.default_rng(rng_seed)
    draws = []
    for _ in range(repeats):
        weights = np.bincount(rng.integers(nships, size=nships), minlength=nships)
        sampled = np.einsum('n,asnij->asij', weights, counts)
        # Preserve the four-class estimand; report rejected draws explicitly.
        if np.any(sampled[0, 0].sum(1) == 0):
            continue
        values = cm_metrics(sampled)
        draws.append((values[1] - values[0]).mean(0))
    interval = np.quantile(draws, [.025, .975], axis=0) if draws else None
    observed = cm_metrics(counts.sum(2))
    delta = observed[1] - observed[0]
    return [dict(metric=metric, mean=float(delta[:, k].mean()), std=sd(delta[:, k]),
                 positive_seeds=int((delta[:, k] > 0).sum()), paired_seeds=len(seeds),
                 seeds=','.join(map(str, seeds)),
                 ci_low=float(interval[0, k]) if interval is not None else None,
                 ci_high=float(interval[1, k]) if interval is not None else None,
                 mmsi_clusters=nships, bootstrap_draws=repeats,
                 accepted_draws=len(draws), skipped_missing_class=repeats-len(draws),
                 bootstrap_seed=rng_seed)
            for k, metric in enumerate(METRICS)]


def kind_of(base, full, y):
    return np.where(base == y, np.where(full == y, 'both_correct', 'harmed'),
                    np.where(full == y, 'rescued', 'both_wrong'))


def margin(score, label):
    return float(score[label] - np.max(np.delete(score, label)))


def provenance(row, manifest, index):
    return dict(id=row['id'], manifest=str(manifest), manifest_index=int(index),
                label=int(row['label']), mmsi=str(row['mmsi']),
                pass_group=row.get('pass_group'), port=row.get('port'),
                original_index=row.get('original_index'), dataset=row.get('dataset'),
                input_scale=row.get('input_scale', 1.0), canvas_size=row.get('canvas_size', 160),
                source_paths=row.get('source_paths'),
                views={pol: row.get(pol) for pol in ('vv', 'vh')})


def make_cases(viz, direction, rows, ids, refs, y, ships, scores, seeds, count):
    from PIL import Image, ImageDraw, ImageFont

    pred = {(arm, s): scores[arm, s].argmax(1) for s in seeds for arm in ARMS}
    kinds = {s: kind_of(pred['plain', s], pred['full', s], y) for s in seeds}
    cases, coverage = [], []
    manifest = OLD / direction / 'evaluation_manifest.json'
    for kind in KINDS:
        candidates = [(s, int(j)) for s in seeds for j in np.flatnonzero(kinds[s] == kind)]
        candidates.sort(key=lambda item: (
            -sum(kinds[s][item[1]] == kind for s in seeds),
            str(rows[ids[item[1]]]['id']), item[0]))
        selected, used_ships, used_labels = [], set(), set()
        # Prefer class diversity, then fill from distinct vessels within this kind.
        for diverse in (True, False):
            for seed, j in candidates:
                if len(selected) >= count:
                    break
                if ships[j] in used_ships or (diverse and int(y[j]) in used_labels):
                    continue
                selected.append((seed, j))
                used_ships.add(ships[j])
                used_labels.add(int(y[j]))
        coverage.append(dict(direction=direction, kind=kind, seed_query_events=len(candidates),
                             distinct_queries=len({j for _, j in candidates}),
                             distinct_mmsi=len({ships[j] for _, j in candidates}),
                             selected=len(selected), absent=not candidates))
        for seed, j in selected:
            row, truth = rows[ids[j]], int(y[j])
            base, full = int(pred['plain', seed][j]), int(pred['full', seed][j])
            rival = base if base != truth else full
            if rival == truth:
                other = scores['plain', seed][j].copy()
                other[truth] = -np.inf
                rival = int(other.argmax())
            pools = [[int(i) for i in refs if rows[i]['label'] == k] for k in (truth, rival)]
            selected_refs = [pool[0] for pool in pools]
            reference_rows = [rows[i] for i in selected_refs]
            agreement = sum(kinds[s][j] == kind for s in seeds)
            evidence = []
            for s in seeds:
                b, f = scores['plain', s][j], scores['full', s][j]
                evidence.append(dict(seed=s, kind=str(kinds[s][j]),
                    plain_prediction=int(pred['plain', s][j]), full_prediction=int(pred['full', s][j]),
                    plain_score=b.tolist(), full_score=f.tolist(),
                    plain_margin=margin(b, truth), full_margin=margin(f, truth),
                    margin_delta=margin(f, truth)-margin(b, truth),
                    plain_true_vs_selected_rival=float(b[truth]-b[rival]),
                    full_true_vs_selected_rival=float(f[truth]-f[rival])))
            case = dict(direction=direction, kind=kind, seed=seed, id=row['id'],
                        truth=truth, confusion=rival, agreement=int(agreement), paired_seeds=seeds,
                        query=provenance(row, manifest, ids[j]), evidence=evidence,
                        same_class_pool=pools[0], confusion_class_pool=pools[1],
                        references=[provenance(rows[i], manifest, i) for i in selected_refs],
                        reference_selection='First in declared 32-source-reference order per class; not attribution.',
                        visual_review='pending', visible_facts=None, mechanism_hypothesis=None,
                        supported_cause=None, unresolved='Human native-chip review and discriminating evidence pending.')
            dest = ROOT / direction / f'cases_s{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            try:
                filename = viz.panel(dest, direction, row, reference_rows, kind,
                                     truth, base, full, agreement)
                # The reused helper hardcodes seed17 and /3: replace its header.
                with Image.open(ROOT / filename) as original:
                    panel = original.copy()
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, panel.width, 32), fill='#f4f5f7')
                font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
                draw.text((12, 8), f'{direction} | {kind} | {row["id"]} | '
                          f'seed{seed}; agreement {agreement}/{len(seeds)}', fill='black', font=font)
                panel.save(ROOT / filename)
                case['image'] = filename
                case['gallery_status'] = 'rendered'
                for item, record in zip([row] + reference_rows, [case['query']] + case['references']):
                    views = viz.prepare_views(item)
                    record['preprocessing'] = {}
                    for p, pol in enumerate(('vv', 'vh')):
                        view = item.get(pol)
                        if view is None:
                            record['preprocessing'][pol] = dict(model_input=False)
                            continue
                        with Image.open(view['image']) as im:
                            size = list(im.size)
                        record['preprocessing'][pol] = dict(native_size=size,
                            display_gamma=0.5, display_scale=min(3.0, 280/max(size)),
                            model_input=views[p] is not None,
                            placement=views[p]['placement'] if views[p] is not None else None)
            except (OSError, ValueError, KeyError) as exc:
                case['gallery_status'] = 'incomplete'
                case['gallery_error'] = f'{type(exc).__name__}: {exc}'
            cases.append(case)
    return cases, coverage


def analyze_direction(viz, direction, args, tables, cases, statuses):
    legacy = OLD / direction
    rows = read(legacy / 'evaluation_manifest.json')
    groups = read(legacy / 'evaluation_groups.json')
    source = read(legacy / 'manifest.json')
    source_groups = read(legacy / 'groups.json')
    ids = np.asarray(groups['evaluation'], dtype=np.int64)
    refs = groups['references']
    if refs != list(range(32)) or set(refs).intersection(ids):
        raise ValueError(f'{direction}: expected 32 leading source references, disjoint from target')
    if [source[i]['id'] for i in source_groups['references']] != [rows[i]['id'] for i in refs]:
        raise ValueError(f'{direction}: source/evaluation reference order differs')
    if any(sum(rows[i]['label'] == k for i in refs) != 8 for k in range(4)):
        raise ValueError(f'{direction}: expected eight source references per common class')
    y = np.array([rows[i]['label'] for i in ids], dtype=np.int64)
    ships = np.array([str(rows[i]['mmsi']) for i in ids])
    scores, valid_scores, direction_status = {}, {}, []
    for seed in args.seeds:
        for arm in ARMS:
            run = ROOT / direction / f'{arm}_s{seed}'
            state = dict(direction=direction, arm=arm, seed=seed, run=str(run))
            missing = [name for name in ('receipt.json', 'validation.npz', 'evaluation.npz')
                       if not (run / name).is_file()]
            if missing:
                state.update(state='incomplete', detail='Missing: ' + ', '.join(missing))
            else:
                try:
                    receipt = read(run / 'receipt.json')
                    if receipt.get('test_read') is not False:
                        raise ValueError('receipt.test_read must be false (source-only checkpoint selection)')
                    _, score = load_prediction(run / 'evaluation.npz', rows, ids)
                    vy, vs = load_prediction(run / 'validation.npz', source, source_groups['validation'])
                    state.update(state='complete', detail='', receipt=receipt)
                    scores[arm, seed] = score
                    valid_scores[arm, seed] = (vy, vs)
                    tables['training'].append(dict(flatten(receipt), direction=direction,
                                                   arm=arm, seed=seed))
                except (OSError, ValueError, KeyError) as exc:
                    state.update(state='invalid', detail=f'{type(exc).__name__}: {exc}')
            direction_status.append(state)
            statuses.append(state)
    paired_seeds = [s for s in args.seeds if all((a, s) in scores for a in ARMS)]
    complete = all(s['state'] == 'complete' for s in direction_status)
    profile = dict(direction=direction, complete=complete, paired_seeds=paired_seeds,
                   available_runs=len(scores), expected_runs=2*len(args.seeds),
                   target_samples=len(ids), target_mmsi=len(set(ships)),
                   source_train_samples=len(source_groups['small']),
                   source_validation_samples=len(source_groups['validation']),
                   source_references=len(refs), manifest_root=str(legacy),
                   analyzed=bool(scores) and (complete or args.partial))
    for split, rr in [('source_train', [source[i] for i in source_groups['small']]),
                      ('source_validation', [source[i] for i in source_groups['validation']]),
                      ('source_references', [rows[i] for i in refs]),
                      ('target_evaluation', [rows[i] for i in ids])]:
        for k in range(4):
            subset = [r for r in rr if r['label'] == k]
            tables['data_profile'].append(dict(direction=direction, split=split, label=k,
                name=viz.NAMES[k], n=len(subset), mmsi=len({str(r['mmsi']) for r in subset}),
                acquisitions=len({r['pass_group'] for r in subset}),
                ports=len({r['port'] for r in subset if r.get('port')}),
                vv=sum(bool(r.get('vv')) for r in subset), vh=sum(bool(r.get('vh')) for r in subset)))
    for i in refs:
        row = rows[i]
        tables['source_references'].append(dict(direction=direction, manifest_index=i,
            id=row['id'], label=row['label'], mmsi=row['mmsi'], pass_group=row['pass_group'],
            port=row.get('port', 'unavailable'),
            vv=row['vv']['image'] if row.get('vv') else None,
            vh=row['vh']['image'] if row.get('vh') else None))
    if not profile['analyzed']:
        return profile
    for (arm, seed), score in scores.items():
        common = dict(direction=direction, arm=arm, seed=seed, paired=seed in paired_seeds)
        pred = score.argmax(1)
        tables['metrics'].append(dict(**common, n=len(y), **metrics(viz, y, score)))
        vy, vs = valid_scores[arm, seed]
        tables['validation'].append(dict(**common, n=len(vy), **metrics(viz, vy, vs)))
        cm = confusion(y, pred)
        for k in range(4):
            tp, n, called = int(cm[k, k]), int(cm[k].sum()), int(cm[:, k].sum())
            tables['per_class'].append(dict(**common, label=k, name=viz.NAMES[k], n=n,
                recall=100*tp/n if n else None, precision=100*tp/called if called else 0.,
                f1=200*tp/(n+called) if n+called else 0.))
            for p in range(4):
                tables['confusion'].append(dict(**common, truth=k, prediction=p, count=int(cm[k, p])))
        for field, name in [('port', 'per_port'), ('pass_group', 'per_acquisition')]:
            values = np.array([str(rows[i].get(field) or 'unavailable') for i in ids])
            for value in sorted(set(values)):
                mask = values == value
                tables[name].append(dict(**common, group=value, n=int(mask.sum()),
                    mmsi=len(set(ships[mask])), classes=','.join(map(str, np.unique(y[mask]))),
                    **metrics(viz, y[mask], score[mask])))
        for j, i in enumerate(ids):
            tables['predictions'].append(dict(**common, manifest_index=int(i), id=rows[i]['id'],
                mmsi=ships[j], label=int(y[j]), prediction=int(pred[j]), margin=margin(score[j], int(y[j])),
                port=rows[i].get('port', 'unavailable'), pass_group=rows[i]['pass_group']))
    if not paired_seeds:
        return profile
    tables['paired_gains'].extend(dict(direction=direction, **row) for row in bootstrap(
        y, ships, scores, paired_seeds, args.bootstrap, args.bootstrap_seed))
    for seed in paired_seeds:
        base, full = scores['plain', seed], scores['full', seed]
        bm, fm = metrics(viz, y, base), metrics(viz, y, full)
        kinds = kind_of(base.argmax(1), full.argmax(1), y)
        tables['paired_by_seed'].append(dict(direction=direction, seed=seed,
                                            **{m: fm[m]-bm[m] for m in METRICS}))
        tables['flips'].append(dict(direction=direction, seed=seed,
                                    **{k: int((kinds == k).sum()) for k in KINDS}))
        class_metrics = [r for r in tables['per_class'] if r['direction'] == direction
                         and r['seed'] == seed]
        for k in range(4):
            b = next(r for r in class_metrics if r['label'] == k and r['arm'] == 'plain')
            f = next(r for r in class_metrics if r['label'] == k and r['arm'] == 'full')
            tables['paired_class'].append(dict(direction=direction, seed=seed, label=k,
                name=viz.NAMES[k], n=b['n'],
                **{f'delta_{m}': f[m]-b[m] if b[m] is not None and f[m] is not None else None
                   for m in ('recall', 'precision', 'f1')}))
        strata = [('class', y),
                  ('port', np.array([str(rows[i].get('port') or 'unavailable') for i in ids])),
                  ('acquisition', np.array([str(rows[i]['pass_group']) for i in ids]))]
        for field, values in strata:
            for value in sorted(set(values)):
                mask = values == value
                bm, fm = metrics(viz, y[mask], base[mask]), metrics(viz, y[mask], full[mask])
                tables['paired_strata'].append(dict(direction=direction, seed=seed, stratum=field,
                    group=str(value), n=int(mask.sum()), mmsi=len(set(ships[mask])),
                    **{f'delta_{m}': fm[m]-bm[m] for m in METRICS},
                    **{k: int((kinds[mask] == k).sum()) for k in KINDS}))
    new_cases, coverage = make_cases(viz, direction, rows, ids, refs, y, ships,
                                      scores, paired_seeds, args.cases_per_kind)
    cases.extend(new_cases)
    tables['case_coverage'].extend(coverage)
    return profile


def write_reports(args, tables, cases, statuses, profiles):
    review_path = ROOT / 'reviewed_cases.json'
    review = read(review_path) if review_path.exists() else None
    if review:
        for case in cases:
            if case['id'] in review['ids']:
                case['visual_review'] = 'reviewed; see ' + review['notes']
    for direction in args.directions:
        paired_seeds = next(p['paired_seeds'] for p in profiles if p['direction'] == direction)
        for arm in ARMS:
            selected = [r for r in tables['metrics'] if r['direction'] == direction
                        and r['arm'] == arm and r['seed'] in paired_seeds]
            if selected:
                tables['summary'].append(dict(direction=direction, arm=arm, paired_seeds=len(selected),
                    seeds=','.join(str(r['seed']) for r in selected),
                    **{f'{m}_{stat}': float(np.mean([r[m] for r in selected])) if stat == 'mean'
                       else sd([r[m] for r in selected]) for m in METRICS for stat in ('mean', 'std')}))
    tables['run_status'] = [{k: v for k, v in s.items() if k != 'receipt'} for s in statuses]
    for name, table in tables.items():
        csv_table(ROOT / f'{name}.csv', table)
    complete = all(p['complete'] for p in profiles)
    gallery_complete = all(c['gallery_status'] == 'rendered' for c in cases)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        result_root=str(ROOT), historical_manifest_root=str(OLD),
        selected_directions=args.directions, requested_seeds=args.seeds, partial=args.partial,
        prediction_exports_complete=complete,
        both_directions_three_seeds_complete=complete and set(args.directions) == set(DIRECTIONS)
            and set(args.seeds) == set(SEEDS),
        status='complete_exports_review_pending' if complete and gallery_complete else 'incomplete',
        analysis_complete=False, native_visual_review='pending',
        gallery_complete=gallery_complete, profiles=profiles, runs=statuses,
        metrics=tables['summary'], paired_gains=tables['paired_gains'], cases=cases,
        definitions=dict(units='OA, BA, macro F1 in percent; gains in percentage points',
            ba='Mean recall over ground-truth classes present in the reported subset',
            macro_f1='Mean over fixed four common classes; undefined class F1 is zero',
            aggregation='Mean of per-seed metrics, never an ensemble; summary uses paired seeds only',
            uncertainty='Paired MMSI cluster bootstrap, fixed observed seeds; not seed-population uncertainty',
            case_selection='Outcome-selected, agreement-ranked; distinct vessels and class diversity per kind',
            evidence='Predictions and native inputs only; no internal attribution or verified physical cause',
            receipt_test_read='Training receipt declaration only; analysis itself reads exported target labels'))
    if review:
        summary['native_visual_review'] = review
        covered = {(c['direction'], c['kind']) for c in cases if c['id'] in review['ids']}
        summary['analysis_complete'] = complete and gallery_complete and all(
            (d, k) in covered for d in args.directions for k in KINDS)
        if summary['analysis_complete']:
            summary['status'] = 'complete_with_native_review'
    write_json(ROOT / 'summary.json', summary)
    if summary['analysis_complete'] and (ROOT / 'status.json').exists():
        state = read(ROOT / 'status.json')
        state.update(analysis_pending=False, analysis_complete=True,
                     state='training_evaluation_and_analysis_complete')
        write_json(ROOT / 'status.json', state)
    write_json(ROOT / 'cases.json', cases)
    case_blocks = []
    for c in cases:
        case_blocks.append(f'### {c["direction"]} / {c["kind"]} / {c["id"]} / seed {c["seed"]}\n\n'
            + (f'![native VV/VH]({c["image"]})\n\n' if c.get('image') else '')
            + markdown_table(c['evidence']) + '\n\n'
            + f'Query manifest index: {c["query"]["manifest_index"]}; MMSI: {c["query"]["mmsi"]}. '
            + f'Reference indices: {[r["manifest_index"] for r in c["references"]]}. '
            + f'Gallery: {c["gallery_status"]}; {c.get("gallery_error", "")}.\n\n'
            + ('Native visual observations and interpretation: see CASE_NOTES_ZH.md.\n'
               if c['visual_review'].startswith('reviewed') else
               'Visual facts and interpretation: pending native-chip review.\n'))
    template = Template((ROOT / 'ANALYSIS_TEMPLATE_ZH.md').read_text(encoding='utf-8'))
    report = template.substitute(status=summary['status'],
        scope=json.dumps(profiles, ensure_ascii=False, indent=2),
        statuses=markdown_table(tables['run_status']), metrics=markdown_table(tables['summary']),
        gains=markdown_table(tables['paired_gains']), flips=markdown_table(tables['flips']),
        profiles=markdown_table(tables['data_profile']),
        coverage=markdown_table(tables['case_coverage']), cases='\n'.join(case_blocks) or '(No eligible cases.)')
    if review:
        report = ('# 本轮结果与原图回看已完成\n\n'
                  '简明结论见[RESULTS_ZH.md](RESULTS_ZH.md)，10组人工原图回看见'
                  '[CASE_NOTES_ZH.md](CASE_NOTES_ZH.md)。下文为自动统计与案例明细，'
                  '其中待解释字段不覆盖上述独立人工记录。\n\n' + report)
    (ROOT / 'FINDINGS_ZH.md').write_text(report, encoding='utf-8')
    cards = []
    for c in cases:
        title = html.escape(f'{c["direction"]} / {c["kind"]} / {c["id"]} / seed {c["seed"]}')
        image = f'<img loading="lazy" src="{html.escape(c["image"], quote=True)}">' if c.get('image') else ''
        cards.append(f'<section><h2>{title}</h2>{image}<pre>'
                     + html.escape(json.dumps(c, ensure_ascii=False, indent=2)) + '</pre></section>')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Cross-dataset fine-tuning analysis</title><style>'
            'body{max-width:1120px;margin:24px auto;padding:0 16px;font:15px Arial;color:#222}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%;height:auto}'
            'section{border-top:1px solid #ccc;padding:16px 0}h1{font-size:24px}h2{font-size:18px}'
            '</style><h1>Cross-dataset fine-tuning: current exports only</h1><p>'
            + html.escape(summary['status']) + '</p>'
            '<p><a href="RESULTS_ZH.md">简明结果</a> | <a href="CASE_NOTES_ZH.md">原图回看</a></p>'
            '<p><a href="FINDINGS_ZH.md">FINDINGS_ZH.md</a> | <a href="summary.json">summary.json</a>'
            ' | <a href="paired_strata.csv">paired strata</a> | <a href="source_references.csv">32 source references</a></p>'
            '<pre>' + html.escape(report.split('<!-- CASES -->')[0]) + '</pre>'
            + ''.join(cards) + '</html>')
    (ROOT / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(status=summary['status'], profiles=profiles,
                          paired_gains=tables['paired_gains']), ensure_ascii=False, indent=2))
    return 2 if (any(s['state'] == 'invalid' for s in statuses) or not gallery_complete
                 or (not args.partial and not complete)) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directions', nargs='+', choices=DIRECTIONS, default=list(DIRECTIONS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--partial', action='store_true', help='Report available current runs and complete seed pairs')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261112)
    parser.add_argument('--cases-per-kind', type=int, default=2)
    args = parser.parse_args()
    if args.bootstrap < 1 or args.cases_per_kind < 1:
        parser.error('--bootstrap and --cases-per-kind must be positive')
    args.seeds = sorted(set(args.seeds))
    args.directions = list(dict.fromkeys(args.directions))
    viz = load_visuals()
    names = ('summary', 'metrics', 'validation', 'training', 'per_class', 'per_port',
             'per_acquisition', 'confusion', 'paired_gains', 'paired_by_seed', 'paired_strata', 'paired_class',
             'flips', 'predictions', 'data_profile', 'source_references', 'case_coverage')
    tables = {name: [] for name in names}
    cases, statuses, profiles = [], [], []
    for direction in args.directions:
        try:
            profiles.append(analyze_direction(viz, direction, args, tables, cases, statuses))
        except (OSError, ValueError, KeyError) as exc:
            statuses.append(dict(direction=direction, arm='manifest', seed=None,
                                 state='invalid', detail=f'{type(exc).__name__}: {exc}'))
            profiles.append(dict(direction=direction, complete=False, paired_seeds=[], analyzed=False))
    return write_reports(args, tables, cases, statuses, profiles)


if __name__ == '__main__':
    raise SystemExit(main())
