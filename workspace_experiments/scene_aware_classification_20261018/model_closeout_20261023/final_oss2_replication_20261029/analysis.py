"""CPU four-class analysis. Formal predictions require explicit main-agent READY."""
import argparse
from collections import Counter
import html
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

from protocol import HERE, PACKAGE, CLASSES, CLASS_INDICES, SEEDS, COUNTS, BUDGET, COEFFICIENT

# Load a private helper module instance; changes below never edit sibling files.
spec = importlib.util.spec_from_file_location('oss2_completed_cpu_helpers',
    HERE.parent/'spatial_m12_full_20261029/analysis.py')
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
base = full.base
METRICS, STATES = base.METRICS, base.STATES
CONDITIONS = ('full_fit', 'plain_fit', 'full_adapt', 'plain_adapt')
PRIMARY = 'full_adapt_minus_plain_adapt'
COMPARISONS = {
    PRIMARY: ('plain_adapt', 'full_adapt'),
    'full_fit_minus_plain_fit': ('plain_fit', 'full_fit'),
    'full_adapt_minus_full_fit': ('full_fit', 'full_adapt'),
    'plain_adapt_minus_plain_fit': ('plain_fit', 'plain_adapt'),
}
SCOPE_NOTE = (
    'OSS2 final spatial M12+M3 versus plainR50, explicit FOUR classes mapped from original [0,2,3,4]. '
    'New architecture on the historical fold0, NOT a new blind test. Evaluation: 378 queries, '
    '369 ships, five pass groups. Reference bank: 32 actual source-fit identities, eight per class. '
    'Plain has NO reference contribution. Full versus plain includes architecture and training-objective '
    'differences; no pure-architecture or physical-causality claim. Fit160/adapt96 steps per chain. '
    'Seeds17/29/43 change order under shared initialization, NOT three independent initializations. '
    'OA/BA are percent; deltas are percentage points; NLL is natural log; Brier sums FOUR classes. '
    'BA averages present classes only; absent groups are blank. SD is sample SD across three orders. '
    'Paired pass bootstrap resamples five whole passes and averages paired fixed-seed differences; '
    'its interval is conditional on these fitted models and only five clusters, not a training-uncertainty CI. '
    'Repeated observations, seeds and references are not independent evidence. '
    'Validation228/selection194 are recorded available only; no checkpoint selection. '
    'Outcome-selected native cases do not estimate population proportions; generated images remain not_reviewed.'
)
CASE_RULE = (
    'Primary full_adapt versus plain_adapt only. Deterministic selection: one available net-rescue, '
    'net-harm, consistent-CC and consistent-WW query, then missing vv_only and vh_only coverage where '
    'available, then fill to eight using the completed-round helper ranking. Never invent absent states. '
    'Representative seed is the largest absolute fixed-rival margin change within its state. '
    'Fixed rival is the strongest nontruth class in mean full_fit logits across three orders, shared by all comparisons.'
)


def per_sample(score, y):
    z = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if z.shape != (len(y), 4) or not np.isin(y, range(4)).all():
        raise ValueError('Expected logits[N,4] and explicit labels 0..3.')
    z = z-z.max(1, keepdims=True)
    logp = z-np.log(np.exp(z).sum(1, keepdims=True))
    prob = np.exp(logp)
    return dict(prediction=np.asarray(score).argmax(1), NLL=-logp[np.arange(len(y)), y],
                Brier=((prob-np.eye(4)[y])**2).sum(1))


def group_masks(rows):
    for field in ('label', 'pass_group', 'available'):
        values = np.array([str(r['analysis_available'] if field == 'available' else r[field]) for r in rows])
        groups = list(map(str, range(4))) if field == 'label' else sorted(set(values))
        for group in groups:
            yield field, group, values == group


# Reuse paired/group/transition and pixel-exact panel helpers with four-class globals.
base.CLASSES = full.CLASSES = CLASSES
base.per_sample = full.per_sample = per_sample
full.group_masks = group_masks
metrics, moments = base.metrics, base.moments


def input_paths(root=HERE):
    predictions, training, evaluation = {}, {}, {}
    for condition in CONDITIONS:
        arm, stage = condition.split('_')
        for seed in SEEDS:
            key = (condition, seed)
            predictions[key] = root/'predictions'/f'evaluation_{arm}_{stage}_s{seed}.npz'
            training[key] = root/'training'/f'{arm}_s{seed}'/stage/'training.json'
            evaluation[key] = root/'evaluation'/f'{arm}_{stage}_s{seed}.json'
    return predictions, training, evaluation


def load_score(path, rows):
    with np.load(path, allow_pickle=False) as pack:
        ids, labels = pack['ids'].astype(str), pack['y']
        score = np.asarray(pack['score'], dtype=np.float64)
        classes = pack['classes'].astype(str).tolist()
    expected = [str(r['id']) for r in rows]
    if ids.shape != (len(rows),) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError(f'{path}: IDs must exactly cover the evaluation cohort.')
    if classes != list(CLASSES) or labels.shape != ids.shape or labels.dtype.kind not in 'iu':
        raise ValueError(f'{path}: four-class names/order or integer labels disagree.')
    if score.shape != (len(rows), 4) or not np.isfinite(score).all():
        raise ValueError(f'{path}: expected finite logits[N,4].')
    lookup = {value: i for i, value in enumerate(ids)}
    order = [lookup[value] for value in expected]
    if not np.array_equal(labels[order], [r['label'] for r in rows]):
        raise ValueError(f'{path}: labels disagree with evaluation manifest.')
    return score[order]


def require_receipts(root=HERE):
    predictions, training_paths, evaluation_paths = input_paths(root)
    manifest_paths = {key: root/'manifests'/f'{key}.json' for key in (*COUNTS, 'references')}
    gate_path = root/'gpu_checks.json'
    required = [gate_path, *manifest_paths.values(), *training_paths.values(),
                *evaluation_paths.values(), *predictions.values()]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError('Incomplete inputs; no NPZ opened:\n'+'\n'.join(missing))
    gate = base.read_json(gate_path)
    if gate.get('status') != 'passed' or gate.get('class_indices') != list(CLASS_INDICES):
        raise ValueError('gpu_checks must pass with the explicit four-class mapping.')
    receipts = dict(gpu=dict(path=str(gate_path), receipt=gate), training={}, evaluation={})
    for key, path in training_paths.items():
        condition, seed = key
        arm, stage = condition.split('_')
        record = base.read_json(path)
        epochs, batch_size, steps = BUDGET[stage]
        expected = dict(arm=arm, stage=stage, seed=seed, steps=steps, completed_epochs=epochs,
            batch_size=batch_size, source_count=478 if stage == 'fit' else 241, reference_count=32,
            classes=list(CLASSES), class_indices=list(CLASS_INDICES), validation_selection=False,
            frozen_unchanged=True)
        if any(record.get(k) != value for k, value in expected.items()) or record.get('smoke_only', False):
            raise ValueError(f'{path}: incomplete or mismatched {steps}-step/{epochs}-epoch receipt.')
        if stage == 'adapt' and record.get('coefficient') != (COEFFICIENT if arm == 'full' else 0.):
            raise ValueError(f'{path}: wrong adaptation objective.')
        for name, tolerance in (('save_replay_error', 1e-5), ('native_cache_error', 1e-4)):
            value = record.get(name)
            if not isinstance(value, (int, float)) or not 0 <= value < tolerance:
                raise ValueError(f'{path}: failed/missing {name}.')
        receipts['training'][f'{condition}_s{seed}'] = dict(path=str(path), receipt=record)
    for (condition, seed), path in evaluation_paths.items():
        arm, stage = condition.split('_')
        record = base.read_json(path)
        if (any(record.get(k) != v for k, v in dict(arm=arm, stage=stage, seed=seed,
                primary='evaluation378', checkpoint_selection=False).items()) or
                not 0 <= record.get('native_cache_error', float('inf')) < 1e-4):
            raise ValueError(f'{path}: evaluation gate failed.')
        receipts['evaluation'][f'{condition}_s{seed}'] = dict(path=str(path), receipt=record)
    return predictions, manifest_paths, receipts


def load_inputs(root=HERE, *, ready=False):
    if not ready:
        raise PermissionError('Explicit main-agent READY is required before reading formal predictions.')
    predictions, manifests, receipts = require_receipts(root)
    cohorts = {name: base.read_json(path) for name, path in manifests.items()}
    for split, rows in cohorts.items():
        count = 32 if split == 'references' else COUNTS[split]
        if len(rows) != count or len({str(r['id']) for r in rows}) != count:
            raise ValueError(f'{split}: count or duplicate IDs disagree.')
        for row in rows:
            if (type(row['label']) is not int or not 0 <= row['label'] < 4 or
                    row['original_class_id'] != CLASS_INDICES[row['label']] or
                    row['class_name'] != CLASSES[row['label']]):
                raise ValueError(f'{split}: original-to-four-class mapping disagrees.')
            row['id'] = str(row['id'])
            row['port'] = 'opensarship2'
            row['analysis_available'] = base.polarization(row)
    evaluation, refs = cohorts['evaluation'], cohorts['references']
    if len({r['mmsi'] for r in evaluation}) != 369 or len({r['pass_group'] for r in evaluation}) != 5:
        raise ValueError('Expected 378 evaluation queries / 369 ships / five passes.')
    if len({r['mmsi'] for r in refs}) != 32 or Counter(r['label'] for r in refs) != dict.fromkeys(range(4), 8):
        raise ValueError('Expected 32 independent source-fit references, eight per class.')
    fit = {r['id']: r for r in cohorts['source_fit']}
    if any(r['id'] not in fit or r['mmsi'] != fit[r['id']]['mmsi'] or
           r['label'] != fit[r['id']]['label'] for r in refs):
        raise ValueError('References must preserve source-fit identities and labels.')
    scores = {key: load_score(path, evaluation) for key, path in predictions.items()}
    return cohorts, scores, receipts


def pair_events(rows, scores, old, new):
    supplied = {**scores, **{('legacy_before', seed): scores['full_fit', seed] for seed in SEEDS}}
    events = full.pair_events(rows, supplied, old, new)
    by_id = {r['id']: i for i, r in enumerate(rows)}
    for event in events:
        index, truth, rival = by_id[event['id']], event['label'], event['fixed_rival']
        for condition in CONDITIONS:
            score = scores[condition, event['seed']][index]
            event[condition+'_prediction'] = int(score.argmax())
            event[condition+'_fixed_margin'] = float(score[truth]-score[rival])
            event[condition+'_top_rival_margin'] = float(score[truth]-np.delete(score, truth).max())
    return events


def pass_bootstrap(rows, scores, old_name, new_name, repeats=2000, seed=20261029):
    """Completed-round paired pass bootstrap, with four explicit class columns."""
    y = np.array([r['label'] for r in rows], int)
    passes, inverse = np.unique([r['pass_group'] for r in rows], return_inverse=True)
    differences = []
    for order_seed in SEEDS:
        old, new = [per_sample(scores[name, order_seed], y) for name in (old_name, new_name)]
        differences.append(np.column_stack([(new['prediction'] == y).astype(float)-(old['prediction'] == y),
                            new['NLL']-old['NLL'], new['Brier']-old['Brier']]))
    delta = np.mean(differences, axis=0)
    totals, counts = np.zeros((len(passes), 3)), np.zeros((len(passes), 4))
    correct = np.zeros_like(counts)
    np.add.at(totals, inverse, delta)
    np.add.at(counts, (inverse, y), 1)
    np.add.at(correct, (inverse, y), delta[:, 0])
    rng, draws, missing = np.random.default_rng(seed), [], 0
    for _ in range(repeats):
        selected = rng.integers(0, len(passes), len(passes))
        n, numerator = counts[selected].sum(0), totals[selected].sum(0)
        present = n > 0
        missing += int(not present[np.unique(y)].all())
        draws.append([100*numerator[0]/n.sum(), 100*np.mean(correct[selected].sum(0)[present]/n[present]),
                      numerator[1]/n.sum(), numerator[2]/n.sum()])
    draws = np.asarray(draws)
    return dict(method='paired whole-pass percentile bootstrap; all paired queries and fixed seeds retained',
        repeats=repeats, bootstrap_seed=seed, pass_count=len(passes), passes=passes.tolist(),
        missing_class_replicates=missing, ba_rule='present classes per resample; missing-class draws counted',
        interpretation='Conditional on fixed models and historical cohort; only five clusters in formal evaluation.',
        metrics={m: dict(point=float(np.mean([metrics(scores[new_name, s], y)[m]-metrics(scores[old_name, s], y)[m]
                      for s in SEEDS])), low=float(np.quantile(draws[:, i], .025)),
                      high=float(np.quantile(draws[:, i], .975))) for i, m in enumerate(METRICS)})


def summarize(rows, scores, repeats=2000):
    y = np.array([r['label'] for r in rows], int)
    tables = dict(metrics=[], groups=[], groups_summary=[], seed_events=[])
    for condition in CONDITIONS:
        for seed in SEEDS:
            tables['metrics'].append(dict(condition=condition, seed=seed, **metrics(scores[condition, seed], y)))
    result = dict(conditions={c: {m: moments([r[m] for r in tables['metrics'] if r['condition'] == c])
                                  for m in METRICS} for c in CONDITIONS}, comparisons={})
    event_sets = {}
    for comparison, (old, new) in COMPARISONS.items():
        events = pair_events(rows, scores, old, new)
        detail, aggregate = full.summarize_comparison(rows, scores, old, new, events)
        for name, values in (('groups', detail), ('groups_summary', aggregate), ('seed_events', events)):
            tables[name].extend(dict(comparison=comparison, **value) for value in values)
        result['comparisons'][comparison] = dict(reference=old, alternative=new, overall=aggregate[0],
            transitions=base.transition_summary(events), pass_bootstrap=pass_bootstrap(rows, scores, old, new, repeats))
        event_sets[comparison] = events
    return result, tables, event_sets


def select_cases(events, count=8):
    ranked = full.select_cases(events, count=len({e['id'] for e in events}))
    selected, used = [], set()
    def add(candidate):
        if candidate and candidate['id'] not in used and len(selected) < count:
            selected.append(candidate)
            used.add(candidate['id'])
    for state in STATES:
        add(next((e for e in ranked if e['selection_category'] == state), None))
    for availability in ('vv_only', 'vh_only'):
        if not any(e['available'] == availability for e in selected):
            add(next((e for e in ranked if e['available'] == availability and e['id'] not in used), None))
    for event in ranked:
        add(event)
    return selected


def case_details(choice, index, query, refs, scores):
    truth, rivals = query['label'], {choice['fixed_rival']}
    for seed in SEEDS:
        for condition in CONDITIONS:
            value = scores[condition, seed][index].copy()
            value[truth] = -np.inf
            rivals.add(int(value.argmax()))
    allowed = [r for r in refs if base.eligible(query, r)]
    detail = dict(choice, comparison=PRIMARY, visual_status='not_reviewed', selection_rule=CASE_RULE,
        reference_use=dict(full='actual fixed-bank candidates; weights/causal contribution not inferred', plain='NONE'),
        fixed_bank_ids=[r['id'] for r in refs], eligible_reference_ids=[r['id'] for r in allowed],
        true_reference_ids=[r['id'] for r in allowed if r['label'] == truth],
        rival_reference_ids=[r['id'] for r in allowed if r['label'] in rivals], rival_classes=sorted(rivals),
        panels={}, per_seed=[], image_observations=None, model_evidence_interpretation=None,
        remaining_uncertainty='Await main-agent native-image review; logits are not physical mechanism evidence.')
    for seed in SEEDS:
        conditions = {}
        for name in CONDITIONS:
            value = scores[name, seed][index]
            sample = per_sample(value[None], np.array([truth]))
            conditions[name] = dict(prediction=int(value.argmax()), logits=value.tolist(),
                fixed_margin=float(value[truth]-value[choice['fixed_rival']]),
                top_rival_margin=float(value[truth]-np.delete(value, truth).max()),
                NLL=float(sample['NLL'][0]), Brier=float(sample['Brier'][0]))
        detail['per_seed'].append(dict(seed=seed, conditions=conditions))
    roles = ['QUERY']+['FULL TRUE REF' if r['label'] == truth else
                       'FULL RIVAL REF' if r['label'] in rivals else 'FULL OTHER REF' for r in allowed]
    return detail, [query, *allowed], roles


def gallery(out, cohorts, scores, events, native):
    folder = out/'cases'
    folder.mkdir(parents=True, exist_ok=True)
    index = {r['id']: (i, r) for i, r in enumerate(cohorts['evaluation'])}
    cases, sources, blocks = [], [], []
    chosen = select_cases(events)
    for number, choice in enumerate(chosen, 1):
        i, query = index[choice['id']]
        case, records, roles = case_details(choice, i, query, cohorts['references'], scores)
        title = f"{number}. {choice['selection_category']} / {query['id']} / representative seed {choice['seed']}"
        blocks += ['<section><h2>'+html.escape(title)+'</h2><p>Truth: '+CLASSES[query['label']]+
                   '; polarization: '+choice['available']+'; pass: '+html.escape(query['pass_group'])+
                   '; not_reviewed. References shown for FULL only; PLAIN has NO reference contribution.</p>',
                   '<table><tr><th>Seed</th>'+''.join('<th>'+c+': prediction / top-rival margin / fixed margin</th>'
                                                       for c in CONDITIONS)+'</tr>']
        for seed_detail in case['per_seed']:
            cells = []
            for name in CONDITIONS:
                value = seed_detail['conditions'][name]
                cells.append(f"<td>{CLASSES[value['prediction']]} / {value['top_rival_margin']:.6f} / {value['fixed_margin']:.6f}</td>")
            blocks.append(f"<tr><td>{seed_detail['seed']}</td>"+''.join(cells)+'</tr>')
        blocks.append('</table>')
        for pol in ('vv', 'vh'):
            path = folder/f'case_{number:02d}_{pol}.png'
            receipts = base.panel(records, roles, pol, path, native)
            expected = [r.get(pol) is not None for r in records]
            if [r['available'] for r in receipts] != expected:
                raise ValueError('Actual selected native availability disagrees with converted manifest.')
            for receipt in receipts:
                receipt.update(case_id=query['id'], seed=choice['seed'], panel=path.name,
                    plain_reference_effect=False, applicable_arm='query' if receipt['role'] == 'QUERY' else 'full',
                    usable_with_query_polarization=bool(receipts[0]['available'] and receipt['available']))
            sources.extend(receipts)
            case['panels'][pol] = str(path)
            blocks.append(f'<h3>{pol.upper()}</h3><div class="image"><img src="{path.name}" alt="{html.escape(title)}"></div>')
        blocks.append('<details><summary>Logits, margins, actual bank IDs</summary><pre>'+
                      html.escape(base.json.dumps(case, indent=2))+'</pre></details></section>')
        cases.append(case)
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>OSS2 four-class native cases</title>"
        "<style>body{font:14px Arial;margin:24px}h1{font-size:22px}h2{font-size:18px}"
        "section{border-top:1px solid #aaa;margin-top:24px}td,th{border:1px solid #bbb;padding:6px}"
        "table{border-collapse:collapse}.image{overflow:auto}img{max-width:none;image-rendering:pixelated}"
        "pre{white-space:pre-wrap}</style><h1>OSS2 full_adapt versus plain_adapt</h1><p>"+
        html.escape(SCOPE_NOTE)+'</p><p>'+html.escape(CASE_RULE)+
        '</p><p>One source pixel per image pixel, no resizing or contrast stretch. VV/VH separated. '
        'Original display chip, exact crop/pad160, duplicate OBB overlay. Missing polarization is unavailable, '
        'not an observed black chip. All eligible actual fixed-bank references are shown for FULL only.</p>'+
        ''.join(blocks)+'</html>')
    (folder/'index.html').write_text(page, encoding='utf-8')
    base.dump(folder/'cases.json', cases)
    base.dump(folder/'native_sources.json', sources)
    present = {e['selection_category'] for e in chosen}
    base.dump(folder/'selection_receipt.json', dict(query_count=len(cases), native_panels=2*len(cases),
        selected_ids=[c['id'] for c in cases], rule=CASE_RULE,
        unavailable_consistent_or_net_states=[state for state in STATES if state not in present],
        selected_availability=dict(Counter(c['available'] for c in cases)), visual_status='not_reviewed'))
    return cases


def analyze(root=HERE, *, ready=False):
    cohorts, scores, receipts = load_inputs(root, ready=ready)
    rows = cohorts['evaluation']
    results, tables, events = summarize(rows, scores)
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    coverage = {name: dict(n=len(group), ships=len({r['mmsi'] for r in group}),
        passes=dict(Counter(r['pass_group'] for r in group)),
        classes={CLASSES[c]: sum(r['label'] == c for r in group) for c in range(4)},
        availability=dict(Counter(r['analysis_available'] for r in group))) for name, group in cohorts.items()}
    summary = dict(scope=SCOPE_NOTE, primary_comparison=PRIMARY, classes=list(CLASSES),
        original_class_indices=list(CLASS_INDICES), coverage=coverage, receipts=receipts,
        results=results, visual_status='not_reviewed')
    for name, values in tables.items():
        base.write_csv(out/f'{name}.csv', values)
    base.write_csv(out/'paired_metrics.csv', [r for r in tables['groups'] if r['dimension'] == 'all'])
    base.dump(out/'summary.json', summary)
    native = full.module_at('oss2_native_input_cpu', PACKAGE/'sar_m123/input.py')
    cases = gallery(out, cohorts, scores, events[PRIMARY], native)
    lines = ['# OSS2 four-class numerical results', '', SCOPE_NOTE, '',
        '| Condition | OA mean / SD | BA mean / SD | NLL mean / SD | Brier mean / SD |',
        '|---|---:|---:|---:|---:|']
    for name, values in results['conditions'].items():
        lines.append('| '+name+' | '+' | '.join(f"{values[m]['mean']:.6f} / {values[m]['sd']:.6f}" for m in METRICS)+' |')
    lines += ['', '| Paired comparison | Metric | Mean delta | Order SD | Pass CI low | Pass CI high |',
              '|---|---|---:|---:|---:|---:|']
    for name, value in results['comparisons'].items():
        for metric in METRICS:
            row, interval = value['overall'], value['pass_bootstrap']['metrics'][metric]
            lines.append(f"| {name} | {metric} | {row['delta_'+metric+'_mean']:.6f} | "
                         f"{row['delta_'+metric+'_sd']:.6f} | {interval['low']:.6f} | {interval['high']:.6f} |")
    for name, value in results['comparisons'].items():
        lines += ['', f"{name}: {value['pass_bootstrap']['missing_class_replicates']}/2000 bootstrap draws omit an originally present class.", '']
    lines += ['| Comparison | State | Seed events | Unique queries | Unique ships |', '|---|---|---:|---:|---:|']
    for name, value in results['comparisons'].items():
        for state in STATES:
            item = value['transitions'][state]
            lines.append(f"| {name} | {state} | {item['seed_events']} | {item['unique_ids']} | {item['unique_ships']} |")
    lines += ['', f'{len(cases)} selected queries / {2*len(cases)} separate native VV/VH panels generated; NOT REVIEWED.',
        'See cases/cases.json for all four endpoints, three orders, actual eligible refs and margins. '
        'Plain reference contribution is NONE. Numeric shifts cannot identify SAR physical causes. '
        'Main-agent review must add visible facts, model-supported evidence, useful conditions and unresolved gaps.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    base.dump(out/'completion.json', dict(numerical_analysis='complete', visual_status='not_reviewed',
        evaluation_queries=378, evaluation_ships=369, pass_groups=5, predictions=12,
        queries=len(cases), panels=2*len(cases), torch_imported='torch' in sys.modules))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready', action='store_true', help='Use only after the main agent explicitly says READY.')
    args = parser.parse_args(argv)
    if not args.ready:
        parser.error('Explicit READY is required; no formal NPZ has been read.')
    analyze(ready=True)


if __name__ == '__main__':
    main()
