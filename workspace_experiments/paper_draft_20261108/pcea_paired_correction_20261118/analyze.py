"""CPU-only cached analysis, after all six new fits AND evaluations complete.

Run: python -B analyze.py
Share aggregate/ only. cases_local.json and gallery/ contain private sample IDs.
No model import, inference, checkpoint selection, or test-pixel access.
"""
from collections import Counter
import html
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'pcea_class_local_20261117'
EXTRA = ROOT.parent / 'pcea_class_local_expansion_20261118'
SEEDS = (17, 29, 43)
ARMS = ('full', 'generic', 'paired_ce', 'guided')
NEW_ARMS = ARMS[2:]
METRICS = ('oa', 'ba', 'macro_f1', 'nll')
KINDS = ('rescued', 'harmed', 'stable_correct', 'stable_wrong')
CONTRASTS = (('paired_ce', 'on'), ('guided', 'off'), ('generic', 'on'), ('full', 'on'))


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The strength helper's main imports a model; only its pure CPU functions are used.
strength = load_helper('paired_strength_analysis_cpu',
                       ROOT.parent / 'pcea_writeback_strength_20261118/analyze.py')
native = load_helper('paired_native_analysis_cpu',
                     ROOT.parent / 'ports24_sync_20261112/analyze.py')
read, write = native.read_json, native.write_json


def cache_path(split, arm, seed, mode):
    if arm in NEW_ARMS:
        name = ('inner.npz' if mode == 'on' else 'inner_off.npz') if split == 'inner' else f'{split}_{mode}.npz'
        return ROOT / 'fits' / f'{arm}_s{seed}' / name
    if split == 'expansion':
        return EXTRA / f'{arm}_s{seed}_{mode}_local.npz'
    return OLD / 'fits' / f'{arm}_s{seed}' / f'{split}_{mode}.npz'


def required_caches():
    for split in ('validation', 'expansion', 'inner'):
        for arm in NEW_ARMS if split == 'inner' else ARMS:
            for seed in SEEDS:
                for mode in ('on',) if arm == 'full' else ('on', 'off'):
                    yield split, arm, seed, mode


def completion():
    paths = [ROOT / 'fits' / f'{a}_s{s}' / 'receipt.json' for a in NEW_ARMS for s in SEEDS]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Six new receipts required before analysis:\n' + '\n'.join(map(str, missing)))
    selections = []
    for arm in ARMS:
        for seed in SEEDS:
            folder = (ROOT if arm in NEW_ARMS else OLD) / 'fits' / f'{arm}_s{seed}'
            receipt, config = read(folder / 'receipt.json'), read(folder / 'config.json')
            curve = read(folder / 'curve.json')
            if (receipt['arm'] != arm or receipt['seed'] != seed or config['steps'] != 240
                    or not curve or curve[-1]['step'] != 240
                    or config['selection'] != 'source-inner BA/OA/-NLL, step0 eligible'
                    or receipt['best']['step'] not in [0] + [r['step'] for r in curve]
                    or (arm in NEW_ARMS and receipt.get('source_only') is not True)):
                raise ValueError(f'Incomplete or inconsistent source selection: {folder}')
            selections.append(dict(arm=arm, seed=seed, best_step=receipt['best']['step'],
                                   updates=config['steps'], reused=arm not in NEW_ARMS))
    missing = [cache_path(*key) for key in required_caches() if not cache_path(*key).is_file()]
    if missing:
        raise FileNotFoundError('Evaluation incomplete; no analysis outputs:\n' + '\n'.join(map(str, missing)))
    return selections


def metadata():
    original, groups = read(strength.STORE / 'manifest.json'), read(strength.STORE / 'groups.json')
    extra = read(EXTRA / 'evaluation_manifest_local.json')
    n = read(EXTRA / 'input_scope.json')['n']
    inner = read(ROOT / 'split_local.json')['inner']
    if n != 198 or len(groups['validation']) != 160 or len(inner) != 1000 or len(groups['references']) != 16:
        raise ValueError('Expected frozen 160/198 external queries, 1000 inner queries and 16 references')
    if not np.array_equal(inner, read(OLD / 'split_local.json')['inner']):
        raise ValueError('Source inner split changed')
    refs = list(range(n, n + 16))
    for old_id, extra_id in zip(groups['references'], refs):
        if any(original[old_id][k] != extra[extra_id][k] for k in ('id', 'corpus_global_row', 'label', 'mmsi')):
            raise ValueError('Expansion bank differs from the fixed source16')
    context = dict(validation=dict(rows=original, ids=np.asarray(groups['validation'], np.int64),
                                   references=groups['references'], recover=True),
                   expansion=dict(rows=extra, ids=np.arange(n), references=refs, recover=False),
                   inner=dict(rows=original, ids=np.asarray(inner, np.int64),
                              references=groups['references'], recover=True))
    for item in context.values():
        ids = item['ids']
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Duplicate expected manifest indices')
        item['label'] = np.array([item['rows'][int(i)]['label'] for i in ids], np.int64)
    return context


def metric_values(item):
    values = native.metrics(item['label'], item['score'])
    return {key: values[key] for key in METRICS}


def source_on_off(cache, context, selections):
    steps = {(r['arm'], r['seed']): r['best_step'] for r in selections}
    records = []
    for arm in NEW_ARMS:
        for seed in SEEDS:
            on, off = (cache['inner', arm, seed, mode] for mode in ('on', 'off'))
            on_metrics, off_metrics = metric_values(on), metric_values(off)
            pair = strength.paired(on, off, context['inner']['rows'])
            row = dict(arm=arm, seed=seed, best_step=steps[arm, seed], selected_mode='on',
                       **{k: v for k, v in pair.items() if k not in ('ports', 'class_net')})
            for key in METRICS:
                row.update({f'on_{key}': on_metrics[key], f'off_{key}': off_metrics[key],
                            f'delta_{key}': on_metrics[key] - off_metrics[key]})
            row.update(mean_margin_on=float(strength.margin(on['score'], on['label']).mean()),
                       mean_margin_off=float(strength.margin(off['score'], off['label']).mean()))
            records.append(row)
    return records


def actual_availability(context):
    base = ROOT.parent / 'ports24_sync_20261112/native_cache'
    audit_path = ROOT.parent / 'ports24_recovered_training_20261113/recovery_audit_local.json'
    extra_path = EXTRA / 'native_local.npz'
    required = (base / 'complete.json', base / 'available.npy', audit_path, extra_path)
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Availability evidence unavailable; no file-presence fallback:\n'
                                + '\n'.join(map(str, missing)))
    original = context['validation']
    raw = np.load(base / 'available.npy', mmap_mode='r', allow_pickle=False)
    if raw.shape != (len(original['rows']), 2) or read(base / 'complete.json')['n'] != len(original['rows']):
        raise ValueError('Native availability cache does not match the original manifest')
    values = dict(validation=np.asarray(raw[original['ids']]).copy())
    audit, restored = read(audit_path), 0
    for j, index in enumerate(original['ids']):
        item = audit.get(str(int(index)))
        if item is None:
            continue
        row = original['rows'][int(index)]
        if (item.get('status') != 'recovered' or item.get('destination') not in ('vv', 'vh')
                or item.get('id') != row['id'] or item.get('corpus_global_row') != row['corpus_global_row']):
            raise ValueError(f'Unconfirmed or mismatched recovery audit at validation index {index}')
        p = ('vv', 'vh').index(item['destination'])
        if values['validation'][j, p] != 0 or values['validation'][j, 1-p] != 1:
            raise ValueError(f'Recovery overlay disagrees with base availability at index {index}')
        values['validation'][j, p] = 1
        restored += 1
    with np.load(extra_path, allow_pickle=False) as data:
        if 'available' not in data:
            raise ValueError('Expansion native cache lacks actual availability')
        raw_extra = data['available']  # Do not load or decode the pixels member.
    extra = context['expansion']
    if raw_extra.shape != (len(extra['rows']), 2):
        raise ValueError('Expansion availability cache does not match its evaluation manifest')
    values['expansion'] = raw_extra[extra['ids']].copy()
    counts = {}
    for split, expected in (('validation', (151, 9)), ('expansion', (171, 27))):
        available = values[split]
        if not np.isin(available, (0, 1)).all() or not available.any(1).all():
            raise ValueError(f'Invalid or entirely unavailable external query in {split}')
        dual = available.astype(bool).all(1)
        actual = int(dual.sum()), int((~dual).sum())
        if actual != expected:
            raise ValueError(f'{split} actual dual/single counts {actual}, expected {expected}; do not guess')
        values[split] = dual
        counts[split] = dict(n=len(dual), dual=actual[0], single=actual[1])
    counts['validation']['recovery_overrides'] = restored
    return values, counts


def availability_contrasts(cache, dual_masks):
    records = []
    jobs = (('guided_vs_paired_ce', 'guided', 'paired_ce', 'on'),
            ('guided_vs_off', 'guided', 'guided', 'off'),
            ('paired_ce_vs_off', 'paired_ce', 'paired_ce', 'off'))
    for split, dual in dual_masks.items():
        for seed in SEEDS:
            for name, arm, baseline, mode in jobs:
                a, b = cache[split, arm, seed, 'on'], cache[split, baseline, seed, mode]
                if not np.array_equal(a['ids'], b['ids']) or not np.array_equal(a['label'], b['label']):
                    raise ValueError('Availability contrast has mismatched query IDs or labels')
                for group, mask in (('dual', dual), ('single', ~dual)):
                    right = a['score'][mask].argmax(1) == a['label'][mask]
                    was = b['score'][mask].argmax(1) == b['label'][mask]
                    rescued, harmed = int((right & ~was).sum()), int((~right & was).sum())
                    records.append(dict(split=split, seed=seed, comparison=name, availability=group,
                        n=int(mask.sum()), correct_before=int(was.sum()), correct_after=int(right.sum()),
                        oa_before=float(was.mean()*100), oa_after=float(right.mean()*100),
                        delta_oa=float((right.astype(int)-was.astype(int)).mean()*100),
                        rescued=rescued, harmed=harmed, net=rescued-harmed,
                        changed=int((a['score'][mask].argmax(1) != b['score'][mask].argmax(1)).sum()),
                        max_abs_logit_delta=float(np.abs(a['score'][mask]-b['score'][mask]).max())))
    return records


def summarize(records, keys, values):
    result = []
    groups = sorted({tuple(row[k] for k in keys) for row in records})
    for group in groups:
        selected = [row for row in records if tuple(row[k] for k in keys) == group]
        if sorted(row['seed'] for row in selected) != list(SEEDS):
            raise ValueError('Expected exactly three matched optimization seeds')
        item = dict(zip(keys, group))
        item['seeds'] = list(SEEDS)
        item['n_per_seed'] = selected[0]['n']
        for key in values:
            vector = [row[key] for row in selected]
            item[key] = dict(mean=float(np.mean(vector)), sd=float(np.std(vector, ddof=1)))
        result.append(item)
    return result


def outcome_masks(a, b):
    right, was = a['score'].argmax(1) == a['label'], b['score'].argmax(1) == b['label']
    return dict(rescued=right & ~was, harmed=~right & was,
                stable_correct=right & was, stable_wrong=~right & ~was)


def comparisons(cache, context):
    records, candidates = [], []
    for split, info in context.items():
        for seed in SEEDS:
            a = cache[split, 'guided', seed, 'on']
            for priority, (arm, mode) in enumerate(CONTRASTS):
                if split == 'inner' and arm not in NEW_ARMS:
                    continue
                b = cache[split, arm, seed, mode]
                name = 'guided_vs_off' if arm == 'guided' else f'guided_vs_{arm}'
                result = strength.paired(a, b, info['rows'])
                states = outcome_masks(a, b)
                result.update(stable_correct=int(states['stable_correct'].sum()),
                              stable_wrong=int(states['stable_wrong'].sum()))
                am, bm = metric_values(a), metric_values(b)
                result.update(delta_macro_f1=am['macro_f1'] - bm['macro_f1'], delta_nll=am['nll'] - bm['nll'])
                records.append(dict(split=split, seed=seed, comparison=name, **result))
                if split == 'inner':
                    continue
                ma, mb = strength.margin(a['score'], a['label']), strength.margin(b['score'], b['label'])
                for kind, mask in states.items():
                    for j in np.flatnonzero(mask):
                        index = int(a['ids'][j])
                        candidates.append(dict(split=split, seed=seed, comparison=name, kind=kind,
                            before_arm=arm, before_mode=mode, index=index, position=int(j),
                            corpus_global_row=info['rows'][index]['corpus_global_row'],
                            priority=priority, magnitude=float(abs(ma[j] - mb[j]))))
    return records, candidates


def select_cases(candidates, limit=12):
    selected, seen, usage = [], set(), Counter()
    while len(selected) < limit:
        added = False
        for kind in KINDS:
            pool = [c for c in candidates if c['kind'] == kind and c['corpus_global_row'] not in seen]
            if not pool or len(selected) == limit:
                continue
            # Prefer the two attribution contrasts, then spread split/contrast/seed.
            chosen = min(pool, key=lambda c: (c['priority'] >= 2,
                usage[c['split']], usage[c['comparison']], usage[c['seed']],
                -c['magnitude'], c['corpus_global_row'], c['priority']))
            selected.append(chosen)
            seen.add(chosen['corpus_global_row'])
            for key in ('split', 'comparison', 'seed'):
                usage[chosen[key]] += 1
            added = True
        if not added:
            break
    return selected


def gallery(candidates, cache, context):
    selected = select_cases(candidates)
    folder = ROOT / 'gallery'
    folder.mkdir(exist_ok=True)
    apis, cases = {}, []
    for item in selected:
        split, seed, j = item['split'], item['seed'], item['position']
        info = context[split]
        if split not in apis:
            apis[split] = native.PanelInputs(info['rows'])
        api = apis[split]
        suffix = '-QueryRef' if info['recover'] else '-Accepted'
        before = item['before_arm'] + '-' + item['before_mode'] + suffix
        after = 'guided-on' + suffix
        if info['recover']:
            for alias in (before, after):
                native.RECOVERY_ARMS[alias] = 'existing_recovery_for_query_and_source_reference'
        index = item['index']
        row = info['rows'][index]
        bank = api.bank(after, index, {'references': info['references']})
        a = cache[split, 'guided', seed, 'on']
        b = cache[split, item['before_arm'], seed, item['before_mode']]
        truth = int(a['label'][j])
        rival_scores = b['score'][j].copy() if item['kind'] == 'rescued' else a['score'][j].copy()
        rival_scores[truth] = -np.inf
        rival = int(rival_scores.argmax())
        filename = f"{len(cases):02d}_{item['kind']}_{row['corpus_global_row']}.png"
        ma, mb = strength.margin(a['score'], a['label']), strength.margin(b['score'], b['label'])
        caption = (f"{split} {item['comparison']} | {item['kind']} s{seed} | truth {truth} | "
                   f"pred {b['score'][j].argmax()} -> {a['score'][j].argmax()} | margin {mb[j]:+.4f} -> {ma[j]:+.4f}")
        provenance, gaps = native.draw_panel(api, index, before, after, bank, bank,
            truth, rival, strength.NAMES, folder / filename, caption)
        evidence = []
        for s in SEEDS:
            x = cache[split, 'guided', s, 'on']
            y = cache[split, item['before_arm'], s, item['before_mode']]
            evidence.append(dict(seed=s, before_prediction=int(y['score'][j].argmax()),
                after_prediction=int(x['score'][j].argmax()), before_scores=y['score'][j].tolist(),
                after_scores=x['score'][j].tolist(),
                before_true_margin=float(strength.margin(y['score'], y['label'])[j]),
                after_true_margin=float(strength.margin(x['score'], x['label'])[j])))
        cases.append(dict(**item, id=row['id'], mmsi=row['mmsi'], port=row['port'],
            product=row['product'], label=truth, rival=rival, file=filename,
            actual_source16_bank=bank, missing_reference_roles=gaps, paths=provenance,
            evidence=evidence, recovery_audit=api.recovered.get(index, (None, None))[1],
            visual_review='pending; generated panels are not a completed visual review'))
    write(ROOT / 'cases_local.json', cases)
    body = ['<!doctype html><meta charset="utf-8"><title>Paired correction: local cases</title>',
            '<p>External diagnostic examples only; not prevalence estimates. References are actual bank examples, not attribution.</p>']
    for case in cases:
        title = f"{case['split']} {case['comparison']} {case['kind']} seed {case['seed']} ID {case['id']}"
        body.append(f'<section><h2>{html.escape(title)}</h2><img style="max-width:100%" src="{html.escape(case["file"], quote=True)}"></section>')
    (folder / 'index.html').write_text('\n'.join(body), encoding='utf-8')
    return [dict(kind=kind, distinct_available=len({c['corpus_global_row'] for c in candidates if c['kind'] == kind}),
                 selected=sum(c['kind'] == kind for c in selected)) for kind in KINDS]


def main():
    selections = completion()
    context = metadata()
    dual_masks, availability_counts = actual_availability(context)
    cache, metrics = {}, []
    for split, arm, seed, mode in required_caches():
        info = context[split]
        score = native.load_prediction(cache_path(split, arm, seed, mode), info['ids'], info['rows'], require_label=True)
        item = dict(ids=info['ids'], label=info['label'], score=score)
        cache[split, arm, seed, mode] = item
        metrics.append(dict(split=split, arm=arm, seed=seed, mode=mode, n=len(info['ids']), **metric_values(item)))
    paired, candidates = comparisons(cache, context)
    source = source_on_off(cache, context, selections)
    strata = availability_contrasts(cache, dual_masks)
    aggregate = ROOT / 'aggregate'
    aggregate.mkdir(exist_ok=True)
    native.write_csv(aggregate / 'metrics.csv', metrics)
    native.write_csv(aggregate / 'selection.csv', selections)
    native.write_csv(aggregate / 'source_inner_on_off.csv', source)
    native.write_csv(aggregate / 'availability_paired.csv', strata)
    write(aggregate / 'availability_paired.json', strata)
    write(aggregate / 'paired.json', paired)
    flat = [{k: v for k, v in r.items() if k not in ('ports', 'class_net')} for r in paired]
    native.write_csv(aggregate / 'paired.csv', flat)
    summary = dict(source_inner_on_off=source,
        availability_counts=availability_counts,
        availability_paired=summarize(strata, ('split', 'comparison', 'availability'),
                                     ('oa_before', 'oa_after', 'delta_oa', 'rescued', 'harmed', 'net', 'changed')),
        metrics=summarize(metrics, ('split', 'arm', 'mode'), METRICS),
        paired=summarize(paired, ('split', 'comparison'),
                         ('rescued', 'harmed', 'net', 'changed', 'delta_oa', 'delta_ba',
                          'delta_macro_f1', 'delta_nll', 'mean_margin_change')),
        class_counts={s: np.bincount(v['label'], minlength=8).tolist() for s, v in context.items()},
        interpretation=dict(ba='Mean recall over classes present in each split',
            macro_f1='Fixed eight classes, absent-class F1 zero; OA/BA/F1 in percent',
            seeds='Matched optimization repeats from the same formal checkpoint17; sample SD, not independent model starts',
            source_inner='Selection diagnostic; earlier encoder saw this pool, not an unseen-domain result',
            source_on_off='Cached on-selected checkpoint and its matching off evaluation; off is not independently selected',
            teacher='Fixed starting classifier on the same augmentation; not pixel interventions or an intervention-trained teacher',
            availability='Original native available cache plus confirmed recovery destinations for160; expansion native available array for198; no pixel decoding or file-presence inference',
            active_write='Dual polarization is necessary, not sufficient, for an active valid write region; these strata are not trace.active groups',
            single_gain='Between-model gains on single-polarization queries may reflect shared-backbone/readout training, not direct PCEA writeback; same-checkpoint on/off is reported separately',
            validation='Historical160 development observations, not a fresh holdout',
            expansion='Preregistered198 new-vessel observations within known validation passages, previously analyzed',
            comparisons='Best selected independently on source inner; no external checkpoint or scale selection',
            on_off='Same selected guided checkpoint; tests current module contribution, not unique causal attribution of training benefit',
            compute='Equal updates and sample order, not equal training compute',
            publication='Only aggregate/ is shareable; local cases and gallery retain private IDs/paths',
            test_accessed=False))
    summary['gallery_coverage'] = gallery(candidates, cache, context)
    write(aggregate / 'summary.json', summary)
    print(f'Completed cached CPU analysis: {len(metrics)} metric rows, {len(paired)} paired rows; {aggregate}')


if __name__ == '__main__':
    main()
