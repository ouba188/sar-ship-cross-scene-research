"""Cached CPU analysis after three routed fits and all twelve evaluations.

Run python -B analyze.py. Publish aggregate/ only; gallery/ and cases_local.json
retain private sample identifiers. No training/model imports or test pixels.
"""
import html
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PAIRED = ROOT.parent / 'pcea_paired_correction_20261118'
spec = importlib.util.spec_from_file_location('routed_analysis_helpers_cpu', PAIRED / 'analyze.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
native, strength = base.native, base.strength
read, write = base.read, base.write
SEEDS, METRICS, KINDS = base.SEEDS, base.METRICS, base.KINDS
ARMS = ('full', 'generic', 'paired_ce', 'guided', 'routed')
PAIRED_ARMS = ('paired_ce', 'guided', 'routed')
CONTRASTS = (('guided', 'on'), ('routed', 'off'), ('paired_ce', 'on'),
             ('generic', 'on'), ('full', 'on'))


def fit_folder(arm, seed):
    parent = ROOT if arm == 'routed' else PAIRED if arm in PAIRED_ARMS else base.OLD
    return parent / 'fits' / f'{arm}_s{seed}'


def cache_path(split, arm, seed, mode):
    if split == 'expansion' and arm in ('full', 'generic'):
        return base.EXTRA / f'{arm}_s{seed}_{mode}_local.npz'
    name = ('inner.npz' if mode == 'on' else 'inner_off.npz') if split == 'inner' else f'{split}_{mode}.npz'
    return fit_folder(arm, seed) / name


def required_caches():
    for split in ('inner', 'validation', 'expansion'):
        for arm in ARMS:
            modes = ('on',) if arm == 'full' or (split == 'inner' and arm == 'generic') else ('on', 'off')
            for seed in SEEDS:
                for mode in modes:
                    yield split, arm, seed, mode


def completion():
    missing = [fit_folder('routed', s) / 'receipt.json' for s in SEEDS
               if not (fit_folder('routed', s) / 'receipt.json').is_file()]
    if missing:
        raise FileNotFoundError('All three routed receipts required before analysis:\n' + '\n'.join(map(str, missing)))
    selections = []
    for arm in ARMS:
        for seed in SEEDS:
            folder = fit_folder(arm, seed)
            receipt, config, curve = (read(folder / name) for name in ('receipt.json', 'config.json', 'curve.json'))
            if (receipt['arm'] != arm or receipt['seed'] != seed or config['steps'] != 240
                    or not curve or curve[-1]['step'] != 240
                    or config['selection'] != 'source-inner BA/OA/-NLL, step0 eligible'
                    or receipt['best']['step'] not in [0] + [r['step'] for r in curve]
                    or (arm in PAIRED_ARMS and receipt.get('source_only') is not True)):
                raise ValueError(f'Incomplete or inconsistent source-selected fit: {folder}')
            if arm == 'routed' and config.get('correction_gradient_targets') != ['ccrel', 'exchange']:
                raise ValueError('Expected the routed correction recipe')
            selections.append(dict(arm=arm, seed=seed, best_step=receipt['best']['step'],
                                   updates=config['steps'], reused=arm != 'routed'))
    missing = [cache_path(*key) for key in required_caches() if not cache_path(*key).is_file()]
    if missing:
        raise FileNotFoundError('Evaluation caches incomplete; no analysis outputs:\n' + '\n'.join(map(str, missing)))
    return selections


def metadata():
    context = base.metadata()
    new, old = read(ROOT / 'split_local.json'), read(PAIRED / 'split_local.json')
    if any(not np.array_equal(new[k], old[k]) for k in ('fit', 'inner')):
        raise ValueError('Routed source split differs from the paired controls')
    return context


def availability(context):
    masks, counts = base.actual_availability(context)
    info = context['inner']
    folder = ROOT.parent / 'ports24_sync_20261112/native_cache'
    raw = np.load(folder / 'available.npy', mmap_mode='r', allow_pickle=False)
    available = np.asarray(raw[info['ids']]).copy()
    audit = read(ROOT.parent / 'ports24_recovered_training_20261113/recovery_audit_local.json')
    restored = 0
    for j, index in enumerate(info['ids']):
        item = audit.get(str(int(index)))
        if item is None:
            continue
        row = info['rows'][int(index)]
        if (item.get('status') != 'recovered' or item.get('destination') not in ('vv', 'vh')
                or item.get('id') != row['id'] or item.get('corpus_global_row') != row['corpus_global_row']):
            raise ValueError('Source inner recovery audit disagrees with the manifest')
        p = ('vv', 'vh').index(item['destination'])
        if available[j, p] != 0 or available[j, 1-p] != 1:
            raise ValueError('Source inner recovery destination disagrees with native availability')
        available[j, p] = 1
        restored += 1
    if not np.isin(available, (0, 1)).all() or available.shape != (1000, 2) or not available.all():
        raise ValueError('Expected the previously confirmed 1000 dual-polarization source inner observations')
    masks['inner'] = available.astype(bool).all(1)
    counts['inner'] = dict(n=1000, dual=1000, single=0, recovery_overrides=restored)
    return masks, counts


def source_on_off(cache, context, selections):
    records = []
    steps = {(r['arm'], r['seed']): r['best_step'] for r in selections}
    for arm in PAIRED_ARMS:
        for seed in SEEDS:
            a, b = (cache['inner', arm, seed, mode] for mode in ('on', 'off'))
            am, bm = base.metric_values(a), base.metric_values(b)
            pair = strength.paired(a, b, context['inner']['rows'])
            item = dict(arm=arm, seed=seed, best_step=steps[arm, seed], selected_mode='on',
                        **{k: v for k, v in pair.items() if k not in ('ports', 'class_net')})
            for key in METRICS:
                item.update({f'on_{key}': am[key], f'off_{key}': bm[key], f'delta_{key}': am[key]-bm[key]})
            records.append(item)
    return records


def comparisons(cache, context, dual_masks):
    paired, strata, candidates = [], [], []
    for split, info in context.items():
        for seed in SEEDS:
            a = cache[split, 'routed', seed, 'on']
            for rank, (arm, mode) in enumerate(CONTRASTS):
                b = cache[split, arm, seed, mode]
                name = 'routed_vs_off' if arm == 'routed' else f'routed_vs_{arm}'
                pair = strength.paired(a, b, info['rows'])
                am, bm = base.metric_values(a), base.metric_values(b)
                states = base.outcome_masks(a, b)
                pair.update(delta_macro_f1=am['macro_f1']-bm['macro_f1'], delta_nll=am['nll']-bm['nll'],
                            stable_correct=int(states['stable_correct'].sum()), stable_wrong=int(states['stable_wrong'].sum()))
                paired.append(dict(split=split, seed=seed, comparison=name, **pair))
                for group, mask in (('dual', dual_masks[split]), ('single', ~dual_masks[split])):
                    n = int(mask.sum())
                    right = a['score'][mask].argmax(1) == a['label'][mask]
                    was = b['score'][mask].argmax(1) == b['label'][mask]
                    rescued, harmed = int((right & ~was).sum()), int((~right & was).sum())
                    strata.append(dict(split=split, seed=seed, comparison=name, availability=group,
                        n=n, correct_before=int(was.sum()), correct_after=int(right.sum()),
                        oa_before=float(was.mean()*100) if n else None,
                        oa_after=float(right.mean()*100) if n else None,
                        delta_oa=100*(rescued-harmed)/n if n else None,
                        rescued=rescued, harmed=harmed, net=rescued-harmed,
                        changed=int((a['score'][mask].argmax(1) != b['score'][mask].argmax(1)).sum()),
                        max_abs_logit_delta=float(np.abs(a['score'][mask]-b['score'][mask]).max()) if n else None))
                if split == 'inner':
                    continue
                delta = strength.margin(a['score'], a['label'])-strength.margin(b['score'], b['label'])
                for kind, mask in states.items():
                    for j in np.flatnonzero(mask):
                        index = int(a['ids'][j])
                        candidates.append(dict(split=split, seed=seed, comparison=name, kind=kind,
                            before_arm=arm, before_mode=mode, index=index, position=int(j),
                            corpus_global_row=info['rows'][index]['corpus_global_row'],
                            priority=min(rank, 1) if rank < 3 else rank,
                            magnitude=float(abs(delta[j]))))
    return paired, strata, candidates


def gallery(candidates, cache, context):
    selected = base.select_cases(candidates, limit=12)
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
        before, after = item['before_arm']+'-'+item['before_mode']+suffix, 'routed-on'+suffix
        if info['recover']:
            for alias in (before, after):
                native.RECOVERY_ARMS[alias] = 'existing_query_and_source_reference_recovery'
        index, row = item['index'], info['rows'][item['index']]
        bank = api.bank(after, index, {'references': info['references']})
        a, b = cache[split, 'routed', seed, 'on'], cache[split, item['before_arm'], seed, item['before_mode']]
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
            x, y = cache[split, 'routed', s, 'on'], cache[split, item['before_arm'], s, item['before_mode']]
            evidence.append(dict(seed=s, before_prediction=int(y['score'][j].argmax()),
                after_prediction=int(x['score'][j].argmax()), before_scores=y['score'][j].tolist(),
                after_scores=x['score'][j].tolist(), before_true_margin=float(strength.margin(y['score'], y['label'])[j]),
                after_true_margin=float(strength.margin(x['score'], x['label'])[j])))
        cases.append(dict(**item, id=row['id'], mmsi=row['mmsi'], port=row['port'], product=row['product'],
            label=truth, rival=rival, file=filename, actual_source16_bank=bank, paths=provenance,
            missing_reference_roles=gaps, evidence=evidence,
            recovery_audit=api.recovered.get(index, (None, None))[1], visual_review='pending actual image viewing'))
    write(ROOT / 'cases_local.json', cases)
    body = ['<!doctype html><meta charset="utf-8"><title>Routed correction: local cases</title>',
            '<p>External diagnostic examples, not prevalence estimates. Actual source-bank examples are not causal attribution.</p>']
    for case in cases:
        title = f"{case['split']} {case['comparison']} {case['kind']} seed {case['seed']} ID {case['id']}"
        body.append(f'<section><h2>{html.escape(title)}</h2><img style="max-width:100%" src="{html.escape(case["file"], quote=True)}"></section>')
    (folder / 'index.html').write_text('\n'.join(body), encoding='utf-8')
    return [dict(kind=k, distinct_available=len({c['corpus_global_row'] for c in candidates if c['kind'] == k}),
                 selected=sum(c['kind'] == k for c in selected)) for k in KINDS]


def main():
    selections = completion()
    context = metadata()
    dual_masks, counts = availability(context)
    cache, metrics = {}, []
    for split, arm, seed, mode in required_caches():
        info = context[split]
        z = native.load_prediction(cache_path(split, arm, seed, mode), info['ids'], info['rows'], require_label=True)
        item = dict(ids=info['ids'], label=info['label'], score=z)
        cache[split, arm, seed, mode] = item
        metrics.append(dict(split=split, arm=arm, seed=seed, mode=mode, n=len(info['ids']), **base.metric_values(item)))
    paired, strata, candidates = comparisons(cache, context, dual_masks)
    source = source_on_off(cache, context, selections)
    out = ROOT / 'aggregate'
    out.mkdir(exist_ok=True)
    for name, records in (('metrics', metrics), ('selection', selections), ('source_inner_on_off', source),
                          ('availability_paired', strata)):
        native.write_csv(out / f'{name}.csv', records)
    write(out / 'paired.json', paired)
    write(out / 'availability_paired.json', strata)
    native.write_csv(out / 'paired.csv', [{k: v for k, v in r.items() if k not in ('ports', 'class_net')} for r in paired])
    summary = dict(metrics=base.summarize(metrics, ('split', 'arm', 'mode'), METRICS),
        paired=base.summarize(paired, ('split', 'comparison'),
            ('rescued', 'harmed', 'net', 'changed', 'delta_oa', 'delta_ba', 'delta_macro_f1', 'delta_nll', 'mean_margin_change')),
        source_inner_on_off=source, availability_counts=counts,
        availability_paired=base.summarize([r for r in strata if r['n']], ('split', 'comparison', 'availability'),
            ('oa_before', 'oa_after', 'delta_oa', 'rescued', 'harmed', 'net', 'changed')),
        class_counts={s: np.bincount(v['label'], minlength=8).tolist() for s, v in context.items()},
        interpretation=dict(ba='Present-class recall mean; macro F1 uses fixed8 classes; OA/BA/F1 in percent',
            seeds='Three matched optimization repeats from one formal checkpoint17, not independent pretrained starts; sample SD',
            selection='On-selected source-inner best, step0 eligible; off uses the same checkpoint; no external reselection',
            source_inner='Previously seen by the original encoder; development evidence only, not unseen-domain generalization',
            external='Historical validation160 and previously analyzed198 new-vessel observations in known passages, not blind scenes',
            teacher='Fixed starting classifier with the same augmentation; not pixel interventions',
            routing='Only extra-loss gradients to reader/exchange; CE continues all trainable parameters; shared gradient clipping remains',
            attribution='Recovery versus old guided is not automatically a gain versus paired CE; on/off is a separate within-checkpoint contrast',
            availability='Actual native available plus confirmed recovery destinations; dual does not guarantee an active write region; empty source single stratum has null OA',
            single_gain='Cross-model gains on single polarization can reflect shared-network training, not direct PCEA writeback',
            compute='Matched240 updates/sample order, not equal compute; no further sweep',
            gallery='At most12 distinct external queries; stable correct/wrong retained with no rescue/harm; actual viewing still required',
            publication='aggregate/ only; sample IDs, reference identities and paths remain in local cases/gallery',
            test_accessed=False))
    summary['gallery_coverage'] = gallery(candidates, cache, context)
    write(out / 'summary.json', summary)
    print(f'Cached CPU analysis complete: {len(metrics)} metrics, {len(paired)} pairs, {len(strata)} availability rows; {out}')


if __name__ == '__main__':
    main()
