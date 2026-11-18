"""CPU-only analysis of five readout conditions on two development cohorts.

Run after the separate ten-evaluation process completes: python -B analysis.py
No model imports or waiting. Only aggregate/ is suitable for public sharing.
"""
import argparse
from collections import Counter
import html
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'm3_coarse_supplement_20261118/fits/continue_s17'
spec = importlib.util.spec_from_file_location(
    'native_readout_analysis_helpers', ROOT.parent / 'm3_coarse_supplement_20261118/analysis.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
base = previous.base
read, write, csv_write = base.read, base.write, base.csv_write
SPLITS = previous.SPLITS
KINDS = previous.KINDS
CONDITIONS = ('legacy', 'fine_anchor', 'fine_live', 'fine_anchor_legacyread', 'fine_live_legacyread')
# Each tuple is (name, after, before, interpretation).
CONTRASTS = (
    ('anchor_vs_continue', 'fine_anchor', 'legacy', 'matched_continuation'),
    ('live_vs_continue', 'fine_live', 'legacy', 'matched_continuation'),
    ('live_vs_anchor', 'fine_live', 'fine_anchor', 'matched_continuation'),
    ('anchor_native_vs_legacyread', 'fine_anchor', 'fine_anchor_legacyread', 'same_weight_readout'),
    ('live_native_vs_legacyread', 'fine_live', 'fine_live_legacyread', 'same_weight_readout'),
    ('anchor_legacyread_vs_continue', 'fine_anchor_legacyread', 'legacy', 'different_weights_legacy_readout'),
    ('live_legacyread_vs_continue', 'fine_live_legacyread', 'legacy', 'different_weights_legacy_readout'),
)


def ready(main_only=False):
    paths = [ROOT / 'results.json', ROOT / 'status.json', ROOT / 'evaluation_inputs/manifest_local.json']
    paths += [ROOT / 'evaluation' / f'{arm}_{split}_local.npz'
              for arm in CONDITIONS for split in SPLITS]
    paths += [ROOT / 'fits' / f'{arm}_s17/receipt.json' for arm in ('fine_anchor', 'fine_live')]
    paths.append(BASE / 'receipt.json')
    if not all(path.is_file() for path in paths):
        return False
    status = read(ROOT / 'status.json')
    return status.get('phase') == 'complete' and status.get('evaluations') in ((6, 10) if main_only else (10,))


def load_scores(rows):
    if len(rows) != 374 or len({r['id'] for r in rows}) != 374:
        raise ValueError('Expected 358 distinct query observations and 16 fixed source references.')
    records = read(ROOT / 'results.json')
    records = [r for r in records if r['arm'] in CONDITIONS]
    table = {(r['split'], r['arm']): r for r in records}
    expected = {(split, arm) for split in SPLITS for arm in CONDITIONS}
    if set(table) != expected or len(records) != len(expected):
        raise ValueError('Missing requested evaluation conditions in results.json; rerun when its current write completes.')
    scores, selections = {}, []
    for arm in ('legacy', 'fine_anchor', 'fine_live'):
        folder = BASE if arm == 'legacy' else ROOT / 'fits' / f'{arm}_s17'
        receipt = read(folder / 'receipt.json')
        step = receipt['best']['step']
        if step not in (0, 30, 60, 90, 120):
            raise ValueError('Unexpected source-selected checkpoint step.')
        selections.append(dict(arm=arm, seed=17, selected_step=step,
                               source_initial=receipt['initial'], source_selected=receipt['best']['metrics']))
        for split in SPLITS:
            variants = [v for v in (arm, arm + '_legacyread') if v in CONDITIONS]
            if any(table[split, v]['selected_step'] != step for v in variants):
                raise ValueError('Same-weight conditions have inconsistent selection metadata.')
    for split, arm in sorted(expected):
        ids = SPLITS[split]
        y = np.array([rows[i]['label'] for i in ids])
        with np.load(ROOT / 'evaluation' / f'{arm}_{split}_local.npz', allow_pickle=False) as cache:
            if not np.array_equal(cache['ids'], ids) or not np.array_equal(cache['label'], y):
                raise ValueError(f'IDs/labels do not match the actual manifest: {split}/{arm}')
            z = cache['score'].astype(np.float64)
        if z.shape != (len(ids), 8) or not np.isfinite(z).all():
            raise ValueError(f'Invalid scores: {split}/{arm}')
        metrics = base.metrics(y, z)
        if any(abs(metrics[k] - table[split, arm][k]) > 1e-8 for k in ('oa', 'ba')):
            raise ValueError(f'NPZ and results.json disagree: {split}/{arm}')
        if arm == 'legacy':
            with np.load(BASE / f'{split}_on_local.npz', allow_pickle=False) as cached:
                if (not np.array_equal(cached['ids'], ids) or not np.array_equal(cached['label'], y)
                        or np.max(abs(cached['score'] - z)) >= 2e-5):
                    raise ValueError('Reused continue predictions do not replay.')
        scores[split, arm] = z
    return scores, selections


def statistics(rows, scores, lengths):
    conditions, paired, candidates = [], [], []
    for split, ids in SPLITS.items():
        y = np.array([rows[i]['label'] for i in ids])
        for group_type, group, mask in previous.groups(y, lengths[ids]):
            selected, labels = ids[mask], y[mask]
            ships = np.array([rows[i]['mmsi'] for i in selected], dtype=object)
            common = dict(split=split, seed=17, group_type=group_type, group=group,
                          n_observations=len(selected), n_unique_ships=len(set(ships)),
                          present_classes=len(np.unique(labels)))
            for arm in CONDITIONS:
                conditions.append(dict(**common, arm=arm, **base.metrics(labels, scores[split, arm][mask])))
            for name, after, before, interpretation in CONTRASTS:
                b, a = scores[split, before][mask], scores[split, after][mask]
                bm, am = base.metrics(labels, b), base.metrics(labels, a)
                states = base.state_masks(b, a, labels)
                delta = base.margin(a, labels) - base.margin(b, labels)
                paired.append(dict(**common, contrast=name, interpretation=interpretation,
                    before=before, after=after, before_oa=bm['oa'], after_oa=am['oa'],
                    before_ba=bm['ba'], after_ba=am['ba'],
                    delta_oa_pp=am['oa'] - bm['oa'] if len(labels) else None,
                    delta_ba_pp=am['ba'] - bm['ba'] if len(labels) else None,
                    delta_margin_mean=float(delta.mean()) if len(labels) else None,
                    delta_margin_median=float(np.median(delta)) if len(labels) else None,
                    **{k: int(v.sum()) for k, v in states.items()},
                    **{f'{k}_unique_ships': len(set(ships[v])) for k, v in states.items()}))
                if group_type == 'overall':
                    for kind, selected_kind in states.items():
                        for j in np.flatnonzero(selected_kind):
                            candidates.append(dict(split=split, seed=17, contrast=name,
                                interpretation=interpretation, before=before, after=after, kind=kind,
                                index=int(ids[j]), position=int(j), delta_margin=float(delta[j])))
    return conditions, paired, candidates


def choose_cases(candidates):
    selected, used = [], set()
    for kind in KINDS:
        pool = [r for r in candidates if r['kind'] == kind and r['index'] not in used]
        if not pool:
            continue
        # Prefer the two main comparisons, then cover cohorts and arms where possible.
        main = [r for r in pool if r['contrast'] in ('anchor_vs_continue', 'live_vs_continue')]
        pool = main or pool
        splits = Counter(r['split'] for r in selected)
        arms = Counter(r['after'] for r in selected)
        item = min(pool, key=lambda r: (splits[r['split']], arms[r['after']],
                                       -abs(r['delta_margin']), r['index'], r['contrast']))
        selected.append(item)
        used.add(item['index'])
    # If a transition type is absent, use a second stable case rather than inventing a flip.
    while len(selected) < 4:
        pool = [r for r in candidates if r['index'] not in used
                and r['contrast'] in ('anchor_vs_continue', 'live_vs_continue')]
        if not pool:
            break
        splits = Counter(r['split'] for r in selected)
        item = min(pool, key=lambda r: (splits[r['split']], -abs(r['delta_margin']), r['index']))
        selected.append(item)
        used.add(item['index'])
    return selected


def gallery(rows, scores, lengths, panels, native, candidates):
    folder = ROOT / 'gallery'
    folder.mkdir(exist_ok=True)
    notes_path = ROOT / 'case_notes_local.json'
    existing = read(notes_path).get('cases', []) if notes_path.exists() else []
    reviews = {(r['id'], r['contrast']): r.get('visual_review') for r in existing}
    cases = []
    for item in choose_cases(candidates):
        split, i, j = (item[k] for k in ('split', 'index', 'position'))
        row, truth = rows[i], int(rows[i]['label'])
        evidence = {}
        for condition in CONDITIONS:
            z = scores[split, condition][j]
            wrong = z.copy()
            wrong[truth] = -np.inf
            evidence[condition] = dict(prediction=int(z.argmax()), strongest_rival=int(wrong.argmax()),
                margin=float(base.margin(z[None], np.array([truth]))[0]), scores=z.tolist())
        b, a = evidence[item['before']], evidence[item['after']]
        rival = b['strongest_rival']
        bank = native.legal_bank(rows, {'references': list(range(358, 374))}, panels.available, i)
        filename = f'{len(cases):02d}_{item["kind"]}_{row["id"]}.png'
        caption = (f'{split} s17 {item["contrast"]} | {item["kind"]} | truth {truth} | '
                   f'pred {b["prediction"]}->{a["prediction"]} | '
                   f'margin {b["margin"]:+.4f}->{a["margin"]:+.4f}')
        paths, gaps = native.draw_panel(panels, i, item['before'], item['after'], bank, bank,
                                       truth, rival, base.NAMES, folder / filename, caption)
        # Local notes record every paired diagnostic for the same actual observation.
        paired = []
        for name, after, before, interpretation in CONTRASTS:
            bb, aa = scores[split, before][j:j+1], scores[split, after][j:j+1]
            states = base.state_masks(bb, aa, np.array([truth]))
            paired.append(dict(contrast=name, interpretation=interpretation,
                kind=next(k for k in KINDS if states[k][0]),
                delta_margin=evidence[after]['margin']-evidence[before]['margin']))
        review = reviews.get((row['id'], item['contrast'])) or dict(status='pending', observations=None,
            limitation='Image appearance is not causal attribution; no endpoint identity or size mechanism is established.')
        cases.append(dict(**item, id=row['id'], label=truth, rival=rival, file=filename,
            max_available_length=float(lengths[i]), evidence=evidence, paired_diagnostics=paired,
            actual_source16_bank=bank, paths=paths, missing_reference_roles=gaps,
            reference_rule='First active legal same-class and strongest non-true baseline-class reference; not ranked by influence.',
            visual_review=review))
    coverage = []
    for split in SPLITS:
        for name, _, _, _ in CONTRASTS:
            for kind in KINDS:
                eligible = sum(r['split'] == split and r['contrast'] == name and r['kind'] == kind for r in candidates)
                shown = sum(r['split'] == split and r['contrast'] == name and r['kind'] == kind for r in cases)
                coverage.append(dict(split=split, contrast=name, kind=kind,
                    eligible_observations=eligible, selected_panels=shown,
                    reason='no_eligible_event' if not eligible else 'selected' if shown else 'four_panel_budget'))
    write(notes_path, dict(selection='Four distinct observations; rescue, harm and stable outcomes where available. Main arm comparisons prioritized. Not a population estimate.',
                          seed=17, cases=cases))
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Native readout cases</title>',
        '<style>body{font:15px/1.5 system-ui;margin:24px}.panel{overflow-x:auto}img{max-width:none}</style>',
        '<h1>M3 native readout: paired cases</h1><p>Seed 17 only. Native 160-pixel canvases, VV/VH separate, '
        'no resizing or intensity stretching. All conditions use the same query pixels and source16 bank. '
        'Shown references are legal examples, not attribution. Available-condition scores and paired outcomes '
        'are in case_notes_local.json. Visual interpretation remains pending until human review.</p>']
    for case in cases:
        body.append(f'<h2>{html.escape(str(case["id"]))}: {case["contrast"]} / {case["kind"]}</h2>'
                    f'<div class="panel"><img src="{html.escape(case["file"])}" alt="Native query and actual references"></div>'
                    f'<p>Review: {html.escape(case["visual_review"]["status"])}</p>')
        for condition, evidence in case['evidence'].items():
            body.append(f'<p>{condition}: prediction {evidence["prediction"]}; margin {evidence["margin"]:+.5f}</p>')
    (folder / 'index.html').write_text('\n'.join(body) + '</html>', encoding='utf-8')
    return cases, coverage


def self_test():
    y = np.array([0, 0, 0, 1])
    before = np.array([[0., 1.], [2., 0.], [2., 0.], [0., 2.]])
    after = np.array([[2., 0.], [2., 0.], [2., 0.], [2., 0.]])
    b, a = base.metrics(y, before), base.metrics(y, after)
    assert a['oa'] - b['oa'] == 0
    assert np.isclose(a['ba'] - b['ba'], -100/3)
    states = base.state_masks(before, after, y)
    assert {k: int(v.sum()) for k, v in states.items()} == dict(rescued=1, harmed=1, stable_correct=2, stable_wrong=0)
    empty = base.metrics(y[:0], before[:0])
    assert empty['oa'] is None and empty['ba'] is None
    lengths = np.array([15., 16., 32., 64.])
    bins = {g: mask for kind, g, mask in previous.groups(y, lengths) if kind == 'length'}
    assert [int(bins[g].sum()) for g in ('<16', '16-32', '32-64', '>=64')] == [1, 1, 1, 1]
    items = [dict(kind=k, index=i, position=i, split='validation', after='fine_live',
                  contrast='live_vs_continue', delta_margin=float(i)) for i, k in enumerate(KINDS)]
    chosen = choose_cases(items)
    assert len({r['index'] for r in chosen}) == 4 and {r['kind'] for r in chosen} == set(KINDS)
    assert 'torch' not in sys.modules
    print('CPU self-test passed: paired OA/BA, empty groups, length boundaries, four-case coverage; no files written.')


def main():
    global CONDITIONS, CONTRASTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--check-inputs', action='store_true', help='Check completion only; no outputs or waiting.')
    parser.add_argument('--main-only', action='store_true', help='Analyze six completed main evaluations; mark same-weight diagnostics pending.')
    args = parser.parse_args()
    if args.main_only:
        CONDITIONS = CONDITIONS[:3]
        CONTRASTS = CONTRASTS[:3]
    if args.self_test:
        self_test()
        return
    if args.check_inputs:
        print('ready' if ready(args.main_only) else 'incomplete: requested evaluations are required')
        return
    if not ready(args.main_only):
        raise SystemExit('Incomplete: need both fits and the requested evaluations. No waiting or analysis outputs.')
    rows = read(ROOT / 'evaluation_inputs/manifest_local.json')
    scores, selections = load_scores(rows)
    native = base.helper()
    panels = base.Panels(rows, native)
    lengths = np.array([max(v['geometry'][2] for p, v in enumerate(pair) if panels.available[i, p])
                        for i, pair in enumerate(panels.views)])
    conditions, paired, candidates = statistics(rows, scores, lengths)
    cases, coverage = gallery(rows, scores, lengths, panels, native, candidates)
    out = ROOT / 'aggregate'
    out.mkdir(exist_ok=True)
    for name, values in (('conditions', conditions), ('paired', paired), ('gallery_coverage', coverage)):
        csv_write(out / f'{name}.csv', values)
    protocol = dict(evaluations=len(CONDITIONS)*2, same_weight_diagnostics='pending' if args.main_only else 'complete',
        continuation_seed=17, n_seeds=1, conditions=list(CONDITIONS),
        observations={s: len(ids) for s, ids in SPLITS.items()},
        unique_query_observations=358, unique_query_ships=len({r['mmsi'] for r in rows[:358]}),
        references=16, class_names=base.NAMES, source_selection=selections,
        selection='Source-inner BA/OA/-NLL, step0 eligible. Readout changes are nonzero at initialization; initial predictions need not match.',
        controls='legacy is reused continue_s17. Effective optimizer groups and 120-step recipe match; no new parameters.',
        estimands='Matched arms include training plus readout differences. native-vs-legacyread holds selected weights fixed and switches both query/ref readers. legacyread-vs-continue compares learned weights under the legacy reader; it is not pure causal training mediation.',
        limitation='A same-weight readout swap can create train/inference mismatch. These ten evaluations do not isolate M1 feedback from live-vs-anchor conv3/BN or isolate layer1 readout from the changed layer2 interpolation/pooling.',
        geometry='Maximum OBB L across actually available polarizations: <16, 16<=L<32, 32<=L<64, >=64; overall, class, length and class-by-length groups.',
        metrics='OA/BA in percent; paired differences in percentage points. BA uses classes present in each group; empty groups null. Margin is truth minus strongest non-true class.',
        units='Observation counts and unique MMSI counts are distinct. One seed; no seed variance or independent replication claim. Contrast events must not be pooled as new ships.',
        mechanism='Existing features are read differently; no new image information, dynamic tokens, pairwise axis matching or bow/stern capability. Length strata and class composition are descriptive, not size causality.',
        scope='Reused 160/198 development cohorts; expansion has new vessels in known passages, not independent scene confirmation. No 3075 test access.',
        privacy='Only aggregate/ is shareable. IDs, file paths, scores and images remain in local notes/gallery.',
        cpu_only=True)
    write(out / 'summary.json', dict(protocol=protocol,
        conditions=[r for r in conditions if r['group_type'] == 'overall'],
        contrasts=[r for r in paired if r['group_type'] == 'overall']))
    assert 'torch' not in sys.modules
    print(f'CPU analysis complete: {len(CONDITIONS)*2} evaluations, {len(cases)} native panels; visual review pending.')
    for row in paired:
        if row['group_type'] == 'overall':
            print(row['split'], row['contrast'], 'delta_OA/BA', row['delta_oa_pp'], row['delta_ba_pp'],
                  'rescue/harm', row['rescued'], row['harmed'])


if __name__ == '__main__':
    main()
