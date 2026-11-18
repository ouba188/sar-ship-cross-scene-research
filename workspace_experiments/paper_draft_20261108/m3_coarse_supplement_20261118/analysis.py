"""CPU-only paired analysis of three continuation arms and supplement on/off.

python -B analysis.py --wait
Only aggregate/ is public. Native gallery and case_notes_local.json stay local.
No model imports, training audits, configuration selection, or test images.
"""
import argparse
from collections import Counter, defaultdict
import html
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
ARMS = ('continue', 'pointwise', 'coarse_plus')
CONDITIONS = (('continue', 'on'), ('pointwise', 'on'), ('pointwise', 'off'),
              ('coarse_plus', 'on'), ('coarse_plus', 'off'))
CONTRASTS = (
    ('coarse_vs_pointwise', ('coarse_plus', 'on'), ('pointwise', 'on')),
    ('coarse_vs_continue', ('coarse_plus', 'on'), ('continue', 'on')),
    ('pointwise_vs_continue', ('pointwise', 'on'), ('continue', 'on')),
    ('coarse_on_vs_off', ('coarse_plus', 'on'), ('coarse_plus', 'off')),
    ('pointwise_on_vs_off', ('pointwise', 'on'), ('pointwise', 'off')),
)
SPLITS = {'validation': np.arange(160), 'expansion': np.arange(160, 358)}
KINDS = ('rescued', 'harmed', 'stable_correct', 'stable_wrong')
MAX_CASES = 12


def load_helper():
    path = ROOT.parent / 'size_adaptive_slots_20261118/analysis.py'
    spec = importlib.util.spec_from_file_location('supplement_native_analysis_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_helper()
read, write, csv_write = base.read, base.write, base.csv_write


def cache_path(split, seed, arm, variant):
    return ROOT / 'fits' / f'{arm}_s{seed}' / f'{split}_{variant}_local.npz'


def ready():
    paths = [ROOT / 'results.json', ROOT / 'evaluation_inputs/manifest_local.json', ROOT / 'status.json']
    paths += [ROOT / 'fits' / f'{a}_s{s}/receipt.json' for a in ARMS for s in SEEDS]
    paths += [cache_path(split, s, a, v) for split in SPLITS for s in SEEDS for a, v in CONDITIONS]
    if not all(p.is_file() for p in paths):
        return False
    try:
        status, results = read(ROOT / 'status.json'), read(ROOT / 'results.json')
    except json.JSONDecodeError:
        return False
    return status.get('phase') == 'complete' and status.get('evaluations') == 30 and len(results) == 30


def load_scores(rows):
    if len(rows) != 374 or len({r['id'] for r in rows}) != 374:
        raise ValueError('Expected 358 distinct observations and 16 references.')
    entries = read(ROOT / 'results.json')
    table = {(r['split'], r['seed'], r['arm'], r['variant']): r for r in entries}
    expected = {(split, s, a, v) for split in SPLITS for s in SEEDS for a, v in CONDITIONS}
    if set(table) != expected or len(entries) != len(expected):
        raise ValueError('Expected exactly 30 evaluation conditions; continue has no off condition.')
    scores = {}
    for key in sorted(expected):
        split, seed, arm, variant = key
        ids = SPLITS[split]
        y = np.array([rows[i]['label'] for i in ids])
        with np.load(cache_path(*key), allow_pickle=False) as data:
            if not np.array_equal(data['ids'], ids) or not np.array_equal(data['label'], y):
                raise ValueError(f'Misaligned evaluation IDs or labels: {key}')
            z = data['score'].astype(np.float64)
        if z.shape != (len(ids), 8) or not np.isfinite(z).all():
            raise ValueError(f'Invalid scores: {key}')
        actual = base.metrics(y, z)
        if any(abs(actual[k]-table[key][k]) > 1e-8 for k in ('oa', 'ba')):
            raise ValueError(f'Cache and results.json disagree: {key}')
        scores[key] = z
    selected = [dict(arm=a, seed=s, selected_step=read(ROOT/'fits'/f'{a}_s{s}/receipt.json')['best']['step'])
                for a in ARMS for s in SEEDS]
    for item in selected:
        if item['selected_step'] not in (0, 30, 60, 90, 120):
            raise ValueError('Unexpected source-selected checkpoint step.')
        for split in SPLITS:
            for arm, variant in CONDITIONS:
                if arm == item['arm'] and table[split, item['seed'], arm, variant]['selected_step'] != item['selected_step']:
                    raise ValueError('Evaluation checkpoint selection metadata disagrees.')
    return scores, selected


def groups(y, lengths):
    yield 'overall', 'all', np.ones(len(y), dtype=bool)
    bins = (('<16', lengths < 16), ('16-32', (lengths >= 16) & (lengths < 32)),
            ('32-64', (lengths >= 32) & (lengths < 64)), ('>=64', lengths >= 64))
    for c in range(8):
        yield 'class', str(c), y == c
    for label, mask in bins:
        yield 'length', label, mask
    for c in range(8):
        for label, mask in bins:
            yield 'class_length', f'{c}:{label}', (y == c) & mask


def seed_summary(records, keys, fields):
    buckets = defaultdict(list)
    for row in records:
        buckets[tuple(row[k] for k in keys)].append(row)
    summaries = []
    for key, records in buckets.items():
        if {r['seed'] for r in records} != set(SEEDS) or len(records) != 3:
            raise ValueError('Seed summary must contain three paired continuation runs.')
        out = dict(zip(keys, key))
        out.update(n_observations=records[0]['n_observations'], n_ships=records[0]['n_ships'], n_seeds=3)
        for field in fields:
            values = [r[field] for r in records if r[field] is not None]
            out.update({f'{field}_seed_mean': float(np.mean(values)) if values else None,
                        f'{field}_seed_min': min(values) if values else None,
                        f'{field}_seed_max': max(values) if values else None})
        summaries.append(out)
    return summaries


def statistics(rows, scores, lengths):
    conditions, paired, unique, candidates = [], [], [], []
    for split, ids in SPLITS.items():
        y = np.array([rows[i]['label'] for i in ids])
        for group_type, group, mask in groups(y, lengths[ids]):
            selected_ids, labels = ids[mask], y[mask]
            ships = np.array([rows[i]['mmsi'] for i in selected_ids], dtype=object)
            common = dict(split=split, group_type=group_type, group=group,
                          n_observations=len(selected_ids), n_ships=len(set(ships)),
                          present_classes=len(np.unique(labels)))
            for arm, variant in CONDITIONS:
                for seed in SEEDS:
                    conditions.append(dict(**common, arm=arm, variant=variant, seed=seed,
                                           **base.metrics(labels, scores[split, seed, arm, variant][mask])))
            for name, after, before in CONTRASTS:
                states = []
                for seed in SEEDS:
                    b, a = scores[split, seed, *before][mask], scores[split, seed, *after][mask]
                    bm, am = base.metrics(labels, b), base.metrics(labels, a)
                    state = base.state_masks(b, a, labels)
                    states.append(state)
                    delta = base.margin(a, labels)-base.margin(b, labels)
                    paired.append(dict(**common, contrast=name, seed=seed,
                        before_oa=bm['oa'], after_oa=am['oa'], before_ba=bm['ba'], after_ba=am['ba'],
                        delta_oa_pp=am['oa']-bm['oa'] if len(labels) else None,
                        delta_ba_pp=am['ba']-bm['ba'] if len(labels) else None,
                        delta_margin_mean=float(delta.mean()) if len(labels) else None,
                        delta_margin_median=float(np.median(delta)) if len(labels) else None,
                        **{k: int(v.sum()) for k, v in state.items()},
                        **{f'{k}_ships': len(set(ships[v])) for k, v in state.items()}))
                    if group_type == 'overall':
                        for kind in KINDS:
                            for j in np.flatnonzero(state[kind]):
                                candidates.append(dict(split=split, seed=seed, contrast=name, kind=kind,
                                    index=int(ids[j]), position=int(j), delta_margin=float(delta[j])))
                row = dict(**common, contrast=name)
                for kind in KINDS:
                    stacked = np.stack([s[kind] for s in states])
                    ship_sets = [set(ships[s[kind]]) for s in states]
                    row.update({f'{kind}_seed_observation_events': int(stacked.sum()),
                                f'{kind}_seed_ship_events': sum(map(len, ship_sets)),
                                f'{kind}_observations_any_seed': int(stacked.any(0).sum()),
                                f'{kind}_observations_all_seeds': int(stacked.all(0).sum()),
                                f'{kind}_ships_any_seed': len(set.union(*ship_sets)),
                                f'{kind}_ships_all_seeds': len(set.intersection(*ship_sets))})
                unique.append(row)
    return conditions, paired, unique, candidates


def choose_cases(candidates):
    selected, used = [], set()
    def pick(pool):
        pool = [r for r in pool if r['index'] not in used]
        if not pool or len(selected) >= MAX_CASES:
            return
        counts = Counter(r['contrast'] for r in selected)
        split_counts = Counter(r['split'] for r in selected)
        item = min(pool, key=lambda r: (counts[r['contrast']], split_counts[r['split']],
                                       -abs(r['delta_margin']), r['index'], r['seed']))
        selected.append(item)
        used.add(item['index'])
    for seed in SEEDS:
        for kind in ('rescued', 'harmed'):
            pick([r for r in candidates if r['seed'] == seed and r['kind'] == kind
                  and r['contrast'].startswith('coarse_')])
        if not any(r['seed'] == seed for r in selected):
            pick([r for r in candidates if r['seed'] == seed and r['contrast'].startswith('coarse_')])
    for kind in ('stable_correct', 'stable_wrong'):
        pick([r for r in candidates if r['kind'] == kind and r['contrast'].startswith('coarse_')])
    for name, _, _ in CONTRASTS:
        if not any(r['contrast'] == name for r in selected):
            pool = [r for r in candidates if r['contrast'] == name]
            flips = [r for r in pool if r['kind'] in ('rescued', 'harmed')]
            pick(flips or pool)
    for split in SPLITS:
        for kind in ('rescued', 'harmed'):
            if not any(r['split'] == split and r['kind'] == kind for r in selected):
                pick([r for r in candidates if r['split'] == split and r['kind'] == kind])
    return selected


def make_gallery(rows, scores, panels, native, candidates, lengths):
    folder = ROOT / 'gallery'
    folder.mkdir(exist_ok=True)
    contrasts = {name: (after, before) for name, after, before in CONTRASTS}
    notes_path = ROOT / 'case_notes_local.json'
    existing = read(notes_path).get('cases', []) if notes_path.exists() else []
    old_notes = {(c['id'], c['contrast'], c['seed']): c.get('visual_review') for c in existing}
    review_path = ROOT / 'visual_review_local.json'
    reviewed = read(review_path) if review_path.exists() else {}
    cases = []
    for item in choose_cases(candidates):
        split, seed, i, j = (item[k] for k in ('split', 'seed', 'index', 'position'))
        after, before = contrasts[item['contrast']]
        b, a = scores[split, seed, *before][j], scores[split, seed, *after][j]
        row, truth = rows[i], int(rows[i]['label'])
        competing = b.copy()
        competing[truth] = -np.inf
        rival = int(competing.argmax())
        bank = native.legal_bank(rows, {'references': list(range(358, 374))}, panels.available, i)
        filename = f'{len(cases):02d}_{item["kind"]}_{row["id"]}.png'
        evidence = []
        for s in SEEDS:
            bb, aa = scores[split, s, *before][j], scores[split, s, *after][j]
            state = base.state_masks(bb[None], aa[None], np.array([truth]))
            evidence.append(dict(seed=s, kind=next(k for k in KINDS if state[k][0]),
                before_prediction=int(bb.argmax()), after_prediction=int(aa.argmax()),
                before_margin=float(base.margin(bb[None], np.array([truth]))[0]),
                after_margin=float(base.margin(aa[None], np.array([truth]))[0]),
                before_scores=bb.tolist(), after_scores=aa.tolist()))
        chosen = next(e for e in evidence if e['seed'] == seed)
        caption = (f'{split} {item["contrast"]} | {item["kind"]} s{seed} | truth {truth} | '
                   f'pred {b.argmax()} -> {a.argmax()} | '
                   f'margin {chosen["before_margin"]:+.4f} -> {chosen["after_margin"]:+.4f}')
        paths, gaps = native.draw_panel(panels, i, '-'.join(before), '-'.join(after), bank, bank,
                                       truth, rival, base.NAMES, folder / filename, caption)
        review = reviewed.get(row['id']) or old_notes.get((row['id'], item['contrast'], seed)) or dict(status='pending', observations=None,
            limitation='Images alone do not establish size causality, physical endpoint identity, or reference influence.')
        cases.append(dict(**item, id=row['id'], label=truth, rival=rival, file=filename,
            max_available_length=float(lengths[i]), before_condition=list(before), after_condition=list(after),
            rival_rule='Strongest non-true class of the selected baseline condition and seed.',
            evidence=evidence, actual_source16_bank=bank, paths=paths,
            missing_reference_roles=gaps, visual_review=review))
    coverage = []
    for contrast, _, _ in CONTRASTS:
        for seed in SEEDS:
            for kind in KINDS:
                available = {r['index'] for r in candidates if r['contrast'] == contrast and r['seed'] == seed and r['kind'] == kind}
                shown = sum(r['contrast'] == contrast and r['seed'] == seed and r['kind'] == kind for r in cases)
                coverage.append(dict(contrast=contrast, seed=seed, kind=kind,
                    eligible_observations=len(available), selected_panels=shown,
                    note='no_eligible_event' if not available else 'representative_panel_budget' if not shown else 'selected'))
    write(notes_path, dict(selection='At most 12 distinct observations; prioritize coarse rescue/harm across three seeds, stable outcomes, and contrast coverage. Cases are not population estimates.',
                          cases=cases))
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Coarse supplement cases</title>',
        '<style>body{font:15px/1.5 system-ui;margin:24px}.panel{overflow-x:auto}img{max-width:none}</style>',
        '<h1>Fine-slot coarse supplement: paired cases</h1><p>Native 160-pixel canvases; no resizing or intensity stretching. '
        'Query pixels and the 16 source references are unchanged across conditions. References shown are the first legal active '
        'same-class and baseline-rival examples, not contribution rankings. Supplement off is a same-checkpoint path intervention; '
        'it is not the matched continue arm. These operators reorganize existing features, adding no raw-image information or endpoint capability. '
        'Three continuation seeds repeat observations and ships. Visual review status is stored in case_notes_local.json.</p>']
    for case in cases:
        body.append(f'<h2>{html.escape(case["id"])} / {case["contrast"]} / {case["kind"]} / seed {case["seed"]}</h2>'
                    f'<div class="panel"><img src="{html.escape(case["file"])}" alt="Native query and actual references"></div>'
                    f'<p>Review: {html.escape(case["visual_review"]["status"])}</p>')
        if case['visual_review'].get('observations'):
            body.append(f'<p>{html.escape(case["visual_review"]["observations"])}</p>')
    (folder / 'index.html').write_text('\n'.join(body)+'</html>', encoding='utf-8')
    return cases, coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true', help='Wait for the existing run; never launches inference.')
    args = parser.parse_args()
    announced = False
    while not ready():
        if not args.wait:
            raise SystemExit('Waiting for nine completed fits and all 30 evaluations; no analysis files written.')
        if not announced:
            print('Waiting for existing training/evaluation completion; CPU analysis only.', flush=True)
            announced = True
        time.sleep(15)
    rows = read(ROOT / 'evaluation_inputs/manifest_local.json')
    scores, selected_steps = load_scores(rows)
    native = base.helper()
    panels = base.Panels(rows, native)
    lengths = np.array([max(v['geometry'][2] for p, v in enumerate(pair) if panels.available[i, p])
                        for i, pair in enumerate(panels.views)])
    conditions, paired, unique, candidates = statistics(rows, scores, lengths)
    condition_summary = seed_summary(conditions, ('split', 'arm', 'variant', 'group_type', 'group'),
                                     ('oa', 'ba', 'margin_mean', 'margin_median'))
    paired_summary = seed_summary(paired, ('split', 'contrast', 'group_type', 'group'),
        ('before_oa', 'after_oa', 'before_ba', 'after_ba', 'delta_oa_pp', 'delta_ba_pp',
         'delta_margin_mean', 'delta_margin_median', *KINDS, *(f'{k}_ships' for k in KINDS)))
    cases, coverage = make_gallery(rows, scores, panels, native, candidates, lengths)
    out = ROOT / 'aggregate'
    out.mkdir(exist_ok=True)
    for name, records in (('condition_per_seed', conditions), ('paired_per_seed', paired),
                          ('condition_seed_summary', condition_summary), ('paired_seed_summary', paired_summary),
                          ('unique_observation_ship_counts', unique), ('gallery_coverage', coverage)):
        csv_write(out / f'{name}.csv', records)
    notes = dict(evaluations=30, arms=list(ARMS), continuation_seeds=list(SEEDS), class_names=base.NAMES,
        observations={'validation': 160, 'expansion': 198}, references=16, source_selected_steps=selected_steps,
        source_selection='120 updates; source-inner BA/OA/-NLL, step 0 eligible. No external-label configuration selection here.',
        start='All runs start from selected routed_s17, with the same new-branch initialization; seeds change order/augmentation.',
        group_definition='Actual available-polarization maximum OBB L: <16, 16<=L<32, 32<=L<64, >=64 pixels; overall, class, length, and class-by-length.',
        metrics='OA/BA percentages, deltas in percentage points. BA uses true classes present in each group; empty groups null. Margins use strongest non-true class.',
        repeated_units='Seed means are descriptive, not independent queries or ensemble predictions. n_observations and n_ships are not multiplied by three.',
        count_definition='Seed-observation events count occurrences; seed-ship events deduplicate MMSI within each seed. Any/all-seed observations intersect observation IDs; any/all-seed ships intersect MMSI sets. Outcomes can overlap across seeds.',
        matched_vs_off='Arm comparisons estimate matched continuation differences; same-weight on/off isolates enabling the learned supplement at inference, not its training effects.',
        information='Reorganizes existing fine-slot features; no new raw SAR information or bow/stern capability.',
        composition='Length and within-class strata describe outcomes, not mechanisms; size, class composition, observation conditions and reference composition remain coupled.',
        evaluation_scope='Reused development cohorts; expansion is new vessels in known passages, not independent passage confirmation.',
        privacy='Only aggregate is shareable; raw images, IDs, individual scores and paths remain in local case notes/gallery.', cpu_only=True)
    write(out / 'summary.json', dict(protocol=notes,
        conditions=[r for r in condition_summary if r['group_type'] == 'overall'],
        contrasts=[r for r in paired_summary if r['group_type'] == 'overall']))
    if 'torch' in sys.modules:
        raise RuntimeError('Analysis unexpectedly imported torch.')
    print(json.dumps(dict(evaluations=30, paired_group_rows=len(paired), native_cases=len(cases), cpu_only=True)), flush=True)
    for row in paired_summary:
        if row['group_type'] == 'overall':
            print(row['split'], row['contrast'], 'delta_OA/BA', row['delta_oa_pp_seed_mean'], row['delta_ba_pp_seed_mean'],
                  'rescue/harm', row['rescued_seed_mean'], row['harmed_seed_mean'], flush=True)


if __name__ == '__main__':
    main()
