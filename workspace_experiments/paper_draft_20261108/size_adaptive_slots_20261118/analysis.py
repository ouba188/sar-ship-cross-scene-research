"""CPU analysis of the 18 main frozen evaluations; side_* probes are excluded.

Run with python -B analysis.py. Only aggregate/ is suitable for public sharing.
Cases and native gallery remain local. No model, checkpoint, or torch import.
"""
from collections import Counter, defaultdict
import csv
import html
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
MODES = ('fixed', 'coarse', 'adaptive')
SPLITS = {'validation': np.arange(160), 'expansion': np.arange(160, 358)}
REFS = list(range(358, 374))
NAMES = ['Bulk', 'Fishing', 'General cargo', 'Chemical/product', 'Container', 'Crude', 'Tug', 'Offshore']
KINDS = ('rescued', 'harmed', 'stable_correct', 'stable_wrong')
NATIVE_REVIEW, PREFERRED = {}, {}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def helper():
    path = ROOT.parent / 'ports24_sync_20261112/analyze.py'
    spec = importlib.util.spec_from_file_location('slot_native_analysis_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scores(rows):
    status = read(ROOT / 'status.json')
    if not (status.get('complete') is True and status.get('smoke') is False
            and status.get('evaluations') == 18 and status.get('trained') is False):
        raise ValueError('All 18 full evaluations must finish; smoke is not accepted.')
    if len(rows) != 374 or len({r['id'] for r in rows}) != 374:
        raise ValueError('Expected 358 distinct queries and 16 distinct references.')
    scores = {}
    for split, ids in SPLITS.items():
        y = np.array([rows[i]['label'] for i in ids])
        for seed in SEEDS:
            for mode in MODES:
                path = ROOT / f'{split}_s{seed}_{mode}_local.npz'
                with np.load(path, allow_pickle=False) as data:
                    if not np.array_equal(data['ids'], ids) or not np.array_equal(data['label'], y):
                        raise ValueError(f'Incomplete or misaligned IDs/labels: {path.name}')
                    z = data['score'].astype(np.float64)
                if z.shape != (len(ids), 8) or not np.isfinite(z).all():
                    raise ValueError(f'Invalid logits: {path.name}')
                scores[split, seed, mode] = z
    return scores


class Panels:
    def __init__(self, rows, native):
        self.rows = rows
        api = native.native_api()
        self.views = [api.prepare(row) for row in rows]
        self.available = np.array([[v is not None and v['valid_target'] for v in pair]
                                   for pair in self.views], dtype=bool)

    def row(self, arm, index, reference=False):
        return self.rows[index]

    def view(self, arm, index, pol, reference=False):
        return self.views[index][('vv', 'vh').index(pol)]


def lengths_from_geometry(rows, geometry, panels):
    tables = {mode: {(r['id'], r['polarization']): r for r in geometry[mode]} for mode in MODES}
    length = []
    for i, row in enumerate(rows):
        values = []
        for p, pol in enumerate(('vv', 'vh')):
            if not panels.available[i, p]:
                continue
            records = [tables[mode][row['id'], pol] for mode in MODES]
            if any(r['length'] != records[0]['length'] or r['width'] != records[0]['width'] for r in records):
                raise ValueError('Geometry changes across modes.')
            actual = panels.views[i][p]['geometry']
            if not np.allclose(actual[2:4], [records[0]['length'], records[0]['width']], atol=1e-4, rtol=0):
                raise ValueError(f'Granularity OBB differs from prepared input: {row["id"]}/{pol}')
            values.append(records[0]['length'])
        if not values:
            raise ValueError('Query/reference has no usable polarization.')
        length.append(max(values))
    return np.array(length), tables


def margin(z, y):
    wrong = z.copy()
    wrong[np.arange(len(y)), y] = -np.inf
    return z[np.arange(len(y)), y] - wrong.max(1)


def state_masks(before, after, y):
    b, a = before.argmax(1) == y, after.argmax(1) == y
    return dict(rescued=~b & a, harmed=b & ~a, stable_correct=b & a, stable_wrong=~b & ~a)


def metrics(y, z):
    if not len(y):
        return dict(oa=None, ba=None, margin_mean=None, margin_median=None)
    correct = z.argmax(1) == y
    recall = [correct[y == c].mean() for c in np.unique(y)]
    m = margin(z, y)
    return dict(oa=float(100 * correct.mean()), ba=float(100 * np.mean(recall)),
                margin_mean=float(m.mean()), margin_median=float(np.median(m)))


def group_masks(y, length):
    yield 'overall', 'all', np.ones(len(y), dtype=bool)
    for c in range(8):
        yield 'class', str(c), y == c
    for label, mask in (('<32', length < 32), ('32-64', (length >= 32) & (length < 64)), ('>=64', length >= 64)):
        yield 'length', label, mask


def statistics(rows, scores, length):
    records, repeated = [], []
    for split, ids in SPLITS.items():
        y = np.array([rows[i]['label'] for i in ids])
        for kind, group, mask in group_masks(y, length[ids]):
            for mode in MODES:
                states_by_seed = []
                for seed in SEEDS:
                    base, z = scores[split, seed, 'fixed'][mask], scores[split, seed, mode][mask]
                    labels = y[mask]
                    bm, am = metrics(labels, base), metrics(labels, z)
                    states = state_masks(base, z, labels)
                    states_by_seed.append(states)
                    dm = margin(z, labels) - margin(base, labels)
                    records.append(dict(split=split, seed=seed, mode=mode, group_type=kind, group=group,
                        n=int(mask.sum()), n_vessels=len({rows[i]['mmsi'] for i in ids[mask]}),
                        present_classes=len(np.unique(labels)), **am,
                        fixed_oa=bm['oa'], fixed_ba=bm['ba'],
                        delta_oa_pp=None if not len(labels) else am['oa']-bm['oa'],
                        delta_ba_pp=None if not len(labels) else am['ba']-bm['ba'],
                        delta_margin_mean=float(dm.mean()) if len(dm) else None,
                        delta_margin_median=float(np.median(dm)) if len(dm) else None,
                        **{k: int(v.sum()) for k, v in states.items()}))
                repeated.append(dict(split=split, mode=mode, group_type=kind, group=group,
                    n_unique_observations=int(mask.sum()),
                    **{f'{k}_{rule}_seed_observations': int(getattr(np.stack([s[k] for s in states_by_seed]), rule)(0).sum())
                       for k in KINDS for rule in ('any', 'all')}))
    buckets = defaultdict(list)
    for row in records:
        buckets[tuple(row[k] for k in ('split', 'mode', 'group_type', 'group'))].append(row)
    means = []
    values = ('oa', 'ba', 'margin_mean', 'margin_median', 'fixed_oa', 'fixed_ba',
              'delta_oa_pp', 'delta_ba_pp', 'delta_margin_mean', 'delta_margin_median', *KINDS)
    for key, group in buckets.items():
        out = dict(zip(('split', 'mode', 'group_type', 'group'), key))
        out.update(n_unique_observations=group[0]['n'], n_vessels=group[0]['n_vessels'], n_seeds=3)
        for name in values:
            a = [r[name] for r in group if r[name] is not None]
            out.update({f'{name}_seed_mean': float(np.mean(a)) if a else None,
                        f'{name}_seed_min': min(a) if a else None,
                        f'{name}_seed_max': max(a) if a else None})
        means.append(out)
    return records, means, repeated


def candidates(rows, scores):
    items = []
    for split, ids in SPLITS.items():
        y = np.array([rows[i]['label'] for i in ids])
        for seed in SEEDS:
            base = scores[split, seed, 'fixed']
            for mode in ('adaptive', 'coarse'):
                z = scores[split, seed, mode]
                before, after = margin(base, y), margin(z, y)
                for kind, mask in state_masks(base, z, y).items():
                    for j in np.flatnonzero(mask):
                        items.append(dict(index=int(ids[j]), position=int(j), split=split, seed=seed,
                            mode=mode, kind=kind, delta_margin=float(after[j]-before[j])))
    return items


def choose_cases(items, rows):
    selected, used = [], set()
    requests = [('validation', 'rescued', PREFERRED.get('rescued', ())),
                ('validation', 'harmed', PREFERRED.get('harmed', ())),
                ('expansion', 'rescued', ()), ('expansion', 'harmed', ()),
                (None, 'stable_correct', ()), (None, 'stable_wrong', ())]
    for split, kind, preferred in requests:
        pool = [r for r in items if r['kind'] == kind and r['index'] not in used
                and (split is None or r['split'] == split)]
        if not pool:
            pool = [r for r in items if r['kind'] == kind and r['index'] not in used]
        if not pool:
            continue
        def order(r):
            names = (rows[r['index']]['id'], str(rows[r['index']].get('corpus_global_row', '')))
            priority = next((i for i, name in enumerate(preferred) if name in names), len(preferred))
            return priority, r['mode'] != 'adaptive', -abs(r['delta_margin']), r['index'], r['seed']
        item = min(pool, key=order)
        selected.append(item)
        used.add(item['index'])
    return selected


def gallery(rows, scores, length, tables, panels, native):
    folder = ROOT / 'gallery'
    folder.mkdir(exist_ok=True)
    items = candidates(rows, scores)
    selected = choose_cases(items, rows)
    cases = []
    for item in selected:
        i, j, seed, mode, split = (item[k] for k in ('index', 'position', 'seed', 'mode', 'split'))
        row, truth = rows[i], int(rows[i]['label'])
        b, a = scores[split, seed, 'fixed'][j], scores[split, seed, mode][j]
        rival_score = b.copy()
        rival_score[truth] = -np.inf
        rival = int(rival_score.argmax())
        bank = native.legal_bank(rows, {'references': REFS}, panels.available, i)
        filename = f'{len(cases):02d}_{item["kind"]}_{row["id"]}.png'
        bm, am = float(margin(b[None], np.array([truth]))[0]), float(margin(a[None], np.array([truth]))[0])
        caption = (f'{split} {mode} vs fixed | {item["kind"]} s{seed} | truth {truth} '
                   f'| pred {b.argmax()} -> {a.argmax()} | margin {bm:+.4f} -> {am:+.4f}')
        paths, gaps = native.draw_panel(panels, i, 'fixed', mode, bank, bank, truth, rival,
                                       NAMES, folder / filename, caption)
        evidence = []
        for s in SEEDS:
            before, after = scores[split, s, 'fixed'][j], scores[split, s, mode][j]
            state = state_masks(before[None], after[None], np.array([truth]))
            evidence.append(dict(seed=s, kind=next(k for k in KINDS if state[k][0]),
                fixed_prediction=int(before.argmax()), prediction=int(after.argmax()),
                fixed_margin=float(margin(before[None], np.array([truth]))[0]),
                margin=float(margin(after[None], np.array([truth]))[0]),
                fixed_scores=before.tolist(), scores=after.tolist()))
        cases.append(dict(**item, id=row['id'], corpus_global_row=row.get('corpus_global_row'),
            label=truth, rival=rival, rival_rule='strongest non-true class under fixed, selected seed',
            max_available_length=float(length[i]), file=filename, evidence=evidence,
            actual_source16_bank=bank, paths=paths, missing_reference_roles=gaps,
            query_geometry={m: [tables[m][row['id'], p] for p in ('vv', 'vh')
                               if (row['id'], p) in tables[m]] for m in MODES},
            visual_review=dict(status='viewed_native_panel' if row['id'] in NATIVE_REVIEW else 'pending',
                observations=NATIVE_REVIEW.get(row['id']),
                interpretation='Observed shape, intensity and context differences do not identify the cause of the prediction change. No physical endpoint identity or reference influence is established.')))
    write(ROOT / 'cases_local.json', cases)
    # Full flip list stays local; it supports checking requested cases not in the six panels.
    write(ROOT / 'flips_local.json', [dict(**r, id=rows[r['index']]['id']) for r in items
                                     if r['kind'] in ('rescued', 'harmed')])
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Slot granularity cases</title>',
        '<style>body{font:15px/1.5 system-ui;margin:24px}.panel{overflow-x:auto}img{max-width:none}</style>',
        '<h1>Frozen slot-granularity cases</h1><p>Native pixels, unchanged intensity and scale. '
        'Fixed vs candidate use identical input images. References are the first active legal examples '
        'in the fixed bank, not influence rankings. Rival is the fixed strongest non-true class. '
        'Outcome-selected cases do not estimate population proportions. Stable wrong means both predictions '
        'are wrong, not necessarily identical. Visual review is recorded in cases_local.json.</p>']
    for case in cases:
        body.append(f'<h2>{html.escape(case["id"])} / {case["kind"]} / {case["mode"]} / seed {case["seed"]}</h2>'
                    f'<div class="panel"><img src="{html.escape(case["file"])}" alt="Native query and references"></div>')
        review = case['visual_review']
        if review['observations']:
            body.append(f'<p>{html.escape(review["observations"])}</p><p>{html.escape(review["interpretation"])}</p>')
    (folder / 'index.html').write_text('\n'.join(body)+'</html>', encoding='utf-8')
    return cases, [dict(kind=k, selected=sum(c['kind'] == k for c in cases),
                        eligible_unique_queries=len({r['index'] for r in items if r['kind'] == k})) for k in KINDS]


def main():
    global NATIVE_REVIEW, PREFERRED
    notes_path = ROOT / 'case_notes_local.json'
    if notes_path.exists():
        private_notes = read(notes_path)
        NATIVE_REVIEW, PREFERRED = private_notes['observations'], private_notes['preferred']
    rows = read(ROOT / 'manifest_local.json')
    scores = load_scores(rows)
    native = helper()
    panels = Panels(rows, native)
    length, tables = lengths_from_geometry(rows, read(ROOT / 'geometry_local.json'), panels)
    records, means, repeated = statistics(rows, scores, length)
    cases, coverage = gallery(rows, scores, length, tables, panels, native)
    out = ROOT / 'aggregate'
    out.mkdir(exist_ok=True)
    csv_write(out / 'per_seed_groups.csv', records)
    csv_write(out / 'seed_summary.csv', means)
    csv_write(out / 'paired_observation_counts.csv', repeated)
    csv_write(out / 'case_coverage.csv', coverage)
    notes = dict(evaluations=18, seeds=list(SEEDS), query_counts={k: len(v) for k, v in SPLITS.items()},
        class_names=NAMES,
        references=16, side_probes_included=False, units='OA/BA and deltas in percent/percentage points; margins in logits',
        ba='Mean recall over true classes present in the group; class-specific BA equals recall; empty groups are null.',
        lengths='Maximum OBB L among actually usable polarizations; bins L<32, 32<=L<64, L>=64.',
        repeated_seeds='Same queries across three continuation seeds; seed means are descriptive, not independent-query evidence or an ensemble.',
        observation_counts='Any/all seed counts are deduplicated by observation within split/mode/group; not disjoint across outcomes.',
        scope='Existing development cohorts. Expansion comprises new vessels in known passages, not independent passage validation.',
        causality='Grouping by length is descriptive. Both query and reference coarsening change nonlinear encoding, matching and writeback; size alone is not established as cause.',
        cases='At most six distinct outcome-selected queries, with preferred validation flip IDs and expansion rescue/harm; no private cases in aggregate.',
        cpu_only=True)
    write(out / 'summary.json', dict(protocol=notes, overall=[r for r in means if r['group_type'] == 'overall']))
    if 'torch' in sys.modules:
        raise RuntimeError('CPU analysis unexpectedly imported torch.')
    print(json.dumps(dict(evaluations=18, grouped_rows=len(records), gallery_cases=len(cases),
                          case_ids=[c['id'] for c in cases], cpu_only=True), ensure_ascii=False))
    for row in means:
        if row['group_type'] == 'overall':
            print(row['split'], row['mode'], 'OA', row['oa_seed_mean'], 'BA', row['ba_seed_mean'],
                  'delta', row['delta_oa_pp_seed_mean'], row['delta_ba_pp_seed_mean'],
                  'rescued/harmed', row['rescued_seed_mean'], row['harmed_seed_mean'])


if __name__ == '__main__':
    main()
