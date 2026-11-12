"""CPU analysis after all three Full-Tuned runs finish.

python -B analyze.py --check-inputs  # Read existing schemas; do not wait or write.
python -B analyze.py --self-test     # In-memory CPU checks; no predictions invented.
python -B analyze.py                # Run manually after training and evaluation.

Outputs: analysis/aggregate, analysis/private, analysis/gallery. Never imports
train.py, torch or a model. Completed recovery arms use recovery.py for panels
only. Historical Plain/SE are recipe-unmatched.
"""
import argparse
from collections import Counter, defaultdict
import csv
import html
import importlib.util
import io
import json
from pathlib import Path
import sys
import textwrap

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
STORE = Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')
SEEDS = (17, 29, 43)
CLASS_COUNT = 8
ARMS = ('Full-Tuned', 'Full', 'Plain', 'SE')
BASELINES = ARMS[1:]
METRICS = ('oa', 'ba', 'macro_f1', 'nll', 'brier')
KINDS = ('rescued', 'harmed', 'stable_correct', 'stable_wrong')
RECOVERY_ARMS = {
    'Old-Query': 'old_query', 'Old-QueryRef': 'old_query_reference',
    'Tuned-Query': 'tuned_query', 'Tuned-QueryRef': 'tuned_query_reference',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, records):
    if not records:
        return
    fields = list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(records)


def orbit(row):
    return '_'.join(row['pass_group'].split('_')[:2])


def metadata():
    rows = read_json(STORE / 'manifest.json')
    groups = read_json(STORE / 'groups.json')
    for name, expected in (('small', 20653), ('validation', 160), ('evaluation', 3075), ('references', 16)):
        if len(groups[name]) != expected:
            raise ValueError(f'Wrong frozen {name} count: {len(groups[name])}')
    with (WORK / 'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv').open(
            encoding='utf-8-sig', newline='') as handle:
        corpus = list(csv.DictReader(handle))
    names = {}
    for row in rows:
        original = corpus[row['corpus_global_row']]
        c, name = int(row['label']), original['class_name']
        if int(original['class_id']) != c or c in names and names[c] != name:
            raise ValueError('Frozen manifest and corpus disagree on class order')
        names[c] = name
    if set(names) != set(range(CLASS_COUNT)):
        raise ValueError('Expected all eight corpus class IDs')
    return rows, groups, [names[c] for c in range(CLASS_COUNT)]


def native_api():
    sys.dont_write_bytecode = True
    path = WORK / 'sar_m123_standalone/sar_m123/input.py'
    spec = importlib.util.spec_from_file_location('ports24_analysis_native_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def availability(rows):
    live, audited = ROOT / 'native_cache', ROOT / 'polarization'
    if (live / 'complete.json').is_file():
        path = live / 'available.npy'
        origin = 'completed_current_native_cache'
    elif (audited / 'native_availability.json').is_file():
        path = audited / 'native_available.npy'
        origin = 'independent_actual_prepare_view_cpu_audit'
    else:
        path = STORE / 'cache/a/available.npy'
        origin = 'original_fullscale_prepare_cache'
    values = np.load(path, mmap_mode='r')
    if values.shape != (len(rows), 2) or not np.isin(values, (0, 1)).all():
        raise ValueError(f'Invalid native availability: {path}')
    return values.astype(bool), dict(path=str(path), source=origin,
        rule='prepare_view returns non-None and valid_target; not merely nonempty spec')


def prediction_path(arm, seed, split):
    if arm in RECOVERY_ARMS:
        return ROOT / f'{RECOVERY_ARMS[arm]}_s{seed}' / f'{split}.npz'
    if arm == 'Full-Tuned':
        return ROOT / f'selected_s{seed}' / f'{split}.npz'
    lr = '0.3' if arm == 'SE' else '1'
    name = 'readout/validation.npz' if split == 'validation' else 'evaluation.npz'
    return STORE / f'{arm.lower()}_s{seed}_lr{lr}' / name


def load_prediction(path, ids, rows, require_label=False):
    ids = np.asarray(ids, np.int64)
    with np.load(path, allow_pickle=False) as data:
        actual, score = data['ids'], data['score']
        if (actual.ndim != 1 or not np.issubdtype(actual.dtype, np.integer)
                or len(np.unique(actual)) != len(actual)
                or not np.array_equal(np.sort(actual), np.sort(ids))):
            raise ValueError(f'Wrong, duplicate or missing manifest indices: {path}')
        if score.shape != (len(ids), CLASS_COUNT) or not np.isfinite(score).all():
            raise ValueError(f'Expected finite (N,8) scores: {path}')
        if require_label and 'label' not in data:
            raise ValueError(f'Missing saved labels: {path}')
        if 'label' in data:
            truth = np.array([rows[int(i)]['label'] for i in actual])
            if not np.array_equal(data['label'], truth):
                raise ValueError(f'Prediction labels disagree with frozen manifest: {path}')
        lookup = {int(i): j for j, i in enumerate(actual)}
        return np.asarray(score[[lookup[int(i)] for i in ids]], np.float64)


def completion():
    missing, receipts = [], []
    for seed in SEEDS:
        folder = ROOT / f'selected_s{seed}'
        path = folder / 'receipt.json'
        if not path.is_file():
            missing.append(str(path))
            continue
        receipt = read_json(path)
        if (receipt.get('stop_reason') not in ('validation_patience', 'max_epochs')
                or receipt.get('epochs', 0) <= 0 or receipt.get('steps', 0) <= 0
                or receipt.get('seed') != seed):
            raise ValueError(f'Unfinished or inconsistent training receipt: {path}')
        for split in ('validation', 'evaluation'):
            if not (folder / f'{split}.npz').is_file():
                missing.append(str(folder / f'{split}.npz'))
        receipts.append(dict(seed=seed, path=str(path), receipt=receipt))
    if missing:
        raise FileNotFoundError('Training/evaluation incomplete; no waiting or analysis outputs:\n' + '\n'.join(missing))
    return receipts


def confusion(y, pred, weights=None):
    return np.bincount(CLASS_COUNT * y + pred, weights=weights,
                       minlength=CLASS_COUNT**2).reshape(CLASS_COUNT, CLASS_COUNT)


def cm_metrics(cm):
    cm = np.asarray(cm, float)
    support, predicted = cm.sum(-1), cm.sum(-2)
    correct = np.diagonal(cm, axis1=-2, axis2=-1)
    recall = np.divide(correct, support, out=np.zeros_like(correct), where=support > 0)
    f1 = np.divide(2*correct, support+predicted, out=np.zeros_like(correct), where=support+predicted > 0)
    total, present = support.sum(-1), (support > 0).sum(-1)
    oa = np.divide(correct.sum(-1), total, out=np.zeros_like(total), where=total > 0)
    ba = np.divide(recall.sum(-1), present, out=np.zeros_like(total), where=present > 0)
    return np.stack((oa, ba, f1.mean(-1)), axis=-1)*100


def probability_losses(y, z):
    centered = z-z.max(-1, keepdims=True)
    logp = centered - np.log(np.exp(centered).sum(-1, keepdims=True))
    p = np.exp(logp)
    true = np.take_along_axis(logp, np.broadcast_to(y, logp.shape[:-1])[..., None], axis=-1)[..., 0]
    target = np.eye(CLASS_COUNT)[y]
    return -true, np.square(p-target).sum(-1)


def metrics(y, score):
    if not len(y):
        raise ValueError('Empty subset')
    out = dict(zip(METRICS[:3], map(float, cm_metrics(confusion(y, score.argmax(-1))))))
    nll, brier = probability_losses(y, score)
    out.update(nll=float(nll.mean()), brier=float(brier.mean()))
    return out


def strata(rows, ids, available):
    av = available[ids]
    yield 'all', 'all', np.ones(len(ids), bool)
    for name, mask in (('both', av.all(1)), ('single', av.sum(1) == 1),
                       ('vv_only', av[:, 0] & ~av[:, 1]), ('vh_only', av[:, 1] & ~av[:, 0]),
                       ('neither', ~av.any(1))):
        if mask.any():
            yield 'availability', name, mask
    for field in ('port', 'pass_group'):
        values = np.array([rows[int(i)][field] for i in ids])
        for value in sorted(set(values)):
            yield field, str(value), values == value


def cluster_intervals(y, ships, candidate, baseline, replicates=2000, seed=20261112):
    """Resample MMSI clusters, keeping observations, methods and fixed seeds paired."""
    unique, inverse = np.unique(ships, return_inverse=True)
    nship, nseed = len(unique), candidate.shape[0]
    cm = np.zeros((nship, 2, nseed, CLASS_COUNT, CLASS_COUNT))
    losses = np.zeros((nship, 2, nseed, 2))
    for method, scores in enumerate((candidate, baseline)):
        for k in range(nseed):
            np.add.at(cm[:, method, k], (inverse, y, scores[k].argmax(-1)), 1)
            nll, brier = probability_losses(y, scores[k])
            np.add.at(losses[:, method, k, 0], inverse, nll)
            np.add.at(losses[:, method, k, 1], inverse, brier)
    rng = np.random.default_rng(seed)
    draws, valid_ba = [], []
    required = np.bincount(y, minlength=CLASS_COUNT) > 0
    size = np.bincount(inverse, minlength=nship)
    for first in range(0, replicates, 128):
        n = min(128, replicates-first)
        multiplicity = rng.multinomial(nship, np.full(nship, 1/nship), size=n)
        sampled = (multiplicity @ cm.reshape(nship, -1)).reshape(n, 2, nseed, CLASS_COUNT, CLASS_COUNT)
        totals = (multiplicity @ losses.reshape(nship, -1)).reshape(n, 2, nseed, 2)
        loss_mean = totals / (multiplicity @ size)[:, None, None, None]
        value = np.concatenate((cm_metrics(sampled), loss_mean), axis=-1)
        draws.append((value[:, 0]-value[:, 1]).mean(1))
        valid_ba.append((sampled[:, 0, 0].sum(-1)[:, required] > 0).all(1))
    draws, valid_ba = np.concatenate(draws), np.concatenate(valid_ba)
    point = np.array([[metrics(y, candidate[k])[m]-metrics(y, baseline[k])[m]
                       for m in METRICS] for k in range(nseed)]).mean(0)
    records = []
    for j, metric in enumerate(METRICS):
        valid = valid_ba if metric == 'ba' else np.ones(replicates, bool)
        lo, hi = np.quantile(draws[valid, j], [.025, .975]) if valid.any() else (None, None)
        records.append(dict(metric=metric, delta=float(point[j]),
            ci_low=float(lo) if lo is not None else None, ci_high=float(hi) if hi is not None else None,
            unit='percentage_points' if j < 3 else 'loss_units', ships=nship,
            replicates=replicates, valid_replicates=int(valid.sum()), bootstrap_seed=seed))
    return records


def outcome_masks(candidate, baseline, y):
    a, b = candidate == y, baseline == y
    return dict(rescued=a & ~b, harmed=~a & b, stable_correct=a & b, stable_wrong=~a & ~b)


def legal_bank(rows, groups, av, index):
    query = rows[index]
    result = []
    for i in groups['references']:
        ref = rows[i]
        identity = ref['mmsi'] != query['mmsi'] and orbit(ref) != orbit(query)
        shared = av[i] & av[index]
        result.append(dict(index=i, id=ref['id'], label=ref['label'], legal_identity=identity,
                           active=bool(identity and shared.any()),
                           shared_polarizations=[p for p, good in zip(('vv', 'vh'), shared) if good]))
    return result


def select_cases(states, y, ids, count=3):
    """Round-robin selection prevents a cross-seed flip case filling two panels."""
    selected, used, class_count = [], set(), Counter()
    for _ in range(count):
        for kind in KINDS:
            repeats = states[kind].sum(0)
            candidates = [j for j in np.flatnonzero(repeats) if j not in used]
            if not candidates:
                continue
            j = min(candidates, key=lambda j: (class_count[kind, int(y[j])], -int(repeats[j]), int(ids[j])))
            selected.append((kind, j))
            used.add(j)
            class_count[kind, int(y[j])] += 1
    return selected


def aggregate_seed_rows(records, keys, values):
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record[k] for k in keys)].append(record)
    result = []
    for key, items in groups.items():
        row = dict(zip(keys, key))
        row.update(seeds=len(items))
        for name in ('n', 'ships', 'classes_present', 'support', 'class_name'):
            if name in items[0]:
                row[name] = items[0][name]
        for value in values:
            x = [r[value] for r in items if r[value] is not None]
            row[value+'_mean'] = float(np.mean(x)) if x else None
            row[value+'_sd'] = float(np.std(x, ddof=1)) if len(x) > 1 else None
        result.append(row)
    return result


def optional_arms():
    complete, coverage = [], []
    for arm, prefix in RECOVERY_ARMS.items():
        files = [ROOT / f'{prefix}_s{s}' / name for s in SEEDS
                 for name in ('validation.npz', 'evaluation.npz', 'complete.json')]
        missing = [str(p) for p in files if not p.is_file()]
        coverage.append(dict(arm=arm, included=not missing, missing=missing))
        if not missing:
            for seed in SEEDS:
                receipt = read_json(ROOT / f'{prefix}_s{seed}/complete.json')
                if (receipt.get('seed') != seed or receipt.get('training') is not False
                        or receipt.get('reference_recovery') != arm.endswith('QueryRef')
                        or receipt.get('weight_source') != ('old' if arm.startswith('Old') else 'tuned')):
                    raise ValueError(f'Recovery completion receipt disagrees with arm: {arm}/{seed}')
            complete.append(arm)
    return complete, coverage


def comparisons(arms):
    result = [('Full-Tuned', 'Full', 'primary_finetuning'),
              ('Full-Tuned', 'Plain', 'historical_unmatched_recipe'),
              ('Full-Tuned', 'SE', 'historical_unmatched_recipe')]
    for prefix, original in (('Old', 'Full'), ('Tuned', 'Full-Tuned')):
        query, both = prefix+'-Query', prefix+'-QueryRef'
        if query in arms:
            result.append((query, original, 'query_recovery'))
        if both in arms:
            result.append((both, original, 'query_reference_total'))
        if query in arms and both in arms:
            result.append((both, query, 'reference_increment_given_query_recovery'))
    if 'Old-QueryRef' in arms and 'Tuned-QueryRef' in arms:
        result.append(('Tuned-QueryRef', 'Old-QueryRef', 'finetuning_under_same_recovered_inputs'))
    return result


class PanelInputs:
    def __init__(self, rows):
        self.rows = rows
        self.native = native_api()
        self.recovery = None
        self.recovered = {}
        self.prepared = {}

    def row(self, arm, index, reference=False):
        restore = arm in RECOVERY_ARMS and (not reference or arm.endswith('QueryRef'))
        if not restore:
            return self.rows[index]
        if self.recovery is None:
            path = ROOT / 'polarization/recovery.py'
            spec = importlib.util.spec_from_file_location('ports24_panel_recovery_cpu', path)
            self.recovery = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.recovery)
        if index not in self.recovered:
            self.recovered[index] = self.recovery.recover(self.rows[index])
        return self.recovered[index][0]

    def view(self, arm, index, pol, reference=False):
        key = arm, index, pol, reference
        if key not in self.prepared:
            self.prepared[key] = self.native.prepare_view(self.row(arm, index, reference).get(pol))
        return self.prepared[key]

    def bank(self, arm, index, groups):
        q = self.row(arm, index)
        qav = [self.view(arm, index, p) for p in ('vv', 'vh')]
        qav = np.array([v is not None and v['valid_target'] for v in qav])
        out = []
        for j in groups['references']:
            r = self.row(arm, j, True)
            rv = [self.view(arm, j, p, True) for p in ('vv', 'vh')]
            rav = np.array([v is not None and v['valid_target'] for v in rv])
            identity = q['mmsi'] != r['mmsi'] and orbit(q) != orbit(r)
            shared = qav & rav
            out.append(dict(index=j, id=r['id'], label=r['label'], legal_identity=identity,
                active=bool(identity and shared.any()),
                shared_polarizations=[p for p, yes in zip(('vv', 'vh'), shared) if yes]))
        return out


def draw_panel(api, index, before, after, bank_before, bank_after, label, rival, names, path, caption):
    from PIL import Image, ImageDraw
    columns = [(before+' query', before, index, False), (after+' query', after, index, False)]
    gaps = []
    for arm, bank in ((before, bank_before), (after, bank_after)):
        for role, c in (('same-class ref', label), ('confusion ref', rival)):
            match = next((r for r in bank if r['active'] and r['label'] == c), None)
            if match is None:
                gaps.append(dict(arm=arm, role=role, label=c, reason='no_active_legal_reference'))
            else:
                columns.append((arm+' '+role, arm, match['index'], True))
    canvas = Image.new('RGB', (230*len(columns), 510), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), caption, fill='black')
    provenance = []
    for j, (role, arm, idx, reference) in enumerate(columns):
        row = api.row(arm, idx, reference)
        x = 10+j*230
        draw.multiline_text((x, 30), '\n'.join(textwrap.wrap(role, 29)), fill='black', spacing=2)
        draw.text((x, 62), f"ID {row['id']}", fill='black')
        draw.multiline_text((x, 78), '\n'.join(textwrap.wrap(names[row['label']], 29)), fill='black')
        for k, pol in enumerate(('vv', 'vh')):
            top = 115+190*k
            spec = row.get(pol)
            view = api.view(arm, idx, pol, reference)
            good = view is not None and view['valid_target']
            draw.text((x, top), pol.upper()+(' active' if good else ' unavailable'), fill='black')
            item = dict(role=role, arm=arm, index=idx, id=row['id'], pol=pol, available=good,
                        image=spec['image'] if spec else None, box_origin=spec.get('box_origin') if spec else None)
            if view is not None:
                canvas.paste(Image.fromarray(view['pixels']).convert('RGB'), (x, top+16))
                offset = np.array([view['placement']['pad_left']-view['placement']['crop_x'],
                                   view['placement']['pad_top']-view['placement']['crop_y']])
                for di, det in enumerate(spec['detections']):
                    if det['confidence'] < .25 or not det.get('shape_usable', True):
                        continue
                    p = np.asarray(det['corners']) + offset + [x, top+16]
                    color = 'red' if di == spec.get('target_index') else 'orange'
                    points = [tuple(point) for point in p]
                    draw.line(points+points[:1], fill=color, width=1)
                item.update(raw_shape=list(view['raw'].shape), placement=view['placement'])
            else:
                draw.text((x, top+65), 'Not used by this model', fill='gray')
            provenance.append(item)
    draw.text((10, 490), 'Native pixels on 160px canvas; red target, orange context; references are examples, not attribution.', fill='black')
    canvas.save(path)
    return provenance, gaps


def gallery(out, rows, groups, scores, names):
    ids = np.asarray(groups['evaluation'], np.int64)
    y = np.array([rows[i]['label'] for i in ids])
    jobs = [('Full-Tuned', 'Full', 3)]
    for prefix, baseline in (('Old', 'Full'), ('Tuned', 'Full-Tuned')):
        arm = next((a for a in (prefix+'-QueryRef', prefix+'-Query') if a in scores), None)
        if arm:
            jobs.append((arm, baseline, 1))
    api = PanelInputs(rows)
    cases, coverage = [], []
    for after, before, count in jobs:
        a, b = scores[after], scores[before]
        states = outcome_masks(a.argmax(-1), b.argmax(-1), y)
        selected = select_cases(states, y, ids, count)
        counts = Counter(kind for kind, _ in selected)
        for kind in KINDS:
            coverage.append(dict(candidate=after, baseline=before, kind=kind, requested=count,
                panels=counts[kind], seed_events=int(states[kind].sum()),
                observations=int(states[kind].any(0).sum()),
                shortfall_reason='' if counts[kind] == count else 'insufficient_distinct_observations_in_this_state'))
        for kind, j in selected:
            k = int(np.flatnonzero(states[kind][:, j])[0])
            index, truth = int(ids[j]), int(y[j])
            rival = int(b[k, j].argmax()) if kind == 'rescued' else int(a[k, j].argmax())
            if rival == truth:
                competing = a[k, j].copy()
                competing[truth] = -np.inf
                rival = int(competing.argmax())
            oldbank, newbank = api.bank(before, index, groups), api.bank(after, index, groups)
            code = f'case_{len(cases)+1:02d}'
            filename = code+'.png'
            caption = f'{after} vs {before} | {kind} | seed {SEEDS[k]} | truth {truth} | before {b[k,j].argmax()} after {a[k,j].argmax()}'
            paths, gaps = draw_panel(api, index, before, after, oldbank, newbank, truth, rival,
                                    names, out/'gallery'/filename, caption)
            evidence = []
            for t, seed in enumerate(SEEDS):
                evidence.append(dict(seed=seed, state=next(kind for kind in KINDS if states[kind][t, j]),
                    before_prediction=int(b[t,j].argmax()), after_prediction=int(a[t,j].argmax()),
                    before_scores=b[t,j].tolist(), after_scores=a[t,j].tolist(),
                    before_true_margin=float(b[t,j,truth]-np.delete(b[t,j],truth).max()),
                    after_true_margin=float(a[t,j,truth]-np.delete(a[t,j],truth).max())))
            cases.append(dict(case=code, candidate=after, baseline=before, kind=kind, manifest_index=index,
                id=rows[index]['id'], mmsi=rows[index]['mmsi'], product=rows[index]['product'],
                port=rows[index]['port'], label=truth, selected_seed=SEEDS[k], file=filename,
                before_legal_bank=oldbank, after_legal_bank=newbank, missing_reference_roles=gaps,
                evidence=evidence, paths=paths, visual_review='pending_main_agent',
                recovery_audit=api.recovered.get(index, (None, None))[1]))
    write_json(out/'private/cases_local.json', cases)
    write_csv(out/'aggregate/case_coverage.csv', coverage)
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>24-port native cases</title>',
            '<style>body{font:15px/1.5 system-ui;margin:24px}section{border-top:1px solid #aaa;padding:12px 0}',
            '.panel{overflow-x:auto}img{max-width:none}</style><h1>Native cases: visual review pending</h1>',
            '<p>Local raw images. Native pixel scale, no extra contrast stretch. Before to after outcomes. ',
            'Reference eligibility excludes same MMSI and same orbit, then requires shared active polarization. ',
            'Displayed same-class/confusion references are legal examples, not the largest contributors. ',
            'Outcome-selected cases are not population proportions. Full scores, identities and paths are in ../private/cases_local.json.</p>']
    for case in cases:
        body.extend([f"<section><h2>{case['case']}: {html.escape(case['candidate'])} vs {html.escape(case['baseline'])} / {case['kind']}</h2>",
            f"<p>Selected seed {case['selected_seed']}; review pending. Missing reference roles: {len(case['missing_reference_roles'])}.</p>",
            f"<div class=panel><img src='{case['file']}' alt='{case['case']}'></div></section>"])
    for c in coverage:
        if c['panels'] < c['requested']:
            body.append(f"<p>{c['candidate']} vs {c['baseline']} {c['kind']}: {c['panels']}/{c['requested']}; {c['shortfall_reason']}.</p>")
    (out/'gallery/index.html').write_text('\n'.join(body)+'</html>', encoding='utf-8')
    return coverage, len(cases)


def self_test():
    y = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7])
    pred = np.array([0, 1, 1, 1, 3, 3, 4, 5, 6, 7])
    z = np.eye(CLASS_COUNT)[pred]*3
    m = metrics(y, z)
    assert np.isclose(m['oa'], 80) and np.isclose(m['ba'], 81.25), m
    expected_f1 = np.mean([2/3, 4/5, 0, 2/3, 1, 1, 1, 1])*100
    assert np.isclose(m['macro_f1'], expected_f1), m
    assert np.isfinite(metrics(y, z*1000)['nll'])
    candidate = np.stack([z, z, z])
    ci = cluster_intervals(y, np.array(['a','a','b','b','c','d','e','f','g','h']), candidate, candidate, 128)
    assert all(abs(r['delta']) < 1e-12 and
               (r['valid_replicates'] == 0 or max(abs(r['ci_low']), abs(r['ci_high'])) < 1e-12)
               for r in ci)
    rows = [dict(id=str(i), label=int(label)) for i, label in enumerate(y)]
    f = io.BytesIO()
    np.savez(f, ids=np.arange(len(y))[::-1], score=z[::-1], label=y[::-1])
    f.seek(0)
    assert np.array_equal(load_prediction(f, np.arange(len(y)), rows, True), z)
    bad = io.BytesIO()
    np.savez(bad, ids=np.zeros(len(y), int), score=z, label=y)
    bad.seek(0)
    try:
        load_prediction(bad, np.arange(len(y)), rows, True)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate IDs were accepted')
    truth = np.zeros(12, int)
    a, b = np.zeros((3, 12), int), np.zeros((3, 12), int)
    b[:, :3], a[:, 3:6] = 1, 1
    a[:, 9:], b[:, 9:] = 1, 2
    states = outcome_masks(a, b, truth)
    selected = select_cases(states, truth, np.arange(12))
    assert Counter(k for k, j in selected) == Counter(dict.fromkeys(KINDS, 3))
    assert len({j for k, j in selected}) == 12
    fake = [dict(id='q', mmsi='1', pass_group='S1A_100_A'),
            dict(id='same_ship', mmsi='1', pass_group='S1B_200_B', label=0),
            dict(id='same_orbit', mmsi='2', pass_group='S1A_100_C', label=0),
            dict(id='legal', mmsi='2', pass_group='S1B_200_B', label=0)]
    bank = legal_bank(fake, dict(references=[1,2,3]), np.ones((4,2), bool), 0)
    assert [r['active'] for r in bank] == [False, False, True]
    assert 'torch' not in sys.modules
    print('SELF_TEST_OK: eight-class metrics, paired MMSI CI, ID ordering/rejection, 12 distinct cases, orbit/MMSI references; no torch; no files written.')


def check_inputs():
    rows, groups, names = metadata()
    available, source = availability(rows)
    checked = 0
    for split in ('validation', 'evaluation'):
        for arm in BASELINES:
            for seed in SEEDS:
                load_prediction(prediction_path(arm, seed, split), groups[split], rows)
                checked += 1
    required = [prediction_path('Full-Tuned', s, part) for s in SEEDS for part in ('validation','evaluation')]
    print(json.dumps(dict(class_order=names, historical_prediction_files_checked=checked,
        native_availability_shape=list(available.shape), availability_source=source,
        candidate_predictions_missing=[str(p) for p in required if not p.is_file()],
        candidate_predictions_read=False, recovery_predictions_read=False,
        analysis_run=False, files_written=False), ensure_ascii=False, indent=2))


def result_report(out, summaries, paired_summary, ci_rows, names, recovery_coverage, panels):
    lines = ['# 24 港 Full-Tuned 分析', '',
        '状态：真实完整预测已统计；原始 chip 面板仍需主代理目视审核。', '',
        '主比较为 Full-Tuned 对原 Full。旧 Plain/SE 仅为历史不同训练配方的系统对照，不能作为匹配微调基线或单独模块因果对照。', '',
        '| 方法 | OA mean +/- SD | BA mean +/- SD | Macro-F1 mean +/- SD | NLL mean | Brier mean |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r['split'] == 'evaluation' and r['dimension'] == 'all':
            scores = ' | '.join(f"{r[m+'_mean']:.3f} +/- {r[m+'_sd']:.3f}" for m in METRICS[:3])
            lines.append(f"| {r['arm']} | {scores} | {r['nll_mean']:.4f} | {r['brier_mean']:.4f} |")
    lines += ['', '| Candidate - Baseline | Metric | Delta | MMSI paired 95% CI |', '|---|---|---:|---|']
    for r in ci_rows:
        if r['split'] == 'evaluation' and r['dimension'] == 'all' and r['baseline'] not in ('Plain','SE'):
            interval = 'unavailable' if r['ci_low'] is None else f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
            lines.append(f"| {r['candidate']} - {r['baseline']} | {r['metric']} | {r['delta']:+.4f} | {interval} |")
    lines += ['', '类别顺序：'+', '.join(f'{i}={name}' for i, name in enumerate(names)), '',
        'aggregate/ 包含逐种子、三种子均值/样本SD、8类逐类、逐港、逐pass_group、原始实际可用性、配对变化和MMSI置信区间。',
        'OA/BA/F1 为百分数，变化为百分点；NLL/Brier 变化为损失单位。BA 仅平均子集真值存在类，macro-F1 固定8类、零分母记0。',
        'MMSI bootstrap 重采样船只并保留其全部观测，方法及三个固定种子成对；先算各种子指标再平均，不把三个种子复制成独立船只。',
        'BA抽样若丢失原子集存在类别则排除并记录有效次数。CI针对 all/availability；逐港和逐过境提供描述性三种子统计，未为每个小组重复bootstrap。',
        'CI条件于当前场景和已训练模型，不估计新港口泛化或选点不确定性；历史测试队列不是新盲测。',
        '可用性分层固定使用原始 prepare_view 的有效输入，recovery 后不重分组，避免比较不同分母。', '',
        'Recovery纳入要求：每臂3种子的validation/evaluation/complete.json全部齐备。未完成臂跳过且写入private/sources_local.json，不推测其结果。',
        'query_recovery、reference_increment_given_query_recovery、query_reference_total 分别报告query单步、给定query恢复后的reference增量及总作用。',
        '原训练权重上的恢复与微调权重上的恢复均为冻结权重输入对照，不是恢复后重新训练。', '',
        f'生成 {panels} 张原生像素面板：gallery/index.html；private/cases_local.json保留逐种子预测、分差、合法参考及原始路径。',
        '主比较目标为3rescues/3harms/3stablecorrect/3stablewrong；若某组不足，不伪造或跨种子重复凑数，缺口见aggregate/case_coverage.csv。',
        '完整recovery臂另选4--8幅代表，优先覆盖救回/误伤；代表案例不是总体比例。',
        '参考排除同MMSI及同orbit（pass_group前两段），再要求与query共享可用极化；图中参考是合法示例，不宣称最高贡献。',
        '评分margin为各模型自身logit尺度，不能把跨模型绝对幅度变化直接当证据强度。', '',
        '**reportpending：主代理仍需查看VV/VH及合法参考，解释代表案例与未解释的失败；脚本不自动编造图像机制或判定论文增益。**',
        'private/和gallery/含私人路径、MMSI/原始图像，保留本地；aggregate/不含逐样本ID/MMSI或原始文件路径。未执行Git、训练或模型推理。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261112)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.check_inputs:
        check_inputs()
        return
    if args.bootstrap < 100:
        parser.error('--bootstrap must be >=100')
    receipts = completion()
    rows, groups, names = metadata()
    available, av_source = availability(rows)
    extra, recovery_coverage = optional_arms()
    arms, pairs = list(ARMS)+extra, comparisons(list(ARMS)+extra)
    scores, sources = {}, []
    for split in ('validation', 'evaluation'):
        scores[split] = {}
        for arm in arms:
            values = []
            for seed in SEEDS:
                path = prediction_path(arm, seed, split)
                values.append(load_prediction(path, groups[split], rows,
                    require_label=arm == 'Full-Tuned' or arm in RECOVERY_ARMS))
                sources.append(dict(arm=arm, seed=seed, split=split, path=str(path)))
            scores[split][arm] = np.stack(values)
    out = ROOT/'analysis'
    for directory in ('aggregate', 'private', 'gallery'):
        (out/directory).mkdir(parents=True, exist_ok=True)
    metric_rows, class_rows, paired, intervals, observations = [], [], [], [], []
    for split, by_arm in scores.items():
        ids = np.asarray(groups[split], np.int64)
        y = np.array([rows[i]['label'] for i in ids])
        ships = np.array([str(rows[i]['mmsi']) for i in ids])
        if any(not ship.isdigit() for ship in ships):
            raise ValueError('MMSI bootstrap requires the frozen valid identities')
        for dimension, value, mask in strata(rows, ids, available):
            truth = y[mask]
            meta = dict(split=split, dimension=dimension, group=value, n=int(mask.sum()),
                        ships=len(set(ships[mask])), classes_present=len(set(truth)))
            for arm in arms:
                for k, seed in enumerate(SEEDS):
                    z = by_arm[arm][k,mask]
                    metric_rows.append(dict(arm=arm, seed=seed, **meta, **metrics(truth,z)))
                    cm = confusion(truth, z.argmax(-1))
                    for c in range(CLASS_COUNT):
                        support, predicted, correct = int(cm[c].sum()), int(cm[:,c].sum()), int(cm[c,c])
                        class_rows.append(dict(arm=arm, seed=seed, **meta, label=c, class_name=names[c],
                            support=support, predicted=predicted, correct=correct,
                            recall=100*correct/support if support else None,
                            precision=100*correct/predicted if predicted else None,
                            f1=200*correct/(support+predicted) if support+predicted else 0.))
            for after, before, purpose in pairs:
                a, b = by_arm[after][:,mask], by_arm[before][:,mask]
                states = outcome_masks(a.argmax(-1),b.argmax(-1),truth)
                for k, seed in enumerate(SEEDS):
                    ma, mb = metrics(truth,a[k]),metrics(truth,b[k])
                    paired.append(dict(candidate=after, baseline=before, purpose=purpose, seed=seed, **meta,
                        **{m+'_delta':ma[m]-mb[m] for m in METRICS},
                        **{kind:int(states[kind][k].sum()) for kind in KINDS},
                        wrong_to_wrong_flips=int(((a[k].argmax(-1)!=b[k].argmax(-1))&states['stable_wrong'][k]).sum())))
                if dimension in ('all','availability'):
                    for ci in cluster_intervals(truth,ships[mask],a,b,args.bootstrap,args.bootstrap_seed):
                        intervals.append(dict(candidate=after, baseline=before, purpose=purpose, **meta, **ci))
                if dimension == 'all':
                    for k, seed in enumerate(SEEDS):
                        ap, bp = a[k].argmax(-1), b[k].argmax(-1)
                        for j, idx in enumerate(ids):
                            observations.append(dict(candidate=after, baseline=before, split=split, seed=seed,
                                index=int(idx), id=rows[idx]['id'], mmsi=rows[idx]['mmsi'],
                                label=int(y[j]), before_prediction=int(bp[j]), after_prediction=int(ap[j]),
                                state=next(kind for kind in KINDS if states[kind][k,j])))
        print('STATISTICS',split,'complete',flush=True)
    keys = ('arm','split','dimension','group')
    summary = aggregate_seed_rows(metric_rows,keys,METRICS)
    per_class = aggregate_seed_rows(class_rows,keys+('label',),('recall','precision','f1'))
    pair_summary = aggregate_seed_rows(paired,('candidate','baseline','purpose','split','dimension','group'),
        tuple(m+'_delta' for m in METRICS)+KINDS+('wrong_to_wrong_flips',))
    for name, records in (('metrics_by_seed.csv',metric_rows),('summary.csv',summary),
                          ('per_class_by_seed.csv',class_rows),('per_class_summary.csv',per_class),
                          ('paired_by_seed.csv',paired),('paired_summary.csv',pair_summary),
                          ('paired_mmsi_ci.csv',intervals)):
        write_csv(out/'aggregate'/name,records)
    for dimension, filename in (('port','per_port.csv'),('pass_group','per_pass.csv'),('availability','per_availability.csv')):
        write_csv(out/'aggregate'/filename,[r for r in summary if r['dimension']==dimension])
    write_csv(out/'private/paired_observations.csv',observations)
    write_json(out/'private/sources_local.json',dict(manifest=str(STORE/'manifest.json'),
        receipts=receipts,predictions=sources,availability=av_source,recovery_coverage=recovery_coverage,
        legal_reference_rule='different MMSI and different first-two-component orbit; shared active polarization'))
    coverage, panels = gallery(out,rows,groups,scores['evaluation'],names)
    result_report(out,summary,pair_summary,intervals,names,recovery_coverage,panels)
    write_json(out/'analysis_receipt.json',dict(status='statistics_complete_visual_review_reportpending',
        arms=arms, seeds=list(SEEDS), bootstrap=args.bootstrap, bootstrap_seed=args.bootstrap_seed,
        panels=panels, expected_primary_panels=12, new_training=False, new_inference=False,
        privacy='private and gallery are local only', missing_panel_groups=[r for r in coverage if r['panels']<r['requested']]))
    print('ANALYSIS_COMPLETE',out/'RESULTS_ZH.md','visual review reportpending',flush=True)


if __name__ == '__main__':
    main()
