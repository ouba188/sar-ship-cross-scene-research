"""Paired aggregate results and native chip examples; no parameter selection."""
from pathlib import Path
import csv
import html
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('class_local_analysis_previous', ROOT.parent/'pcea_ports24_startpoint_20261117/analyze.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
read, write, margin = old.read, old.write, old.margin
old.ROOT = ROOT


def prediction(arm, seed, mode='on'):
    return old.prediction(ROOT/'fits'/f'{arm}_s{seed}'/f'validation_{mode}.npz')


def main():
    rows, groups, names = old.a.metadata()
    results = read(ROOT/'results.json')
    assert len(results) == 9
    sampled = {}
    for seed in (17, 29, 43):
        ids_train = np.load(ROOT/'fits'/f'full_s{seed}/training_ids_local.npy')
        assert len(set(ids_train.tolist())) == len(ids_train)
        for arm in ('generic', 'class_detail'):
            assert np.array_equal(ids_train, np.load(ROOT/'fits'/f'{arm}_s{seed}/training_ids_local.npy'))
        sampled[str(seed)] = dict(n=len(ids_train), classes=np.bincount([rows[i]['label'] for i in ids_train], minlength=8).tolist(),
                                 vessels=len({rows[i]['mmsi'] for i in ids_train}), ports=len({rows[i]['port'] for i in ids_train}),
                                 passes=len({'_'.join(rows[i]['pass_group'].split('_')[:2]) for i in ids_train}))
    write(ROOT/'actual_training_scope.json', sampled)
    base = old.prediction(ROOT.parent/'ports24_recovered_training_20261113/selected_s17/validation.npz')
    ids, y, initial = base['ids'], base['label'], base['score']
    summaries, table, comparisons = {}, [], []
    for seed in (17, 29, 43):
        f = prediction('full', seed)['score']
        g = prediction('generic', seed)['score']
        c = prediction('class_detail', seed)['score']
        u = prediction('class_detail', seed, 'uniform')['score']
        p = prediction('class_detail', seed, 'permuted')['score']
        off = prediction('class_detail', seed, 'off')['score']
        tests = [('class_vs_generic', c, g), ('class_vs_full', c, f),
                 ('generic_vs_full', g, f), ('class_vs_uniform', c, u),
                 ('class_vs_permuted', c, p), ('class_vs_off', c, off),
                 ('class_vs_initial', c, initial), ('full_vs_initial', f, initial)]
        for name, z, b in tests:
            item = old.paired(z, b, y, rows, ids)
            delta = z-b
            item.update(max_logit_difference=float(abs(delta).max()),
                        centered_logit_rms=float(np.mean((delta-delta.mean(1, keepdims=True))**2)**.5),
                        mean_margin_difference=float(np.mean(margin(z, y)-margin(b, y))))
            summaries[f'{name}_s{seed}'] = item
            if name in ('class_vs_generic', 'class_vs_uniform', 'class_vs_initial'):
                comparisons.append((f'{name}_s{seed}', z, b))
    for r in results:
        table.append(dict(arm=r['arm'], seed=r['seed'], selected_step=r['best']['step'],
                          inner_oa=r['best']['metrics']['oa'], inner_ba=r['best']['metrics']['ba'],
                          **{k:r['on'][k] for k in ('oa', 'ba', 'macro_f1', 'nll')}))
    means = {}
    for arm in ('full', 'generic', 'class_detail'):
        a = [x for x in table if x['arm'] == arm]
        means[arm] = {k:dict(mean=float(np.mean([x[k] for x in a])), std=float(np.std([x[k] for x in a], ddof=1)))
                      for k in ('oa', 'ba', 'macro_f1', 'nll')}
    write(ROOT/'paired.json', dict(means=means, comparisons=summaries, table=table))
    with (ROOT/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=table[0])
        writer.writeheader()
        writer.writerows(table)
    folder = ROOT/'gallery'
    folder.mkdir(exist_ok=True)
    api = old.a.PanelInputs(rows)
    panel_arm = 'PCEA-ClassDetail'
    old.a.RECOVERY_ARMS[panel_arm] = ''
    candidates, seen = [], set()
    for name, z, b in comparisons:
        right, was = z.argmax(1) == y, b.argmax(1) == y
        states = dict(rescued=right & ~was, harmed=~right & was,
                      correct=right & was, wrong=~right & ~was)
        for kind, mask in states.items():
            if kind in ('correct', 'wrong') and not name.endswith('s17'):
                continue
            indexes = np.flatnonzero(mask)
            if not len(indexes):
                continue
            delta = margin(z, y)-margin(b, y)
            j = int(indexes[np.argmax(abs(delta[indexes]))])
            if (int(ids[j]), kind) in seen:
                continue
            seen.add((int(ids[j]), kind))
            candidates.append((name, kind, j, z, b))
    # Retain the two previously motivating cases, without interpreting them as a population estimate.
    for wanted in ('9574', '7672'):
        match = [j for j, idx in enumerate(ids) if str(rows[idx]['id']) == wanted]
        if match:
            candidates.append(('class_vs_generic_s17', 'prior_case', match[0],
                               prediction('class_detail', 17)['score'], prediction('generic', 17)['score']))
    for seed in (17, 29, 43):
        on = prediction('class_detail', seed)['score']
        off = prediction('class_detail', seed, 'off')['score']
        for j in np.flatnonzero(on.argmax(1) != off.argmax(1)):
            kind = 'branch_rescued' if on[j].argmax() == y[j] else 'branch_changed'
            candidates.append((f'class_vs_off_s{seed}', kind, int(j), on, off))
    cases = []
    for name, kind, j, z, b in candidates:
        idx, truth = int(ids[j]), int(y[j])
        rival_score = b[j].copy()
        rival_score[truth] = -np.inf
        rival = int(rival_score.argmax())
        bank = api.bank(panel_arm, idx, groups)
        fname = f'{len(cases):02d}_{kind}_{rows[idx]["id"]}.png'
        caption = f'{name} | {kind} | truth {truth}; {b[j].argmax()} -> {z[j].argmax()}'
        paths, gaps = old.a.draw_panel(api, idx, panel_arm, panel_arm, bank, bank, truth, rival, names, folder/fname, caption)
        cases.append(dict(comparison=name, kind=kind, id=rows[idx]['id'], index=idx, port=rows[idx]['port'],
                          label=truth, name=names[truth], file=fname, paths=paths, reference_gaps=gaps,
                          before=b[j].tolist(), after=z[j].tolist(),
                          before_margin=float(margin(b, y)[j]), after_margin=float(margin(z, y)[j])))
    write(folder/'cases_local.json', cases)
    body = ['<!doctype html><meta charset="utf-8"><title>Class-local paired chip review</title>',
            '<h1>Class-response local detail: paired native cases</h1>',
            '<p>VV/VH shown separately at native160 crop/pad. Fixed legal same/rival-class references are examples, not maximal-influence attribution. Input pixels are unchanged.</p>']
    for c in cases:
        body.append(f'<h2>{html.escape(c["comparison"])} / {c["kind"]}: {c["id"]}</h2>'
                    f'<p>{html.escape(c["name"])}; {c["port"]}; margin {c["before_margin"]:.5f} to {c["after_margin"]:.5f}</p>'
                    f'<img src="{c["file"]}" style="max-width:100%">')
    (folder/'index.html').write_text('\n'.join(body), encoding='utf-8')
    print(json.dumps(dict(means=means, cases=[{k:v for k,v in c.items() if k not in ('paths','before','after')} for c in cases]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
