"""Paired predictions, internal contribution diagnostics and native chip gallery."""
import html
import importlib.util
from run import *

spec = importlib.util.spec_from_file_location('oss2_m1_covariance_gallery', task.ROOT/'review_task.py')
gallery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)


def comparison(name, control, z, b, y, scope):
    good, old = z.argmax(1) == y, b.argmax(1) == y
    mz, mb = task.metrics(y, z), task.metrics(y, b)
    return dict(candidate=name, control=control, scope=scope, rescues=int((good & ~old).sum()),
                harms=int((~good & old).sum()), changes=int((z.argmax(1) != b.argmax(1)).sum()),
                delta_oa=mz['oa']-mb['oa'], delta_ba=mz['ba']-mb['ba'], delta_nll=mz['nll']-mb['nll'])


def main():
    rows, train, ids, rid = task.records()
    y = np.array([rows[i]['label'] for i in ids])
    entries = {}
    for path in ROOT.glob('*/receipt.json'):
        r = json.loads(path.read_text())
        with np.load(path.parent/'selected.npz') as f:
            assert np.array_equal(f['ids'], ids)
            entries[path.parent.name] = (r, f['score'].copy())
    pairs, counts = [], []
    for name, (r, z) in entries.items():
        for cls in range(4):
            m = y == cls
            counts.append(dict(run=name, label=cls, n=int(m.sum()), correct=int(((z.argmax(1) == y) & m).sum())))
        for control in ('mean', 'second'):
            other = f'{r["position"]}_{control}_g{int(r["gain"]*10)}_s{r["seed"]}'
            if r['kind'] == 'class' and other in entries:
                pairs.append(comparison(name, other, z, entries[other][1], y, 'matched_selected'))
                last = np.load(ROOT/name/'epoch6.npz')['score']
                previous = np.load(ROOT/other/'epoch6.npz')['score']
                pairs.append(comparison(name, other, last, previous, y, 'matched_last'))
        for mode in ('off', 'relation_off', 'merged', 'spatial_mean'):
            pairs.append(comparison(name, mode, z, np.load(ROOT/name/(mode+'.npz'))['score'], y, 'same_weights'))
    best = {}
    for kind in ('mean', 'second', 'class'):
        pool = [(name, r, z) for name, (r, z) in entries.items() if r['kind'] == kind and r['seed'] == 17]
        if not pool:
            continue
        name, r, z = max(pool, key=lambda item: prior.rank(item[1]['selected']))
        best[kind] = dict(run=name, position=r['position'], gain=r['gain'], epoch=r['selected']['epoch'],
                          metrics=task.metrics(y, z))
    if len(best) < 3:
        return
    name = best['class']['run']
    z = entries[name][1]
    scores = {kind: entries[item['run']][1] for kind, item in best.items()}
    scores['baseline'] = np.load(prior.OLD/'anchor.npz')['score']
    old_mean = json.loads((ROOT.parent/'m1_tuning_position_20261101/selection.json').read_text())['mean_gate']['run']
    scores['previous_best_mean'] = np.load(ROOT.parent/'m1_tuning_position_20261101'/old_mean/'selected.npz')['score']
    for kind in ('mean', 'second'):
        pairs.append(comparison(name, best[kind]['run'], z, scores[kind], y, 'independently_tuned_best'))
    for kind in ('baseline', 'previous_best_mean'):
        pairs.append(comparison(name, kind, z, scores[kind], y, 'historical_reference'))
    for mode in ('off', 'relation_off', 'merged', 'spatial_mean', 'vv_only', 'vh_only', 'covariance_zero'):
        if not (ROOT/name/(mode+'.npz')).exists():
            continue
        scores[mode] = np.load(ROOT/name/(mode+'.npz'))['score']
    pd.DataFrame(pairs).to_csv(ROOT/'pairs.csv', index=False)
    pd.DataFrame(counts).to_csv(ROOT/'class_counts.csv', index=False)
    task.write(ROOT/'selection.json', best)
    orders = [json.loads((ROOT/n/'orders.json').read_text()) for n, (r, _) in entries.items() if r['seed'] == 17]
    assert all(o == orders[0] for o in orders)
    good = z.argmax(1) == y
    mean = scores['mean'].argmax(1) == y
    second = scores['second'].argmax(1) == y
    original = scores['baseline'].argmax(1) == y
    relation_off = scores['relation_off'].argmax(1) == y
    groups = [('rescue_mean', good & ~mean), ('harm_mean', ~good & mean),
              ('relation_rescue', good & ~relation_off), ('relation_harm', ~good & relation_off),
              ('rescue_second', good & ~second), ('harm_second', ~good & second),
              ('rescue_anchor', good & ~original), ('harm_anchor', ~good & original),
              ('stable_correct', good & mean & second), ('stable_wrong', ~good & ~mean & ~second)]
    gap = abs(gallery.margins(z, y)-gallery.margins(scores['mean'], y))
    seen, cases, status = set(), [], []
    gallery.ROOT = ROOT
    a = task.arrays()
    for label, mask in groups:
        eligible = sorted(np.flatnonzero(mask), key=lambda i: -gap[i])
        pos = next((int(i) for i in eligible if int(i) not in seen), None)
        status.append(dict(group=label, n=len(eligible), selected=pos))
        if pos is not None:
            seen.add(pos)
            cases.append(gallery.draw_case(rows, a, rid, int(ids[pos]), label, scores, y, pos, focus='class'))
    task.write(ROOT/'cases.json', dict(selection=best, groups=status, cases=cases, orders_equal=True))
    event = []
    for i, idx in enumerate(ids):
        rec = dict(index=int(idx), id=rows[idx]['id'], mmsi=rows[idx]['mmsi'], label=int(y[i]),
                   pass_group=rows[idx]['pass_group'], available=np.asarray(a['available'][idx]).tolist())
        for kind, s in scores.items():
            rec[kind+'_prediction'] = int(s[i].argmax())
            rec[kind+'_margin'] = float(gallery.margins(s[i:i+1], y[i:i+1])[0])
        event.append(rec)
    pd.DataFrame(event).to_csv(ROOT/'events.csv', index=False)
    body = ['<html><meta charset="utf-8"><title>M1 class-competitive covariance pilot</title>',
            '<style>body{font-family:Arial;margin:24px}img{max-width:100%;height:auto}td,th{padding:5px;border:1px solid #bbb}table{border-collapse:collapse}</style>',
            '<h1>M1: mean vs low-rank second-order vs class-competitive relation</h1>',
            '<p>OSS2 four-class source719 / historical development228 / fixed32 source references. Development-selected results, not independent confirmation. M1 + position-specific conv3 + head trained; M2/M3 frozen. VV/VH separate. No explicit frequency.</p>',
            '<p><a href="FINDINGS_ZH.md">Findings and mechanism boundaries</a> | <a href="repeat_gallery.html">Seeds29/43 native rescue/harm gallery</a></p>',
            '<pre>'+html.escape(json.dumps(best, indent=2))+'</pre>',
            pd.read_csv(ROOT/'summary.csv').to_html(index=False),
            pd.DataFrame(pairs).query('scope != "same_weights"').to_html(index=False)]
    if (ROOT/'repeats.csv').exists():
        body += ['<h2>Frozen-choice training repeats</h2>',
                 pd.read_csv(ROOT/'repeats.csv').to_html(index=False),
                 '<p>The class candidate beats the mean on OA/BA in all three seeds. However, covariance-zero and merged-class inference diagnostics have zero top-1 changes in all three: package-level gain does not establish unique covariance necessity.</p>',
                 pd.read_csv(ROOT/'repeat_pairs.csv').to_html(index=False)]
    for case in cases:
        body.extend(['<h2>'+html.escape(case['group']+' '+case['id'])+'</h2>', '<img src="'+case['image']+'">',
                     '<pre>'+html.escape(json.dumps(case['evidence'], indent=2))+'</pre>'])
    (ROOT/'index.html').write_text('\n'.join(body+['</html>']), encoding='utf-8')
    print(json.dumps(best, indent=2), flush=True)
    print(pd.DataFrame(pairs).query('scope in ["independently_tuned_best", "historical_reference"]').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
