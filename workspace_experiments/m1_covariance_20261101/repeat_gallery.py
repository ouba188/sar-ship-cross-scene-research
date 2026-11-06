"""Native examples for frozen-choice repeats; no fitting or selection changes."""
import html
from review import gallery, ROOT, task, np, json

rows, train, ids, rid = task.records()
y = np.array([rows[i]['label'] for i in ids])
a = task.arrays()
selection = json.loads((ROOT/'selection.json').read_text())
gallery.ROOT = ROOT
cases = []
for seed in (29, 43):
    scores = {}
    for kind, sel in selection.items():
        name = f'{sel["position"]}_{kind}_g{int(sel["gain"]*10)}_s{seed}'
        scores[kind] = np.load(ROOT/name/'selected.npz')['score']
        if kind == 'class':
            for mode in ('off', 'relation_off', 'covariance_zero', 'vv_only', 'vh_only'):
                scores[mode] = np.load(ROOT/name/(mode+'.npz'))['score']
    scores['baseline'] = scores['mean']
    good, old = scores['class'].argmax(1) == y, scores['mean'].argmax(1) == y
    for label, mask in (('rescue', good & ~old), ('harm', ~good & old)):
        for position in np.flatnonzero(mask):
            cases.append(gallery.draw_case(rows, a, rid, int(ids[position]), f'seed{seed}_{label}',
                                          scores, y, int(position), focus='class'))
task.write(ROOT/'repeat_cases.json', cases)
body = ['<html><meta charset="utf-8"><title>M1 frozen-choice repeats</title>',
        '<style>body{font-family:Arial;margin:24px}img{max-width:100%;height:auto}</style>',
        '<h1>Frozen-choice repeats: native VV/VH and fixed source references</h1>',
        '<p>Every rescue and harm relative to the independently selected mean configuration, seeds29/43. Baseline in this page means the trained mean control, not the historical carrier.</p>']
for case in cases:
    body.extend(['<h2>'+html.escape(case['group']+' '+case['id'])+'</h2>', '<img src="'+case['image']+'">',
                 '<pre>'+html.escape(json.dumps(case['evidence'], indent=2))+'</pre>'])
(ROOT/'repeat_gallery.html').write_text('\n'.join(body+['</html>']), encoding='utf-8')
print([(c['group'], c['id'], c['image']) for c in cases])
