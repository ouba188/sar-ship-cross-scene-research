"""Fixed final-epoch diagnosis, distinct from the predeclared selected checkpoint."""
import pandas as pd
import review_task as gallery
from model_task import *


def main():
    current.setup(DEVICE)
    rows, train, hold, rid = records()
    a = arrays()
    y = np.array([rows[i]['label'] for i in hold])
    out = ROOT / 'last_epoch'
    out.mkdir(exist_ok=True)
    gallery.ROOT = out
    scores = {'baseline': np.load(ROOT/'baseline.npz')['score'][hold]}
    summary = []
    for arm in ('none', 'original', 'aggregate', 'competitive'):
        model = Model(arm)
        state = torch.load(ROOT/(arm+'_17')/'last.pt', map_location='cpu', weights_only=True)
        model.restore(state['trainable_state'])
        modes = [(arm, None)]
        if arm != 'none':
            modes.append((arm+'_off', 'off'))
        if arm == 'competitive':
            modes.append(('competitive_pool', 'aggregate'))
        for name, mode in modes:
            p = predict(model, a, rows, hold, rid, override=mode)
            np.savez_compressed(out/(name+'.npz'), **p)
            scores[name] = p['score']
            summary.append(dict(arm=name, **metrics(y, p['score']), mean_update_ratio=float(p['update_ratio'].mean())))
        del model
        print('LAST_DONE', arm, flush=True)
    pd.DataFrame(summary).to_csv(out/'summary.csv', index=False)
    pairs = []
    comparisons = [(k,'baseline') for k in ('none','original','aggregate','competitive')]
    comparisons += [('aggregate','none'),('competitive','none'),('competitive','aggregate'),('competitive','competitive_pool')]
    comparisons += [(k,k+'_off') for k in ('original','aggregate','competitive')]
    for new, old in comparisons:
        good = scores[new].argmax(1)==y
        was = scores[old].argmax(1)==y
        mn, mo = metrics(y,scores[new]), metrics(y,scores[old])
        pairs.append(dict(candidate=new, control=old, rescues=int((good&~was).sum()),
            harms=int((~good&was).sum()), changes=int((scores[new].argmax(1)!=scores[old].argmax(1)).sum()),
            delta_oa=mn['oa']-mo['oa'], delta_ba=mn['ba']-mo['ba']))
    pd.DataFrame(pairs).to_csv(out/'pairs.csv',index=False)
    sp, ag, base = [scores[k].argmax(1)==y for k in ('competitive','aggregate','baseline')]
    groups = [('matching_rescue',sp&~ag),('matching_harm',~sp&ag),('base_rescue',sp&~base),
              ('base_harm',~sp&base),('stable_correct',sp&ag&base),('stable_wrong',~sp&~ag&~base)]
    rank = abs(gallery.margins(scores['competitive'],y)-gallery.margins(scores['aggregate'],y))
    used, cases, status = set(), [], []
    for name, mask in groups:
        pool = sorted(np.flatnonzero(mask),key=lambda i:-rank[i])
        i = next((int(i) for i in pool if i not in used),None)
        status.append(dict(group=name,n=len(pool),selected=i))
        if i is not None:
            used.add(i)
            cases.append(gallery.draw_case(rows,a,rid,int(hold[i]),name,scores,y,i))
    write(out/'cases.json',dict(cases=cases,groups=status,epoch=4,steps=120,
        scope='Posthoc fixed-last-epoch diagnosis, NOT replacement for BA-selected checkpoint'))
    body = ['<html><meta charset="utf-8"><title>Fixed final epoch</title><style>body{font-family:Arial;margin:24px}img{max-width:100%}</style>',
            '<h1>Fixed epoch 4: diagnostic only</h1><p>All arms: 120 steps. Does not replace the selected-checkpoint report.</p>',
            pd.DataFrame(summary).to_html(index=False),pd.DataFrame(pairs).to_html(index=False)]
    for c in cases:
        body.append('<h2>'+c['group']+' '+c['id']+'</h2><img src="'+c['image']+'">')
    (out/'index.html').write_text('\n'.join(body)+ '</html>',encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__=='__main__':
    main()
