"""Paired readout comparisons and native VV/VH cases, without fitting."""
from pathlib import Path
import importlib.util
import json
import csv
import html
import numpy as np
from train_probe import metrics

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('probe_panels',ROOT.parent/'ports24_sync_20261112/analyze.py')
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def load(arm,seed):
    with np.load(ROOT/f'{arm}_s{seed}_predictions_local.npz') as z:
        return z['score'].copy()


def margin(z,y):
    wrong=z.copy();wrong[np.arange(len(y)),y]=-np.inf
    return z[np.arange(len(y)),y]-wrong.max(1)


def counts(after,before,y,mask):
    good=after.argmax(1)==y;old=before.argmax(1)==y
    return dict(n=int(mask.sum()),rescued=int((good&~old&mask).sum()),
                harmed=int((~good&old&mask).sum()),changed=int(((after.argmax(1)!=before.argmax(1))&mask).sum()))


def main():
    rows,groups,names=a.metadata()
    with np.load(ROOT/'validation_local.npz') as d:
        ids,y,baseline,active=d['ids'],d['label'],d['logits'],d['active']
    result=a.read_json(ROOT/'results.json')
    # Fixed eight-class macro-F1 matches the inherited 24-port reporting convention.
    result['baseline']=metrics(y,baseline)
    result['baseline_dual']=metrics(y[active],baseline[active])
    for row in result['results']:
        z=load(row['arm'],row['seed'])
        row['validation']=metrics(y,z)
        row['dual']=metrics(y[active],z[active])
    result['metric_convention']='BA over present classes; macro-F1 over all eight configured classes'
    a.write_json(ROOT/'results.json',result)
    aggregate={arm:{k:dict(mean=float(np.mean(v:=[r['validation'][k] for r in result['results'] if r['arm']==arm])),
        std=float(np.std(v,ddof=1))) for k in ('oa','ba','macro_f1')}
        for arm in ('summary','detail_mean','detail_grid')}
    paired={}; scores={}
    for seed in (17,29,43):
        summary=load('summary',seed)
        for arm in ('summary','detail_mean','detail_grid'):
            z=load(arm,seed);b=baseline if arm=='summary' else summary
            key=f'{arm}_s{seed}'
            scores[key]=(z,b)
            allmask=np.ones(len(y),bool)
            c=counts(z,b,y,allmask)
            changed=np.flatnonzero((z.argmax(1)==y)!=(b.argmax(1)==y))
            c['flips']=[dict(id=rows[ids[i]]['id'],class_name=names[y[i]],port=rows[ids[i]]['port'],
                kind='rescued' if z[i].argmax()==y[i] else 'harmed',
                before_margin=float(margin(b,y)[i]),after_margin=float(margin(z,y)[i])) for i in changed]
            c['classes']={names[k]:counts(z,b,y,y==k) for k in range(8)}
            c['ports']={p:counts(z,b,y,np.array([rows[i]['port']==p for i in ids]))
                for p in sorted({rows[i]['port'] for i in ids})}
            paired[key]=c
    a.write_json(ROOT/'paired.json',dict(aggregate=aggregate,comparisons=paired))
    table=[dict(arm=r['arm'],seed=r['seed'],epoch=r['best']['epoch'],
                weight_decay=r['weight_decay'],**r['validation']) for r in result['results']]
    a.write_csv(ROOT/'metrics.csv',table)
    targets=[('detail_mean_s29','29641'),('detail_mean_s17','13481'),
             ('detail_grid_s17','5915'),('detail_grid_s17','35709'),
             ('detail_grid_s29','9574'),('detail_grid_s29','7672')]
    z,b=scores['detail_grid_s29']
    for mask in ((z.argmax(1)==y)&(b.argmax(1)==y),(z.argmax(1)!=y)&(b.argmax(1)!=y)):
        ix=np.flatnonzero(mask)
        j=ix[np.argmax(abs(margin(z,y)[ix]-margin(b,y)[ix]))]
        targets.append(('detail_grid_s29',str(rows[ids[j]]['id'])))
    gallery=ROOT/'gallery';gallery.mkdir(exist_ok=True)
    api=a.PanelInputs(rows);arm='PCEA-QueryRef';a.RECOVERY_ARMS[arm]=''
    cases=[]
    for comparison,shipid in targets:
        j=next(j for j,i in enumerate(ids) if str(rows[i]['id'])==shipid)
        idx,truth=int(ids[j]),int(y[j]);z,b=scores[comparison]
        rival=b[j].copy();rival[truth]=-np.inf;rival=int(rival.argmax())
        kind=('correct' if z[j].argmax()==truth else 'wrong') if (z[j].argmax()==truth)==(b[j].argmax()==truth) else ('rescued' if z[j].argmax()==truth else 'harmed')
        bank=api.bank(arm,idx,groups)
        fname=f'{len(cases):02d}_{kind}_{shipid}.png'
        title=f'{comparison} vs summary | {kind} | class {truth}; {b[j].argmax()} -> {z[j].argmax()}'
        paths,gaps=a.draw_panel(api,idx,arm,arm,bank,bank,truth,rival,names,gallery/fname,title)
        cases.append(dict(comparison=comparison,id=shipid,kind=kind,file=fname,
            label=truth,class_name=names[truth],port=rows[idx]['port'],
            before_margin=float(margin(b,y)[j]),after_margin=float(margin(z,y)[j]),
            paths=paths,reference_gaps=gaps))
    a.write_json(gallery/'cases_local.json',cases)
    body=['<!doctype html><meta charset="utf-8"><title>PCEA readout comparison</title>',
          '<h1>PCEA readout comparison, fixed checkpoint</h1>',
          '<p>Same-capacity source-trained readouts. Native-scale VV/VH; source references are examples, not attribution.</p>']
    for c in cases:
        body.append(f'<h2>{html.escape(c["comparison"])} / {c["kind"]}: {c["id"]}</h2>'
            f'<p>{c["port"]}, {c["class_name"]}; margin {c["before_margin"]:.4f} to {c["after_margin"]:.4f}</p>'
            f'<img src="{c["file"]}" style="max-width:100%">')
    (gallery/'index.html').write_text('\n'.join(body),encoding='utf-8')
    print(json.dumps(dict(aggregate=aggregate,cases=[{k:v for k,v in c.items() if k not in ('paths','reference_gaps')} for c in cases]),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
