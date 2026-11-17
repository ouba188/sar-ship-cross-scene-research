"""Paired validation results and native query/reference panels, no retraining."""
from pathlib import Path
import argparse
import csv
import html
import importlib.util
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ports24_sp_panels',ROOT.parent/'ports24_sync_20261112/analyze.py')
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
read=a.read_json
write=a.write_json


def prediction(path):
    with np.load(path) as z:
        return {k:z[k].copy() for k in ('ids','label','score')}


def margin(z,y):
    wrong=z.copy()
    wrong[np.arange(len(y)),y]=-np.inf
    return z[np.arange(len(y)),y]-wrong.max(1)


def paired(z,b,y,rows,ids):
    rescue=(z.argmax(1)==y)&(b.argmax(1)!=y)
    harm=(z.argmax(1)!=y)&(b.argmax(1)==y)
    def counts(mask):
        return dict(n=int(mask.sum()),rescued=int((rescue&mask).sum()),harmed=int((harm&mask).sum()),
            net=int((rescue&mask).sum()-(harm&mask).sum()))
    available=np.load(ROOT.parent/'ports24_sync_20261112/native_cache/available.npy',mmap_mode='r')[ids].copy()
    restored=read(ROOT.parent/'ports24_recovered_training_20261113/recovery_audit_local.json')
    available[[str(int(i)) in restored for i in ids]]=1
    dual=available.astype(bool).all(1)
    return dict(changed=int((z.argmax(1)!=b.argmax(1)).sum()),
        overall=counts(np.ones(len(y),bool)),dual=counts(dual),single=counts(~dual),
        classes={str(k):counts(y==k) for k in range(8)},
        ports={p:counts(np.array([rows[i]['port']==p for i in ids])) for p in sorted({rows[i]['port'] for i in ids})})


def panels(rows,groups,names,ids,y,comparisons,folder):
    folder.mkdir(parents=True,exist_ok=True)
    api=a.PanelInputs(rows)
    arm='PCEA-QueryRef'
    a.RECOVERY_ARMS[arm]=''
    cases=[]
    for comparison,z,b in comparisons:
        right=z.argmax(1)==y
        old=b.argmax(1)==y
        states=dict(rescued=right&~old,harmed=~right&old,correct=right&old,wrong=~right&~old)
        for kind,mask in states.items():
            options=np.flatnonzero(mask)
            if not len(options):
                continue
            change=margin(z,y)-margin(b,y)
            j=int(options[np.argmax(abs(change[options]))])
            idx,truth=int(ids[j]),int(y[j])
            competing=b[j].copy()
            competing[truth]=-np.inf
            rival=int(competing.argmax())
            bank=api.bank(arm,idx,groups)
            fname=f'{len(cases):02d}_{kind}_{rows[idx]["id"]}.png'
            caption=f'{comparison} | {kind} | truth {truth}; {b[j].argmax()} -> {z[j].argmax()}'
            paths,gaps=a.draw_panel(api,idx,arm,arm,bank,bank,truth,rival,names,folder/fname,caption)
            cases.append(dict(comparison=comparison,kind=kind,id=rows[idx]['id'],index=idx,
                port=rows[idx]['port'],label=truth,name=names[truth],file=fname,
                before=b[j].tolist(),after=z[j].tolist(),before_margin=float(margin(b,y)[j]),
                after_margin=float(margin(z,y)[j]),paths=paths,reference_gaps=gaps))
    write(folder/'cases_local.json',cases)
    body=['<!doctype html><meta charset="utf-8"><title>24-port PCEA native cases</title>',
        '<h1>24-port paired validation cases, seed17</h1>',
        '<p>Native160, VV/VH separate. Identical recovered inputs; fixed legal source references are examples, not attribution.</p>']
    for c in cases:
        body.append(f'<h2>{html.escape(c["comparison"])} / {c["kind"]}: {c["id"]}</h2>'
            f'<p>{html.escape(c["name"])}; {c["port"]}; margin {c["before_margin"]:.4f} to {c["after_margin"]:.4f}</p>'
            f'<img src="{c["file"]}" style="max-width:100%">')
    (folder/'index.html').write_text('\n'.join(body),encoding='utf-8')
    print(json.dumps([{k:v for k,v in c.items() if k not in ('paths','before','after')} for c in cases],ensure_ascii=False))


def main(preview=False):
    rows,groups,names=a.metadata()
    initial=prediction(ROOT.parent/'ports24_recovered_training_20261113/selected_s17/validation.npz')
    ids,y,b=initial['ids'],initial['label'],initial['score']
    if preview:
        panels(rows,groups,names,ids,y,[('initial Full',b,b)],ROOT/'input_gallery')
        return
    table=[]
    scores={}
    final_scores={}
    comparisons=[]
    report={}
    for strength in (0.,.1):
        for arm in ('full','cross_mean'):
            key=f'{arm}_s17_sp{strength:g}'
            dest=ROOT/'fits'/key
            receipt=read(dest/'receipt.json')
            z=prediction(dest/'validation.npz')
            assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['label'])
            scores[key]=z['score']
            final_scores[key]=prediction(dest/'lastvalidation.npz')['score']
            table.append(dict(arm=arm,strength=strength,step=receipt['best']['step'],
                **{k:receipt['best']['metrics'][k] for k in ('oa','ba','macro_f1','nll')},
                last_oa=receipt['last']['oa'],last_ba=receipt['last']['ba'],
                backbone_drift=receipt['final_drift']['backbone']['l2_distance']))
            report[key+'_vs_initial']=paired(z['score'],b,y,rows,ids)
            if arm=='cross_mean':
                if (dest/'validation_off.npz').exists():
                    off=prediction(dest/'validation_off.npz')['score']
                    report[key+'_best_on_off']=paired(z['score'],off,y,rows,ids)
                    report[key+'_best_on_off']['max_logit_difference']=float(abs(z['score']-off).max())
                    comparisons.append((key+' best on/off',z['score'],off))
                on=prediction(dest/'lastvalidation.npz')['score']
                off=prediction(dest/'lastvalidation_off.npz')['score']
                report[key+'_last_on_off']=paired(on,off,y,rows,ids)
                report[key+'_last_on_off']['max_logit_difference']=float(abs(on-off).max())
                comparisons.append((key+' last on/off',on,off))
                report[key+'_last_vs_initial']=paired(on,b,y,rows,ids)
                comparisons.append((key+' last vs initial',on,b))
    for strength in (0.,.1):
        z=scores[f'cross_mean_s17_sp{strength:g}']
        f=scores[f'full_s17_sp{strength:g}']
        name=f'PCEA vs Full sp{strength:g}'
        report[name]=paired(z,f,y,rows,ids)
        comparisons.append((name,z,f))
        report[name+'_final']=paired(final_scores[f'cross_mean_s17_sp{strength:g}'],
            final_scores[f'full_s17_sp{strength:g}'],y,rows,ids)
    with (ROOT/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=table[0])
        writer.writeheader();writer.writerows(table)
    common=next(iter(final_scores.values())).argmax(1)==y
    write(ROOT/'paired.json',dict(table=table,comparisons=report,
        all_final_correctness_identical=all(np.array_equal(z.argmax(1)==y,common) for z in final_scores.values())))
    panels(rows,groups,names,ids,y,comparisons,ROOT/'gallery')
    print(json.dumps(table,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview',action='store_true')
    main(parser.parse_args().preview)
