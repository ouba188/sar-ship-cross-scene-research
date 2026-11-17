"""Analyze cached strength contrasts and render actual native chip examples."""
from pathlib import Path
import csv
import html
import importlib.util
import json
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'pcea_class_local_20261117'
EXTRA = ROOT.parent/'pcea_class_local_expansion_20261118'
STORE = Path('C:/Users/25791/AppData/Local/sar_ports24_fullscale_20261110')
NAMES = ['Bulk','Fishing','General cargo','Chemical/product','Container','Crude','Tug','Offshore']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def prediction(path):
    with np.load(path) as item:
        return {k:item[k] for k in ('ids','label','score')}


def margin(z,y):
    other=z.copy()
    other[np.arange(len(y)),y]=-np.inf
    return z[np.arange(len(y)),y]-other.max(1)


def paired(a,b,rows):
    assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['label'],b['label'])
    z,old,y=a['score'],b['score'],a['label']
    right,was=z.argmax(1)==y,old.argmax(1)==y
    net=right.astype(int)-was.astype(int)
    delta=margin(z,y)-margin(old,y)
    ports=np.array([rows[i]['port'] for i in a['ids']])
    return dict(n=len(y),rescued=int((right&~was).sum()),harmed=int((~right&was).sum()),
                net=int(net.sum()),changed=int((z.argmax(1)!=old.argmax(1)).sum()),
                delta_oa=float(net.mean()*100),
                delta_ba=float(np.mean([net[y==k].mean() for k in np.unique(y)])*100),
                mean_margin_change=float(delta.mean()),
                class_net=[int(net[y==k].sum()) for k in range(8)],
                ports={p:dict(n=int((ports==p).sum()),net=int(net[ports==p].sum())) for p in np.unique(ports)})


def main():
    source=read(ROOT/'source_metrics.json')
    external=read(ROOT/'external_metrics.json')
    selection=read(ROOT/'selection.json')
    selected=selection['scale']
    original=read(STORE/'manifest.json')
    groups=read(STORE/'groups.json')
    extra=read(EXTRA/'evaluation_manifest_local.json')
    comparisons={}
    candidates=[]
    for seed in (17,29,43):
        one=prediction(ROOT/f'inner_s{seed}_a1_local.npz')
        off=prediction(ROOT/f'inner_s{seed}_a0_local.npz')
        for scale in (0,1,2,4,8):
            item=prediction(ROOT/f'inner_s{seed}_a{scale}_local.npz')
            key=f'inner_s{seed}_a{scale}_vs_a1'
            comparisons[key]=paired(item,one,original)
            comparisons[f'inner_s{seed}_a{scale}_vs_off']=paired(item,off,original)
            if scale in {selected,8} and scale!=1:
                candidates.append((key,item,one,original,'original'))
        for split,rows in [('validation160',original),('expansion198',extra)]:
            item=prediction(ROOT/f'{split}_s{seed}_selected_local.npz')
            if split=='validation160':
                one=prediction(OLD/'fits'/f'generic_s{seed}/validation_on.npz')
                off=prediction(OLD/'fits'/f'generic_s{seed}/validation_off.npz')
                full=prediction(OLD/'fits'/f'full_s{seed}/validation_on.npz')
            else:
                one=prediction(EXTRA/f'generic_s{seed}_on_local.npz')
                off=prediction(EXTRA/f'generic_s{seed}_off_local.npz')
                full=prediction(EXTRA/f'full_s{seed}_on_local.npz')
            for name,b in [('a1',one),('off',off),('full',full)]:
                key=f'{split}_s{seed}_selected_vs_{name}'
                comparisons[key]=paired(item,b,rows)
                if name=='a1':
                    candidates.append((key,item,b,rows,'extra' if split=='expansion198' else 'original'))
    means={}
    for split in ('validation160','expansion198'):
        a=[r for r in external if r['split']==split]
        means[split]={k:dict(mean=float(np.mean([r[k] for r in a])),std=float(np.std([r[k] for r in a],ddof=1)))
                      for k in ('oa','ba','macro_f1','nll')}
    write(ROOT/'paired.json',dict(selection=selection,external_means=means,comparisons=comparisons))
    with (ROOT/'source_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['seed','scale','oa','ba','macro_f1','nll','update_input_rms_median','update_input_rms_p95']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(source)

    spec=importlib.util.spec_from_file_location('strength_gallery_model',OLD/'model.py')
    model=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=model
    spec.loader.exec_module(model)
    cache={}
    def pixels(row):
        key=row['id']
        if key not in cache:
            fixed,_=model.old.recovered.recovery.recover(row)
            views=model.old.t.e.task.data.prepare(fixed)
            cache[key]=[(v['pixels'] if v is not None and v['valid_target'] else np.zeros((160,160),np.uint8)) for v in views]
        return cache[key]
    gallery=ROOT/'gallery'
    gallery.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    cases=[];seen=set()
    for key,a,b,rows,origin in candidates:
        y=a['label'];z=a['score'];old=b['score']
        right,was=z.argmax(1)==y,old.argmax(1)==y
        delta=margin(z,y)-margin(old,y)
        for kind,mask in [('rescued',right&~was),('harmed',~right&was),('correct',right&was),('wrong',~right&~was)]:
            if kind in ('correct','wrong') and not ('s17' in key and (key.startswith('inner') or key.startswith('expansion'))):
                continue
            possible=np.flatnonzero(mask)
            if not len(possible):continue
            j=int(possible[np.argmax(abs(delta[possible]))])
            row=rows[int(a['ids'][j])]
            tag=(row['id'],kind,'inner' if key.startswith('inner') else key.split('_s')[0])
            if tag in seen:continue
            seen.add(tag)
            rival=old[j].copy();rival[y[j]]=-np.inf;rival=int(rival.argmax())
            refs=[original[i] for i in groups['references']]
            same=next(r for r in refs if r['label']==y[j])
            wrong=next(r for r in refs if r['label']==rival)
            c=dict(comparison=key,kind=kind,id=row['id'],label=int(y[j]),port=row['port'],
                   before=int(old[j].argmax()),after=int(z[j].argmax()),before_margin=float(margin(old,y)[j]),
                   after_margin=float(margin(z,y)[j]),same_reference=same['id'],rival_reference=wrong['id'])
            canvas=Image.new('RGB',(760,660),'white');draw=ImageDraw.Draw(canvas)
            draw.text((10,8),f"{key} / {kind} / {row['id']}",font=font,fill='black')
            draw.text((10,30),f"{row['port']} | {NAMES[y[j]]} | {NAMES[c['before']]} -> {NAMES[c['after']]}",font=small,fill='black')
            draw.text((10,50),f"Margin {c['before_margin']:.6f} -> {c['after_margin']:.6f}",font=small,fill='black')
            for ri,(r,title) in enumerate(zip((row,same,wrong),('Query','Same-class reference','Competing-class reference'))):
                top=90+ri*185
                draw.text((10,top),title,font=font,fill='black')
                draw.text((10,top+25),str(r['id'])+' | '+NAMES[r['label']],font=small,fill='black')
                for p,im in enumerate(pixels(r)):
                    x=290+p*220
                    draw.text((x,top-20),('VV','VH')[p],font=small,fill='black')
                    canvas.paste(Image.fromarray(im).convert('RGB'),(x,top))
            c['file']=f'{len(cases):02d}_{kind}_{row["id"]}.png'
            canvas.save(gallery/c['file']);cases.append(c)
    write(ROOT/'cases_local.json',cases)
    page=['<!doctype html><meta charset="utf-8"><title>PCEA strength cases</title>',
          '<h1>Actual-feature writeback strength</h1><p>Native160 crop/pad; actual fixed-bank references shown as examples, not influence attribution. Source strength8 is diagnostic when not selected. External data use only the source-selected strength.</p>']
    for c in cases:
        page.append(f'<h2>{html.escape(c["comparison"])} / {c["kind"]}: {c["id"]}</h2><img src="{c["file"]}">')
    (gallery/'index.html').write_text('\n'.join(page),encoding='utf-8')
    print(json.dumps(dict(selection=selection,external_means=means,
          contrasts={k:{x:v[x] for x in ('rescued','harmed','net')} for k,v in comparisons.items()},cases=len(cases)),indent=2))


if __name__=='__main__':main()
