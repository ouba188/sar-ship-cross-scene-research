"""Cached paired metrics and native-scale query/reference case panels."""
from pathlib import Path
import html
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
NAMES = ['Bulk', 'Fishing', 'General cargo', 'Chemical/product', 'Container', 'Crude', 'Tug', 'Offshore']


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def write(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def score(arm, seed, mode='on'):
    data = np.load(ROOT/f'{arm}_s{seed}_{mode}_local.npz')
    return data['score']


def margin(z, y):
    rival = z.copy()
    rival[np.arange(len(y)), y] = -np.inf
    return z[np.arange(len(y)),y]-rival.max(1)


def compare(z, b, y):
    right, was = z.argmax(1)==y, b.argmax(1)==y
    counts = np.bincount(y, minlength=8)
    net = right.astype(int)-was.astype(int)
    d = z-b
    change = margin(z,y)-margin(b,y)
    useful = (~was) & (change > 0)
    return dict(n=len(y), rescued=int((right & ~was).sum()), harmed=int((~right & was).sum()),
                changed=int((z.argmax(1)!=b.argmax(1)).sum()), net=int(net.sum()),
                delta_oa=float(net.mean()*100),
                delta_ba=float(np.mean([net[y==c].mean() for c in range(8) if counts[c]])*100),
                class_counts=counts.tolist(), class_net=[int(net[y==c].sum()) for c in range(8)],
                centered_logit_rms=float(np.mean((d-d.mean(1,keepdims=True))**2)**.5),
                max_absolute_margin_change=float(abs(change).max()),
                wrong_cases_with_improved_margin=int(useful.sum()),
                favorable_wrong_deficit_fraction_median=float(np.median(change[useful]/(-margin(b,y)[useful]))) if useful.any() else None)


def main():
    rows = read('evaluation_manifest_local.json')
    n = read('input_scope.json')['n']
    y = np.array([r['label'] for r in rows[:n]])
    native = np.load(ROOT/'native_local.npz')
    results = read('results.json')
    means = {}
    for arm in ('full','generic','class_detail'):
        a = [r for r in results if r['arm']==arm and r['mode']=='on']
        means[arm] = {key:dict(mean=float(np.mean([r[key] for r in a])), std=float(np.std([r[key] for r in a],ddof=1)))
                      for key in ('oa','ba','macro_f1','nll')}
    comparisons, cases, seen = {}, [], set()
    dual = native['available'][:n].all(1)
    for seed in (17,29,43):
        c, g, f = (score(a,seed) for a in ('class_detail','generic','full'))
        contrasts = [('class_vs_full',c,f), ('generic_vs_full',g,f), ('class_vs_generic',c,g),
                     ('class_vs_uniform',c,score('class_detail',seed,'uniform')),
                     ('class_vs_permuted',c,score('class_detail',seed,'permuted')),
                     ('class_vs_off',c,score('class_detail',seed,'off')),
                     ('generic_vs_off',g,score('generic',seed,'off'))]
        for name,z,b in contrasts:
            key = f'{name}_s{seed}'
            item = compare(z,b,y)
            item['ports'] = {p:compare(z[mask],b[mask],y[mask]) for p in sorted({r['port'] for r in rows[:n]})
                             if (mask:=np.array([r['port']==p for r in rows[:n]])).any()}
            item['polarization'] = {k:compare(z[mask],b[mask],y[mask]) for k,mask in [('dual',dual),('single',~dual)] if mask.any()}
            comparisons[key] = item
            if name not in ('class_vs_full','class_vs_generic','class_vs_uniform','class_vs_off'):
                continue
            right,was = z.argmax(1)==y,b.argmax(1)==y
            for kind, mask in [('rescued',right & ~was),('harmed',~right & was),('correct',right & was),('wrong',~right & ~was)]:
                if kind in ('correct','wrong') and (seed!=17 or name not in ('class_vs_generic','class_vs_uniform','class_vs_off')):
                    continue
                eligible = np.flatnonzero(mask)
                if not len(eligible):
                    continue
                j = int(eligible[np.argmax(abs((margin(z,y)-margin(b,y))[eligible]))])
                if (j,kind,name) in seen:
                    continue
                seen.add((j,kind,name))
                rival = b[j].copy()
                rival[y[j]] = -np.inf
                rival = int(rival.argmax())
                same = next(k for k in range(n,n+16) if rows[k]['label']==y[j])
                other = next(k for k in range(n,n+16) if rows[k]['label']==rival)
                cases.append(dict(comparison=key,kind=kind,index=j,id=rows[j]['id'],port=rows[j]['port'],
                                  label=int(y[j]),before=int(b[j].argmax()),after=int(z[j].argmax()),
                                  before_margin=float(margin(b,y)[j]),after_margin=float(margin(z,y)[j]),
                                  same_reference=rows[same]['id'],rival_reference=rows[other]['id'],
                                  panel_indices=[j,same,other]))
    write('paired.json',dict(means=means,comparisons=comparisons))
    folder = ROOT/'gallery'
    folder.mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    page = ['<!doctype html><meta charset="utf-8"><title>PCEA new-vessel expansion</title>',
            '<h1>New vessels, known passages: frozen PCEA evaluation</h1>',
            '<p>Native160 crop/pad, VV/VH separate. Reference examples from the actual fixed16 bank, not ranked causal attribution. Cases are selected for diagnosis, not a population estimate.</p>']
    for k,c in enumerate(cases):
        canvas = Image.new('RGB',(750,660),'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12,8),f"{c['comparison']} / {c['kind']} / {c['id']}",fill='black',font=font)
        draw.text((12,32),f"{c['port']} | {NAMES[c['label']]} | {NAMES[c['before']]} -> {NAMES[c['after']]}",fill='black',font=small)
        draw.text((12,52),f"Margin {c['before_margin']:.6f} -> {c['after_margin']:.6f}",fill='black',font=small)
        for ri,(idx,title) in enumerate(zip(c['panel_indices'],('Query','Same-class reference','Competing-class reference'))):
            top = 90+ri*185
            draw.text((12,top),title,fill='black',font=font)
            draw.text((12,top+24),rows[idx]['id'],fill='black',font=small)
            draw.text((12,top+44),NAMES[rows[idx]['label']],fill='black',font=small)
            for p,pol in enumerate(('VV','VH')):
                x = 290+p*220
                draw.text((x,top-20),pol,fill='black',font=small)
                canvas.paste(Image.fromarray(native['pixels'][idx,p]).convert('RGB'),(x,top))
                if not native['available'][idx,p]:
                    draw.text((x+8,top+70),'Unavailable',fill='red',font=small)
        filename = f'{k:02d}_{c["kind"]}_{c["id"]}.png'
        canvas.save(folder/filename)
        c['file'] = filename
        page.append(f'<h2>{html.escape(c["comparison"])}: {c["id"]} / {c["kind"]}</h2><img src="{filename}">')
    write('cases_local.json',cases)
    (folder/'index.html').write_text('\n'.join(page),encoding='utf-8')
    print(json.dumps(dict(means=means,contrasts={k:{x:v[x] for x in ('rescued','harmed','net','changed')} for k,v in comparisons.items()},cases=len(cases)),indent=2))


if __name__ == '__main__':
    main()
