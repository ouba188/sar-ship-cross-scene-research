"""Paired, class-stratified summaries and native-chip evidence panels."""
from collections import Counter
from pathlib import Path
import csv
import html
import importlib.util
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from audit import read,save
NAMES = ['Bulk','Fishing','General cargo','Chemical/product','Container','Crude','Tug','Offshore']
SEEDS = (17,29,43)
SPLITS = {'validation':np.arange(160),'expansion':np.arange(160,358)}


def csvsave(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def margin(z,y):
    wrong = z.copy()
    wrong[np.arange(len(y)),y] = -np.inf
    return z[np.arange(len(y)),y]-wrong.max(1)


def main():
    assert read(ROOT/'status.json')['complete']
    # Exposure means observations consumed before each selected checkpoint,
    # not all later updates or the complete pretraining history.
    selection = read(ROOT/'selection_local.json')
    selection_summary = read(ROOT/'selection_summary.json')
    exposure = {}
    for seed in SEEDS:
        dest = ROOT.parent/'pcea_routed_correction_20261118/fits'/f'routed_s{seed}'
        step = read(dest/'receipt.json')['best']['step']
        batch = read(dest/'config.json')['batch']
        seen = set(np.load(dest/'training_ids_local.npy')[:step*batch].tolist())
        exposure[str(seed)] = {bank:sum(r['source_index'] in seen for r in selection if r['bank'] == bank)
                              for bank in sorted({r['bank'] for r in selection})}
    selection_summary['continuation_training_exposure'] = exposure
    selection_summary['exposure_scope'] = 'observation exposure before selected continuation checkpoint only'
    save(ROOT/'selection_summary.json',selection_summary)
    rows = read(ROOT/'manifest_local.json')
    indices = read(ROOT/'bank_indices_local.json')
    banks = list(indices)
    out = ROOT/'aggregate'
    out.mkdir(exist_ok=True)
    summary, classes, ports, passages, stability, cluster, scores = [],[],[],[],[],[],{}
    for split,ids in SPLITS.items():
        y = np.array([rows[i]['label'] for i in ids])
        ships = np.array([rows[i]['mmsi'] for i in ids])
        z = {}
        for bank in ['m3_off']+banks:
            values = []
            for seed in SEEDS:
                data = np.load(ROOT/f'{split}_s{seed}_{bank}_local.npz')
                assert np.array_equal(data['ids'],ids) and np.array_equal(data['label'],y)
                values.append(data['score'])
            z[bank] = np.stack(values)
        scores[split] = z
        correct = {k:v.argmax(2) == y for k,v in z.items()}
        off, original = correct['m3_off'],correct['original']
        for bank,ok in correct.items():
            delta = ok.astype(float)-off
            recalls = np.array([ok[:,y == c].mean(1) for c in np.unique(y)])
            offrec = np.array([off[:,y == c].mean(1) for c in np.unique(y)])
            summary.append(dict(split=split,bank=bank,n=len(ids),oa=100*ok.mean(),ba=100*recalls.mean(),
                delta_off_oa=100*delta.mean(),delta_off_ba=100*(recalls-offrec).mean(),
                delta_original_oa=100*(ok.astype(float)-original).mean(),
                mean_rescued=(~off & ok).sum(1).mean(),mean_harmed=(off & ~ok).sum(1).mean(),
                seed_delta_oa_min=100*delta.mean(1).min(),seed_delta_oa_max=100*delta.mean(1).max()))
            oldrescue = ~off & original
            stability.append(dict(split=split,bank=bank,original_rescue_decisions=int(oldrescue.sum()),
                original_rescues_retained=int((oldrescue & ok).sum()),
                all_seed_rescue_queries=int((~off & ok).all(0).sum()),
                all_seed_harm_queries=int((off & ~ok).all(0).sum())))
            for c in np.unique(y):
                mask = y == c
                classes.append(dict(split=split,bank=bank,label=int(c),name=NAMES[c],n=int(mask.sum()),
                    delta_recall=100*delta[:,mask].mean(),
                    mean_rescued=(~off[:,mask]&ok[:,mask]).sum(1).mean(),
                    mean_harmed=(off[:,mask]&~ok[:,mask]).sum(1).mean()))
            for field,target in (('port',ports),('pass_group',passages)):
                group = np.array([rows[i][field] for i in ids])
                for value in np.unique(group):
                    mask = group == value
                    # Equal category weight within this same fixed group.
                    class_delta = [delta[:,mask & (y == c)].mean() for c in np.unique(y[mask])]
                    target.append(dict(split=split,bank=bank,group=value,n=int(mask.sum()),
                        classes=len(class_delta),delta_oa=100*delta[:,mask].mean(),
                        delta_macro_recall=100*np.mean(class_delta)))
            byship = np.array([delta[:,ships == s].mean() for s in np.unique(ships)])
            rng = np.random.default_rng(20261118)
            boot = byship[rng.integers(len(byship),size=(2000,len(byship)))].mean(1)*100
            cluster.append(dict(split=split,bank=bank,vessels=len(byship),
                equal_vessel_delta=100*byship.mean(),descriptive_ci_low=float(np.quantile(boot,.025)),
                descriptive_ci_high=float(np.quantile(boot,.975))))
    for name,values in [('summary',summary),('by_class',classes),('by_port',ports),('by_passage',passages),
                        ('rescue_stability',stability),('vessel_summary',cluster)]:
        csvsave(out/f'{name}.csv',values)
        save(out/f'{name}.json',values)
    families = []
    robust, robust_local = [], []
    for split in SPLITS:
        ids = SPLITS[split]
        y = np.array([rows[i]['label'] for i in ids])
        z = scores[split]
        off = z['m3_off'].argmax(2) == y
        foreign = np.stack([z['outside_query_ports_'+str(i)].argmax(2) == y for i in (1,2,3)])
        for kind, mask in (
            ('rescued_all',(~off).all(0) & foreign.all((0,1))),
            ('harmed_all',off.all(0) & (~foreign).all((0,1)))):
            for j in np.flatnonzero(mask):
                row = rows[ids[j]]
                robust_local.append(dict(split=split,kind=kind,id=row['id'],mmsi=row['mmsi'],
                    label=row['label'],port=row['port']))
        robust.append(dict(split=split,
            all_three_seeds_off_wrong=int((~off).all(0).sum()),
            rescued_all_three_foreign_banks_all_three_seeds=int(((~off).all(0) & foreign.all((0,1))).sum()),
            harmed_all_three_foreign_banks_all_three_seeds=int((off.all(0) & (~foreign).all((0,1))).sum())))
        for family in ('original','same_port','outside_query_ports'):
            matches = [r for r in summary if r['split'] == split and r['bank'].startswith(family)]
            families.append(dict(split=split,family=family,
                **{key:float(np.mean([r[key] for r in matches])) for key in
                   ('oa','ba','delta_off_oa','delta_off_ba','mean_rescued','mean_harmed')}))
    save(out/'family_summary.json',families)
    save(out/'robust_case_counts.json',robust)
    save(ROOT/'robust_cases_local.json',robust_local)
    make_cases(rows,indices,scores)
    print(json.dumps(families,ensure_ascii=False,indent=2))


def make_cases(rows,indices,scores):
    from PIL import Image,ImageDraw,ImageFont
    spec = importlib.util.spec_from_file_location('reference_native_input',ROOT.parents[1]/'sar_m123_standalone/sar_m123/input.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    gallery = ROOT/'gallery'
    gallery.mkdir(exist_ok=True)
    selected, used = [],set()
    foreign = ['outside_query_ports_'+str(i) for i in (1,2,3)]
    candidates = []
    for split,ids in SPLITS.items():
        y = np.array([rows[i]['label'] for i in ids])
        z = scores[split]
        off = z['m3_off'].argmax(2) == y
        orig = z['original'].argmax(2) == y
        allforeign = np.stack([z[k].argmax(2) == y for k in foreign])
        for j,idx in enumerate(ids):
            rate = allforeign[:,:,j].mean()
            for bank in foreign:
                after = z[bank][0,j].argmax() == y[j]
                typ = None
                priority = 0
                if not off[0,j] and after:
                    typ,priority = 'rescued',rate+off[:,j].mean()*-.1
                elif off[0,j] and not after:
                    typ,priority = 'harmed',1-rate
                elif off[:,j].all() and allforeign[:,:,j].all():
                    typ = 'stable_correct'
                elif not off[:,j].any() and not allforeign[:,:,j].any():
                    typ = 'stable_wrong'
                if typ:
                    candidates.append(dict(kind=typ,priority=float(priority),split=split,index=int(idx),
                        local_index=j,bank=bank,foreign_correct_fraction=float(rate),
                        original_correct_all_seeds=bool(orig[:,j].all())))
                if orig[0,j] and not after and not off[0,j]:
                    candidates.append(dict(kind='original_rescue_lost',priority=1-rate,split=split,
                        index=int(idx),local_index=j,bank=bank,foreign_correct_fraction=float(rate),
                        original_correct_all_seeds=bool(orig[:,j].all())))
    for kind in ('rescued','harmed','original_rescue_lost','stable_correct','stable_wrong'):
        choices = sorted([c for c in candidates if c['kind'] == kind and c['index'] not in used],key=lambda x:-x['priority'])
        if choices:
            c = choices[0]
            selected.append(c)
            used.add(c['index'])
    for c in candidates:
        if c['kind'] == 'rescued' and c['priority'] == 1 and c['index'] not in used:
            selected.append(c)
            used.add(c['index'])
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    pages = []
    for number,case in enumerate(selected):
        q = rows[case['index']]
        z = scores[case['split']]
        j = case['local_index']
        y = q['label']
        offz = z['m3_off'][0,j]
        rival = int(np.argmax(np.where(np.arange(8) == y,-np.inf,offz)))
        rivals = {rival}
        rivals.update(int(z[k][0,j].argmax()) for k in ('original',case['bank'])
                      if int(z[k][0,j].argmax()) != y)
        panels = [('Query',case['index'])]
        for bank,tag in (('original','Original'),(case['bank'],'Changed')):
            for label,role in [(y,'same class')]+[(r,'rival class') for r in sorted(rivals)]:
                panels.extend((tag+' '+role,i) for i in indices[bank] if rows[i]['label'] == label)
        canvas = Image.new('RGB',(190*len(panels),500),'white')
        draw = ImageDraw.Draw(canvas)
        title = f"{case['kind']} | query {q['id']} | {q['port']} | {NAMES[y]} | seed17 | {case['bank']}"
        draw.text((8,5),title,fill='black',font=font)
        sources = []
        for col,(role,idx) in enumerate(panels):
            row = rows[idx]
            x = col*190+8
            draw.text((x,32),role,fill='black',font=font)
            draw.text((x,51),f"{row['id']} {NAMES[row['label']]}",fill='black',font=font)
            draw.text((x,70),row['port'],fill='black',font=font)
            for p,pol in enumerate(('vv','vh')):
                view = native.prepare_view(row.get(pol))
                top = 96+p*190
                good = view is not None and view['valid_target']
                draw.text((x,top),pol.upper()+('' if good else ' unavailable'),fill='black',font=font)
                if view is not None:
                    canvas.paste(Image.fromarray(view['pixels']).convert('RGB'),(x,top+20))
                sources.append(dict(role=role,id=row['id'],polarization=pol,
                    image=row[pol]['image'] if row.get(pol) else None,active=good))
        name=f'{number:02d}_{case["kind"]}_{q["id"]}.png'
        canvas.save(gallery/name)
        outcomes = {}
        for bank in ('m3_off','original',case['bank']):
            logits = z[bank][0,j]
            outcomes[bank] = dict(prediction=NAMES[int(logits.argmax())],
                margin=float(margin(logits[None],np.array([y]))[0]))
        case.update(id=q['id'],mmsi=q['mmsi'],label=y,port=q['port'],seed=17,
            image=name,outcomes=outcomes,sources=sources)
        pages.append(f'<section><h2>{html.escape(title)}</h2><p>Foreign-bank correct fraction across 3 banks x 3 seeds: {case["foreign_correct_fraction"]:.3f}</p>'
            f'<pre>{html.escape(json.dumps(outcomes,indent=2))}</pre><img src="{name}" alt="Native VV and VH chips with actual references"></section>')
    save(ROOT/'cases_local.json',selected)
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Reference controls</title>'
        '<style>body{font:16px Arial;margin:24px;color:#202020}section{border-top:1px solid #aaa;margin-top:28px}img{max-width:100%;height:auto}pre{white-space:pre-wrap}</style>'
        '<h1>Frozen M3 reference-dependence controls</h1><p>Native160 pixels, no resizing or new contrast stretch. Both references for the true class and each displayed competing class are shown for both banks. Diagnostic examples, not population proportions.</p>'
        +''.join(pages),encoding='utf-8')


if __name__ == '__main__':
    main()
