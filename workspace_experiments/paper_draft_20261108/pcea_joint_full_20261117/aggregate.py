"""Paired seed/class/polarization accounting; repeated predictions are not new ships."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
manifest=read(ROOT.parent.parent/'m1_coverage_20261101/manifest.json')
selection=read(ROOT/'selection.json')['selected']
arms=('full','cross_mean','cross_local')
seeds=(17,29,43)
out={}
for split in ('validation','evaluation'):
    byseed=[];groups=[]
    for seed in seeds:
        scores={}
        for arm in arms:
            dest=(ROOT/'evaluation'/f'{arm}_s{seed}'/'predictions.npz') if split=='evaluation' else (
                ROOT/'fits'/f'{arm}_s{seed}_{selection[arm]["config"]}'/'validation.npz')
            with np.load(dest) as f:
                if scores:assert np.array_equal(ids,f['ids']) and np.array_equal(y,f['label'])
                ids,y=f['ids'].copy(),f['label'].copy()
                scores[arm]=f['score'].copy()
        dual=np.array([manifest[i]['vv'] is not None and manifest[i]['vh'] is not None for i in ids])
        masks={'all':np.ones(len(y),bool),'dual':dual,'single':~dual}
        masks.update({f'class{k}':y==k for k in range(4)})
        masks.update({'pass:'+p:np.array([manifest[i]['pass_group']==p for i in ids])
                      for p in sorted({manifest[i]['pass_group'] for i in ids})})
        base=scores['full'].argmax(1)==y
        mean=scores['cross_mean'].argmax(1)==y
        for arm,z in scores.items():
            correct=z.argmax(1)==y
            oa=float(correct.mean()*100)
            ba=float(np.mean([correct[y==k].mean() for k in range(4)])*100)
            byseed.append(dict(seed=seed,arm=arm,oa=oa,ba=ba,
                rescued=int((correct & ~base).sum()),harmed=int((~correct & base).sum()),
                vs_mean_rescued=int((correct & ~mean).sum()),vs_mean_harmed=int((~correct & mean).sum())))
            for name,mask in masks.items():
                groups.append(dict(seed=seed,arm=arm,group=name,n=int(mask.sum()),
                    correct=int(correct[mask].sum()),full_correct=int(base[mask].sum()),
                    rescued=int((correct[mask] & ~base[mask]).sum()),
                    harmed=int((~correct[mask] & base[mask]).sum())))
    means={}
    for population,ss in [('all_three',seeds),('confirmation_only',(29,43))]:
        means[population]={a:{k:dict(mean=float(np.mean(v:=[r[k] for r in byseed if r['arm']==a and r['seed'] in ss])),
                                   std=float(np.std(v,ddof=1))) for k in ('oa','ba')} for a in arms}
    totals=[]
    for arm in arms:
        for name in masks:
            rr=[r for r in groups if r['arm']==arm and r['group']==name]
            totals.append(dict(arm=arm,group=name,**{k:sum(r[k] for r in rr) for k in ('n','correct','full_correct','rescued','harmed')}))
    out[split]=dict(rows=byseed,means=means,groups=groups,repeated_prediction_totals=totals)
out['diagnostics']=[]
for seed in seeds:
    for arm in ('cross_mean','cross_local'):
        path=ROOT/f'diagnostics_{arm}_s{seed}.json'
        if path.exists():out['diagnostics'].append(dict(seed=seed,**read(path)))
out['note']='All-three includes configuration-selection seed17. Confirmation29/43 uses locked LR. Totals repeat the same ships across seeds; not independent sample counts. Exposed evaluation is not a new blind test.'
if (ROOT/'evaluation_pathways.json').exists():
    out['evaluation_pathways']=read(ROOT/'evaluation_pathways.json')
out['training_curves']=[]
for seed in seeds:
    for arm in arms:
        folder=ROOT/'fits'/f'{arm}_s{seed}_{selection[arm]["config"]}'
        receipt=read(folder/'receipt.json');curve=read(folder/'curve.json')
        best=next(r for r in curve if r['epoch']==receipt['best']['epoch'])
        out['training_curves'].append(dict(seed=seed,arm=arm,best=best,last=curve[-1]))
(ROOT/'paired_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for split in ('validation','evaluation'):
    print(split, out[split]['means'])
    print([r for r in out[split]['repeated_prediction_totals'] if r['group'] in ('dual','single')])
