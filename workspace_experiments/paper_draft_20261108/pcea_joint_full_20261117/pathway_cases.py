"""CPU-only native-input panels for actual same-weight branch-dependent decisions."""
from pathlib import Path
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PAPER=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
rows=read(PAPER.parent/'m1_coverage_20261101/manifest.json')
groups=read(PAPER.parent/'m1_coverage_20261101/groups.json')
pixels=np.load(PAPER/'recent_baselines_20261111/native_cache/pixels.npy',mmap_mode='r')
available=np.load(PAPER/'recent_baselines_20261111/native_cache/available.npy')
selection=read(ROOT/'selection.json')['selected']
names=('Bulk','General cargo','Chemical/product tanker','Container')
plt.rcParams['svg.fonttype']='none'
folder=ROOT/'pathway_cases_s17';folder.mkdir(exist_ok=True)
records=[]


def margin(z,y):
    wrong=np.array(z,copy=True);wrong[y]=-np.inf
    return float(z[y]-wrong.max())


for arm in ('cross_mean','cross_local'):
    with np.load(ROOT/f'diagnostics_{arm}_s17.npz') as f:
        data={k:f[k].copy() for k in f.files}
    ids,y=data['ids'],data['label']
    correct=data['actual'].argmax(1)==y
    flags=('disabled','no_product','own_context','uniform_attention')
    chosen=np.flatnonzero(np.logical_or.reduce([correct != (data[k].argmax(1)==y) for k in flags]))
    for j in chosen:
        idx=int(ids[j]);row=rows[idx]
        rival=int(data['disabled'][j].argmax())
        if rival==y[j]:
            rival=int(data['actual'][j].argmax())
        affected=[k for k in flags if correct[j] != (data[k][j].argmax()==y[j])]
        values={k:dict(prediction=int(data[k][j].argmax()),margin=margin(data[k][j],int(y[j])))
                for k in ('actual',)+flags}
        fig,axes=plt.subplots(2,3,figsize=(10,7));references=[]
        for p,pol in enumerate(('VV','VH')):
            for ax in axes[p]:ax.axis('off')
            axes[p,0].imshow(pixels[idx,p],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[p,0].set_title(pol+' query native160',fontsize=9)
            for col,k in ((1,int(y[j])),(2,rival)):
                usable=[i for i in groups['references'] if rows[i]['label']==k and available[i,p]]
                if usable:
                    ri=usable[0]
                    assert rows[ri]['mmsi']!=row['mmsi'] and rows[ri]['pass_group']!=row['pass_group']
                    axes[p,col].imshow(pixels[ri,p],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    axes[p,col].set_title(('True' if col==1 else 'Rival')+' '+rows[ri]['id'],fontsize=7)
                    references.append(dict(index=ri,id=rows[ri]['id'],polarization=pol,role='true' if col==1 else 'rival'))
        fig.suptitle(f'{row["id"]} | {names[int(y[j])]} | {arm} seed17\n'+
            ' / '.join(f'{k}: {v["margin"]:+.4f}' for k,v in values.items()),fontsize=9)
        fig.text(.5,.025,'Same selected weights; original classification vs pathway replacement. Native VV/VH; no intensity restretch.',ha='center',fontsize=8)
        fig.subplots_adjust(top=.85,bottom=.06,left=.02,right=.98,wspace=.12,hspace=.25)
        stem=f'{arm}_{idx}';fig.savefig(folder/(stem+'.png'),dpi=130);fig.savefig(folder/(stem+'.svg'));plt.close(fig)
        records.append(dict(arm=arm,index=idx,id=row['id'],label=int(y[j]),actual_correct=bool(correct[j]),
            affected=affected,details=values,image=stem+'.png',references=references,vv=row['vv'],vh=row['vh']))
(folder/'cases_local.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
body=''.join(f'<h2>{html.escape(r["arm"]+" "+r["id"])}</h2><img style="max-width:100%" src="{r["image"]}"><pre>{html.escape(str(r["details"]))}</pre>' for r in records)
(folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Same-weight pathway cases</title><h1>Actual pathway-sensitive decisions</h1>'+body,encoding='utf-8')
print(json.dumps([{k:r[k] for k in ('arm','index','id','label','actual_correct','affected','details')} for r in records],ensure_ascii=False,indent=2))
