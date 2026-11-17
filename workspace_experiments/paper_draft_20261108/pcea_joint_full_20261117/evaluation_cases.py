"""Native evaluation inputs for paired rescue/harm and unchanged decisions."""
from pathlib import Path
import argparse
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
PAPER=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
parser=argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=17)
args=parser.parse_args()
rows=read(PAPER.parent/'m1_coverage_20261101/manifest.json')
groups=read(PAPER.parent/'m1_coverage_20261101/groups.json')
pixels=np.load(PAPER/'recent_baselines_20261111/native_cache/pixels.npy',mmap_mode='r')
available=np.load(PAPER/'recent_baselines_20261111/native_cache/available.npy')
names=('Bulk','General cargo','Chemical/product tanker','Container')
scores={}
for arm in ('full','cross_mean','cross_local'):
    with np.load(ROOT/'evaluation'/f'{arm}_s{args.seed}'/'predictions.npz') as f:
        if scores:
            assert np.array_equal(ids,f['ids']) and np.array_equal(y,f['label'])
        else:
            ids,y=f['ids'].copy(),f['label'].copy()
        scores[arm]=f['score'].copy()

def margin(z):
    wrong=z.copy();wrong[np.arange(len(y)),y]=-np.inf
    return z[np.arange(len(y)),y]-wrong.max(1)

folder=ROOT/f'evaluation_cases_s{args.seed}';folder.mkdir(exist_ok=True)
plt.rcParams['svg.fonttype']='none'
records=[]
base=scores['full'].argmax(1)==y
for arm in ('cross_mean','cross_local'):
    good=scores[arm].argmax(1)==y
    for state,mask in (
        ('rescue',good & ~base),('harm',~good & base),
        ('stable_correct',good & base),('stable_wrong',~good & ~base)):
        candidates=np.flatnonzero(mask)
        if not len(candidates):continue
        ranked=candidates[np.argsort(np.abs(margin(scores[arm])[candidates]))]
        j=int(ranked[len(ranked)//2])
        idx=int(ids[j]);row=rows[idx]
        wrong=scores['full'][j].copy();wrong[y[j]]=-np.inf
        rival=int(wrong.argmax())
        detail={k:dict(prediction=int(z[j].argmax()),margin=float(margin(z)[j])) for k,z in scores.items()}
        fig,axes=plt.subplots(2,3,figsize=(10,7));references=[]
        for p,pol in enumerate(('VV','VH')):
            for ax in axes[p]:ax.axis('off')
            axes[p,0].imshow(pixels[idx,p],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[p,0].set_title(pol+' query native160',fontsize=9)
            for col,k in ((1,int(y[j])),(2,rival)):
                usable=[i for i in groups['references'] if rows[i]['label']==k and available[i,p]
                        and rows[i]['mmsi']!=row['mmsi'] and rows[i]['pass_group']!=row['pass_group']]
                if usable:
                    ri=usable[0]
                    axes[p,col].imshow(pixels[ri,p],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    axes[p,col].set_title(('True ' if col==1 else 'Rival ')+rows[ri]['id'],fontsize=7)
                    references.append(dict(index=ri,id=rows[ri]['id'],polarization=pol))
        fig.suptitle(f'{row["id"]} | {names[y[j]]} | {arm} seed{args.seed} | {state}\n'+
                     ' / '.join(f'{k}: {v["margin"]:+.4f}' for k,v in detail.items()),fontsize=9)
        fig.text(.5,.02,'Historically exposed evaluation split. Frozen validation choices. Legal reference examples, not influence ranking.',ha='center',fontsize=7)
        fig.subplots_adjust(top=.85,bottom=.06,left=.02,right=.98,wspace=.12,hspace=.25)
        stem=f'{arm}_{state}_{idx}'
        fig.savefig(folder/(stem+'.png'),dpi=130);fig.savefig(folder/(stem+'.svg'));plt.close(fig)
        records.append(dict(arm=arm,state=state,index=idx,id=row['id'],label=int(y[j]),image=stem+'.png',
                            group_n=int(mask.sum()),details=detail,references=references,vv=row['vv'],vh=row['vh']))
(folder/'cases_local.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
body=''.join(f'<h2>{r["arm"]}: {r["state"]} ({r["group_n"]} total)</h2><img style="max-width:100%" src="{r["image"]}"><pre>{html.escape(str(r["details"]))}</pre>' for r in records)
(folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Frozen evaluation cases</title>'+body,encoding='utf-8')
print(folder)
