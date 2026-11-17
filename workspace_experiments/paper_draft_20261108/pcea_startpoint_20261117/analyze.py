"""Paired validation summaries and native query/reference case sheets."""
from pathlib import Path
import argparse
import csv
import html
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
PAPER=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def score(path):
    with np.load(path) as data:
        return {k:data[k].copy() for k in ('ids','label','score')}


def margin(z,y):
    wrong=z.copy();wrong[np.arange(len(y)),y]=-np.inf
    return z[np.arange(len(y)),y]-wrong.max(1)


def paired(z, base, y, dual, rows, ids):
    right=z.argmax(1)==y; old=base.argmax(1)==y
    rescue=right & ~old; harm=old & ~right
    def count(mask):
        return dict(n=int(mask.sum()),rescued=int((rescue&mask).sum()),
                    harmed=int((harm&mask).sum()),net=int((rescue&mask).sum()-(harm&mask).sum()))
    return dict(overall=count(np.ones(len(y),dtype=bool)),dual=count(dual),single=count(~dual),
        classes={str(k):count(y==k) for k in range(4)},
        passes={k:count(np.array([rows[i]['pass_group']==k for i in ids]))
                for k in sorted({rows[i]['pass_group'] for i in ids})})


def summary(seed=17):
    rows=read(PAPER.parent/'m1_coverage_20261101/manifest.json')
    available=np.load(PAPER/'recent_baselines_20261111/native_cache/available.npy')
    initial=score(PAPER/f'full_finetune_20261111/selected_s{seed}/validation.npz')
    ids,y=initial['ids'],initial['label']
    dual=available[ids].astype(bool).all(1)
    fits={}; table=[]
    for strength in (0.,.1):
        for arm in ('full','cross_mean'):
            name=f'{arm}_s{seed}_sp{strength:g}'
            dest=ROOT/'fits'/name
            receipt=read(dest/'receipt.json'); curve=read(dest/'curve.json')
            data=score(dest/'validation.npz')
            assert np.array_equal(ids,data['ids']) and np.array_equal(y,data['label'])
            fits[name]=dict(receipt=receipt,curve=curve,score=data['score'])
            table.append(dict(arm=arm,seed=seed,strength=strength,epoch=receipt['best']['epoch'],
                **{k:receipt['best']['metrics'][k] for k in ('oa','ba','macro_f1','nll')},
                last_oa=receipt['last']['oa'],last_ba=receipt['last']['ba'],
                final_backbone_drift=receipt['final_drift']['backbone']['l2_distance'],
                final_module_drift=receipt['final_drift']['old_modules']['l2_distance']))
    with (ROOT/f'metrics_s{seed}.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=table[0]);writer.writeheader();writer.writerows(table)
    comparisons={}
    for strength in (0.,.1):
        base=f'full_s{seed}_sp{strength:g}';candidate=f'cross_mean_s{seed}_sp{strength:g}'
        comparisons[f'pcea_minus_full_sp{strength:g}']=paired(fits[candidate]['score'],fits[base]['score'],y,dual,rows,ids)
    for arm in ('full','cross_mean'):
        comparisons[arm+'_anchor_minus_plain']=paired(fits[f'{arm}_s{seed}_sp0.1']['score'],
            fits[f'{arm}_s{seed}_sp0']['score'],y,dual,rows,ids)
        comparisons[arm+'_anchor_minus_initial']=paired(fits[f'{arm}_s{seed}_sp0.1']['score'],
            initial['score'],y,dual,rows,ids)
        path=ROOT/'fits'/f'{arm}_s{seed}_sp0.1'/'lastvalidation.npz'
        if path.exists():
            comparisons[arm+'_last_anchor_minus_initial']=paired(score(path)['score'],
                initial['score'],y,dual,rows,ids)
    write(ROOT/f'paired_s{seed}.json',dict(table=table,comparisons=comparisons))
    if seed==17:
        base=fits['full_s17_sp0.1']['receipt']['best']['metrics']
        candidate=fits['cross_mean_s17_sp0.1']['receipt']['best']['metrics']
        init=fits['full_s17_sp0.1']['receipt']['initial']
        better=lambda a,b:(a['ba']>=b['ba'] and a['oa']>b['oa']) or (a['ba']>b['ba'] and a['oa']>=b['oa'])
        write(ROOT/'decision.json',dict(confirm=better(candidate,base) and better(candidate,init),
            candidate=candidate,matched_full=base,initial=init,
            rule='Predeclared validation Pareto gain over BOTH same-anchor Full and initial Full. No evaluation used.'))
    print(json.dumps(dict(table=table,comparisons=comparisons),indent=2))


def gallery(seed=17, last=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read(PAPER.parent/'m1_coverage_20261101/manifest.json')
    groups=read(PAPER.parent/'m1_coverage_20261101/groups.json')
    pixels=np.load(PAPER/'recent_baselines_20261111/native_cache/pixels.npy',mmap_mode='r')
    available=np.load(PAPER/'recent_baselines_20261111/native_cache/available.npy')
    names=('Bulk','General cargo','Chemical/product tanker','Container')
    scores={}; epochs={}
    for arm in ('full','cross_mean'):
        for strength in (0.,.1):
            data=score(ROOT/'fits'/f'{arm}_s{seed}_sp{strength:g}'/('lastvalidation.npz' if last else 'validation.npz'))
            ids,y=data['ids'],data['label'];scores[f'{arm}_{strength:g}']=data['score']
            receipt=read(ROOT/'fits'/f'{arm}_s{seed}_sp{strength:g}'/'receipt.json')
            epochs[f'{arm}_{strength:g}']=8 if last else receipt['best']['epoch']
    folder=ROOT/(f'last_gallery_s{seed}' if last else f'gallery_s{seed}');folder.mkdir(exist_ok=True)
    records=[]
    initial=score(PAPER/f'full_finetune_20261111/selected_s{seed}/validation.npz')
    scores['initial']=initial['score']
    epochs['initial']=0
    contrasts=[('PCEA','full_0.1','cross_mean_0.1'),('anchor','full_0','full_0.1')]
    if last:
        contrasts.append(('continuation','initial','cross_mean_0.1'))
        off=score(ROOT/'fits'/f'cross_mean_s{seed}_sp0'/'lastvalidation_off.npz')
        scores['cross_mean_0_off']=off['score'];epochs['cross_mean_0_off']=8
        contrasts.append(('direct_branch','cross_mean_0_off','cross_mean_0'))
    for label,base_key,key in contrasts:
        base=scores[base_key].argmax(1)==y;good=scores[key].argmax(1)==y
        for state,mask in (('rescue',good&~base),('harm',~good&base),
                           ('stable_correct',good&base),('stable_wrong',~good&~base)):
            candidates=np.flatnonzero(mask)
            if not len(candidates):continue
            ranked=candidates[np.argsort(abs(margin(scores[key],y)[candidates]))]
            j=int(ranked[len(ranked)//2]);idx=int(ids[j]);row=rows[idx]
            wrong=scores[base_key][j].copy();wrong[y[j]]=-np.inf;rival=int(wrong.argmax())
            detail={k:dict(prediction=int(z[j].argmax()),margin=float(margin(z,y)[j]),epoch=epochs[k]) for k,z in scores.items()}
            fig,axes=plt.subplots(2,3,figsize=(10,7));refs=[]
            for p,pol in enumerate(('VV','VH')):
                for ax in axes[p]:ax.axis('off')
                axes[p,0].imshow(pixels[idx,p],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[p,0].set_title(f'{pol} query native160 / valid {int(available[idx,p])}',fontsize=9)
                for col,k in ((1,int(y[j])),(2,rival)):
                    pool=[i for i in groups['references'] if rows[i]['label']==k and available[i,p]
                          and rows[i]['mmsi']!=row['mmsi'] and rows[i]['pass_group']!=row['pass_group']]
                    if pool:
                        ri=pool[0];refs.append(dict(index=ri,id=rows[ri]['id'],polarization=pol))
                        axes[p,col].imshow(pixels[ri,p],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                        axes[p,col].set_title(('True ' if col==1 else 'Rival ')+rows[ri]['id'],fontsize=7)
            shown=(base_key,key,'initial')
            fig.suptitle(f'{row["id"]} | {names[y[j]]} | {label} {state} | seed{seed} | '+('last' if last else 'best')+'\n'+
                         ' / '.join(f'{k}: {detail[k]["margin"]:+.4f}' for k in shown),fontsize=9)
            fig.text(.5,.02,f'{base_key} e{epochs[base_key]} vs {key} e{epochs[key]}. Epoch0 is original Full, NOT learned PCEA benefit. Legal reference examples.',ha='center',fontsize=7)
            fig.subplots_adjust(top=.85,bottom=.06,left=.02,right=.98,wspace=.12,hspace=.25)
            stem=f'{label}_{state}_{idx}';fig.savefig(folder/(stem+'.png'),dpi=130);plt.close(fig)
            records.append(dict(contrast=label,state=state,n=int(mask.sum()),index=idx,id=row['id'],
                label=int(y[j]),image=stem+'.png',details=detail,references=refs,vv=row['vv'],vh=row['vh']))
    write(folder/'cases_local.json',records)
    body=''.join(f'<h2>{r["contrast"]}: {r["state"]} (n={r["n"]})</h2><img style="max-width:100%" src="{r["image"]}"><pre>{html.escape(str(r["details"]))}</pre>' for r in records)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Startpoint cases</title>'+body,encoding='utf-8')
    print(folder)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--gallery',action='store_true');parser.add_argument('--last',action='store_true');args=parser.parse_args()
    summary(args.seed)
    if args.gallery:gallery(args.seed,args.last)
