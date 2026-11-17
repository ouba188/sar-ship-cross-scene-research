"""Paired class/scene effects and native-grid evidence panels; no model execution."""
from pathlib import Path
import csv
import html
import json
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import run

ROOT,WORK=run.ROOT,run.WORK
sys.path.insert(0,str(WORK/'sar_m123_standalone'))
from sar_m123.tokens import soft_regions
native=run.load('gallery_native',WORK/'sar_m123_standalone/sar_m123/input.py')
NAMES=['Bulk','General cargo','Chemical/product tanker','Container']
ARMS=['closed_full','global_full','transfer_full','transfer_pcea']
rows=run.read(WORK/'m1_coverage_20261101/manifest.json')
groups=run.read(WORK/'m1_coverage_20261101/groups.json')
records={r['index']:r for r in run.read(ROOT/'records_local.json')}
assets=dict(np.load(ROOT/'recovered_inputs.npz'))
alookup={int(i):j for j,i in enumerate(assets['ids'])}
bundles={(split,seed):dict(np.load(ROOT/f'{split}_s{seed}.npz')) for split in ('validation','evaluation') for seed in (17,29,43)}


def csv_write(path,values):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)


def stats():
    grouped=[];flips=[];overall=[]
    contrasts=[('global_full','closed_full'),('transfer_full','closed_full'),('transfer_full','global_full'),('transfer_pcea','transfer_full')]
    for (split,seed),a in bundles.items():
        y,ids,affected=a['label'],a['ids'],a['affected']
        for new,old in contrasts:
            pn,po=a[new].argmax(1),a[old].argmax(1)
            for group,values in [('class',y),('pass',np.array([rows[i]['pass_group'] for i in ids])),('scope',np.where(affected,'recovered','unchanged'))]:
                for value in np.unique(values):
                    sel=values==value
                    paired=run.paired(y[sel],a[new][sel],a[old][sel])
                    grouped.append(dict(split=split,seed=seed,new=new,old=old,group=group,value=str(value),n=int(sel.sum()),
                                        oa_delta=float(100*((pn[sel]==y[sel]).mean()-(po[sel]==y[sel]).mean())),**paired))
            overall.append(dict(split=split,seed=seed,new=new,old=old,**run.paired(y,a[new],a[old]),
                affected_logit_max=float(abs(a[new][affected]-a[old][affected]).max())))
            for j in np.flatnonzero(pn!=po):
                flips.append(dict(split=split,seed=seed,index=int(ids[j]),id=rows[ids[j]]['id'],label=int(y[j]),
                    new=new,old=old,pred_old=int(po[j]),pred_new=int(pn[j]),
                    rescued=bool(po[j]!=y[j] and pn[j]==y[j]),harmed=bool(po[j]==y[j] and pn[j]!=y[j])))
    csv_write(ROOT/'groups.csv',grouped)
    csv_write(ROOT/'contrasts.csv',overall)
    run.write(ROOT/'flips_local.json',flips)


def gallery():
    folder=ROOT/'gallery';folder.mkdir(exist_ok=True)
    chosen=[('evaluation',29,'002912','global rescue, extra regional margin'),
            ('evaluation',29,'002831','region-specific rescue'),
            ('evaluation',43,'000260','region-specific rescue'),
            ('evaluation',17,'000230','regional harm'),
            ('evaluation',43,'005566','regional harm'),
            ('evaluation',17,'000277','global rescue revoked by regions'),
            ('evaluation',17,'000316','persistent error, changed rival')]
    a=bundles['evaluation',17]
    stable=[int(i) for j,i in enumerate(a['ids']) if a['affected'][j] and all(a[k][j].argmax()==a['label'][j] for k in ARMS)]
    chosen.append(('evaluation',17,rows[stable[0]]['id'][-6:],'persistent correct'))
    a=bundles['validation',29]
    old,new=a['closed_full'].argmax(1),a['transfer_full'].argmax(1)
    for name,mask in [('validation rescue',(old!=a['label'])&(new==a['label'])),('validation harm',(old==a['label'])&(new!=a['label']))]:
        ix=np.flatnonzero(mask)
        if len(ix):chosen.append(('validation',29,rows[a['ids'][ix[0]]]['id'][-6:],name))
    exported=[]
    for split,seed,short,reason in chosen:
        a=bundles[split,seed]
        j=next(j for j,i in enumerate(a['ids']) if rows[i]['id'].endswith('_'+short))
        idx=int(a['ids'][j]);r=records[idx];y=int(a['label'][j]);target=run.POLS.index(r['target_pol'])
        recovered=assets['pixels'][alookup[idx]];foot=assets['footprint'][alookup[idx]];g=assets['geometry'][alookup[idx]]
        corners=np.asarray(r['corners'])
        fig,axs=plt.subplots(2,4,figsize=(15,7.5))
        for p,pol in enumerate(run.POLS):
            path=r['donor_png'] if pol==r['donor_pol'] else r['target_png']
            raw=np.array(Image.open(path).convert('L'))
            axs[0,p].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axs[0,p].add_patch(Polygon(corners,closed=True,fill=False,edgecolor='#e69f00' if p==target else '#009e73',linewidth=1))
            axs[0,p].set_title(pol.upper()+' raw: '+('transferred OBB' if p==target else 'donor OBB'))
        axs[0,2].imshow(recovered,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        mask=soft_regions(g)*foot
        rgb=np.zeros((160,160,4));colors=np.array([[.2,.7,.5],[.9,.65,.1],[.25,.5,.9]])
        rgb[...,:3]=np.einsum('rhw,rc->hwc',mask,colors)
        rgb[...,3]=.35*mask.sum(0).clip(0,1)
        axs[0,2].imshow(rgb,interpolation='nearest');axs[0,2].set_title(r['target_pol'].upper()+' native160: body/end/side')
        scores={}
        text=[]
        for arm in ARMS:
            z=a[arm][j];others=z.copy();others[y]=-np.inf
            margin=float(z[y]-others.max());pred=int(z.argmax())
            scores[arm]=dict(pred=pred,margin=margin,logits=z.tolist())
            text.append(arm+'\n'+NAMES[pred]+f' | margin {margin:+.4f}')
        axs[0,3].text(0,.95,'\n\n'.join(text),va='top',fontsize=10,transform=axs[0,3].transAxes)
        original=a['closed_full'][j].copy();original[y]=-np.inf;rival=int(original.argmax())
        refs=[]
        for k,cls in enumerate((y,rival)):
            legal=[i for i in groups['references'] if rows[i]['label']==cls and rows[i]['mmsi']!=rows[idx]['mmsi']
                   and '_'.join(rows[i]['pass_group'].split('_')[:2])!='_'.join(rows[idx]['pass_group'].split('_')[:2])]
            # Fixed legal first source reference, not posthoc highest influence.
            ri=legal[0];refs.append(dict(index=ri,id=rows[ri]['id'],label=cls))
            for p,pol in enumerate(run.POLS):
                ax=axs[1,2*k+p];spec=rows[ri].get(pol)
                if spec is not None:
                    view=native.prepare_view(spec);ax.imshow(view['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                ax.set_title(('Same-class' if k==0 else 'Rival')+f' ref {rows[ri]["id"][-6:]} '+pol.upper(),fontsize=10)
        for ax in axs.flat:ax.set_axis_off()
        fig.suptitle(f'{short} | {NAMES[y]} | {split} seed{seed}\n{reason}; donor {r["donor_pol"].upper()} confidence {r["donor_confidence"]:.3f}',fontsize=13)
        fig.text(.5,.015,'Raw grids unresized; native160 crop/pad. Green=donor OBB; amber=transferred OBB. References are legal examples, not influence rankings.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.93))
        stem=f'{split}_s{seed}_{short}'
        fig.savefig(folder/(stem+'.png'),dpi=110);fig.savefig(folder/(stem+'.svg'));plt.close(fig)
        exported.append(dict(split=split,seed=seed,id=r['id'],index=idx,reason=reason,label=y,
                             target_pol=r['target_pol'],scores=scores,references=refs,panel=stem+'.png'))
    run.write(folder/'cases_local.json',exported)
    cards=''.join(f'<section><h2>{html.escape(c["id"])} · {html.escape(c["reason"])}</h2><img src="{c["panel"]}" alt="paired evidence panel"></section>' for c in exported)
    (folder/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>双极化OBB迁移对照</title><style>body{font-family:Arial,sans-serif;max-width:1500px;margin:24px auto;color:#222}img{width:100%;height:auto}section{border-top:1px solid #ddd;margin-top:24px}h2{font-size:18px}</style><h1>双极化OBB迁移：真实输入与分类对照</h1><p>固定权重，closed/global/transfer分别为关闭、全图回退、迁移框；PCEA为新增调制开关。展示正负及持续正确/错误案例，不代表总体比例。SVG文件可编辑。</p>'+cards+'</html>',encoding='utf-8')
    print('GALLERY',len(exported),folder/'index.html')


if __name__=='__main__':
    stats();gallery()
