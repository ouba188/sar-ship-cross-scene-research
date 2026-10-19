"""Native VV/VH panels and additive regional residuals for actual test cases."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
from model import LocalReader

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    f=pd.read_csv(ROOT.parent/'conditional_evidence_pilot/cohort.csv',dtype={'mmsi':str}).set_index('obs_id')
    p=pd.read_csv(ROOT/'predictions.csv',dtype={'mmsi':str})
    b=p.query('arm=="pre" and seed==20261019 and selection=="base"').set_index('obs_id')
    n=p.query('arm=="pre" and seed==20261019 and selection=="source_selected"').set_index('obs_id')
    pair=b[['correct','margin']].join(n[['correct','margin']],rsuffix='_new')
    pair['gain']=pair.margin_new-pair.margin
    views=json.loads((ROOT/'view_snapshot.json').read_text(encoding='utf-8'))
    lookup={v['obs_id']:(i,v) for i,v in enumerate(views)}
    classes=f[['class_id','class_name']].drop_duplicates().set_index('class_id').class_name.to_dict()
    maps=np.load(ROOT/'maps.npy',mmap_mode='r'); masks=np.load(ROOT/'masks.npy',mmap_mode='r')
    alignment=pd.read_csv(ROOT/'crossview_alignment.csv').set_index('obs_id')
    records=[]; used=set()
    mapped=pair.index.to_series().map(alignment.status).eq('target')
    for reason,mask,asc in [('rescue',~pair.correct&pair.correct_new,False),('rescue_aligned',~pair.correct&pair.correct_new&mapped,False),('harm',pair.correct&~pair.correct_new,True),('persistent_correct',pair.correct&pair.correct_new,False),('persistent_wrong',~pair.correct&~pair.correct_new,True)]:
        available=pair[mask&~pair.index.isin(used)].sort_values('gain',ascending=asc)
        if available.empty: continue
        oid=int(available.index[0]); used.add(oid); i,view=lookup[oid]; row=f.loc[oid]; fold=int(n.loc[oid,'fold'])
        net=LocalReader('pre'); net.load_state_dict(torch.load(ROOT/f'model_f{fold}_s20261019_pre.pt',map_location='cpu',weights_only=True)); net.eval()
        cond=np.load(ROOT/f'condition_fold{fold}.npz')['correct'][i:i+1]
        mm=np.array(masks[i:i+1]); conflict=bool(alignment.loc[oid,'disable_new_vh'])
        if conflict: mm[:,1]=0
        with torch.no_grad():
            desc=net.descriptor(torch.from_numpy(np.array(maps[i:i+1],dtype=np.float32)),torch.from_numpy(mm),torch.from_numpy(cond))
        true=int(row.class_id); rival=int(b.loc[oid,'prediction'])
        if rival==true:
            base=np.load(ROOT/f'baseline_fold{fold}.npy',mmap_mode='r')[i].copy(); base[true]=-np.inf; rival=int(base.argmax())
        weight=(net.head.weight[true]-net.head.weight[rival]).detach().numpy().reshape(2,3,12)
        contribution=(desc.numpy()[0]*weight).sum(-1)
        fig,axes=plt.subplots(1,2,figsize=(10,5))
        for polidx,pol in enumerate(('vv','vh')):
            image=np.asarray(Image.open(row['image_'+pol]).convert('L')); ax=axes[polidx]
            ax.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            pri=view['images'][pol]['priors']
            for j,d in enumerate(pri.get('detections',[])):
                if d['confidence']>=.25 and d['shape_usable']:
                    ax.add_patch(Polygon(d['corners'],closed=True,fill=False,edgecolor='#00ad70' if j==pri['target_index'] else '#f1a320',linewidth=1))
            ax.set_xlim(-.5,image.shape[1]-.5); ax.set_ylim(image.shape[0]-.5,-.5)
            ax.set_title(f"{pol.upper()} {image.shape[1]}x{image.shape[0]} | alignment {alignment.loc[oid,'status']}\nresidual true-vs-{classes[rival]}:\nT {contribution[polidx,0]:+.3f} O {contribution[polidx,1]:+.3f} R {contribution[polidx,2]:+.3f}",fontsize=9)
            records.append(dict(reason=reason,obs_id=oid,mmsi=row.mmsi,fold=fold,seed=20261019,pol=pol,path=row['image_'+pol],class_name=row.class_name,base_prediction=classes[int(b.loc[oid,'prediction'])],new_prediction=classes[int(n.loc[oid,'prediction'])],base_margin=b.loc[oid,'margin'],new_margin=n.loc[oid,'margin'],target_contribution=contribution[polidx,0],other_contribution=contribution[polidx,1],rest_contribution=contribution[polidx,2],comparison_rival=classes[rival],alignment=alignment.loc[oid,'status']))
        fig.suptitle(f"{reason} obs {oid} MMSI {row.mmsi} | {row['port']} | {row['time']}\ntrue {row.class_name} | base {classes[int(b.loc[oid,'prediction'])]} {b.loc[oid,'margin']:+.3f} -> local {classes[int(n.loc[oid,'prediction'])]} {n.loc[oid,'margin']:+.3f}",fontsize=10)
        fig.tight_layout(rect=(0,0,1,.87)); fig.savefig(ROOT/f'case_{reason}_{oid}.png',dpi=135); plt.close(fig)
    pd.DataFrame(records).to_csv(ROOT/'case_review.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__': main()
