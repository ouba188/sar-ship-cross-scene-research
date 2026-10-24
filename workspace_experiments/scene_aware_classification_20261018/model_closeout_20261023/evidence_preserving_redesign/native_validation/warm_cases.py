"""Raw-chip review panels for the bounded warm-start correction."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_native as tr

def main():
    f=pd.read_csv(tr.OLD/'table.csv');y=f.class_id.to_numpy(int);te=np.flatnonzero(f.role.eq('evaluation'))
    parent=np.load(HERE/'results/moment_17.npz')['score'];after=np.load(HERE/'results/warm_axis_moment_17.npz')['score']
    changed=te[parent[te].argmax(1)!=after[te].argmax(1)]
    geo=tr.load('warm_geo',tr.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up=pd.read_csv(tr.UP/'table.csv');valid=np.load(tr.UP/'summaries.npz')['valid'];refs=np.load(tr.LOCAL/'scores_layer1.npz')['refs']
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    records=[]
    for j in changed:
        i=int(f.iloc[j].up_index);c=int(y[j]);rival=int(after[j].argmax());obs=int(f.iloc[j].obs_id);rr=refs[i,[c,rival]].ravel()
        fig,axes=plt.subplots(2,5,figsize=(16,7))
        paths=[]
        for p in range(2):
            for col,ri in enumerate([i,*rr]):
                if valid[ri,p]:
                    pix,fp,g,_=geo.native(up,snapshots,int(ri),p)
                    axes[p,col].imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                    for region,color in zip(geo.soft_regions(g)*fp,('#20a780','#d77329','#3287d0')):
                        if region.max()>.5:axes[p,col].contour(region,levels=[.5],colors=[color],linewidths=.5)
                else:axes[p,col].text(.5,.5,'MISSING',ha='center')
                label='QUERY' if col==0 else 'TRUE' if col<3 else 'RIVAL'
                axes[p,col].set_title(f'{("VV","VH")[p]} {label} {int(up.iloc[ri].obs_id)}\n{up.iloc[ri].class_name}',fontsize=8)
                axes[p,col].set(xticks=[],yticks=[])
                paths.append(str(up.iloc[ri]['image_'+('vv','vh')[p]]))
        case='harm' if parent[j].argmax()==c else 'WW_change'
        fig.suptitle(f'warm_axis {case} seed17 query{obs} true{c}: parent {parent[j].argmax()} -> warm {rival}\nFixed rival margin {parent[j,c]-parent[j,rival]:+.5f} -> {after[j,c]-after[j,rival]:+.5f}')
        fig.text(.02,.02,'Native160; no image resize. Parent=M2 moment, not old_full. Only convolution trained; references unchanged.',fontsize=9)
        fig.subplots_adjust(top=.84,bottom=.09,wspace=.12,hspace=.15)
        path=HERE/'analysis'/f'warm_{case}_{obs}.png';fig.savefig(path,dpi=120);plt.close(fig)
        records.append(dict(index=int(j),obs_id=obs,case=case,true_class=c,parent_pred=int(parent[j].argmax()),new_pred=rival,parent_margin=float(parent[j,c]-parent[j,rival]),new_margin=float(after[j,c]-after[j,rival]),reference_obs=up.iloc[rr].obs_id.astype(int).tolist(),paths=paths,figure=str(path)))
    (HERE/'analysis/warm_cases.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(pd.DataFrame(records)[['obs_id','case','parent_margin','new_margin','reference_obs']].to_string(index=False))

if __name__=='__main__':main()
