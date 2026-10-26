"""Paired fixed-cohort analysis and native VV/VH review panels."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run
from analyze_transfer import input_records, margin, CLASSES
base=run.base


def main():
    out=HERE/'analysis'
    out.mkdir(exist_ok=True)
    metrics=pd.concat([pd.read_csv(HERE/s/'metrics.csv') for s in ('oss1','oss2')])
    metrics['arm']=metrics.method.str.replace(r'_seed\d+$','',regex=True)
    metrics.groupby(['source','cohort','arm'])[['OA','BA','NLL','Brier']].agg(['mean','std']).to_csv(out/'summary.csv')
    paired,classes,intervals,cases=[],[],[],[]
    for source,other in [('oss1','oss2'),('oss2','oss1')]:
        cross=base.TRANSFER/f'{source}-to-{other}'
        table=base.tr.read_table(cross/'target_table.csv',labels=True)
        y=table.class_id.to_numpy(int)
        z=base.tr.load_npz(HERE/source/'cross_predictions.npz')
        np.testing.assert_array_equal(z['obs_id'],table.obs_id)
        groups=list(table.groupby('pass_group').indices.values())
        counts=np.array([np.bincount(y[ix],minlength=4) for ix in groups])
        draws=np.random.default_rng(1026).integers(len(groups),size=(2000,len(groups)))
        ct=counts[draws].sum(1)
        complete=(ct>0).all(1)
        combined=pd.read_csv(cross/'table.csv',dtype={'mmsi':str})
        records=input_records(combined,base.tr.read_json(cross/'protocol.json'))
        cached=base.tr.load_npz(run.REPAIR/cross.name/'seed17/predictions.npz')
        np.testing.assert_array_equal(cached['obs_id'],table.obs_id)
        for arm in ('raw_summary','anchored_raw_summary'):
            effects=[]
            for seed in base.SEEDS:
                b=z[f'raw_only_seed{seed}']
                a=z[f'{arm}_seed{seed}']
                bc,ac=b.argmax(1)==y,a.argmax(1)==y
                d=ac.astype(float)-bc
                effects.append(d)
                paired.append(dict(source=source,arm=arm,seed=seed,rescue=int((ac&~bc).sum()),harm=int((bc&~ac).sum()),OA_pp=100*d.mean(),BA_pp=100*np.mean([d[y==c].mean() for c in range(4)])))
                for c in range(4):
                    classes.append(dict(source=source,arm=arm,seed=seed,class_name=CLASSES[c],n=int((y==c).sum()),recall_pp=100*d[y==c].mean()))
            d=np.mean(effects,0)
            sums=np.array([np.bincount(y[ix],weights=d[ix],minlength=4) for ix in groups])
            sm=sums[draws].sum(1)
            oa=100*sm.sum(1)/ct.sum(1)
            ba=100*(sm[complete]/ct[complete]).mean(1)
            intervals.append(dict(source=source,arm=arm,OA_low=np.quantile(oa,.025),OA_high=np.quantile(oa,.975),BA_low=np.quantile(ba,.025),BA_high=np.quantile(ba,.975)))
            a,b=z[f'{arm}_seed17'],z['raw_only_seed17']
            ac,bc=a.argmax(1)==y,b.argmax(1)==y
            selected=[]
            for state,mask in dict(rescue=ac&~bc,harm=bc&~ac,CC=ac&bc,WW=~ac&~bc).items():
                ids=np.flatnonzero(mask)
                if len(ids):
                    selected.append((state,int(ids[np.argmax(abs(margin(a,y)[ids]-margin(b,y)[ids]))])))
            fig,axes=plt.subplots(2,len(selected),figsize=(4*len(selected),8),squeeze=False)
            name=f'{source}_{arm}.png'
            for col,(state,j) in enumerate(selected):
                record=records[int(cached['query_rows'][j])]
                views=base.tr.extract.prepare(record)
                provenance=[]
                for p,v in enumerate(views):
                    ax=axes[p,col]
                    if v is not None:
                        ax.imshow(v['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                        if v['masks'][0].any():
                            ax.contour(v['masks'][0],levels=[.5],colors=['#20a780'],linewidths=.7)
                        provenance.append(dict(pol=('VV','VH')[p],path=record[('vv','vh')[p]]['image'],placement=v['placement']))
                    ax.set_title(f'{state}: {table.iloc[j].obs_id} {CLASSES[y[j]]} {("VV","VH")[p]}\n{CLASSES[b[j].argmax()]} -> {CLASSES[a[j].argmax()]}',fontsize=10)
                    ax.set(xticks=[],yticks=[])
                cases.append(dict(source=source,arm=arm,status=state,obs_id=int(table.iloc[j].obs_id),seed=17,true=int(y[j]),before_pred=int(b[j].argmax()),after_pred=int(a[j].argmax()),before_margin=float(margin(b,y)[j]),after_margin=float(margin(a,y)[j]),image=name,provenance=provenance))
            fig.suptitle(f'{source}->{other}: {arm} vs raw_only, seed17')
            fig.text(.02,.02,'Native160 crop/pad, fixed grayscale. Feature-readout comparison; no reference retrieval at this stage.',fontsize=9)
            fig.subplots_adjust(top=.89,bottom=.07,hspace=.14,wspace=.12)
            fig.savefig(out/name,dpi=120)
            plt.close(fig)
    for name,rows in [('paired',paired),('by_class',classes),('uncertainty',intervals)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    base.write_json(out/'cases.json',cases)
    base.write_json(out/'receipt.json',dict(panels=4,case_positions=len(cases),raw_review_pending=True,bootstrap='2000 pass-cluster resamples of seed-mean paired effects; BA conditional all4classes'))
    print(metrics[metrics.cohort.eq('cross')].groupby(['source','arm'])[['OA','BA']].mean().to_string())
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__=='__main__':main()
