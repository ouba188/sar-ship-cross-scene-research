"""Paired results and native query panels for the frozen compression experiment."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import probe_compression as run
from analyze_transfer import input_records, margin, CLASSES

base=run.base


def main():
    out=HERE/'compression_analysis'
    out.mkdir(exist_ok=True)
    frames=[]
    for source in ('oss1','oss2'):
        for family in ('compression','compact_coordinates'):
            frames.append(pd.read_csv(HERE/family/source/'metrics.csv'))
    metric=pd.concat(frames,ignore_index=True)
    metric['arm']=metric.method.str.replace(r'_seed\d+$','',regex=True)
    metric.to_csv(out/'all_metrics.csv',index=False)
    metric.groupby(['source','cohort','arm'])[['OA','BA','NLL','Brier']].agg(['mean','std','count']).to_csv(out/'summary.csv')
    pairs=[('direct','reader_span'),('reader_complement','reader_span'),
           ('direct','reader_coords'),('reader_complement','reader_coords'),
           ('pca_coords','reader_coords'),('direct','actual_m2_summary'),
           ('reader_complement','direct')]
    paired,classes,passes,intervals,cases=[],[],[],[],[]
    for source,other in (('oss1','oss2'),('oss2','oss1')):
        direction=f'{source}-to-{other}'
        transfer=base.TRANSFER/direction
        target=base.tr.read_table(transfer/'target_table.csv',labels=True)
        y=target.class_id.to_numpy(int)
        scores={}
        for family in ('compression','compact_coordinates'):
            z=base.tr.load_npz(HERE/family/source/'cross_predictions.npz')
            np.testing.assert_array_equal(z.pop('obs_id'),target.obs_id)
            scores.update(z)
        group=list(target.groupby('pass_group').indices.values())
        counts=np.array([np.bincount(y[ix],minlength=4) for ix in group])
        draws=np.random.default_rng(1026).integers(len(group),size=(2000,len(group)))
        ct=counts[draws].sum(1)
        complete=(ct>0).all(1)
        for after,before in pairs:
            diffs=[]
            for seed in base.SEEDS:
                a,b=scores[f'{after}_mlp_seed{seed}'],scores[f'{before}_mlp_seed{seed}']
                ac,bc=a.argmax(1)==y,b.argmax(1)==y
                d=ac.astype(float)-bc
                diffs.append(d)
                paired.append(dict(source=source,after=after,before=before,seed=seed,
                    rescued=int((ac&~bc).sum()),harmed=int((bc&~ac).sum()),
                    OA_pp=100*d.mean(),BA_pp=100*np.mean([d[y==c].mean() for c in range(4)])))
                for c in range(4):
                    ix=y==c
                    classes.append(dict(source=source,after=after,before=before,seed=seed,class_id=c,
                        class_name=CLASSES[c],n=int(ix.sum()),recall_pp=100*d[ix].mean()))
                for name,ix in target.groupby('pass_group').indices.items():
                    passes.append(dict(source=source,after=after,before=before,seed=seed,pass_group=name,
                        n=len(ix),OA_pp=100*d[ix].mean()))
            d=np.mean(diffs,0)
            sums=np.array([np.bincount(y[ix],weights=d[ix],minlength=4) for ix in group])
            sm=sums[draws].sum(1)
            oa=100*sm.sum(1)/ct.sum(1)
            ba=100*(sm[complete]/ct[complete]).mean(1)
            intervals.append(dict(source=source,after=after,before=before,OA_low=np.quantile(oa,.025),
                OA_high=np.quantile(oa,.975),BA_low=np.quantile(ba,.025),BA_high=np.quantile(ba,.975),
                complete_draws=int(complete.sum()),passes=len(group)))
        combined=pd.read_csv(transfer/'table.csv',dtype={'mmsi':str})
        records=input_records(combined,base.tr.read_json(transfer/'protocol.json'))
        z=base.tr.load_npz(base.TRANSFER/'normalization_repair_20261026'/direction/'seed17/predictions.npz')
        np.testing.assert_array_equal(z['obs_id'],target.obs_id)
        for after,before in [('direct','reader_coords'),('reader_complement','reader_coords')]:
            a,b=scores[f'{after}_mlp_seed17'],scores[f'{before}_mlp_seed17']
            ac,bc=a.argmax(1)==y,b.argmax(1)==y
            selected=[]
            for status,mask in dict(rescue=ac&~bc,harm=bc&~ac,CC=ac&bc,WW=~ac&~bc).items():
                ix=np.flatnonzero(mask)
                if len(ix):
                    j=int(ix[np.argmax(abs(margin(a,y)[ix]-margin(b,y)[ix]))])
                    selected.append((status,j))
            fig,axes=plt.subplots(2,len(selected),figsize=(4*len(selected),8),squeeze=False)
            image_name=f'{source}_{after}_vs_{before}.png'
            for col,(status,j) in enumerate(selected):
                views=base.tr.extract.prepare(records[int(z['query_rows'][j])])
                prov=[]
                for p,v in enumerate(views):
                    ax=axes[p,col]
                    if v is not None:
                        ax.imshow(v['pixels'],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                        if v['masks'][0].any():
                            ax.contour(v['masks'][0],levels=[.5],colors=['#20a780'],linewidths=.7)
                        record=records[int(z['query_rows'][j])][('vv','vh')[p]]
                        prov.append(dict(polarization=('VV','VH')[p],path=record['image'],placement=v['placement']))
                    ax.set_title(f'{status}: {int(target.iloc[j].obs_id)} {CLASSES[y[j]]} {("VV","VH")[p]}\n'
                        f'{CLASSES[b[j].argmax()]} -> {CLASSES[a[j].argmax()]}',fontsize=10)
                    ax.set(xticks=[],yticks=[])
                cases.append(dict(source=source,after=after,before=before,status=status,seed=17,
                    obs_id=int(target.iloc[j].obs_id),index=j,true=int(y[j]),before_pred=int(b[j].argmax()),
                    after_pred=int(a[j].argmax()),before_scores=b[j].tolist(),after_scores=a[j].tolist(),
                    before_margin=float(margin(b,y)[j]),after_margin=float(margin(a,y)[j]),
                    image=image_name,provenance=prov))
            fig.suptitle(f'{direction}: {after} vs {before}, seed17\nSame raw input; projections happen in feature space, not pixel masks.')
            fig.text(.02,.02,'Native160 crop/pad, scale=1. Independent classifiers, not calibrated margins. These heads do not retrieve references.',fontsize=9)
            fig.subplots_adjust(top=.88,bottom=.07,wspace=.1,hspace=.15)
            fig.savefig(out/image_name,dpi=120)
            plt.close(fig)
    for name,rows in [('paired',paired),('by_class',classes),('by_pass',passes),('uncertainty',intervals)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    base.write_json(out/'cases.json',cases)
    base.write_json(out/'receipt.json',dict(panels=4,case_positions=len(cases),unique_queries=len({(r['source'],r['obs_id']) for r in cases}),
        visual_review_pending=True,selection='within each status max absolute uncalibrated margin difference, seed17',
        bootstrap='pass clusters, average paired seed effects first; BA conditional all classes'))
    print(metric[metric.cohort.eq('cross')].groupby(['source','arm'])[['OA','BA']].mean().to_string(),flush=True)
    print(pd.DataFrame(cases)[['source','after','status','obs_id','before_pred','after_pred']].to_string(index=False),flush=True)


if __name__=='__main__':main()
