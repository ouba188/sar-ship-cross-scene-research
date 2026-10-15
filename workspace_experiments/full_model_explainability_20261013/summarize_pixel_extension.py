"""Aggregate fixed-context diagnostics and independently check saved readouts."""

from pixel_extension import DEST, OUT, ROOT, read, write, CONDITIONS, identity
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def audit(folder):
    receipt=read(folder/'COMPLETE.json')
    saved=np.load(folder/'predictions.npz')
    p,y,q=saved['probability'],saved['labels'],saved['query_indices']
    frame=pd.read_csv(folder/'selected.csv',dtype={'identity':str},keep_default_na=False)
    old=OUT/'tasks'/receipt['task']
    support=np.load(old/'input/support.npy')
    assert len(q)==len(np.unique(q)) and not np.isin(q,support).any()
    np.testing.assert_allclose(p.sum(-1),1,atol=1e-12,rtol=0)
    original=np.load(old/'predictions.npy',mmap_mode='r')[:,7,q]
    np.testing.assert_array_equal(p[0].argmax(-1),original)
    m=pd.read_csv(folder/'metrics.csv');checks=0
    for row in m[m.subset.isin(['all','sham_valid'])].itertuples():
        use=np.ones(len(y),bool) if row.subset=='all' else saved['sham_valid']
        truth=y[use];pred=p[list(saved['conditions']).index(row.condition),row.draw,use].argmax(-1)
        base=original[row.draw,use]
        ba=100*np.mean([np.mean(pred[truth==c]==c) for c in np.unique(truth)])
        baseba=100*np.mean([np.mean(base[truth==c]==c) for c in np.unique(truth)])
        np.testing.assert_allclose([row.oa,row.ba,row.delta_ba],
                                  [100*np.mean(pred==truth),ba,ba-baseba],atol=1e-10,rtol=0)
        checks+=3
    return dict(task=receipt['task'],metric_checks=checks,original_decisions=int(original.size),
                query_count=len(q),sham_n=int(saved['sham_valid'].sum()),
                ships=int(frame.identity.nunique()),classes=int(len(np.unique(y))),
                feature_error=receipt['feature_max_relative_l2'],probability_error=receipt['probability_max_error'])


def main():
    tasks=[t for t in read(ROOT/'paper_core_freeze_20261013/FROZEN_REFERENCE.json')['tasks']
           if not t['task'].startswith('oss2/')]
    all_metrics=[];all_classes=[];audits=[]
    for cohort,items in [('main',tasks),('sham_supplement',[t for t in tasks if t['task']=='f0921/f0921'])]:
        for task in items:
            folder=(DEST if cohort=='main' else DEST/cohort)/task['task']
            assert (folder/'COMPLETE.json').exists(),str(folder)
            audits.append(dict(cohort=cohort,**audit(folder)))
            all_metrics.append(pd.read_csv(folder/'metrics.csv').assign(cohort=cohort))
            all_classes.append(pd.read_csv(folder/'classes.csv').assign(cohort=cohort))
    m=pd.concat(all_metrics,ignore_index=True);c=pd.concat(all_classes,ignore_index=True)
    m.to_csv(DEST/'metrics.csv',index=False);c.to_csv(DEST/'classes.csv',index=False)
    keys=['cohort','dataset','task','subset','condition']
    tm=m.groupby(keys).mean(numeric_only=True).reset_index()
    tm.to_csv(DEST/'task_summary.csv',index=False)
    summary=tm.groupby(['cohort','dataset','subset','condition']).mean(numeric_only=True).reset_index()
    summary.to_csv(DEST/'summary.csv',index=False)
    contrasts=[]
    index=['cohort','dataset','task','draw','subset']
    for label,left,right,subset in [('target_minus_outer_shuffle','target_shuffle','outer_shuffle','all'),
        ('target_minus_sham_flat','target_flat','sham_flat','sham_valid'),
        ('outer_minus_target_display','outer_display','target_display','all')]:
        l=m[(m.condition==left)&(m.subset==subset)].set_index(index)
        r=m[(m.condition==right)&(m.subset==subset)].set_index(index)
        z=l[['n','classes']].copy();z['contrast']=label
        z['ba_difference']=l.ba-r.ba;z['oa_difference']=l.oa-r.oa
        contrasts.append(z.reset_index())
    contrast=pd.concat(contrasts,ignore_index=True)
    contrast.to_csv(DEST/'paired_contrasts.csv',index=False)
    ct=contrast.groupby(['cohort','dataset','task','contrast']).mean(numeric_only=True).reset_index()
    ct.to_csv(DEST/'task_contrasts.csv',index=False)
    cs=ct.groupby(['cohort','dataset','contrast']).agg(mean_ba=('ba_difference','mean'),
        min_ba=('ba_difference','min'),max_ba=('ba_difference','max'),tasks=('task','size'),
        negative_tasks=('ba_difference',lambda a:int((a<0).sum())),
        positive_tasks=('ba_difference',lambda a:int((a>0).sum()))).reset_index()
    cs.to_csv(DEST/'contrast_summary.csv',index=False)
    coverage=pd.DataFrame(audits);coverage.to_csv(DEST/'coverage.csv',index=False)
    write(DEST/'AUDIT.json',dict(contexts=len(audits),metric_checks=int(coverage.metric_checks.sum()),
        original_decisions=int(coverage.original_decisions.sum()),
        feature_max_relative_l2=float(coverage.feature_error.max()),
        probability_max_error=float(coverage.probability_error.max()),all_passed=True))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    port=ct[(ct.cohort=='main')&(ct.dataset=='self24ports')]
    pivot=port.pivot(index='task',columns='contrast',values='ba_difference').sort_index()
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
    v=pivot['target_minus_outer_shuffle'].to_numpy();yy=np.arange(len(v))
    counts=port[port.contrast=='target_minus_outer_shuffle'].set_index('task').n
    bars=ax.barh(yy,v,color=np.where(v<0,'#347db3','#c07438'))
    ax.bar_label(bars,labels=[f'{a:+.2f}' for a in v],padding=3,fontsize=9)
    ax.axvline(0,color='#777777',lw=.8)
    ax.set_yticks(yy,[f'{a.split("/")[-1]} (n={int(counts[a])})' for a in pivot.index])
    ax.set_xlim(-22,8);ax.set_xlabel('Target - outer shuffle: full-model BA difference (pp)')
    ax.invert_yaxis()
    fig.suptitle('23 frozen full models: target-neighborhood shuffle is more harmful in 22 ports')
    fig.savefig(DEST/'PORT_CONTRASTS.png');fig.savefig(DEST/'PORT_CONTRASTS.pdf');plt.close(fig)
    labels={'target_shuffle':'Target shuffle','outer_shuffle':'Outer shuffle',
            'target_flat':'Target flatten','target_display':'Target display','outer_display':'Outer display'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,ds,title in zip(axes,['self24ports','f0921'],['23 ports (equal-port mean)','F_0921 main sample']):
        part=summary[(summary.cohort=='main')&(summary.dataset==ds)&(summary.subset=='all')].set_index('condition')
        v=part.loc[list(labels),'delta_ba'].to_numpy()
        bars=ax.barh(list(labels.values()),v,color=['#347db3','#9c9c9c','#347db3','#579963','#c07438'])
        ax.bar_label(bars,labels=[f'{a:+.2f}' for a in v],padding=3)
        ax.set_xlim(min(v.min()-4,-2),max(v.max()+4,2));ax.axvline(0,color='#777777',lw=.8)
        ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('BA change from original (pp)')
    fig.savefig(DEST/'DATASET_EFFECTS.png');fig.savefig(DEST/'DATASET_EFFECTS.pdf');plt.close(fig)
    print(summary[(summary.subset=='all')&(summary.cohort=='main')][['dataset','condition','n','ba','delta_ba']].to_string(index=False))
    print(cs.to_string(index=False));print(coverage.to_string(index=False))


if __name__=='__main__':main()
