"""Separate score separability, support-only calibration and open-set utility."""

from study import OUT, METHODS, write
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def weights(y,unknown):
    known=y!=unknown;w=np.zeros(len(y),float)
    classes=np.unique(y[known])
    for c in classes:w[y==c]=1/(len(classes)*(y==c).sum())
    u=(y==unknown)/(y==unknown).sum()
    return known,w,u


def oscr(score,y,pred,unknown):
    known,w,u=weights(y,unknown)
    order=np.argsort(score,kind='stable')
    ends=np.r_[np.flatnonzero(np.diff(score[order]))+1,len(score)]-1
    far=np.r_[0,np.cumsum(u[order])[ends]]
    ccr=np.r_[0,np.cumsum((w*(pred==y))[order])[ends]]
    return 100*float(np.trapezoid(ccr,far))


def oracle95(score,y,unknown):
    known,w,_=weights(y,unknown)
    ss=score[known];ww=w[known];order=np.argsort(ss)
    index=np.searchsorted(np.cumsum(ww[order]),.95)
    threshold=ss[order[min(index,len(ss)-1)]]
    return float(threshold),100*np.mean(score[~known]>threshold)


def readout(score,threshold,y,pred,unknown):
    known,w,u=weights(y,unknown);reject=score>threshold
    before=100*float(w@(pred==y));after=100*float(w@((pred==y)&~reject))
    tpr=100*float(u@reject);frr=100*float(w@reject)
    h=2*after*tpr/(after+tpr) if after+tpr>0 else 0.
    answer=dict(known_ba_before=before,known_ba_after=after,known_correct_lost=before-after,
                unknown_tpr=tpr,known_false_reject=frr,h_score=h,
                unknown_n=int((~known).sum()),known_n=int(known.sum()))
    for rate in (.05,.20,.50):answer[f'utility_{int(rate*100):02d}']=rate*tpr-(1-rate)*(before-after)
    return answer


def main():
    # Distinguish a correct open-set curve implementation from ranking all known samples as correct.
    y=np.array([0,1,2,2]);pred=np.array([0,1,0,0]);score=np.array([0.,0.,1.,1.])
    assert oscr(score,y,pred,2)==100.
    assert oscr(np.zeros(4),y,pred,2)==50.
    assert oscr(-score,y,pred,2)==0.
    assert abs(oscr(score,y,np.array([0,0,0,0]),2)-50.)<1e-12
    records=[];partial=[];support_stats=[]
    for path in sorted((OUT/'tasks').rglob('unknown_*.npz')):
        task=path.parent.relative_to(OUT/'tasks').as_posix();unknown=int(path.stem.split('_')[1])
        z=np.load(path);y=z['labels'];known,w,u=weights(y,unknown)
        assert np.array_equal(z['classes'],np.delete(np.arange(8),unknown))
        assert not (z['support_labels']==unknown).any()
        for draw in range(5):
            sy=z['support_labels'][draw];counts=np.bincount(sy,minlength=8)
            pred=z['classes'][z['probability'][draw].argmax(1)]
            scores=z['score'][draw]
            for mi,method in enumerate(METHODS):
                ss=scores[:,mi]
                auc=100*roc_auc_score(y==unknown,ss,sample_weight=w+u)
                curve=oscr(ss,y,pred,unknown);oracle_threshold,otpr=oracle95(ss,y,unknown)
                for ti,quantile in enumerate((.90,.95)):
                    threshold=float(z['thresholds'][draw,ti,mi])
                    np.testing.assert_allclose(threshold,np.quantile(z['calibration_score'][draw,:,mi],quantile),atol=1e-12)
                    records.append(dict(task=task,unknown=unknown,draw=draw,method=method,quantile=quantile,
                        threshold=threshold,auroc=auc,oscr=curve,oracle95_unknown_tpr=otpr,
                        oracle95_threshold=oracle_threshold,**readout(ss,threshold,y,pred,unknown)))
            # Same actual target known queries: 8-way source reference versus absent-class masking.
            p8=z['eight_probability'][draw];ref=p8.argmax(1)
            oracle=p8.copy();oracle[:,unknown]=-np.inf
            simple=p8.copy();simple[:,counts==0]=-np.inf
            base=100*float(w@(ref==y))
            for mode,pp in [('unmasked',ref),('oracle_absent_only',oracle.argmax(1)),('support_zero_mask',simple.argmax(1))]:
                partial.append(dict(task=task,absent=unknown,draw=draw,mode=mode,
                    ba=100*float(w@(pp==y)),oa=100*np.mean(pp[known]==y[known]),
                    delta_ba=100*float(w@((pp==y).astype(int)-(ref==y))),
                    absent_prediction_rate=100*float(w@(pp==unknown)),
                    present_no_support_classes=int(np.isin(np.unique(y[known]),np.flatnonzero(counts==0)).sum())))
            unsupported=np.isin(y,np.flatnonzero(counts==0))&known
            support_stats.append(dict(task=task,unknown=unknown,draw=draw,
                known_classes_in_query=len(np.unique(y[known])),known_classes_in_support=int((counts>0).sum()),
                unsupported_known_query_n=int(unsupported.sum()),
                unsupported_known_class_mass=100*float(w@unsupported)))
    m=pd.DataFrame(records);p=pd.DataFrame(partial);support=pd.DataFrame(support_stats)
    m.to_csv(OUT/'metrics.csv',index=False);p.to_csv(OUT/'partial_metrics.csv',index=False)
    support.to_csv(OUT/'support_coverage.csv',index=False)
    keys=['task','unknown','draw','quantile']
    fields=['utility_20','auroc','oscr','known_ba_after','unknown_tpr','known_false_reject']
    base=m[m.method=='msp'][keys+fields].rename(columns={k:'msp_'+k for k in fields})
    comparison=m.merge(base,on=keys,validate='many_to_one')
    for col in fields:
        comparison['delta_'+col]=comparison[col]-comparison['msp_'+col]
    comparison.to_csv(OUT/'paired_effects.csv',index=False)
    summary=m.groupby(['task','method','quantile']).mean(numeric_only=True).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    m.groupby(['task','unknown','method','quantile']).mean(numeric_only=True).reset_index().to_csv(OUT/'class_summary.csv',index=False)
    effects=comparison.groupby(['task','method','quantile']).mean(numeric_only=True).reset_index()
    effects.to_csv(OUT/'effect_summary.csv',index=False)
    p.groupby(['task','mode']).mean(numeric_only=True).reset_index().to_csv(OUT/'partial_summary.csv',index=False)
    primary=summary[summary['quantile']==.95]
    task_names=list(primary.task.unique());colors=['#666666','#bb713d','#3788a8','#4c966b']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,col,title in zip(axes,['auroc','utility_20'],
        ['Unknown-class score separation','Net benefit at 20% unknown prevalence']):
        for j,method in enumerate(METHODS):
            table=primary[primary.method==method].set_index('task')
            vals=table.loc[task_names,col].to_numpy()
            ax.bar(np.arange(len(task_names))+(j-1.5)*.2,vals,.2,label=method,color=colors[j])
        ax.set_xticks(np.arange(len(task_names)),[n.removeprefix('ports/') for n in task_names],rotation=20)
        ax.set_title(title);ax.set_ylabel('AUROC (%)' if col=='auroc' else 'Percentage-point utility vs no rejection')
        ax.axhline(50 if col=='auroc' else 0,color='#777777',lw=.8)
    axes[0].legend(ncol=2,fontsize=8,frameon=False)
    fig.suptitle('Frozen-feature pilot: class-held-out readouts, not the full reference model',fontsize=11)
    fig.savefig(OUT/'SEPARATION_AND_UTILITY.png',dpi=180);fig.savefig(OUT/'SEPARATION_AND_UTILITY.pdf');plt.close(fig)
    write(OUT/'ANALYSIS_COMPLETE.json',dict(metric_rows=len(m),partial_rows=len(p),
        contexts=len(support),methods=METHODS,metric_unit='percentage / percentage points',
        primary_quantile=.95,utility_unknown_prevalence=.20,tests=4))
    print(primary[['task','method','auroc','oscr','known_ba_before','known_ba_after','known_false_reject','unknown_tpr','utility_20','oracle95_unknown_tpr']].to_string(index=False))
    print('PARTIAL');print(p.groupby(['task','mode'])[['ba','delta_ba','absent_prediction_rate']].mean().to_string())


if __name__=='__main__':main()
