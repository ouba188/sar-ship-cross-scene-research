"""Score quality, calibration, coverage strata and net utility without target selection."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import OUT, BASE, METHODS, core
from analyze import weights, oscr, oracle95, readout


def all_decisions(z, calibration):
    th=z[calibration+'_thresholds']
    result={m:z['score'][:,i]>th[i] for i,m in enumerate(METHODS)}
    result['none']=np.zeros(len(z['labels']),bool)
    result['source_selected']=result[str(z['selected'])]
    return result


def metrics_for(z, decision, score):
    y=z['labels']; pred=z['pred']; u=int(z['unknown'])
    known,w,wu=weights(y,u)
    # Direct group means independently check the shared metric implementation.
    ba0=np.mean([np.mean(pred[y==c]==c) for c in np.unique(y[known])])*100
    ba1=np.mean([np.mean((pred[y==c]==c)&~decision[y==c]) for c in np.unique(y[known])])*100
    frr=np.mean([np.mean(decision[y==c]) for c in np.unique(y[known])])*100
    tpr=np.mean(decision[~known])*100
    shared=readout(decision.astype(float),.5,y,pred,u)
    np.testing.assert_allclose([ba0,ba1,frr,tpr],[shared['known_ba_before'],shared['known_ba_after'],
                                              shared['known_false_reject'],shared['unknown_tpr']],atol=1e-10)
    answer=dict(**shared,known_oa_before=100*np.mean(pred[known]==y[known]),
                known_oa_after=100*np.mean((pred[known]==y[known])&~decision[known]),
                rejected=int(decision.sum()),known_rejected=int(decision[known].sum()))
    if score is not None:
        answer.update(auroc=100*roc_auc_score(~known,score,sample_weight=w+wu),
                      oscr=oscr(score,y,pred,u),tpr_at_frr5=oracle95(score,y,u)[1])
    else:answer.update(auroc=np.nan,oscr=np.nan,tpr_at_frr5=np.nan)
    return answer


def main():
    contexts=pd.read_csv(OUT/'target_contexts.csv')
    metrics=[]; strata=[]; events=[]; classrows=[]; stored={}; checks=0
    for row in contexts.itertuples():
        z=np.load(OUT/row.path); y=z['labels']; pred=z['pred']; known,w,wu=weights(y,row.outer)
        random=np.load(OUT/'target'/row.task/f'u{row.outer}_d{row.draw}_random.npz')
        original_counts=np.bincount(random['support_labels'],minlength=8)
        current_counts=np.bincount(z['support_labels'],minlength=8)
        base=dict(task=row.task,unknown=row.outer,draw=row.draw,arm=row.arm)
        for calibration in ('source','support'):
            decisions=all_decisions(z,calibration)
            for method,decision in decisions.items():
                effective=str(z['selected']) if method=='source_selected' else method
                score=None if effective=='none' else z['score'][:,METHODS.index(effective)]
                result=metrics_for(z,decision,score);checks+=4
                metrics.append(dict(**base,calibration=calibration,method=method,**result))
                for group,mask in [('originally_missing',known&(original_counts[y]==0)),
                                   ('originally_supported',known&(original_counts[y]>0)),
                                   ('currently_missing',known&(current_counts[y]==0))]:
                    if not mask.any():continue
                    cls=np.unique(y[mask]);conditional=np.zeros(len(y))
                    for c in cls:conditional[y==c]=1/(len(cls)*(y==c).sum())
                    strata.append(dict(**base,calibration=calibration,method=method,group=group,
                                       n=int(mask.sum()),classes=len(cls),frr=100*conditional@decision,
                                       ba=100*conditional@((pred==y)&~decision),
                                       correct_lost=100*conditional@((pred==y)&decision)))
                for c in np.unique(y):
                    mask=y==c
                    classrows.append(dict(**base,calibration=calibration,method=method,class_id=int(c),
                        n=int(mask.sum()),is_unknown=bool(c==row.outer),support_count=int(current_counts[c]),
                        rejection=100*decision[mask].mean(),correct_after=100*((pred[mask]==c)&~decision[mask]).mean()))
            for reference in ('support_knn','source_knn','ova','fixed_union'):
                old=decisions[reference]; new=decisions['joint']
                freed=known&old&~new; added=known&~old&new
                events.append(dict(**base,calibration=calibration,reference=reference,
                    known_freed=int(freed.sum()),correct_known_freed=int((freed&(pred==y)).sum()),
                    new_known_reject=int(added.sum()),new_correct_known_reject=int((added&(pred==y)).sum()),
                    extra_unknown_detected=int((~known&~old&new).sum()),unknown_newly_missed=int((~known&old&~new).sum()),
                    delta_known_ba=100*w@((pred==y)*(old.astype(int)-new)),
                    delta_unknown_tpr=100*wu@(new.astype(int)-old),
                    original_missing_freed=int((freed&(original_counts[y]==0)).sum())))
            if row.arm=='random':
                for method in ('joint','support_knn','source_knn','ova','source_selected'):
                    decision=decisions[method]
                    for pi in (.05,.20):
                        # Preserve the shared-query dependence across all held classes and draws.
                        contribution=100*(pi*wu*decision-(1-pi)*w*(pred==y)*decision)
                        key=(row.task,calibration,method,pi)
                        stored.setdefault(key,[]).append((y.copy(),contribution))
    m=pd.DataFrame(metrics);s=pd.DataFrame(strata);e=pd.DataFrame(events);c=pd.DataFrame(classrows)
    m.to_csv(OUT/'metrics.csv',index=False);s.to_csv(OUT/'strata.csv',index=False)
    e.to_csv(OUT/'events.csv',index=False);c.to_csv(OUT/'classes.csv',index=False)
    fields=['auroc','oscr','tpr_at_frr5','known_ba_before','known_ba_after','known_oa_before',
            'known_oa_after','known_false_reject','unknown_tpr','utility_05','utility_20']
    summary=m.groupby(['task','arm','calibration','method'])[fields].mean().reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    m.groupby(['task','arm','calibration','method','draw'])[fields].mean().reset_index().to_csv(OUT/'draws.csv',index=False)
    m.groupby(['task','arm','calibration','method','unknown'])[fields].mean().reset_index().to_csv(OUT/'unknown_classes.csv',index=False)
    s.groupby(['task','arm','calibration','method','group'])[['frr','ba','correct_lost','n','classes']].mean().reset_index().to_csv(OUT/'strata_summary.csv',index=False)
    e.groupby(['task','arm','calibration','reference']).sum(numeric_only=True).reset_index().to_csv(OUT/'event_totals.csv',index=False)
    source=pd.read_csv(OUT/'source_metrics.csv')
    source.groupby(['role','method'])[fields[:1]+['known_false_reject','unknown_tpr','utility_05','utility_20']].mean().reset_index().to_csv(OUT/'source_summary.csv',index=False)
    bootstrap=[]
    for (task,calibration,method,pi),items in stored.items():
        y=items[0][0]
        for yy,_ in items:np.testing.assert_array_equal(yy,y)
        a=np.stack([v for _,v in items]).mean(0)
        rng=np.random.default_rng(20261014); estimates=[]
        for _ in range(400):
            ix=np.concatenate([rng.choice(np.flatnonzero(y==cl),(y==cl).sum(),replace=True) for cl in np.unique(y)])
            estimates.append(a[ix].sum())
        bootstrap.append(dict(task=task,calibration=calibration,method=method,prevalence=pi,
                              utility=float(a.sum()),low=float(np.quantile(estimates,.025)),
                              high=float(np.quantile(estimates,.975)),unit='class-stratified query ship; fixed supports and fits'))
    pd.DataFrame(bootstrap).to_csv(OUT/'query_ship_bootstrap.csv',index=False)
    primary=summary[(summary.arm=='random')&(summary.calibration=='source')]
    print('PRIMARY SOURCE CALIBRATION\n'+primary[['task','method','auroc','known_false_reject','unknown_tpr','utility_05','utility_20']].to_string(index=False))
    print('TARGET LOO\n'+summary[(summary.arm=='random')&(summary.calibration=='support')][['task','method','known_false_reject','unknown_tpr','utility_20']].to_string(index=False))
    print('SOURCE\n'+pd.read_csv(OUT/'source_summary.csv').to_string(index=False))
    figure(summary)
    core.write(OUT/'ANALYSIS_COMPLETE.json',dict(metric_rows=len(m),strata_rows=len(s),event_rows=len(e),
               independent_group_metric_checks=checks,query_bootstrap_repeats=400))


def figure(summary):
    tasks=['oss2','f0921','ports/Port Klang']; methods=['source_knn','support_knn','ova','fixed_union','joint']
    colors=['#69757c','#cc6849','#9473b4','#289d86','#337dbc']
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,field,cal,title in zip(axs,['auroc','utility_20','utility_20'],['source','source','support'],
                                ['Ranking of truly held-out classes','Net utility: source threshold','Net utility: support LOO threshold']):
        for i,method in enumerate(methods):
            df=summary[(summary.arm=='random')&(summary.calibration==cal)&(summary.method==method)].set_index('task')
            ax.bar(np.arange(3)+(i-2)*.16,df.loc[tasks,field],.15,label=method,color=colors[i])
        ax.set_xticks(np.arange(3),['OSS2','F_0921','Port Klang']);ax.set_title(title,fontsize=10)
        ax.axhline(50 if field=='auroc' else 0,color='gray',lw=.8)
        ax.set_ylabel('AUROC (%)' if field=='auroc' else 'U(20%), percentage points')
    axs[0].legend(fontsize=8,frameon=False)
    fig.suptitle('Frozen-feature open-set pilot, B=64; not the full paper model',fontsize=12)
    fig.savefig(OUT/'KNOWNNESS_RESULTS.png',dpi=170);fig.savefig(OUT/'KNOWNNESS_RESULTS.pdf');plt.close(fig)


if __name__=='__main__':main()
