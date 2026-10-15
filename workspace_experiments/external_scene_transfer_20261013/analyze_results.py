"""Read paired external predictions, data strata, and independent solver checks."""
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score,recall_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import OUT,DATASETS,MODES,metadata,write


def main():
    plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    frames={n:metadata(n) for n in DATASETS}
    records=[];classrows=[];transitions=[];products=[];coverage=[];supportinfo=[];checked=0
    for e in plans:
        z=np.load(OUT/'predictions'/f"episode_{e['episode']:03d}.npz")
        d=frames[e['target']];q=z['query_rows'];y=d.iloc[q].class_id.to_numpy(int);present=np.unique(y)
        p=z['probabilities'];prior=z['source_only']
        assert np.isfinite(p).all() and np.isfinite(prior).all()
        np.testing.assert_allclose(p.sum(-1),1,atol=1e-12);np.testing.assert_allclose(prior.sum(-1),1,atol=1e-12)
        base={k:e[k] for k in ['episode','study','target','budget','acquisition']}
        coverage.append(dict(**base,queries=len(q),ships=d.iloc[q].identity.nunique(),query_acquisitions=d.iloc[q].loc[d.iloc[q].acquisition_known.astype(bool),'acquisition'].nunique(),
            query_classes=','.join(map(str,present)),min_class_n=int(pd.Series(y).value_counts().min())))
        cells=[(-1,'source_only',prior)]+[(draw,arm,p[draw,ai]) for draw in range(5) for ai,arm in enumerate(e['arms'])]
        for draw,arm,probs in cells:
            for mi,mode in enumerate(MODES):
                pred=probs[mi].argmax(-1);ok=pred==y
                oa=100*float(ok.mean());ba=100*float(np.mean([ok[y==c].mean() for c in present]))
                np.testing.assert_allclose([oa,ba],[100*accuracy_score(y,pred),100*recall_score(y,pred,labels=present,average='macro',zero_division=0)],atol=1e-10)
                checked+=1
                records.append(dict(**base,draw=draw,arm=arm,mode=mode,n=len(q),oa=oa,ba=ba))
                for c in range(8):
                    m=y==c
                    classrows.append(dict(**base,draw=draw,arm=arm,mode=mode,class_id=c,n=int(m.sum()),
                        recall=100*float(ok[m].mean()) if m.any() else np.nan,
                        oa_contribution=100*float(ok[m].sum())/len(q),
                        ba_contribution=100*float(ok[m].mean())/len(present) if m.any() else 0.))
                if e['target']=='oss2':
                    for pt in ['GRDH','SLC']:
                        m=d.iloc[q].product_type.eq(pt).to_numpy()
                        if not m.any():continue
                        yy=y[m];pp=pred[m]
                        products.append(dict(**base,draw=draw,arm=arm,mode=mode,product_type=pt,n=int(m.sum()),
                            oa=100*float(np.mean(yy==pp)),ba=100*recall_score(yy,pp,labels=np.unique(yy),average='macro',zero_division=0)))
        for draw in range(5):
            for ai,(arm,sn) in enumerate(zip(e['arms'],e['support_datasets'])):
                s=z['support_rows'][draw,ai];ds=frames[sn].iloc[s]
                supportinfo.append(dict(**base,draw=draw,arm=arm,support_dataset=sn,
                    classes=ds.class_id.nunique(),class_counts=json.dumps(ds.class_id.value_counts().sort_index().to_dict()),
                    acquisitions=ds.loc[ds.acquisition_known.astype(bool),'acquisition'].nunique(),
                    same_query_acquisition_fraction=float(ds.acquisition.isin(set(d.iloc[q].loc[d.iloc[q].acquisition_known.astype(bool),'acquisition'])).mean()),
                    slc_fraction=float(ds.product_type.eq('SLC').mean()) if sn=='oss2' else 0.))
            for mi,mode in enumerate(MODES):
                contrasts=[(arm+'_minus_source',prior[mi],p[draw,ai,mi]) for ai,arm in enumerate(e['arms'])]
                if e['study']=='acquisition_transfer':contrasts.append(('other_minus_same',p[draw,0,mi],p[draw,1,mi]))
                if e['study']=='support_exchange':
                    same=DATASETS.index(e['target']);contrasts.append(('foreign_minus_local_support',p[draw,same,mi],p[draw,1-same,mi]))
                for contrast,a,b in contrasts:
                    ca=a.argmax(-1)==y;cb=b.argmax(-1)==y
                    for c in range(8):
                        m=y==c
                        transitions.append(dict(**base,draw=draw,mode=mode,contrast=contrast,class_id=c,n=int(m.sum()),
                            rescued=int((~ca&cb&m).sum()),lost=int((ca&~cb&m).sum()),
                            delta_oa=100*float(cb[m].sum()-ca[m].sum())/len(q),
                            delta_ba=100*float(cb[m].mean()-ca[m].mean())/len(present) if m.any() else 0.))
    df=pd.DataFrame(records);df.to_csv(OUT/'metrics.csv',index=False)
    cd=pd.DataFrame(classrows);cd.to_csv(OUT/'classes.csv',index=False)
    pd.DataFrame(products).to_csv(OUT/'product_metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    pd.DataFrame(supportinfo).to_csv(OUT/'support_conditions.csv',index=False)
    tr=pd.DataFrame(transitions);tr.to_csv(OUT/'transitions.csv',index=False)
    keys=['episode','study','target','budget','acquisition','draw','mode']
    effects=tr.groupby(keys+['contrast'])[['delta_oa','delta_ba']].sum().reset_index()
    effects.to_csv(OUT/'effects.csv',index=False)
    summary=effects.groupby(['study','target','budget','mode','contrast'])[['delta_oa','delta_ba']].mean().reset_index()
    summary.to_csv(OUT/'effect_summary.csv',index=False)
    means=effects.groupby(['study','target','budget','mode','contrast','draw'])[['delta_oa','delta_ba']].mean().reset_index()
    means.groupby(['study','target','budget','mode','contrast']).agg(mean_oa=('delta_oa','mean'),sd_support_oa=('delta_oa','std'),
        mean_ba=('delta_ba','mean'),sd_support_ba=('delta_ba','std'),positive_support_draws_ba=('delta_ba',lambda v:int((v>0).sum()))).to_csv(OUT/'support_variability.csv')
    absolute=df.groupby(['study','target','budget','mode','arm'])[['oa','ba']].mean().reset_index()
    absolute.to_csv(OUT/'absolute_summary.csv',index=False)
    ck=['study','target','budget','mode','contrast','class_id']
    tr.groupby(ck)[['delta_oa','delta_ba','rescued','lost']].mean().to_csv(OUT/'class_effect_summary.csv')
    branchkeys=[k for k in keys if k!='mode']+['arm']
    gain=(df[df['mode']=='dual_late'].set_index(branchkeys)[['oa','ba']]-df[df['mode']=='r50_late'].set_index(branchkeys)[['oa','ba']]).reset_index()
    gain.groupby(['study','target','budget','arm'])[['oa','ba']].mean().to_csv(OUT/'branch_summary.csv')
    productdf=pd.DataFrame(products)
    productdf.groupby(['study','target','budget','mode','arm','product_type'])[['oa','ba']].mean().to_csv(OUT/'product_summary.csv')
    # Pair budgets only when queries actually match, including acquisition enrollment.
    bm=['study','target','acquisition','draw','mode','arm']
    paired=df[df.budget==64].merge(df[df.budget==16],on=bm,suffixes=('_64','_16'),validate='one_to_one')
    qmap={(e['study'],e['target'],e['acquisition'],e['budget']):e['query'] for e in plans}
    for e in plans:
        if e['budget']==64:
            k=(e['study'],e['target'],e['acquisition'])
            assert qmap[k+(16,)]==qmap[k+(64,)]
    paired['delta_oa']=paired.oa_64-paired.oa_16;paired['delta_ba']=paired.ba_64-paired.ba_16
    paired.groupby(['study','target','mode','arm'])[['delta_oa','delta_ba']].mean().to_csv(OUT/'budget_summary.csv')
    # Independent predictions cover cross-dataset indexing and both support budgets.
    banks={n:np.load(OUT/(n+'_encoded.npz')) for n in DATASETS}
    solver_cases=0;maxerr=0.
    for study in ['dataset_adaptation','support_exchange','acquisition_transfer']:
        for target in DATASETS:
            for budget in [16,64]:
                e=next(e for e in plans if e['study']==study and e['target']==target and e['budget']==budget)
                z=np.load(OUT/'predictions'/f"episode_{e['episode']:03d}.npz")
                fq=banks[target];qi=np.searchsorted(fq['rows'],z['query_rows'])
                for ai,sn in enumerate(e['support_datasets']):
                    fs=banks[sn];sr=z['support_rows'][0,ai];si=np.searchsorted(fs['rows'],sr);sy=frames[sn].iloc[sr].class_id.to_numpy(int)
                    w=compute_sample_weight('balanced',sy);desired=2*np.eye(8)[sy]-1;outs=[]
                    for sq,ss,pk in [(fq['b'][qi],fs['b'][si],'pb'),(np.c_[fq['r'][qi],fq['t'][qi]],np.c_[fs['r'][si],fs['t'][si]],'pd')]:
                        fit=KernelRidge(alpha=8.,kernel='precomputed').fit(ss@ss.T+1,desired-fs[pk][si],sample_weight=w)
                        outs.append(softmax(fq[pk][qi]+fit.predict(sq@ss.T+1),axis=1))
                    actual=.5*(outs[0]+outs[1]);expected=z['probabilities'][0,ai,2]
                    maxerr=max(maxerr,float(np.max(np.abs(actual-expected))))
                    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-9);solver_cases+=1
    # Class contributions must reconstruct every reported whole-query metric.
    sums=cd.groupby(keys+['arm'])[['oa_contribution','ba_contribution']].sum().reset_index()
    compared=sums.merge(df,on=keys+['arm'],validate='one_to_one')
    np.testing.assert_allclose(compared.oa_contribution,compared.oa,atol=1e-10)
    np.testing.assert_allclose(compared.ba_contribution,compared.ba,atol=1e-10)
    scope=[]
    for study in ['dataset_adaptation','support_exchange','acquisition_transfer']:
        for target in DATASETS:
            for budget in [16,64]:
                es=[e for e in plans if e['study']==study and e['target']==target and e['budget']==budget]
                q=np.unique(np.concatenate([e['query'] for e in es]));d=frames[target].iloc[q]
                scope.append(dict(study=study,target=target,budget=budget,episodes=len(es),queries=len(q),vessels=d.identity.nunique(),
                    acquisitions=d.loc[d.acquisition_known.astype(bool),'acquisition'].nunique(),
                    class_counts={int(k):int(v) for k,v in d.class_id.value_counts().sort_index().items()}))
    write('SCOPE.json',scope)
    write('VERIFIED.json',dict(independent_metric_rows=checked,independent_solver_cases=solver_cases,
        max_probability_error=maxerr,class_sum_checks=len(compared),main_model_changed=False))
    render(absolute,summary)
    write('COMPLETE.json',dict(status='complete',datasets=DATASETS,episodes=len(plans),no_new_encoder_training=True,
        new_source_geometry=1,source_linear_heads=3,main_model_changed=False,scope=scope))
    print('ADAPTATION\n',absolute[(absolute.study=='dataset_adaptation')&(absolute['mode']=='dual_late')].to_string(index=False))
    print('PAIRED EFFECTS\n',summary[(summary['mode']=='dual_late')&summary.contrast.isin(['target_support_minus_source','other_minus_same','foreign_minus_local_support'])].to_string(index=False))
    print('VERIFIED',checked,solver_cases,maxerr)


def render(absolute,summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,target in zip(axes,DATASETS):
        for mode,color in [('r50_late','#287f83'),('dual_late','#b74962')]:
            s=absolute[(absolute.study=='dataset_adaptation')&(absolute.target==target)&(absolute['mode']==mode)]
            baseline=float(s[(s.budget==16)&(s.arm=='source_only')].ba.iloc[0])
            values=[baseline]+[float(s[(s.budget==b)&(s.arm=='target_support')].ba.iloc[0]) for b in [16,64]]
            ax.plot([0,1,2],values,marker='o',label='R50' if mode=='r50_late' else 'R50 + R18',color=color)
        ax.set_xticks([0,1,2],['Source only','16 supports','64 supports']);ax.set_ylabel('BA (%)')
        ax.set_title('24-port source -> '+('F_0921' if target=='f0921' else 'OpenSARShip 2.0'))
        ax.legend();ax.grid(alpha=.15)
    fig.savefig(OUT/'cross_dataset_adaptation.png',dpi=170);fig.savefig(OUT/'cross_dataset_adaptation.svg');plt.close(fig)


if __name__=='__main__':main()
