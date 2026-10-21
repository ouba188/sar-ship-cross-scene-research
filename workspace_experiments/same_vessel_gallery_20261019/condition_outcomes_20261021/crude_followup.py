"""Bounded post-hoc crude-tanker diagnostic, not an inference module."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_conditions import OUT, ROOT, OLD, corr, diagnostic, cases


def main():
    p=pd.read_csv(OUT/'pair_factors.csv')
    q=p[p.class_name.eq('crude_oil_tanker')].copy().reset_index(drop=True)
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    columns=['vv_contrast','vh_contrast']
    y=q.delta_margin.to_numpy();pred=np.empty(len(q));null=np.empty(len(q));metrics=[];coef=[]
    for fold in range(3):
        test=q.fold.eq(fold);train=~test
        model=make_pipeline(StandardScaler(),Ridge(alpha=10))
        model.fit(q.loc[train,columns],y[train]);pred[test]=model.predict(q.loc[test,columns]);null[test]=y[train].mean()
        metrics.append(dict(fold=str(fold),ships=int(test.sum()),mse=np.mean((pred[test]-y[test])**2),
            constant_mse=np.mean((null[test]-y[test])**2),rho=corr(pred[test],y[test])))
        coef.append(dict(fold=fold,vv_weight=model[-1].coef_[0],vh_weight=model[-1].coef_[1]))
    mse=np.mean((pred-y)**2);nmse=np.mean((null-y)**2)
    metrics.append(dict(fold='pooled',ships=len(q),mse=mse,constant_mse=nmse,rho=corr(pred,y),error_reduction=1-mse/nmse))
    pd.DataFrame(metrics).to_csv(OUT/'crude_diagnostic_metrics.csv',index=False)
    pd.DataFrame(coef).to_csv(OUT/'crude_diagnostic_coefficients.csv',index=False)
    q[['pair_id','fold','delta_margin']].assign(predicted=pred,constant=null).to_csv(OUT/'crude_diagnostic_predictions.csv',index=False)
    cohort=pd.read_csv(ROOT/'conditional_evidence_pilot/cohort.csv')
    score_rows=[]
    for fold in range(3):
        for seed in range(3):
            with np.load(OLD/f'scores_f{fold}_baseline_s{seed}.npz') as d:
                rr,ss=d['rows'],d['scores']
            for j,i in enumerate(rr):
                row=cohort.iloc[i];c=int(row.class_id)
                if row.class_name!='crude_oil_tanker':continue
                w=ss[j].copy();w[c]=-np.inf
                score_rows.append(dict(obs_id=row.obs_id,seed=seed,true_score=ss[j,c],wrong_score=w.max(),rival=w.argmax(),pred=ss[j].argmax()))
    sr=pd.DataFrame(score_rows)
    av=sr.groupby('obs_id')[['true_score','wrong_score']].mean()
    for name in ('true_score','wrong_score'):
        q['delta_'+name]=q.obs_id_b.map(av[name])-q.obs_id_a.map(av[name])
    np.testing.assert_allclose(q.delta_true_score-q.delta_wrong_score,q.delta_margin,atol=1e-6)
    decomposition=[]
    for feat in columns:
        for outcome in ('delta_margin','delta_true_score','delta_wrong_score'):
            decomposition.append(dict(feature=feat,outcome=outcome,rho=corr(q[feat],q[outcome])))
    pd.DataFrame(decomposition).to_csv(OUT/'crude_score_decomposition.csv',index=False)
    q['mean_contrast_delta']=q[columns].mean(axis=1)
    q['contrast_direction']=np.where(q.mean_contrast_delta>=0,'increase','decrease')
    q['margin_direction']=np.where(q.delta_margin>=0,'increase','decrease')
    q.groupby(['contrast_direction','margin_direction']).size().rename('ships').reset_index().to_csv(OUT/'crude_quadrants.csv',index=False)
    q.to_csv(OUT/'crude_pair_details.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,feat in zip(axs,columns):
        for fold in range(3):
            s=q[q.fold.eq(fold)];ax.scatter(s[feat],s.delta_margin,s=15,alpha=.7,label=f'fold {fold}')
        ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7)
        ax.set_xlabel(f'{feat}: B - A');ax.set_ylabel('True-vs-best-wrong margin: B - A')
        ax.set_title('100 crude-oil-tanker pairs');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'crude_contrast_relationship.png',dpi=140);plt.close(fig)
    # Two outcome-selected explanatory examples, not a frequency estimate.
    views=pd.read_csv(OLD/'data_review/main_pair_views.csv').set_index(['obs_id','pol'])
    from PIL import Image
    selected=[]
    for label,g in q.groupby('contrast_direction'):
        match=g[g.margin_direction.ne(label)]
        r=match.loc[match.mean_contrast_delta.abs().idxmax()]
        fig,axs=plt.subplots(2,2,figsize=(10,8))
        fig.suptitle(f'Post-hoc example: {r.pair_id}; contrast {label}, margin {r.margin_direction}\n'
                     f'mean contrast change={r.mean_contrast_delta:+.3f}, margin change={r.delta_margin:+.3f}')
        for a,side in enumerate(('a','b')):
            oid=int(r['obs_id_'+side]);outcome=sr[sr.obs_id.eq(oid)].sort_values('seed')
            for b,pol in enumerate(('vv','vh')):
                v=views.loc[(oid,pol)]
                with Image.open(v.image_path) as im: raw=np.asarray(im.convert('L'))
                axs[a,b].imshow(raw,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axs[a,b].set_title(f'{pol.upper()} {oid} {r["port_"+side]} | native {v.width}x{v.height}\n'
                    f'pred seeds={outcome.pred.tolist()}, truth=5; rel={v.relative_axis_deg:.1f}, inc={v.incidence:.1f}',fontsize=9)
        fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/f'crude_case_{label}_{r.pair_id}.png',dpi=120);plt.close(fig)
        selected.append(dict(selection='contrast_'+label,**r.to_dict()))
    pd.DataFrame(selected).to_csv(OUT/'crude_selected_cases.csv',index=False)
    receipt=json.loads((OUT/'RECEIPT.json').read_text(encoding='utf-8'))
    receipt['diagnostic_ridge_fits']=12
    receipt['posthoc_crude_diagnostic_fits']=3
    (OUT/'RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False));print(pd.DataFrame(decomposition).to_string(index=False))


if __name__=='__main__':
    main()
