"""Mechanism and task summaries from already frozen predictions."""

import run_targets as t
from run_source import np,pd,c,OUT,metrics,probabilities
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def loss_rows(base_probability,prior,residual,gate,y):
    p=.5*(base_probability+softmax(prior+residual*gate,axis=1))
    return np.square(p-np.eye(8)[y]).sum(1)


def alignment():
    rows,matrices=[],[]
    for ds,directory in t.folders():
        task=directory.name
        meta=pd.read_csv(c.read_json(directory/'PREDICTIONS_FROZEN.json')['target_manifest'],usecols=['class_id'])
        for draw in range(5):
            for budget in c.BUDGETS:
                old=c.archive(directory/f'{draw}_{budget}.npz')
                new=c.archive(t.CACHE/ds/task/f'{draw}_{budget}.npz')
                oof=c.archive(OUT/'target_oof'/ds/task/f'{draw}_{budget}.npz')
                y=meta.iloc[old['query_ids']].class_id.to_numpy(int)
                oy=oof['labels']
                pq=new['prior']
                dq=old['expert_raw']-pq
                bq=softmax(old['base'],axis=1)
                rawq=loss_rows(bq,pq,dq,np.ones(8),y)
                rawo=loss_rows(oof['base_probability'],oof['prior'],oof['residual'],np.ones(8),oy)
                for cls in range(8):
                    gate=np.ones(8)
                    gate[cls]=oof['gate_joint'][cls]
                    gainq=rawq-loss_rows(bq,pq,dq,gate,y)
                    gaino=rawo-loss_rows(oof['base_probability'],oof['prior'],oof['residual'],gate,oy)
                    actual=np.array([gainq[y==k].mean() if (y==k).any() else np.nan for k in range(8)])
                    estimated=np.array([gaino[oy==k].mean() if (oy==k).any() else np.nan for k in range(8)])
                    common=np.isfinite(actual)&np.isfinite(estimated)
                    key=dict(dataset=ds,task=task,draw=draw,budget=budget,updated_class=cls,
                        gamma=gate[cls],support_n=int((old['support_labels']==cls).sum()))
                    rows.append(dict(**key,estimated_gain=float(np.nanmean(estimated)),actual_gain=float(np.nanmean(actual)),
                        matched_class_actual_gain=float(actual[common].mean()),
                        matched_class_estimated_gain=float(estimated[common].mean()),
                        own_gain=float(actual[cls]),
                        cross_gain=float(np.nanmean(np.delete(actual,cls)))))
                    for truth in range(8):
                        matrices.append(dict(**key,true_class=truth,estimated_gain=estimated[truth],actual_gain=actual[truth]))
        print('UTILITY DIAGNOSED',ds,task,flush=True)
    f=pd.DataFrame(rows)
    f.to_csv(OUT/'utility_alignment.csv',index=False)
    pd.DataFrame(matrices).to_csv(OUT/'utility_matrix.csv',index=False)
    summary=[]
    for (ds,budget),g in f.groupby(['dataset','budget']):
        active=g[abs(g.gamma-1)>1e-8]
        for common in (False,True):
            x=active['matched_class_estimated_gain' if common else 'estimated_gain']
            y=active['matched_class_actual_gain' if common else 'actual_gain']
            rho=float(spearmanr(x,y).statistic) if x.nunique()>1 and y.nunique()>1 else np.nan
            summary.append(dict(dataset=ds,budget=budget,matched_classes=common,n_updates=len(active),
                spearman=rho,sign_agreement=float(np.mean(np.sign(x)==np.sign(y))),
                always_positive_accuracy=float((y>0).mean()),
                utility_auc=float(roc_auc_score(y>0,x)) if (y>0).nunique()==2 else np.nan,
                estimated_positive=int((x>0).sum()),actual_positive=int((y>0).sum()),
                false_positive=int(((x>0)&(y<0)).sum())))
    pd.DataFrame(summary).to_csv(OUT/'utility_alignment_summary.csv',index=False)


def markdown(frame,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,r in frame.iterrows():
        lines.append('| '+' | '.join(f'{r[k]:.4f}' if isinstance(r[k],(float,np.floating)) else str(r[k]) for k in columns)+' |')
    return '\n'.join(lines)


def summarize():
    f=pd.read_csv(OUT/'target_metrics.csv')
    pair=[]
    keys=['dataset','task','budget','draw','layer']
    for reference in ('raw','global','self','count','initial','simple'):
        joined=f[f.arm=='joint'].merge(f[f.arm==reference],on=keys,suffixes=('','_ref'),validate='one_to_one')
        joined['reference']=reference
        for k in ('oa','ba','macro_brier'):
            joined['difference_'+k]=joined[k]-joined[k+'_ref']
        pair.append(joined[keys+['reference','difference_oa','difference_ba','difference_macro_brier']])
    pair=pd.concat(pair,ignore_index=True)
    pair.to_csv(OUT/'target_comparisons.csv',index=False)
    port=pair.groupby(['dataset','task','budget','layer','reference']).agg(
        delta_oa=('difference_oa','mean'),delta_ba=('difference_ba','mean'),
        delta_macro_brier=('difference_macro_brier','mean'),
        positive_draws=('difference_ba',lambda x:int((x>1e-10).sum())),
        negative_draws=('difference_ba',lambda x:int((x< -1e-10).sum()))).reset_index()
    port.to_csv(OUT/'target_comparison_ports.csv',index=False)
    summary=port.groupby(['dataset','budget','layer','reference']).agg(
        delta_oa=('delta_oa','mean'),delta_ba=('delta_ba','mean'),delta_macro_brier=('delta_macro_brier','mean'),
        positive_tasks=('delta_ba',lambda x:int((x>1e-10).sum())),
        negative_tasks=('delta_ba',lambda x:int((x< -1e-10).sum()))).reset_index()
    summary.to_csv(OUT/'target_comparison_summary.csv',index=False)
    cp=pd.read_csv(OUT/'target_classes.csv')
    classport=cp.groupby(['dataset','task','budget','layer','arm','class_id']).agg(
        recall=('recall','mean'),delta_ba=('delta_ba','mean'),rescue_ba=('rescue_ba','mean'),harm_ba=('harm_ba','mean')).reset_index()
    classport.to_csv(OUT/'class_ports.csv',index=False)
    summary_class=[]
    for (ds,b,layer,arm,cls),g in classport.groupby(['dataset','budget','layer','arm','class_id']):
        n=23 if ds=='self24ports' else 1
        summary_class.append(dict(dataset=ds,budget=b,layer=layer,arm=arm,class_id=cls,
            recall_present_tasks=g.recall.mean(),delta_ba=g.delta_ba.sum()/n,
            rescue_ba=g.rescue_ba.sum()/n,harm_ba=g.harm_ba.sum()/n))
    pd.DataFrame(summary_class).to_csv(OUT/'class_summary.csv',index=False)
    whole=pd.read_csv(OUT/'target_summary.csv')
    check=pd.DataFrame(summary_class).groupby(['dataset','budget','layer','arm']).delta_ba.sum().sort_index()
    np.testing.assert_allclose(check,whole.set_index(['dataset','budget','layer','arm']).delta_ba.sort_index(),atol=1e-10)
    small=f[(f.dataset=='self24ports')&(f.task!='Mombasa')].groupby(
        ['budget','layer','arm'])[['delta_oa','delta_ba','delta_macro_brier']].mean().reset_index()
    small.to_csv(OUT/'without_mombasa.csv',index=False)
    text=['# 逐类更新效用：完整数值表','单位：OA/BA为百分数，差值为百分点；宏Brier差负值为改善。23港先平均抽样再等权港口。',
        '## 完整模型',markdown(whole[whole.layer=='fusion'],['dataset','budget','arm','oa','ba','delta_oa','delta_ba','delta_macro_brier']),
        '## 联合版本相对替换对照',markdown(summary[summary.layer=='fusion'],
        ['dataset','budget','reference','delta_oa','delta_ba','delta_macro_brier','positive_tasks','negative_tasks']),
        '## 逐港联合版本相对原模型',markdown(port[(port.layer=='fusion')&(port.reference=='raw')],
        ['dataset','task','budget','delta_oa','delta_ba','positive_draws','negative_draws'])]
    (OUT/'TABLES_ZH.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for j,(ds,title) in enumerate([('self24ports','23 ports'),('f0921','F_0921'),('oss2','OpenSARShip 2.0')]):
        for arm,color in [('joint','#277f91'),('global','#b05d32'),('count','#795493')]:
            g=whole[(whole.dataset==ds)&(whole.layer=='fusion')&(whole.arm==arm)].sort_values('budget')
            for row,m in enumerate(('delta_oa','delta_ba')):
                axes[row,j].plot(range(3),g[m],marker='o',label=arm,color=color)
                axes[row,j].set_xticks(range(3),['16','64','128'])
        axes[0,j].set_title(title)
        axes[1,j].set_xlabel('Labeled support budget')
    for i,row in enumerate(axes):
        for ax in row:
            ax.axhline(0,color='#555555',linewidth=.8)
            ax.grid(axis='y',alpha=.2)
            ax.spines[['top','right']].set_visible(False)
        row[0].set_ylabel(('OA' if i==0 else 'BA')+' change (pp)')
    axes[0,0].legend(frameon=False)
    fig.savefig(OUT/'utility_adaptation.png',dpi=150)
    plt.close(fig)
    print(summary[(summary.layer=='fusion')&(summary.reference.isin(['raw','global']))].to_string(index=False),flush=True)


if __name__=='__main__':
    with c.threadpool_limits(limits=2):
        summarize()
        alignment()
