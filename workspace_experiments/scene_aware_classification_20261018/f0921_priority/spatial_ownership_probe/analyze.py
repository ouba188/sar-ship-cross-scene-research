"""Separate relation changes, decision headroom, and confidence shrinkage."""
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from run import HERE,PREV,cell_fractions,build_kernels,choose_support


def main():
    meta=pd.read_csv(PREV/'cohort.csv');features=dict(np.load(PREV/'features.npz'))
    p=cell_fractions(np.load(PREV/'inputs.npz')['regions'])
    y=meta.label.to_numpy();products=meta['product'].to_numpy();classes=np.unique(y)
    rows=[];cellrows=[];diagnostic_metrics=[]
    for product in sorted(set(products)):
        q=np.flatnonzero(products==product);src=np.flatnonzero(products!=product)
        for seed in range(5):
            s=choose_support(src,y,seed);kernels,_=build_kernels(features,p,s)
            oldsim=kernels['global_grid'][np.ix_(q,s)]
            newsim=kernels['spatial_ownership3'][np.ix_(q,s)]
            b=np.stack([oldsim[:,y[s]==c].mean(1) for c in classes],1)
            n=np.stack([newsim[:,y[s]==c].mean(1) for c in classes],1)
            bc=b-b.mean(1,keepdims=True);nc=n-n.mean(1,keepdims=True)
            bn=np.linalg.norm(bc,axis=1);nn=np.linalg.norm(nc,axis=1)
            scale=nn/np.maximum(bn,1e-12)
            # Label-free diagnostic: same logit norm as new method, old direction.
            matched=bc*scale[:,None]
            prob={name:softmax(score/.1,axis=1) for name,score in [('baseline',b),('ownership',n),('norm_matched_baseline',matched)]}
            for name,pp in prob.items():
                diagnostic_metrics.append(dict(fold=product[-4:],draw=seed,arm=name,
                    oa=float(np.mean(pp.argmax(1)==y[q])),
                    brier=float(np.square(pp-np.eye(len(classes))[y[q]]).sum(1).mean()),
                    nll=float(-np.log(np.maximum(pp[np.arange(len(q)),y[q]],1e-12)).mean())))
            comp=np.einsum('qcr,scr->qsc',np.sqrt(p[q]),np.sqrt(p[s]))
            for j,i in enumerate(q):
                top=np.argsort(-b[j]);gap=float(b[j,top[0]]-b[j,top[1]])
                correction=float(np.abs(nc[j]-bc[j]).max())
                original_margin=float(b[j,y[i]]-max(b[j,c] for c in classes if c!=y[i]))
                new_margin=float(n[j,y[i]]-max(n[j,c] for c in classes if c!=y[i]))
                rows.append(dict(fold=product[-4:],draw=seed,row=int(i),class_raw=meta.iloc[i].class_raw,
                    neighbor_present=bool(meta.iloc[i].image_other_fraction>0),baseline_correct=bool(top[0]==y[i]),
                    baseline_top2_gap=gap,twice_max_centered_logit_change=2*correction,
                    no_flip_bound=bool(2*correction<gap),delta_margin=new_margin-original_margin,
                    logit_norm_ratio=float(scale[j]),
                    logit_direction_cosine=float(nc[j]@bc[j]/max(nn[j]*bn[j],1e-12)),
                    same_compatibility=float(comp[j,y[s]==y[i]].mean()),
                    wrong_compatibility=float(comp[j,y[s]!=y[i]].mean()),
                    brier_old=float(np.square(prob['baseline'][j]-np.eye(len(classes))[y[i]]).sum()),
                    brier_new=float(np.square(prob['ownership'][j]-np.eye(len(classes))[y[i]]).sum()),
                    brier_norm_matched=float(np.square(prob['norm_matched_baseline'][j]-np.eye(len(classes))[y[i]]).sum())))
                for cell in range(4):
                    cellrows.append(dict(fold=product[-4:],draw=seed,row=int(i),cell=cell,
                        same_compatibility=float(comp[j,y[s]==y[i],cell].mean()),
                        wrong_compatibility=float(comp[j,y[s]!=y[i],cell].mean())))
    df=pd.DataFrame(rows);dm=pd.DataFrame(diagnostic_metrics)
    df.to_csv(HERE/'mechanism_by_query.csv',index=False)
    pd.DataFrame(cellrows).to_csv(HERE/'compatibility_by_cell.csv',index=False)
    dm.to_csv(HERE/'confidence_diagnostic_metrics.csv',index=False)
    conf=dm.groupby('arm')[['oa','brier','nll']].mean()
    conf.to_csv(HERE/'confidence_diagnostic_summary.csv')
    pred=pd.read_csv(HERE/'predictions.csv')
    base=pred[pred.arm.eq('global_grid')].set_index(['fold','draw','row'])
    new=pred[pred.arm.eq('spatial_ownership3')].set_index(['fold','draw','row']).loc[base.index]
    met=pd.read_csv(HERE/'metrics.csv')
    draws=met.groupby(['draw','arm'])[['oa','ba','map_r','brier']].mean()
    draws.to_csv(HERE/'draw_means.csv')
    draw_delta=draws.xs('spatial_ownership3',level='arm')-draws.xs('global_grid',level='arm')
    draw_delta.to_csv(HERE/'paired_draw_deltas.csv')
    df['delta_brier']=df.brier_new-df.brier_old
    df.groupby(['neighbor_present','baseline_correct'])[['delta_margin','delta_brier','logit_norm_ratio']].agg(['mean','count']).to_csv(HERE/'mechanism_groups.csv')
    stats=dict(mean_cell_fractions=p.mean((0,1)).tolist(),
        compatibility_same=float(df.same_compatibility.mean()),compatibility_wrong=float(df.wrong_compatibility.mean()),
        compatibility_gap=float((df.same_compatibility-df.wrong_compatibility).mean()),
        no_flip_bound_count=int(df.no_flip_bound.sum()),query_events=len(df),
        median_top2_gap=float(df.baseline_top2_gap.median()),
        median_twice_max_logit_change=float(df.twice_max_centered_logit_change.median()),
        mean_logit_norm_ratio=float(df.logit_norm_ratio.mean()),
        mean_logit_direction_cosine=float(df.logit_direction_cosine.mean()),
        events_margin_improved=int(df.delta_margin.gt(0).sum()),
        events_brier_improved=int(df.delta_brier.lt(0).sum()),
        brier_draw_improvements=int(draw_delta.brier.lt(0).sum()),
        brier_effect_mean=float(draw_delta.brier.mean()),brier_effect_draw_std=float(draw_delta.brier.std()),
        changed_top1_events=int((new.prediction!=base.prediction).sum()),
        changed_ap_events=int((np.abs(new.ap-base.ap)>1e-10).sum()),
        improved_ap_events=int((new.ap-base.ap>1e-10).sum()),
        worsened_ap_events=int((new.ap-base.ap < -1e-10).sum()),
        changed_nearest_support_events=int((new.top_support_row!=base.top_support_row).sum()),
        interpretation='Posthoc norm-matched diagnostic separates confidence magnitude from class-score direction; not a deployment method or target-selected temperature.')
    (HERE/'mechanism_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2));print(conf.to_string());print(df.groupby(['neighbor_present','baseline_correct'])[['delta_margin','delta_brier']].mean().to_string())


if __name__=='__main__':main()
