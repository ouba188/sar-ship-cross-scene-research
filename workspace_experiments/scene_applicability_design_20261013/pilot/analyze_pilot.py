"""Paired task, probability-risk, retrieval and independent solver evaluation."""
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from run_pilot import OUT, ARMS, metadata, kernel, write


def scores(y, p):
    present=np.unique(y); hit=p.argmax(1)==y
    brier=np.sum((p-np.eye(8)[y])**2,axis=1)
    nll=-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1))
    return dict(oa=100*hit.mean(),ba=100*np.mean([hit[y==c].mean() for c in present]),
        brier=brier.mean(),macro_brier=np.mean([brier[y==c].mean() for c in present]),
        nll=nll.mean(),macro_nll=np.mean([nll[y==c].mean() for c in present]))


def main():
    d=metadata(); plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    f=np.load(OUT/'encoded.npz'); rows=f['rows']; z=np.c_[f['r'],f['t']]
    metrics=[]; changes=[]; cov=[]; checks=0; solver_checks=0; maxerr=0.; cases=[]
    for e in plans:
        q=np.array(e['query']); y=d.iloc[q].class_id.to_numpy(int); qi=np.searchsorted(rows,q)
        cov.append(dict(episode=e['episode'],port=e['port'],acquisition=e['acquisition'],queries=len(q),
            ships=d.iloc[q].mmsi.nunique(),class_counts=json.dumps(d.iloc[q].class_id.value_counts().sort_index().to_dict())))
        for budget in [16,64]:
            p=np.load(OUT/'predictions'/f"episode_{e['episode']:02d}_b{budget}.npz")
            probs=p['probabilities']
            np.testing.assert_allclose(probs.sum(-1),1,atol=1e-12)
            assert np.isfinite(probs).all()
            for modei,mode in enumerate(['expert','compact_full']):
                prior=p['source_only'][modei]
                metrics.append(dict(episode=e['episode'],port=e['port'],budget=budget,draw=-1,cell='source_only',arm='source_only',mode=mode,**scores(y,prior)))
            for draw in range(5):
                for ci,cell in enumerate(['same','other']):
                    for ai,arm in enumerate(ARMS):
                        for mi,mode in enumerate(['expert','compact_full']):
                            pp=probs[draw,ci,ai,mi]; ref=probs[draw,ci,0,mi]
                            m=scores(y,pp)
                            np.testing.assert_allclose([m['oa'],m['ba']], [100*accuracy_score(y,pp.argmax(1)),100*recall_score(y,pp.argmax(1),labels=np.unique(y),average='macro',zero_division=0)],atol=1e-10)
                            checks+=1
                            base=dict(episode=e['episode'],port=e['port'],budget=budget,draw=draw,cell=cell,arm=arm,mode=mode)
                            metrics.append(dict(**base,**m))
                            old=ref.argmax(1)==y; new=pp.argmax(1)==y
                            for c in range(8):
                                mask=y==c; rescue=np.sum(~old&new&mask); harm=np.sum(old&~new&mask)
                                changes.append(dict(**base,class_id=c,n=int(mask.sum()),rescued=int(rescue),harmed=int(harm),
                                    delta_oa=100*(rescue-harm)/len(y),delta_ba=100*(rescue-harm)/mask.sum()/len(np.unique(y)) if mask.any() else 0))
                            if arm=='scene_memory' and mode=='compact_full' and cell=='other' and draw==0:
                                for i in np.flatnonzero(old!=new)[:12]:
                                    cases.append(dict(**base,row=int(q[i]),ship=d.iloc[q[i]].mmsi,class_id=int(y[i]),
                                        old_prediction=int(ref[i].argmax()),new_prediction=int(pp[i].argmax()),
                                        outcome='rescue' if new[i] else 'harm'))
                    # Independent sklearn solve for all kernels in one draw/scene/budget.
                    if draw==0 and ci==1:
                        s=p['support_rows'][draw,ci]; si=np.searchsorted(rows,s); sy=d.iloc[s].class_id.to_numpy(int)
                        desired=2*np.eye(8)[sy]-1; weights=compute_sample_weight('balanced',sy)
                        bb=KernelRidge(alpha=8.,kernel='precomputed').fit(f['b'][si]@f['b'][si].T+1,desired-f['pb'][si],sample_weight=weights)
                        bp=softmax(f['pb'][qi]+bb.predict(f['b'][qi]@f['b'][si].T+1),axis=1)
                        for ai,arm in enumerate(ARMS):
                            r=None if arm=='original' else f['rel_'+arm]
                            ks=kernel(z[si],z[si],None if r is None else r[si],None if r is None else r[si])
                            kq=kernel(z[qi],z[si],None if r is None else r[qi],None if r is None else r[si])
                            fitted=KernelRidge(alpha=8.,kernel='precomputed').fit(ks,desired-f['pd'][si],sample_weight=weights)
                            ep=softmax(f['pd'][qi]+fitted.predict(kq),axis=1)
                            actual=.5*(bp+ep); expected=probs[draw,ci,ai,1]
                            maxerr=max(maxerr,float(np.max(np.abs(actual-expected))))
                            np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
                            solver_checks+=1
    m=pd.DataFrame(metrics); t=pd.DataFrame(changes)
    m.to_csv(OUT/'metrics.csv',index=False); t.to_csv(OUT/'transitions.csv',index=False)
    pd.DataFrame(cov).to_csv(OUT/'coverage.csv',index=False); pd.DataFrame(cases).to_csv(OUT/'cases.csv',index=False)
    keys=['episode','port','budget','draw','cell','mode']
    actual=m[m.arm!='source_only']; ref=actual[actual.arm=='original'].drop(columns='arm')
    joined=actual.merge(ref,on=keys,suffixes=('','_ref'),validate='many_to_one')
    values=['oa','ba','brier','macro_brier','nll','macro_nll']
    for v in values: joined['delta_'+v]=joined[v]-joined[v+'_ref']
    joined.to_csv(OUT/'effects.csv',index=False)
    delta=['delta_'+v for v in values]
    joined.groupby(['port','budget','cell','mode','arm'])[delta].mean().to_csv(OUT/'port_effects.csv')
    joined.groupby(['budget','cell','mode','arm'])[delta].mean().to_csv(OUT/'effect_summary.csv')
    actual.groupby(['budget','cell','mode','arm'])[values].mean().to_csv(OUT/'absolute_summary.csv')
    drawing=joined.groupby(['budget','cell','mode','arm','draw'])[delta].mean().reset_index()
    drawing.groupby(['budget','cell','mode','arm']).agg(ba_mean=('delta_ba','mean'),ba_support_sd=('delta_ba','std'),
        ba_positive_draws=('delta_ba',lambda a:int((a>0).sum())),oa_mean=('delta_oa','mean')).to_csv(OUT/'support_variability.csv')
    byscene=joined.groupby(['episode','port','budget','cell','mode','arm'])[delta].mean().reset_index()
    byscene.to_csv(OUT/'scene_effects.csv',index=False)
    t.groupby(['budget','cell','mode','arm','class_id'])[['rescued','harmed','delta_oa','delta_ba']].mean().to_csv(OUT/'class_effects.csv')
    sums=t.groupby(keys+['arm'])[['delta_oa','delta_ba']].sum().reset_index()
    expected=joined[keys+['arm','delta_oa','delta_ba']]
    j=sums.merge(expected,on=keys+['arm'],suffixes=('_sum','_direct'),validate='one_to_one')
    np.testing.assert_allclose(j.delta_oa_sum,j.delta_oa_direct,atol=1e-10)
    np.testing.assert_allclose(j.delta_ba_sum,j.delta_ba_direct,atol=1e-10)
    ret=pd.read_csv(OUT/'retrieval.csv')
    # Class then scene means; scenes/ports have equal weight. Missing gallery classes stay explicit.
    rc=ret.groupby(['episode','port','arm','class_id']).map_at_r.mean().reset_index()
    rc.to_csv(OUT/'retrieval_classes.csv',index=False)
    rs=rc.groupby(['episode','port','arm']).map_at_r.mean().reset_index()
    rs.to_csv(OUT/'retrieval_scene.csv',index=False)
    rs.groupby('arm').map_at_r.mean().to_csv(OUT/'retrieval_summary.csv')
    write('VERIFIED.json',dict(independent_metric_rows=checks,independent_kernel_ridge_cases=solver_checks,
        max_probability_error=maxerr,class_decomposition_rows=len(j),
        toy_checks=['explicit_joint_feature_map','positive_semidefinite','rho0_original','hand_calculated_MAP_R']))
    qq=np.unique(np.concatenate([e['query'] for e in plans]))
    write('SCOPE.json',dict(unique_queries=len(qq),unique_query_ships=d.iloc[qq].mmsi.nunique(),
        query_class_counts=d.iloc[qq].class_id.value_counts().sort_index().to_dict(),
        ports=sorted(d.iloc[qq].port.unique()),scenes=len(plans),query_events=sum(len(e['query']) for e in plans),
        retrieval_eligible_queries=int(ret[ret.arm=='original'].map_at_r.notna().sum()),
        retrieval_query_events=len(ret[ret.arm=='original'])))
    print('COMPACT FULL EFFECTS\n',joined[joined['mode']=='compact_full'].groupby(['budget','cell','arm'])[delta].mean().to_string(),flush=True)
    print('RETRIEVAL\n',rs.groupby('arm').map_at_r.mean().to_string(),flush=True)
    print('VERIFIED',checks,solver_checks,maxerr,flush=True)


if __name__=='__main__':
    main()
