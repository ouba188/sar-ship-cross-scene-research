"""Paired expansion effects, including scene means and pooled observations."""
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score,recall_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.class_weight import compute_sample_weight
from expand import OUT,PILOT,ARMS,metadata,kernel,write
from analyze_pilot import scores

VALUES=['oa','ba','brier','macro_brier','nll','macro_nll']
DELTA=['delta_'+v for v in VALUES]


def summarize(df,name):
    group=['cohort','subset','budget','cell','mode','arm']
    # Average draws inside scene, scenes inside port, then ports equally.
    scenes=df.groupby(group+['port','episode'])[DELTA].mean().reset_index()
    ports=scenes.groupby(group+['port'])[DELTA].mean().reset_index()
    ports.to_csv(OUT/(name+'_ports.csv'),index=False)
    scenes.to_csv(OUT/(name+'_scenes.csv'),index=False)
    ports.groupby(group)[DELTA].mean().to_csv(OUT/(name+'_summary.csv'))
    dr=df.groupby(group+['port','draw'])[DELTA].mean().reset_index()
    dr=dr.groupby(group+['draw'])[DELTA].mean().reset_index()
    dr.groupby(group).agg(ba_mean=('delta_ba','mean'),ba_support_sd=('delta_ba','std'),
        ba_positive_draws=('delta_ba',lambda x:int((x>1e-12).sum())),oa_mean=('delta_oa','mean'),
        oa_positive_draws=('delta_oa',lambda x:int((x>1e-12).sum()))).to_csv(OUT/(name+'_draws.csv'))
    return ports


def main():
    d=metadata();plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    pilot=json.loads((PILOT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    pilot_roles=np.unique(np.concatenate([e['query']+e['gallery']+sum([s for a in e['draws'] for ss in a.values() for s in ss],[]) for e in pilot]))
    seen_roles=set(d.iloc[pilot_roles].mmsi)
    with np.load(OUT/'encoded.npz') as cached:
        f={key:cached[key] for key in cached.files}
    rows=f['rows'];z=np.c_[f['r'],f['t']]
    records=[];transitions=[];events=[];ship_events=[];independent=0;solver_errors=[];checked_ports=set()
    for e in plans:
        q=np.array(e['query']);qi=np.searchsorted(rows,q);y=d.iloc[q].class_id.to_numpy(int)
        masks={'all_queries':np.ones(len(q),bool),'pilot_unseen_ships':np.array(e['pilot_unseen_ship_mask'],bool),
               'pilot_all_roles_unseen':~d.iloc[q].mmsi.isin(seen_roles).to_numpy()}
        for b in [16,64]:
            p=np.load(OUT/'predictions'/f"episode_{e['episode']:02d}_b{b}.npz");probs=p['probabilities']
            assert np.isfinite(probs).all();np.testing.assert_allclose(probs.sum(-1),1,atol=1e-12)
            for draw in range(5):
                for ci,cell in enumerate(['same','other']):
                    for ai,arm in enumerate(ARMS):
                        for mi,mode in enumerate(['expert','compact_full']):
                            for subset,mask in masks.items():
                                if not mask.any():continue
                                yy=y[mask]; pp=probs[draw,ci,ai,mi,mask]; ref=probs[draw,ci,0,mi,mask]
                                m=scores(yy,pp);old=scores(yy,ref)
                                np.testing.assert_allclose([m['oa'],m['ba']],[100*accuracy_score(yy,pp.argmax(1)),
                                    100*recall_score(yy,pp.argmax(1),labels=np.unique(yy),average='macro',zero_division=0)],atol=1e-10)
                                independent+=1
                                base=dict(episode=e['episode'],port=e['port'],cohort=e['cohort'],subset=subset,
                                    budget=b,draw=draw,cell=cell,arm=arm,mode=mode)
                                ca=ref.argmax(1)==yy;cb=pp.argmax(1)==yy
                                records.append(dict(**base,n=len(yy),classes=len(np.unique(yy)),correct=int(cb.sum()),
                                    ref_correct=int(ca.sum()),**m,**{'delta_'+k:m[k]-old[k] for k in VALUES}))
                                for c in np.unique(yy):
                                    cm=yy==c;rescue=int(np.sum(~ca&cb&cm));harm=int(np.sum(ca&~cb&cm))
                                    transitions.append(dict(**base,class_id=int(c),n=int(cm.sum()),rescued=rescue,harmed=harm,
                                        delta_oa=100*(rescue-harm)/len(yy),delta_ba=100*(rescue-harm)/cm.sum()/len(np.unique(yy))))
                            if mode=='compact_full' and arm=='scene_memory':
                                ca=probs[draw,ci,0,mi].argmax(1)==y;cb=probs[draw,ci,ai,mi].argmax(1)==y
                                for j,row in enumerate(q):
                                    ship_events.append(dict(episode=e['episode'],port=e['port'],cohort=e['cohort'],budget=b,draw=draw,
                                        cell=cell,row=int(row),ship=d.iloc[row].mmsi,class_id=int(y[j]),
                                        new_query_ship=bool(e['pilot_unseen_ship_mask'][j]),new_role_ship=bool(masks['pilot_all_roles_unseen'][j]),
                                        old_correct=float(ca[j]),new_correct=float(cb[j])))
                                for j in np.flatnonzero(ca!=cb):
                                    events.append(dict(episode=e['episode'],port=e['port'],cohort=e['cohort'],budget=b,draw=draw,
                                        cell=cell,row=int(q[j]),ship=d.iloc[q[j]].mmsi,class_id=int(y[j]),
                                        new_ship=bool(e['pilot_unseen_ship_mask'][j]),outcome='rescue' if cb[j] else 'harm',
                                        old_pred=int(probs[draw,ci,0,mi,j].argmax()),new_pred=int(probs[draw,ci,ai,mi,j].argmax())))
            if e['cohort']=='new_acquisition' and (e['port'],b) not in checked_ports:
                checked_ports.add((e['port'],b))
                for ci in range(2):
                    s=p['support_rows'][0,ci];si=np.searchsorted(rows,s);sy=d.iloc[s].class_id.to_numpy(int)
                    wanted=2*np.eye(8)[sy]-1;w=compute_sample_weight('balanced',sy)
                    bb=KernelRidge(alpha=8.,kernel='precomputed').fit(f['b'][si]@f['b'][si].T+1,wanted-f['pb'][si],sample_weight=w)
                    bp=softmax(f['pb'][qi]+bb.predict(f['b'][qi]@f['b'][si].T+1),axis=1)
                    for ai,arm in enumerate(ARMS):
                        r=None if arm=='original' else f['rel_'+arm]
                        ks=kernel(z[si],z[si],None if r is None else r[si],None if r is None else r[si])
                        kq=kernel(z[qi],z[si],None if r is None else r[qi],None if r is None else r[si])
                        ee=KernelRidge(alpha=8.,kernel='precomputed').fit(ks,wanted-f['pd'][si],sample_weight=w)
                        value=.5*(bp+softmax(f['pd'][qi]+ee.predict(kq),axis=1))
                        expected=probs[0,ci,ai,1];error=float(np.max(np.abs(value-expected)))
                        np.testing.assert_allclose(value,expected,atol=1e-10,rtol=1e-10);solver_errors.append(error)
    df=pd.DataFrame(records);df.to_csv(OUT/'metrics.csv',index=False)
    tr=pd.DataFrame(transitions);tr.to_csv(OUT/'transitions.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'changed_cases.csv',index=False)
    sd=pd.DataFrame(ship_events);sd.to_csv(OUT/'ship_events.csv',index=False)
    sr=[]
    for subset,ss in [('all_queries',sd),('pilot_unseen_ships',sd[sd.new_query_ship]),('pilot_all_roles_unseen',sd[sd.new_role_ship])]:
        shipmean=ss.groupby(['cohort','budget','cell','ship','class_id'])[['old_correct','new_correct']].mean().reset_index()
        for (cohort,b,cell),g in shipmean.groupby(['cohort','budget','cell']):
            byclass=g.groupby('class_id')[['old_correct','new_correct']].mean()
            sr.append(dict(cohort=cohort,budget=b,cell=cell,subset=subset,ships=len(g),
                ship_oa_delta=100*float((g.new_correct-g.old_correct).mean()),
                ship_ba_delta=100*float((byclass.new_correct-byclass.old_correct).mean())))
    pd.DataFrame(sr).to_csv(OUT/'vessel_balanced_summary.csv',index=False)
    ports=summarize(df,'effects')
    group=['cohort','subset','budget','cell','mode','arm']
    absolute=df.groupby(group+['port','episode'])[VALUES].mean().reset_index().groupby(group+['port'])[VALUES].mean().reset_index()
    absolute.groupby(group)[VALUES].mean().to_csv(OUT/'absolute_summary.csv')
    pooled=df.groupby(group)[['n','correct','ref_correct']].sum().reset_index()
    pooled['pooled_oa']=100*pooled.correct/pooled.n
    pooled['pooled_delta_oa']=100*(pooled.correct-pooled.ref_correct)/pooled.n
    pooled.to_csv(OUT/'pooled_summary.csv',index=False)
    # Full-task class contributions, zero for absent classes, reconstruct macro differences.
    ck=['episode','port','cohort','subset','budget','draw','cell','mode','arm']
    sums=tr.groupby(ck)[['delta_oa','delta_ba']].sum().sort_index()
    np.testing.assert_allclose(sums,df.set_index(ck)[['delta_oa','delta_ba']].sort_index(),atol=1e-10)
    cc=df[ck].merge(pd.DataFrame({'class_id':range(8)}),how='cross')
    cc=cc.merge(tr,on=ck+['class_id'],how='left').fillna({'delta_oa':0,'delta_ba':0,'rescued':0,'harmed':0,'n':0})
    vv=['delta_oa','delta_ba','rescued','harmed']
    cs=cc.groupby(group+['port','class_id'])[vv].mean().reset_index()
    cs.to_csv(OUT/'class_ports.csv',index=False)
    cs.groupby(group+['class_id'])[vv].mean().to_csv(OUT/'class_summary.csv')
    rt=pd.read_csv(OUT/'retrieval.csv');rets=[]
    rt['new_role_ship']=~d.iloc[rt.row.to_numpy()].mmsi.isin(seen_roles).to_numpy()
    for subset,a in [('all_queries',rt),('pilot_unseen_ships',rt[rt.new_ship]),('pilot_all_roles_unseen',rt[rt.new_role_ship])]:
        perclass=a.groupby(['cohort','episode','port','arm','class_id']).map_at_r.mean().reset_index()
        scenes=perclass.groupby(['cohort','episode','port','arm']).map_at_r.mean().reset_index()
        scenes['subset']=subset;rets.append(scenes)
    rr=pd.concat(rets);rr.to_csv(OUT/'retrieval_scenes.csv',index=False)
    rr.groupby(['cohort','subset','port','arm']).map_at_r.mean().groupby(['cohort','subset','arm']).mean().to_csv(OUT/'retrieval_summary.csv')
    scope=[]
    for cohort in ['pilot','new_acquisition']:
        es=[e for e in plans if e['cohort']==cohort]
        for subset in ['all_queries','pilot_unseen_ships','pilot_all_roles_unseen']:
            q=np.concatenate([np.array(e['query'])[np.ones(len(e['query']),bool) if subset=='all_queries' else
                (e['pilot_unseen_ship_mask'] if subset=='pilot_unseen_ships' else ~d.iloc[e['query']].mmsi.isin(seen_roles).to_numpy())] for e in es])
            unique=np.unique(q)
            scope.append(dict(cohort=cohort,subset=subset,scenes=len(es),query_events=len(q),unique_rows=len(unique),
                unique_ships=int(d.iloc[unique].mmsi.nunique()),class_counts=d.iloc[unique].class_id.value_counts().sort_index().to_dict()))
    write('SCOPE.json',scope)
    write('VERIFIED.json',dict(independent_metric_rows=independent,independent_kernel_ridge_cases=len(solver_errors),
        max_probability_error=max(solver_errors),class_sum_rows=len(sums),pilot_replay_arrays=12))
    selected=ports[(ports.cohort=='new_acquisition')&(ports.subset=='all_queries')&(ports['mode']=='compact_full')]
    print('NEW ACQUISITION EQUAL PORT EFFECTS\n',selected.groupby(['budget','cell','arm'])[DELTA].mean().to_string(),flush=True)
    print('SCENE PORT EFFECTS\n',selected[selected.arm=='scene_memory'].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
