"""Source-defined class-shared versus within-class mode decomposition."""
import json
import pickle
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from expand import OUT,PILOT,SEED,metadata,operator,write
from core import RHO
from analyze_pilot import scores
from analyze_expansion import summarize,VALUES,DELTA


def main():
    started=time.perf_counter();d=metadata();ev=operator()
    plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    with np.load(OUT/'encoded.npz') as data:f={k:data[k] for k in data.files}
    rows=f['rows'];z=np.c_[f['r'],f['t']];r=f['rel_scene_memory']
    with (PILOT/'memories.pkl').open('rb') as h:memory=pickle.load(h)['scene_memory']
    classes=memory['classes'];projector=np.zeros((len(classes),8))
    for c in range(8):projector[classes==c,c]=1/np.sqrt(np.sum(classes==c))
    shared=r@projector;within=r-shared@projector.T
    np.testing.assert_allclose(r[:200]@r[:200].T,shared[:200]@shared[:200].T+within[:200]@within[:200].T,atol=1e-11)
    np.testing.assert_allclose(shared.T@within@projector,0,atol=1e-10)
    source=np.load(PILOT/'source_rows.npy');ships=d.iloc[source].mmsi.to_numpy();rng=np.random.default_rng(SEED)
    chosen=np.array([rng.choice(np.flatnonzero(ships==s)) for s in np.unique(ships)])
    with np.load(PILOT/'relative_repair/relations.npz') as old:
        sr=old['scene_memory'][np.searchsorted(old['rows'],source[chosen])]
    total_energy=float(np.mean(np.sum(sr**2,axis=1)))
    shared_energy=float(np.mean(np.sum((sr@projector)**2,axis=1)))
    scale=np.sqrt(total_energy/shared_energy)
    names=['shared_keep','within_keep','shared_scaled']
    maps=[shared,within,shared*scale]
    dest=OUT/'class_shared_diagnosis';dest.mkdir(exist_ok=True)
    (dest/'PROTOCOL.json').write_text(json.dumps(dict(date='2026-10-14',status='Posthoc one decomposition and one source-scaled repair',
        arms=names,source_shared_energy_fraction=shared_energy/total_energy,source_scale=float(scale),
        meaning='Class block projection averages references of each known SOURCE class; target query labels not used',
        fixed='Same queries/supports/prior/rho.25/ridge8; each arm re-solves its support coefficients',
        correction='shared_scaled drops within-class mode responses and matches source energy by fixed rule, no parameter sweep',
        controls='shared_keep/within_keep preserve native component amplitudes; their score gains are not additive after re-solving',
        evaluation='Same expanded development observations, not independent confirmation of posthoc repair'),indent=2),encoding='utf-8')
    pilot=json.loads((PILOT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    prows=np.unique(np.concatenate([e['query']+e['gallery']+sum([s for a in e['draws'] for ss in a.values() for s in ss],[]) for e in pilot]))
    seen=set(d.iloc[prows].mmsi)
    records=[];terms=[];risks=[];verify=[];energy=[]
    for e in plans:
        q=np.array(e['query']);qi=np.searchsorted(rows,q);y=d.iloc[q].class_id.to_numpy(int)
        masks={'all_queries':np.ones(len(q),bool),'pilot_all_roles_unseen':~d.iloc[q].mmsi.isin(seen).to_numpy()}
        energy.append(dict(episode=e['episode'],port=e['port'],cohort=e['cohort'],
            shared_energy_fraction=float(np.sum(shared[qi]**2)/np.sum(r[qi]**2))))
        for b in [16,64]:
            p=np.load(OUT/'predictions'/f"episode_{e['episode']:02d}_b{b}.npz")
            pred=np.empty((5,2,3,2,len(q),8))
            for draw in range(5):
                for ci,cell in enumerate(['same','other']):
                    s=p['support_rows'][draw,ci];si=np.searchsorted(rows,s);sy=d.iloc[s].class_id.to_numpy(int)
                    ks=z[si]@z[si].T+1;kq=z[qi]@z[si].T+1
                    ab=ev.coefficients(f['b'][si]@f['b'][si].T+1,f['pb'][si],sy,8.)
                    bp=softmax(f['pb'][qi]+(f['b'][qi]@f['b'][si].T+1)@ab,axis=1)
                    full_alpha=ev.coefficients((1-RHO)*ks+RHO*(r[si]@r[si].T),f['pd'][si],sy,8.)
                    baseline_expert=p['probabilities'][draw,ci,0,0]
                    rivals=np.argsort(-baseline_expert,axis=1)
                    rival=np.array([a[0] if a[0]!=yy else a[1] for a,yy in zip(rivals,y)])
                    # Freeze full coefficients for signed class-margin pathway accounting.
                    for component,rr in [('shared',shared),('within',within)]:
                        delta=RHO*(rr[qi]@rr[si].T)@full_alpha
                        margins=delta[np.arange(len(y)),y]-delta[np.arange(len(y)),rival]
                        for c in np.unique(y):
                            mask=y==c
                            terms.append(dict(episode=e['episode'],port=e['port'],cohort=e['cohort'],budget=b,draw=draw,
                                cell=cell,component=component,class_id=int(c),mean_true_rival_margin=float(margins[mask].mean())))
                    for ai,(arm,rr) in enumerate(zip(names,maps)):
                        ss=(1-RHO)*ks+RHO*(rr[si]@rr[si].T);qs=(1-RHO)*kq+RHO*(rr[qi]@rr[si].T)
                        alpha=ev.coefficients(ss,f['pd'][si],sy,8.)
                        ep=softmax(f['pd'][qi]+qs@alpha,axis=1);pred[draw,ci,ai]=[ep,.5*(bp+ep)]
                        for mi,mode in enumerate(['expert','compact_full']):
                            for subset,mask in masks.items():
                                if not mask.any():continue
                                yy=y[mask];pp=pred[draw,ci,ai,mi,mask];ref=p['probabilities'][draw,ci,0,mi,mask]
                                m=scores(yy,pp);old=scores(yy,ref)
                                base=dict(episode=e['episode'],port=e['port'],cohort=e['cohort'],budget=b,draw=draw,cell=cell,
                                    arm=arm,mode=mode,subset=subset)
                                ca=ref.argmax(1)==yy;cb=pp.argmax(1)==yy
                                records.append(dict(**base,n=len(yy),**m,**{'delta_'+k:m[k]-old[k] for k in VALUES},
                                    rescued=int(np.sum(~ca&cb)),harmed=int(np.sum(ca&~cb))))
                                for c in np.unique(yy):
                                    cm=yy==c;net=np.sum(~ca&cb&cm)-np.sum(ca&~cb&cm)
                                    risks.append(dict(**base,class_id=int(c),delta_ba=100*net/cm.sum()/len(np.unique(yy))))
                        if draw==0 and ci==1 and e['episode'] in [1,15,35]:
                            other=KernelRidge(alpha=8.,kernel='precomputed').fit(ss,2*np.eye(8)[sy]-1-f['pd'][si],sample_weight=compute_sample_weight('balanced',sy))
                            val=softmax(f['pd'][qi]+other.predict(qs),axis=1)
                            np.testing.assert_allclose(ep,val,atol=1e-10,rtol=1e-10);verify.append(float(np.max(np.abs(ep-val))))
            np.savez_compressed(dest/f"episode_{e['episode']:02d}_b{b}.npz",probabilities=pred,arms=np.array(names),query_rows=q,support_rows=p['support_rows'])
    df=pd.DataFrame(records);df.to_csv(dest/'metrics.csv',index=False)
    summarize(df,'class_shared')
    pd.DataFrame(terms).to_csv(dest/'margin_terms.csv',index=False)
    pd.DataFrame(risks).to_csv(dest/'class_effects.csv',index=False)
    pd.DataFrame(energy).to_csv(dest/'energy.csv',index=False)
    pooled=df.groupby(['cohort','subset','budget','cell','mode','arm'])[['n','rescued','harmed']].sum()
    pooled['pooled_delta_oa']=100*(pooled.rescued-pooled.harmed)/pooled.n
    pooled.to_csv(dest/'pooled.csv')
    (dest/'COMPLETE.json').write_text(json.dumps(dict(expert_solves=len(plans)*2*5*2*3,neural_training=0,
        seconds=time.perf_counter()-started,independent_solver_cases=len(verify),max_error=max(verify)),indent=2),encoding='utf-8')
    chosen=df[(df.cohort=='new_acquisition')&(df.subset=='all_queries')&(df['mode']=='compact_full')]
    print('CLASS_SHARED_ENERGY_FRACTION',shared_energy/total_energy,'SCALE',scale,flush=True)
    print(chosen.groupby(['port','budget','cell','arm'])[DELTA].mean().groupby(['budget','cell','arm']).mean().to_string(),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
