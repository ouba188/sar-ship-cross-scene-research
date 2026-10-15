"""Bounded post-hoc diagnosis: common reference response versus contrast."""
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.class_weight import compute_sample_weight
from run_pilot import OUT, SEED, RHO, metadata, operator, write
from analyze_pilot import scores
from core import map_at_r


def main():
    d=metadata(); f=np.load(OUT/'encoded.npz'); rows=f['rows']; z=np.c_[f['r'],f['t']]
    source=np.load(OUT/'source_rows.npy'); si=np.searchsorted(rows,source)
    rng=np.random.default_rng(SEED); ships=d.iloc[source].mmsi.to_numpy()
    chosen=np.array([rng.choice(np.flatnonzero(ships==s)) for s in np.unique(ships)])
    r=f['rel_scene_memory']; rs=r[si[chosen]]
    _,sv,vt=np.linalg.svd(rs,full_matrices=False)
    v=vt[0]; common=(r@v)[:,None]; contrast=r-common*v[None,:]
    np.testing.assert_allclose(r[:100]@r[:100].T,common[:100]@common[:100].T+contrast[:100]@contrast[:100].T,atol=1e-12)
    spectrum=(sv**2)/(sv@sv)
    effrank=float(np.exp(-np.sum(spectrum*np.log(np.maximum(spectrum,1e-300)))))
    source_norm=np.linalg.norm(z[si[chosen]],axis=1)
    common_amp=np.abs(common[si[chosen],0])
    corr=float(np.corrcoef(source_norm,common_amp)[0,1])
    plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    ev=operator(); records=[]; retrieval=[]; checks=0; errs=[]; agreement=[]
    names=['shrink_only','common_only','contrast_only']
    for e in plans:
        q=np.array(e['query']); qi=np.searchsorted(rows,q); y=d.iloc[q].class_id.to_numpy(int)
        gi=np.searchsorted(rows,e['gallery']); gy=d.iloc[e['gallery']].class_id.to_numpy(int)
        k0=z[qi]@z[gi].T+1
        for name,sim in [('memory_only',r[qi]@r[gi].T),('common_only',common[qi]@common[gi].T),
                         ('contrast_only',contrast[qi]@contrast[gi].T),
                         ('mix_common',(1-RHO)*k0+RHO*(common[qi]@common[gi].T)),
                         ('mix_contrast',(1-RHO)*k0+RHO*(contrast[qi]@contrast[gi].T))]:
            values=map_at_r(sim,y,gy)
            for c in np.unique(y):
                retrieval.append(dict(episode=e['episode'],port=e['port'],arm=name,class_id=int(c),map_at_r=float(np.nanmean(values[y==c])) if np.isfinite(values[y==c]).any() else np.nan))
        for b in [16,64]:
            p=np.load(OUT/'predictions'/f"episode_{e['episode']:02d}_b{b}.npz")
            out=np.empty((5,2,3,len(q),8))
            for draw in range(5):
                for ci,cell in enumerate(['same','other']):
                    s=p['support_rows'][draw,ci]; ix=np.searchsorted(rows,s); sy=d.iloc[s].class_id.to_numpy(int)
                    kb=f['b'][ix]@f['b'][ix].T+1
                    ab=ev.coefficients(kb,f['pb'][ix],sy,8.)
                    bp=softmax(f['pb'][qi]+(f['b'][qi]@f['b'][ix].T+1)@ab,axis=1)
                    ss=z[ix]@z[ix].T+1; qs=z[qi]@z[ix].T+1
                    kss=[(1-RHO)*ss,(1-RHO)*ss+RHO*(common[ix]@common[ix].T),(1-RHO)*ss+RHO*(contrast[ix]@contrast[ix].T)]
                    kqs=[(1-RHO)*qs,(1-RHO)*qs+RHO*(common[qi]@common[ix].T),(1-RHO)*qs+RHO*(contrast[qi]@contrast[ix].T)]
                    ref=p['probabilities'][draw,ci,0,1]; full=p['probabilities'][draw,ci,3,1]
                    for ai,name in enumerate(names):
                        a=ev.coefficients(kss[ai],f['pd'][ix],sy,8.)
                        prob=.5*(bp+softmax(f['pd'][qi]+kqs[ai]@a,axis=1)); out[draw,ci,ai]=prob
                        m=scores(y,prob); old=scores(y,ref)
                        records.append(dict(episode=e['episode'],port=e['port'],budget=b,draw=draw,cell=cell,arm=name,
                            **{'delta_'+k:m[k]-old[k] for k in m},
                            changed_vs_scene=int(np.sum(prob.argmax(1)!=full.argmax(1))),
                            scene_prob_rms=float(np.sqrt(np.mean((prob-full)**2)))))
                        if draw==0 and ci==1:
                            independent=KernelRidge(alpha=8.,kernel='precomputed').fit(kss[ai],2*np.eye(8)[sy]-1-f['pd'][ix],sample_weight=compute_sample_weight('balanced',sy))
                            other=.5*(bp+softmax(f['pd'][qi]+independent.predict(kqs[ai]),axis=1))
                            errs.append(float(np.max(np.abs(other-prob))))
                            np.testing.assert_allclose(other,prob,atol=1e-10,rtol=1e-10); checks+=1
                    agreement.append(dict(episode=e['episode'],budget=b,draw=draw,cell=cell,n=len(q),
                        mean_scene_changes=int(np.sum(p['probabilities'][draw,ci,1,1].argmax(1)!=full.argmax(1))),
                        cluster_scene_changes=int(np.sum(p['probabilities'][draw,ci,2,1].argmax(1)!=full.argmax(1)))))
            np.savez_compressed(OUT/'predictions'/f"diagnostic_{e['episode']:02d}_b{b}.npz",probabilities=out,arms=np.array(names),query_rows=q)
    df=pd.DataFrame(records); df.to_csv(OUT/'diagnostic_effects.csv',index=False)
    df.groupby(['budget','cell','arm']).mean(numeric_only=True).to_csv(OUT/'diagnostic_summary.csv')
    rr=pd.DataFrame(retrieval);rr.to_csv(OUT/'diagnostic_retrieval.csv',index=False)
    rr.groupby(['episode','port','arm']).map_at_r.mean().groupby('arm').mean().to_csv(OUT/'diagnostic_retrieval_summary.csv')
    agree=pd.DataFrame(agreement);agree.to_csv(OUT/'prediction_agreement.csv',index=False)
    write('DIAGNOSIS.json',dict(source_reference_first_component_energy=float(spectrum[0]),effective_rank=effrank,
        source_common_amplitude_vs_feature_norm_correlation=corr,source_reference_components=spectrum.tolist(),
        primary_exposures=int(agree.n.sum()),class_mean_vs_scene_changed=int(agree.mean_scene_changes.sum()),
        cluster_vs_scene_changed=int(agree.cluster_scene_changes.sum()),
        independent_solver_cases=checks,max_probability_error=max(errs),new_neural_training=0,
        interpretation='Posthoc source-fitted rank1 decomposition; not a new selected candidate or physical factor separation',
        limitation='Contrast-only keeps original amplitude after decomposition; it is not retuned to favorable target outcomes'))
    print('FIRST_COMPONENT',spectrum[0],'EFFECTIVE_RANK',effrank,'NORM_CORRELATION',corr,flush=True)
    print(df.groupby(['budget','cell','arm'])[['delta_oa','delta_ba','delta_macro_brier','changed_vs_scene']].mean().to_string(),flush=True)


if __name__=='__main__':
    main()
