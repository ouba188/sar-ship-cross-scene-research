"""One mechanism-led repair, identical source-only normalization for all memories."""
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.class_weight import compute_sample_weight
from core import ARMS, kernel, map_at_r
from run_pilot import OUT, SEED, metadata, operator, write
from analyze_pilot import scores


def main():
    d=metadata(); f=np.load(OUT/'encoded.npz'); rows=f['rows']; z=np.c_[f['r'],f['t']]
    source=np.load(OUT/'source_rows.npy'); ix=np.searchsorted(rows,source)
    rng=np.random.default_rng(SEED); ships=d.iloc[source].mmsi.to_numpy()
    chosen=np.array([rng.choice(np.flatnonzero(ships==s)) for s in np.unique(ships)])
    ref=ix[chosen]; energy=np.mean(np.sum(z[ref]**2,axis=1)+1)
    maps={}; scalings={}; spectra={}
    for arm in ARMS[1:]:
        r=f['rel_'+arm]
        unit=r/np.maximum(np.linalg.norm(r,axis=1,keepdims=True),1e-15)
        mean=unit[ref].mean(0)
        centered=unit-mean
        pre=float(np.mean(np.sum(centered[ref]**2,axis=1)))
        scale=float(np.sqrt(energy/pre))
        maps[arm]=centered*scale
        sv=np.linalg.svd(maps[arm][ref],compute_uv=False); proportion=sv**2/(sv@sv)
        spectra[arm]=dict(first_component_energy=float(proportion[0]),effective_rank=float(np.exp(-np.sum(proportion*np.log(np.maximum(proportion,1e-300))))))
        scalings[arm]=dict(center=mean.tolist(),scale=scale,pre_scale_source_energy=pre)
    dest=OUT/'relative_repair';dest.mkdir(exist_ok=True)
    (dest/'PROTOCOL.json').write_text(json.dumps(dict(status='Posthoc single targeted repair, not new independent confirmation',
        reason='Raw reference responses have99.895percent energy in rank1; rank1-only exactly reproduces all compact decisions',
        transform='L2-normalize reference profile per sample, subtract source-unique-vessel mean, source mean-energy match',
        change='Remove common amplitude and expose profile differences, applied to all3 memory controls',
        fixed='Same sources/prior/query/support/rho.25/ridge8/T1; no parameter sweep or encoder training',
        tradeoff='Memory no longer carries absolute distance-to-bank; unchanged original kernel still carries original features',
        calibration=scalings,spectra=spectra),ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(dest/'relations.npz',rows=rows,**maps)
    plans=json.loads((OUT/'ENROLLMENT.json').read_text(encoding='utf-8'))
    ev=operator(); records=[]; transitions=[]; retrieval=[]; costs=[]; solvererr=[]; examples=[]
    import time
    for e in plans:
        q=np.array(e['query']); qi=np.searchsorted(rows,q); y=d.iloc[q].class_id.to_numpy(int)
        gi=np.searchsorted(rows,e['gallery']); gy=d.iloc[e['gallery']].class_id.to_numpy(int)
        for arm in ARMS[1:]:
            r=maps[arm]; sim=kernel(z[qi],z[gi],r[qi],r[gi]); ap=map_at_r(sim,y,gy)
            for c in np.unique(y):
                retrieval.append(dict(episode=e['episode'],port=e['port'],arm=arm,class_id=int(c),map_at_r=float(np.nanmean(ap[y==c])) if np.isfinite(ap[y==c]).any() else np.nan))
        for b in [16,64]:
            p=np.load(OUT/'predictions'/f"episode_{e['episode']:02d}_b{b}.npz")
            outputs=np.empty((5,2,3,2,len(q),8))
            for draw in range(5):
                for ci,cell in enumerate(['same','other']):
                    s=p['support_rows'][draw,ci]; si=np.searchsorted(rows,s); sy=d.iloc[s].class_id.to_numpy(int)
                    ab=ev.coefficients(f['b'][si]@f['b'][si].T+1,f['pb'][si],sy,8.)
                    bp=softmax(f['pb'][qi]+(f['b'][qi]@f['b'][si].T+1)@ab,axis=1)
                    for ai,arm in enumerate(ARMS[1:]):
                        tick=time.perf_counter()
                        r=maps[arm]; ks=kernel(z[si],z[si],r[si],r[si]); kq=kernel(z[qi],z[si],r[qi],r[si])
                        a=ev.coefficients(ks,f['pd'][si],sy,8.)
                        ep=softmax(f['pd'][qi]+kq@a,axis=1)
                        outputs[draw,ci,ai]=[ep,.5*(bp+ep)]
                        costs.append(dict(episode=e['episode'],budget=b,draw=draw,cell=cell,arm=arm,seconds=time.perf_counter()-tick))
                        for mi,mode in enumerate(['expert','compact_full']):
                            pp=outputs[draw,ci,ai,mi]; baseline=p['probabilities'][draw,ci,0,mi]
                            m=scores(y,pp); old=scores(y,baseline)
                            base=dict(episode=e['episode'],port=e['port'],budget=b,draw=draw,cell=cell,arm=arm,mode=mode)
                            records.append(dict(**base,**m,**{'delta_'+k:m[k]-old[k] for k in m}))
                            ca=baseline.argmax(1)==y; cb=pp.argmax(1)==y
                            for c in range(8):
                                mask=y==c; rescue=int(np.sum(~ca&cb&mask)); harm=int(np.sum(ca&~cb&mask))
                                transitions.append(dict(**base,class_id=c,n=int(mask.sum()),rescued=rescue,harmed=harm,
                                    delta_oa=100*(rescue-harm)/len(y),delta_ba=100*(rescue-harm)/mask.sum()/len(np.unique(y)) if mask.any() else 0))
                            if arm=='scene_memory' and mode=='compact_full' and cell=='other' and draw==0:
                                for ii in np.flatnonzero(ca!=cb):
                                    # Saved explanatory cases use the exact fixed original/candidate kernels.
                                    original_sim=z[qi[ii]]@z[si].T+1
                                    jj=int(np.argmax(kq[ii])); kk=int(np.argmax(original_sim))
                                    examples.append(dict(**base,row=int(q[ii]),ship=d.iloc[q[ii]].mmsi,class_id=int(y[ii]),
                                        old_prediction=int(baseline[ii].argmax()),new_prediction=int(pp[ii].argmax()),
                                        outcome='rescue' if cb[ii] else 'harm',original_top_support=int(s[kk]),
                                        original_support_class=int(sy[kk]),new_top_support=int(s[jj]),new_support_class=int(sy[jj])))
                        if draw==0 and ci==1:
                            independent=KernelRidge(alpha=8.,kernel='precomputed').fit(ks,2*np.eye(8)[sy]-1-f['pd'][si],sample_weight=compute_sample_weight('balanced',sy))
                            other=softmax(f['pd'][qi]+independent.predict(kq),axis=1)
                            solvererr.append(float(np.max(np.abs(ep-other))))
                            np.testing.assert_allclose(ep,other,atol=1e-10,rtol=1e-10)
            np.savez_compressed(dest/f"episode_{e['episode']:02d}_b{b}.npz",probabilities=outputs,arms=np.array(ARMS[1:]),query_rows=q,support_rows=p['support_rows'])
    df=pd.DataFrame(records);df.to_csv(dest/'metrics.csv',index=False)
    vals=['delta_oa','delta_ba','delta_brier','delta_macro_brier','delta_nll','delta_macro_nll']
    df.groupby(['budget','cell','mode','arm'])[vals].mean().to_csv(dest/'effect_summary.csv')
    df.groupby(['port','budget','cell','mode','arm'])[vals].mean().to_csv(dest/'port_effects.csv')
    df.groupby(['episode','port','budget','cell','mode','arm'])[vals].mean().to_csv(dest/'scene_effects.csv')
    draw=df.groupby(['budget','cell','mode','arm','draw'])[vals].mean().reset_index()
    draw.groupby(['budget','cell','mode','arm']).agg(ba_mean=('delta_ba','mean'),ba_support_sd=('delta_ba','std'),
        ba_positive_draws=('delta_ba',lambda a:int((a>0).sum())),oa_mean=('delta_oa','mean'),oa_positive_draws=('delta_oa',lambda a:int((a>0).sum()))).to_csv(dest/'support_variability.csv')
    tr=pd.DataFrame(transitions);tr.to_csv(dest/'transitions.csv',index=False)
    tr.groupby(['budget','cell','mode','arm','class_id'])[['rescued','harmed','delta_oa','delta_ba']].mean().to_csv(dest/'class_effects.csv')
    rr=pd.DataFrame(retrieval);rr.to_csv(dest/'retrieval.csv',index=False)
    rr.groupby(['episode','port','arm']).map_at_r.mean().groupby('arm').mean().to_csv(dest/'retrieval_summary.csv')
    pd.DataFrame(costs).to_csv(dest/'cached_solver_timing.csv',index=False)
    pd.DataFrame(examples).to_csv(dest/'cases.csv',index=False)
    k=['episode','port','budget','draw','cell','mode','arm']
    sums=tr.groupby(k)[['delta_oa','delta_ba']].sum()
    np.testing.assert_allclose(sums.sort_index(),df.set_index(k)[['delta_oa','delta_ba']].sort_index(),atol=1e-10)
    (dest/'VERIFIED.json').write_text(json.dumps(dict(independent_solver_cases=len(solvererr),max_error=max(solvererr),class_sum_rows=len(df)),indent=2),encoding='utf-8')
    print('REPAIR FULL EFFECTS\n',df[df['mode']=='compact_full'].groupby(['budget','cell','arm'])[vals].mean().to_string(),flush=True)
    print('REPAIR RETRIEVAL\n',rr.groupby(['episode','port','arm']).map_at_r.mean().groupby('arm').mean().to_string(),flush=True)
    print('REPAIR SPECTRA',spectra,flush=True)


if __name__=='__main__':
    main()
