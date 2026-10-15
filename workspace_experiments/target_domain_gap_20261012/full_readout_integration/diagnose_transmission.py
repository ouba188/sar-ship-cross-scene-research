"""Trace existing head updates through residual fitting and fixed probability fusion."""

from core import *
import sys
sys.path.insert(0,'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check')
from compact_fusion_source import ROOT,SOURCE
from prepare_learned_frequency import DATASETS


def centered_log(p):
    z=np.log(np.maximum(p,1e-300))
    return z-z.mean(axis=-1,keepdims=True)


def main():
    stages,groups,events=[],[],[]
    cases=[('primary',OUT/'full'/task,task) for task in ('Los Angeles','Port Klang','Santos')]
    cases += [('primary',OUT/'matched_full/oss2','oss2'),('legacy_oss2',OUT/'full/oss2','oss2')]
    for version,folder,task in cases:
        with np.load(folder/'predictions.npz') as z:
            saved={k:z[k] for k in z.files}
        if task=='oss2':
            frame=pd.read_csv(ROOT/DATASETS['oss2'][0]/'target_manifest.csv',keep_default_na=False,
                dtype={'mmsi_or_ship_id':str}).iloc[saved['rows']]
            shipcol='mmsi_or_ship_id'
        else:
            frame=pd.read_csv(SOURCE/'manifest.csv',usecols=['instance_id','class_id','mmsi'],
                keep_default_na=False,dtype={'mmsi':str}).iloc[saved['rows']]
            shipcol='mmsi'
        y=frame.class_id.to_numpy()
        arms=list(saved['arms'])
        a,b=arms.index('original'),arms.index('source_fixed')
        for draw in range(5):
            p0,p1=saved['expert_prior'][draw,[a,b]]
            e0,e1=saved['expert'][draw,[a,b]]
            f0,f1=saved['full'][draw,[a,b]]
            base=2*f0-e0
            np.testing.assert_allclose(base,2*f1-e1,atol=1e-14,rtol=1e-14)
            dp=centered_log(p1)-centered_log(p0)
            de=centered_log(e1)-centered_log(e0)
            dr=de-dp
            ep0,ep1,fp0,fp1=(v.argmax(1) for v in (e0,e1,f0,f1))
            for scope,mask in [('original',saved['eval_masks'][draw]),('common',saved['common_eval'])]:
                yy=y[mask]
                counts=np.bincount(yy,minlength=8)
                weights=100./(8*counts[yy])
                prior_rms=float(np.sqrt(np.mean(dp[mask]**2)))
                final_rms=float(np.sqrt(np.mean(de[mask]**2)))
                cosine=float(np.sum(dp[mask]*dr[mask])/
                    max(np.linalg.norm(dp[mask])*np.linalg.norm(dr[mask]),1e-30))
                er=(ep0[mask]!=yy)&(ep1[mask]==yy)
                eh=(ep0[mask]==yy)&(ep1[mask]!=yy)
                fr=(fp0[mask]!=yy)&(fp1[mask]==yy)
                fh=(fp0[mask]==yy)&(fp1[mask]!=yy)
                stages.append(dict(version=version,task=task,draw=draw,scope=scope,
                    prior_delta_rms=prior_rms,adapted_delta_rms=final_rms,
                    norm_retained_ratio=final_rms/prior_rms if prior_rms else 0.,
                    residual_change_cosine=cosine,
                    expert_rescues=int(er.sum()),expert_harms=int(eh.sum()),
                    full_rescues=int(fr.sum()),full_harms=int(fh.sum()),
                    expert_rescues_final_still_wrong=int((er&(fp1[mask]!=yy)).sum())))
                for name,sub in [('expert_argmax_changed',ep0[mask]!=ep1[mask]),
                                 ('expert_argmax_unchanged',ep0[mask]==ep1[mask])]:
                    groups.append(dict(version=version,task=task,draw=draw,scope=scope,group=name,
                        rows=int(sub.sum()),rescues=int((fr&sub).sum()),harms=int((fh&sub).sum()),
                        net_ba=float(np.sum(weights*sub*(fr.astype(int)-fh.astype(int))))))
                for j in np.flatnonzero(mask&((fp0!=fp1)|(ep0!=ep1))):
                    competing=f0[j].copy()
                    competing[y[j]]=-np.inf
                    rival=int(competing.argmax())
                    change=.5*((e1[j,y[j]]-e1[j,rival])-(e0[j,y[j]]-e0[j,rival]))
                    observed=(f1[j,y[j]]-f1[j,rival])-(f0[j,y[j]]-f0[j,rival])
                    if abs(change-observed)>1e-12:
                        raise ValueError('Fixed-base fusion margin attribution is inconsistent')
                    events.append(dict(version=version,task=task,draw=draw,scope=scope,
                        row=int(saved['rows'][j]),sample_id=str(saved['ids'][j]),ship=str(frame.iloc[j][shipcol]),
                        true_class=int(y[j]),expert_before=int(ep0[j]),expert_after=int(ep1[j]),
                        full_before=int(fp0[j]),full_after=int(fp1[j]),base_prediction=int(base[j].argmax()),
                        fusion_margin_delta=float(change)))
    for name,values in [('transmission',stages),('fusion_event_groups',groups),('transmission_events',events)]:
        pd.DataFrame(values).to_csv(OUT/(name+'.csv'),index=False)
    d=pd.DataFrame(stages)
    print('TRANSMISSION',flush=True)
    print(d[d.scope.eq('original')].groupby(['version','task']).mean(numeric_only=True).to_string(),flush=True)
    g=pd.DataFrame(groups)
    print('FUSION GROUPS',flush=True)
    print(g[g.scope.eq('original')].groupby(['version','task','group'])[['rescues','harms','net_ba']].mean().to_string(),flush=True)
    write(OUT/'TRANSMISSION_COMPLETE.json',dict(new_fits=0,base_unchanged=True,
        exact_fusion_margin_change=True,logit_differences_centered=True))


if __name__=='__main__':
    main()
