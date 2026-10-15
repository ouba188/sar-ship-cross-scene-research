"""Posthoc additive score accounting, not a new policy or physical cause test."""

import evaluate_evidence as a
from evaluate_evidence import np, pd, OUT, PREV, ARMS, get, load, read, reference, class_weights, gain_labels


def main():
    rows=[];checks=[];labels={}
    for u in read(OUT/'PREDICTIONS_FROZEN.json')['units']:
        ctx='pooled' if u['family']=='external' else u['task']
        z,e=get(u);ref=reference(ctx)
        if u['label_path'] not in labels:
            labels[u['label_path']]=pd.read_csv(u['label_path'],usecols=['class_id']).class_id.to_numpy(int)
        y=labels[u['label_path']][z['query_ids']];w=class_weights(y)
        x=(z['x']-ref['mean'])/ref['scale'];oldlogit=x@ref['coef']+ref['intercept']
        gain=gain_labels(z['raw'],z['initial'],y)
        d=load(OUT/'decisions'/(u['uid']+'.npz'))
        for arm,cols in ARMS.items():
            m=load(OUT/'models'/f'{ctx}__{arm}.npz')
            ne=(e-m['mean'])/m['scale']
            neww=np.zeros(14);neww[cols]=m['coef'][16:]
            parts=dict(old16_refit=x@(m['coef'][:16]-ref['coef']),
                cases=ne[:,:8]@neww[:8],deletion=ne[:,8:]@neww[8:],
                intercept=np.full(len(y),float(m['intercept']-ref['intercept'])))
            newlogit=oldlogit+sum(parts.values())
            np.testing.assert_allclose(a.expit(newlogit),d[arm],atol=1e-10,rtol=0)
            checks.append(dict(uid=u['uid'],arm=arm,max_error=float(abs(a.expit(newlogit)-d[arm]).max())))
            old,new=oldlogit>0,newlogit>0
            masks={'restored_rescue':(gain==1)&~old&new,'newly_lost_rescue':(gain==1)&old&~new,
                   'new_harm':(gain==-1)&~old&new,'newly_avoided_harm':(gain==-1)&old&~new}
            for event,mask in masks.items():
                for cls in [-1,*np.unique(y)]:
                    keep=mask if cls==-1 else mask&(y==cls)
                    n=int(keep.sum());weight=float(w[keep].sum())
                    row=dict(uid=u['uid'],family=u['family'],task=u['task'],cohort=u['cohort'],arm=arm,
                             event=event,class_id=int(cls),n=n,ba_mass=100*weight,
                             old_logit_sum=float(oldlogit[keep].sum()),new_logit_sum=float(newlogit[keep].sum()))
                    for name,v in parts.items():
                        row[name+'_sum']=float(v[keep].sum())
                    rows.append(row)
    pd.DataFrame(checks).to_csv(OUT/'logit_replays.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(OUT/'decision_transitions.csv',index=False)
    keys=['family','task','cohort','arm','event','class_id']
    means=df.groupby(keys)[['n','ba_mass']].mean()
    totals=df.groupby(keys)[['n','old_logit_sum','new_logit_sum','old16_refit_sum','cases_sum','deletion_sum','intercept_sum']].sum()
    for col in totals.columns[1:]:
        means[col.replace('_sum','_mean')]=totals[col]/totals.n.replace(0,np.nan)
    means.reset_index().to_csv(OUT/'decision_transition_summary.csv',index=False)
    print('TRANSITION ACCOUNTING',len(checks),flush=True)


if __name__=='__main__':
    main()
