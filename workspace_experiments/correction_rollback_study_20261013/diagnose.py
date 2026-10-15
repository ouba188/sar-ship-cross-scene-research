"""Source transition coverage and exact additive rollback-logit accounting."""

import numpy as np
import pandas as pd
from scipy.special import expit
from run_probe import OUT, ROOT, PREV, GROUPS, read, load, source_units, base_model, class_weights, gain_labels


def get_events(u, table, source=False):
    z = load(u['path'])
    ctx = u['context'] if source else ('pooled' if u['family']=='external' else u['task'])
    ref = base_model(ctx)
    q = table.iloc[z['query_ids']]
    y = q.class_id.to_numpy(int)
    g = gain_labels(z['raw'],z['initial'],y)
    r,i = z['raw'].argmax(1),z['initial'].argmax(1)
    terms = ((z['x']-ref['mean'])/ref['scale'])*ref['coef']
    logit = terms.sum(1)+ref['intercept']
    score = expit(logit)
    if not source:
        old = load(PREV/'decisions'/(u['uid']+'.npz'))['expanded__contextual']
        np.testing.assert_allclose(score,old,atol=1e-12,rtol=0)
    keep = g!=0
    idkey = 'mmsi' if 'mmsi' in q else 'mmsi_or_ship_id'
    ship = q[idkey].fillna('').astype(str).to_numpy()
    frame = pd.DataFrame(dict(uid=u['uid'],context=ctx,family='source_fit' if source else u['family'],
        task=ctx if source else u['task'],cohort='expanded' if source else u['cohort'],
        obs_id=z['query_ids'][keep],ship=ship[keep],class_id=y[keep],raw_class=r[keep],initial_class=i[keep],
        gain=g[keep],weight=class_weights(y)[keep],probability=score[keep],logit=logit[keep],
        intercept=float(ref['intercept'])))
    frame['status']=np.where(frame.gain==1,np.where(frame.probability>.5,'kept_rescue','lost_rescue'),
                             np.where(frame.probability>.5,'kept_harm','avoided_harm'))
    frame['net']=np.where(frame.probability>.5,0,-frame.gain)
    for name,sl in GROUPS.items():
        frame[name]=terms[keep,sl].sum(1)
    np.testing.assert_allclose(frame[list(GROUPS)].sum(axis=1)+frame.intercept,frame.logit,atol=1e-12)
    return frame


def main():
    meta = pd.read_csv(ROOT/'same_vessel_condition_study_20261010/observations.csv',dtype={'mmsi':str})
    events = pd.concat([get_events(u,meta,True) for u in source_units()],ignore_index=True)
    events.to_csv(OUT/'source_events.csv',index=False)
    source_pairs=[]
    for ctx in ('pooled',)+tuple(events.context.unique()):
        subset=events if ctx=='pooled' else events.loc[events.context==ctx]
        nunit=108 if ctx=='pooled' else 36
        for r in range(8):
            for i in range(8):
                f=subset.loc[(subset.raw_class==r)&(subset.initial_class==i)]
                good=f.gain==1;bad=f.gain==-1
                rw,hw=f.loc[good,'weight'].sum(),f.loc[bad,'weight'].sum()
                source_pairs.append(dict(context=ctx,raw_class=r,initial_class=i,event_exposures=len(f),
                    event_obs=f.obs_id.nunique(),event_ships=f.ship.nunique(),rescue_exposures=int(good.sum()),
                    harm_exposures=int(bad.sum()),rescue_weight=rw/nunit,harm_weight=hw/nunit,
                    rescue_fraction=rw/(rw+hw) if rw+hw else np.nan))
    sf=pd.DataFrame(source_pairs);sf.to_csv(OUT/'source_transitions.csv',index=False)
    lookup=sf.set_index(['context','raw_class','initial_class'])
    units=read(PREV/'PREDICTIONS_FROZEN.json')['units']
    tables={};target=[];pairs=[];groups=[];checks=[]
    for u in units:
        if u['label_path'] not in tables:
            cols=pd.read_csv(u['label_path'],nrows=0).columns
            keep=[k for k in ('class_id','mmsi','mmsi_or_ship_id') if k in cols]
            tables[u['label_path']]=pd.read_csv(u['label_path'],usecols=keep,dtype={k:str for k in keep if k!='class_id'})
        f=get_events(u,tables[u['label_path']])
        target.append(f)
        ctx='pooled' if u['family']=='external' else u['task']
        key={k:u[k] for k in ('family','task','cohort','uid')}
        for r in range(8):
            for i in range(8):
                cell=f.loc[(f.raw_class==r)&(f.initial_class==i)]
                ref=lookup.loc[(ctx,r,i)]
                row=dict(**key,raw_class=r,initial_class=i,event_n=len(cell),event_ships=cell.ship.nunique(),
                    source_event_obs=int(ref.event_obs),source_event_ships=int(ref.event_ships),
                    source_rescue_fraction=ref.rescue_fraction,
                    event_ba_mass=100*cell.weight.sum(),delta_ba=100*(cell.weight*cell.net).sum())
                for status in ('lost_rescue','kept_rescue','avoided_harm','kept_harm'):
                    row[status+'_n']=int((cell.status==status).sum())
                    row[status+'_ba']=100*cell.loc[cell.status==status,'weight'].sum()
                pairs.append(row)
        for (cls,status),cell in f.groupby(['class_id','status']):
            row=dict(**key,class_id=int(cls),status=status,event_n=len(cell),ships=cell.ship.nunique(),
                ba_mass=100*cell.weight.sum())
            for col in ['intercept','logit']+list(GROUPS):
                row[col]=float(np.average(cell[col],weights=cell.weight))
            groups.append(row)
        old=pd.read_csv(PREV/'metrics.csv') if not checks else None
        if old is not None:
            refs=old.set_index(['uid','arm'])
        actual=100*(f.weight*f.net).sum()
        expected=refs.loc[(u['uid'],'expanded__contextual'),'versus_initial_ba']
        np.testing.assert_allclose(actual,expected,atol=1e-10)
        checks.append(dict(uid=u['uid'],event_account_error=float(abs(actual-expected))))
    tf=pd.concat(target,ignore_index=True)
    tf.to_csv(OUT/'evaluation_events.csv',index=False)
    pf=pd.DataFrame(pairs);pf.to_csv(OUT/'evaluation_transitions.csv',index=False)
    pf.groupby(['family','task','cohort','raw_class','initial_class']).mean(numeric_only=True).reset_index().to_csv(OUT/'transition_summary.csv',index=False)
    gf=pd.DataFrame(groups);gf.to_csv(OUT/'logit_groups.csv',index=False)
    # This mean describes event-present cells; no zero-event cell has a logit mean.
    gf.groupby(['family','task','cohort','class_id','status']).mean(numeric_only=True).reset_index().to_csv(OUT/'logit_summary.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'diagnostic_replays.csv',index=False)
    joined=tf.merge(sf[['context','raw_class','initial_class','event_obs','rescue_fraction']],
                    on=['context','raw_class','initial_class'],validate='many_to_one')
    joined['coverage']=np.where(joined.event_obs==0,'unseen',np.where(joined.rescue_fraction>=.5,'source_favors_initial','source_favors_raw'))
    rows=[]
    for key,cell in joined.groupby(['family','task','cohort','uid']):
        for cov in ('unseen','source_favors_initial','source_favors_raw'):
            f=cell.loc[cell.coverage==cov]
            rows.append(dict(zip(['family','task','cohort','uid'],key),coverage=cov,events=len(f),
                event_ba_mass=100*f.weight.sum(),lost_rescue_ba=100*f.loc[f.status=='lost_rescue','weight'].sum(),
                avoided_harm_ba=100*f.loc[f.status=='avoided_harm','weight'].sum(),delta_ba=100*(f.weight*f.net).sum()))
    cf=pd.DataFrame(rows);cf.to_csv(OUT/'coverage_cells.csv',index=False)
    cf.groupby(['family','task','cohort','coverage']).mean(numeric_only=True).reset_index().to_csv(OUT/'coverage_summary.csv',index=False)
    print('DIAGNOSED',len(events),'source events',len(tf),'evaluation events',len(checks),'replays',flush=True)


if __name__=='__main__':
    main()
