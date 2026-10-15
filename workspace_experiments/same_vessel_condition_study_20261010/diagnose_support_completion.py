"""Fixed role interventions and vessel/class attribution for completion pilots."""

from case_support_completion import *

PILOT = OUT
CONTROL = BASE/'case_completion_class_control_20261012'
OUT = PILOT/'diagnostics'


def scatter(x, meta):
    w=weights(meta); y=meta.class_id.to_numpy()
    mean=np.average(x,axis=0,weights=w)
    within,between=0.,0.
    for c in np.unique(y):
        m=y==c;center=np.average(x[m],axis=0,weights=w[m])
        within+=float(w[m]@np.square(x[m]-center).sum(1))
        between+=float(w[m].sum()*np.square(center-mean).sum())
    return dict(within=within,between=between,ratio=between/within)


def main():
    OUT.mkdir(exist_ok=True)
    meta=pd.read_csv(BASE/'observations.csv')
    metrics,events,contributions,geometry,readout,roles=[],[],[],[],[],[]
    all_class=pd.concat([pd.read_csv(PILOT/'classes.csv'),pd.read_csv(CONTROL/'classes.csv')])
    table=all_class.groupby(['arm','class_id']).recall.mean().unstack(0)
    table.subtract(table['raw'],axis=0).to_csv(OUT/'class_delta.csv')
    table.to_csv(OUT/'class_recall.csv')
    for port in PORTS:
        with np.load(TEACHER/port/'maps.npz') as z:
            ids,raw,prior,teacher=(z[k] for k in ('obs_ids','single','prior','equal_r18'))
        index=pd.Index(ids);mm=meta.iloc[ids].reset_index(drop=True);y=mm.class_id.to_numpy()
        with np.load(PILOT/port/'final_model.npz') as z:
            history={k:z[k] for k in z.files}
        with np.load(CONTROL/port/'final_model.npz') as z:
            control={k:z[k] for k in z.files}
        held=np.arange(1296,len(ids))
        for name,model,arm in [('image_history',history,'image_history'),('class_image_history',history,'class_image_history'),('image_class_target',control,'image_history')]:
            x=raw[held][:,COLS]
            corrected=x+predict(model,x,y[held],arm)
            for state,value in [('raw',x),('corrected',corrected)]:
                geometry.append(dict(context=port,arm=name,state=state,**scatter(value,mm.iloc[held])))
        # Historical seed tags share support splits; attribution counts only three unique draws.
        seed=SEEDS[0]
        for draw in range(3):
            with np.load(TEACHER/port/f'{seed}_{draw}_single_native.npz') as z:
                so,qo,baseline=(z[k] for k in ('support_obs_ids','query_obs_ids','scores'))
            s,q=index.get_indexer(so),index.get_indexer(qo)
            sy,truth=y[s],y[q];bp=baseline.argmax(1)
            ds=predict(control,raw[s][:,COLS],sy,'image_history')
            dq=predict(control,raw[q][:,COLS],np.zeros(len(q),dtype=int),'image_history')
            ps=adapt_support(raw[s],ds,1.)
            candidate=raw[s][:,COLS]+ds
            rc=raw[s][:,COLS].mean(0);tc=candidate.mean(0)
            gain=np.sqrt(np.square(raw[s][:,COLS]-rc).sum(1).mean()/np.square(candidate-tc).sum(1).mean())
            pq=raw[q].copy();pq[:,COLS]=(raw[q][:,COLS]+dq-tc)*gain+rc
            for role in ('raw','support_only','query_only','both'):
                rs=ps if role in ('support_only','both') else raw[s]
                rq=pq if role in ('query_only','both') else raw[q]
                scores=prior[q]+(rq@rs.T)@coefficients(rs@rs.T,prior[s],sy,8.)
                roles.append(dict(context=port,draw=draw,role=role,**metric(truth,scores.argmax(1))))
                if role=='support_only':
                    with np.load(CONTROL/port/f'{seed}_{draw}_image_class_target.npz') as z:
                        np.testing.assert_allclose(scores,z['scores'],rtol=0,atol=1e-10)
            for name,root in [('image_history',PILOT),('class_image_history',PILOT),('image_class_target',CONTROL)]:
                with np.load(root/port/f'{seed}_{draw}_{name}.npz') as z:
                    scores=z['scores']
                with np.load(BASE/'case_role_headroom_20261012'/port/f'{seed}_{draw}_support_only_matched.npz') as z:
                    taught=z['scores']
                pred=scores.argmax(1)
                change=(pred==truth).astype(float)-(bp==truth).astype(float)
                nc=np.bincount(truth,minlength=8)
                rowweight=100/((nc>0).sum()*nc[truth])
                cm=scores-scores.mean(1,keepdims=True)
                tm=taught-taught.mean(1,keepdims=True)
                readout.append(dict(context=port,draw=draw,arm=name,teacher_score_mse=float(np.mean((cm-tm)**2)),
                    baseline_teacher_score_mse=float(np.mean((baseline-baseline.mean(1,keepdims=True)-tm)**2)),
                    ba=float(metric(truth,pred)['ba'])))
                ships=mm.iloc[q].mmsi.to_numpy()
                for ship in np.unique(ships):
                    m=ships==ship
                    contributions.append(dict(context=port,draw=draw,arm=name,mmsi=str(ship),ba_contribution=float(np.sum(change[m]*rowweight[m]))))
                for j in np.flatnonzero(change):
                    events.append(dict(context=port,draw=draw,arm=name,obs_id=int(qo[j]),mmsi=str(ships[j]),class_id=int(truth[j]),rescue=bool(change[j]>0)))
                metrics.append(dict(context=port,draw=draw,arm=name,**metric(truth,pred),delta_ba=float(np.sum(change*rowweight))))
    pd.DataFrame(geometry).to_csv(OUT/'geometry.csv',index=False)
    pd.DataFrame(readout).to_csv(OUT/'readout.csv',index=False)
    pd.DataFrame(roles).to_csv(OUT/'role_intervention.csv',index=False)
    pd.DataFrame(roles).groupby('role')[['oa','ba']].mean().to_csv(OUT/'role_summary.csv')
    ev=pd.DataFrame(events);ev.to_csv(OUT/'events_unique_draws.csv',index=False)
    con=pd.DataFrame(contributions)
    byship=con.groupby(['arm','mmsi']).ba_contribution.sum()/9
    byship.to_csv(OUT/'ship_contributions.csv')
    attribution={}
    for arm in ('image_history','class_image_history','image_class_target'):
        es=ev[ev.arm==arm];values=byship.loc[arm].sort_values(ascending=False)
        top=values.index[0]
        removal=[]
        root=CONTROL if arm=='image_class_target' else PILOT
        for port in PORTS:
            for draw in range(3):
                with np.load(root/port/f'{SEEDS[0]}_{draw}_{arm}.npz') as z:
                    qo,scores=z['query_obs_ids'],z['scores']
                with np.load(TEACHER/port/f'{SEEDS[0]}_{draw}_single_native.npz') as z:
                    baseline=z['scores']
                keep=meta.iloc[qo].mmsi.astype(str).to_numpy()!=top
                truth=meta.iloc[qo].class_id.to_numpy()[keep]
                removal.append(metric(truth,scores[keep].argmax(1))['ba']-metric(truth,baseline[keep].argmax(1))['ba'])
        attribution[arm]=dict(rescued_unique_ships=int(es[es.rescue].mmsi.nunique()),harmed_unique_ships=int(es[~es.rescue].mmsi.nunique()),
            rescued_unique_observations=int(es[es.rescue].obs_id.nunique()),harmed_unique_observations=int(es[~es.rescue].obs_id.nunique()),
            top_ship=str(top),top_ship_ba_contribution=float(values.iloc[0]),total_ba_delta=float(values.sum()),
            remove_top_ship_rebalanced_delta_ba=float(np.mean(removal)),unique_readouts=9,
            scope='Same 3 draws across 3 geometry contexts; ships/observations may be rescued and harmed in different readouts')
    write(OUT/'ATTRIBUTION.json',attribution)
    print(table.subtract(table['raw'],axis=0).to_string(),flush=True)
    print(pd.DataFrame(roles).groupby('role')[['oa','ba']].mean().to_string(),flush=True)
    print(attribution,flush=True)


if __name__=='__main__':
    main()
