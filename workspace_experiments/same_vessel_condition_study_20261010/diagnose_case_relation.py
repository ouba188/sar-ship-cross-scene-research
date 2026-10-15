"""Saved-score replay, vessel attribution, and shape/scale interventions."""

from case_relation_headroom import *
from full_teacher_data import bank_from_extra
from unified_head_expert import _conditions
import json

HEADROOM = OUT
STUDENT = BASE/'case_relation_student_20261012'


def headroom_attribution(meta):
    records,checks = [],[]
    for port in PORTS:
        for seed in SEEDS:
            for draw in range(3):
                path = HEADROOM/port/f'{seed}_{draw}_equal_r18_matched.npz'
                with np.load(path) as z:
                    alpha = __import__('unified_head_core').coefficients(
                        z['kernel_support'],z['prior_support'],z['support_labels'],8.)
                    replay = z['prior_query']+z['kernel_query']@alpha
                    np.testing.assert_allclose(replay,z['scores'],atol=1e-12,rtol=1e-12)
                    checks.append(float(np.max(abs(replay-z['scores']))))
                    ids=z['query_obs_ids']; pred=z['scores'].argmax(1); old=z['baseline'].argmax(1)
                truth=meta.iloc[ids].class_id.to_numpy()
                _,cnt=np.unique(truth,return_counts=True)
                nc=len(cnt); class_counts=np.bincount(truth,minlength=8)
                for j,obs in enumerate(ids):
                    delta=int(pred[j]==truth[j])-int(old[j]==truth[j])
                    if delta:
                        records.append(dict(context=port,seed=seed,draw=draw,obs_id=int(obs),
                            mmsi=str(meta.iloc[obs].mmsi),class_id=int(truth[j]),
                            contribution=100*delta/(nc*class_counts[truth[j]]*18),rescue=delta>0))
    frame=pd.DataFrame(records)
    ship=frame.groupby('mmsi').contribution.sum().sort_values(ascending=False)
    ship.to_csv(HEADROOM/'vessel_ba_contribution.csv')
    frame.to_csv(HEADROOM/'weighted_events.csv',index=False)
    top=ship.index[0]
    removed=[]
    for port in PORTS:
        for seed in SEEDS:
            for draw in range(3):
                with np.load(HEADROOM/port/f'{seed}_{draw}_equal_r18_matched.npz') as z:
                    ids=z['query_obs_ids']; pred=z['scores'].argmax(1); old=z['baseline'].argmax(1)
                use=meta.iloc[ids].mmsi.astype(str).to_numpy()!=top
                truth=meta.iloc[ids].class_id.to_numpy()[use]
                removed.append(metric(truth,pred[use])['ba']-metric(truth,old[use])['ba'])
    write(HEADROOM/'ATTRIBUTION.json',dict(replay_max_error=max(checks),
        rescue_unique_ships=int(frame.loc[frame.rescue].mmsi.nunique()),
        harm_unique_ships=int(frame.loc[~frame.rescue].mmsi.nunique()),
        rescued_unique_observations=int(frame.loc[frame.rescue].obs_id.nunique()),
        harmed_unique_observations=int(frame.loc[~frame.rescue].obs_id.nunique()),
        net_ba_delta=float(ship.sum()),top_ship=str(top),top_ship_contribution=float(ship.iloc[0]),
        top_ship_removed_rebalanced_ba_delta=float(np.mean(removed)),
        unit='Events repeated across contexts/support draws; unique ships may appear in both directions'))


def diagnose_students(meta):
    if not (STUDENT/'COMPLETE.json').exists():
        raise RuntimeError('Complete student arms before diagnosis')
    out=STUDENT/'diagnostics'; out.mkdir(exist_ok=True)
    shapes,readability,gradients=[],[],[]
    replay_error=0.
    incidence=_conditions()
    for port in PORTS:
        context=ExpertContext(ROOT/'learned_frequency_residual_20261009/tasks/self24ports'/port)
        context._prepare_source()
        with np.load(HEADROOM/port/'maps.npz') as z:
            ids=z['obs_ids']; prior=z['prior']
        rows=meta.iloc[ids].manifest_row.to_numpy()
        pos=pd.Index(context.source_rows).get_indexer(rows)
        index=pd.Index(ids); cols=r18_columns()
        for seed in SEEDS:
            phi={}
            for arm in ('task','score','relation','joint'):
                path=STUDENT/port/str(seed)/arm
                extra=np.load(path/'extra.npy')
                bank=bank_from_extra(context.bank[pos],extra,context.geometry)
                phi[arm]=numpy_phi(bank,context.raw[rows],extra,incidence[rows],context.geometry)
                readability.append(dict(context=port,seed=seed,arm=arm,
                    **json.loads((path/'geometry.json').read_text())))
                for step in (0,31,63):
                    gradients.append(dict(context=port,seed=seed,arm=arm,step=step,
                        **json.loads((path/f'gradient_{step:02d}.json').read_text())))
            for draw_seed in SEEDS:
                for draw in range(3):
                    with np.load(HEADROOM/port/f'{draw_seed}_{draw}_single_native.npz') as z:
                        s=index.get_indexer(z['support_obs_ids']); qo=z['query_obs_ids']; q=index.get_indexer(qo)
                        sy=z['support_labels']
                    truth=meta.iloc[qo].class_id.to_numpy()
                    for arm in phi:
                        computed=score(phi[arm]@phi[arm][s].T,prior,s,sy)[0][q]
                        with np.load(STUDENT/port/str(seed)/arm/f'{draw_seed}_{draw}.npz') as z:
                            np.testing.assert_allclose(computed,z['scores'],atol=2e-10,rtol=2e-10)
                            replay_error=max(replay_error,float(np.max(abs(computed-z['scores']))))
                    task=phi['task']; rel=phi['relation']
                    shape,energy=match_energy(task,rel,s,cols)
                    gain=np.sqrt(energy['teacher_energy']/energy['raw_energy'])
                    scale=task.copy()
                    tm=task[s][:,cols].mean(0); rm=rel[s][:,cols].mean(0)
                    scale[:,cols]=(task[:,cols]-tm)*gain+rm
                    reconstructed=shape.copy(); reconstructed[:,cols]=(shape[:,cols]-tm)*gain+rm
                    np.testing.assert_allclose(reconstructed,rel,atol=1e-10,rtol=1e-10)
                    for name,value in [('task',task),('relation_shape_only',shape),
                                        ('relation_scale_centroid_only',scale),('relation_full',rel)]:
                        pred=score(value@value[s].T,prior,s,sy)[0][q].argmax(1)
                        shapes.append(dict(context=port,seed=seed,draw_seed=draw_seed,draw=draw,
                            arm=name,**metric(truth,pred),relative_energy=float(gain**2)))
        del context,phi;gc.collect()
    pd.DataFrame(readability).to_csv(out/'geometry.csv',index=False)
    pd.DataFrame(gradients).to_csv(out/'gradients.csv',index=False)
    frame=pd.DataFrame(shapes);frame.to_csv(out/'shape_scale.csv',index=False)
    frame.groupby('arm')[['oa','ba']].mean().to_csv(out/'shape_scale_summary.csv')
    write(out/'REPLAY.json',dict(student_readouts=144,max_error=replay_error,
        shape_and_centroid_scale_composition_exact=True,new_fits=0))
    print(frame.groupby('arm')[['oa','ba']].mean().to_string(),flush=True)


def diagnose_repair(meta):
    repair=BASE/'case_relation_readout_repair_20261012'
    errors,classes=[],[]
    max_error=0.
    incidence=_conditions()
    for port in PORTS:
        context=ExpertContext(ROOT/'learned_frequency_residual_20261009/tasks/self24ports'/port)
        context._prepare_source()
        with np.load(HEADROOM/port/'maps.npz') as z:
            ids=z['obs_ids'];prior=z['prior']
        rows=meta.iloc[ids].manifest_row.to_numpy()
        pos=pd.Index(context.source_rows).get_indexer(rows);index=pd.Index(ids)
        for seed in SEEDS:
            extra=np.load(repair/port/str(seed)/'readout/extra.npy')
            bank=bank_from_extra(context.bank[pos],extra,context.geometry)
            phi=numpy_phi(bank,context.raw[rows],extra,incidence[rows],context.geometry)
            for ds in SEEDS:
                for draw in range(3):
                    name=f'{ds}_{draw}.npz'
                    with np.load(HEADROOM/port/f'{ds}_{draw}_equal_r18_native.npz') as z:
                        s=index.get_indexer(z['support_obs_ids']);q=index.get_indexer(z['query_obs_ids'])
                        teacher=z['scores'];sy=z['support_labels'];qo=z['query_obs_ids']
                    replay=score(phi@phi[s].T,prior,s,sy)[0][q]
                    with np.load(repair/port/str(seed)/'readout'/name) as z:
                        np.testing.assert_allclose(replay,z['scores'],atol=2e-10,rtol=2e-10)
                        max_error=max(max_error,float(np.max(abs(replay-z['scores']))))
                    truth=meta.iloc[qo].class_id.to_numpy()
                    for arm in ('task','score','relation','joint','readout'):
                        root=repair if arm=='readout' else STUDENT
                        with np.load(root/port/str(seed)/arm/name) as z:
                            scores=z['scores'];pred=scores.argmax(1)
                        diff=scores-teacher;diff-=diff.mean(1,keepdims=True)
                        counts=np.bincount(truth,minlength=8)
                        w=len(truth)/((counts>0).sum()*counts[truth])
                        errors.append(dict(context=port,seed=seed,draw_seed=ds,draw=draw,arm=arm,
                            mse=float(np.mean(diff**2)),balanced_mse=float(np.mean(w[:,None]*diff**2))))
                        for c in np.unique(truth):
                            keep=truth==c
                            classes.append(dict(context=port,seed=seed,draw_seed=ds,draw=draw,arm=arm,
                                class_id=int(c),recall=float(100*np.mean(pred[keep]==c)),n=int(keep.sum())))
        del context;gc.collect()
    error_frame=pd.DataFrame(errors);error_frame.to_csv(repair/'all_adapted_teacher_errors.csv',index=False)
    error_frame.groupby('arm')[['mse','balanced_mse']].mean().to_csv(repair/'teacher_error_summary.csv')
    frame=pd.DataFrame(classes);frame.to_csv(repair/'class_metrics.csv',index=False)
    table=frame.groupby(['class_id','arm']).recall.mean().unstack('arm')
    table.sub(table.task,axis=0).to_csv(repair/'class_delta.csv')
    write(repair/'REPLAY.json',dict(readouts=36,max_error=max_error,
        no_query_labels_or_identity_used_in_score_reconstruction=True))
    print('REPAIR',error_frame.groupby('arm')[['mse','balanced_mse']].mean().to_string(),flush=True)
    print(table.sub(table.task,axis=0).to_string(),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--headroom-only',action='store_true')
    parser.add_argument('--repair-only',action='store_true')
    args=parser.parse_args()
    meta=pd.read_csv(BASE/'observations.csv')
    if args.repair_only:
        diagnose_repair(meta)
    else:
        headroom_attribution(meta)
        if not args.headroom_only:
            diagnose_students(meta)
