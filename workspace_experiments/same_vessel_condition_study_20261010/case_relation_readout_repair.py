"""One prespecified repair: distill the actual support solver's correction."""

import case_relation_student as study
from case_relation_student import *
from unified_head_expert import _conditions
import json

PARENT = study.OUT
OUT = BASE/'case_relation_readout_repair_20261012'


def report_repair(meta):
    records,events,errors = [],[],[]
    for port in PORTS:
        for seed in SEEDS:
            for ds in SEEDS:
                for draw in range(3):
                    name=f'{ds}_{draw}.npz'
                    with np.load(OUT/port/str(seed)/'readout'/name) as z:
                        ids=z['query_obs_ids']; scores=z['scores']
                    with np.load(PARENT/port/str(seed)/'task'/name) as z:
                        np.testing.assert_array_equal(ids,z['query_obs_ids'])
                        before=z['scores'].argmax(1)
                    with np.load(study.HEADROOM/port/f'{ds}_{draw}_equal_r18_native.npz') as z:
                        teacher=z['scores']
                    truth=meta.iloc[ids].class_id.to_numpy(); pred=scores.argmax(1)
                    delta=scores-teacher;delta-=delta.mean(1,keepdims=True)
                    errors.append(dict(context=port,seed=seed,draw_seed=ds,draw=draw,
                                       arm='readout',mse=float(np.mean(delta**2))))
                    records.append(dict(context=port,seed=seed,draw_seed=ds,draw=draw,arm='readout',
                        **metric(truth,pred),rescue=int(np.sum((before!=truth)&(pred==truth))),
                        harm=int(np.sum((before==truth)&(pred!=truth)))))
                    for j in np.flatnonzero((before==truth)!=(pred==truth)):
                        row=meta.iloc[ids[j]]
                        events.append(dict(context=port,seed=seed,draw_seed=ds,draw=draw,
                            obs_id=int(ids[j]),mmsi=str(row.mmsi),port=row.port,class_id=int(truth[j]),
                            before=int(before[j]),after=int(pred[j]),rescue=bool(pred[j]==truth[j])))
    result=pd.DataFrame(records)
    result.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(events).to_csv(OUT/'events.csv',index=False)
    pd.DataFrame(errors).to_csv(OUT/'adapted_teacher_error.csv',index=False)
    joint=pd.concat([pd.read_csv(PARENT/'metrics.csv'),result],ignore_index=True)
    summary=joint.groupby('arm')[['oa','ba','rescue','harm']].mean()
    summary.to_csv(OUT/'summary.csv')
    joint.groupby(['context','seed','arm'])[['oa','ba']].mean().to_csv(OUT/'context_seed.csv')
    write(OUT/'COMPLETE.json',dict(neural_fits=6,steps=64,training_seeds=2,
        original_negative_preserved=True,target_runs=0,main_model_unchanged=True,
        reason='Normalized geometry improved while adapted-score imitation and macro recognition did not',
        next='No automatic strength/step sweep; report source feasibility only'))
    print(summary.to_string(),flush=True)


def main():
    if not (PARENT/'COMPLETE.json').exists():
        raise RuntimeError('Original matched pilot required')
    OUT.mkdir(exist_ok=True)
    write(OUT/'PROTOCOL.json',dict(date='2026-10-12',parent=str(PARENT),
        change='Replace row-normalized Gram loss by class-balanced centered adapted-logit MSE',
        teacher='Same equal-history case map, same original source prior, actual support64 ridge8',
        training='Same1296sourcefit, same folds/seeds64steps/lr1e-5/taskCE; all302heldships excluded from readers',
        loss_scale='Match initial teaching gradient to0.5 oldscoreMSE; fixed thereafter, same rule as original relation arm',
        inputs='Teacher uses source training history/support labels; student prediction never uses identities/history',
        scope='One diagnosis-led correction, not new global sweep; source expert kernel only, not final fusion'))
    study.OUT=OUT;study.ARMS=('readout',)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    meta=pd.read_csv(BASE/'observations.csv')
    with np.load(BASE/'crossfit_evidence_20261011/source_folds.npz') as z:
        folds=z['assignment'];nfit=len(z['fit_obs_ids'])
    incidence=_conditions();cache=np.load(PREFIX,mmap_mode='r')
    for port in PORTS:
        study.run_student(port,meta,folds,nfit,incidence,cache)
    report_repair(meta)


if __name__ == '__main__':
    main()
