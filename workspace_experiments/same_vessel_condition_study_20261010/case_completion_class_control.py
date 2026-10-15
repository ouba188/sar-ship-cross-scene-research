"""Matched image-to-residual control with other-vessel class targets, no history."""

from case_support_completion import *

OUT = BASE/'case_completion_class_control_20261012'


def fit_control(x, meta):
    w = weights(meta)
    y, ships = meta.class_id.to_numpy(), meta.mmsi.to_numpy()
    target = np.zeros_like(x)
    for c in np.unique(y):
        members = np.flatnonzero(y==c)
        for ship in np.unique(ships[members]):
            focal = members[ships[members]==ship]
            others = members[ships[members]!=ship]
            if not len(others):
                raise ValueError('Class control needs at least two fitting vessels per class')
            target[focal] = np.average(x[others],axis=0,weights=w[others])
    delta = target-x
    mx,md = np.average(x,axis=0,weights=w),np.average(delta,axis=0,weights=w)
    xc,dc = x-mx,delta-md
    gram = xc.T@(w[:,None]*xc)
    ridge = max(.1*np.trace(gram)/len(gram),1e-12)
    coef = np.linalg.solve(gram+ridge*np.eye(len(gram)),xc.T@(w[:,None]*dc))
    return dict(image_mean_x=mx,image_mean_d=md,image_coef=coef,image_ridge=ridge)


def main():
    OUT.mkdir(exist_ok=True)
    write(OUT/'PROTOCOL.json',dict(
        question='Does same-vessel history outperform matched generic class-average supervision?',
        matched='Exactly image_history fitting rows, 576D inputs/outputs, residual ridge, weights, source folds, doses and solver',
        only_difference='Replace other-product same-ship target with same-class other-ship mean; exclude entire focal vessel',
        query='No query labels or history; support completion also does not input support labels',
        role='Mechanism comparison after previous pilot; source-only dose selection; historical held reuse explicitly acknowledged',
        no_extra_neural_training=True))
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(BASE/'crossfit_evidence_20261011/source_folds.npz') as z:
        fit_ids,folds=z['fit_obs_ids'],z['assignment']
    nfit=len(fit_ids)
    cv,models,cache=[],{},{}
    for port in PORTS:
        dest=OUT/port;dest.mkdir(exist_ok=True)
        with np.load(TEACHER/port/'maps.npz') as z:
            ids,raw,prior,multi=(z[k] for k in ('obs_ids','single','prior','multi'))
        np.testing.assert_array_equal(ids[:nfit],fit_ids)
        mm=meta.iloc[ids].reset_index(drop=True);y=mm.class_id.to_numpy()
        with np.load(BASE/'case_relation_student_20261012'/port/'source_readers.npz') as z:
            oof_prior=z['fit_prior']
        for fold in range(3):
            fitting=np.flatnonzero((folds!=fold)&multi[:nfit]);val=np.flatnonzero(folds==fold)
            model=fit_control(raw[fitting][:,COLS],mm.iloc[fitting])
            np.savez_compressed(dest/f'fold_{fold}_model.npz',**model)
            for draw in range(3):
                ls,lq=identity_episode(mm.iloc[val],np.random.default_rng(101200+fold*10+draw),query_size=None)
                s,q=val[ls],val[lq]
                delta=predict(model,raw[s][:,COLS],y[s],'image_history')
                for dose in DOSES:
                    scores=classify(raw[s],raw[q],oof_prior[s],oof_prior[q],y[s],delta,dose)
                    cv.append(dict(context=port,fold=fold,draw=draw,dose=dose,**metric(y[q],scores.argmax(1))))
        fitting=np.flatnonzero(multi[:nfit])
        models[port]=fit_control(raw[fitting][:,COLS],mm.iloc[fitting])
        np.savez_compressed(dest/'final_model.npz',**models[port])
        cache[port]=(ids,raw,prior,mm,y)
        print('FITTED',port,flush=True)
    cv=pd.DataFrame(cv);cv.to_csv(OUT/'source_oof.csv',index=False)
    summary=cv.groupby('dose')[['oa','ba']].mean().reset_index()
    dose=float(summary.sort_values(['ba','dose'],ascending=[False,True],kind='stable').iloc[0].dose)
    summary.to_csv(OUT/'selection.csv',index=False)
    write(OUT/'FROZEN_CHOICE.json',dict(dose=dose))
    results,classes=[] ,[]
    for port,(ids,raw,prior,mm,y) in cache.items():
        index=pd.Index(ids)
        for seed in SEEDS:
            for draw in range(3):
                with np.load(TEACHER/port/f'{seed}_{draw}_single_native.npz') as z:
                    so,qo,baseline=(z[k] for k in ('support_obs_ids','query_obs_ids','scores'))
                s,q=index.get_indexer(so),index.get_indexer(qo)
                assert (s>=nfit).all() and (q>=nfit).all()
                delta=predict(models[port],raw[s][:,COLS],y[s],'image_history')
                scores=classify(raw[s],raw[q],prior[s],prior[q],y[s],delta,dose)
                pred,bp=scores.argmax(1),baseline.argmax(1)
                row=dict(context=port,seed=seed,draw=draw,arm='image_class_target',dose=dose)
                results.append(dict(**row,**metric(y[q],pred),rescue=int(np.sum((bp!=y[q])&(pred==y[q]))),harm=int(np.sum((bp==y[q])&(pred!=y[q])))))
                for c in np.unique(y[q]):
                    mask=y[q]==c
                    classes.append(dict(**row,class_id=int(c),n=int(mask.sum()),recall=float(100*np.mean(pred[mask]==c))))
                np.savez_compressed(OUT/port/f'{seed}_{draw}_image_class_target.npz',support_obs_ids=so,query_obs_ids=qo,scores=scores)
    frame=pd.DataFrame(results);frame.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'classes.csv',index=False)
    frame.groupby('context')[['oa','ba']].mean().to_csv(OUT/'contexts.csv')
    frame.groupby('arm')[['oa','ba','rescue','harm']].mean().to_csv(OUT/'summary.csv')
    write(OUT/'COMPLETE.json',dict(linear_fits=12,neural_fits=0,held_selected=False))
    print('FROZEN DOSE',dose,flush=True)
    print(frame.groupby('arm')[['oa','ba','rescue','harm']].mean().to_string(),flush=True)


if __name__ == '__main__':
    main()
