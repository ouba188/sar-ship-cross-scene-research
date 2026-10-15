"""Bounded label-aware support completion; queries never use labels or history."""

from full_teacher_data import BASE, PORTS, write
from case_relation_headroom import OUT as TEACHER, SEEDS, metric, r18_columns
from unified_head_core import coefficients
from explicit_candidate_projection import identity_episode
import numpy as np
import pandas as pd

OUT = BASE/'case_support_completion_20261012'
ARMS = ('class_center', 'class_history_mean', 'image_history', 'class_image_history')
DOSES = (0., .25, .5, 1.)
COLS = r18_columns()


def weights(meta):
    result = np.zeros(len(meta))
    for _, group in meta.reset_index(drop=True).groupby('class_id'):
        for _, ship in group.groupby('mmsi'):
            result[ship.index] = 1/(meta.class_id.nunique()*group.mmsi.nunique()*len(ship))
    return result


def fit_models(x, teacher, y, meta):
    w = weights(meta)
    delta = teacher-x
    means, residuals = np.zeros((8,x.shape[1])), np.zeros((8,x.shape[1]))
    present = np.zeros(8,bool)
    for c in np.unique(y):
        m = y == c
        means[c] = np.average(x[m],axis=0,weights=w[m])
        residuals[c] = np.average(delta[m],axis=0,weights=w[m])
        present[c] = True
    fitted = dict(means=means,residuals=residuals,present=present)
    for kind in ('image','class_image'):
        mx = means[y] if kind == 'class_image' else np.average(x,axis=0,weights=w)
        md = residuals[y] if kind == 'class_image' else np.average(delta,axis=0,weights=w)
        xc, dc = x-mx, delta-md
        gram = xc.T@(w[:,None]*xc)
        ridge = max(.1*np.trace(gram)/len(gram),1e-12)
        coef = np.linalg.solve(gram+ridge*np.eye(len(gram)),xc.T@(w[:,None]*dc))
        fitted[kind+'_coef'] = coef
        fitted[kind+'_ridge'] = ridge
        if kind == 'image':
            fitted['image_mean_x'], fitted['image_mean_d'] = mx,md
    return fitted


def predict(model, x, labels, arm):
    if arm == 'class_center':
        return (model['means'][labels]-x)*model['present'][labels,None]
    if arm == 'class_history_mean':
        return model['residuals'][labels]
    if arm == 'image_history':
        return model['image_mean_d']+(x-model['image_mean_x'])@model['image_coef']
    if arm == 'class_image_history':
        answer = model['residuals'][labels]+(x-model['means'][labels])@model['class_image_coef']
        return answer*model['present'][labels,None]
    raise ValueError(arm)


def adapt_support(raw_s, delta, dose):
    if dose == 0:
        return raw_s.copy()
    result = raw_s.copy()
    block = raw_s[:,COLS]+dose*delta
    rc, tc = raw_s[:,COLS].mean(0), block.mean(0)
    re = np.square(raw_s[:,COLS]-rc).sum(1).mean()
    te = np.square(block-tc).sum(1).mean()
    result[:,COLS] = (block-tc)*np.sqrt(re/max(te,1e-20))+rc
    return result


def classify(raw_s, raw_q, prior_s, prior_q, labels, delta, dose):
    ps = adapt_support(raw_s,delta,dose)
    alpha = coefficients(ps@ps.T,prior_s,labels,8.)
    return prior_q+(raw_q@ps.T)@alpha


def main():
    OUT.mkdir(exist_ok=True)
    write(OUT/'PROTOCOL.json',dict(
        premise='Matched support-only real-history teacher has +1.620321 BA points',
        primary='class_image_history versus unchanged and class_center controls',
        arms=ARMS,doses=DOSES,ridge='Fixed .1 trace(weighted feature Gram)/dimension',
        source='1296 source-fit, only 1040 multi-observation rows train completion; equal class/vessel weights',
        controls='All predictors and raw class centers use exactly the same fitting rows',
        selection='Three vessel folds x three support draws x three geometry contexts; source OOF BA, smaller dose ties; shared dose across contexts',
        priors='Cached source OOF priors exclude validation fold ships AND all held ships',
        support='64 distinct ships per episode, SAR representation plus existing support class label',
        query='Unchanged SAR feature and prior, no query label, history, MMSI, port or AIS as model input',
        scaling='Original support centroid and centered energy restored after correction',
        evaluation='Same historical 823 source-held observations; not fresh data or target generalization',
        scope='Fixed initial full expert, not final base/expert fusion and not retrained backbone',
        no_held_parameter_selection=True))
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(BASE/'crossfit_evidence_20261011/source_folds.npz') as z:
        fit_ids, folds = z['fit_obs_ids'],z['assignment']
    nfit = len(fit_ids)
    cv, diagnostics, contexts = [], [], {}
    for port in PORTS:
        dest = OUT/port
        dest.mkdir(exist_ok=True)
        with np.load(TEACHER/port/'maps.npz') as z:
            ids, raw, teacher, prior, multi = (z[k] for k in ('obs_ids','single','equal_r18','prior','multi'))
        np.testing.assert_array_equal(ids[:nfit],fit_ids)
        mm = meta.iloc[ids].reset_index(drop=True)
        y = mm.class_id.to_numpy()
        assert not set(mm.iloc[:nfit].mmsi)&set(mm.iloc[nfit:].mmsi)
        with np.load(BASE/'case_relation_student_20261012'/port/'source_readers.npz') as z:
            oof_prior = z['fit_prior']
        for fold in range(3):
            fitting = np.flatnonzero((folds!=fold)&multi[:nfit])
            val = np.flatnonzero(folds==fold)
            assert not set(mm.iloc[fitting].mmsi)&set(mm.iloc[val].mmsi)
            model = fit_models(raw[fitting][:,COLS],teacher[fitting][:,COLS],y[fitting],mm.iloc[fitting])
            np.savez_compressed(dest/f'fold_{fold}_model.npz',**model)
            # Each fold is vessel-disjoint, including every historical teacher donor.
            for draw in range(3):
                ls,lq = identity_episode(mm.iloc[val],np.random.default_rng(101200+fold*10+draw),query_size=None)
                s,q = val[ls],val[lq]
                np.savez_compressed(dest/f'fold_{fold}_draw_{draw}.npz',support_obs_ids=ids[s],query_obs_ids=ids[q])
                for arm in ARMS:
                    delta = predict(model,raw[s][:,COLS],y[s],arm)
                    for dose in DOSES:
                        scores = classify(raw[s],raw[q],oof_prior[s],oof_prior[q],y[s],delta,dose)
                        cv.append(dict(context=port,fold=fold,draw=draw,arm=arm,dose=dose,**metric(y[q],scores.argmax(1))))
            print('OOF',port,fold,flush=True)
        fitting = np.flatnonzero(multi[:nfit])
        model = fit_models(raw[fitting][:,COLS],teacher[fitting][:,COLS],y[fitting],mm.iloc[fitting])
        np.savez_compressed(dest/'final_model.npz',**model)
        contexts[port] = (ids,raw,teacher,prior,multi,mm,y,model)
    cv = pd.DataFrame(cv)
    cv.to_csv(OUT/'source_oof.csv',index=False)
    selection, chosen = [], {}
    for arm in ARMS:
        avg = cv[cv.arm==arm].groupby('dose')[['oa','ba']].mean().reset_index()
        best = avg.sort_values(['ba','dose'],ascending=[False,True],kind='stable').iloc[0]
        chosen[arm] = float(best.dose)
        for r in avg.to_dict('records'):
            selection.append(dict(arm=arm,selected=r['dose']==chosen[arm],**r))
    pd.DataFrame(selection).to_csv(OUT/'selection.csv',index=False)
    write(OUT/'FROZEN_CHOICES.json',chosen)
    print('FROZEN',chosen,flush=True)
    results,classes,events = [],[],[]
    for port,(ids,raw,teacher,prior,multi,mm,y,model) in contexts.items():
        dest,index = OUT/port,pd.Index(ids)
        held_multi = np.flatnonzero(multi & (np.arange(len(ids))>=nfit))
        for arm in ARMS:
            delta = predict(model,raw[held_multi][:,COLS],y[held_multi],arm)
            target = teacher[held_multi][:,COLS]-raw[held_multi][:,COLS]
            w = weights(mm.iloc[held_multi])
            zero = float(w@np.square(target).mean(1))
            mse = float(w@np.square(target-delta).mean(1))
            diagnostics.append(dict(context=port,arm=arm,cohort='held_history_rows_known_class_diagnostic_only',
                zero_mse=zero,mse=mse,explained_percent=100*(1-mse/zero)))
        split_keys = []
        for seed in SEEDS:
            for draw in range(3):
                with np.load(TEACHER/port/f'{seed}_{draw}_single_native.npz') as z:
                    so,qo,baseline = (z[k] for k in ('support_obs_ids','query_obs_ids','scores'))
                split_keys.append(tuple(so.tolist()))
                s,q = index.get_indexer(so),index.get_indexer(qo)
                assert (s>=nfit).all() and (q>=nfit).all()
                bp = baseline.argmax(1)
                replay = classify(raw[s],raw[q],prior[s],prior[q],y[s],np.zeros((64,len(COLS))),0)
                np.testing.assert_allclose(replay,baseline,atol=1e-10,rtol=0)
                for arm in ('raw',)+ARMS:
                    dose = 0 if arm=='raw' else chosen[arm]
                    delta = np.zeros((len(s),len(COLS))) if arm=='raw' else predict(model,raw[s][:,COLS],y[s],arm)
                    scores = classify(raw[s],raw[q],prior[s],prior[q],y[s],delta,dose)
                    pred = scores.argmax(1)
                    row = dict(context=port,seed=seed,draw=draw,arm=arm,dose=dose)
                    results.append(dict(**row,**metric(y[q],pred),rescue=int(np.sum((bp!=y[q])&(pred==y[q]))),harm=int(np.sum((bp==y[q])&(pred!=y[q])))))
                    for c in np.unique(y[q]):
                        mask = y[q]==c
                        classes.append(dict(**row,class_id=int(c),n=int(mask.sum()),recall=float(100*np.mean(pred[mask]==c))))
                    for j in np.flatnonzero((bp==y[q])!=(pred==y[q])):
                        events.append(dict(**row,obs_id=int(qo[j]),mmsi=str(mm.iloc[q[j]].mmsi),class_id=int(y[q[j]]),before=int(bp[j]),after=int(pred[j]),rescue=bool(pred[j]==y[q[j]])))
                    np.savez_compressed(dest/f'{seed}_{draw}_{arm}.npz',support_obs_ids=so,query_obs_ids=qo,scores=scores)
        write(dest/'SPLIT_COUNTS.json',dict(historical_tags=len(split_keys),unique_support_splits=len(set(split_keys))))
    df = pd.DataFrame(results)
    for name,frame in [('metrics',df),('classes',pd.DataFrame(classes)),('events',pd.DataFrame(events)),('prediction_diagnostics',pd.DataFrame(diagnostics))]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    summary=df.groupby('arm')[['oa','ba','rescue','harm']].mean()
    summary.to_csv(OUT/'summary.csv')
    df.groupby(['context','arm'])[['oa','ba']].mean().to_csv(OUT/'contexts.csv')
    write(OUT/'COMPLETE.json',dict(neural_fits=0,linear_residual_fits=24,held_choice_used=False,backbone_unchanged=True,final_fusion_not_evaluated=True))
    print(summary.to_string(),flush=True)
    print(pd.DataFrame(diagnostics).groupby('arm').explained_percent.mean().to_string(),flush=True)


if __name__ == '__main__':
    main()
