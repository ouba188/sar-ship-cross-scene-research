"""Choose frozen metric amplitudes using support-only out-of-fold predictions."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.kernel_ridge import KernelRidge
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
OLD = ROOT/'class_reference_metric_pilot_20261014'
ORACLE = ROOT/'metric_alpha_oracle_20261014'
spec = importlib.util.spec_from_file_location('support_select_pilot', OLD/'run_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
ALPHAS = np.array([0., .25, .5])
STAGES = {'initial': OLD, 'matched': OLD/'matched_training'}


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def support_folds(labels, acquisitions, seed):
    """Hold whole acquisitions out; one-acquisition cells use unique-ship folds."""
    groups, inverse, counts = np.unique(acquisitions, return_inverse=True, return_counts=True)
    fold = np.full(len(labels), -1, dtype=int)
    load = np.zeros(4, dtype=int)
    if len(groups) > 1:
        for group in np.argsort(-counts, kind='stable'):
            k = int(load.argmin())
            fold[inverse == group] = k
            load[k] += counts[group]
        method = 'acquisition_grouped'
    else:
        rng = np.random.default_rng(seed)
        offset = 0
        for c in np.unique(labels):
            ix = np.flatnonzero(labels == c)
            rng.shuffle(ix)
            fold[ix] = (np.arange(len(ix))+offset) % 4
            offset += len(ix)
        method = 'ship_stratified_single_acquisition'
    assert np.all(fold >= 0) and len(np.unique(fold)) == 4
    if method == 'acquisition_grouped':
        for k in range(4):
            assert not set(acquisitions[fold == k]) & set(acquisitions[fold != k])
    return fold, method


def predict_fold(kernel, prior, labels, train, valid):
    coef = pilot.reader.coefficients(kernel[np.ix_(train, train)], prior[train], labels[train], 8.)
    return softmax(prior[valid]+kernel[np.ix_(valid, train)]@coef, axis=1)


def base_oof(kernel, prior, labels, fold):
    result = np.zeros_like(prior)
    for k in range(4):
        train, valid = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        result[valid] = predict_fold(kernel, prior, labels, train, valid)
    return result


def metric_oof(visual_kernel, response, prior, labels, fold, base, learned):
    probabilities = []
    for alpha in ALPHAS:
        matrix = (1-alpha)*np.eye(8)+alpha*learned
        kernel = .75*visual_kernel+.25*(response@matrix@response.T)
        expert = base_oof(kernel, prior, labels, fold)
        probabilities.append(.5*(base+expert))
    return np.stack(probabilities)


def select_all():
    started = time.perf_counter()
    (OUT/'oof').mkdir(exist_ok=True)
    write('PROTOCOL.json', dict(date='2026-10-14',
        purpose='One bounded support-only selection test after held-label oracle exploration',
        alphas=ALPHAS.tolist(), primary='Pooled support OOF BA, exact ties choose smaller alpha',
        secondary='Pooled support OOF macro NLL, exact ties choose smaller alpha; no winner selection across policies',
        folds='4 folds grouped by acquisition, descending group size balanced by support count; single-acquisition cells use deterministic class-stratified unique-ship folds',
        refits='Refit BOTH base and expert on each fold training subset with lambda8; class weights from that subset only',
        final='Use all original64 supports for final prediction; frozen previous probability cache is exact full-support refit',
        stages=list(STAGES), seeds=pilot.SEEDS, scope='Existing48 scenes/3ports/5draws, same and other acquisition separately',
        freeze='All choices persisted before query probability caches or query outcome metrics are loaded',
        cost='No new metric/source/backbone fits; support-fold ridge fits only',
        caveats='Recipe/grid informed by prior query exploration; support selection itself uses no query labels. Not independent confirmation. Existing support quotas originated from original tasks. Fold train size differs from64. Missing support classes cannot contribute to CV BA.',
        prohibited='No finer grid, new gates, query-selected sampler/policy, external run or main model replacement'))
    d, f, plans, _, _, _ = pilot.load_data()
    enrollment = [e for e in json.loads((OLD/'ENROLLMENT.json').read_text(encoding='utf-8')) if e['role'] == 'held']
    models = {}
    for stage, path in STAGES.items():
        for held in sorted({e['held'] for e in enrollment}):
            for seed in pilot.SEEDS:
                name = f"{held.replace(' ', '_')}_class_mean_task_{seed}.npz"
                with np.load(path/'models'/name) as a:
                    models[(stage,held,seed)] = a['metric']
    choices, cv, folds_log, checks = [], [], [], []
    number = {}
    for ei, e in enumerate(enrollment):
        held = e['held']; ti = number.get(held, 0); number[held] = ti+1
        s = np.array(e['support']); sf = d.iloc[s]
        assert len(s) == sf.mmsi.nunique() == 64
        ix = pilot.indices(f,s)
        sy = sf.class_id.to_numpy(int)
        acquisition = sf.acquisition.to_numpy()
        fold, method = support_folds(sy, acquisition, 20261014+e['episode']*10+e['draw'])
        key = {k:e[k] for k in ['port','episode','cohort','draw','cell']}
        fold_missing = np.zeros(64, dtype=bool)
        for k in range(4):
            train, valid = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            fold_missing[valid] = ~np.isin(sy[valid],sy[train])
            folds_log.append(dict(**key,fold=k,method=method,train_n=len(train),valid_n=len(valid),
                train_classes=len(np.unique(sy[train])),valid_classes=len(np.unique(sy[valid])),
                missing_class_queries=int(fold_missing[valid].sum()),
                train_acquisitions=len(np.unique(acquisition[train])),valid_acquisitions=len(np.unique(acquisition[valid]))))
        start = time.perf_counter()
        kb = f['b'][ix]@f['b'][ix].T+1
        kv = f['z'][ix]@f['z'][ix].T+1
        pb, pd_, response = f['pb'][ix], f['pd'][ix], f['class_mean'][ix]
        base = base_oof(kb,pb,sy,fold)
        common_ms = 1000*(time.perf_counter()-start)
        all_prob = []
        for stage in STAGES:
            stage_prob = []
            for seed in pilot.SEEDS:
                start = time.perf_counter()
                prob = metric_oof(kv,response,pd_,sy,fold,base,models[(stage,held,seed)])
                ms = common_ms+1000*(time.perf_counter()-start)
                stage_prob.append(prob)
                values = [pilot.metrics(sy,p) for p in prob]
                for alpha,v in zip(ALPHAS,values):
                    cv.append(dict(stage=stage,**key,seed=seed,alpha=alpha,**v))
                ba = np.array([v['ba'] for v in values])
                nll = np.array([v['macro_nll'] for v in values])
                decisions = (prob.argmax(2) != prob[0].argmax(1)).sum(1)
                for policy,best in [('support_ba',int(ba.argmax())),('support_nll',int(nll.argmin()))]:
                    choices.append(dict(stage=stage,**key,seed=seed,policy=policy,alpha=ALPHAS[best],
                        cv_ba=ba[best],cv_ba_gain=ba[best]-ba[0],cv_nll=nll[best],
                        cv_nll_gain=nll[best]-nll[0],ba_tied_candidates=int((ba == ba.max()).sum()),
                        changed_oof_predictions=int(decisions[best]),support_classes=len(np.unique(sy)),
                        support_acquisitions=len(np.unique(acquisition)),fold_missing_n=int(fold_missing.sum()),
                        selection_ms=ms,fold_method=method))
                if ti == 0 and seed == pilot.SEEDS[0]:
                    train,valid = np.flatnonzero(fold != 0),np.flatnonzero(fold == 0)
                    m=.75*np.eye(8)+.25*models[(stage,held,seed)]
                    kernel=.75*kv+.25*response@m@response.T
                    root=pilot.reader.class_weights(sy[train])
                    target=2*np.eye(8)[sy[train]]-1-pd_[train]
                    independent=KernelRidge(alpha=8.,kernel='precomputed').fit(kernel[np.ix_(train,train)],target,sample_weight=root)
                    pe=softmax(pd_[valid]+independent.predict(kernel[np.ix_(valid,train)]),axis=1)
                    independent_b=KernelRidge(alpha=8.,kernel='precomputed').fit(kb[np.ix_(train,train)],2*np.eye(8)[sy[train]]-1-pb[train],sample_weight=root)
                    bp=softmax(pb[valid]+independent_b.predict(kb[np.ix_(valid,train)]),axis=1)
                    expected=.5*(pe+bp)
                    np.testing.assert_allclose(prob[1,valid],expected,atol=1e-11,rtol=1e-11)
                    checks.append(dict(stage=stage,port=held,kind='independent_full_fold_krr',error=float(abs(expected-prob[1,valid]).max())))
            all_prob.append(stage_prob)
        np.savez_compressed(OUT/'oof'/f"{held.replace(' ','_')}_{ti:03d}.npz",full=np.array(all_prob),
            s=s,sy=sy,fold=fold,acquisition=acquisition.astype(str),alphas=ALPHAS,seeds=pilot.SEEDS,stages=np.array(list(STAGES)))
        if ei % 80 == 0:
            print('SUPPORT_SELECTION',ei,'/',len(enrollment),flush=True)
    for name,rows in [('choices',choices),('cv_metrics',cv),('folds',folds_log),('checks',checks)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    write('SELECTION_FROZEN.json',dict(seconds=time.perf_counter()-started,support_cells=len(enrollment),
        choices=len(choices),expert_fold_solves=len(enrollment)*2*3*3*4,base_fold_solves=len(enrollment)*4,
        query_prediction_caches_opened=0,query_metrics_used_for_selection=False))
    return d,f,enrollment


def evaluate(d,f,enrollment):
    assert (OUT/'SELECTION_FROZEN.json').exists()
    choices=pd.read_csv(OUT/'choices.csv')
    keyed=choices.set_index(['stage','port','episode','draw','cell','seed','policy'])
    records,events,changes,checks=[],[],[],[]
    number={}
    for e in enrollment:
        held=e['held'];ti=number.get(held,0);number[held]=ti+1
        key={k:e[k] for k in ['port','episode','cohort','draw','cell']}
        for stage in STAGES:
            with np.load(ORACLE/'predictions'/f"{stage}_{held.replace(' ','_')}_{ti:03d}.npz") as a:
                np.testing.assert_array_equal(a['s'],e['support'])
                np.testing.assert_array_equal(a['q'],e['query'])
                np.testing.assert_array_equal(a['seeds'],pilot.SEEDS)
                y,sy,q,s=a['y'],a['sy'],a['q'],a['s']
                np.testing.assert_array_equal(y,d.iloc[q].class_id)
                indices=[int(np.flatnonzero(a['alphas']==alpha)[0]) for alpha in ALPHAS]
                probs=a['full'][:,indices].copy()
            if ti == 0:
                qi,si=pilot.indices(f,q),pilot.indices(f,s)
                with np.load(STAGES[stage]/'models'/f"{held.replace(' ','_')}_class_mean_task_{pilot.SEEDS[0]}.npz") as a:
                    m=.75*np.eye(8)+.25*a['metric']
                rs,rq=f['class_mean'][si],f['class_mean'][qi]
                ks=.75*(f['z'][si]@f['z'][si].T+1)+.25*rs@m@rs.T
                kq=.75*(f['z'][qi]@f['z'][si].T+1)+.25*rq@m@rs.T
                coef=pilot.reader.coefficients(ks,f['pd'][si],sy,8.)
                ab=pilot.reader.coefficients(f['b'][si]@f['b'][si].T+1,f['pb'][si],sy,8.)
                pe=softmax(f['pd'][qi]+kq@coef,axis=1)
                pb=softmax(f['pb'][qi]+(f['b'][qi]@f['b'][si].T+1)@ab,axis=1)
                expected=.5*(pe+pb)
                np.testing.assert_allclose(expected,probs[0,1],atol=1e-10,rtol=1e-10)
                checks.append(dict(stage=stage,port=held,kind='full_support_cache_replay',error=float(abs(expected-probs[0,1]).max())))
            covered=np.isin(y,sy)
            for si,seed in enumerate(pilot.SEEDS):
                baseline=probs[si,0];before=baseline.argmax(1)
                policies=[(f'fixed_{alpha:g}',alpha) for alpha in ALPHAS]
                for policy in ['support_ba','support_nll']:
                    selected=keyed.loc[(stage,held,e['episode'],e['draw'],e['cell'],seed,policy)]
                    policies.append((policy,float(selected.alpha)))
                for policy,alpha in policies:
                    prob=probs[si,int(np.flatnonzero(ALPHAS==alpha)[0])]
                    pred=prob.argmax(1)
                    item=dict(stage=stage,**key,seed=seed,policy=policy,alpha=alpha)
                    records.append(dict(**item,**pilot.metrics(y,prob),query_n=len(y),support_classes=len(np.unique(sy)),
                        query_classes=len(np.unique(y)),uncovered_query_n=int((~covered).sum()),
                        uncovered_query_classes=len(set(y)-set(sy))))
                    for c in np.unique(y):
                        mask=y==c;b=before[mask]==c;p=pred[mask]==c
                        events.append(dict(**item,class_id=int(c),query_n=int(mask.sum()),support_n=int((sy==c).sum()),
                            rescues=int((~b&p).sum()),harms=int((b&~p).sum()),recall=100*p.mean(),baseline_recall=100*b.mean()))
                    if policy.startswith('support_'):
                        for j in np.flatnonzero((before==y)!=(pred==y)):
                            changes.append(dict(**item,row=int(q[j]),ship=d.iloc[q[j]].mmsi,class_id=int(y[j]),
                                support_n=int((sy==y[j]).sum()),event='rescue' if pred[j]==y[j] else 'harm'))
    for name,rows in [('metrics',records),('class_events',events),('changed_cases',changes),('replay_checks',checks)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    write('EVALUATION_COMPLETE.json',dict(metrics=len(records),class_events=len(events),new_training_fits=0,
        main_model_changed=False,external_runs=0))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        evaluate(*select_all())
