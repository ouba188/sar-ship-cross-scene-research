"""Source-trained knownness probe with nested class exclusion on frozen features."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
sys.path.insert(0, str(BASE))
import study as core
from analyze import weights, readout

METHODS = ['msp', 'source_knn', 'support_knn', 'ova', 'fixed_union', 'joint']
GROUPS = {'train': ['Jebel Ali', 'Qingdao'], 'calibration': ['Singapore', 'Ningbo-Zhoushan']}
TARGETS = ['oss2', 'f0921', 'ports/Port Klang']
SEED = 20261014


def save(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def ova_fit(x, y, classes):
    models = []
    for c in classes:
        model = LogisticRegression(C=1., class_weight='balanced', solver='liblinear',
                                   max_iter=500, random_state=0).fit(x, y == c)
        assert model.n_iter_.max() < 500
        models.append(model)
    return models


def ova_score(x, models):
    return 1. - np.column_stack([m.predict_proba(x)[:, 1] for m in models]).max(1)


def fit_geometry(values, y):
    r, c, e = values
    pca = PCA(n_components=128, svd_solver='randomized', random_state=0).fit(r)
    ssl = PCA(n_components=64, svd_solver='randomized', random_state=0).fit(e)
    x = pca.transform(r); center = x.mean(0); scale = x.std(0) + 1e-6
    xx = (x-center)/scale; x4 = (pca.transform(c)-center)/scale
    joint = np.c_[xx, ssl.transform(e)]; ec = joint.mean(0); es = joint.std(0)+1e-6
    joint = (joint-ec)/es; classes = np.unique(y)
    model = dict(pca=pca, ssl=ssl, center=center, scale=scale, ec=ec, es=es, classes=classes,
                 expert=RidgeClassifier(alpha=1., class_weight='balanced').fit(joint, y),
                 base=RidgeClassifier(alpha=1., class_weight='balanced').fit(x4, y))
    f = core.encode(values, model)
    rng = np.random.default_rng(core.SEED)
    bank = np.concatenate([rng.choice(np.flatnonzero(y == cl), min(128, (y == cl).sum()),
                                      replace=False) for cl in classes])
    model['source_bank'] = f['direction'][bank]
    return model, ova_fit(f['x'], y, classes)


def evidence(f, s, q, sy, prob, cal, model, ova):
    aq, ac = core.anomaly(f, s, q, prob, cal, model['source_bank'])
    vq = ova_score(f['x'][q], ova); vc = ova_score(f['x'][s], ova)
    pq = model['classes'][prob.argmax(1)]; pc = model['classes'][cal.argmax(1)]
    counts = np.bincount(sy, minlength=8)
    cq = counts[pq]; cc = counts[pc] - (pc == sy)
    assert (cc >= 0).all()
    xq = np.c_[vq, aq[:, 1:3], np.log1p(cq)]
    xc = np.c_[vc, ac[:, 1:3], np.log1p(cc)]
    center = ac[:, 1:3].mean(0); scale = np.maximum(ac[:, 1:3].std(0), 1e-6)
    sq = np.c_[aq[:, :3], vq, ((aq[:, 1:3]-center)/scale).min(1)]
    sc = np.c_[ac[:, :3], vc, ((ac[:, 1:3]-center)/scale).min(1)]
    return xq, xc, sq, sc


def observation_weights(y, unknown):
    known, w, u = weights(y, unknown)
    return .5*(w+u)


def weighted_quantile(x, w, p=.95):
    order = np.argsort(x, kind='stable')
    index = np.searchsorted(np.cumsum(w[order]), p*w.sum())
    return float(x[order[min(index, len(x)-1)]])


def source_plan(frame):
    plans = {}
    for role, ports in GROUPS.items():
        rows = np.flatnonzero(frame.port.isin(ports))
        rng = np.random.default_rng(SEED)
        shuffled = rng.permutation(rows)
        pool = shuffled[:int(np.ceil(.35*len(rows)))]; rest = shuffled[len(pool):]
        query = np.concatenate([rng.permutation(rest[frame.class_id.to_numpy()[rest] == cl])[:40]
                                for cl in range(8)])
        plans[role] = dict(pool=np.sort(pool), query=np.sort(query))
    core.write(OUT/'SOURCE_PLAN.json', {r: {k: v.tolist() for k, v in p.items()} for r,p in plans.items()})
    return plans


def source_episodes(frame, values, plans):
    y = frame.class_id.to_numpy(); records = []; fits = 0; start = time.perf_counter()
    assert frame.identity.nunique() == len(frame)
    for a in range(8):
        for b in range(a+1, 8):
            for role, ports in GROUPS.items():
                keep = ~frame.port.isin(ports).to_numpy() & ~np.isin(y, [a,b])
                assert len(np.unique(y[keep])) == 6
                model, ova = fit_geometry(tuple(v[keep] for v in values), y[keep]); fits += 1
                f = core.encode(values, model)
                for outer, pseudo in [(a,b), (b,a)]:
                    query = plans[role]['query']; query = query[y[query] != outer]
                    pool = plans[role]['pool']; pool = pool[~np.isin(y[pool], [outer,pseudo])]
                    assert (y[query] == pseudo).any() and len(pool) >= 64
                    assert not set(frame.identity.iloc[np.flatnonzero(keep)]) & set(frame.identity.iloc[query])
                    for draw in range(3):
                        eligible = [c for c in model['classes'] if (y[query] == c).any() and
                                    (y[pool] == c).any() and (y[pool] != c).sum() >= 64]
                        missing = int(eligible[(outer+pseudo+draw) % len(eligible)])
                        for arm in ('random', 'missing'):
                            rr = np.random.default_rng(SEED + outer*101 + pseudo*13 + draw)
                            p = pool if arm == 'random' else pool[y[pool] != missing]
                            s = rr.choice(p, 64, replace=False)
                            assert not np.intersect1d(s, query).size
                            prob, cal = core.predict(f, s, query, y[s], model['classes'], loo=True)
                            x, xc, score, cs = evidence(f, s, query, y[s], prob, cal, model, ova)
                            rel = Path('source')/f'outer_{outer}'/role/f'v{pseudo}_d{draw}_{arm}.npz'
                            save(OUT/rel, inputs=x, cal_inputs=xc, scores=score, cal_scores=cs,
                                 labels=y[query], pred=model['classes'][prob.argmax(1)], query=query,
                                 supports=s, support_labels=y[s], classes=model['classes'],
                                 unknown=pseudo, outer=outer)
                            records.append(dict(outer=outer, pseudo=pseudo, role=role, draw=draw, arm=arm,
                                                missing=missing if arm == 'missing' else -1,
                                                n=len(query), fit_n=int(keep.sum()), path=rel.as_posix()))
            print('SOURCE_PAIR', a, b, 'fits', fits, 'seconds', round(time.perf_counter()-start,1), flush=True)
    pd.DataFrame(records).to_csv(OUT/'source_contexts.csv', index=False)
    core.write(OUT/'SOURCE_COMPLETE.json', dict(fits=fits, contexts=len(records), seconds=time.perf_counter()-start))
    return pd.DataFrame(records)


def train_gates(records):
    frozen = {}; validation = []; fitrows = []
    for outer in range(8):
        block = records[records.outer == outer]
        xx=[]; yy=[]; ww=[]
        for row in block[block.role == 'train'].itertuples():
            z=np.load(OUT/row.path); y=z['labels']; unknown=int(z['unknown'])
            xx.append(z['inputs']); yy.append(y==unknown); ww.append(observation_weights(y,unknown))
        x=np.concatenate(xx); y=np.concatenate(yy); w=np.concatenate(ww); w *= len(w)/w.sum()
        scaler=StandardScaler().fit(x, sample_weight=w)
        gate=LogisticRegression(C=1., solver='lbfgs', max_iter=500, random_state=0).fit(scaler.transform(x),y,sample_weight=w)
        assert gate.n_iter_.max() < 500
        entries=[]
        for row in block.itertuples():
            z=np.load(OUT/row.path); score=np.c_[z['scores'], gate.predict_proba(scaler.transform(z['inputs']))[:,1]]
            known, wknown, _=weights(z['labels'],int(z['unknown']))
            entries.append((row, z['labels'], z['pred'], score, wknown))
        calentries=[e for e in entries if e[0].role=='calibration']
        thresholds=[]
        for i in range(len(METHODS)):
            thresholds.append(weighted_quantile(np.concatenate([e[3][:,i] for e in calentries]),
                                                np.concatenate([e[4] for e in calentries])))
        values={m:[] for m in METHODS}
        for row, labels, pred, score, wk in entries:
            for i,method in enumerate(METHODS):
                result=readout(score[:,i],thresholds[i],labels,pred,row.pseudo)
                from sklearn.metrics import roc_auc_score
                result['auroc']=100*roc_auc_score(labels==row.pseudo,score[:,i],sample_weight=observation_weights(labels,row.pseudo))
                validation.append(dict(outer=outer,pseudo=row.pseudo,role=row.role,draw=row.draw,arm=row.arm,method=method,**result))
                if row.role=='calibration':values[method].append(result['utility_20'])
        choices={m:float(np.mean(v)) for m,v in values.items()};choices['none']=0.
        selected=max(choices,key=choices.get)
        frozen[outer]=dict(scaler=scaler,gate=gate,thresholds=np.array(thresholds),selected=selected)
        fitrows.append(dict(outer=outer,selected=selected,validation_utility=choices,
                            coefficients=gate.coef_[0].tolist(),intercept=float(gate.intercept_[0])))
    pd.DataFrame(validation).to_csv(OUT/'source_metrics.csv',index=False)
    with (OUT/'frozen_gates.pkl').open('wb') as h:pickle.dump(frozen,h)
    core.write(OUT/'FROZEN_POLICY.json',dict(methods=METHODS,features=['ova','source_distance','support_distance','log_predicted_class_count'],
               policies=fitrows,target_labels_used=False,primary_threshold='source_calibration_macro_known_p95'))
    return frozen


def evaluate_targets(source_frame, source_values, tasks, enrollment, frozen):
    plan=json.loads((BASE/'coverage_intervention/SUPPORT_PLAN.json').read_text(encoding='utf-8'))
    replay=[]; timing=[]; records=[]; start=time.perf_counter()
    for task in TARGETS:
        t=tasks[task]; rows=[r for r in plan if r['task']==task and r['arm'] in ('random','cover1')]
        t['union']=np.unique(np.r_[t['query'],np.array([r['supports'] for r in rows]).ravel()])
        t['values']=core.raw_take(t['arrays'],t['frame'],t['union'])
        t['frame'].iloc[t['union']].assign(local_row=t['union']).to_csv(OUT/(task.replace('/','_')+'_manifest.csv'),index=False)
    sy=source_frame.class_id.to_numpy()
    for outer in range(8):
        with (BASE/'fits'/f'without_{outer}.pkl').open('rb') as h:model=pickle.load(h)
        keep=sy!=outer
        fs=core.encode(tuple(v[keep] for v in source_values),model)
        tt=time.perf_counter(); ova=ova_fit(fs['x'],sy[keep],model['classes'])
        ova_seconds=time.perf_counter()-tt
        policy=frozen[outer]
        for task in TARGETS:
            t=tasks[task]; f=core.encode(t['values'],model); q=np.searchsorted(t['union'],t['query'])
            labels=t['frame'].class_id.to_numpy(); qy=labels[t['query']]
            for arm in ('random','cover1'):
                old=np.load(BASE/'coverage_intervention/tasks'/task/f'unknown_{outer}_{arm}.npz')
                np.testing.assert_array_equal(old['query'],t['query'])
                for draw in range(5):
                    tt=time.perf_counter(); support=old['supports'][draw]; s=np.searchsorted(t['union'],support)
                    assert not np.intersect1d(s,q).size and not (labels[support]==outer).any()
                    prob,cal=core.predict(f,s,q,labels[support],model['classes'],loo=True)
                    np.testing.assert_array_equal(prob.argmax(1),old['probability'][draw].argmax(1))
                    x,xc,score,cs=evidence(f,s,q,labels[support],prob,cal,model,ova)
                    score=np.c_[score,policy['gate'].predict_proba(policy['scaler'].transform(x))[:,1]]
                    cs=np.c_[cs,policy['gate'].predict_proba(policy['scaler'].transform(xc))[:,1]]
                    # Replayed legacy score decisions distinguish numeric drift from actual changes.
                    oldreject=old['score'][draw,:,:3]>old['thresholds'][draw,:3]
                    np.testing.assert_array_equal(score[:,:3]>np.quantile(cs[:,:3],.95,axis=0),oldreject)
                    rel=Path('target')/task/f'u{outer}_d{draw}_{arm}.npz'
                    save(OUT/rel,score=score,cal_score=cs,inputs=x,cal_inputs=xc,probability=prob,
                         pred=model['classes'][prob.argmax(1)],labels=qy,query=t['query'],
                         supports=support,support_labels=labels[support],source_thresholds=policy['thresholds'],
                         support_thresholds=np.quantile(cs,.95,axis=0),selected=policy['selected'],unknown=outer)
                    elapsed=time.perf_counter()-tt
                    records.append(dict(task=task,outer=outer,draw=draw,arm=arm,path=rel.as_posix()))
                    replay.append(dict(task=task,outer=outer,draw=draw,arm=arm,prob_max=float(abs(prob-old['probability'][draw]).max()),
                                       legacy_argmax_exact=True,legacy_rejection_exact=True))
                    timing.append(dict(task=task,outer=outer,draw=draw,arm=arm,seconds=elapsed,ova_fit_seconds=ova_seconds))
        print('TARGET_OUTER',outer,'seconds',round(time.perf_counter()-start,1),flush=True)
    pd.DataFrame(records).to_csv(OUT/'target_contexts.csv',index=False)
    pd.DataFrame(replay).to_csv(OUT/'replay.csv',index=False)
    pd.DataFrame(timing).to_csv(OUT/'timing.csv',index=False)
    core.write(OUT/'PREDICTIONS_COMPLETE.json',dict(target_contexts=len(records),target_fits=8,
              neural_training=False,full_reference_model=False,seconds=time.perf_counter()-start))


def main():
    started=time.perf_counter(); core.OUT=OUT/'input_replay'; core.OUT.mkdir(parents=True,exist_ok=True)
    source,rows,arrays,tasks,enrollment=core.inputs()
    assert enrollment==json.loads((BASE/'ENROLLMENT.json').read_text(encoding='utf-8'))
    np.testing.assert_array_equal(rows,np.load(BASE/'source_rows.npy'))
    frame=source.iloc[rows].reset_index(drop=True)
    values=core.raw_take(arrays,source,rows)
    frame.groupby(['port','class_id']).size().rename('ships').to_csv(OUT/'source_inventory.csv')
    if (OUT/'SOURCE_COMPLETE.json').exists():records=pd.read_csv(OUT/'source_contexts.csv')
    else:records=source_episodes(frame,values,source_plan(frame))
    frozen=train_gates(records)
    evaluate_targets(frame,values,tasks,enrollment,frozen)
    core.write(OUT/'RUN_COMPLETE.json',dict(seconds=time.perf_counter()-started,source_geometry_fits=56,
                                         ova_fits=64,joint_fits=8))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
