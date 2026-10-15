"""Matched integration, anchor replacement and score-path diagnostics."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from legacy_model import ReferenceSupportModel, reference_map, cluster_bank

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PREV = ROOT/'class_reference_external_20261014'
EXPAND = ROOT/'scene_applicability_expansion_20261014'
PILOT = ROOT/'scene_applicability_design_20261013/pilot'
spec = importlib.util.spec_from_file_location('old_class_reference', PREV/'experiment.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
ARMS = ['original', 'shrink', 'class_mean', 'cluster_0', 'cluster_1', 'cluster_2']
SEEDS = [20261014, 20261015, 20261016]


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load_npz(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def select(f, rows):
    ix = np.searchsorted(f['rows'], rows)
    np.testing.assert_array_equal(f['rows'][ix], rows)
    return {k: f[k][ix] for k in ['z', 'b', 'pb', 'pd']}


def banks(source_z, labels, template, name):
    started = time.perf_counter()
    out = {'original': None, 'shrink': None, 'class_mean': template}
    composition = []
    for i, seed in enumerate(SEEDS):
        bank, assignments = cluster_bank(source_z, template, seed)
        out[f'cluster_{i}'] = bank
        for k in range(8):
            for c in range(8):
                composition.append(dict(domain=name, seed=seed, cluster=k, class_id=c,
                                        n=int(np.sum((assignments == k) & (labels == c)))))
    np.testing.assert_allclose(template['centers'],
                              np.stack([source_z[labels == c].mean(0) for c in range(8)]),
                              rtol=1e-10, atol=1e-10)
    with (OUT/f'{name}_banks.pkl').open('wb') as h:
        pickle.dump(out, h)
    pd.DataFrame(composition).to_csv(OUT/f'{name}_cluster_composition.csv', index=False)
    print('BANKS', name, len(source_z), 'seconds', time.perf_counter()-started, flush=True)
    return out, dict(domain=name, n=len(source_z), dim=source_z.shape[1],
                     seconds=time.perf_counter()-started, class_counts=np.bincount(labels, minlength=8).tolist())


def prepare_external(ev):
    with (PREV/'deployment_memory.pkl').open('rb') as h:
        template = pickle.load(h)
    with (old.OLD/'fits/geometry.pkl').open('rb') as h:
        geometry = pickle.load(h)
    source = template['source_rows']
    sf = old.encoded(ev, old.source_arrays(), source, geometry)
    d = old.source_metadata()
    bank, info = banks(sf['z'], d.iloc[source].class_id.to_numpy(int), template, 'external')
    plans = json.loads((PREV/'EXTERNAL_ENROLLMENT.json').read_text(encoding='utf-8'))
    contexts = []
    for name in old.ext.DATASETS:
        f = load_npz(old.OLD/f'{name}_encoded.npz')
        f['z'] = np.c_[f.pop('r'), f.pop('t')]
        meta = pd.read_csv(old.OLD/f'{name}_metadata.csv', dtype={'identity': str}, keep_default_na=False)
        tasks = []
        for e in plans:
            if e['target'] != name:
                continue
            archive = load_npz(PREV/'predictions'/f"episode_{e['episode']:03d}.npz")
            for draw, supports in enumerate(e['support_rows']):
                for ci, support in enumerate(supports):
                    tasks.append(dict(dataset=name, episode=e['episode'], port=name,
                        cohort='external', budget=e['budget'], draw=draw,
                        cell=e['arms'][ci], study=e['study'], query=e['query'], support=support,
                        old_original=.5*(archive['base'][draw,ci]+archive['expert'][draw,ci,0]),
                        old_class=.5*(archive['base'][draw,ci]+archive['expert'][draw,ci,1])))
        contexts.append((name, f, meta, bank, tasks))
    return contexts, info


def prepare_ports():
    f = load_npz(EXPAND/'encoded.npz'); f['z'] = np.c_[f.pop('r'), f.pop('t')]
    source_f = load_npz(PILOT/'encoded.npz')
    source = np.load(PILOT/'source_rows.npy')
    d = old.source_metadata()
    si = np.searchsorted(source_f['rows'], source)
    np.testing.assert_array_equal(source_f['rows'][si], source)
    ships = d.iloc[source].identity.to_numpy()
    rng = np.random.default_rng(20261013)
    chosen = np.array([rng.choice(np.flatnonzero(ships == ship)) for ship in np.unique(ships)])
    z = np.c_[source_f['r'][si[chosen]], source_f['t'][si[chosen]]]
    labels = d.iloc[source[chosen]].class_id.to_numpy(int)
    with (PILOT/'memories.pkl').open('rb') as h:
        template = pickle.load(h)['class_mean']
    calibration = json.loads((PILOT/'relative_repair/PROTOCOL.json').read_text(encoding='utf-8'))['calibration']['class_mean']
    template = dict(centers=template['centers'], tau2=template['tau2'],
                    center=np.array(calibration['center']), scale=calibration['scale'])
    bank, info = banks(z, labels, template, 'ports')
    plans = json.loads((EXPAND/'ENROLLMENT.json').read_text(encoding='utf-8'))
    tasks = []
    for e in plans:
        archive = load_npz(EXPAND/'predictions'/f"episode_{e['episode']:02d}_b64.npz")
        for draw, draws in enumerate(e['draws']):
            for ci, support in enumerate(draws['64']):
                tasks.append(dict(dataset='ports', episode=e['episode'], port=e['port'],
                    cohort=e['cohort'], budget=64, draw=draw,
                    cell=['same_acquisition','other_acquisition'][ci], study='acquisition_transfer',
                    query=e['query'], support=support,
                    old_original=archive['probabilities'][draw,ci,0,1],
                    old_class=archive['probabilities'][draw,ci,1,1]))
    return ('ports', f, d, bank, tasks), info


def ap_at_r(k, y, sy):
    order = np.argsort(-k, axis=1, kind='stable')
    hit = sy[order] == y[:, None]
    r = np.sum(sy[None, :] == y[:, None], axis=1)
    rank = np.arange(1, len(sy)+1)
    value = np.sum((np.cumsum(hit, axis=1)/rank)*hit*(rank <= r[:,None]), axis=1)
    return np.divide(value, r, out=np.full(len(y), np.nan), where=r>0)


def margin(x, y, rival):
    row = np.arange(len(y))
    return x[row,y]-x[row,rival]


def main():
    started = time.perf_counter(); ev = old.operator()
    contexts, info = prepare_external(ev)
    port_context, pinfo = prepare_ports(); contexts.append(port_context)
    write('BANK_FITS.json', [info, pinfo])
    evaluate(contexts, ev, started)


def evaluate(contexts, ev, started, model_type=ReferenceSupportModel,
             replay_arms=(('original','old_original'),('class_mean','old_class'))):
    (OUT/'predictions').mkdir(exist_ok=True)
    metrics = []; cases = []; checks = []; timings = []; conditions = []; contributors = []
    for name, f, meta, bank, tasks in contexts:
        print('CONTEXT', name, 'cells', len(tasks), flush=True)
        for ti, task in enumerate(tasks):
            q = np.array(task['query']); s = np.array(task['support'])
            query, support = select(f, q), select(f, s)
            y = meta.iloc[q].class_id.to_numpy(int); sy = meta.iloc[s].class_id.to_numpy(int)
            assert not set(meta.iloc[q].identity)&set(meta.iloc[s].identity)
            assert len(s) == meta.iloc[s].identity.nunique()
            key = {k:task[k] for k in ['dataset','episode','port','cohort','budget','draw','cell','study']}
            counts = np.bincount(sy, minlength=8)
            conditions.append(dict(**key, queries=len(q), query_ships=meta.iloc[q].identity.nunique(),
                                   support_counts=json.dumps(counts.tolist()),
                                   query_counts=json.dumps(np.bincount(y,minlength=8).tolist())))
            results = {}; models = {}
            for arm in ARMS:
                model = model_type(ev.coefficients, bank[arm], rho=0. if arm=='original' else .25)
                t = time.perf_counter(); model.fit_support(support, sy); fit_s = time.perf_counter()-t
                t = time.perf_counter(); result = model.predict(query); pred_s = time.perf_counter()-t
                results[arm] = result; models[arm] = model
                timings.append(dict(**key, arm=arm, fit_seconds=fit_s, predict_seconds=pred_s, query_n=len(q)))
                for mode in ['full','expert']:
                    metrics.append(dict(**key, arm=arm, mode=mode, **old.scores(y, result[mode])))
                for c in np.unique(y):
                    metrics.append(dict(**key, arm=arm, mode=f'class_{c}', **old.scores(y[y==c],result['full'][y==c])))
                if ti == 0:
                    ks = model.kernel(support,support)
                    check = KernelRidge(alpha=8.,kernel='precomputed').fit(ks,
                        2*np.eye(8)[sy]-1-support['pd'], sample_weight=compute_sample_weight('balanced',sy))
                    checked = softmax(query['pd']+check.predict(result['kernel']),axis=1)
                    np.testing.assert_allclose(checked,result['expert'],rtol=1e-10,atol=1e-10)
                    one = model.predict({k:v[:1] for k,v in query.items()})['full']
                    np.testing.assert_allclose(one,result['full'][:1],rtol=1e-10,atol=1e-10)
                    checks.append(dict(dataset=name,arm=arm,kind='independent_krr',error=float(np.max(abs(checked-result['expert'])))))
            for arm, field in replay_arms:
                error = np.max(abs(results[arm]['full']-task[field]))
                np.testing.assert_allclose(results[arm]['full'],task[field],rtol=1e-9,atol=1e-9)
                np.testing.assert_array_equal(results[arm]['full'].argmax(1),task[field].argmax(1))
                checks.append(dict(**key,arm=arm,kind='archive_replay',error=float(error)))
            base = results['original']; pred0=base['full'].argmax(1)
            rival_scores=base['scores'].copy();rival_scores[np.arange(len(y)),y]=-np.inf
            rival=rival_scores.argmax(1); ap0=ap_at_r(base['kernel'],y,sy)
            for arm in ARMS[1:]:
                result=results[arm]; pred=result['full'].argmax(1)
                # Exact algebraic path, not a physical causal decomposition.
                geom=(result['kernel']-base['kernel'])@models['original'].coef
                refit=result['kernel']@(models[arm].coef-models['original'].coef)
                delta=result['scores']-base['scores']
                np.testing.assert_allclose(geom+refit,delta,rtol=1e-9,atol=1e-9)
                ap=ap_at_r(result['kernel'],y,sy)
                dm=margin(delta,y,rival); gm=margin(geom,y,rival); fm=margin(refit,y,rival)
                m0=margin(base['scores'],y,rival)
                for i,row in enumerate(q):
                    cases.append(dict(**key,arm=arm,row=int(row),ship=str(meta.iloc[row].identity),
                        class_id=int(y[i]),rival=int(rival[i]),correct_support=int(counts[y[i]]),
                        old_pred=int(pred0[i]),new_pred=int(pred[i]),old_correct=bool(pred0[i]==y[i]),
                        new_correct=bool(pred[i]==y[i]),old_margin=float(m0[i]),margin_delta=float(dm[i]),
                        relation_margin=float(gm[i]),refit_margin=float(fm[i]),
                        map_old=float(ap0[i]),map_new=float(ap[i])))
                if arm=='class_mean':
                    changed=np.flatnonzero((pred0==y)!=(pred==y))
                    for i in changed:
                        c=int(y[i]);r=int(rival[i])
                        old_contrib=base['kernel'][i]*(models['original'].coef[:,c]-models['original'].coef[:,r])
                        new_contrib=result['kernel'][i]*(models[arm].coef[:,c]-models[arm].coef[:,r])
                        diff=new_contrib-old_contrib
                        for j in np.argsort(-abs(diff))[:3]:
                            contributors.append(dict(**key,query_row=int(q[i]),query_ship=str(meta.iloc[q[i]].identity),
                                class_id=c,rival=r,rescue=bool(pred[i]==c),support_row=int(s[j]),
                                support_ship=str(meta.iloc[s[j]].identity),support_class=int(sy[j]),
                                old_contribution=float(old_contrib[j]),new_contribution=float(new_contrib[j]),
                                delta=float(diff[j])))
            if ti % 40 == 0:
                print('PREDICTED',name,ti,'/',len(tasks),flush=True)
            tag=f"{name}_{task['episode']:03d}_{task['budget']}_{task['draw']}_{task['cell']}"
            np.savez_compressed(OUT/'predictions'/f'{tag}.npz',
                full=np.stack([results[a]['full'] for a in ARMS]),
                expert=np.stack([results[a]['expert'] for a in ARMS]),
                scores=np.stack([results[a]['scores'] for a in ARMS]),
                query_rows=q,support_rows=s,arms=np.array(ARMS))
        print('CONTEXT_DONE',name,flush=True)
    for filename, rows in [('metrics',metrics),('case_paths',cases),('checks',checks),
                           ('timings',timings),('conditions',conditions),('top_support_contributors',contributors)]:
        pd.DataFrame(rows).to_csv(OUT/f'{filename}.csv',index=False)
    write('COMPUTATION_COMPLETE.json',dict(seconds=time.perf_counter()-started,
        support_cells=len(conditions),expert_solves=len(conditions)*len(ARMS),
        arms=ARMS,cluster_seeds=SEEDS,neural_training=0,source_head_fits=0,
        expert_dimension=int(contexts[0][1]['z'].shape[1]),
        integration='Complete encoded-feature candidate; frozen image encoders and source geometry reused; NOT legacy full replacement',
        archive_replays=sum(c['kind']=='archive_replay' for c in checks),
        independent_krr=sum(c['kind']=='independent_krr' for c in checks),
        max_replay_error=max((c['error'] for c in checks if c['kind']=='archive_replay'),default=None)))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
