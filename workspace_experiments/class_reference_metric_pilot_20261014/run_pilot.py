"""Leave-port-out learning of source-reference metrics; frozen visual reader."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import importlib.util
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.class_weight import compute_sample_weight
import torch
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
CORE = ROOT / 'class_reference_core_20261014'
EXPAND = ROOT / 'scene_applicability_expansion_20261014'
PILOT = ROOT / 'scene_applicability_design_20261013/pilot'
SEEDS = [20261014, 20261015, 20261016]
ANCHORS = ['class_mean', 'cluster_0']
STEPS = 240
DTYPE = torch.float64
TRAIN_SAMPLING = 'random'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


core = module('metric_pilot_core', CORE / 'model.py')
reader = module('metric_pilot_reader', Path('E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/unified_head_core.py'))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def tensor(x):
    return torch.as_tensor(x, dtype=DTYPE)


def load_data():
    d = pd.read_csv(ROOT / 'visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv',
                    usecols=['instance_id', 'port', 'mmsi', 'class_id', 'product'],
                    dtype={'mmsi': str}, keep_default_na=False)
    d['acquisition'] = d['product'].str.split('_').map(lambda s: '_'.join([s[0], s[6], s[7]]))
    d['known'] = d.mmsi.str.fullmatch('[1-9][0-9]{8}')
    conflict = d[d.known].groupby('mmsi').class_id.nunique()
    d['valid'] = d.known & ~d.mmsi.isin(conflict[conflict > 1].index)
    with np.load(EXPAND / 'encoded.npz') as a:
        f = {k: a[k] for k in ['rows', 'r', 't', 'b', 'pb', 'pd']}
    f['z'] = np.c_[f.pop('r'), f.pop('t')]
    with (CORE / 'normalization_repair/ports_banks.pkl').open('rb') as h:
        banks = pickle.load(h)
    plans = json.loads((EXPAND / 'ENROLLMENT.json').read_text(encoding='utf-8'))
    source = np.load(PILOT / 'source_rows.npy')
    with np.load(PILOT / 'encoded.npz') as a:
        ix = np.searchsorted(a['rows'], source)
        np.testing.assert_array_equal(a['rows'][ix], source)
        ships = d.iloc[source].mmsi.to_numpy()
        rng = np.random.default_rng(20261013)
        chosen = np.array([rng.choice(np.flatnonzero(ships == s)) for s in np.unique(ships)])
        z = np.c_[a['r'][ix[chosen]], a['t'][ix[chosen]]]
    cov = {}
    for anchor in ANCHORS:
        r = core.reference_map(z, banks[anchor])
        cov[anchor] = tensor(r.T @ r / len(r))
        f[anchor] = core.reference_map(f['z'], banks[anchor])
    held_all = d[d.port.isin({e['port'] for e in plans})]
    assert not set(d.iloc[source].mmsi) & set(held_all.mmsi)
    assert not set(d.iloc[source].acquisition) & set(held_all.acquisition)
    assert not set(d.iloc[source].port) & set(held_all.port)
    return d, f, plans, banks, cov, source


def indices(f, rows):
    ix = np.searchsorted(f['rows'], rows)
    np.testing.assert_array_equal(f['rows'][ix], rows)
    return ix


def make_task(d, f, e, q, s, draw, cell, held, role):
    q, s = np.array(q, dtype=int), np.array(s, dtype=int)
    qi, si = indices(f, q), indices(f, s)
    y = d.iloc[q].class_id.to_numpy(int)
    sy = d.iloc[s].class_id.to_numpy(int)
    assert len(s) == 64 == d.iloc[s].mmsi.nunique()
    assert not set(d.iloc[q].mmsi) & set(d.iloc[s].mmsi)
    assert d.iloc[s].port.eq(e['port']).all()
    if cell == 'other_acquisition':
        assert d.iloc[s].acquisition.ne(e['acquisition']).all()
    else:
        assert d.iloc[s].acquisition.eq(e['acquisition']).all()
    kb = f['b'][si] @ f['b'][si].T + 1
    ab = reader.coefficients(kb, f['pb'][si], sy, 8.)
    bp = softmax(f['pb'][qi] + (f['b'][qi] @ f['b'][si].T + 1) @ ab, axis=1)
    counts = np.bincount(sy, minlength=8)
    qw = compute_sample_weight('balanced', y) / len(y)
    sw = np.sqrt(compute_sample_weight('balanced', sy))
    task = dict(key=dict(held=held, role=role, port=e['port'], episode=e['episode'],
                        cohort=e['cohort'], draw=draw, cell=cell),
                q=q, s=s, y=y, sy=sy, counts=counts,
                qweight=tensor(qw), root=tensor(sw), base=tensor(bp),
                prior=tensor(f['pd'][qi]), support_prior=f['pd'][si],
                target=tensor(2*np.eye(8)[sy]-1-f['pd'][si]),
                ks=tensor(f['z'][si] @ f['z'][si].T+1),
                kq=tensor(f['z'][qi] @ f['z'][si].T+1),
                eye=torch.eye(64, dtype=DTYPE),
                yt=torch.tensor(y, dtype=torch.long),
                positive=tensor(y[:, None] == sy[None, :]))
    for anchor in ANCHORS:
        task[anchor] = (tensor(f[anchor][qi]), tensor(f[anchor][si]))
    return task


def enrollment(d, f, plans):
    ports = sorted({e['port'] for e in plans})
    train, test, records, scopes = {}, {}, [], []
    available = d.iloc[f['rows']]
    for fi, held in enumerate(ports):
        held_frame = d[d.port.eq(held)]
        excluded_ships = set(held_frame.loc[held_frame.known, 'mmsi'])
        excluded_acq = set(held_frame.acquisition)
        train[held], test[held] = [], []
        training_plans = [e for e in plans if e['port'] != held]
        all_queries = np.unique(sum([e['query'] for e in training_plans], []))
        reserved = set(d.iloc[all_queries].mmsi)
        allowed = available[available.valid & ~available.port.eq(held) &
                            ~available.mmsi.isin(excluded_ships) &
                            ~available.acquisition.isin(excluded_acq)]
        pool = allowed[~allowed.mmsi.isin(reserved)]
        for e in training_plans:
            q = [i for i in e['query'] if d.at[i, 'mmsi'] not in excluded_ships and
                 d.at[i, 'acquisition'] not in excluded_acq]
            p = pool[pool.port.eq(e['port']) & pool.acquisition.ne(e['acquisition'])]
            if len(q) < 5 or d.iloc[q].class_id.nunique() < 2 or p.mmsi.nunique() < 64:
                scopes.append(dict(held=held, episode=e['episode'], status='not_enrolled',
                                   queries=len(q), support_ships=p.mmsi.nunique()))
                continue
            groups = p.groupby('mmsi', sort=True).groups
            ship_ids = np.array(list(groups))
            for draw in range(3):
                rng = np.random.default_rng(20261014 + fi*10000 + e['episode']*10 + draw)
                if TRAIN_SAMPLING == 'matched_scene_counts':
                    selected = []
                    for c, count in enumerate(e['quotas']['64']):
                        candidates = p.loc[p.class_id.eq(c), 'mmsi'].unique()
                        assert len(candidates) >= count
                        selected.extend(rng.choice(candidates, int(count), replace=False))
                else:
                    selected = rng.choice(ship_ids, 64, replace=False)
                s = [int(rng.choice(groups[ship])) for ship in selected]
                if TRAIN_SAMPLING == 'matched_scene_counts':
                    np.testing.assert_array_equal(np.bincount(d.iloc[s].class_id, minlength=8), e['quotas']['64'])
                train[held].append(make_task(d, f, e, q, s, draw, 'other_acquisition', held, 'train'))
            scopes.append(dict(held=held, episode=e['episode'], status='enrolled',
                               queries=len(q), support_ships=p.mmsi.nunique()))
        for e in [e for e in plans if e['port'] == held]:
            for draw, old in enumerate(e['draws']):
                for ci, cell in enumerate(['same_acquisition', 'other_acquisition']):
                    test[held].append(make_task(d, f, e, e['query'], old['64'][ci], draw, cell, held, 'held'))
        rows = np.unique(np.concatenate([np.r_[t['q'], t['s']] for t in train[held]]))
        assert not set(d.iloc[rows].mmsi) & excluded_ships
        assert not set(d.iloc[rows].acquisition) & excluded_acq
        for t in train[held] + test[held]:
            records.append(dict(**t['key'], query=t['q'].tolist(), support=t['s'].tolist(),
                                query_counts=np.bincount(t['y'], minlength=8).tolist(),
                                support_counts=t['counts'].tolist()))
        print('ENROLLED', held, 'train', len(train[held]), 'held', len(test[held]), flush=True)
    write('ENROLLMENT.json', records)
    pd.DataFrame(scopes).to_csv(OUT/'training_capacity.csv', index=False)
    return train, test


def normalized_metric(l, covariance):
    raw = l @ l.T
    return raw * (torch.trace(covariance) / torch.sum(raw * covariance))


def forward(task, anchor, m, original=False):
    ks, kq = task['ks'], task['kq']
    if not original:
        rq, rs = task[anchor]
        ks = .75*ks + .25*(rs @ m @ rs.T)
        kq = .75*kq + .25*(rq @ m @ rs.T)
    root = task['root']
    a = root[:, None]*torch.linalg.solve(root[:, None]*ks*root[None, :] + 8*task['eye'],
                                         root[:, None]*task['target'])
    scores = task['prior'] + kq @ a
    expert = torch.softmax(scores, dim=1)
    return dict(full=.5*(task['base']+expert), expert=expert, scores=scores,
                kernel=kq, ks=ks, coef=a)


def task_loss(task, anchor, m, objective):
    if objective == 'task':
        p = forward(task, anchor, m)['full']
        nll = -p[torch.arange(len(task['y'])), task['yt']].clamp_min(1e-15).log()
        return (task['qweight']*nll).sum()
    rq, rs = task[anchor]
    qnorm = torch.sqrt((rq @ m * rq).sum(1).clamp_min(1e-15))
    snorm = torch.sqrt((rs @ m * rs).sum(1).clamp_min(1e-15))
    logits = (rq @ m @ rs.T) / (qnorm[:, None]*snorm[None, :]) / .1
    npos = task['positive'].sum(1)
    valid = npos > 0
    if not valid.any():
        return logits.sum()*0
    loss = -(torch.log_softmax(logits, dim=1)*task['positive']).sum(1) / npos.clamp_min(1)
    weight = task['qweight']*valid
    return (weight*loss).sum()/weight.sum()


def checks(train, test, covariance):
    records = []
    eye = torch.eye(8, dtype=DTYPE)
    for held, tasks in test.items():
        t = tasks[0]
        for anchor in ANCHORS:
            r = forward(t, anchor, eye)
            checked = KernelRidge(alpha=8., kernel='precomputed').fit(
                r['ks'].numpy(), t['target'].numpy(), sample_weight=t['root'].numpy()**2)
            score = t['prior'].numpy()+checked.predict(r['kernel'].numpy())
            np.testing.assert_allclose(score, r['scores'].numpy(), rtol=1e-10, atol=1e-10)
            records.append(dict(kind='independent_sklearn_krr', held=held, anchor=anchor,
                                error=float(np.max(abs(score-r['scores'].numpy())))))
        for t in tasks:
            key = t['key']
            path = CORE/'normalization_repair/predictions'/f"ports_{key['episode']:03d}_64_{key['draw']}_{key['cell']}.npz"
            with np.load(path) as archive:
                for anchor, ai in [('original', 0), ('class_mean', 2), ('cluster_0', 3)]:
                    result = forward(t, 'class_mean' if anchor == 'original' else anchor, eye,
                                     original=anchor == 'original')['full'].numpy()
                    np.testing.assert_allclose(result, archive['full'][ai], rtol=1e-9, atol=1e-9)
                    records.append(dict(kind='fixed_archive_replay', held=held, anchor=anchor,
                                        episode=key['episode'], draw=key['draw'], cell=key['cell'],
                                        error=float(np.max(abs(result-archive['full'][ai])))))
    t = next(iter(train.values()))[0]
    for objective in ['task', 'contrastive']:
        l = eye.clone().requires_grad_(True)
        # A directional finite difference tests the gradient through the actual solve.
        cov = covariance['class_mean']
        v = torch.arange(64, dtype=DTYPE).reshape(8, 8).sin()/8
        loss = task_loss(t, 'class_mean', normalized_metric(l, cov), objective)
        g = torch.autograd.grad(loss, l)[0]
        eps = 1e-5
        finite = (task_loss(t, 'class_mean', normalized_metric(l.detach()+eps*v, cov), objective)-
                  task_loss(t, 'class_mean', normalized_metric(l.detach()-eps*v, cov), objective))/(2*eps)
        exact = (g*v).sum()
        np.testing.assert_allclose(exact.item(), finite.item(), rtol=2e-4, atol=2e-7)
        records.append(dict(kind='directional_gradient', objective=objective,
                            error=abs(exact.item()-finite.item()), derivative=exact.item()))
    return records


def fit(tasks, held, anchor, objective, seed, covariance):
    l = torch.eye(8, dtype=DTYPE).requires_grad_(True)
    optimizer = torch.optim.Adam([l], lr=.01)
    rng = np.random.default_rng(seed)
    byport = {p: [t for t in tasks if t['key']['port'] == p] for p in sorted({t['key']['port'] for t in tasks})}
    log = []
    started = time.perf_counter()
    for step in range(STEPS+1):
        if step % 40 == 0:
            with torch.no_grad():
                m = normalized_metric(l, covariance)
                values = {p: np.mean([task_loss(t, anchor, m, objective).item() for t in ts]) for p, ts in byport.items()}
                risk = {p: np.mean([task_loss(t, anchor, m, 'task').item() for t in ts]) for p, ts in byport.items()}
                log.append(dict(held=held, anchor=anchor, objective=objective, seed=seed, step=step,
                                objective_loss=float(np.mean(list(values.values()))),
                                full_macro_nll=float(np.mean(list(risk.values()))),
                                metric_change=float(torch.linalg.norm(m-torch.eye(8, dtype=DTYPE))),
                                eigen_min=float(torch.linalg.eigvalsh(m).min()),
                                eigen_max=float(torch.linalg.eigvalsh(m).max())))
        if step == STEPS:
            break
        optimizer.zero_grad()
        m = normalized_metric(l, covariance)
        batch = [ts[int(rng.integers(len(ts)))] for ts in byport.values()]
        loss = torch.stack([task_loss(t, anchor, m, objective) for t in batch]).mean()
        loss = loss + .01*((m-torch.eye(8, dtype=DTYPE))**2).mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    m = normalized_metric(l.detach(), covariance)
    np.testing.assert_allclose(torch.sum(m*covariance).item(), torch.trace(covariance).item(), rtol=1e-12)
    model_key = f"{held.replace(' ', '_')}_{anchor}_{objective}_{seed}"
    np.savez_compressed(OUT/'models'/f'{model_key}.npz', metric=m.numpy(), factor=l.detach().numpy(), covariance=covariance.numpy())
    print('TRAINED', model_key, 'NLL', round(log[0]['full_macro_nll'], 5), '->',
          round(log[-1]['full_macro_nll'], 5), 'seconds', round(time.perf_counter()-started, 1), flush=True)
    return m, log


def metrics(y, p):
    pred = p.argmax(1)
    br = ((p-np.eye(8)[y])**2).sum(1)
    nl = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))
    return dict(oa=100*np.mean(pred == y), ba=100*np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]),
                macro_brier=np.mean([br[y == c].mean() for c in np.unique(y)]),
                macro_nll=np.mean([nl[y == c].mean() for c in np.unique(y)]))


def evaluate_task(t, results, d):
    records, cases = [], []
    for (anchor, objective, seed), r in results.items():
        key = dict(**t['key'], anchor=anchor, objective=objective, seed=seed)
        p = r['full'].numpy()
        k = r['kernel'].numpy()
        order = np.argsort(-k, axis=1, kind='stable')
        hit = t['sy'][order] == t['y'][:, None]
        npos = (t['sy'][None, :] == t['y'][:, None]).sum(1)
        ranks = np.arange(1, 65)
        ap = ((np.cumsum(hit, axis=1)/ranks)*hit*(ranks <= npos[:, None])).sum(1)
        ap = np.divide(ap, npos, out=np.full(len(ap), np.nan), where=npos > 0)
        macro_ap = np.mean([np.nanmean(ap[t['y'] == c]) for c in np.unique(t['y']) if np.isfinite(ap[t['y'] == c]).any()])
        records.append(dict(**key, mode='full', **metrics(t['y'], p), map_at_r=100*macro_ap))
        records.append(dict(**key, mode='expert', **metrics(t['y'], r['expert'].numpy())))
        baseline = results[(anchor, 'fixed', -1)] if anchor != 'original' else r
        p0 = baseline['full'].numpy()
        pred, pred0 = p.argmax(1), p0.argmax(1)
        # Current strongest wrong class, not just the historical competing class.
        wrong = p.copy(); wrong[np.arange(len(p)), t['y']] = -np.inf
        margin = p[np.arange(len(p)), t['y']]-wrong.max(1)
        for c in np.unique(t['y']):
            mask = t['y'] == c
            before = pred0[mask] == c; after = pred[mask] == c
            cases.append(dict(**key, class_id=int(c), query_n=int(mask.sum()), support_n=int(t['counts'][c]),
                              recall=100*after.mean(), baseline_recall=100*before.mean(),
                              rescues=int((~before & after).sum()), harms=int((before & ~after).sum()),
                              margin=float(margin[mask].mean()),
                              macro_nll=float(-np.log(np.maximum(p[mask, c], 1e-15)).mean())))
        if objective != 'fixed':
            oldk = baseline['kernel']; olda = baseline['coef']
            geometry = (r['kernel']-oldk) @ olda
            refit = r['kernel'] @ (r['coef']-olda)
            np.testing.assert_allclose((geometry+refit).numpy(), (r['scores']-baseline['scores']).numpy(), atol=1e-9, rtol=1e-9)
    return records, cases


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    for folder in ['models', 'predictions']:
        (OUT/folder).mkdir(exist_ok=True)
    write('PROTOCOL.json', dict(date='2026-10-14', status='Frozen before metric-training outcomes',
        question='Does learning class-reference geometry through deployed support residual solve improve held-port decisions?',
        anchors=ANCHORS, cluster_control='Preselected first of3 existing seed banks, no target choice',
        objectives=['supervised multi-positive cosine contrastive at T=.1', 'macro NLL after weighted residual ridge and .5 full fusion'],
        seeds=SEEDS, steps=STEPS, optimizer='Adam lr.01, identity start, .01 mean-square metric-to-identity penalty',
        sampler='One random training task from each of two training ports per step, equal port weights',
        metric='M=L L^T; source covariance trace matched to identity at every step',
        frozen='visual2752, source prior, reference banks, rho.25, ridge8, probability fusion.5',
        training='Leave one of3source-held ports out. Remove all its known MMSIs/acquisitions from metric training. Original query cohort filtered; 3new64-unique-ship support draws per scene from other acquisitions, excluding all training query ships.',
        support_sampling=TRAIN_SAMPLING,
        matched_sampling='When enabled, reuse each TRAINING scene original class quota; no held-port counts or outcomes used. One posthoc sampler control, not independent confirmation.',
        evaluation='Original48 scenes and5support draws; both same/other acquisition. Final240-step model only, no held selection.',
        primary='Other-acquisition, equal scenes within port then equal3ports; report training-seed and support-draw variability separately',
        scope='Source-corpus pilot only; no F0921/OSS2, no new backbone/source-head fits, historical exposure retained',
        main_model_changed=False))
    d, f, plans, banks, cov, source = load_data()
    train, test = enrollment(d, f, plans)
    check_records = checks(train, test, cov)
    allmetrics, allcases, logs = [], [], []
    identity = torch.eye(8, dtype=DTYPE)
    for held in sorted(train):
        matrices = {(a, 'fixed', -1): identity for a in ANCHORS}
        for anchor in ANCHORS:
            for objective in ['contrastive', 'task']:
                for seed in SEEDS:
                    m, trace = fit(train[held], held, anchor, objective, seed, cov[anchor])
                    matrices[(anchor, objective, seed)] = m
                    logs.extend(trace)
                    pd.DataFrame(logs).to_csv(OUT/'training_log.csv', index=False)
        for ti, t in enumerate(test[held]):
            with torch.no_grad():
                results = {key: forward(t, key[0], m) for key, m in matrices.items()}
                results[('original', 'fixed', -1)] = forward(t, 'class_mean', identity, original=True)
                records, cases = evaluate_task(t, results, d)
            allmetrics.extend(records); allcases.extend(cases)
            names = ['|'.join(map(str, key)) for key in results]
            np.savez_compressed(OUT/'predictions'/f"{held.replace(' ', '_')}_{ti:03d}.npz",
                arms=np.array(names), full=np.stack([r['full'].numpy() for r in results.values()]),
                expert=np.stack([r['expert'].numpy() for r in results.values()]),
                q=t['q'], s=t['s'], y=t['y'], sy=t['sy'])
        pd.DataFrame(allmetrics).to_csv(OUT/'metrics.csv', index=False)
        pd.DataFrame(allcases).to_csv(OUT/'class_events.csv', index=False)
        print('EVALUATED', held, len(test[held]), flush=True)
    pd.DataFrame(check_records).to_csv(OUT/'checks.csv', index=False)
    write('COMPLETE.json', dict(seconds=time.perf_counter()-started, metric_fits=36,
        train_cells={k:len(v) for k,v in train.items()}, held_cells=sum(map(len, test.values())),
        source_fit_rows=len(source), backbone_fits=0, source_head_fits=0, final_checkpoint_step=STEPS))
    print('COMPLETE', round(time.perf_counter()-started, 1), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
