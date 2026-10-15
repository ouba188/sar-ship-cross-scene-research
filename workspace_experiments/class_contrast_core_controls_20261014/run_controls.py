"""Matched semantic-anchor and rank-seven metric controls, one frozen geometry."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
PREV = OUT.parent / 'class_contrast_utility_20261014'
sys.path.insert(0, str(PREV))
import train_and_test as utility
probe, run = utility.probe, utility.run
from model import unit_response, reference_map
SEEDS = utility.SEEDS
LOWER = np.tril_indices(7)
CHECKS = []


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def project(z, spec):
    if spec['kind'] == 'pca':
        return ((z - spec['mean']) @ spec['vectors']) / spec['std'] * spec['scale']
    if spec['kind'] == 'reference':
        return reference_map(z, spec['bank'])
    value = probe.mapping(z, spec['bank'])
    if spec['kind'] == 'class_subspace':
        value = value @ spec['vectors']
    return value


def fit_specs(z, labels, bank):
    specs = {'class_edge': dict(kind='edge', bank=bank)}
    x = project(z, specs['class_edge'])
    cov = x.T @ x / len(x)
    eig, vectors = np.linalg.eigh(cov)
    vectors = vectors[:, np.argsort(eig)[-7:][::-1]]
    specs['class_metric'] = dict(kind='class_subspace', bank=bank, vectors=vectors)
    compact = x @ vectors
    np.testing.assert_allclose(x[:64] @ x[:64].T, compact[:64] @ compact[:64].T, atol=1e-9)
    CHECKS.append(dict(kind='class_rank7_initial_kernel', error=float(np.max(abs(x[:64] @ x[:64].T - compact[:64] @ compact[:64].T)))))
    pca = PCA(n_components=7, svd_solver='randomized', random_state=20261014).fit(z)
    projected = pca.transform(z)
    energy = float(np.mean(np.sum(z * z, axis=1) + 1))
    specs['pca_metric'] = dict(kind='pca', mean=pca.mean_, vectors=pca.components_.T,
                               std=projected.std(0), scale=np.sqrt(energy / 7))
    composition = []
    for i, seed in enumerate(SEEDS):
        km = KMeans(n_clusters=8, n_init=5, random_state=seed).fit(z)
        cluster = utility.fit_linear_bank(z, dict(bank, centers=km.cluster_centers_))
        specs[f'cluster_edge_{i}'] = dict(kind='edge', bank=cluster)
        for k in range(8):
            for c in range(8):
                composition.append(dict(seed=seed, cluster=k, class_id=c,
                                        n=int(np.sum((km.labels_ == k) & (labels == c)))))
        print('CLUSTER', seed, 'fit', flush=True)
    r = unit_response(z, bank['centers'], bank['tau2'])
    mean = r.mean(0)
    ref = dict(bank, center=mean, scale=np.sqrt(energy / np.mean(np.sum((r - mean) ** 2, axis=1))))
    specs['class_mean'] = dict(kind='reference', bank=ref)
    pd.DataFrame(composition).to_csv(OUT / 'cluster_composition.csv', index=False)
    for spec in specs.values():
        values = project(z, spec)
        spec['cov'] = values.T @ values / len(values)
        np.testing.assert_allclose(np.trace(spec['cov']), energy, atol=1e-9)
    with (OUT / 'specs.pkl').open('wb') as h:
        pickle.dump(specs, h)
    return specs


def make_metric(theta, kind, covariance):
    if kind == 'edge':
        return torch.diag(28 * torch.softmax(theta, dim=0))
    matrix = torch.zeros((7, 7), dtype=torch.float64)
    for i, (a, b) in enumerate(zip(*LOWER)):
        if a == b:
            matrix[a, b] = theta[i]
        else:
            matrix[a, b] = theta[i] / np.sqrt(2)
            matrix[b, a] = theta[i] / np.sqrt(2)
    matrix = matrix - torch.trace(matrix) / 7 * torch.eye(7, dtype=torch.float64)
    metric = torch.matrix_exp(matrix)
    return metric * torch.trace(covariance) / torch.sum(metric * covariance.T)


def penalty(theta, kind):
    if kind == 'edge':
        return .001 * torch.mean((theta - theta.mean()) ** 2)
    diagonal = torch.tensor(LOWER[0] == LOWER[1])
    centered = theta - diagonal * theta[diagonal].mean()
    return .001 * torch.mean(centered ** 2)


def forward(metric, task):
    ss = task['kss'] + .25 * ((task['vs'] @ metric) @ task['vs'].T)
    qs = task['kqs'] + .25 * ((task['vq'] @ metric) @ task['vs'].T)
    root = task['root']
    alpha = root[:, None] * torch.linalg.solve(root[:, None] * ss * root[None, :] + 8 * task['eye'],
                                              root[:, None] * task['target'])
    full = .5 * (task['base'] + torch.softmax(task['prior'] + qs @ alpha, dim=1))
    risk = torch.sum(task['qw'] * torch.sum((full - task['truth']) ** 2, dim=1))
    return risk, full


def prepare_training():
    ev = run.old.operator()
    with (run.PREV / 'validation_fit/geometry.pkl').open('rb') as h:
        geometry = pickle.load(h)
    with (run.PREV / 'validation_memory.pkl').open('rb') as h:
        template = pickle.load(h)
    with (PREV / 'balanced_objective/training_bank.pkl').open('rb') as h:
        bank = pickle.load(h)
    rows = np.sort(np.asarray(template['source_rows']))
    f = run.old.encoded(ev, run.old.source_arrays(), rows, geometry)
    meta = run.old.source_metadata()
    tasks = json.loads((PREV / 'balanced_objective/TRAIN_ENROLLMENT.json').read_text(encoding='utf-8'))
    specs = fit_specs(f['z'], meta.loc[rows].class_id.to_numpy(int), bank)
    prepared = [utility.prepare_task(f, meta, bank, t, ev) for t in tasks]
    presence = pd.DataFrame([dict(port=p['port'], c=int(c)) for p in prepared for c in np.unique(p['y'])])
    counts = presence.groupby(['port', 'c']).size()
    ports = presence.groupby('c').port.nunique()
    masses = np.zeros(8)
    for p in prepared:
        y = p['y']; nc = np.bincount(y, minlength=8)
        weight = np.array([len(tasks) / (8 * ports[c] * counts[(p['port'], c)] * nc[c]) for c in y])
        p['qw'] = utility.tensor(weight)
        for c in np.unique(y):
            masses[c] += weight[y == c].sum() / len(tasks)
    np.testing.assert_allclose(masses, np.repeat(.125, 8), atol=1e-12)
    write('ENROLLMENT.json', tasks)
    write('OBJECTIVE_CLASS_MASS.json', masses.tolist())
    return ev, geometry, f, meta, tasks, specs, prepared


def train_controls(f, tasks, specs, prepared):
    frozen = np.load(PREV / 'balanced_objective/frozen_weights.npy')
    models = {
        'original': dict(spec=None, metric=None),
        'class_mean': dict(spec='class_mean', metric=np.eye(8)),
        'class_fixed': dict(spec='class_edge', metric=np.eye(28)),
        'pca_fixed': dict(spec='pca_metric', metric=np.eye(7)),
    }
    for i in range(3):
        models[f'class_utility_{i}'] = dict(spec='class_edge', metric=np.diag(frozen[i]))
        models[f'cluster_fixed_{i}'] = dict(spec=f'cluster_edge_{i}', metric=np.eye(28))
    logs = []
    for family in ('class_metric', 'pca_metric', 'cluster_edge'):
        for i, seed in enumerate(SEEDS):
            key = f'cluster_edge_{i}' if family == 'cluster_edge' else family
            spec = specs[key]
            kind = 'edge' if family == 'cluster_edge' else 'free'
            x = project(f['z'], spec)
            ps = []
            for task, common in zip(tasks, prepared):
                qs = np.searchsorted(f['rows'], task['query'])
                ss = np.searchsorted(f['rows'], task['support'])
                ps.append(dict(common, vs=utility.tensor(x[ss]), vq=utility.tensor(x[qs])))
            covariance = utility.tensor(spec['cov'])
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            theta = torch.nn.Parameter(torch.randn(28, dtype=torch.float64) * .001)
            risk, _ = forward(make_metric(theta, kind, covariance), ps[0])
            gradient = torch.autograd.grad(risk, theta)[0]
            direction = torch.linspace(-1, 1, 28, dtype=torch.float64); direction /= torch.linalg.norm(direction)
            eps = 1e-5
            finite = (forward(make_metric(theta.detach() + eps * direction, kind, covariance), ps[0])[0] -
                      forward(make_metric(theta.detach() - eps * direction, kind, covariance), ps[0])[0]).item() / (2 * eps)
            automatic = float(gradient @ direction)
            np.testing.assert_allclose(finite, automatic, atol=1e-8, rtol=1e-4)
            CHECKS.append(dict(kind='gradient', family=family, seed=seed, finite=finite, automatic=automatic))
            optimizer = torch.optim.Adam([theta], lr=.03)
            for epoch in range(41):
                if epoch:
                    indexes = rng.permutation(len(tasks))
                    for start in range(0, len(tasks), 8):
                        optimizer.zero_grad()
                        metric = make_metric(theta, kind, covariance)
                        loss = torch.stack([forward(metric, ps[j])[0] for j in indexes[start:start + 8]]).mean()
                        loss = loss + penalty(theta, kind)
                        loss.backward(); optimizer.step()
                if epoch in (0, 10, 20, 30, 40):
                    with torch.no_grad():
                        metric = make_metric(theta, kind, covariance)
                        risk = np.mean([forward(metric, p)[0].item() for p in ps])
                    logs.append(dict(family=family, seed=seed, epoch=epoch, risk=risk))
                    print('TRAIN', family, seed, epoch, risk, flush=True)
            metric = make_metric(theta.detach(), kind, covariance).numpy()
            np.testing.assert_allclose(np.sum(metric * spec['cov'].T), np.trace(spec['cov']), atol=1e-9)
            assert np.linalg.eigvalsh(metric).min() > 0
            models[f'{family}_{i}'] = dict(spec=key, metric=metric)
            np.save(OUT / f'{family}_{seed}_metric.npy', metric)
    pd.DataFrame(logs).to_csv(OUT / 'training_curve.csv', index=False)
    with (OUT / 'models.pkl').open('wb') as h:
        pickle.dump(models, h)
    write('TRAIN_COMPLETE.json', dict(new_small_models=9, seeds=SEEDS, train_tasks=len(tasks),
        epochs=40, scalar_parameters=28, effective_scale_free_parameters=27,
        class_utility='Reused all3 frozen balanced-objective models, identical training data/protocol',
        frozen_before_test=True))
    return models


def enrollment():
    plans = json.loads((run.PREV / 'EXTERNAL_ENROLLMENT.json').read_text(encoding='utf-8'))
    tasks = []
    for e in plans:
        if e['study'] != 'acquisition_transfer':
            continue
        for draw, supports in enumerate(e['support_rows']):
            for ci, support in enumerate(supports):
                tasks.append(dict(dataset=e['target'], episode=e['episode'], port=e['target'], cohort='external',
                                  budget=e['budget'], draw=draw, cell=e['arms'][ci], study=e['study'],
                                  query=e['query'], support=support))
    plans = json.loads((run.EXPAND / 'ENROLLMENT.json').read_text(encoding='utf-8'))
    for e in plans:
        for draw, d in enumerate(e['draws']):
            tasks.append(dict(dataset='ports', episode=e['episode'], port=e['port'], cohort=e['cohort'],
                              budget=64, draw=draw, cell='other_acquisition', study='acquisition_transfer',
                              query=e['query'], support=d['64'][1]))
    return tasks


def evaluate(ev, geometry, specs, models, source_meta):
    tasks = enrollment()
    write('EVALUATION_ENROLLMENT.json', tasks)
    (OUT / 'predictions').mkdir(exist_ok=True)
    metrics, changes, conditions, timings = [], [], [], []
    arms = list(models)
    training_ids = np.load(run.PREV / 'validation_fit_rows.npy')
    for dataset in ('f0921', 'oss2', 'ports'):
        active = [t for t in tasks if t['dataset'] == dataset]
        rows = np.unique(sum([t['query'] + t['support'] for t in active], []))
        if dataset == 'ports':
            meta = source_meta
            arrays = run.old.source_arrays()
            for field in ('port', 'identity', 'acquisition'):
                assert not set(meta.loc[training_ids, field]) & set(meta.loc[rows, field])
        else:
            meta = pd.read_csv(run.old.OLD / f'{dataset}_metadata.csv', dtype={'identity': str}, keep_default_na=False)
            arrays = run.old.ext.arrays_for(dataset, meta)
            assert not set(source_meta.loc[training_ids].identity) & set(meta.loc[rows].identity)
        f = run.old.encoded(ev, arrays, rows, geometry)
        maps = {key: project(f['z'], spec) for key, spec in specs.items()}
        print('EVALUATE', dataset, len(active), flush=True)
        for ti, task in enumerate(active):
            q, s = np.array(task['query']), np.array(task['support'])
            qi, si = np.searchsorted(rows, q), np.searchsorted(rows, s)
            y, sy = meta.loc[q].class_id.to_numpy(int), meta.loc[s].class_id.to_numpy(int)
            assert not set(meta.loc[q].identity) & set(meta.loc[s].identity)
            assert len(s) == meta.loc[s].identity.nunique()
            query, support = run.select(f, q), run.select(f, s)
            kss = support['z'] @ support['z'].T + 1
            kqs = query['z'] @ support['z'].T + 1
            bss = support['b'] @ support['b'].T + 1
            bqs = query['b'] @ support['b'].T + 1
            base = softmax(query['pb'] + bqs @ ev.coefficients(bss, support['pb'], sy, 8.), axis=1)
            key = {k: task[k] for k in ('dataset', 'episode', 'port', 'cohort', 'budget', 'draw', 'cell', 'study')}
            conditions.append(dict(**key, queries=len(q), query_ships=meta.loc[q].identity.nunique(),
                                   support_counts=json.dumps(np.bincount(sy, minlength=8).tolist()),
                                   query_counts=json.dumps(np.bincount(y, minlength=8).tolist())))
            outputs = {}
            for arm, model in models.items():
                start = time.perf_counter()
                if model['spec'] is None:
                    ss, qs = kss, kqs
                else:
                    mapped = maps[model['spec']]
                    ss = .75 * kss + .25 * (mapped[si] @ model['metric'] @ mapped[si].T)
                    qs = .75 * kqs + .25 * (mapped[qi] @ model['metric'] @ mapped[si].T)
                coef = ev.coefficients(ss, support['pd'], sy, 8.)
                expert = softmax(query['pd'] + qs @ coef, axis=1)
                full = .5 * (base + expert)
                outputs[arm] = full
                timings.append(dict(**key, arm=arm, seconds=time.perf_counter() - start))
                for mode, p in [('full', full), ('expert', expert)]:
                    metrics.append(dict(**key, arm=arm, mode=mode, **run.old.scores(y, p)))
                for c in np.unique(y):
                    metrics.append(dict(**key, arm=arm, mode=f'class_{c}', **run.old.scores(y[y == c], full[y == c])))
                if ti == 0:
                    independent = KernelRidge(alpha=8., kernel='precomputed').fit(ss,
                        2 * np.eye(8)[sy] - 1 - support['pd'], sample_weight=compute_sample_weight('balanced', sy))
                    checked = .5 * (base + softmax(query['pd'] + independent.predict(qs), axis=1))
                    np.testing.assert_allclose(checked, full, atol=1e-10)
                    CHECKS.append(dict(kind='sklearn', dataset=dataset, arm=arm, error=float(abs(checked - full).max())))
            for i in range(3):
                reference = outputs[f'class_utility_{i}'].argmax(1)
                for family in ('pca_metric', 'class_metric', 'cluster_edge'):
                    pred = outputs[f'{family}_{i}'].argmax(1)
                    for j in np.flatnonzero((reference == y) != (pred == y)):
                        changes.append(dict(**key, seed=SEEDS[i], competitor=family, row=int(q[j]),
                                            ship=str(meta.loc[q[j]].identity), class_id=int(y[j]),
                                            ours_correct=bool(reference[j] == y[j]), competitor_correct=bool(pred[j] == y[j])))
            if dataset == 'ports':
                before = pd.read_csv(PREV / 'balanced_objective/same_geometry_held_metrics.csv') if ti == 0 else before
                selected = before[(before.episode == task['episode']) & (before.draw == task['draw'])]
                for i in range(3):
                    score = run.old.scores(y, outputs[f'class_utility_{i}'])
                    archived = selected[selected.arm == f'utility_{i}'].iloc[0]
                    np.testing.assert_allclose([score[v] for v in probe.analysis.VALUES], archived[probe.analysis.VALUES].to_numpy(float), atol=1e-9)
                CHECKS.append(dict(kind='same_geometry_metric_replay', episode=task['episode'], draw=task['draw']))
            tag = f"{dataset}_{task['episode']:03d}_{task['budget']}_{task['draw']}_{task['cell']}"
            np.savez_compressed(OUT / 'predictions' / f'{tag}.npz', full=np.stack([outputs[a] for a in arms]),
                                arms=np.array(arms), query_rows=q, support_rows=s)
        print('DONE', dataset, flush=True)
    for name, records in [('metrics', metrics), ('change_events', changes), ('conditions', conditions), ('timings', timings)]:
        pd.DataFrame(records).to_csv(OUT / f'{name}.csv', index=False)
    write('CHECKS.json', CHECKS)
    write('COMPUTATION_COMPLETE.json', dict(support_cells=len(tasks), arms=arms, solves=len(tasks) * len(arms),
        new_backbone_training=0, new_source_heads=0, source_geometry='One frozen validation geometry for ALL source and target mappings; no source refit or cluster-index alignment at deployment'))


def summarize():
    analysis = probe.analysis
    data = pd.read_csv(OUT / 'metrics.csv')
    draw = analysis.aggregate(analysis.add_scopes(data), analysis.KEY + ['arm'], analysis.VALUES)
    draw.to_csv(OUT / 'draw_metrics.csv', index=False)
    result = []
    for keys, g in draw.groupby(analysis.KEY):
        for metric in analysis.VALUES:
            pivot = g.pivot(index='draw', columns='arm', values=metric)
            families = {'ours': 'class_utility', 'pca': 'pca_metric', 'free_class': 'class_metric', 'cluster': 'cluster_edge'}
            for family, prefix in families.items():
                for i in range(3):
                    values = pivot[f'{prefix}_{i}']
                    baselines = {'core': pivot.class_mean, 'ours': pivot[f'class_utility_{i}'],
                                 'own_fixed': pivot[f'cluster_fixed_{i}'] if family == 'cluster' else pivot.pca_fixed if family == 'pca' else pivot.class_fixed}
                    for baseline, base in baselines.items():
                        delta = values - base
                        result.append(dict(zip(analysis.KEY, keys), family=family, seed=SEEDS[i], metric=metric,
                                           baseline=baseline, absolute=values.mean(), delta=delta.mean(),
                                           positive=int((delta > 1e-10).sum()), negative=int((delta < -1e-10).sum())))
    result = pd.DataFrame(result)
    result.to_csv(OUT / 'seed_effects.csv', index=False)
    summary = result.groupby(analysis.KEY + ['family', 'metric', 'baseline']).agg(
        absolute=('absolute', 'mean'), delta=('delta', 'mean'), sd_seed=('delta', 'std'),
        positive_seeds=('delta', lambda x: int((x > 1e-10).sum()))).reset_index()
    summary.to_csv(OUT / 'effects_summary.csv', index=False)
    print(summary[(summary['mode'] == 'full') & (summary.metric == 'ba') & (summary.baseline == 'ours')].to_string(index=False), flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    write('PROTOCOL.json', dict(status='Frozen before control outcomes', purpose='Semantic class anchors vs density anchors; constrained graph metric vs general rank7 PSD',
        geometry='Existing validation geometry frozen for training AND ALL evaluation data, avoiding cluster permutation/PCA frame refit confounds',
        comparability='Same440 support/query cells; absolute scores and deltas are a NEW common-geometry table, not copied from previous deployment-geometry results',
        controls=dict(pca_metric='Source-only unlabelled PCA7, standardized coordinates, general SPD metric',
                      class_metric='Exact rank7 factor of current equal class-contrast kernel, general SPD metric; shares class information but removes nonnegative-edge constraint',
                      cluster_edge='8 unlabelled KMeans anchors, 28 pair contrasts, same source calibration and positive weight learning'),
        capacity='28 trainable scalars each, one redundant scale removed/normalized; rank<=7 for learned contrast/PCA kernels. The unchanged nonlinear class-reference baseline has8features.',
        training='Same116source tasks/global8class-port-task balanced Brier; same Adam.03/batch8/40epochs/three seeds; fixed final epoch',
        regularization='Same .001 coefficient on centered log-edge parameters or trace-free symmetric log-metric parameters; different parameterizations are explicit',
        source_fit='Same8825source ships; KMeans3predeclared seeds n_init5, seedpaired with optimizer seeds; no best-seed choice',
        fixed='rho.25,ridge8,fusion.5, frozen backbones/source priors, no target gradients or target-based checkpoint choices',
        scope='No main-thread modifications; reused development data, not independent confirmation'))
    start = time.perf_counter()
    ev, geometry, f, meta, tasks, specs, prepared = prepare_training()
    models = train_controls(f, tasks, specs, prepared)
    evaluate(ev, geometry, specs, models, meta)
    summarize()
    write('COMPLETE.json', dict(seconds=time.perf_counter() - start, status='All runs complete'))


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
