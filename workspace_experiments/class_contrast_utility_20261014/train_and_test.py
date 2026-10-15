"""Learn 28 positive contrast weights through the actual support residual solve."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
BALANCED = '--balanced' in sys.argv
if BALANCED:
    OUT = OUT / 'balanced_objective'
PROBE = OUT.parent / 'class_contrast_probe_20261014'
if BALANCED:
    PROBE = OUT.parent.parent / 'class_contrast_probe_20261014'
sys.path.insert(0, str(PROBE))
import run_probe as probe
run = probe.run
SEEDS = [20261014, 20261015, 20261016]
ARMS = ['original', 'class_mean', 'contrast_linear', 'contrast_weighted'] + [f'utility_{i}' for i in range(3)]
EPOCHS = 40


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


def weighted_bank(linear, weights):
    return dict(linear, weights=weights, center=linear['center'] * np.sqrt(weights))


def fit_linear_bank(z, template):
    d = probe.contrast(z, template)
    pair_scale = d.std(0)
    v = d / pair_scale
    energy = np.mean(np.sum(z * z, axis=1) + 1)
    return dict(template, kind='contrast_linear', weights=np.ones(28), pair_scale=pair_scale,
                center=v.mean(0), scale=np.sqrt(energy / np.mean(np.sum((v - v.mean(0)) ** 2, axis=1))))


def enroll(meta):
    rng = np.random.default_rng(20261014)
    tasks, coverage = [], []
    for port, p in meta.groupby('port', sort=True):
        eligible = []
        for acquisition, group in p.groupby('acquisition', sort=True):
            if len(group) >= 8 and group.class_id.nunique() >= 2 and len(p) - len(group) >= 64:
                eligible.append((acquisition, group))
        coverage.append(dict(port=port, ships=len(p), classes=p.class_id.nunique(), eligible_scenes=len(eligible)))
        if not eligible:
            continue
        picks = np.unique(np.rint(np.linspace(0, len(eligible) - 1, min(2, len(eligible)))).astype(int))
        for i in picks:
            acq, group = eligible[i]
            # One source observation per vessel, so scene exclusion also isolates identities.
            q = rng.choice(group.index.to_numpy(), min(48, len(group)), replace=False)
            pool = p[p.acquisition != acq]
            for draw in range(2):
                support = run.old.ext.sample_ships(pool, 64, rng)
                for budget in (16, 64):
                    tasks.append(dict(port=port, acquisition=acq, draw=draw, budget=budget,
                                      query=q.tolist(), support=support[:budget]))
    pd.DataFrame(coverage).to_csv(OUT / 'source_coverage.csv', index=False)
    return tasks


def prepare_task(f, meta, bank, task, ev):
    q, s = np.array(task['query']), np.array(task['support'])
    query, support = run.select(f, q), run.select(f, s)
    y, sy = meta.loc[q].class_id.to_numpy(int), meta.loc[s].class_id.to_numpy(int)
    assert not set(meta.loc[q].identity) & set(meta.loc[s].identity)
    assert not set(meta.loc[q].acquisition) & set(meta.loc[s].acquisition)
    vs = probe.contrast(support['z'], bank) / bank['pair_scale'] - bank['center']
    vq = probe.contrast(query['z'], bank) / bank['pair_scale'] - bank['center']
    kbss = support['b'] @ support['b'].T + 1
    kbqs = query['b'] @ support['b'].T + 1
    bp = softmax(query['pb'] + kbqs @ ev.coefficients(kbss, support['pb'], sy, 8.), axis=1)
    _, inv, counts = np.unique(sy, return_inverse=True, return_counts=True)
    root = np.sqrt(len(sy) / (len(counts) * counts[inv]))
    _, invq, nq = np.unique(y, return_inverse=True, return_counts=True)
    qw = 1. / (len(nq) * nq[invq])
    return dict(kss=tensor(.75 * (support['z'] @ support['z'].T + 1)),
                kqs=tensor(.75 * (query['z'] @ support['z'].T + 1)),
                vs=tensor(vs), vq=tensor(vq), root=tensor(root), eye=tensor(np.eye(len(s))),
                target=tensor(2 * np.eye(8)[sy] - 1 - support['pd']),
                prior=tensor(query['pd']), base=tensor(bp), truth=tensor(np.eye(8)[y]),
                qw=tensor(qw), gain2=.25 * bank['scale'] ** 2, port=task['port'], y=y)


def forward(theta, task):
    weights = 28 * torch.softmax(theta, dim=0)
    kss = task['kss'] + task['gain2'] * ((task['vs'] * weights) @ task['vs'].T)
    kqs = task['kqs'] + task['gain2'] * ((task['vq'] * weights) @ task['vs'].T)
    root = task['root']
    coef = root[:, None] * torch.linalg.solve(root[:, None] * kss * root[None, :] + 8 * task['eye'],
                                             root[:, None] * task['target'])
    expert = torch.softmax(task['prior'] + kqs @ coef, dim=1)
    full = .5 * (task['base'] + expert)
    risk = torch.sum(task['qw'] * torch.sum((full - task['truth']) ** 2, dim=1))
    return risk, full


def train():
    started = time.perf_counter()
    ev = run.old.operator()
    with (run.PREV / 'validation_fit/geometry.pkl').open('rb') as h:
        geometry = pickle.load(h)
    with (run.PREV / 'validation_memory.pkl').open('rb') as h:
        template = pickle.load(h)
    rows = np.asarray(template['source_rows'])
    meta_all = run.old.source_metadata()
    meta = meta_all.loc[rows]
    f = run.old.encoded(ev, run.old.source_arrays(), rows, geometry)
    order = np.argsort(f['rows'])
    f = {k: v[order] for k, v in f.items()}
    bank = fit_linear_bank(f['z'], template)
    tasks = enroll(meta)
    assert tasks
    write('TRAIN_ENROLLMENT.json', tasks)
    # Check actual new supervision against all external known identities.
    isolation = {}
    for name in ('f0921', 'oss2'):
        target = pd.read_csv(run.old.OLD / f'{name}_metadata.csv', dtype={'identity': str})
        overlap = set(meta.identity) & set(target.identity)
        isolation[name + '_training_ship_overlap'] = len(overlap)
        assert not overlap
    held = json.loads((run.PREV / 'SOURCE_ISOLATION.json').read_text())
    assert not set(meta.port) & set(held['held_ports'])
    isolation['held_ports'] = held['held_ports']
    isolation['source_geometry'] = str(run.PREV / 'validation_fit/geometry.pkl')
    write('ISOLATION.json', isolation)
    prepared = [prepare_task(f, meta_all, bank, task, ev) for task in tasks]
    # Port-balanced training prevents ports with more enrolled scenes dominating.
    port_counts = pd.Series([t['port'] for t in tasks]).value_counts()
    task_weights = np.array([len(tasks) / (len(port_counts) * port_counts[t['port']]) for t in tasks])
    if BALANCED:
        # Equal classes globally, then equal eligible ports and tasks within each class.
        presence = pd.DataFrame([dict(port=p['port'], class_id=int(c))
                                 for p in prepared for c in np.unique(p['y'])])
        counts = presence.groupby(['port', 'class_id']).size()
        ports_per_class = presence.groupby('class_id').port.nunique()
        assert len(ports_per_class) == 8
        masses = np.zeros(8)
        for p in prepared:
            y = p['y']
            nc = np.bincount(y, minlength=8)
            qw = np.array([len(tasks) / (8 * ports_per_class[c] * counts[(p['port'], c)] * nc[c]) for c in y])
            p['qw'] = tensor(qw)
            for c in np.unique(y):
                masses[c] += qw[y == c].sum() / len(tasks)
        np.testing.assert_allclose(masses, np.repeat(.125, 8), atol=1e-12)
        task_weights = np.ones(len(tasks))
        write('OBJECTIVE_CLASS_MASS.json', dict(mass=masses.tolist(),
              rule='Equal8classes, then equal eligible ports, then equal tasks containing class; query samples uniform within task/class'))
    example = tasks[0]
    baseline = probe.ContrastModel(ev.coefficients, bank, rho=.25).fit_support(
        run.select(f, example['support']), meta_all.loc[example['support']].class_id.to_numpy(int))
    actual = baseline.predict(run.select(f, example['query']))['full']
    _, result = forward(torch.zeros(28, dtype=torch.float64), prepared[0])
    np.testing.assert_allclose(actual, result.numpy(), atol=1e-10, rtol=1e-10)
    theta = torch.zeros(28, dtype=torch.float64, requires_grad=True)
    risk, _ = forward(theta, prepared[0])
    grad = torch.autograd.grad(risk, theta)[0]
    plus = torch.zeros(28, dtype=torch.float64); plus[0] = 1e-5
    fd = (forward(plus, prepared[0])[0] - forward(-plus, prepared[0])[0]).item() / 2e-5
    np.testing.assert_allclose(fd, grad[0].item(), atol=1e-8, rtol=1e-4)
    write('TRAIN_CHECKS.json', dict(torch_numpy_max_error=float(abs(actual - result.numpy()).max()),
                                  gradient_finite_difference=fd, autograd=float(grad[0]), tasks=len(tasks)))
    logs, learned = [], []
    for seed_index, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        theta = torch.nn.Parameter(torch.randn(28, dtype=torch.float64) * .001)
        optimizer = torch.optim.Adam([theta], lr=.03)
        for epoch in range(EPOCHS + 1):
            if epoch:
                order = rng.permutation(len(tasks))
                for start in range(0, len(order), 8):
                    indexes = order[start:start + 8]
                    optimizer.zero_grad()
                    losses = [forward(theta, prepared[i])[0] * task_weights[i] for i in indexes]
                    loss = torch.stack(losses).mean() + .001 * torch.mean((theta - theta.mean()) ** 2)
                    loss.backward()
                    optimizer.step()
            if epoch % 10 == 0 or epoch == EPOCHS:
                records = []
                with torch.no_grad():
                    for p in prepared:
                        risk, full = forward(theta, p)
                        records.append(dict(port=p['port'], risk=risk.item(), **run.old.scores(p['y'], full.numpy())))
                means = pd.DataFrame(records).groupby('port').mean().mean()
                if BALANCED:
                    means['risk'] = pd.DataFrame(records).risk.mean()
                logs.append(dict(seed=seed, epoch=epoch, **means.to_dict()))
                print('TRAIN', seed, epoch, 'risk', means['risk'], 'BA', means['ba'], flush=True)
        weights = (28 * torch.softmax(theta.detach(), 0)).numpy()
        learned.append(weights)
        pd.DataFrame(dict(class_a=probe.PAIRS[:, 0], class_b=probe.PAIRS[:, 1], weight=weights)).to_csv(
            OUT / f'weights_{seed}.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'training_curve.csv', index=False)
    np.save(OUT / 'frozen_weights.npy', np.array(learned))
    with (OUT / 'training_bank.pkl').open('wb') as h:
        pickle.dump(bank, h)
    write('TRAIN_COMPLETE.json', dict(seconds=time.perf_counter() - started, seeds=SEEDS, epochs=EPOCHS,
        tasks=len(tasks), source_ships=len(meta), task_ports=list(port_counts.index),
        weight_rule='Final fixed epoch, no held-target selection, all three seeds reported',
        status='Weights frozen before external evaluation', parameters=28))
    return np.array(learned)


def load_banks(z, labels, template, name):
    with (PROBE / f'{name}_banks.pkl').open('rb') as h:
        prior = pickle.load(h)
    banks = {arm: prior[arm] for arm in ARMS[:4]}
    weights = np.load(OUT / 'frozen_weights.npy')
    linear = prior['contrast_linear']
    for i, w in enumerate(weights):
        banks[f'utility_{i}'] = weighted_bank(linear, w)
        # Unit-variance source pair coordinates preserve energy for mean-one weights.
        v = probe.contrast(z, linear) / linear['pair_scale']
        energy = np.mean(np.sum((v - v.mean(0)) ** 2 * w, axis=1))
        np.testing.assert_allclose(energy, 28., atol=1e-8)
    with (OUT / f'{name}_banks.pkl').open('wb') as h:
        pickle.dump(banks, h)
    return banks, dict(domain=name, n=len(z), deployment='Frozen learned weights transferred to existing source-calibrated pair coordinates')


def test():
    start = time.perf_counter()
    ev = run.old.operator()
    run.OUT = OUT; run.ARMS = ARMS; run.banks = load_banks
    contexts, ei = run.prepare_external(ev)
    ports, pi = run.prepare_ports(); contexts.append(ports)
    filtered = []
    for name, f, meta, banks, tasks in contexts:
        tasks = [t for t in tasks if t['study'] == 'acquisition_transfer' and
                 (name != 'ports' or t['cell'] == 'other_acquisition')]
        for task in tasks:
            tag = f"{name}_{task['episode']:03d}_{task['budget']}_{task['draw']}_{task['cell']}"
            archive = run.load_npz(PROBE / 'predictions' / f'{tag}.npz')
            np.testing.assert_array_equal(archive['query_rows'], task['query'])
            np.testing.assert_array_equal(archive['support_rows'], task['support'])
            task['old_class'] = archive['full'][list(archive['arms']).index('class_mean')]
        filtered.append((name, f, meta, banks, tasks))
    run.write('BANK_FITS.json', [ei, pi])
    run.evaluate(filtered, ev, start, model_type=probe.ContrastModel)
    receipt = json.loads((OUT / 'COMPUTATION_COMPLETE.json').read_text())
    receipt['cluster_seeds'] = []; receipt['readout_training_seeds'] = SEEDS
    receipt['neural_training'] = 'Only 28 contrast logits x3 seeds; zero backbone/head fits'
    run.write('COMPUTATION_COMPLETE.json', receipt)
    probe.OUT = OUT; probe.ARMS = ARMS
    # Existing summary needs a named equal control; use linear as the simple counterpart.
    summarize()


def summarize():
    data = pd.read_csv(OUT / 'metrics.csv')
    analysis = probe.analysis
    draw = analysis.aggregate(analysis.add_scopes(data), analysis.KEY + ['arm'], analysis.VALUES)
    draw.to_csv(OUT / 'draw_metrics.csv', index=False)
    rows = []
    for keys, group in draw.groupby(analysis.KEY):
        for metric in analysis.VALUES:
            pivot = group.pivot(index='draw', columns='arm', values=metric)
            for baseline in ['class_mean', 'contrast_linear', 'contrast_weighted', 'original']:
                candidates = {f'utility_{i}': pivot[f'utility_{i}'] for i in range(3)}
                candidates['utility_seed_mean'] = pivot[[f'utility_{i}' for i in range(3)]].mean(axis=1)
                for arm, values in candidates.items():
                    delta = values - pivot[baseline]
                    rows.append(dict(zip(analysis.KEY, keys), arm=arm, baseline=baseline, metric=metric,
                                     absolute=values.mean(), delta=delta.mean(), sd_draw=delta.std(),
                                     positive=int((delta > 1e-10).sum()), negative=int((delta < -1e-10).sum())))
    effects = pd.DataFrame(rows)
    effects.to_csv(OUT / 'effects_summary.csv', index=False)
    print(effects[(effects['mode'] == 'full') & (effects.metric == 'ba') &
                  (effects.arm == 'utility_seed_mean') & (effects.baseline == 'class_mean')].to_string(index=False), flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    write('PROTOCOL.json', dict(status='Fixed before training and new target outcomes', source='Existing external validation geometry/source memory excluding Singapore/JebelAli/PortKlang and their identities/acquisitions',
        source_tasks='At most2 evenly spaced eligible acquisition groups per training port; >=8ships/2classes; other-scene support64, prefix16;2draws;query<=48',
        target='Previous440 acquisition support cells, no target gradients, stopping or version selection',
        method='28 positive mean-one linear contrast weights; autograd through class-balanced support ridge; fixed .5 probability fusion',
        objective=('Global8class/eligible-port/task-balanced Brier' if BALANCED else 'Port-balanced query macro-Brier') + ' + .001 centered-logit L2', optimizer='Adam lr.03, batch8 tasks',
        epochs=EPOCHS, seeds=SEEDS, checkpoint='Fixed final epoch, all seeds reported; mean performance is not a prediction ensemble',
        unchanged='Frozen encoders, source priors, reference centers, source pair scales, ridge8,rho.25,fusion.5',
        limitations='Source-training query labels previously fit source heads; training loss is not held-out utility; historical exposure/development reuse remains',
        scope='Independent side experiment; no main-thread source/model/manuscript/memory edits'))
    train()
    test()


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
