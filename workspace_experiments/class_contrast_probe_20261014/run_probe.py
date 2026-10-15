"""Frozen source-only class-contrast feasibility probe, isolated from main work."""
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
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
CORE = OUT.parent / 'class_reference_core_20261014'
sys.path.insert(0, str(CORE))
from model import ReferenceSupportModel, reference_map
spec = importlib.util.spec_from_file_location('contrast_core_runner', CORE / 'run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
spec = importlib.util.spec_from_file_location('contrast_core_analysis', CORE / 'analyze.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
ARMS = ['original', 'class_mean', 'contrast_linear', 'contrast_equal', 'contrast_weighted']
PAIRS = np.array([(a, b) for a in range(8) for b in range(a + 1, 8)])


def contrast(z, bank):
    centers = bank['centers']
    h = (2 * z @ centers.T - np.sum(centers * centers, axis=1)) / bank['tau2']
    return h[:, PAIRS[:, 0]] - h[:, PAIRS[:, 1]]


def mapping(z, bank):
    if bank['kind'] == 'class_mean':
        return reference_map(z, bank)
    v = contrast(z, bank) / bank['pair_scale']
    if bank['kind'] != 'contrast_linear':
        v = np.tanh(v)
    v = v * np.sqrt(bank['weights'])
    return (v - bank['center']) * bank['scale']


class ContrastModel(ReferenceSupportModel):
    def kernel(self, left, right):
        k = (1 - self.rho) * (left['z'] @ right['z'].T + 1)
        if self.bank is not None:
            k += self.rho * (mapping(left['z'], self.bank) @ mapping(right['z'], self.bank).T)
        return k


def source_rows(name, template):
    if name == 'external':
        return template['source_rows']
    rows = np.load(run.PILOT / 'source_rows.npy')
    d = run.old.source_metadata()
    ships = d.iloc[rows].identity.to_numpy()
    rng = np.random.default_rng(20261013)
    chosen = [rng.choice(np.flatnonzero(ships == ship)) for ship in np.unique(ships)]
    return rows[chosen]


def fit_banks(z, labels, template, name):
    t = time.perf_counter()
    with (CORE / 'normalization_repair' / f'{name}_banks.pkl').open('rb') as h:
        previous = pickle.load(h)['class_mean']
    bank0 = dict(previous, kind='class_mean')
    rows = source_rows(name, template)
    meta = run.old.source_metadata().iloc[rows]
    np.testing.assert_array_equal(labels, meta.class_id.to_numpy(int))
    np.testing.assert_allclose(bank0['centers'], np.stack([z[labels == c].mean(0) for c in range(8)]), atol=1e-10)
    d = contrast(z, bank0)
    scale = d.std(axis=0)
    assert np.all(scale > 0)
    stats = []
    for j, (a, b) in enumerate(PAIRS):
        gap = d[labels == a, j].mean() - d[labels == b, j].mean()
        within = .5 * (d[labels == a, j].var() + d[labels == b, j].var())
        drifts, group_counts = [], []
        for c in (a, b):
            frame = pd.DataFrame({'acquisition': meta.acquisition.to_numpy()[labels == c], 'value': d[labels == c, j]})
            grouped = frame.groupby('acquisition').value.agg(['mean', 'size'])
            means = grouped.loc[grouped['size'] >= 2, 'mean'].to_numpy()
            group_counts.append(len(means))
            drifts.append(float(np.var(means)) if len(means) >= 2 else 0.)
        drift = float(np.mean(drifts))
        score = gap * gap / (within + drift + .01 * scale[j] ** 2)
        stats.append(dict(class_a=int(a), class_b=int(b), gap=gap, within=within,
                          acquisition_scatter=drift, acquisitions_a=group_counts[0],
                          acquisitions_b=group_counts[1], raw_weight=score, pair_scale=scale[j]))
    weights = np.array([r['raw_weight'] for r in stats])
    weights /= weights.mean()
    for i, row in enumerate(stats):
        row['weight'] = weights[i]
    pd.DataFrame(stats).to_csv(OUT / f'{name}_pair_weights.csv', index=False)
    banks = {'original': None, 'class_mean': bank0}
    energy = np.mean(np.sum(z * z, axis=1) + 1)
    diagnostics = []
    for arm in ARMS[2:]:
        v = d / scale
        if arm != 'contrast_linear':
            v = np.tanh(v)
        w = weights if arm == 'contrast_weighted' else np.ones(len(PAIRS))
        v = v * np.sqrt(w)
        center = v.mean(0)
        gain = np.sqrt(energy / np.mean(np.sum((v - center) ** 2, axis=1)))
        banks[arm] = dict(bank0, kind=arm, pair_scale=scale, weights=w, center=center, scale=gain)
        mapped = mapping(z[:128], banks[arm])
        np.testing.assert_allclose(mapped, (v[:128] - center) * gain, atol=1e-10)
        eig = np.linalg.eigvalsh(mapped @ mapped.T)
        assert eig.min() > -1e-8 * max(1., eig.max())
        diagnostics.append(dict(arm=arm, energy=energy, map_scale=gain,
                                weight_effective_pairs=float(w.sum() ** 2 / np.sum(w * w))))
    with (OUT / f'{name}_banks.pkl').open('wb') as h:
        pickle.dump(banks, h)
    print('BANK', name, len(z), 'seconds', time.perf_counter() - t, flush=True)
    return banks, dict(domain=name, n=len(z), ships=meta.identity.nunique(),
                       acquisitions=meta.acquisition.nunique(), source_class_counts=np.bincount(labels, minlength=8).tolist(),
                       diagnostics=diagnostics)


def summarize():
    metrics = pd.read_csv(OUT / 'metrics.csv')
    draw = analysis.aggregate(analysis.add_scopes(metrics), analysis.KEY + ['arm'], analysis.VALUES)
    draw.to_csv(OUT / 'draw_metrics.csv', index=False)
    records = []
    for keys, group in draw.groupby(analysis.KEY):
        for metric in analysis.VALUES:
            pivot = group.pivot(index='draw', columns='arm', values=metric)
            for arm in ARMS[1:]:
                for baseline in ['original', 'class_mean', 'contrast_equal']:
                    if arm == baseline or (arm == 'class_mean' and baseline != 'original'):
                        continue
                    delta = pivot[arm] - pivot[baseline]
                    records.append(dict(zip(analysis.KEY, keys), arm=arm, baseline=baseline, metric=metric,
                                        absolute=float(pivot[arm].mean()), delta=float(delta.mean()),
                                        sd=float(delta.std()), positive=int((delta > 1e-10).sum()),
                                        negative=int((delta < -1e-10).sum()), draws=len(delta)))
    effects = pd.DataFrame(records)
    effects.to_csv(OUT / 'effects_summary.csv', index=False)
    cases = pd.read_csv(OUT / 'case_paths.csv', dtype={'ship': str})
    cases['net'] = cases.new_correct.astype(int) - cases.old_correct.astype(int)
    cases['map_delta'] = cases.map_new - cases.map_old
    ck = ['dataset', 'study', 'cohort', 'budget', 'cell', 'arm', 'draw', 'port', 'episode', 'class_id']
    class_effects = cases.groupby(ck)[['net', 'margin_delta', 'relation_margin', 'refit_margin', 'map_delta']].mean().reset_index()
    class_effects.to_csv(OUT / 'class_effects.csv', index=False)
    scene = class_effects.groupby(ck[:-1])[['net', 'margin_delta', 'map_delta']].mean().reset_index()
    path = analysis.aggregate(analysis.add_scopes(scene), ['dataset', 'study', 'cohort', 'budget', 'cell', 'arm'], ['net', 'margin_delta', 'map_delta'])
    path.to_csv(OUT / 'path_draws.csv', index=False)
    for _, row in path.iterrows():
        m = draw[(draw.dataset == row.dataset) & (draw.study == row.study) & (draw.cohort == row.cohort) &
                 (draw.budget == row.budget) & (draw.cell == row.cell) & (draw.draw == row.draw) & (draw['mode'] == 'full')]
        expected = m.loc[m.arm == row.arm, 'ba'].iloc[0] - m.loc[m.arm == 'original', 'ba'].iloc[0]
        np.testing.assert_allclose(row.net * 100, expected, atol=1e-10)
    events = []
    for keys, g in cases.groupby(['dataset', 'cohort', 'budget', 'cell', 'arm', 'class_id']):
        events.append(dict(zip(['dataset', 'cohort', 'budget', 'cell', 'arm', 'class_id'], keys),
                           exposures=len(g), ships=g.ship.nunique(), rescued=int((g.net > 0).sum()),
                           harmed=int((g.net < 0).sum()), rescued_ships=g.loc[g.net > 0, 'ship'].nunique(),
                           harmed_ships=g.loc[g.net < 0, 'ship'].nunique(), missing_support=int((g.correct_support == 0).sum())))
    pd.DataFrame(events).to_csv(OUT / 'class_events.csv', index=False)
    (OUT / 'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(macro_reconstructions=len(path)), indent=2), encoding='utf-8')
    print(effects[(effects['mode'] == 'full') & (effects.metric == 'ba') & (effects.baseline == 'class_mean')].to_string(index=False), flush=True)


def main():
    protocol = dict(status='Frozen before target outcomes', scope='Small cached-feature operator probe; no main-thread files changed',
                    arms=ARMS, pairs=PAIRS.tolist(), formula='d_ab=(2z dot(mu_a-mu_b)-||mu_a||2+||mu_b||2)/tau2',
                    scaling='Per-pair source standard deviation; source centering and common energy match in every arm',
                    weight='gap_ab^2/(0.5*(within_a+within_b)+0.5*(acqmean_variance_a+acqmean_variance_b)+0.01*pair_variance); mean weight=1',
                    drift='Class-conditional acquisition mean scatter, >=2 ships per acquisition; fewer than2 eligible acquisitions use0 and report coverage. Not same-vessel causal drift.',
                    fixed=dict(rho=.25, ridge=8, expert_fusion=.5, expert_dimensions=2752),
                    enrollment='Existing acquisition_transfer cells only; all external5scenes B16/B64 same/other; all48 portscenes B64 other only; five fixed support draws',
                    exclusion='No global task, target tuning, neural training, dynamic attention, new source head, or episodic loss training',
                    evidence='Development data reused; five support draws are not five backbone trainings; no independent confirmation claim')
    (OUT / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    start = time.perf_counter()
    run.OUT = OUT
    run.ARMS = ARMS
    run.banks = fit_banks
    ev = run.old.operator()
    contexts, external_info = run.prepare_external(ev)
    ports, port_info = run.prepare_ports()
    contexts.append(ports)
    filtered = []
    replays = 0
    for name, f, meta, banks, tasks in contexts:
        tasks = [t for t in tasks if t['study'] == 'acquisition_transfer' and
                 (name != 'ports' or t['cell'] == 'other_acquisition')]
        for task in tasks:
            tag = f"{name}_{task['episode']:03d}_{task['budget']}_{task['draw']}_{task['cell']}"
            archive = run.load_npz(CORE / 'normalization_repair/predictions' / f'{tag}.npz')
            np.testing.assert_array_equal(archive['query_rows'], task['query'])
            np.testing.assert_array_equal(archive['support_rows'], task['support'])
            task['old_class'] = archive['full'][list(archive['arms']).index('class_mean')]
            replays += 1
        filtered.append((name, f, meta, banks, tasks))
    run.write('BANK_FITS.json', [external_info, port_info])
    print('TASKS', [(c[0], len(c[4])) for c in filtered], 'paired archives', replays, flush=True)
    run.evaluate(filtered, ev, start, model_type=ContrastModel)
    receipt = json.loads((OUT / 'COMPUTATION_COMPLETE.json').read_text())
    receipt['cluster_seeds'] = []
    receipt['source_weight_rule_fits'] = 2
    run.write('COMPUTATION_COMPLETE.json', receipt)
    summarize()


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
