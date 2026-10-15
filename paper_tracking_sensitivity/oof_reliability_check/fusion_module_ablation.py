"""Matched frozen-representation module ablations under both current readouts."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import time

import numpy as np
import pandas as pd

from prepare_learned_frequency import ROOT, SOURCE, DATASETS, load_npz
from frozen_comparison import folders, write, ba
from unified_head_core import score, combined, late

OUT = ROOT/'fusion_module_ablation_20261010'
REFERENCE = ROOT/'unified_head_20261010'
ARMS = ('phi', 'mean_plain', 'mean_uniform', 'mean_condition',
        'frequency_condition', 'full_plain', 'full_uniform', 'full')
READOUTS = ('base', 'fusion', 'unified')
EDGES = (
    ('mean_added', 'phi', 'mean_plain'),
    ('mean_shrinkage', 'mean_plain', 'mean_uniform'),
    ('mean_pair_condition', 'mean_uniform', 'mean_condition'),
    ('mean_condition_total', 'mean_plain', 'mean_condition'),
    ('frequency_added', 'mean_condition', 'full'),
    ('mean_given_frequency', 'frequency_condition', 'full'),
    ('full_shrinkage', 'full_plain', 'full_uniform'),
    ('full_pair_condition', 'full_uniform', 'full'),
    ('full_condition_total', 'full_plain', 'full'),
    ('full_stack', 'phi', 'full'),
)


def parts(data, embedding, support):
    distance = np.sum((data['conditions'][:, None]
                       - data['conditions'][support][None])**2, axis=2)
    condition = np.exp(-distance/.5)
    n = len(support)
    rho = float((condition[support].sum()-n)/(n*(n-1)))
    return dict(phi=data['phi']@data['phi'][support].T,
                mean=data['mean']@data['mean'][support].T,
                frequency=embedding@embedding[support].T,
                condition=condition, rho=rho)


def uniform_kernel(kernel, support, rho):
    result = rho*kernel
    result[support, np.arange(len(support))] = kernel[support, np.arange(len(support))]
    return result


def kernel_for(p, support, eta, arm):
    kernel = p['phi'].copy()
    if arm == 'phi':
        return kernel
    use_mean = arm != 'frequency_condition'
    use_frequency = arm.startswith('full') or arm == 'frequency_condition'
    for name, strength in [('mean', eta if use_mean else 0.),
                           ('frequency', .25 if use_frequency else 0.)]:
        if strength == 0:
            continue
        addition = p[name]
        if arm.endswith('uniform'):
            addition = uniform_kernel(addition, support, p['rho'])
        elif arm.endswith('condition') or arm == 'full':
            addition = addition*p['condition']
        kernel += strength*addition
    return kernel


def metadata(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    return info, f"{info['dataset']}/{info['task']}"


def protocol():
    path = OUT/'PROTOCOL.json'
    if path.exists():
        return
    write(path, dict(date='2026-10-10', tasks=25, draws_per_task=5, support_labels=64,
        arms=ARMS, readouts=READOUTS, paired_edges=EDGES, strength=8., ratio=.5,
        eta='Inherited per-task source choice, no new search', frequency_strength=.25,
        trainable_encoders='Frozen; ablates deployed components, not optimally retrained architectures',
        condition='Inherited unlabelled cohort ranks; RBF exp(-squared_distance/.5)',
        uniform='Support-pair offdiagonal mean RBF in inherited cohort coordinates; preserves diagonal',
        removed_condition='Only the outer relationship kernel; frozen frequency gate may still use conditions',
        prediction_boundary='Support labels only in solves; evaluation readout after all policies saved. Legacy expert metadata helper may load unused class_id.',
        unchanged='Source priors, expert, support draws, query masks, source geometry and feature caches',
        selection='None: all fixed contrasts, no target-best choice',
        interpretation='Development evidence, not independent confirmation; 5 draws are not network seeds'))


def predict_task(folder):
    from unified_head_expert import ExpertContext
    info, suffix = metadata(folder)
    dest = OUT/'targets'/suffix
    if (dest/'COMPLETE.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    data = load_npz(folder/'target.npz')
    z = load_npz(folder/'selective_embedding.npz')
    old = load_npz(REFERENCE/'targets'/suffix/'policies.npz')
    np.testing.assert_array_equal(data['rows'], z['rows'])
    for key in ('rows', 'ids', 'support', 'eval_masks'):
        np.testing.assert_array_equal(data[key], old[key])
    context = ExpertContext(folder)
    predictions = np.empty((5, len(READOUTS), len(ARMS), len(data['rows'])), np.int8)
    coefficients = np.empty((5, 2, len(ARMS), 64, 8), np.float64)
    expert_predictions, checks = [], []
    started = time.perf_counter()
    for draw, support in enumerate(data['support']):
        sy = data['support_labels'][draw]
        if len(np.unique(support)) != 64 or data['eval_masks'][draw, support].any():
            raise ValueError('Original support/query split is not preserved')
        pe, ke = context.target_components(data, support)
        expert, _ = score(ke, pe, support, sy)
        np.testing.assert_allclose(expert, data['expert'][draw], atol=3e-5, rtol=1e-6)
        expert_predictions.append(expert.argmax(1))
        p = parts(data, z['embedding'].astype(float), support)
        for ai, arm in enumerate(ARMS):
            kernel = kernel_for(p, support, info['eta'], arm)
            base, ab = score(kernel, data['prior'], support, sy)
            joint, au = combined(kernel, ke, data['prior'], pe, support, sy)
            coefficients[draw, 0, ai], coefficients[draw, 1, ai] = ab, au
            for ri, values in enumerate((base, late(base, expert), joint)):
                predictions[draw, ri, ai] = values.argmax(1)
        for ri, ref in enumerate(('base', 'late_softmax_fixed', 'unified_fixed')):
            np.testing.assert_array_equal(predictions[draw, ri, ARMS.index('full')],
                old['predictions'][draw, list(old['modes']).index(ref)])
        np.testing.assert_array_equal(expert_predictions[-1],
            old['predictions'][draw, list(old['modes']).index('expert')])
        if info['eta'] == 0:
            for ai in range(1, 4):
                np.testing.assert_array_equal(predictions[draw, :, ai], predictions[draw, :, 0])
        checks.append(dict(draw=draw, eta=info['eta'], uniform_rho=p['rho'],
                           full_prediction_replay=True))
        print(f'{suffix} draw {draw+1}/5: 8 arms x 3 readouts; full replay exact', flush=True)
    np.savez_compressed(dest/'policies.npz', predictions=predictions,
        expert_predictions=np.asarray(expert_predictions, dtype=np.int8),
        coefficients=coefficients, arms=np.array(ARMS), readouts=np.array(READOUTS),
        rows=data['rows'], ids=data['ids'], support=data['support'], eval_masks=data['eval_masks'])
    pd.DataFrame(checks).to_csv(dest/'checks.csv', index=False)
    write(dest/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'],
        seconds=time.perf_counter()-started, draws=5, arms=ARMS, readouts=READOUTS,
        prediction_replay_exact=True, query_labels_used_in_fitting=False))


def report():
    tasks = folders()
    if len(tasks) != 25 or not all((OUT/'targets'/metadata(f)[1]/'COMPLETE.json').exists() for f in tasks):
        raise RuntimeError('Freeze all 25 tasks before evaluation')
    metrics, comparisons, classes, transitions = [], [], [], []
    for folder in tasks:
        info, suffix = metadata(folder)
        p = load_npz(OUT/'targets'/suffix/'policies.npz')
        manifest, idcol = ((SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
                          else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'))
        frame = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(frame[idcol].to_numpy(str), p['ids'])
        truth = frame.class_id.to_numpy()
        common = dict(dataset=info['dataset'], task=info['task'])
        for draw, ev in enumerate(p['eval_masks']):
            y = truth[ev]
            sy = truth[p['support'][draw]]
            present = np.unique(y)
            for ri, readout in enumerate(READOUTS):
                for ai, arm in enumerate(ARMS):
                    pred = p['predictions'][draw, ri, ai, ev]
                    metrics.append(dict(**common, draw=draw, readout=readout, arm=arm,
                                        oa=100*np.mean(pred == y), ba=ba(y, pred), n=len(y)))
                for name, before, after in EDGES:
                    old = p['predictions'][draw, ri, ARMS.index(before), ev]
                    new = p['predictions'][draw, ri, ARMS.index(after), ev]
                    oc, nc = old == y, new == y
                    comparisons.append(dict(**common, draw=draw, readout=readout, contrast=name,
                        delta_oa=100*np.mean(nc.astype(float)-oc), delta_ba=ba(y, new)-ba(y, old),
                        rescued=int(np.sum(nc & ~oc)), harmed=int(np.sum(oc & ~nc)), changed=int(np.sum(old != new))))
                    for c in present:
                        mask = y == c
                        delta = 100*np.mean(nc[mask].astype(float)-oc[mask])
                        classes.append(dict(**common, draw=draw, readout=readout, contrast=name,
                            class_id=int(c), n=int(mask.sum()), support_n=int(np.sum(sy == c)),
                            recall_delta=delta, ba_contribution=delta/len(present),
                            rescued=int(np.sum(nc[mask] & ~oc[mask])), harmed=int(np.sum(oc[mask] & ~nc[mask]))))
            for name, before, after in EDGES:
                a, b = ARMS.index(before), ARMS.index(after)
                b0, b1 = p['predictions'][draw, 0, a, ev] == y, p['predictions'][draw, 0, b, ev] == y
                f0, f1 = p['predictions'][draw, 1, a, ev] == y, p['predictions'][draw, 1, b, ev] == y
                rescue, harm = ~b0 & b1, b0 & ~b1
                transitions.append(dict(**common, draw=draw, contrast=name,
                    base_rescues=int(rescue.sum()), base_harms=int(harm.sum()),
                    rescued_already_correct_in_both_fusions=int(np.sum(rescue & f0 & f1)),
                    rescued_transmitted_to_fusion=int(np.sum(rescue & ~f0 & f1)),
                    rescued_but_both_fusions_wrong=int(np.sum(rescue & ~f0 & ~f1)),
                    rescued_but_fusion_harmed=int(np.sum(rescue & f0 & ~f1)),
                    base_harms_protected_by_both_fusions=int(np.sum(harm & f0 & f1)),
                    base_harms_transmitted_to_fusion=int(np.sum(harm & f0 & ~f1))))
    m, c, cl, tr = map(pd.DataFrame, (metrics, comparisons, classes, transitions))
    m.to_csv(OUT/'metrics.csv', index=False)
    c.to_csv(OUT/'paired_metrics.csv', index=False)
    cl.to_csv(OUT/'class_effects.csv', index=False)
    tr.to_csv(OUT/'fusion_transitions.csv', index=False)
    summary = m.groupby(['dataset', 'task', 'readout', 'arm'])[['oa', 'ba']].mean().groupby(
        ['dataset', 'readout', 'arm']).mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    draws = c.groupby(['dataset', 'draw', 'readout', 'contrast'])[['delta_oa', 'delta_ba']].mean().reset_index()
    draws.to_csv(OUT/'draw_effects.csv', index=False)
    effects = draws.groupby(['dataset', 'readout', 'contrast']).agg(
        delta_oa=('delta_oa', 'mean'), delta_ba=('delta_ba', 'mean'),
        oa_positive=('delta_oa', lambda x: int((x > 1e-10).sum())),
        ba_positive=('delta_ba', lambda x: int((x > 1e-10).sum()))).reset_index()
    effects.to_csv(OUT/'effects.csv', index=False)
    # A missing class in a port contributes zero to the dataset-level BA contrast.
    task_count = m.groupby('dataset').task.nunique()
    class_summary = cl.groupby(['dataset', 'readout', 'contrast', 'class_id']).agg(
        ba_contribution=('ba_contribution', 'sum'), rescued=('rescued', 'sum'), harmed=('harmed', 'sum')).reset_index()
    class_summary['ba_contribution'] /= class_summary.dataset.map(task_count)*5
    class_summary.to_csv(OUT/'class_ba_contributions.csv', index=False)
    tr.groupby(['dataset', 'contrast']).sum(numeric_only=True).drop(columns='draw').to_csv(OUT/'transition_totals.csv')
    ref = pd.read_csv(REFERENCE/'summary.csv')
    for readout, mode in zip(READOUTS, ('base', 'late_softmax_fixed', 'unified_fixed')):
        actual = summary[(summary.readout == readout) & (summary.arm == 'full')].sort_values('dataset')
        expected = ref[ref['mode'] == mode].sort_values('dataset')
        np.testing.assert_allclose(actual[['oa', 'ba']], expected[['oa', 'ba']], atol=1e-10, rtol=0)
    reconstructed = class_summary.groupby(['dataset', 'readout', 'contrast']).ba_contribution.sum().sort_index()
    np.testing.assert_allclose(reconstructed.to_numpy(),
        effects.set_index(['dataset', 'readout', 'contrast']).delta_ba.sort_index().to_numpy(), atol=1e-10)
    write(OUT/'COMPLETE.json', dict(tasks=25, draws=125, metric_rows=len(m),
        full_replay_exact=True, class_ba_accounting_exact=True, no_target_selection=True,
        new_training=False, independent_confirmation=False))
    print(effects[effects.readout == 'fusion'].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('predict', 'report', 'all'), default='all')
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    protocol()
    if args.stage in ('predict', 'all'):
        for folder in folders():
            if not args.tasks or metadata(folder)[0]['task'] in args.tasks:
                predict_task(folder)
    if args.stage in ('report', 'all'):
        report()


if __name__ == '__main__':
    main()
