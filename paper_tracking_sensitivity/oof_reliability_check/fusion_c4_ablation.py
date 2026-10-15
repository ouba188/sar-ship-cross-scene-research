"""Original display versus C4 global inputs in the frozen current full model."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from fusion_module_ablation import (ROOT, SOURCE, DATASETS, READOUTS, folders, metadata,
    load_npz, parts, kernel_for, write, ba)
from unified_head_core import score, combined, late
from prepare_learned_frequency import (PARENT, REFERENCE, C4, EXTERNAL, load_pickle,
    base_phi, replace_r50, normalize_direction, base_coordinates, external_phi)

OUT = ROOT/'fusion_c4_ablation_20261010'
ARMS = ('zero', 'c4')


def coordinates(folder, data, name):
    info, _ = metadata(folder)
    if info['dataset'] == 'self24ports':
        raw = np.load(C4/f'pooled_{name}.npy', mmap_mode='r')[data['rows']].astype(float)
        g = load_pickle(PARENT/info['task']/'geometry.pkl')
        head = load_pickle(REFERENCE/info['task']/'direct_heads.pkl')['joint_base']
        bank = replace_r50(raw, np.load(PARENT/info['task']/'target_bank.npy'), g)
        u = data['phi'][:, -3:-1]*np.sqrt(2)
        return dict(phi=base_phi(bank, g, u),
            prior=np.c_[bank, u]@head['coef']+head['intercept'],
            mean=normalize_direction((bank[:, :128]-g['base_center'])/g['base_scale']),
            conditions=data['conditions'])
    provenance = json.loads((EXTERNAL/info['dataset']/'FROZEN.json').read_text())
    model = load_pickle(Path(provenance.get('model_path',
        str(ROOT/DATASETS[info['dataset']][1]/'source_model.pkl'))))
    raw = np.load(EXTERNAL/info['dataset']/f'pooled_{name}.npy', mmap_mode='r')
    x = base_coordinates(raw, model)
    return dict(phi=external_phi(x, model),
        prior=model['base'].decision_function(x).astype(np.float32).astype(float),
        mean=normalize_direction((x-model['residual_center'])/model['residual_scale']),
        conditions=data['conditions'])


def predict(folder):
    from unified_head_expert import ExpertContext
    info, suffix = metadata(folder)
    dest = OUT/'targets'/suffix
    if (dest/'COMPLETE.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    data = load_npz(folder/'target.npz')
    z = load_npz(folder/'selective_embedding.npz')['embedding'].astype(float)
    expected = load_npz(ROOT/'unified_head_20261010/targets'/suffix/'policies.npz')
    for key in ('rows', 'ids', 'support', 'eval_masks'):
        np.testing.assert_array_equal(data[key], expected[key])
    matching = coordinates(folder, data, 'c4')
    for key in ('phi', 'prior', 'mean', 'conditions'):
        np.testing.assert_allclose(matching[key], data[key], atol=1e-10, rtol=1e-10)
    zero = coordinates(folder, data, 'zero')
    predictions = np.empty((5, 3, 2, len(data['rows'])), np.int8)
    context = ExpertContext(folder)
    for draw, support in enumerate(data['support']):
        pe, ke = context.target_components(data, support)
        sy = data['support_labels'][draw]
        expert, _ = score(ke, pe, support, sy)
        for ai, values in enumerate((zero, matching)):
            p = parts(values, z, support)
            kb = kernel_for(p, support, info['eta'], 'full')
            base, _ = score(kb, values['prior'], support, sy)
            unified, _ = combined(kb, ke, values['prior'], pe, support, sy)
            for ri, prediction in enumerate((base, late(base, expert), unified)):
                predictions[draw, ri, ai] = prediction.argmax(1)
        for ri, mode in enumerate(('base', 'late_softmax_fixed', 'unified_fixed')):
            np.testing.assert_array_equal(predictions[draw, ri, 1],
                expected['predictions'][draw, list(expected['modes']).index(mode)])
        print(f'C4 {suffix} draw {draw+1}/5: current inputs and full predictions exact', flush=True)
    np.savez_compressed(dest/'policies.npz', predictions=predictions, arms=np.array(ARMS),
        readouts=np.array(READOUTS), rows=data['rows'], ids=data['ids'],
        support=data['support'], eval_masks=data['eval_masks'])
    write(dest/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'],
        c4_coordinates_and_predictions_replay=True, same_64_labels=True))


def report():
    from report_fusion_module_ablation import truth_for
    if not all((OUT/'targets'/metadata(f)[1]/'COMPLETE.json').exists() for f in folders()):
        raise RuntimeError('Complete all predictions first')
    rows, classes = [], []
    for folder in folders():
        info, suffix = metadata(folder)
        p = load_npz(OUT/'targets'/suffix/'policies.npz')
        truth = truth_for(info, p)
        for draw, ev in enumerate(p['eval_masks']):
            y = truth[ev]
            for ri, readout in enumerate(READOUTS):
                before, after = p['predictions'][draw, ri, :, :][:, ev]
                for ai, arm in enumerate(ARMS):
                    pred = p['predictions'][draw, ri, ai, ev]
                    rows.append(dict(dataset=info['dataset'], task=info['task'], draw=draw,
                        readout=readout, arm=arm, oa=100*np.mean(pred == y), ba=ba(y, pred)))
                for c in np.unique(y):
                    mask = y == c
                    delta = 100*np.mean((after[mask] == c).astype(float)-(before[mask] == c))
                    classes.append(dict(dataset=info['dataset'], task=info['task'], draw=draw,
                        readout=readout, class_id=int(c), recall_delta=delta,
                        ba_contribution=delta/len(np.unique(y)), n=int(mask.sum())))
    m = pd.DataFrame(rows)
    m.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_effects.csv', index=False)
    summary = m.groupby(['dataset', 'task', 'readout', 'arm'])[['oa', 'ba']].mean().groupby(
        ['dataset', 'readout', 'arm']).mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    d = m.pivot(index=['dataset', 'task', 'draw', 'readout'], columns='arm', values=['oa', 'ba'])
    delta = pd.DataFrame(dict(delta_oa=d['oa']['c4']-d['oa']['zero'],
                              delta_ba=d['ba']['c4']-d['ba']['zero'])).reset_index()
    delta.to_csv(OUT/'paired_metrics.csv', index=False)
    draws = delta.groupby(['dataset', 'draw', 'readout'])[['delta_oa', 'delta_ba']].mean().reset_index()
    effects = draws.groupby(['dataset', 'readout']).agg(delta_oa=('delta_oa', 'mean'),
        delta_ba=('delta_ba', 'mean'), oa_positive=('delta_oa', lambda x: int((x > 0).sum())),
        ba_positive=('delta_ba', lambda x: int((x > 0).sum()))).reset_index()
    effects.to_csv(OUT/'effects.csv', index=False)
    write(OUT/'COMPLETE.json', dict(tasks=25, draws=125, metric_rows=len(m),
        representation_replay_exact=True, no_new_parameter_selection=True))
    print(effects.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('predict', 'report', 'all'), default='all')
    args = parser.parse_args()
    if not (OUT/'PROTOCOL.json').exists():
        write(OUT/'PROTOCOL.json', dict(arms=ARMS, readouts=READOUTS, target_labels=64, draws=5,
            intervention='C4 averaged global input versus matched zero-rotation extraction; source classifier parameters frozen, source scores recomputed',
            fixed='Frequency embedding and its conditions, expert, source geometry, kernel weights, ridge8',
            fitting='Refit each support residual with matched support and query input',
            scope='Frozen deployed-input ablation, not end-to-end retraining or physical angle intervention',
            new_target_selection=False, evaluation='All policies saved before truth readout'))
    if args.stage in ('predict', 'all'):
        for folder in folders():
            predict(folder)
    if args.stage in ('report', 'all'):
        report()
