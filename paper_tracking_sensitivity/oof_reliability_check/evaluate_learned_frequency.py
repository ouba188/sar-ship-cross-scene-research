"""Matched mean-correction replay and frozen learned frequency target readout."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json

import numpy as np
import pandas as pd

from prepare_learned_frequency import OUT, ROOT, SOURCE, PARENT, DATASETS, load_npz, load_pickle, write_json
from conditional_direction_residual import fit_predict, direction_kernel
from cross_dataset_c4 import route

MODES = ('baseline', 'unsplit', 'fixed', 'selective', 'selective_uniform')


def predict(data, embedding, support, labels, eta):
    phi, prior = data['phi'], data['prior']
    kernel = direction_kernel(data['mean'], data['conditions'], support, True)*eta
    kernel += .25*direction_kernel(embedding.astype(float), data['conditions'], support, True)
    _, inverse, count = np.unique(labels, return_inverse=True, return_counts=True)
    root = np.sqrt(len(labels)/(len(count)*count[inverse]))
    ks = phi[support]@phi[support].T+kernel[support]
    desired = 2*np.eye(8)[labels]-1
    dual = np.linalg.solve(root[:, None]*ks*root[None, :]+8*np.eye(len(support)),
                           root[:, None]*(desired-prior[support]))
    alpha = root[:, None]*dual
    return prior+phi@(phi[support].T@alpha)+kernel@alpha


def evaluate(folder, baseline_only=False):
    info = json.loads((folder/'PREPARED.json').read_text())
    data = load_npz(folder/'target.npz')
    if info['dataset'] == 'self24ports':
        from receiver_consistent_handoff import confidence_parts
        from evidence_aware_handoff import predict_gate, route_scores
        gate = load_pickle(PARENT/info['task']/'source_models.pkl')['gates']['joint']
        def routing(score, expert, labels, ev):
            em, eh = confidence_parts(expert)
            bm, bh = confidence_parts(score)
            result = route_scores(score, expert, predict_gate(gate, np.c_[em, bm, eh, bh]),
                                  labels, ev, data['route_ids'], budget=.05)
            return result['predictions'], result['selected']
    else:
        from pathlib import Path
        gate = load_pickle(Path(info['model_path']))['gate']
        def routing(score, expert, labels, ev):
            return route(score, expert, gate, labels, ev, data['route_ids'])
    modes = MODES[:1] if baseline_only else MODES
    embeds = {}
    gate_records = []
    for mode in modes[1:]:
        emb = load_npz(folder/f'{mode}_embedding.npz')
        np.testing.assert_array_equal(emb['rows'], data['rows'])
        np.testing.assert_allclose(np.linalg.norm(emb['embedding'], axis=1), 1, atol=2e-6)
        embeds[mode] = emb['embedding']
        for j, band in enumerate(['LL', 'LH', 'HL', 'HH']):
            gate_records.append(dict(mode=mode, band=band, mean=float(emb['weights'][:, j].mean()),
                                    std=float(emb['weights'][:, j].std()), min=float(emb['weights'][:, j].min()),
                                    max=float(emb['weights'][:, j].max())))
    shape = (5, len(modes), len(data['rows']))
    bases, fulls = np.empty(shape, np.int8), np.empty(shape, np.int8)
    selected = np.empty(shape, bool)
    scores_all = np.empty(shape+(8,), np.float32)
    diagnostics = []
    for draw, support in enumerate(data['support']):
        ev, sy = data['eval_masks'][draw], data['support_labels'][draw]
        if len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('Support mismatch or query overlap')
        baseline = fit_predict(data['phi'], data['prior'], data['mean'], data['conditions'],
                               support, sy, info['eta'], conditional=True)
        for i, mode in enumerate(modes):
            score = baseline if i == 0 else predict(data, embeds[mode], support, sy, info['eta'])
            pred, sel = routing(score, data['expert'][draw], sy, ev)
            if sel[~ev].any() or sel.sum() > int(.05*ev.sum()):
                raise ValueError('Budget exceeded')
            bp = score.argmax(1)
            if i == 0:
                for name, values in [('base', bp), ('full', pred), ('selected', sel)]:
                    np.testing.assert_array_equal(values, data['expected_'+name][draw], err_msg=folder.name)
            bases[draw, i], fulls[draw, i], selected[draw, i] = bp, pred, sel
            scores_all[draw, i] = score
            diagnostics.append(dict(draw=draw, mode=mode, score_delta_rms=float(np.sqrt(np.mean((score[ev]-baseline[ev])**2))),
                base_changes=int(np.sum(bp[ev] != baseline[ev].argmax(1))), selected=int(sel.sum())))
    if baseline_only:
        write_json(folder/'BASELINE_REPLAY.json', dict(draws=5, base_exact=True, full_exact=True, routing_exact=True,
                                                      baseline='Previous mean-representation conditional_selected'))
    else:
        np.savez_compressed(folder/'policies.npz', modes=np.array(modes), base_predictions=bases,
            predictions=fulls, selected=selected, scores=scores_all, query_rows=data['rows'], sample_ids=data['ids'],
            support_rows=data['support'], eval_masks=data['eval_masks'])
        pd.DataFrame(diagnostics).to_csv(folder/'diagnostics.csv', index=False)
        pd.DataFrame(gate_records).to_csv(folder/'gate_statistics.csv', index=False)
        write_json(folder/'EVALUATED.json', dict(dataset=info['dataset'], task=info['task'], modes=modes,
            draws=5, labels_per_draw=64, handoff_fraction=.05, baseline_exact=True,
            eta=info['eta'], beta=.25, target_query_labels_used_in_fit=0, independent_confirmation=False))
    print(('BASELINE EXACT ' if baseline_only else 'EVALUATED ')+folder.name, flush=True)


def report(folders):
    source_y = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    records, classes = [], []
    for folder in folders:
        info = json.loads((folder/'PREPARED.json').read_text())
        dataset = info['dataset']
        p, data = load_npz(folder/'policies.npz'), load_npz(folder/'target.npz')
        y = source_y[p['query_rows']] if dataset == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[dataset][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        for draw, ev in enumerate(p['eval_masks']):
            for stage, key in [('base', 'base_predictions'), ('full', 'predictions')]:
                ref = p[key][draw, 0]
                for mode, pred in zip(p['modes'], p[key][draw]):
                    acc, base = pred == y, ref == y
                    oa, boa = 100*acc[ev].mean(), 100*base[ev].mean()
                    ba = 100*np.mean([acc[ev & (y == c)].mean() for c in np.unique(y[ev])])
                    bba = 100*np.mean([base[ev & (y == c)].mean() for c in np.unique(y[ev])])
                    records.append(dict(dataset=dataset, task=info['task'], draw=draw, stage=stage, mode=mode,
                        oa=oa, ba=ba, delta_oa=oa-boa, delta_ba=ba-bba,
                        rescues=int(np.sum(ev & acc & ~base)), harms=int(np.sum(ev & ~acc & base))))
                    for c in np.unique(y[ev]):
                        mask = ev & (y == c)
                        classes.append(dict(dataset=dataset, task=info['task'], draw=draw, stage=stage, mode=mode,
                            class_id=int(c), n=int(mask.sum()), support_n=int(np.sum(data['support_labels'][draw] == c)),
                            recall=100*acc[mask].mean(), delta_recall=100*(acc[mask].mean()-base[mask].mean())))
    result = pd.DataFrame(records)
    result.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    fields = ['oa', 'ba', 'delta_oa', 'delta_ba']
    task = result.groupby(['dataset', 'task', 'stage', 'mode'])[fields].mean().reset_index()
    task.to_csv(OUT/'task_means.csv', index=False)
    summary = task.groupby(['dataset', 'stage', 'mode'])[fields].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    result.groupby(['dataset', 'draw', 'stage', 'mode'])[fields].mean().to_csv(OUT/'draw_metrics.csv')
    result.groupby(['dataset', 'stage', 'mode'])[['rescues', 'harms']].sum().to_csv(OUT/'event_summary.csv')
    print(summary[summary.stage.eq('full')].to_string(index=False), flush=True)
    write_json(OUT/'READOUT_STATUS.json', dict(completed_tasks=len(folders), expected_tasks=25,
                                             complete=len(folders) == 25, no_target_selection=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['self24ports', 'oss2', 'f0921'])
    parser.add_argument('--ports', nargs='+')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    folders = []
    if 'self24ports' in args.datasets:
        folders += [p for p in sorted((OUT/'tasks/self24ports').iterdir()) if not args.ports or p.name in args.ports]
    folders += [OUT/'tasks'/d for d in args.datasets if d != 'self24ports']
    if not args.report_only:
        for folder in folders:
            if not args.baseline_only and (folder/'EVALUATED.json').exists():
                continue
            evaluate(folder, args.baseline_only)
    if not args.baseline_only:
        report([p.parent for p in sorted((OUT/'tasks').rglob('EVALUATED.json'))])


if __name__ == '__main__':
    main()
