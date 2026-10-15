"""Locate representation vs routing effects and sample-specific gate usage."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import numpy as np
import pandas as pd

from evaluate_learned_frequency import predict
from prepare_learned_frequency import OUT, ROOT, SOURCE, PARENT, DATASETS, load_npz, load_pickle, write_json
from cross_dataset_c4 import route


def metric(y, pred):
    acc = y == pred
    return 100*acc.mean(), 100*np.mean([acc[y == c].mean() for c in np.unique(y)])


def main():
    ys = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    records, gates = [], []
    for marker in sorted((OUT/'tasks').rglob('EVALUATED.json')):
        folder = marker.parent
        info = json.loads((folder/'PREPARED.json').read_text())
        data, p = load_npz(folder/'target.npz'), load_npz(folder/'policies.npz')
        y = ys[p['query_rows']] if info['dataset'] == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        emb = load_npz(folder/'selective_embedding.npz')
        weights = emb['weights'].astype(float)
        global_weights = weights.mean(0)
        z = emb['embedding'].astype(float).reshape(len(weights), 4, 64)
        global_embedding = (z/np.sqrt(weights[:, :, None])*np.sqrt(global_weights[None, :, None])).reshape(len(weights), -1)
        np.testing.assert_allclose(np.linalg.norm(global_embedding, axis=1), 1, atol=2e-6)
        gate_info = dict(dataset=info['dataset'], task=info['task'],
            dominant_band=['LL', 'LH', 'HL', 'HH'][global_weights.argmax()],
            dominant_weight=global_weights.max(), max_sample_std=weights.std(0).max(),
            gate_entropy=float(-(weights*np.log(np.maximum(weights, 1e-30))).sum(1).mean()))
        gates.append(gate_info)
        if info['dataset'] == 'self24ports':
            from receiver_consistent_handoff import confidence_parts
            from evidence_aware_handoff import predict_gate, route_scores
            gate = load_pickle(PARENT/info['task']/'source_models.pkl')['gates']['joint']
            def routing(score, expert, labels, ev):
                em, eh = confidence_parts(expert)
                bm, bh = confidence_parts(score)
                result = route_scores(score, expert, predict_gate(gate, np.c_[em, bm, eh, bh]),
                                      labels, ev, data['route_ids'], budget=.05)
                return result['predictions']
        else:
            gate = load_pickle(Path(info['model_path']))['gate']
            def routing(score, expert, labels, ev):
                return route(score, expert, gate, labels, ev, data['route_ids'])[0]
        arm = list(p['modes']).index('selective')
        for draw, support in enumerate(data['support']):
            ev = data['eval_masks'][draw]
            sy = data['support_labels'][draw]
            current_base = p['base_predictions'][draw, arm]
            expert = data['expert'][draw].argmax(1)
            frozen_route = np.where(p['selected'][draw, 0], expert, current_base)
            global_scores = predict(data, global_embedding, support, sy, info['eta'])
            global_full = routing(global_scores, data['expert'][draw], sy, ev)
            arms = dict(baseline_base=p['base_predictions'][draw, 0], baseline_full=p['predictions'][draw, 0],
                        selective_base=current_base, selective_full=p['predictions'][draw, arm],
                        selective_old_route=frozen_route, global_gate_base=global_scores.argmax(1),
                        global_gate_full=global_full)
            for name, pred in arms.items():
                oa, ba = metric(y[ev], pred[ev])
                records.append(dict(dataset=info['dataset'], task=info['task'], draw=draw, diagnostic=name,
                    oa=oa, ba=ba, changes_from_selective=int(np.sum(ev & (pred != p['predictions'][draw, arm])))))
            for ai, name in [(0, 'baseline'), (arm, 'selective')]:
                selected = p['selected'][draw, ai]
                bp = p['base_predictions'][draw, ai]
                records.append(dict(dataset=info['dataset'], task=info['task'], draw=draw,
                    diagnostic=name+'_handoff_events', oa=np.nan, ba=np.nan,
                    rescues=int(np.sum(selected & (bp != y) & (expert == y))),
                    harms=int(np.sum(selected & (bp == y) & (expert != y))),
                    selected=int(selected.sum())))
        print('DIAGNOSED '+folder.name, flush=True)
    data = pd.DataFrame(records)
    data.to_csv(OUT/'mechanism_metrics.csv', index=False)
    pd.DataFrame(gates).to_csv(OUT/'gate_collapse.csv', index=False)
    summary = data.groupby(['dataset', 'task', 'diagnostic'])[['oa', 'ba']].mean().groupby(
        ['dataset', 'diagnostic']).mean().reset_index()
    summary.to_csv(OUT/'mechanism_summary.csv', index=False)
    print(summary.dropna().to_string(index=False), flush=True)
    write_json(OUT/'DIAGNOSIS_COMPLETE.json', dict(tasks=len(gates),
        global_gate='Mean neural weights over the unlabelled target cohort, same trained band processors',
        old_route='Previous baseline admission mask, current base predictions and original expert',
        purpose='Post-readout mechanism diagnostics, not target-selected new models'))


if __name__ == '__main__':
    main()
