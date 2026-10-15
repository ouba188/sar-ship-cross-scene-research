"""Read out predeclared information-removal interventions on all tasks."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_learned_frequency import OUT as EXP, ROOT, SOURCE, PARENT, DATASETS, load_npz, load_pickle
from frequency_information_content import pooled_intervention
from evaluate_learned_frequency import predict
from explain_frequency_gain import metrics
from cross_dataset_c4 import route

OUT = ROOT/'frequency_information_content_20261009'


def evaluate(folder, retrained=False):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/info['dataset']/info['task']
    if (out/'EVALUATED.json').exists():
        return
    if not (out/'ENCODED.json').exists():
        raise FileNotFoundError('Wait for band encoding: '+str(out))
    data = load_npz(folder/'target.npz')
    policy = load_npz(folder/'policies.npz')
    archive = load_npz(folder/'selective_embedding.npz')
    np.testing.assert_array_equal(archive['rows'], data['rows'])
    embeddings = {'actual': archive['embedding']}
    for mode in (() if retrained else ('no_quadrant_layout', 'no_channel_identity')):
        embeddings[mode] = pooled_intervention(archive['embedding'], archive['weights'], mode)
    for mode in ('unsigned_bands', 'no_local_band_layout'):
        z = load_npz(out/f'{mode}.npz')
        np.testing.assert_array_equal(z['rows'], data['rows'])
        embeddings[mode] = z['embedding']
    if info['dataset'] == 'self24ports':
        y = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()[data['rows']]
        from receiver_consistent_handoff import confidence_parts
        from evidence_aware_handoff import predict_gate, route_scores
        gate = load_pickle(PARENT/info['task']/'source_models.pkl')['gates']['joint']
        def routing(score, expert, labels, ev):
            em, eh = confidence_parts(expert)
            bm, bh = confidence_parts(score)
            ret = route_scores(score, expert, predict_gate(gate, np.c_[em, bm, eh, bh]),
                               labels, ev, data['route_ids'], budget=.05)
            return ret['predictions'], ret['selected']
    else:
        y = pd.read_csv(ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        gate = load_pickle(Path(info['model_path']))['gate']
        def routing(score, expert, labels, ev):
            return route(score, expert, gate, labels, ev, data['route_ids'])
    arm = list(policy['modes']).index('selective')
    records, classes, magnitudes, strata = [], [], [], []
    meta = dict(dataset=info['dataset'], task=info['task'])
    for mode, emb in embeddings.items():
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1., atol=2e-6)
        magnitudes.append(dict(**meta, mode=mode,
            mean_cosine_to_actual=(np.nan if retrained and mode != 'actual' else
                float(np.mean(np.sum(emb*embeddings['actual'], axis=1))))))
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        for mode, emb in embeddings.items():
            score = predict(data, emb, support, sy, info['eta'])
            base = score.argmax(1)
            full, selected = routing(score, data['expert'][draw], sy, ev)
            if selected[~ev].any() or selected.sum() > int(.05*ev.sum()):
                raise ValueError('Handoff budget mismatch')
            if mode == 'actual':
                np.testing.assert_array_equal(base, policy['base_predictions'][draw, arm])
                np.testing.assert_array_equal(full, policy['predictions'][draw, arm])
                np.testing.assert_array_equal(selected, policy['selected'][draw, arm])
            frozen = np.where(policy['selected'][draw, arm], data['expert'][draw].argmax(1), base)
            for stage, pred in [('base', base), ('full', full), ('fixed_actual_route', frozen)]:
                reference = policy['base_predictions' if stage == 'base' else 'predictions'][draw, arm]
                baseline = data['expected_base' if stage == 'base' else 'expected_full'][draw]
                oa, ba = metrics(y[ev], pred[ev])
                oldoa, oldba = metrics(y[ev], baseline[ev])
                records.append(dict(**meta, draw=draw, mode=mode, stage=stage, oa=oa, ba=ba,
                    delta_old_oa=oa-oldoa, delta_old_ba=ba-oldba,
                    losses_vs_actual=int(np.sum(ev & (pred != y) & (reference == y))),
                    rescues_vs_actual=int(np.sum(ev & (pred == y) & (reference != y)))))
                if stage != 'fixed_actual_route':
                    for c in np.unique(y[ev]):
                        mask = ev & (y == c)
                        classes.append(dict(**meta, draw=draw, mode=mode, stage=stage, class_id=int(c),
                            n=int(mask.sum()), recall=100*float(np.mean(pred[mask] == c))))
                    for j, feature in enumerate(('elongation_rank', 'center_excess_rank')):
                        bins = np.digitize(data['conditions'][:, j], [.25, .75])
                        for bin_id, bin_name in enumerate(('low', 'middle', 'high')):
                            mask = ev & (bins == bin_id)
                            if not mask.any():
                                continue
                            ao, ab = metrics(y[mask], pred[mask])
                            bo, bb = metrics(y[mask], baseline[mask])
                            strata.append(dict(**meta, draw=draw, mode=mode, stage=stage,
                                feature=feature, bin=bin_name, n=int(mask.sum()), oa=ao, ba=ab,
                                delta_old_oa=ao-bo, delta_old_ba=ab-bb))
        print(f'INFO EVAL {info["task"]} draw{draw}', flush=True)
    pd.DataFrame(records).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(out/'classes.csv', index=False)
    pd.DataFrame(magnitudes).to_csv(out/'embedding_change.csv', index=False)
    pd.DataFrame(strata).to_csv(out/'strata.csv', index=False)
    (out/'EVALUATED.json').write_text(json.dumps(dict(**meta, exact_replay=True, draws=5,
        neural_retraining=retrained, support_refit=True, target_query_labels_in_fit=False), indent=2))


def report():
    folders = [p.parent for p in sorted(OUT.glob('*/*/EVALUATED.json'))]
    data = pd.concat([pd.read_csv(f/'metrics.csv') for f in folders], ignore_index=True)
    data.to_csv(OUT/'metrics_all.csv', index=False)
    fields = ['oa', 'ba', 'delta_old_oa', 'delta_old_ba']
    task = data.groupby(['dataset', 'task', 'stage', 'mode'])[fields].mean().reset_index()
    task.to_csv(OUT/'task_means.csv', index=False)
    summary = task.groupby(['dataset', 'stage', 'mode'])[fields].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    data.groupby(['dataset', 'stage', 'mode'])[['losses_vs_actual', 'rescues_vs_actual']].sum().to_csv(OUT/'events.csv')
    data.groupby(['dataset', 'draw', 'stage', 'mode'])[fields].mean().to_csv(OUT/'draw_metrics.csv')
    classes = pd.concat([pd.read_csv(f/'classes.csv') for f in folders], ignore_index=True)
    classes.groupby(['dataset', 'task', 'stage', 'mode', 'class_id']).recall.mean().groupby(
        ['dataset', 'stage', 'mode', 'class_id']).mean().to_csv(OUT/'class_means.csv')
    pd.concat([pd.read_csv(f/'embedding_change.csv') for f in folders], ignore_index=True).to_csv(OUT/'embedding_change.csv', index=False)
    strata = pd.concat([pd.read_csv(f/'strata.csv') for f in folders], ignore_index=True)
    strata.to_csv(OUT/'strata_all.csv', index=False)
    group = ['dataset', 'stage', 'mode', 'feature', 'bin']
    strata.groupby(group+['task'])[fields].mean().groupby(group).mean().to_csv(OUT/'strata_summary.csv')
    (OUT/'READOUT.json').write_text(json.dumps(dict(tasks=len(folders), expected=25, complete=len(folders) == 25,
        target_selection=False, independent_confirmation=False, information_removal_not_new_architecture=True,
        source_retrained=OUT.name == 'retrained'), indent=2))
    print(summary[summary.stage.eq('full')].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrained', action='store_true')
    args = parser.parse_args()
    if args.retrained:
        OUT = OUT/'retrained'
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        evaluate(marker.parent, args.retrained)
    report()
