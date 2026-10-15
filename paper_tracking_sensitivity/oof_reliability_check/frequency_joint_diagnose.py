"""Matched training attribution and classifier/decision-stage diagnosis."""

import json
import numpy as np
import pandas as pd

from prepare_learned_frequency import ROOT, OUT as EXP, SOURCE, DATASETS, load_npz
from frequency_reliability_run import statistics

OUT = ROOT/'frequency_joint_20261009'


def main():
    if not json.loads((OUT/'STATUS.json').read_text())['complete']:
        raise ValueError('All target outputs must be frozen before diagnostic label reads')
    labels_source = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    records, classes, training, comparisons, source_validation = [], [], [], [], []
    pairs = [('continued_ce', 'learned'), ('joint_kernel', 'learned'),
             ('joint_kernel', 'continued_ce'), ('joint_compare', 'joint_kernel'),
             ('source_selected', 'learned')]
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        folder = marker.parent
        info = json.loads((folder/'PREPARED.json').read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        dst = OUT/'tasks'/info['dataset']/info['task']
        p = load_npz(OUT/'targets'/info['dataset']/info['task']/'policies.npz')
        choice = load_npz(OUT/'targets'/info['dataset']/info['task']/'comparisons.npz')
        modes = list(p['modes'])
        y = labels_source[p['rows']] if info['dataset'] == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        validation = pd.read_csv(OUT/'source'/info['dataset']/info['task']/'validation.csv')
        source_validation.extend([dict(**meta, **r) for r in validation.to_dict('records')])
        old_emb = load_npz(folder/'selective_embedding.npz')['embedding']
        for arm in ('continued_ce', 'joint_kernel'):
            log = json.loads((dst/f'{arm}_training.json').read_text())
            grad = log['initial_gradient']
            emb = load_npz(dst/f'{arm}_embedding.npz')['embedding']
            weights = np.sum(emb.reshape(-1, 4, 64)**2, axis=2).mean(0, dtype=np.float64)
            cosine = np.sum(emb*old_emb, axis=1)
            training.append(dict(**meta, mode=arm, parameter_l2_change=log['parameter_l2_change'],
                final_ce=log['epochs'][-1]['ce'], final_pair=log['epochs'][-1]['pair'],
                final_pair_accuracy=log['epochs'][-1]['pair_accuracy'],
                first_ce_grad=grad['ce_embedding_grad'], first_pair_grad=grad['pair_embedding_grad'],
                weighted_pair_to_ce_grad=.1*grad['pair_embedding_grad']/max(grad['ce_embedding_grad'], 1e-12),
                first_gradient_cosine=grad['gradient_cosine'],
                cosine_to_original=float(cosine.mean(dtype=np.float64)),
                **{f'gate_{band}': float(w) for band, w in zip(('LL', 'LH', 'HL', 'HH'), weights)}))
        for draw, ev in enumerate(p['eval_masks']):
            for stage, key in [('base', 'base_predictions'), ('frozen_route', 'frozen_predictions'), ('full', 'predictions')]:
                for arm, reference in pairs:
                    pred, ref = p[key][draw, modes.index(arm)], p[key][draw, modes.index(reference)]
                    oa, ba, f1 = statistics(y[ev], pred[ev])
                    ro, rb, rf = statistics(y[ev], ref[ev])
                    records.append(dict(**meta, draw=draw, stage=stage, mode=arm, reference=reference,
                        delta_oa=oa-ro, delta_ba=ba-rb, delta_f1=f1-rf,
                        rescued=int(np.sum(ev & (pred == y) & (ref != y))),
                        harmed=int(np.sum(ev & (pred != y) & (ref == y)))))
                    for cl in np.unique(y[ev]):
                        mask = ev & (y == cl)
                        classes.append(dict(**meta, draw=draw, stage=stage, mode=arm, reference=reference,
                            class_id=int(cl), n=int(mask.sum()),
                            delta_recall=100*float(np.mean(pred[mask] == cl)-np.mean(ref[mask] == cl))))
            rows = choice[f'rows_{draw}']
            reject = choice[f'reject_{draw}']
            old = p['base_predictions'][draw, modes.index('learned'), rows]
            new = p['base_predictions'][draw, modes.index('joint_kernel'), rows]
            utility = (new == y[rows]).astype(float)-(old == y[rows])
            comparisons.append(dict(**meta, draw=draw, conflicts=len(rows), reverted=int(reject.sum()),
                new_rescues=int(np.sum(utility > 0)), new_harms=int(np.sum(utility < 0)),
                lost_rescues=int(np.sum(reject & (utility > 0))), repaired_harms=int(np.sum(reject & (utility < 0)))))
    m, tr = pd.DataFrame(records), pd.DataFrame(training)
    m.to_csv(OUT/'matched_metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'matched_classes.csv', index=False)
    tr.to_csv(OUT/'training_diagnostics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'comparison_events.csv', index=False)
    pd.DataFrame(source_validation).to_csv(OUT/'source_validation_all.csv', index=False)
    means = m.groupby(['dataset', 'task', 'mode', 'reference', 'stage'])[['delta_oa', 'delta_ba', 'delta_f1']].mean().groupby(
        ['dataset', 'mode', 'reference', 'stage']).mean()
    means.to_csv(OUT/'matched_summary.csv')
    print('MATCHED EFFECTS\n'+means.to_string())
    print('TRAINING\n'+tr.groupby(['dataset', 'mode'])[['final_ce', 'final_pair', 'weighted_pair_to_ce_grad',
        'first_gradient_cosine', 'cosine_to_original']].mean().to_string())
    print('COMPARE EVENTS\n'+pd.DataFrame(comparisons).groupby('dataset').sum(numeric_only=True).to_string())


if __name__ == '__main__':
    main()
