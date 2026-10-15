"""Diagnostic-only fixed 2x2 of stable source prior and stable support kernel."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from external_knowledge_run import (ROOT, OUT, TARGETS, TASKS, load, unpickle,
                                    coordinates, inputs, maps, READOUTS)
from external_knowledge_metadata import write
from unified_head_expert import ExpertContext
from unified_head_core import score, combined, late
from unified_head_run import base_kernel

ARMS = ('matched_00', 'stable_prior_only_10', 'stable_kernel_only_01', 'stable_both_11')


def freeze(dataset):
    destination = OUT/'diagnostic_prior_kernel'/dataset
    if (destination/'FROZEN.json').exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    folder = TASKS/dataset
    fit = unpickle(OUT/dataset/'source_fit.pkl')
    geo, heads = fit['geometry'], fit['heads']
    info = json.loads((folder/'PREPARED.json').read_text())
    data = load(folder/'target.npz', ['rows', 'ids', 'support', 'support_labels', 'eval_masks',
                                    'phi', 'mean', 'conditions', 'prior'])
    embedding = load(folder/'selective_embedding.npz', ['rows', 'embedding'])
    np.testing.assert_array_equal(embedding['rows'], data['rows'])
    z = embedding['embedding'].astype(float)
    context = ExpertContext(folder)
    model_b = unpickle(Path(info['model_path']))
    model_e, stats = context.model, context.stats
    target = ROOT/TARGETS[dataset]
    raw = (np.load(target/'paired_r50.npy', mmap_mode='r') if dataset == 'f0921'
           else load(target/'features.npz', ['paired_r50'])['paired_r50'])
    extra = np.load(context.joint/'target_extra.npy', mmap_mode='r')
    c4 = np.load(ROOT/'cross_dataset_c4_20261009'/dataset/'pooled_c4.npy', mmap_mode='r')
    cb, ce = coordinates(c4, model_b, geo), coordinates(raw, model_e, geo, extra)
    priors = {}
    zero = np.zeros((len(raw), 2))
    for arm in ('matched_refit', 'stable_only'):
        priors[arm] = []
        for head, coords, model in [('base', cb, model_b), ('expert', ce, model_e)]:
            h = heads[arm][head]
            values = inputs(coords, arm, maps(model, stats, geo)[0], zero, head, model)
            priors[arm].append(values@h['coef']+h['intercept'])
    old = load(OUT/dataset/'policies.npz')
    predictions = np.empty((5, 4, 4, len(raw)), np.int8)
    score_shift = []
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        kb0 = base_kernel(data, z, support, info['eta'])
        _, ke0 = context.target_components(data, support)
        sb = cb['s']@cb['s'][support].T/cb['s'].shape[1]
        se = .25*ce['s']@ce['s'][support].T/ce['s'].shape[1]
        original_scores = None
        for ai, arm in enumerate(ARMS):
            new_prior, new_kernel = ai in (1, 3), ai in (2, 3)
            pb, pe = priors['stable_only' if new_prior else 'matched_refit']
            kb, ke = (kb0+sb, ke0+se) if new_kernel else (kb0, ke0)
            b, e = score(kb, pb, support, sy)[0], score(ke, pe, support, sy)[0]
            j = combined(kb, ke, pb, pe, support, sy)[0]
            results = b, e, late(b, e), j
            if ai == 0:
                original_scores = results
            for ri, values in enumerate(results):
                predictions[draw, ai, ri] = values.argmax(1)
                shift = values[ev]-original_scores[ri][ev]
                score_shift.append(dict(dataset=dataset, draw=draw, arm=arm, readout=READOUTS[ri],
                                        score_rms=float(np.sqrt(np.mean(shift**2)))))
            if ai in (0, 3):
                slot = list(old['arms']).index('matched_refit' if ai == 0 else 'stable_only')
                np.testing.assert_array_equal(predictions[draw, ai], old['predictions'][draw, slot])
        print(f'DIAGNOSTIC {dataset} draw{draw}: both factorial endpoints exact', flush=True)
    np.savez_compressed(destination/'policies.npz', predictions=predictions, arms=np.array(ARMS),
        readouts=np.array(READOUTS), ids=data['ids'], support=data['support'],
        support_labels=data['support_labels'], eval_masks=data['eval_masks'])
    pd.DataFrame(score_shift).to_csv(destination/'score_shift.csv', index=False)
    write(destination/'FROZEN.json', dict(dataset=dataset, diagnostic_only=True,
        full_and_matched_endpoints_exact=True, target_query_labels_used=False,
        parameters_changed=False, shared_supports=True))


def report():
    if not all((OUT/'diagnostic_prior_kernel'/t/'FROZEN.json').exists() for t in TARGETS):
        raise ValueError('Complete both diagnostic freezes before readout')
    rows, classes = [], []
    for dataset, directory in TARGETS.items():
        p = load(OUT/'diagnostic_prior_kernel'/dataset/'policies.npz')
        f = pd.read_csv(ROOT/directory/'target_manifest.csv', usecols=['sample_id', 'class_id'])
        np.testing.assert_array_equal(f.sample_id.to_numpy(str), p['ids'])
        truth = f.class_id.to_numpy()
        for draw, ev in enumerate(p['eval_masks']):
            y = truth[ev]
            for ai, arm in enumerate(ARMS):
                for ri, readout in enumerate(READOUTS):
                    pred = p['predictions'][draw, ai, ri, ev]
                    baseline = p['predictions'][draw, 0, ri, ev]
                    rows.append(dict(dataset=dataset, draw=draw, arm=arm, readout=readout,
                        oa=100*np.mean(pred == y), ba=np.mean([100*np.mean(pred[y == c] == c) for c in np.unique(y)]),
                        rescues=int(np.sum((pred == y) & (baseline != y))), harms=int(np.sum((pred != y) & (baseline == y)))))
                    for c in np.unique(y):
                        classes.append(dict(dataset=dataset, draw=draw, arm=arm, readout=readout,
                            class_id=int(c), recall=100*np.mean(pred[y == c] == c), n=int((y == c).sum())))
    m = pd.DataFrame(rows)
    dest = OUT/'diagnostic_prior_kernel'
    m.to_csv(dest/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(dest/'class_metrics.csv', index=False)
    summary = m.groupby(['dataset', 'arm', 'readout'])[['oa', 'ba', 'rescues', 'harms']].mean().reset_index()
    summary.to_csv(dest/'summary.csv', index=False)
    write(dest/'COMPLETE.json', dict(diagnostic_only=True, no_parameter_search=True,
        contrasts='Prior-only vs matched; kernel-only vs matched; interaction=11-10-01+00',
        endpoints_exact=True, all_predictions_frozen_before_readout=True))
    print(summary[summary.readout.isin(['fusion', 'unified'])].to_string(index=False), flush=True)


def main():
    with threadpool_limits(limits=2):
        for dataset in TARGETS:
            freeze(dataset)
            gc.collect()
        report()


if __name__ == '__main__':
    main()
