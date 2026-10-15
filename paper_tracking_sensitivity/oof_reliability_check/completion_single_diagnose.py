"""Fixed prior/kernel crossover after the negative source-finetuning pilot."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from unified_head_core import score

ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
PARENT = ROOT/'completion_single_finetune_20261010'
OUT = PARENT/'prior_kernel_diagnosis'


def main():
    OUT.mkdir(exist_ok=True)
    with np.load(PARENT/'policies.npz') as saved:
        data = {k: saved[k] for k in saved.files}
    views = []
    for name, path in [('frozen', ROOT/'cross_dataset_c4_20261009/oss2/pooled_c4.npy'),
                       ('finetuned', PARENT/'oss2_c4.npy')]:
        with (PARENT/f'{name}_single_core_head.pkl').open('rb') as stream:
            fitted = pickle.load(stream)
        x = (fitted['pca'].transform(np.load(path)).astype(float)-fitted['center'])/fitted['scale']
        views.append(dict(prior=fitted['head'].decision_function(x),
            phi=np.c_[x/np.sqrt(128), np.ones(len(x))],
            mean=x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)))
    names = ('frozen', 'finetuned')
    arms = [f'prior_{p}__kernel_{k}' for p in names for k in names]
    protocol = dict(trigger='Fine-tuned same64 single-core BA below matched frozen arm',
        intervention='Cross source-prior scores and source-fitted target kernels, no new fitting or tuning',
        weights='Original eta0.25, residual lambda8, original5x64 supports',
        interpretation='Computational-path attribution on development data, not physical causality',
        selection='None; retain original negative experiment and all four diagnostic arms')
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    predictions = np.empty((5, 4, len(data['rows'])), np.int8)
    for draw, support in enumerate(data['support']):
        condition = np.exp(-np.sum((data['conditions'][:, None]-data['conditions'][support][None])**2, axis=2)/.5)
        kernels = [v['phi']@v['phi'][support].T+.25*(v['mean']@v['mean'][support].T)*condition for v in views]
        for pi, p in enumerate(views):
            for ki, kernel in enumerate(kernels):
                values, _ = score(kernel, p['prior'], support, data['support_labels'][draw])
                predictions[draw, 2*pi+ki] = values.argmax(1)
        np.testing.assert_array_equal(predictions[draw, 0], data['predictions'][draw, 0])
        np.testing.assert_array_equal(predictions[draw, 3], data['predictions'][draw, 1])
    np.savez_compressed(OUT/'policies.npz', predictions=predictions, arms=np.array(arms),
        rows=data['rows'], ids=data['ids'], support=data['support'], eval_masks=data['eval_masks'])
    frame = pd.read_csv(ROOT/'opensarship2_transfer_20261007/target_manifest.csv', usecols=['sample_id', 'class_id'])
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str), data['ids'])
    metrics, classes = [], []
    for draw, ev in enumerate(data['eval_masks']):
        y = frame.class_id.to_numpy()[ev]
        for ai, arm in enumerate(arms):
            pred = predictions[draw, ai, ev]
            recalls = [100*np.mean(pred[y == c] == c) for c in np.unique(y)]
            metrics.append(dict(draw=draw, arm=arm, oa=100*np.mean(pred == y), ba=np.mean(recalls)))
            for c, recall in zip(np.unique(y), recalls):
                classes.append(dict(draw=draw, arm=arm, class_id=int(c), recall=recall))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT/'metrics.csv', index=False)
    summary = metrics.groupby('arm')[['oa', 'ba']].mean()
    summary.to_csv(OUT/'summary.csv')
    pd.DataFrame(classes).to_csv(OUT/'class_recalls.csv', index=False)
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(arms=4, draws=5, both_diagonals_exact=True,
        neural_retraining=False, target_model_selection=False), indent=2), encoding='utf-8')
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
