"""Source-only matched-objective control for the visual comparator."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from frequency_candidate_run import OUT, CANDIDATES, WIDTHS
from prepare_learned_frequency import load_npz


def fit(features, utility, weights):
    scale = np.maximum(np.sqrt(np.average(features**2, axis=0, weights=weights)), 1e-6)
    model = Ridge(alpha=10., fit_intercept=False).fit(features/scale, utility,
        sample_weight=weights/weights.mean())
    return model, scale


def main():
    records = []
    for marker in sorted((OUT/'source').glob('*/*/SELECTION.json')):
        info = json.loads(marker.read_text())
        d = load_npz(marker.parent/'examples.npz')
        u = (d['new'].argmax(1) == d['labels']).astype(float)-(d['old'].argmax(1) == d['labels'])
        ep, eligible = d['episode'], d['eligible']
        for mode, width in zip(CANDIDATES, WIDTHS):
            value = np.zeros(len(u))
            for held in np.unique(ep):
                train, valid = (ep != held) & eligible, ep == held
                if train.any():
                    model, scale = fit(d['candidate'][train, :width], u[train], d['weights'][train])
                    value[valid] = model.predict(d['candidate'][valid, :width]/scale)
            reject, event = eligible & (value < 0), eligible & (u != 0)
            delta = -100*float(np.sum(d['weights'][reject]*u[reject]))/info['source_ports']
            records.append(dict(dataset=info['dataset'], task=info['task'], mode=mode,
                source_pairs=int(eligible.sum()), delta_ba=delta,
                previous_pair_delta_ba=info['source_delta_ba'][mode],
                delta_over_previous=delta-info['source_delta_ba'][mode],
                lost=int(np.sum(reject & (u > 0))), repaired=int(np.sum(reject & (u < 0))),
                auc=float(roc_auc_score(u[event] > 0, value[event])) if len(np.unique(u[event])) == 2 else np.nan))
    result = pd.DataFrame(records)
    result.to_csv(OUT/'objective_control_source.csv', index=False)
    print(result.groupby(['dataset', 'mode'])[['auc', 'delta_ba', 'previous_pair_delta_ba', 'delta_over_previous']].mean().to_string())


if __name__ == '__main__':
    main()
