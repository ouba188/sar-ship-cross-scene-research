"""Source-only fixed-capacity diagnostic, with no target prediction or selection."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from frequency_decision_run import OUT, CANDIDATES
from frequency_decision_diagnose import action_record
from prepare_learned_frequency import load_npz


def main():
    records = []
    for marker in sorted((OUT/'source').glob('*/*/SELECTION.json')):
        info = json.loads(marker.read_text())
        d = load_npz(marker.parent/'examples.npz')
        a, b = d['old'].argmax(1), d['new'].argmax(1)
        conflict = a != b
        utility = (b == d['labels']).astype(float)-(a == d['labels'])
        for mode in CANDIDATES:
            f = d['features'][:, :info['decision_width']] if mode == 'decision_only' else d['features']
            benefit = np.zeros(len(utility))
            for held in np.unique(d['episode']):
                train, valid = (d['episode'] != held) & conflict, d['episode'] == held
                w = d['weights'][train]/d['weights'][train].mean()
                scaler = StandardScaler().fit(f[train], sample_weight=w)
                model = Ridge(alpha=10.).fit(scaler.transform(f[train]), utility[train], sample_weight=w)
                benefit[valid] = model.predict(scaler.transform(f[valid]))
            records.append(dict(dataset=info['dataset'], task=info['task'], mode=mode,
                **action_record(utility[conflict], benefit[conflict],
                    d['weights'][conflict]/info['source_ports'])))
    result = pd.DataFrame(records)
    result.to_csv(OUT/'diagnosis_linear_source.csv', index=False)
    print(result.groupby(['dataset', 'mode'])[['auc', 'balanced_auc', 'delta_ba']].mean().to_string())


if __name__ == '__main__':
    main()
