"""Replay exported held-source forecasts without fitting or reading outcome labels."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from condition_evidence_response import ARMS, predict_model

BASE = Path(__file__).resolve().parent
OUT = BASE/'condition_evidence_20261011'


def main():
    meta = pd.read_csv(BASE/'observations.csv', usecols=['incidence_deg'])
    angle = meta.incidence_deg.to_numpy(float)
    with np.load(OUT/'evidence_features.npz') as f:
        names = list(f['feature_names'])
        features = np.full((len(meta), len(names)), np.nan)
        features[f['obs_ids']] = f['features']
    total = np.full(len(meta), np.nan)
    known = np.isfinite(features).all(1)
    total[known] = np.logaddexp.reduce(features[known, :3], axis=1)
    count, max_error = 0, 0.
    for path in sorted((OUT/'predictions').glob('*.npz')):
        seed, rest = path.stem.split('_', 1)
        with np.load(path) as saved:
            directed = saved['directed']
            a, b = directed.T
            for arm in ARMS:
                with np.load(OUT/'models'/f'{path.stem}_{arm}.npz') as f:
                    model = {k: f[k] for k in f.files}
                mode = str(model['mode'])
                name = rest.removesuffix('_'+mode)
                if seed == '0':
                    anchor, context = features[a, names.index(name)], total[a]
                else:
                    with np.load(BASE/'incidence_transport_20261011'/f'full_{seed}_inputs.npz') as f:
                        score = f['prior'][a].astype(np.float64)
                    ordered = np.sort(score, axis=1)
                    gap = ordered[:, -1]-ordered[:, -2]
                    if name.startswith('class_'):
                        parts = name.split('_')
                        anchor, context = score[:, int(parts[1])]-score[:, int(parts[3])], gap
                    else:
                        anchor, context = gap, ordered[:, -1]
                prediction = predict_model(anchor, context, angle[a], angle[b], model)
                error = float(np.max(np.abs(prediction-saved[arm])))
                np.testing.assert_allclose(prediction, saved[arm], rtol=0, atol=1e-12,
                                           err_msg=f'{path.name}: {arm}')
                max_error = max(max_error, error)
                count += 1
    receipt = dict(replayed=count, maximum_absolute_error=max_error,
                   outcome_labels_read=False, fitting=False, target_port_evaluation=False)
    (OUT/'REPLAY.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
