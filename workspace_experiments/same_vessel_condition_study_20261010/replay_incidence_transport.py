"""Check metadata-driven correction exported from raw frozen features."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
import torch
from train_explicit_candidate import load_data, SEEDS, save_json
from run_incidence_transport import OUT
from replay_crossfit_evidence import exported_prediction
from incidence_transport import ARMS, canonical_delta
from paired_evidence_factors import support_operator, adapted_scores


def main():
    torch.set_num_threads(1)
    meta, split, roles, source, raw, norm, fixed, h = load_data()
    ids = split['test_source']; mm = meta.iloc[ids].reset_index(drop=True)
    labels = meta.class_id.to_numpy(); positions = {int(v): j for j, v in enumerate(ids)}
    records = []
    for seed in SEEDS:
        base = exported_prediction(raw[ids], seed)
        with np.load(OUT/f'full_{seed}_inputs.npz') as f:
            mapping = f['whitened_to_scores'].copy()
            old_prior, old_kernel = f['prior'][ids], f['kernel'][ids]
        for arm in ARMS:
            with np.load(OUT/f'full_{seed}_{arm}.npz') as f:
                model = {k: f[k] for k in f.files}
            dz, valid = canonical_delta(base['features']['class_only'], mm, model)
            with np.load(OUT/f'canonical_{seed}_{arm}.npz') as f:
                np.testing.assert_allclose(dz, f['delta_z'][ids], atol=1e-5, rtol=1e-5)
                np.testing.assert_array_equal(valid, f['valid'][ids])
                reference_dz, selected = f['delta_z'][ids], float(f['gain'])
            for gain in sorted(set((selected, 1.))):
                prior = base['prior']+gain*dz@mapping
                reference_prior = old_prior+gain*reference_dz@mapping
                np.testing.assert_array_equal(prior[~valid], base['prior'][~valid])
                for draw in range(3):
                    with np.load(OUT/f'{seed}_source_{draw}_predictions.npz') as f:
                        s, q = [np.array([positions[int(v)] for v in f[key]]) for key in ('support_obs_ids', 'query_obs_ids')]
                        sy = labels[f['support_obs_ids']]
                        op, present = support_operator(base['kernel'][s], base['kernel'][q], sy)
                        score = adapted_scores(prior[s], prior[q], op, present, sy)
                        if gain == selected:
                            reference = f[arm]
                        else:
                            old_op, old_present = support_operator(old_kernel[s], old_kernel[q], sy)
                            reference = adapted_scores(reference_prior[s], reference_prior[q], old_op, old_present, sy)
                        np.testing.assert_allclose(score, reference, atol=1e-5, rtol=1e-5)
                        np.testing.assert_array_equal(score.argmax(1), reference.argmax(1))
                        records.append(dict(seed=seed, arm=arm, draw=draw, gain=gain, selected=gain == selected,
                            max_score_error=float(np.abs(score-reference).max()), changed_predictions=0))
    pd.DataFrame(records).to_csv(OUT/'deployment_replay.csv', index=False)
    save_json(OUT/'DEPLOYMENT_VERIFIED.json', dict(cases=len(records), raw_features=True,
        unknown_metadata_prior_unchanged=True, predictions_identical=True,
        max_score_error=max(r['max_score_error'] for r in records)))
    print(pd.DataFrame(records).groupby('arm').max_score_error.max().to_string(), flush=True)


if __name__ == '__main__':
    main()
