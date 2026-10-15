"""Rebuild held predictions from exported coefficients and raw cached evidence."""

from run_task_pair_relation import BASE, OUT, PARENT, SEEDS, load, make_episode, apply, write_json
import numpy as np
import pandas as pd
from task_pair_relation import ARMS, observation_coordinates


def main():
    meta = pd.read_csv(BASE/'observations.csv')
    e = load(BASE/'condition_evidence_20261011/evidence_features.npz')
    features = np.full((len(meta), 6), np.nan); features[e['obs_ids']] = e['features']
    normalization = load(OUT/'normalization.npz')
    coordinates, _ = observation_coordinates(features, None, normalization)
    angles = meta.incidence_deg.to_numpy(float)
    scores, multipliers = 0, 0
    for seed in SEEDS:
        bundle = load(PARENT/f'full_{seed}_inputs.npz')
        models = {arm: load(OUT/f'{seed}_{arm}_model.npz') for arm in ARMS}
        for draw in range(3):
            saved = load(OUT/f'{seed}_source_{draw}_predictions.npz')
            ep = make_episode(meta, bundle, saved['support_obs_ids'], saved['query_obs_ids'], coordinates,
                              angles, angles, normalization['limits'])
            np.testing.assert_array_equal(saved['baseline'], ep['base']); scores += 1
            np.testing.assert_array_equal(saved['valid'], ep['valid'])
            for arm in ARMS:
                for state in ('selected', 'last'):
                    prediction, factor = apply(ep, arm, models[arm], state)
                    np.testing.assert_array_equal(prediction, saved[f'{arm}_{state}']); scores += 1
                    np.testing.assert_array_equal(factor, saved[f'{arm}_{state}_multiplier']); multipliers += 1
    receipt = dict(score_arrays_exact=scores, multiplier_arrays_exact=multipliers,
                   query_labels_used_in_prediction=False)
    write_json(OUT/'REPLAY.json', receipt)
    print(receipt)


if __name__ == '__main__':
    main()
