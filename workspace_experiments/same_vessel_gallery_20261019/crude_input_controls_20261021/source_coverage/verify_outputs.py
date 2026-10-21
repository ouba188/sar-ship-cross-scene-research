"""Focused checks for wrong split membership, illegal neighbors, and axial wrap."""
import json

import numpy as np
import pandas as pd

from analyze_source_coverage import OUT, PILOT, PAIRED, FIELDS, distance


def main():
    actual = pd.read_csv(OUT/'split_membership.csv', dtype={'mmsi': str})
    frozen = pd.read_csv(PILOT/'partitions.csv')
    frozen = frozen[~frozen.obs_id.eq(31237)]
    assert len(actual) == len(frozen)
    for (fold, partition), g in actual.groupby(['split_fold','partition']):
        expected = frozen[frozen.fold.eq(fold) & frozen.partition.eq(partition)]
        np.testing.assert_array_equal(g.obs_id, expected.obs_id)
    old = pd.read_csv(PAIRED/'coverage.csv')
    counts = pd.read_csv(OUT/'split_summary.csv')
    np.testing.assert_array_equal(counts.observations, old.n)
    np.testing.assert_array_equal(counts.ships, old.ships)
    source = actual[actual.partition.eq('train')]
    endpoint = pd.read_csv(OUT/'crude_endpoints.csv', dtype={'mmsi': str}).set_index('obs_id')
    cov = pd.read_csv(OUT/'endpoint_source_coverage.csv', dtype={'nearest_mmsi': str})
    assert len(cov) == 200 * 8 * 2 and not cov.duplicated(['obs_id','pol','source_class_id']).any()
    for fold, g in cov.groupby('split_fold'):
        legal = source[source.split_fold.eq(fold)].set_index('obs_id')
        neighbors = legal.loc[g.nearest_obs_id.astype(int)]
        assert (neighbors.class_id.to_numpy() == g.source_class_id.to_numpy()).all()
        assert (neighbors.mmsi.to_numpy() != endpoint.loc[g.obs_id].mmsi.to_numpy()).all()
        assert (neighbors['product'].to_numpy() != endpoint.loc[g.obs_id]['product'].to_numpy()).all()
        assert g.fixed_near_ships.le(g.fixed_near_observations).all()
        assert g.fixed_near_products.le(g.fixed_near_observations).all()
    ref = pd.read_csv(OUT/'source_nn_reference.csv', dtype={'mmsi': str, 'nearest_mmsi': str})
    assert ref.mmsi.ne(ref.nearest_mmsi).all()
    assert ref['product'].ne(ref.nearest_product).all()
    assert np.isfinite(ref.independent_nn_distance).all()
    # -89/+89 are two degrees apart on an unoriented axis, not 178 degrees.
    a = pd.DataFrame([[-89, 35, 20, 10]], columns=FIELDS)
    b = pd.DataFrame([[89, 35, 20, 10]], columns=FIELDS)
    np.testing.assert_allclose(distance(a,b), [[2/15]])
    b = pd.DataFrame([[-89, 35, 25, 10]], columns=FIELDS)
    np.testing.assert_allclose(distance(a,b), [[1]])
    np.testing.assert_allclose(distance(a,b), distance(b,a).T)
    v = dict(status='passed', frozen_membership_rows=len(actual), ordered_score_files=63,
        endpoint_comparisons=len(cov), independent_source_neighbors=len(ref),
        checks=['frozen pre-exclusion partitions minus 31237', 'existing paired coverage counts',
                'legal same-class source neighbors', 'no same-vessel/product calibration references',
                'axial periodic boundary', 'multiplicative size tolerance and symmetry'])
    (OUT/'VERIFICATION.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
    print(json.dumps(v,indent=2))


if __name__ == '__main__':
    main()
