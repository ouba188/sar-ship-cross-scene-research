"""Replay persisted predictions and check the matched experimental population."""
import json
import numpy as np
import pandas as pd
import input_controls as study


def main():
    root = study.ROOT
    frame = pd.read_csv(study.prior.ROOT/'cohort.csv', dtype={'mmsi': str})
    cache = np.load(root/'features.npz')
    partitions = pd.read_csv(root/'partitions.csv')
    source = pd.read_csv(root/'source_coverage/split_membership.csv')
    source = source.drop(columns='fold').rename(columns={'split_fold': 'fold'})
    keys = ['fold', 'partition', 'obs_id']
    pd.testing.assert_frame_equal(partitions[keys].sort_values(keys).reset_index(drop=True),
                                  source[keys].sort_values(keys).reset_index(drop=True))
    predictions = pd.read_csv(root/'predictions.csv')
    maximum = 0.
    for arm in study.ARMS:
        bundle = np.load(root/f'scores_{arm}.npz')
        rows, scores = bundle['rows'], bundle['scores']
        assert len(rows) == 4744 and len(set(rows)) == 4744
        recomputed = np.zeros_like(scores)
        for fold in range(3):
            select = frame.iloc[rows].fold.to_numpy() == fold
            head = np.load(root/f'head_f{fold}_{arm}.npz')
            raw = cache[arm][rows[select]]
            recomputed[select] = ((raw-head['mean'])/head['std']/np.sqrt(raw.shape[1])) @ head['coef'].T + head['intercept']
        error = float(np.max(np.abs(scores-recomputed)))
        assert error < 1e-6
        maximum = max(maximum, error)
        p = predictions[predictions.arm.eq(arm)].set_index('obs_id').loc[frame.iloc[rows].obs_id]
        assert np.array_equal(p.prediction, scores.argmax(1))
        recorded = pd.read_csv(root/'summary.csv').set_index('arm').loc[arm]
        for metric, value in study.metrics(frame.iloc[rows].class_id.to_numpy(), scores).items():
            assert abs(recorded[metric]-value) < 1e-10
    coverage = pd.read_csv(root/'input_coverage.csv')
    assert (coverage.native_target_pixels == coverage.retained_target_pixels).all()
    assert (coverage.input_scale == 1).all()
    checks = pd.read_csv(root/'input_checks.csv')
    assert checks.loc[checks.test.str.startswith('outside_change'), 'max_abs'].eq(0).all()
    perturb = pd.read_csv(root/'perturbations.csv')
    assert len(perturb) == 1800 and perturb.obs_id.nunique() == 200
    baseline = perturb[perturb.region.eq('baseline')].set_index('obs_id')
    replay = predictions[predictions.arm.eq('full_global')].set_index('obs_id').loc[baseline.index]
    baseline_error = float(np.max(np.abs(baseline.margin-replay.margin)))
    assert baseline_error < 1e-5
    pair_results = pd.read_csv(root/'crude_pairs.csv')
    assert pair_results.groupby('arm').size().eq(100).all()
    result = dict(status='passed', matched_split_rows=len(partitions), test_rows=4744,
        heads_replayed=15, score_max_abs=maximum, input_scale=1,
        target_pixels_retained=True, body_input_outside_value_invariance=True,
        perturb_records=1800, perturb_baseline_max_abs=baseline_error,
        scope='Protocol, persistence and numerical checks; not evidence of a physical cause.')
    (root/'VERIFICATION.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
