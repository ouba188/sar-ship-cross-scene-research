"""Paired old-partition accounting and native-image review without new choices."""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'source_utility_alignment/matched_controls'))
import analyze_controls as review


def main():
    p = pd.read_csv(HERE/'predictions.csv', dtype={'mmsi': str})
    primary = p[p.selector.eq('J')].copy()
    rows, pairs, intervals = [], [], []
    for (arm, seed), g in primary.groupby(['arm', 'seed']):
        current = review.assessment(g.class_id.to_numpy(), g[['s0', 's1', 's2']].to_numpy())
        base = review.assessment(g.class_id.to_numpy(), g[['b0', 'b1', 'b2']].to_numpy())
        rows.append(dict(arm=arm, seed=seed, **current, net=int(g.net.sum()),
                         delta_OA=current['OA']-base['OA'], delta_BA=current['BA']-base['BA']))
    for seed in [17, 29, 43]:
        a = primary[primary.arm.eq('signed_only') & primary.seed.eq(seed)].set_index('instance_id').sort_index()
        for arm in ['A', 'score_only', 'conditional_std', 'linear_distribution']:
            if arm == 'A':
                right = a.base_prediction.to_numpy()
            else:
                rs = -1 if arm == 'linear_distribution' else seed
                b = primary[primary.arm.eq(arm) & primary.seed.eq(rs)].set_index('instance_id').loc[a.index]
                np.testing.assert_array_equal(a.class_id, b.class_id)
                np.testing.assert_allclose(a[['b0', 'b1', 'b2']], b[['b0', 'b1', 'b2']], atol=1e-12, rtol=0)
                right = b.prediction.to_numpy()
            y = a.class_id.to_numpy()
            lc, rc = a.prediction.to_numpy() == y, right == y
            diff = lc.astype(int)-rc.astype(int)
            meta = dict(seed=seed, reference=arm)
            pairs.append(dict(**meta, candidate_only_correct=int((lc & ~rc).sum()),
                              reference_only_correct=int((~lc & rc).sum()), net=int(diff.sum()),
                              delta_OA=100*diff.mean(), delta_BA=100*np.mean([diff[y == c].mean() for c in range(3)])))
            for block in ['mmsi', 'pass_group']:
                intervals.append(dict(**meta, block=block, **review.block_intervals(y, diff, a[block])))
    pd.DataFrame(rows).to_csv(HERE/'primary_metrics.csv', index=False)
    pd.DataFrame(rows).groupby('arm').mean(numeric_only=True).to_csv(HERE/'arm_means.csv')
    pd.DataFrame(pairs).to_csv(HERE/'candidate_comparison.csv', index=False)
    pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv', index=False)
    review.HERE = HERE
    review.ARMS = ['signed_only', 'score_only', 'conditional_std', 'linear_distribution']
    table = pd.read_csv(review.parent.EXP/'readout/targets.csv', dtype={'mmsi': str})
    panels, missing = review.render(table, primary)
    score = p[['b0', 'b1', 'b2']].to_numpy()+p[['delta0', 'delta1', 'delta2']].to_numpy()
    assert np.array_equal(score.argmax(1), p.prediction)
    error = float(abs(score-p[['s0', 's1', 's2']].to_numpy()).max())
    assert error < 1e-12
    (HERE/'analysis_summary.json').write_text(json.dumps(dict(panels=panels, missing_outcomes=missing,
        score_reconstruction_error=error, intervals='conditional on trained models, not selection uncertainty'), indent=2), encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
