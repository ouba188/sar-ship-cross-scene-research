"""Paired increment versus frozen calibration, plus native case panels."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'source_utility_alignment/matched_controls'))
import analyze_controls as review


def main():
    p = pd.read_csv(HERE/'predictions.csv', dtype={'mmsi': str})
    table = pd.read_csv(review.parent.EXP/'readout/targets.csv', dtype={'mmsi': str})
    pairs, intervals, panels = [], [], []
    for partition, part in p.groupby('partition'):
        for seed in [17, 29, 43]:
            control = part[part.arm.eq('score_only') & part.seed.eq(seed)].set_index('instance_id').sort_index()
            for arm in ['signed_only', 'conditional_std', 'score_only']:
                g = part[part.arm.eq(arm) & part.seed.eq(seed)].set_index('instance_id').loc[control.index]
                y = g.class_id.to_numpy()
                np.testing.assert_array_equal(y, control.class_id)
                np.testing.assert_allclose(g[['b0','b1','b2']], control[['b0','b1','b2']], atol=1e-12, rtol=0)
                left = g.prediction.to_numpy() == y
                refs = ['B'] if arm == 'score_only' else ['B', 'extra_score']
                for name in refs:
                    right = (g.base_prediction if name == 'B' else control.prediction).to_numpy() == y
                    diff = left.astype(int)-right.astype(int)
                    meta = dict(partition=partition, arm=arm, seed=seed, reference=name)
                    pairs.append(dict(**meta, only_left=int((left & ~right).sum()), only_right=int((~left & right).sum()),
                        net=int(diff.sum()), delta_OA=100*diff.mean(), delta_BA=100*np.mean([diff[y == c].mean() for c in range(3)])))
                    for block in ['pass_group', 'mmsi']:
                        intervals.append(dict(**meta, block=block, **review.block_intervals(y, diff, g[block])))
        folder = HERE/partition
        folder.mkdir(exist_ok=True)
        review.HERE = folder
        review.ARMS = ['signed_only', 'conditional_std', 'score_only']
        paths, missing = review.render(table, part, baseline_label='B (frozen calibration)',
            subtitle='Primary J; seed17. All heads add to frozen B. Green=recorded OBB; native pixels, nearest display.')
        panels.extend(paths)
        (folder/'review_summary.json').write_text(json.dumps(dict(panels=paths, missing=missing), indent=2), encoding='utf8')
    pd.DataFrame(pairs).to_csv(HERE/'paired_effects.csv', index=False)
    pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv', index=False)
    pd.read_csv(HERE/'pooled.csv').groupby(['partition','arm'])[['OA','BA','net','delta_OA','delta_BA','NLL','Brier']].mean().to_csv(HERE/'arm_means.csv')
    sel = pd.read_csv(HERE/'selections.csv')
    sel.groupby(['partition','arm']).agg(heads=('epoch','size'), epoch0=('epoch',lambda x:int((x==0).sum())),
        epoch_median=('epoch','median'), epoch_max=('epoch','max')).to_csv(HERE/'selection_summary.csv')
    rebuilt = p[['b0','b1','b2']].to_numpy()+p[['delta0','delta1','delta2']].to_numpy()
    error = float(abs(rebuilt-p[['s0','s1','s2']].to_numpy()).max())
    assert error < 1e-12 and np.array_equal(rebuilt.argmax(1), p.prediction)
    (HERE/'analysis_summary.json').write_text(json.dumps(dict(panels=panels, score_reconstruction_error=error,
        intervals='fixed saved models, separate pass/MMSI sampling; excludes model-selection uncertainty'), indent=2), encoding='utf8')
    print(pd.read_csv(HERE/'arm_means.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
