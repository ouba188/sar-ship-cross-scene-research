"""Compare saved constant-moment heads with frozen B and parent increments."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT.parent/'source_utility_alignment/matched_controls'))
import analyze_controls as review

ARMS = ['constant_signed', 'signed_only', 'score_only']
COMPARISONS = [('constant_signed', 'B'), ('signed_only', 'constant_signed'),
               ('constant_signed', 'score_only')]
METRICS = ['OA', 'BA', 'J', 'NLL', 'balancedNLL', 'Brier']


def main():
    metrics, pairs, classes, intervals, samples, panels, fold_metrics = [], [], [], [], [], [], []
    means_for_intervals = {}
    table = pd.read_csv(review.parent.EXP/'readout/targets.csv', dtype={'mmsi': str})
    for role, filename in [('validation', 'validation_predictions.csv'), ('test', 'predictions.csv')]:
        current = pd.read_csv(HERE/filename, dtype={'mmsi': str})
        if role == 'test':
            parent = pd.read_csv(PARENT/filename, dtype={'mmsi': str})
            parent = parent[parent.arm.isin(ARMS[1:])]
            data = pd.concat([current, parent], ignore_index=True)
        else:
            data = current
        for (partition, fold, seed), g in current.groupby(['partition', 'fold', 'seed']):
            y = g.class_id.to_numpy()
            baseline = review.assessment(y, g[['b0', 'b1', 'b2']].to_numpy())
            selected = review.assessment(y, g[['s0', 's1', 's2']].to_numpy())
            fold_metrics.append(dict(role=role, partition=partition, fold=fold, seed=seed, n=len(g),
                net=int(g.net.sum()), **selected,
                **{f'delta_{k}': selected[k]-baseline[k] for k in METRICS}))
        for (partition, arm, seed), g in data.groupby(['partition', 'arm', 'seed']):
            y = g.class_id.to_numpy()
            baseline = review.assessment(y, g[['b0', 'b1', 'b2']].to_numpy())
            selected = review.assessment(y, g[['s0', 's1', 's2']].to_numpy())
            metrics.append(dict(role=role, partition=partition, arm=arm, seed=seed, n=len(g),
                net=int(g.net.sum()), **selected,
                **{f'delta_{k}': selected[k]-baseline[k] for k in METRICS}))
        if role == 'validation':
            continue
        for (partition, seed), group in data.groupby(['partition', 'seed']):
            indexed = {arm: group[group.arm.eq(arm)].set_index(['fold', 'instance_id']).sort_index()
                       for arm in ARMS}
            for left_arm, right_arm in COMPARISONS:
                left = indexed[left_arm]
                right = left if right_arm == 'B' else indexed[right_arm].loc[left.index]
                assert left.index.is_unique and left.index.equals(right.index)
                np.testing.assert_array_equal(left.class_id, right.class_id)
                np.testing.assert_allclose(left[['b0', 'b1', 'b2']], right[['b0', 'b1', 'b2']], atol=1e-12, rtol=0)
                y = left.class_id.to_numpy()
                lp = left.prediction.to_numpy()
                rp = (right.base_prediction if right_arm == 'B' else right.prediction).to_numpy()
                ls = left[['s0', 's1', 's2']].to_numpy()
                rs = right[['b0', 'b1', 'b2'] if right_arm == 'B' else ['s0', 's1', 's2']].to_numpy()
                lc, rc = lp == y, rp == y
                diff = lc.astype(int)-rc.astype(int)
                lm, rm = review.assessment(y, ls), review.assessment(y, rs)
                meta = dict(role=role, partition=partition, seed=seed, left=left_arm, right=right_arm)
                pairs.append(dict(**meta, left_only=int((lc & ~rc).sum()), right_only=int((~lc & rc).sum()),
                    net=int(diff.sum()), disagreements=int((lp != rp).sum()),
                    **{f'delta_{k}': lm[k]-rm[k] for k in METRICS}))
                for c in range(3):
                    mask = y == c
                    classes.append(dict(**meta, class_id=c, n=int(mask.sum()), net=int(diff[mask].sum()),
                        delta_recall_pp=100*diff[mask].mean()))
                if role == 'test':
                    sample = left[['class_id', 'port', 'pass_group', 'mmsi']].copy()
                    sample['left_prediction'], sample['right_prediction'], sample['paired_net'] = lp, rp, diff
                    for key, value in meta.items():
                        sample[key] = value
                    samples.append(sample.reset_index())
                    for block in ['pass_group', 'mmsi']:
                        intervals.append(dict(**meta, block=block, **review.block_intervals(y, diff, left[block])))
                    means_for_intervals.setdefault((partition, left_arm, right_arm), []).append(sample)
        if role == 'test':
            for partition, part in data.groupby('partition'):
                folder = HERE/partition
                folder.mkdir(exist_ok=True)
                review.HERE, review.ARMS = folder, ARMS
                paths, missing = review.render(table, part, baseline_label='B (frozen calibration)',
                    subtitle='Primary J, seed17. All increments beyond frozen B; green=OBB; native pixels, nearest display.')
                panels.extend(paths)
                (folder/'review_summary.json').write_text(json.dumps(dict(panels=paths, missing=missing), indent=2), encoding='utf8')
    # Average paired seed effects within an object before drawing shared blocks.
    for (partition, left_arm, right_arm), frames in means_for_intervals.items():
        first = frames[0]
        diff = np.mean([frame.loc[first.index, 'paired_net'].to_numpy() for frame in frames], axis=0)
        for block in ['pass_group', 'mmsi']:
            intervals.append(dict(role='test', partition=partition, seed='mean', left=left_arm, right=right_arm,
                block=block, **review.block_intervals(first.class_id.to_numpy(), diff, first[block])))
    mf, pf, cf = pd.DataFrame(metrics), pd.DataFrame(pairs), pd.DataFrame(classes)
    mf.to_csv(HERE/'role_metrics.csv', index=False)
    pd.DataFrame(fold_metrics).to_csv(HERE/'fold_metrics.csv', index=False)
    mf.groupby(['role', 'partition', 'arm'])[METRICS+['net', 'delta_OA', 'delta_BA']].mean().to_csv(HERE/'arm_means.csv')
    pf.to_csv(HERE/'paired_effects.csv', index=False)
    cf.to_csv(HERE/'paired_class_effects.csv', index=False)
    pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv', index=False)
    sf = pd.concat(samples, ignore_index=True)
    sf.to_csv(HERE/'paired_samples.csv', index=False)
    sf.groupby(['partition', 'seed', 'left', 'right', 'pass_group']).agg(n=('paired_net', 'size'),
        net=('paired_net', 'sum')).to_csv(HERE/'paired_pass_effects.csv')
    selections = pd.read_csv(HERE/'selections.csv')
    selections.groupby('partition').agg(heads=('epoch', 'size'), epoch0=('epoch', lambda x: int(x.eq(0).sum())),
        epoch_median=('epoch', 'median'), epoch_max=('epoch', 'max')).to_csv(HERE/'selection_summary.csv')
    p = pd.read_csv(HERE/'predictions.csv')
    reconstructed = p[['b0', 'b1', 'b2']].to_numpy()+p[['delta0', 'delta1', 'delta2']].to_numpy()
    error = float(abs(reconstructed-p[['s0', 's1', 's2']].to_numpy()).max())
    assert error < 1e-12 and np.array_equal(reconstructed.argmax(1), p.prediction)
    receipt = dict(panels=panels, score_reconstruction_error=error, baseline_label='B (frozen calibration)',
        intervals='1000 fixed-prediction pass and MMSI block draws; seed mean keeps each object once; excludes fitting/selection uncertainty')
    (HERE/'analysis_summary.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(mf.query('role == "test"').to_string(index=False), flush=True)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
