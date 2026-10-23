"""Reuse native-image review and evaluate paired, fixed-prediction differences."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze_results as parent


def main():
    parent.main('polarization_readout', show_used_polarization_moments=True)
    p = pd.read_csv(HERE / 'predictions.csv', dtype={'mmsi': str})
    test = p[p.role.eq('test')]
    paired, pooled = [], []
    for seed in [17, 29, 43]:
        real = test[test.seed.eq(seed) & test.arm.eq('pol_preserved')].set_index('instance_id').sort_index()
        for arm in ['aggregate_repeated', 'pol_preserved_capped']:
            control = test[test.seed.eq(seed) & test.arm.eq(arm)].set_index('instance_id').loc[real.index]
            diff = real.net.to_numpy()-control.net.to_numpy()
            paired += [dict(seed=seed, **r) for r in parent.intervals(real, diff, 'pol_preserved-'+arm)]
        for arm, g in test[test.seed.eq(seed)].groupby('arm'):
            pooled.append(dict(seed=seed, arm=arm, **parent.metrics(g)))
    pd.DataFrame(paired).to_csv(HERE / 'matched_intervals.csv', index=False)
    pd.DataFrame(pooled).to_csv(HERE / 'paired_metrics.csv', index=False)

    # Explain score changes against both the old and the new strongest rival.
    cases = pd.read_csv(HERE / 'review_cases.csv')
    indexed = test.set_index(['arm', 'seed', 'instance_id'])
    explained = []
    for row in cases.itertuples():
        g = indexed.loc[(row.arm, row.seed, row.instance_id)]
        y = int(g.class_id)
        base = g[[f'b{c}' for c in range(8)]].to_numpy(float)
        score = g[[f's{c}' for c in range(8)]].to_numpy(float)
        old = max((c for c in range(8) if c != y), key=lambda c: base[c])
        new = max((c for c in range(8) if c != y), key=lambda c: score[c])
        for rival, name in [(old, 'old_rival'), (new, 'new_rival')]:
            vv = float(g[f'vv_delta{y}']-g[f'vv_delta{rival}'])
            vh = float(g[f'vh_delta{y}']-g[f'vh_delta{rival}'])
            change = float((score[y]-score[rival])-(base[y]-base[rival]))
            assert abs(vv+vh-change) < 1e-10
            explained.append(dict(arm=row.arm, seed=row.seed, instance_id=row.instance_id,
                                  outcome=row.outcome, truth=y, comparison=name, rival=rival,
                                  rival_changed=old != new, before_margin=base[y]-base[rival],
                                  after_margin=score[y]-score[rival], vv_change=vv, vh_change=vh,
                                  decomposition_error=abs(vv+vh-change)))
    pd.DataFrame(explained).to_csv(HERE / 'case_polarization_contributions.csv', index=False)
    print(pd.DataFrame(pooled)[['arm', 'seed', 'OA', 'BA', 'net', 'delta_OA', 'delta_BA']].to_string(index=False))


if __name__ == '__main__':
    main()
