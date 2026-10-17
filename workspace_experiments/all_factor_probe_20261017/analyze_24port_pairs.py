"""Bounded same-vessel motion/image association; polarization kept separate."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent


def main():
    frame = pd.read_csv(HERE/'ports_motion/same_vessel_conditions.csv', dtype={'mmsi': str})
    frame = frame[frame.both_motion_exact].sort_values(['mmsi', 'first', 'second'])
    frame['max_gap'] = frame[['first_ais_minus_sar_midpoint_s', 'second_ais_minus_sar_midpoint_s']].abs().max(axis=1)
    rows = []
    for pol, group in frame.groupby('pol'):
        group = group.drop_duplicates('mmsi')
        for scope, p in [('exact_pairs', group), ('coverage>=.5', group[group.min_coverage >= .5]),
                         ('coverage>=.5_gap<=60s', group[(group.min_coverage >= .5) & (group.max_gap <= 60)])]:
            for factor in ['sog_change_knots', 'axis_change_deg', 'true_heading_change_deg']:
                for outcome in ['delta_contrast', 'delta_balance']:
                    f = p.dropna(subset=[factor, outcome])
                    absolute = factor != 'sog_change_knots'
                    x, y = f[factor].to_numpy(), f[outcome].to_numpy()
                    if absolute:
                        x, y = abs(x), abs(y)
                    rho = float(spearmanr(x, y).statistic) if len(f) >= 8 and np.std(x) > 0 and np.std(y) > 0 else np.nan
                    rows.append(dict(pol=pol, scope=scope, factor=factor, outcome=outcome, ships=len(f),
                        absolute_change=absolute, spearman=rho, motion_change_ge2kn=int(f.sog_change_knots.abs().ge(2).sum()),
                        input='stretched uint8; native per-pol crops may differ; no calibrated-power interpretation'))
    pd.DataFrame(rows).to_csv(HERE/'ports_motion_image_associations.csv', index=False)
    print(pd.DataFrame(rows).query('factor == "sog_change_knots"').to_string(index=False))


if __name__ == '__main__':
    main()
