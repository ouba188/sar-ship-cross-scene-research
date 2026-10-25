"""Report paired gains by actual acquisition, class and missing polarization."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'oss1_m123_replication_20261026'))
from train_m3 import metrics, write_json


def analyze(root):
    table = pd.read_csv(root/'table.csv', dtype={'mmsi': str})
    te = np.flatnonzero(table.role.eq('evaluation'))
    y = table.class_id.to_numpy(int)
    data = {int(p.name[4:]): dict(np.load(p/'predictions.npz'))
            for p in sorted(root.glob('seed*')) if (p/'predictions.npz').exists()}
    arms = ('old_both', 'B_fixed', 'spatial', 'frequency')
    rows = []
    for seed, z in data.items():
        strata = [('all', 'all', te)]
        for key in ('pass_group', 'class_id'):
            for value, inds in table.iloc[te].groupby(key).groups.items():
                strata.append((key, str(value), np.array(inds)))
        paired = z['available'].all(1)
        for value, mask in [('both', paired), ('single', ~paired)]:
            if mask[te].any():
                strata.append(('polarization', value, te[mask[te]]))
        for key, value, ix in strata:
            for arm in arms:
                rows.append(dict(seed=seed, group_type=key, group=value, arm=arm,
                                 **metrics(y[ix], z[arm][ix])))
    frame = pd.DataFrame(rows)
    frame.to_csv(root/'group_metrics.csv', index=False)
    frame[frame.group_type.eq('pass_group')].groupby(['seed','arm'])[['OA','BA']].mean().to_csv(root/'equal_pass_metrics.csv')
    # Repeated seeds are averaged within each query, not treated as new ships.
    rng = np.random.default_rng(1026)
    groups = [np.array(ix) for ix in table.iloc[te].groupby('pass_group').groups.values()]
    pairs = [('frequency','B_fixed'), ('frequency','spatial'), ('frequency','old_both'),
             ('spatial','B_fixed'), ('B_fixed','old_both')]
    uncertainty = []
    for arm, base in pairs:
        d = np.mean([(z[arm].argmax(1) == y).astype(float)-(z[base].argmax(1) == y)
                     for z in data.values()], axis=0)
        draws = []
        for _ in range(2000):
            ix = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
            draws.append([d[ix].mean()*100, np.mean([d[ix[y[ix] == c]].mean()
                          for c in np.unique(y[ix])])*100])
        low, high = np.quantile(draws, [.025,.975], axis=0)
        uncertainty.append(dict(arm=arm, baseline=base, OA_low=low[0], OA_high=high[0],
            BA_low=low[1], BA_high=high[1], acquisition_groups=len(groups),
            scope='resample target acquisitions; average seeds within observation; BA over represented classes'))
    pd.DataFrame(uncertainty).to_csv(root/'acquisition_bootstrap.csv', index=False)
    write_json(root/'coverage_analysis.json', dict(query_n=len(te), query_mmsi=table.iloc[te].mmsi.nunique(),
        query_passes=len(groups), seeds=list(data),
        note='Few acquisitions limit interval precision. Earlier class-stratified intervals condition on the fixed observed mix.'))
    print(frame[frame.group_type.eq('all')].to_string(index=False))


if __name__ == '__main__':
    analyze(Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'m3_results')
