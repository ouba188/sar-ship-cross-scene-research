"""Descriptive paired correctness and B-relative overlap; no oracle selection."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def counts(a, b):
    y = a.class_id.to_numpy()
    base = a.base_prediction.to_numpy() == y
    mean = a.prediction.to_numpy() == y
    local = b.prediction.to_numpy() == y
    lr, mr = ~base & local, ~base & mean
    lh, mh = base & ~local, base & ~mean
    return dict(n=len(a), unique_observations=a.instance_id.nunique(),
        base_correct=int(base.sum()), base_wrong=int((~base).sum()),
        both_correct=int((local & mean).sum()),
        local_only_correct=int((local & ~mean).sum()),
        mean_only_correct=int((mean & ~local).sum()),
        both_wrong=int((~local & ~mean).sum()),
        local_rescued=int(lr.sum()), mean_rescued=int(mr.sum()),
        both_rescued=int((lr & mr).sum()),
        local_only_rescued=int((lr & ~mr).sum()),
        mean_only_rescued=int((mr & ~lr).sum()),
        neither_rescued=int((~base & ~local & ~mean).sum()),
        local_harmed=int(lh.sum()), mean_harmed=int(mh.sum()),
        both_harmed=int((lh & mh).sum()),
        local_only_harmed=int((lh & ~mh).sum()),
        mean_only_harmed=int((mh & ~lh).sum()),
        neither_harmed=int((base & local & mean).sum()))


def main():
    mean = pd.read_csv(HERE/'predictions.csv')
    old = pd.read_csv(HERE.parent/'strong_calibration_increment/predictions.csv')
    local = old[old.arm.eq('local')].set_index(['fold', 'seed', 'role', 'instance_id'])
    pooled, classes, unique_val = [], [], []
    for (role, seed), g in mean.groupby(['role', 'seed']):
        a = g.sort_values(['fold', 'instance_id']).reset_index(drop=True)
        key = pd.MultiIndex.from_frame(a[['fold', 'seed', 'role', 'instance_id']])
        b = local.loc[key].reset_index()
        np.testing.assert_array_equal(a.class_id, b.class_id)
        np.testing.assert_array_equal(a.base_prediction, b.base_prediction)
        np.testing.assert_array_equal(a[[f'b{c}' for c in range(8)]], b[[f'b{c}' for c in range(8)]])
        meta = dict(role=role, seed=seed)
        pooled.append(dict(**meta, **counts(a, b)))
        for c, sub in a.groupby('class_id'):
            classes.append(dict(**meta, class_id=c, class_name=sub.class_name.iloc[0],
                **counts(sub, b.loc[sub.index])))
        if role == 'validation':
            ix = a.drop_duplicates('instance_id').index
            unique_val.append(dict(**meta, dedup_rule='first fold per instance', **counts(a.loc[ix], b.loc[ix])))
    for name, rows in [('overlap_pooled', pooled), ('overlap_classes', classes),
                       ('overlap_validation_unique', unique_val)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv', index=False)
    print(pd.DataFrame(pooled).to_string(index=False))
    print(pd.DataFrame(classes)[['role','seed','class_name','n','local_only_correct','mean_only_correct',
        'both_rescued','local_only_rescued','mean_only_rescued','both_harmed',
        'local_only_harmed','mean_only_harmed']].to_string(index=False))


if __name__ == '__main__':
    main()
