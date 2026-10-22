"""Compare whole vectors only where the previous native windows are identical."""
import json
import numpy as np
import pandas as pd
from extract_context import HERE, BASE, dump


def main():
    old = BASE/'context_relation_pilot_20261022/ports24'
    ot, nt = pd.read_csv(old/'targets.csv'), pd.read_csv(HERE/'targets.csv')
    oe, ne = pd.read_csv(old/'neighbors.csv'), pd.read_csv(HERE/'neighbors.csv')
    with np.load(old/'features.npz') as f:
        old_target, old_neighbor, old_mask = f['target'], f['neighbor'], f['mask']
    with np.load(HERE/'features.npz') as f:
        new_target, new_neighbor = f['target'], f['neighbor']
    new_ids = dict(zip(nt.instance_id, nt.index))
    comparisons = []
    path_key = lambda p: str(p).replace('\\', '/').lower()
    for i, row in ot.iterrows():
        if row.instance_id not in new_ids:
            continue
        j = new_ids[row.instance_id]
        new = nt.iloc[j]
        if not all(path_key(row[f'raster_{p}']) == path_key(new[f'raster_{p}']) for p in ('vv', 'vh')):
            continue
        if not all(json.loads(row.target_window) == json.loads(new[f'target_window_{p}']) for p in ('vv', 'vh')):
            continue
        comparisons.append(dict(kind='target', instance_id=row.instance_id, detection_id='',
                                max_abs=float(np.abs(old_target[i]-new_target[j]).max()), exact=bool(np.array_equal(old_target[i], new_target[j]))))
    oe['target_index'], oe['slot'] = np.nonzero(old_mask)
    lookup = {(r.product, r.detection_id, r.read_window): r for r in oe.itertuples()}
    for row in ne.itertuples():
        previous = lookup.get((row.product, row.detection_id, row.read_window))
        if previous is None or json.loads(row.read_window_vh) != json.loads(previous.read_window):
            continue
        old_row = ot.iloc[previous.target_index]
        if not all(path_key(old_row[f'raster_{p}']) == path_key(getattr(row, f'raster_{p}')) for p in ('vv', 'vh')):
            continue
        a, b = old_neighbor[previous.target_index, previous.slot], new_neighbor[row.target_index, row.slot]
        comparisons.append(dict(kind='neighbor', instance_id=row.instance_id, detection_id=row.detection_id,
                                max_abs=float(np.abs(a-b).max()), exact=bool(np.array_equal(a, b))))
    result = pd.DataFrame(comparisons)
    assert len(result) and result.max_abs.max() <= 1e-5
    result.to_csv(HERE/'whole_replay.csv', index=False)
    summary = dict(counts=result.kind.value_counts().to_dict(), max_abs=float(result.max_abs.max()), exact=bool(result.exact.all()),
                   scope='same product/raster/native windows only; changed target windows are not expected to reproduce old vectors')
    dump(HERE/'whole_replay_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
