"""Separate active region updates from source-selected zero-step fallback."""
from pathlib import Path
import pandas as pd
from analyze_results import paired, SCORE, BASE

HERE = Path(__file__).resolve().parent
f = pd.read_csv(HERE.parent / 'test_predictions.csv', float_precision='round_trip')
rows = []
for (seed, fold), g in f.groupby(['seed', 'fold']):
    arms = {arm: part.sort_values('instance_id').reset_index(drop=True) for arm, part in g.groupby('arm')}
    dyn = arms['dynamic']
    for right in ['baseline', 'fixed', 'global', 'merged']:
        epoch_left = int(dyn.epoch.iloc[0])
        epoch_right = 0 if right == 'baseline' else int(arms[right].epoch.iloc[0])
        before = dyn[BASE].to_numpy() if right == 'baseline' else arms[right][SCORE].to_numpy()
        rows.append(dict(seed=seed, fold=fold, left='dynamic', right=right,
            epoch_left=epoch_left, epoch_right=epoch_right,
            both_updates_active=epoch_left > 0 and epoch_right > 0,
            **paired(dyn.class_id.to_numpy(int), dyn[SCORE].to_numpy(), before)))
out = pd.DataFrame(rows)
out.to_csv(HERE / 'test_active_scope.csv', index=False)
print(out.loc[out.right.eq('fixed')].to_string(index=False))
