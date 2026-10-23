"""Fixed substitutions distinguish matching changes from selected gain/scale."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UP = ROOT.parent / 'regional_utility_pilot/upstream'
sys.path.insert(0, str(ROOT))
from train_readout import metrics


def main():
    cache = np.load(HERE / 'scores.npz'); idx = cache['idx']
    table = pd.read_csv(UP / 'table.csv'); y = table.class_id.to_numpy()
    ids = np.flatnonzero(table.iloc[idx].role.eq('evaluation'))
    scales = json.loads((HERE / 'transforms.json').read_text(encoding='utf-8'))
    selected = pd.read_csv(HERE / 'source_metrics.csv').query("role == 'selection'")
    valid = cache['available']
    rows = []; predictions = {}
    for seed in (17, 29, 43):
        base = np.load(UP / f'strategies_seed{seed}.npz')['logits'][:, 2]
        gain = selected.query('seed == @seed').set_index('arm').weight.to_dict()
        for values_arm, scale_arm, gain_arm in (
                ('coupled', 'coupled', 'coupled'), ('fixed', 'fixed', 'fixed'),
                ('fixed', 'fixed', 'coupled'), ('fixed', 'coupled', 'coupled'),
                ('coupled', 'fixed', 'fixed')):
            z = cache[values_arm] / np.asarray(scales[scale_arm])[None, None]
            z = (z * valid).sum((2, 3)) / np.maximum(valid.sum((2, 3)), 1)
            z -= z.mean(-1, keepdims=True)
            score = base[idx] + gain[gain_arm] * z
            key = f'{seed}_{values_arm}_scale{scale_arm}_gain{gain_arm}'
            predictions[key] = score
            rows.append(dict(seed=seed, values=values_arm, scale=scale_arm, gain_source=gain_arm,
                             gain=gain[gain_arm], **metrics(y[idx[ids]], score[ids])))
    pd.DataFrame(rows).to_csv(HERE / 'common_readout.csv', index=False)
    np.savez_compressed(HERE / 'common_readout.npz', idx=idx, **predictions)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
