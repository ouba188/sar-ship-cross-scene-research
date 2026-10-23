"""Set descriptive size boundaries from source TRAIN geometry, before outcomes."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
train = []
for fold in range(3):
    with np.load(ROOT.parent / 'baseline' / f'fold{fold}_inputs.npz') as z:
        train.extend(z['train'].tolist())
train = np.unique(train)
with np.load(ROOT / 'source_geometry.npz') as z:
    geom = z['geometry'][train, 0]
assert np.isfinite(geom[:, 2:4]).all()
values = dict(vv_length=geom[:, 2], vv_width=geom[:, 3],
              vv_aspect=geom[:, 2] / geom[:, 3])
result = dict(source_train_unique=len(train), pol='VV', rule='source train tertiles',
              source_validation_or_test_outcomes_used=False,
              boundaries={k: np.quantile(v, [1/3, 2/3]).tolist() for k, v in values.items()})
(HERE / 'strata_frozen.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
