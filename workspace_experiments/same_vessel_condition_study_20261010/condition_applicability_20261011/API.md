# Frozen applicability exports

Import from `condition_applicability.py`:

```python
meta, fit, held, features = load_features()
# features: full global-observation-row N x 6, float64.
# Uncached rows are NaN. Column 5 is center_border_log_ratio.
# Total logenergy is logaddexp.reduce(features[:, :3], axis=1)
# on finite rows only; this is the sum of overlapping soft-branch energies.
inputs = gate_inputs(anchor_a, total_logenergy_a, angle_a, angle_b)
# N x 4: raw center A, total logenergy A, abs angle gap, mean angle.
with np.load(APP / 'gate.npz') as model:
    g = predict_gate(inputs, model)
```

`predict_gate` returns length-N condition weights in [0,1]. It accepts either
an `np.load` NpzFile or a dictionary. Angles are degrees. No normalization is
required by the caller. Nonfinite gate input rows return zero; callers must
still exclude invalid response predictions rather than multiply NaN by zero.

Exports:
- `gate.npz`: feature_index (scalar int; -1 for root), threshold (float),
  leaf_mix (length 2), feature_names (length 4), source-training leaf statistics.
- `anchor.npz`, `condition.npz`: unchanged response model dictionaries.
- `capacity.npz`, `sham.npz`: old control dictionaries.

Response forecasts use the original API:

```python
from condition_evidence_response import predict_model
with np.load(APP / 'anchor.npz') as model:
    anchor = predict_model(center_a, total_a, angle_a, angle_b, model)
with np.load(APP / 'condition.npz') as model:
    condition = predict_model(center_a, total_a, angle_a, angle_b, model)
prediction = (1 - g) * anchor + g * condition
```

Prediction is absolute center/border response change, not signed change,
classification probability, or probability of causality. To reverse A/B,
use B's center/total and reversed angles. Gate output may differ by direction
when the selected gate feature is center/total. Current deployed split is mean
angle, so current blend is symmetric but the response forecast need not be.

Frozen rule: mean_incidence_deg <= 37.380153402147705 -> mix 0;
otherwise -> mix 0.05361370737910942. No held or target outcomes selected it.

`source_gate_training_oof.npz` is a gate-training artifact, not honest gate
evaluation. Use `source_nested_predictions.npz` for outer-fold evaluation.
`outer0/`, `outer1/`, `outer2/` contain all corresponding deployment models.
Source/held metrics concern matched same-vessel pairs. Arbitrary different-ship
support-query transfer is a separate downstream hypothesis, not validated by
these response errors. No support matching or classification code lives here.
