# Case Manifest Interface

Created by `python -B analyze.py` after the completed predictions are available:

`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/shallow_m2_expansion_20261028/analysis/cases/cases.json`

The top level is a JSON list of eight unique query objects. For the bounded same-weight on/off diagnostic, use `id`, `representative_seed`, and `fixed_rival`; query preprocessing comes from the matching `query_records.json` row. This analysis does not run the diagnostic or load any model.

| Field | Type / meaning |
|---|---|
| `id` | String query ID matching prediction `ids` and query records |
| `label`, `class_name` | Corrected integer 0..7 and canonical output class name |
| `state` | Selected category: `rescue`, `harm`, `CC`, or `WW` |
| `port`, `mmsi`, `pass_group`, `product` | Source query metadata |
| `representative_seed` | Integer 17, 29, or 43; selected state holds at this seed |
| `outcome_counts` | Number of the three seeds in each of the four states |
| `mean_margin_delta` | Mean across seeds of layer2 minus baseline top1 margin |
| `fixed_rival` | Strongest baseline nontruth class at representative seed, held fixed for all paired fixed margins |
| `confusion_classes` | Union of nontruth predictions across all six models plus fixed rival |
| `eligible_reference_ids` | Actual bank excluding same observation/MMSI |
| `true_reference_ids` | Eligible references of the corrected true class |
| `confusion_reference_ids` | Eligible references of confusion classes |
| `per_seed` | Three objects, in seed order 17,29,43 |
| `selection`, `margin_rule`, `visual_status` | Selection definition, margin definition, and pending manual review |

Each `per_seed` object contains:

- `seed`, `state`, `baseline_prediction`, `layer2_prediction`.
- `baseline_logits`, `layer2_logits`: eight unnormalized scores, class IDs 0..7.
- `baseline_margin`, `layer2_margin`, `delta_margin`: true logit minus current strongest nontruth logit; each model can have a different strongest rival.
- `baseline_fixed_margin`, `layer2_fixed_margin`: true logit minus the common top-level `fixed_rival` logit.
- `baseline_NLL`, `layer2_NLL`, `delta_NLL`, and corresponding `Brier` fields.

Selection makes two rescue/harm/CC/WW rounds: descending state consistency across three seeds, then in round two new class/port coverage within the state, then absolute mean margin change, then lexical ID. Missing categories are reported through zero counts and filled from available categories. Representative seed maximizes absolute margin change among seeds with the selected state; ties use seed order. This is posthoc representation, not prevalence estimation or checkpoint selection.

Class IDs: 0 bulk_carrier, 1 fishing, 2 general_cargo, 3 chemical_tanker, 4 container, 5 crude_oil_tanker, 6 tug, 7 offshore.

`analysis/verification/cases/cases.json` contains synthetic verification examples only. It must not be used for the actual model diagnostic.
