# Consistent cross-polarization recovery

Computed from frozen predictions. Main reviewed all 12 native panels; see ../FINDINGS_ZH.md.

- A: Original Full fine-tuning; original queries and references.
- B: A weights; restored queries, original references.
- C: A weights; restored queries and references.
- D: Matched fine-tuning with restored training, queries and references.

Primary D-C: matched training/validation organization under identical restored inference inputs.
D-A: whole-process comparison; both training and inference inputs change.
Original test missing30 is a protocol group; no full-scene claims.

| Split | Arm | OA mean +/- SD | BA mean +/- SD | Macro-F1 mean +/- SD |
|---|---|---:|---:|---:|
| validation | A | 65.936 +/- 2.920 | 61.224 +/- 1.583 | 59.390 +/- 2.085 |
| validation | B | 66.520 +/- 2.570 | 62.825 +/- 0.906 | 60.716 +/- 2.111 |
| validation | C | 66.520 +/- 3.407 | 62.557 +/- 1.593 | 60.536 +/- 3.062 |
| validation | D | 65.497 +/- 0.913 | 62.163 +/- 0.338 | 59.876 +/- 0.227 |
| evaluation | A | 64.727 +/- 2.702 | 59.396 +/- 3.831 | 57.693 +/- 3.187 |
| evaluation | B | 65.344 +/- 2.100 | 59.430 +/- 3.425 | 57.842 +/- 2.864 |
| evaluation | C | 65.785 +/- 2.400 | 59.776 +/- 3.610 | 58.198 +/- 3.060 |
| evaluation | D | 61.464 +/- 2.702 | 59.123 +/- 3.420 | 55.795 +/- 2.903 |

| Test comparison | Metric | Delta pp | Paired MMSI 95% CI | Positive seeds |
|---|---|---:|---|---:|
| D-C | oa | -4.321 | [-7.143, -1.511] | 0/3 |
| D-C | ba | -0.653 | [-4.761, +3.597] | 2/3 |
| D-C | macro_f1 | -2.403 | [-5.491, +0.668] | 0/3 |
| D-A | oa | -3.263 | [-6.349, -0.178] | 0/3 |
| D-A | ba | -0.273 | [-4.198, +3.904] | 2/3 |
| D-A | macro_f1 | -1.898 | [-5.113, +1.359] | 1/3 |

| Test original-input group | Arm | N | OA | BA | Macro-F1 |
|---|---|---:|---:|---:|---:|
| original_both | A | 348 | 66.762 | 60.522 | 58.803 |
| original_both | B | 348 | 66.762 | 60.522 | 58.803 |
| original_both | C | 348 | 67.146 | 60.836 | 59.107 |
| original_both | D | 348 | 62.739 | 59.481 | 56.270 |
| original_missing | A | 30 | 41.111 | 44.381 | 39.108 |
| original_missing | B | 30 | 48.889 | 50.821 | 46.461 |
| original_missing | C | 30 | 50.000 | 51.515 | 47.220 |
| original_missing | D | 30 | 46.667 | 52.146 | 45.872 |

Class recalls/F1, original availability and scene/pass/orbit denominators are in the CSV tables.
data_scope.csv gives original missing VV/VH counts; class coverage is observational, not causal evidence.
Selected A/D costs are in training_budget*.csv; these exclude scouts and inherited staged training.
Sample SD is across three seeds. Ship intervals condition on existing runs/scenes and do not include selection uncertainty.
Validation is descriptive; the evaluation cohort has been used historically. No hypothesis tests.
See ANALYSIS_NOTES.md for metric definitions, design limits and interpretation.

Native panels: 12. Main visual review complete. cases_local.json, index.html and cases/ remain PRIVATE.
