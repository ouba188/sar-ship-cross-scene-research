# oss2 development

Completed paired seeds: [17, 29, 43].
Three-seed complete: True.
Analyzed queries: 228; expected: 228.

| Candidate - reference | Seed | Delta OA | Delta BA | Rescue | Harm | WW changes |
|---|---|---|---|---|---|---|
| adapt_only - saved_baseline | 17 | 0.0 | -0.9439100674381962 | 5 | 5 | 4 |
| source_mix - saved_baseline | 17 | 0.0 | -0.36273997832544325 | 4 | 4 | 4 |
| source_mix - adapt_only | 17 | 0.0 | 0.5811700891127529 | 1 | 1 | 0 |
| adapt_only - saved_baseline | 29 | 0.0 | 1.3743128435782097 | 3 | 3 | 1 |
| source_mix - saved_baseline | 29 | -0.877192982456144 | 0.0 | 2 | 4 | 1 |
| source_mix - adapt_only | 29 | -0.877192982456144 | -1.3743128435782097 | 0 | 2 | 0 |
| adapt_only - saved_baseline | 43 | -0.43859649122807554 | -1.0998714126083016 | 2 | 3 | 2 |
| source_mix - saved_baseline | 43 | -0.8771929824561298 | -0.8189725362038089 | 4 | 6 | 1 |
| source_mix - adapt_only | 43 | -0.4385964912280542 | 0.2808988764044926 | 2 | 3 | 1 |

## Missing Inputs

```json
[]
```

Only complete paired seeds with both new prediction files and training reports are analyzed.
No old prediction files or old training results are substituted. Baseline comes from these new NPZs.
Scores are logits; OA/BA and differences are percent/percentage points; NLL/Brier are unscaled.
BA averages present classes. Absent classes have null recall, not zero recall.
Means and sample SD are over available paired seeds; incomplete results are provisional.
Repeated query events across seeds are not independent samples or ships.
Positive OA/BA differences improve accuracy; negative NLL/Brier differences improve risk.
Training reports are retained as reported; gradients alone do not establish a causal benefit.
