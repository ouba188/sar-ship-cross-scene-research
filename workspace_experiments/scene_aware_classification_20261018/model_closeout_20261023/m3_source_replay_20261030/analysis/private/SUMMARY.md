# private development

Completed paired seeds: [17, 29, 43].
Three-seed complete: True.
Analyzed queries: 160; expected: 160.

| Candidate - reference | Seed | Delta OA | Delta BA | Rescue | Harm | WW changes |
|---|---|---|---|---|---|---|
| adapt_only - saved_baseline | 17 | -0.625 | -0.2499999999999929 | 2 | 3 | 8 |
| source_mix - saved_baseline | 17 | 0.0 | 0.3214285714285765 | 2 | 2 | 8 |
| source_mix - adapt_only | 17 | 0.625 | 0.5714285714285694 | 1 | 0 | 0 |
| adapt_only - saved_baseline | 29 | -2.5 | -0.816326530612244 | 0 | 4 | 13 |
| source_mix - saved_baseline | 29 | -2.5 | -0.816326530612244 | 0 | 4 | 8 |
| source_mix - adapt_only | 29 | 0.0 | 0.0 | 0 | 0 | 5 |
| adapt_only - saved_baseline | 43 | 0.625 | 0.8928571428571388 | 2 | 1 | 6 |
| source_mix - saved_baseline | 43 | 0.625 | 0.8928571428571388 | 2 | 1 | 6 |
| source_mix - adapt_only | 43 | 0.0 | 0.0 | 0 | 0 | 0 |

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
