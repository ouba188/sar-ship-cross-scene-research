# Parent analysis interface

Import this directory's run.py under a unique module name with importlib.util.spec_from_file_location. Imports do not train. Reader and Metric expose the implementation; trace_query provides a CPU float64 reconstruction for image analysis.

## Saved state and predictions

- `weights.npz`: keys `{arm}_{seed}_{Ub,Vb,Un,Vn,a,Mb,Mn}`. U/V matrices are 32x4; Mb/Mn are 32x32; a has four entries in VV_side,VV_end,VH_side,VH_end order. Every key stores the selection-chosen state, including the exact initialization if selected_step=0.
- `checkpoints_{arm}_{seed}.npz`: `{Ub,Vb,Un,Vn,a,Mb,Mn}_{step}` for steps0/20/50/100/200. These are diagnostic checkpoints, not alternate evaluation candidates.
- `predictions_{seed}.npz`: `idx`, `bank`, `full`, and each arm. Logit arrays use ORIGINAL table-row indexing. Only idx (utility/selection/evaluation) receives the learned correction; other rows retain full logits and were not inferred by the learned model. No ordinary anchor is included.
- `features_{seed}.npz`: `idx`, `bank`, `{arm}`, `{arm}_raw`, `{arm}_referencefeatures`, all original-row x 8-class x 4-group. `{arm}` is Ztheta-Zidentity; raw is unstandardized learned class-reference mean; referencefeatures is frozen standardized Zidentity. Rows outside idx are NaN, intentionally not computed. Learned standardized features = relative + referencefeatures.
- `scales.csv` contains per-arm/per-group frozen utility identity RMS and divisor. Independent is body-and-neighbor mean reconstruction, so its scale intentionally differs from parent's neighbor-only control.

To reconstruct final logits on idx: d=(features[arm]*a).sum(-1); d-=d.mean(-1,keepdims=True); logits=full+d. Existing prediction caches are float32; independent float64 tracing may differ at roundoff scale. `training_receipts.csv` records selected steps and replay error.

## Per-query trace

Call `trace_query(original_table_index, arm, Mb, Mn)` or pass the already-loaded x,mass,bank as optional keywords. It returns:

- `raw_scores`, `valid`: 8 classes x 4 groups.
- `reference_scores`, `reference_valid`: 4 groups x 1 query x 8 classes x 8 references. Invalid references contribute zero, not a counted zero observation. Average only valid references.
- `A`, and independent-only `An`: 4 groups x 1 query x 64 flattened references x 8 query edges x 8 reference edges. Flat reference index=8*class+reference index. Invalid edge rows/columns remain zero.
- `T`, `V`, and independent-only `body`,`neighbor`: 4 groups x 1 query x 64 references. These are before validity masking; use reference_valid when reporting. T is neighborhood transfer improvement; V is the exact reference-neighbor permutation risk correction.
- `qb`,`qn`: 4 x 1 x 8 x 32; `sb`,`sn`: 4 x 64 x 8 x 32. These are centered, transformed tokens, without renormalization. Explicit predicted neighborhoods are A@sn[:,None].
- `mq`: 4 x 1; `ms`,`lambda_body`: 4 x 64. `bank`: 8x8 original table indices. `legal_edges` and `neighbor_map`: query followed by bank.ravel(), then group,edge.

Call once with saved Mb/Mn and once with identity. For each class/group: average valid reference_scores; center across valid classes; divide by frozen divisor; subtract identity; multiply by a. Summing the four group contributions gives the corrected class logit. A true-minus-fixed-rival margin uses the difference of the two class contributions. Identity subtraction also subtracts the numerical reference method; the returned map is not itself a classifier probability or physical explanation.

True/reference permutation algebra remains exact after fixed linear transformations because centered rows remain centered. Shuffled arms replay the existing full-table sequential RNG mapping; do not slice or reorder x before rebuilding their Reader. The direct scalar `fixed.transfer` recenters its arguments and can independently check a selected unpadded trace after applying the learned transformations.

All training/configuration results should be reported relative to clean full. Configurations17/29/43 refer to existing full baselines; transform initialization uses fixed seed17 for all configurations. Their shared data and initialization do not create independent dataset replications.
