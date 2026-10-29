# Development CPU analysis

Only `analysis.py` and `analysis/` belong to this analysis task. No training,
checkpoint loading, mainrun edits, Git operations or final-test reads occur.
The preceding coverage analysis provides the NPZ alignment and metrics;
its gallery algorithm and native crop/pad panel helpers are reused.

After results are authorized and completion JSON receipts exist, run from
the experiment directory:

```powershell
python -B analysis.py --dataset private --seeds 17 29 43 --gallery
python -B analysis.py --dataset oss2 --seeds 17 --gallery
python -B analysis.py --dataset all --gallery --transplant
```

Omitting `--seeds` requests 17/29/43 but reads only completed paired orders.
An NPZ without both arm completion receipts never enters the analysis.
`paired_three_orders_complete` remains false until all three are paired.
There is deliberately no final/evaluation scope or arbitrary manifest option.

Outputs per dataset:

- `summary.json`, `SUMMARY.md`, `metrics.csv`: sourcefit before from receipts,
  saved complete baseline and each after endpoint, with paired deltas and SD.
- `paired.csv`, `paired_means.csv`, `events.csv`, `changed_queries.csv`: all
  four correctness states, top1 changes including wrong-to-wrong, unique IDs
  separate from repeated order events, and rescue/harm versus saved status.
- `class_tradeoffs.csv`: per-class recall/NLL/Brier changes joined to source
  sample/MMSI/pass counts and inverse-frequency weights.
- `source_adapt.csv`: float32 N/(K*n_c) sample weights, class totals and exact
  six-epoch order-dependent CE/KL coefficient exposure after batch weight-sum
  normalization, using original teacher-correct/useful flags. This does not
  estimate gradients or prove why a class changed.
- `four_state_cases/`: center versus saved baseline, all four states.
- `incremental_cases/`: independently selected center versus raw cases,
  all four states even with zero top1 changes; no fallback comparator.
- `transplant.csv`, `transplant_means.csv`, `TRANSPLANT.md`: optional completed
  M3-only/LIVE-only diagnostics versus saved baseline, their full joint model,
  and raw-component counterpart. No extra case galleries or weight loading.

Galleries keep VV/VH separate, original pixel scale, actual 160px crop/pad and
separate OBB overlays. Actual fixed-bank references exclude self/same MMSI;
displayed references follow bank order rather than inferred attention.
`cases.json` records all order predictions/margins, and `native_sources.json`
records paths and crop/pad provenance. Absent states are explicit. Generated
images are `not_reviewed` until a separate visual note records inspection.

Before has no per-query NPZ and cannot support a before/after case attribution.
Selected parameters without gradients (including initial reader gain entries)
are listed separately from nonzero first-step gradients. Neither a selection
list nor a gradient norm alone establishes a parameter's causal contribution.

Synthetic verification:

```powershell
python -B analysis/test_analysis.py
```
