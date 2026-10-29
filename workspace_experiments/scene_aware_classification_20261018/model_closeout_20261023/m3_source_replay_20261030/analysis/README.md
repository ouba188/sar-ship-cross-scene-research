# M3 source replay CPU analysis

Run from this experiment directory:

```powershell
python -B analysis.py --dataset all --gallery
python -B analysis.py --dataset private --seeds 17 --gallery
python -B analysis.py --dataset all --seeds 17 29 43 --gallery
python -B analysis/test_analysis.py
```

Input files are exclusively this round's `{dataset}/predictions/{arm}_s{seed}.npz`
(`score`, `baseline`, `y`, `ids`) and matching `training/{arm}_s{seed}.json`.
Arms: `adapt_only`, `source_mix`; default requested seed: 17. A complete three-seed
summary requires 17, 29, 43, but expansion is the main agent's decision. Analysis
never starts or expands training. `source_mix` is source-fit mixing, not centering.
Scores are logits. The development query/reference manifests are the same as
`m3_center_joint_20261030/analysis.py`; no final evaluation manifest is read.

Each seed is included only when both arms have predictions and training reports.
Missing files/samples are recorded in `summary.json` and `SUMMARY.md`. A partial
sample cohort is not silently evaluated. Three-seed completion is explicit;
one-seed SD is null. Old predictions are never substituted.

Outputs per dataset:

- `metrics.csv`, `endpoint_means.csv`: OA/BA/NLL/Brier per endpoint and seed mean/SD.
- `paired.csv`, `paired_means.csv`: both arms versus saved baseline, and source_mix versus adapt_only.
- `class_tradeoffs.csv`: per-class rescue/harm, recall, NLL/Brier change, and missing-class flags.
- `events.csv`, `changed_queries.csv`: query IDs, states, seed, margins, and available vessel/pass/port metadata.
- `summary.json`: completion, input paths, training reports, paired means/SD, event counts and unique query counts.
- Three comparison galleries: `index.html`, `cases.json`, `selection.json`, `native_sources.json`, separate VV/VH PNGs.

Gallery selection is deterministic: one distinct query per state (rescued,
harmed, persistently correct, persistently incorrect), ranked by state frequency
then absolute fixed-rival margin change, ID, seed. Missing states are explicit.
All seed predictions and fixed-rival margins are saved for each selected query.
Reference chips come from the actual fixed bank, excluding identical IDs/MMSIs;
the first two same-class and rival-class entries are displayed in bank order.
Bank order is not an attention ranking. Original pixels and exact crop/pad are
shown at scale 1; OBB overlays are separate. No chip is resized or restretched.

Rendered galleries start as `not_reviewed`; image facts and mechanism claims
require actual inspection. Repeated events across seeds are not independent
ships. Outcome-selected cases do not estimate population proportions. No
physical cause is inferred automatically. The analysis does not import PyTorch,
run GPU work, alter training scripts, or perform Git operations.

Focused checks use synthetic logits only inside temporary `analysis/` folders,
plus one real query/reference native-pixel equality check. They do not manufacture
experimental results or populate experiment prediction directories.
