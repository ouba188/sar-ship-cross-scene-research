# Matched source-fit analysis handoff

CPU analysis completed. All six actual training receipts report 8 completed epochs,
408 steps, frozen_unchanged=true, save_replay_max_error=0, validation_selection=false,
and selection_and_adaptation_not_run=true. Models, predictions, runner, protocol and
main findings were not modified by this analysis. Git synchronization remains with main.

This is the fixed source-fit first stage, not the full intervention-adapted system.
The historically reused development cohort has 160 observations, 158 ships, seven
present classes, three ports and three passes. Fit/development share Jebel Ali,
Port Klang and Singapore, so this is not strict LOPO. Each development port is
represented by one pass here; matching port/pass tables are the same sample
partition, not two independent confirmations. Three order seeds reuse the queries
and fixed references and are not independent datasets. No checkpoint or recipe
selection was performed from these results.

## Primary Two-Arm Results

Mean +/- sample SD across order seeds 17/29/43. OA/BA in percent; paired differences
in percentage points. BA averages the seven present development classes. NLL is
natural-log cross entropy; Brier sums squared error over all eight classes.

| Arm | OA | BA | NLL | Brier |
|---|---:|---:|---:|---:|
| legacy | 41.666667 +/- 2.953635 | 33.222892 +/- 1.943356 | 2.647204 +/- 0.282654 | 0.887192 +/- 0.063569 |
| spatial | 41.041667 +/- 2.818281 | 33.199186 +/- 1.759171 | 2.664567 +/- 0.283742 | 0.893252 +/- 0.066502 |
| paired spatial minus legacy | -0.625000 +/- 1.082532 | -0.023706 +/- 1.074824 | +0.017363 +/- 0.018482 | +0.006060 +/- 0.005816 |

| Seed | dOA | dBA | dNLL | dBrier | Rescue / harm events |
|---|---:|---:|---:|---:|---:|
| 17 | +0.625000 | +1.190476 | -0.001999 | +0.000405 | 1 / 0 |
| 29 | -1.250000 | -0.408163 | +0.019270 | +0.005750 | 0 / 2 |
| 43 | -1.250000 | -0.853432 | +0.034818 | +0.012024 | 0 / 2 |

Spatial has no demonstrated mean performance advantage under this fixed recipe:
OA is lower, BA is nearly tied, and probability losses are higher. Brier worsens
in all three seeds. The limited experiment does not establish equivalence or an
overall final-system ranking.

There is one rescue event on one unique query and four harm events on three unique
queries. CC has 196 events / 85 unique IDs; WW has 279 events / 111 unique IDs.
Unique-ID counts across states overlap because a query's state can vary by seed.

The single general-cargo rescue raises that class's mean recall by 2.777778 points
(12 development observations); three chemical-tanker harms lower its mean recall
by 1.428571 points (70 observations), and one container harm lowers its mean recall
by 1.515152 points (22 observations). Their macro average over seven classes is
the observed -0.023706 BA difference. The event-weighted OA difference is
(1-4)/480*100 = -0.625 points. This explains the OA/BA arithmetic, not a physical
or optimization cause. No fishing query occurs in development.

Port mean dOA/dBA: Jebel Ali -0.966184/-0.272109; Port Klang 0/+1.948052;
Singapore -0.529101/-0.144300. NLL and Brier both worsen on average in each port.
Full per-seed class/port/pass data are in groups.csv and groups_summary.csv.

## Separate Posthoc Pruning Diagnostic

Pruned is the trained legacy model converted to remove explicit M1/M2 frequency
branches, with retained weights and M3 unchanged and no refit. It is not a third
fairly trained arm. These tables remain separate from the primary two-arm summary.

| Comparison | dOA | dBA | dNLL | dBrier |
|---|---:|---:|---:|---:|
| pruned minus legacy | 0 +/- 0 | 0 +/- 0 | -0.001088 +/- 0.001026 | -0.000377 +/- 0.000318 |
| spatial minus pruned | -0.625000 +/- 1.082532 | -0.023706 +/- 1.074824 | +0.018451 +/- 0.017976 | +0.006437 +/- 0.005601 |

Pruning changes zero top-1 predictions across 480 seed events; its NLL/Brier are
slightly lower in every seed. Spatial versus pruned changes 10 top-1 predictions
on eight unique IDs, comprising one rescue, four harms and five WW class changes.

Truth-minus-case-rival logit margins for every correctness flip:

| Query | Seed | True / rival class | Legacy | Pruned | Spatial |
|---|---:|---|---:|---:|---:|
| 37321 | 17 | general cargo / chemical tanker | -0.025014 | -0.042443 | +0.043169 |
| 6194 | 29 | chemical tanker / general cargo | +0.080435 | +0.082848 | -0.050838 |
| 6194 | 43 | chemical tanker / general cargo | +0.087471 | +0.087619 | -0.227678 |
| 31779 | 29 | chemical tanker / general cargo | +0.023548 | +0.030222 | -0.256770 |
| 34278 | 43 | container / bulk carrier | +0.135765 | +0.150019 | -0.020959 |

The observed flips are not reproduced by removing the branches after legacy
training. They depend on the separately optimized spatial path under this recipe.
This does not isolate the responsible parameter, establish a physical cause, or
show that frequency information was unused during training. M3 still contains
spatial-frequency matching in all conditions.

## Cases And Native Panels

Eight unique queries; separate VV/VH panels, original display pixels at 1:1 scale,
the exact model crop/pad, and a duplicate crop with the target OBB. Same-class,
selected-rival and additional current-rival references are from the fixed bank,
with ID/MMSI exclusion and unavailable polarizations explicitly marked.

| Panel prefix | ID | Selected seed/state | States at 17 / 29 / 43 |
|---|---:|---|---|
| case_01 | 37321 | 17 rescue | rescue / WW / CC |
| case_02 | 6194 | 29 harm | CC / harm / harm |
| case_03 | 31779 | 29 harm | WW / harm / WW |
| case_04 | 27820 | 29 CC | CC / CC / CC |
| case_05 | 23235 | 43 WW | WW / WW / WW |
| case_06 | 3951 | 17 CC | CC / CC / CC |
| case_07 | 31141 | 29 WW | WW / WW / WW |
| case_08 | 34278 | 43 harm | CC / CC / harm |

Panel paths are cases/<prefix>_vv.png and cases/<prefix>_vh.png.
Exact seed-specific logits, class margins, IDs and references are in cases/cases.json;
source image paths, native sizes and crop transforms are in cases/native_sources.json.
31141 remains wrong in seed29 but changes from bulk carrier to crude-oil tanker;
its gallery includes both competing classes' actual references. Selected cases are
illustrations, not an estimate of prevalence.

Main reported native review of 37321 VV/VH and references 6913 (VV/VH), 22755
(VV only), 24527 (VV/VH) and 7790 (VV/VH), plus queries 6194 and 31779 VV/VH.
Main's visible observation for the latter two: cross-shaped bright extensions in
VV and a more concentrated appearance in VH. This is an attributed observation,
not a finding independently re-established by this analysis or a physical cause.
No further visible facts for 37321 were supplied. The analysis agent opened only
case_01_vh.png to inspect rendering, including the missing-VH marker for 22755,
clean native image, crop and OBB layout. Full remaining pixel review stays with
main. Automated gallery flags do not certify that all cases have been reviewed.

## Output Links

- [Primary numeric report](NUMERIC_REPORT.md)
- [Posthoc pruning report](posthoc_pruning/NUMERIC_REPORT.md)
- [Native gallery](cases/index.html)
- [Cohort report](COHORT_REPORT.md)
- [Per-query primary events](seed_events.csv)
- [Per-query pruning events](posthoc_pruning/seed_events.csv)

No raw dataset, checkpoints or prediction arrays were uploaded. Native image paths
and derived panels remain local for main's existing archive/exclusion policy.
