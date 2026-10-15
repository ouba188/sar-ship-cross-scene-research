# Projection-Residual Expert: Matched 64-Label Pilot

Completed 2026-10-08. Status: usable OA-oriented development candidate with
explicit BA tradeoffs, not a replacement for the current BA-oriented model.

## What Changed

The preceding diagnosis suggested that current source-PCA views omit or make
harder to read some useful external-data information. This pilot tests that
direction in the ACTUAL 64-label, 5%-handoff setup, not the 512/4096-label probe.

For each frozen encoder, remove the source PCA mean and retained-PC projection:

    r(x) = (x - mu) - ((x - mu) P^T) P

Compute this separately for paired R50 (2048D, retaining128 PCs) and VH SSL
R18 (512D, retaining64 PCs). Concatenate the remaining coordinates, standardize
with the corresponding identity-purged source cohort, and divide by sqrt(2560).
The existing normalized 192D source-PCA feature branch remains available.

Fit the expert's residual to the same 64 observed labels around the unchanged
source expert prior. The source prior, cached adapted base, gate coefficients,
observed-class restriction, positive gate threshold, tie order and 5% cap stay
unchanged. One penalized bias is shared by all feature branches.

Four fixed arms, with Kp/Kr/Kd denoting bias-free retained/raw/discarded kernels:

- `pca_reference`: Kp + 1, replay of the existing model.
- `raw_joint`: Kr + 1, replace only the residual correction representation.
- `retained_half_control`: 0.5 Kp + 1, change branch strength without new info.
- `projection_residual`: 0.5 Kp + 0.5 Kd + 1, fixed equal-weight complement.

The kernel formulation is exactly weighted residual ridge with strength8; it
keeps the solve at64x64. It is not a new nonlinear kernel or attention layer.
No extra target labels, target metadata, geometry missingness, AIS, backbone
training or gate refitting. Hyperparameters were fixed before pilot readout.
All three datasets' policies were saved before reading their query outcomes.

## Full Model Results

Values are mean percentages over the exact five historical support draws.
Each draw has64 target labels. All results use the corrected F0921 inputs or
the existing fine-compatible public OSS cohorts.

| Dataset | Arm | OA | BA |
| --- | --- | ---: | ---: |
| F0921 | Reference | 29.633543 | 25.105149 |
| F0921 | Raw joint | 29.739032 | 24.994089 |
| F0921 | Half-retained control | 29.573911 | 25.303587 |
| F0921 | Projection residual | **29.828803** | 25.003274 |
| Public OSS1 | Reference | 43.066148 | 32.330577 |
| Public OSS1 | Raw joint | 42.879377 | 32.250331 |
| Public OSS1 | Half-retained control | 42.474708 | 32.075349 |
| Public OSS1 | Projection residual | **43.284047** | 31.970674 |
| Public OSS2 | Reference | 32.382505 | 29.938364 |
| Public OSS2 | Raw joint | 32.165173 | 29.851440 |
| Public OSS2 | Half-retained control | 31.991307 | 29.750051 |
| Public OSS2 | Projection residual | **32.442271** | **29.971866** |

Projection-residual changes relative to reference:

| Dataset | Delta OA pp | OA-positive draws | Delta BA pp | BA-positive draws |
| --- | ---: | ---: | ---: | ---: |
| F0921 | +0.195260 | 3/5 | -0.101876 | 2/5 |
| Public OSS1 | +0.217899 | 5/5 | -0.359902 | 0/5 |
| Public OSS2 | +0.059766 | 3/5 | +0.033502 | 3/5 |

These are useful measured OA gains under the user's pragmatic adoption policy.
They are small and do not justify claims of uniform improvement or significance.

## The Expert Genuinely Uses The Added Information

Evaluating the expert on all query rows is diagnostic, NOT the budgeted result:

| Dataset | Reference expert OA / BA | Complement expert OA / BA | Expert OA delta pp |
| --- | ---: | ---: | ---: |
| F0921 | 31.054811 / 25.982106 | 32.809074 / 25.834987 | +1.754264 |
| Public OSS1 | 50.817121 / 30.551436 | 53.136187 / 29.924705 | +2.319066 |
| Public OSS2 | 34.947025 / 30.909838 | 36.761750 / 31.384500 | +1.814724 |

Expert OA improves in all five draws on EACH dataset. Thus this branch has an
observed expert-level benefit, unlike the preceding spatial-attention pilots.
Expert BA changes -0.147118/-0.626731/+0.474662 pp respectively: the difficulty
is not only handing improved predictions to the gate; class balance is also
an unresolved expert-level issue.

Relative to half-retained control, adding the complement raises full OA by
+0.254892/+0.809339/+0.450964 pp on F0921/OSS1/OSS2. Corresponding BA changes
are -0.300313/-0.104675/+0.221815 pp. The OA benefit is not explained solely
by weakening the old branch. No random-direction or universal novelty claim
is made, and the inherited gate may not be calibrated for the new expert.

All arms use the same cap: F0921 6171/123425 query rows, OSS1 64/1285, OSS2
184/3681. This is a handoff count, not a feature-computation budget. A stronger
all-query expert does not translate one-for-one into full-model improvement.

## Adoption And Follow-Up

Retain `projection_residual` as an OA-oriented candidate, with the complete
negative BA record. Do not replace F0921's aligned-global BA25.435555 candidate
with this BA25.003274 variant. The half-retained control is separately a cheap
F0921 BA candidate (+0.198438 pp, 4/5 draws), not a new information mechanism.
Raw replacement is not preferred: the complement arm has higher mean OA and
BA than raw replacement on F0921; on both public sets raw replacement loses
both mean metrics to the original reference.

A bounded follow-up attempted to compress the complement into eight source-
supervised discriminant directions. It failed on all three datasets' mean OA
and BA and is not enabled. See
`../source_discriminant_pilot_20261008/SOURCE_DISCRIMINANT_FINDINGS.md`.
Do not package this unsuccessful follow-up as another effective module.

## Verification And Scope

The kernel reference reproduces every cached full prediction AND admission
mask on all15 dataset/draw pairs. Maximum score differences: OSS1 9.23e-8,
OSS2 1.78e-7, F0921 3.30e-7. Supports, evaluation masks, row IDs and budget
accounting are checked in the actual runs. Focused unit tests cover weighted
ridge equivalence, discarded-space behavior, source-only statistics, support
order/exclusion, observed classes, gate ordering, zero-kernel behavior and
absence of query truth from the fit API.
The extended suite also checks source-only supervised projection and explicit
guided-kernel equivalence: 12 tests passed in the existing ClearSAR environment.

These remain historical development cohorts with row-level support draws,
not vessel-disjoint independent confirmation. The encoder has historical
F0921 unlabeled exposure. Public cohorts are fine-compatible subsets, not
standard full-dataset benchmarks. Source gate is inherited, not newly trained
on these expert outputs. The 24-port feature diagnosis is NOT a completed
24-port evaluation of this module; that has not been run in this pilot.

No main-model artifacts, manuscript performance tables or earlier results
were overwritten. `PROTOCOL.json`, `summary.json`, `metrics.csv` and per-dataset
policies record all arms, including regressions.
