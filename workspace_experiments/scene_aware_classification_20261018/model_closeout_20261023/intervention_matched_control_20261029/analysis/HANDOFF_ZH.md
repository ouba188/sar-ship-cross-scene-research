# Matched intervention: CPU analysis handoff

Formal analysis completed after main-agent notification; process exited 0.
All outputs are under this analysis directory. No training, inference, GPU,
runner/public-source edits or Git operations were performed by this analysis.
Native gallery contains 8 unique queries and 16 separate VV/VH panels.
All cases and source receipts remain `not_reviewed`; main-agent image review is pending.

## Main conclusion

Guided improves expanded OA over matched CE + 0.1 KL control in all three order
seeds, but decreases BA and increases NLL in all three. This is an accuracy/class
tradeoff, not uniform improvement. Guided-before alone does not isolate guidance:
ordinary continuation already improves BA and probability losses substantially.
The principal attribution is guided-control, conditional on this matched design.

Both cohorts are historical, not blind tests. Seeds17/29/43 vary training order
with the same initialization, not independent initializations. Bootstrap intervals
condition on fixed fitted models; they are not independent-training uncertainty.
Class and port associations and prediction margins do not prove physical causes.

## Three conditions: mean +/- sample SD

OA/BA in percent. NLL uses natural logs; Brier sums over eight classes.

| Cohort | Condition | OA | BA | NLL | Brier |
|---|---|---:|---:|---:|---:|
| development160 | before | 41.041667 +/- 2.818281 | 33.199186 +/- 1.759171 | 2.664567 +/- 0.283742 | 0.893252 +/- 0.066502 |
| development160 | control | 40.208333 +/- 2.009094 | 33.283189 +/- 1.814038 | 2.231615 +/- 0.072852 | 0.855070 +/- 0.006755 |
| development160 | guided | 45.000000 +/- 1.082532 | 35.733663 +/- 0.740113 | 2.430989 +/- 0.162197 | 0.840963 +/- 0.018059 |
| expanded3075 | before | 46.352304 +/- 1.122933 | 38.682940 +/- 1.856018 | 2.344226 +/- 0.076481 | 0.815594 +/- 0.001003 |
| expanded3075 | control | 45.127371 +/- 1.561427 | 41.039015 +/- 0.600579 | 1.939647 +/- 0.023102 | 0.783313 +/- 0.010681 |
| expanded3075 | guided | 47.317073 +/- 1.125128 | 40.402194 +/- 1.067667 | 2.137044 +/- 0.047298 | 0.786188 +/- 0.004741 |

## Guided-control: paired order-seed differences

OA/BA deltas in percentage points; positive NLL/Brier means worse.

| Cohort | Seed | OA | BA | NLL | Brier | Rescue / harm events |
|---|---:|---:|---:|---:|---:|---:|
| development | 17 | +6.250000 | +2.899351 | +0.245737 | -0.023023 | 13 / 3 |
| development | 29 | +3.125000 | +1.020408 | +0.094988 | -0.020864 | 8 / 3 |
| development | 43 | +5.000000 | +3.431664 | +0.257398 | +0.001566 | 9 / 1 |
| expanded | 17 | +1.918699 | -1.024539 | +0.218152 | +0.007348 | 167 / 108 |
| expanded | 29 | +2.731707 | -0.103982 | +0.138735 | -0.007753 | 181 / 97 |
| expanded | 43 | +1.918699 | -0.781942 | +0.235305 | +0.009032 | 177 / 118 |

| Cohort | Metric | Mean +/- SD | Whole-pass bootstrap 95% CI |
|---|---|---:|---:|
| development | OA | +4.791667 +/- 1.572882 | [3.174603, 6.763285] |
| development | BA | +2.450474 +/- 1.266750 | [1.875902, 3.246753] |
| development | NLL | +0.199374 +/- 0.090589 | [0.000482, 0.350547] |
| development | Brier | -0.014107 +/- 0.013616 | [-0.070953, 0.030271] |
| expanded | OA | +2.189702 +/- 0.469390 | [1.183688, 3.249427] |
| expanded | BA | -0.636821 +/- 0.477128 | [-1.606833, 0.366585] |
| expanded | NLL | +0.197397 +/- 0.051522 | [0.141813, 0.252697] |
| expanded | Brier | +0.002875 +/- 0.009243 | [-0.010469, 0.015555] |

2000 whole-pass resamples, bootstrap seed20261029. Development has only 3 passes;
expanded has 85. No resample omitted an originally present class. Development
interval resolution is limited by its three clusters. See NUMERIC_REPORT.md for
both after-before comparisons and their intervals.

## Class composition

Recall difference equals within-class OA difference. Absent development fishing
class is not included in development BA. R/H counts are repeated seed events.

| Class | Expanded n | Expanded recall delta +/- SD | Expanded R/H | Development n | Development recall delta |
|---|---:|---:|---:|---:|---:|
| bulk_carrier | 947 | +7.426962 +/- 0.703097 | 230 / 19 | 25 | +5.333333 |
| fishing | 102 | +1.307190 +/- 1.132059 | 8 / 4 | 0 | absent |
| general_cargo | 301 | +3.875969 +/- 2.212069 | 58 / 23 | 12 | +2.777778 |
| chemical_tanker | 803 | +2.407638 +/- 1.496125 | 175 / 117 | 70 | +8.095238 |
| container | 361 | -1.108033 +/- 2.366760 | 28 / 40 | 22 | +3.030303 |
| crude_oil_tanker | 256 | -1.041667 +/- 2.983448 | 23 / 31 | 16 | -2.083333 |
| tug | 138 | -4.589372 +/- 1.673479 | 3 / 22 | 7 | 0 |
| offshore | 167 | -13.373253 +/- 1.829372 | 0 / 67 | 8 | 0 |

Expanded bulk contributes +211 net seed events, chemical +58, general cargo +35;
offshore contributes -67 and tug -19. OA weights the larger classes while BA gives
each present class equal weight, explaining the measured OA/BA divergence.
Expanded offshore NLL delta is +2.450566 and tug +1.321943; development offshore
and tug also worsen in NLL (+2.157530/+1.433267) despite no top-1 changes.
This is a measured class-competition tradeoff, not a demonstrated cause in imagery.

Actual polarizations: both2876 has OA/BA delta +2.294854/-0.768780;
VV-only199 has +0.670017/-0.815738. Both groups have higher NLL.
Port differences include Richards Bay n195: +9.914530/+4.029487 OA/BA;
Los Angeles n103: -4.854369/-3.232630; Santos n239: +5.020921/-8.149100;
Port Klang n798: +2.255639/-0.158513; Jebel Ali n628: -0.424628/-0.113458.
These groups differ in composition; they are not matched causal port effects.
Full class/port/pass/polarization results are in groups.csv/groups_summary.csv.

## Before-to-continuation timelines

Control and guided are parallel continuations of before, NOT sequential stages.
The B/C/G ordering below is a display convention only.

| Cohort | Guided-control event | Before correct | Before wrong | Total events |
|---|---|---:|---:|---:|
| expanded | rescued | 455 | 70 | 525 |
| expanded | harmed | 21 | 302 | 323 |
| expanded | CC | 3550 | 290 | 3840 |
| expanded | WW | 250 | 4287 | 4537 |
| development | rescued | 20 | 10 | 30 |
| development | harmed | 2 | 5 | 7 |
| development | CC | 168 | 18 | 186 |
| development | WW | 7 | 250 | 257 |

Thus most expanded relative rescues preserve correct before predictions lost by
control (455); most relative harms miss corrections achieved by control (302).
There are also 70 before-wrong/control-wrong/guided-right events and 21
before-right/control-right/guided-wrong events. Expanded unique rescue/harm queries
are 376/246 (development22/6); IDs may change categories across seeds, so counts
must not be interpreted as mutually exclusive independent observations.

## Native cases for main-agent review

Gallery: [cases/index.html](cases/index.html). Every panel remains not_reviewed.
Margin means truth logit minus largest nontruth logit. Values below use the
selected representative seed; all seeds, logits and margins are in cases.json.

| ID | Guided-control state | Truth | Seed | Predictions B/C/G | Margins B/C/G | Interpretation from predictions only |
|---|---|---|---:|---|---|---|
| 7821 | rescue, all seeds | general_cargo | 43 | general_cargo / offshore / general_cargo | +4.1965 / -0.0395 / +3.7305 | preserves before-correct result |
| 25475 | rescue, all seeds | chemical_tanker | 43 | chemical_tanker / offshore / chemical_tanker | +0.3315 / -1.6909 / +0.3749 | preserves before-correct result; VV-only |
| 30214 | harm, all seeds | offshore | 43 | general_cargo / offshore / general_cargo | -3.6594 / +1.6304 / -2.9238 | misses control's new correction |
| 6180 | harm, all seeds | offshore | 43 | general_cargo / offshore / general_cargo | -3.9441 / +0.8491 / -3.0182 | misses control's new correction |
| 6183 | CC, all seeds | offshore | 17 | offshore / offshore / offshore | +0.7520 / +4.3027 / +1.4234 | both correct; guided margin weaker than control |
| 36026 | CC, all seeds | bulk_carrier | 17 | bulk_carrier / bulk_carrier / bulk_carrier | +12.4562 / +5.5009 / +9.9229 | both correct; guided margin stronger than control |
| 22726 | WW, all seeds | offshore | 17 | chemical_tanker / chemical_tanker / chemical_tanker | -27.2688 / -14.7703 / -24.8606 | no top-1 flip; guided evidence worse than control |
| 22750 | WW, all seeds | offshore | 43 | bulk_carrier / chemical_tanker / chemical_tanker | -13.8173 / -4.5277 / -11.4315 | wrong class changes from before; guided still worse margin |

7821/25475/30214/6180 have the same B/C/G prediction sequence in every seed.
6183 seed43 is general_cargo/offshore/offshore: both continuations rescue a
before error; seeds17/29 were already correct. 22750 seeds17/43 start bulk_carrier,
seed29 starts chemical_tanker; both continuations predict chemical_tanker throughout.

Actual same-class refs: general_cargo6913/22755, chemical_tanker24527/7790,
offshore13000/13276, bulk_carrier34478/31512. Panels also include the actual eligible
rival refs across all conditions/seeds, recorded in cases.json/native_sources.json.
Examples include paired general_cargo/offshore competition in7821 versus30214/6180.
No physical explanation or visual observation is asserted before main-agent review.

## Receipts and files

All six continuation receipts show 6 epochs/54 steps, frozen_unchanged=true and
save_replay_max_error=0. Control seeds all have selected_count=0, coefficient=0,
intervention_outcomes_used=false, original_teacher_reused=true. Guided selected
counts are101/109/102 and coefficient0.3683066666126251. Supplied
evaluation_check.json records native replay errors0 for all three control models,
development160/expanded3075 and default_changed=false.

Outputs: metrics.csv, paired_metrics.csv, groups.csv, groups_summary.csv,
seed_events.csv, cohort_groups.csv, cohort_overlap.csv, reference_pairs.csv,
summary.json, NUMERIC_REPORT.md, complete.json, this handoff and cases/.
The analysis consumed 6 new control and 12 reused before/guided NPZs. No default
model change or further experiment is performed or recommended as automatic work.
