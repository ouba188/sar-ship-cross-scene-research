# Current source -> public OpenSARShip2: completed 2026-10-07

## Result

The frozen same-64, 5%-handoff method has a positive but modest additional
benefit on the label-compatible OpenSARShip2 cohort. Do not attribute the
whole adaptation gain to expert handoff.

| Fixed policy | Mean balanced accuracy (%) | Mean accuracy (%) |
| --- | ---: | ---: |
| Original source base | 19.457723 | 18.565607 |
| Base adapted with 64 target labels | 29.295273 | 30.638413 |
| Same-64 adapted branches + 5% expert handoff | 29.938364 | 32.382505 |

- Base adaptation over the original base: +9.837550 BA percentage points.
- Handoff over the adapted base: +0.643091 BA points; +1.744091 accuracy points.
- Full method over the original base: +10.480641 BA points.
- Handoff BA increments for seeds 0..4: +0.256231, +0.541346, +1.092434,
  +0.894763, +0.430681 points. All five are positive.
- Every draw admits 184 / 3,681 evaluation rows (4.998642%). Both branches
  share the same 64 labels; no second set of expert labels is purchased.

All five policies were saved before query readout. No target tuning, new
modules, failed-draw removal, or post-result parameter changes were used.

## Data boundary

Native manifest: 9,120 rows. Explicit fine-label alignment retains 3,745
rows: bulk 868, fishing 490, general cargo 316, product/chemical tanker 818,
container 973, crude-oil tanker 149, tug 76, offshore supply 55.
The other 5,375 rows are outside this compatible fine-label cohort, NOT model
errors removed from an eight-class test. Quality/interference filtering is
not applied; 483 interference-positive included rows remain.

Source training removes 1,095 rows sharing 473 target vessel identifiers,
leaving 36,996 rows across 24 ports. No source scene identifiers match the
target manifest. Source PCA, classifiers, and all 24 OOF folds/gate were refit
after this exclusion. The existing image encoders were not retrained.

Target support/query follows the current row-level protocol. Query rows
sharing a support vessel number 58, 50, 59, 64, 53 across the five draws.
This is NOT vessel-disjoint target adaptation. Some draws lack tug or
offshore support labels; the original observed-class guard remains unchanged.

## Model correction and cache reuse

The current source base is paired VV/VH through a TWO-channel ImageNet R50
V2 (2048D), not VV-only. The expert adds VH through a frozen domain-SSL R18
e206 epoch40 (512D). This corrects previous simplified model descriptions;
the original main experiment results were not changed.

The supplied external_os2_fusar/feats_os2.npz is valid for its old experiment,
but is not interchangeable with the present base: it uses VV RGB through a
standard R50 and an interference0/GRDH/shared-three cohort of 5,006 chips.
Its SSL preprocessing also forces square images. Preserve it as historical
evidence; this run extracts the actual current representation for all 3,745
eligible chips. Non-square SSL chips retain aspect ratio via shape grouping.

Eight fixed source-row encoder replays have paired-R50 relative L2 errors
0.005804..0.007682 and SSL errors 0.000001187..0.000002575. These are recorded
numerical discrepancies, not bitwise reproduction. The recipe is traced to
the original producer; no checkpoint was trained on this target experiment.
The SSL checkpoint itself was pretrained on E:/F_0921_pool_v1, so later
F_0921 experiments must disclose this encoder exposure.

## Interpretation

This extends the current mechanism from within-corpus cross-port development
to a frozen cross-dataset transfer protocol. It does not establish universal
improvement: mean crude-oil-tanker and offshore recalls decline under handoff
by about 3.15 and 0.38 points, despite positive overall BA in all draws.

Absolute BA remains only 29.94%; domain shift is not solved. Do not directly
compare that number with the old 23-port macro mean, since datasets and
aggregation differ. The incremental handoff comparison is the relevant
within-run evidence. Five overlapping support draws are descriptive
repetitions, not five independent datasets or a statistical significance test.

Public OpenSARShip2 has prior historical experiments (e181 and the supplied
external_os2_fusar run). Call this cross-dataset transfer, not untouched blind
external confirmation. The 5% cap concerns expert admissions; all expert
scores are available before selection, so it is not an inference-cost result.

## Artifacts and verification

- TRANSFER_PROTOCOL.md: frozen settings and intended interpretation.
- summary.json, metrics.json/csv: complete three-policy readout.
- policies.npz: all five supports, evaluation masks, corrections, scores,
  predictions and handoff masks, stored before query scoring.
- source_split.json/npz, source_oof/, source_oof.npz, source_model.pkl:
  identity-purged source fits and frozen gate.
- label_mapping.csv, label_mapping_summary.json, target_manifest.csv:
  inclusion/exclusion provenance and retained target order.
- features.npz, feature_recipe.json, encoder_replay.json: current embeddings.
- AVAILABLE_FEATURES.md: user-supplied inventory and compatibility status.

Label bridge: 42 focused pytest cases passed in the delegated implementation.
Handoff integration: three unittest cases passed (support exclusion and exact
cap/changes, absent-class guard, rejection of query labels in the support API).
The full run completed all 24 source folds and all five target draws.

Commands (use the GPU interpreter for extraction and ClearSAR for fitting):

```powershell
& 'E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe' 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/extract_opensarship2_transfer.py' --out <new-output>
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_opensarship2_transfer.py' --out <same-output>
```

The completed result directory is protected against accidental rerun overwrite.
No additional dataset experiments were launched from the newly supplied list.
