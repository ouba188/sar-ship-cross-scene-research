# Analysis interpretation and interfaces

Run `python -B analyze.py` after all three seeds have both validation/evaluation
NPZs for A/B/C/D. Defaults: 2000 paired ship bootstrap draws, seed 20261112,
12 D-versus-C panels. `--panels 8` requests eight; `--include-d-vs-a` adds four
paired before/after panels. `--out PATH` changes the default ROOT/analysis output.
No training, inference, waiting, status-file access, formal edits, or Git actions.

A: original full fine-tuning, original inference inputs. B: A weights, recovered
queries only. C: A weights, recovered queries AND fixed32 references. D: matched
full/1e-4 fine-tuning from the SAME staged Full seed initialization as A, same
loss/RNG/schedule, with recovered training/reference/query inputs. D does not
continue A's fine-tuned checkpoint. These are the main trainer's protocol;
prediction arrays alone cannot establish initialization or RNG matching.

D-C isolates training/validation input organization under the same restored
inference inputs, subject to the matched recipe. Validation-selected epochs and
early stopping may differ; this is not a fixed-step optimization ablation.
D-A compares the complete process, including changed inference inputs. C-B
describes fixed-weight reference recovery; B-A describes query-only recovery.
Recovery can change pixels, masks, availability, geometry and reference relations.
The original absent view is an omitted input branch, not a zero-intensity SAR view.

data_scope.csv retains original missingness by split/class/polarization:
source719 / validation228 / test378 / fixed32 references; missing48/22/30/5.
References are a source subset, not 32 extra independent training observations.
The test original-missing30 group is a protocol-defined group, not all scene
ships, a detector recall denominator, or a causal missing-view experiment.
Class coverage differences offer explanations to investigate, not causal proof.
Scene/pass/orbit tables concern only the included manifest observations; they
cannot establish full-scene, unseen-port or all-vessel performance.

Metrics are percentages, differences percentage points. BA averages truth-present
classes; macro-F1 uses all four labels and zero-division=0. Absent-class recall
and precision with no predicted positives are blank. SD is sample SD over three
seeds; seeds share data and are not independent scenes. Bootstrap resamples MMSI
clusters with all observations retained, pairs arms and the same fixed three
seeds in each draw, then averages seed deltas. BA draws losing an originally
present class are excluded and valid counts recorded. Intervals condition on
these trained runs and scenes, excluding model-selection/new-scene uncertainty.
No hypothesis tests or p-values. Validation is not independent confirmation;
the historical evaluation cohort is not a fresh blind test.

Gallery panels reuse hstats.draw_panel and sar_m123.input.prepare_view: VV/VH
are separate 160x160 native crop/pad views with scale1, no resize or enhancement.
C/D show identical recovered queries and actual recovered fixed32-bank members.
Legality uses different MMSI and platform/orbit; shared usable polarization is
recorded separately. Supports are examples, not highest-contribution claims.
All32 restored rows/audits, per-arm legal bank, selected IDs/paths, crop placement,
per-seed scores/margins and original availability are in cases_local.json.
If a rescue/harm type is absent, case_coverage.csv records zero; stable examples
fill the requested count. Class balance is best effort within available outcomes.
Stable-wrong includes wrong-to-wrong flips, separately counted in paired tables.
Cases are illustrative, not prevalence estimates; score scales can differ across
trained weights. No physical causes are inferred automatically. Main must inspect
the panels, verify actual evidence and write findings after training completes.

Training budget tables read selected A/D receipts when available; absent receipts
remain explicitly incomplete. They count three selected continuations per arm,
not inherited staged training, A's unselected scouts, B/C inference, or preparation.
Query exposures=719*epochs are repeated presentations, not independent examples;
reference passes are excluded. Epoch0 selection is valid. Matched schedules need
not produce equal realized epochs/cost; these are recorded rather than assumed.

Aggregate CSVs, RESULTS.md, summary.json and these notes contain no raw chip paths
or per-record mappings. Keep cases_local.json, index.html and cases/ PRIVATE and
local; do not publish raw galleries, NPZs, weights or source mappings. No exports
are made automatically by this script. analysis_receipt.json marks visual review
pending, never completed research or verified training provenance.
