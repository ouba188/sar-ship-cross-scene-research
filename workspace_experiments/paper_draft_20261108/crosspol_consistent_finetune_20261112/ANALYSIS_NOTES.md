# CPU analysis handoff

Run `python -B analyze.py` only after the main trainer has produced all A/B/C/D
validation/evaluation NPZs for seeds 17/29/43. No status files or trainer imports
are needed. The default output is `analysis/`; `--out PATH` overrides it.

- A: `../full_finetune_20261111/selected_s{seed}/{split}.npz`, score/label/ids.
- B: `../crosspol_box_recovery_20261112/{split}_s{seed}.npz`, recovered/label/ids.
- C: `frozen_restored_s{seed}/{split}.npz`, score/label/ids.
- D: `selected_s{seed}/{split}.npz`, score/label/ids.

IDs are original manifest indices, aligned before comparison. Split names are
validation/evaluation. Expected cohorts remain source719/val228/test378/ref32.
Helpers come directly from `../r50_tuned_20261111/analyze.py`, recovery `pilot.py`
and `sar_m123.input`; no full-analysis wrapper or training code is imported.

D-C is primary: identical restored query/reference inputs, with matched training
recipe and staged initialization as specified by the main trainer. D-A is the
whole-process comparison and also changes inference inputs. C-B and B-A are
secondary paired tables. Prediction files cannot verify matched initialization
or RNG. Selection and early stopping can produce different realized budgets.

Outputs include OA/BA/macro-F1, paired seed deltas/outcome counts, class metrics,
original-availability and scene/pass/orbit aggregates, paired MMSI bootstrap
D-C/D-A intervals, and selected A/D receipt budget tables. No hypothesis tests.
`data_scope.csv` records original missing VV/VH by split/class; references are
part of source719, not additional independent training examples. Missing totals
are 48/22/30/5 for source/validation/test/references. Test missing30 is a protocol
group; neither scene tables nor class coverage establish full-scene or causal
claims. Runtime `ANALYSIS_NOTES.md` contains full metric and bootstrap definitions.

Defaults: `--bootstrap 2000 --bootstrap-seed 20261112 --panels 12`.
`--panels` accepts 8-12 D-C cases, balancing outcomes/classes/original availability
when feasible. Absent outcome types are explicitly counted and filled with real
stable cases. `--include-d-vs-a` adds four cases with original A inputs above
restored D inputs. Both polarizations use the existing draw_panel helper and
160x160 native crop/pad without resize. C/D references come from the actual legal
fixed32 bank, with missing views restored. Examples prioritize restored bank
members where available but do not claim highest contribution.

`cases_local.json`, `index.html` and `cases/` are PRIVATE local artifacts. They
contain raw native chips, case IDs, source paths, actual bank membership and
per-seed score/margin evidence. Aggregate CSVs, RESULTS.md and summary.json omit
raw paths and per-record mappings. No Git actions, training, inference, formal
edits or waiting are performed. Main owns post-training execution, visual review,
findings and any later scoped publication. Completion always marks review pending.
