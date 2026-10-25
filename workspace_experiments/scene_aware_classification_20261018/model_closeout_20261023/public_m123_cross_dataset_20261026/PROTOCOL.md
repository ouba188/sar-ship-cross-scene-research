# Frozen OSS1 / OSS2 transfer

Completion update (2026-10-26): the main task has completed both full-cohort
directions and OSS2 source replay, followed by paired/group analysis and actual
native-chip/reference review. Results and a separately recorded upstream
normalization diagnostic are in FINDINGS_ZH.md. The implementation-handoff
statements below describe the earlier subtask scope, not pending experiments.

`transfer.py` is inference only. It imports the adjacent OSS1 algorithms without
calling training, scale fitting, PCA fitting, clustering or model selection.
The sole authorized shared-file change is
`extract_frontend.extract_summaries(..., frozen_tau=None)`. Its default still
fits source tau exactly as before; a supplied tau bypasses the median fit, writes
`temperature_fit_role=frozen_source`, an empty local `fit_obs_id`, and the source
scales path. The transfer protocol additionally records the actual source tau
and source fit observation IDs.

## Inputs and scope

- `--source-root`: source replication directory containing `stack_table.csv`
  and `frontend/`. `--source-frontend` can explicitly override the latter.
- `--source-m3`: complete source M3 directory (`m3_results_grid01` for OSS1;
  `m3_results` for the new OSS2 run). It must contain `table.csv`, `projection.npz`,
  `local.npz`, `references.npz`, and `seed17/29/43` checkpoints.
- `--target-root`: raw inventory/cache root. Default cohort is every row of
  `stack_table.csv`, regardless of its original role. Use `--full-table` to
  replace this with the prepared complete cross-dataset cohort. This is essential
  for the intended 1619 / 921 query coverage; the historical stack alone is smaller.
- `--view-inventory` and `--geometry` are mandatory explicit target paths. There
  is no implicit OSS1 geometry fallback. `--target-cache` defaults to the target
  root's `frontend_cache`. Missing raw-cache observations alone are extracted
  under the output directory, then merged in cohort order. Existing caches are
  read-only. If no complete pair of raw caches exists, the requested cohort is
  extracted there. These caches must use the declared native160/C4 pipeline,
  the supplied inventory/geometry, and the original ImageNet R50 V2 weights.
- `--out` must be an empty subdirectory of this branch. Default seeds are all
  three source seeds (17, 29, 43). Default device is CUDA; source replay below is CPU.

The prepared target cohort must use fixed class IDs
`bulk=0, general=1, chemical=2, container=3`, corresponding to original IDs
`0,2,3,4`. Class membership is declared by the data preparation step, not inferred
or adjusted from model output. Main inference excludes class/label fields,
including duplicate `y`, from its target metadata read. All requested seeds and
arms are saved before target `class_id` is read for metrics. The separately
requested cohort audit in `verify_core.py --check-cohorts` checks this mapping;
it neither runs inference nor selects any parameter.

## Frozen computation

1. Reconstruct the source baseline from axis Ridge, Five, Neighborhood and
   Calibration checkpoints and their saved normalizers. Reuse their original
   arithmetic precision and batching.
2. For actual transfer, run target raw chips through the original C4 backbone
   and **source** `reader_layer1/3.npz`. Reuse the **source** `scales.json` tau.
   Never reuse target-trained class-dependent summaries. Reuse only the
   target's reader-independent raw cache.
3. Apply source M1 and M2 residual checkpoints and their saved scalers to obtain
   `s12`. Project target raw M3 moments with source `projection.npz`.
4. Use exactly the saved 32 source references (8 per candidate class, distinct
   fit_train MMSIs). Retrieve two per candidate class with the original sort
   rule, independent of the query label. No target references are selected.
5. Reuse the source local metric, transport implementation, joint head, polar
   heads, scalers, B alpha, spatial gate and frequency gate. Emit all four arms:
   `old_both`, `B_fixed`, `spatial`, `frequency`. Missing polarizations preserve
   the source B anchor and availability mask, as in the original model.

No target scaler, PCA, reader, tau, gate, reference bank, checkpoint, threshold
or hyperparameter is fitted or selected. The spatial/frequency arms use the
same source alpha; no target result chooses between them.

## Cohort separation

Before inference, independently exclude any target MMSI or physical acquisition
overlapping source `fit_train`, `fit_validation`, `utility_train` or `selection`.
The exclusion uses the entire source table, not the reference subset. Source
`evaluation` is not treated as supervision. Source stack and M3 table identity,
role and class alignment must agree. Physical pass keys are recovered from the
Sentinel product's satellite/orbit/datatake fields, with canonical overpass keys
as fallback; dataset prefixes do not create artificial separation.

At least one present OBB with confidence >= .25 must be usable. If inventory
`valid_target` is available, it is also required. An invalid independent
`axis_valid` flag alone does not remove an accepted OBB: the original preparation
derives geometry and an axis from its corners. No label-dependent sampling,
balancing, target-role restriction or performance-based exclusion is applied.

Independent prepared-cohort check on 2026-10-26:

| Direction | Queries | Ships | Physical passes | Source supervised rows |
| --- | ---: | ---: | ---: | ---: |
| OSS1 to OSS2 | 1619 | 1347 | 27 | 803 |
| OSS2 to OSS1 | 921 | 742 | 21 | 1141 |

Both had zero additional exclusions and correct original-to-four-class mapping.
These are historically exposed public datasets, not a new blind test. These
counts are inventory checks, not transfer performance results.

## Output contract for analysis

- `table.csv`: full source table first, then selected target queries; `obs_id`
  and `row_index` are unique combined indices. `original_obs_id`, `dataset`,
  `dataset_side`, `dataset_row`, and `original_role` retain provenance. Restore
  `original_obs_id` before joining the original view inventory. Target rows get
  role `evaluation`; target labels are filled only after prediction saving.
- `target_table.csv`: prediction-row order, original target `obs_id`,
  `query_row=0..N-1`, `combined_row`, and canonical `physical_pass_group`.
  `class_id` is added only after all prediction files are saved.
- `source_bank.csv`: 32 rows, `bank_index` in `[0,31]`, `combined_row` pointing
  to the corresponding source row in `table.csv`, and original source IDs.
- `seed*/predictions.npz`: arrays are **target-row aligned**, not combined-table
  length. `query_rows` points into `table.csv`; `refs[N,4,2]` points only to source
  rows of that same table. `refs_bank` is the equivalent compact bank index.
  Included scores: all four arms, `base` (float32 M1+M2 input to M3),
  `frontend_base`, `m1`, `s12`, `axis`, `A`, `U`, `delta_m1`, `delta_m2`.
  Included M3 evidence: `delta_vv`, `delta_vh`, `local`, `gate_raw`, `available`,
  `weight_B_fixed`, `weight_spatial`, `weight_frequency`. `old_both` is a joint
  head, so it has no polar-fusion weight array.
- `protocol.json`: source paths, target input paths, tau provenance, seeds,
  separation mode, and mappings. `target_labels_read=false` records the state
  at inference setup. `predictions_saved.json` records the save barrier before
  evaluation. `complete.json` records whether post-save evaluation completed.
- `target_eligibility.csv`: independent overlap and availability decisions.
  `metrics.csv` and `summary.csv`: post-save target metrics only, no arm selection.
  BA is the mean recall over classes present in the evaluated cohort, matching
  the original implementation.

The main agent's `analyze_transfer.py` consumes these mappings directly and owns
paired/pass statistics and native VV/VH query plus actual-reference galleries.
This branch does not fabricate gallery review or physical mechanism conclusions.

## Source replay and verification

`--validate-source-replay` requires `target-root=source-root` and no full-table
override. It bypasses cross-domain exclusions, reads existing source raw caches
and **existing source summaries**, and never extracts raw images. It recomputes
the entire frozen frontend and M3 for every source observation, then compares
source saved scores, local features, projections, deltas, weights and references.
No evaluation metrics are produced in replay mode. It must never be presented
as cross-dataset validation.

References and predicted classes must match exactly. Numeric score tolerance is
absolute `2e-5` with zero relative tolerance; each checked array also reports
its actual maximum error and bitwise equality. CPU/GPU batching may change low
bits, so numerical parity is not described as bitwise identity.

`verify_core.py` checks all-supervised-role exclusion, physical-pass naming,
ID collisions, target-label isolation/save order, frozen tau with median fitting
explicitly forbidden, partial-cache merge order, and 6 representative source
rows across all 3 seeds. Training helpers are patched to raise during the cached
checkpoint check. It also verifies missing-polarization gate behavior. The
frozen-tau extraction check uses synthetic empty views on CPU; it is an interface
test, not a new real-chip extraction claim. Actual cross-domain summary extraction
provides the subsequent real-image use of that interface.

Full OSS1 CPU cache replay completed for all 866 observations and seeds 17/29/43.
Reference indices and predicted classes matched exactly. Maximum absolute error
across the compared arrays was 5.245208740234375e-6, 8.58306884765625e-6 and
5.4836273193359375e-6 respectively. These are numerical parity results, not
bitwise identity. No new raw-image extraction was used for this replay.

No training, GPU extraction, Git operation or cross-dataset performance run was
performed by this implementation task. See `core_validation/checks.json` and
`source_replay_oss1/source_replay.json` for actual verification results. Only
small JSON test receipts are retained; regenerable replay NPZ/CSV files were
removed at handoff. `predictions_saved.json` records that predictions were saved
during that successful validation run, not that its temporary arrays remain.
Formal transfer outputs are unaffected by this test cleanup. Run commands are
in `RUN_COMMANDS.ps1`.

## Handoff commands

Run after OSS2 frontend and M3 source checkpoints are complete, and when the
main agent's GPU extraction has finished:

```powershell
cd 'E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/public_m123_cross_dataset_20261026'
./RUN_COMMANDS.ps1 -Operation oss1-to-oss2
./RUN_COMMANDS.ps1 -Operation oss2-to-oss1
```

The wrapper uses the two full-cohort tables under the OSS2 root and the correct
explicit inventory/geometry for each target. Output folders default to the
operation name. Use `-OutputName <new-name>` for a deliberate fresh run.
The underlying CLI is equivalent to:

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -B ./transfer.py `
  --source-root <source-root> --source-m3 <source-m3> `
  --target-root <target-root> --full-table <prepared-cross-table> `
  --view-inventory <target-view-inventory.csv> --geometry <target-geometry.csv> `
  --out <empty-output-directory> --device cuda
```

Optional source cache replay is `./RUN_COMMANDS.ps1 -Operation replay-oss2`.
It uses CPU and never re-extracts source images. The main agent owns subsequent
`analyze_transfer.py` execution and visual review. No additional validation is
required by this handoff beyond the main agent's planned integration run.
