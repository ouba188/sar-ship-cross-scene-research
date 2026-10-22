# Expanded context readout

Completed 2026-10-22. All changes and outputs are confined to this readout directory. No training, dependency installation, Git operations, or edits to the parent training script were performed.

## Ready for training

- `targets.csv` and `features.npz`: 5,695 aligned labeled targets; 178 products, 163 acquisition/pass groups, 15 ports, 4,312 MMSI.
- Classes: bulk_carrier 3,170; fishing_vessel 1,147; general_cargo 1,378. All source fields and supplied labels/identities are preserved.
- `neighbors.csv`: 2,397 edges for 1,165 targets, with explicit target index, slot, edge index, detection ID, windows and OBB geometry.
- 6,134 linked targets are fully accounted for: 5,695 retained and 439 excluded because the nearest eligible VV detection is farther than 5 pixels from the recorded VV window center. No threshold relaxation or relabeling was performed.
- Missing MMSI: zero. Cross-class MMSI conflicts in the linked input: zero. The conflict report is still saved; no identity or label cleanup was applied.

## Features and checks

`target`, `target_fg`, `target_bg` are N x 256. `neighbor`, `neighbor_fg`, `neighbor_bg` are N x 16 x 256. `geometry` is N x 16 x 7, with node mask, combined foreground/background validity, additional per-polarization validity, instance IDs and edge indices. Combined validity requires both polarizations. This cohort has no invalid target or retained neighbor FG/BG regions.

Frozen ImageNet R50 conv1/bn1/relu, mean/max pooling and four exact quarter-turns are reused from the old code. VV and VH retain their independently recorded native target windows. A common full-image OBB is mapped to each window independently. Only feature-map pooling weights are masked; input pixels are unchanged. Background is outside a square radius-3 dilation.

Persisted arrays are nonempty and finite. All IDs, slots and geometry align with the tables. All source labels and identities are unchanged. All linked IDs are represented either in the target table or target exclusions. Every retained neighbor center is outside the target VV window and within 1 km. The largest retained radius is 999.810759 m. Both actual polarization windows have zero intersection area with the target OBB. Planned windows are checked before neighbor reading and asserted equal to the actual `crop_world` read windows.

The old filters remain: VV confidence >= 0.25, valid pixels, full-image coordinates, WorldCover land-interior exclusion, target center match <= 5 pixels and at most 16 nearest valid neighbors. `excluded.csv` records catalog rejection reasons and each rejected target/neighbor relation. In this run, 435 centers were inside target windows and 147 candidate read windows intersected target OBBs; neither group was used as context. No eligible candidate was removed by the land-interior or max16 rule in this cohort.

`whole_replay.csv` compares only identical raster/native-window inputs: 61 targets and 85 neighbor edges exactly reproduce the previous whole vectors, maximum absolute error 0. Changed target windows are outside that replay claim. New windows have been extracted, not claimed to match old crops.

## Runtime and recovery

178 product shards completed in a summed 172.766912 seconds. There were 2,101 unique neighbor encodes and 296 cache hits across targets within products. The final resume/merge took 8.546395 seconds; these timings exclude code preparation and visual review.

The first final merge stopped at a provenance check because `set_index('instance_id')` removed the column subsequently being checked. This was corrected with `drop=False`. All existing product shards were reused; no pixel extraction or GPU encoding was repeated. Final verification and replay completed successfully. Per-product completion receipts and the original failure log remain available.

## Native image review

Actually inspected using `view_image`:

- Melbourne `pair_0030918`: VV 64 x 64 and VH 72 x 72; common OBB follows the target across different windows.
- Richards Bay `pair_0065191`: VV 104 x 104 and VH 96 x 96; separate window offsets correctly preserve OBB placement.
- Mombasa `pair_0031576`, Port Said `pair_0055988`, Qingdao `pair_0059313`, Richards Bay `pair_0065308`: target and nearest retained neighbor, both polarizations and foreground/background pooling footprints. Cases were selected by new port and deterministic ID order, without predictions. Exact paths, windows and detection IDs are in `context_review_cases.csv`.

Visible boundaries: Mombasa chips contain additional bright objects outside the selected OBB; these remain in the whole/background readout by design. Qingdao VV/VH target windows have a one-pixel horizontal offset and the displayed common OBB follows that offset. Richards Bay `pair_0065308` has bright response extending beyond the fitted target OBB, and its neighbor has a much weaker VH response. These observations do not establish physical causes or detection truth. Conv1 has a 7 x 7 receptive field, so FG/BG readouts are not pure physical component separation. WorldCover is a static 2021 external map.

## Read-only training review

Reviewed `../train_expanded.py`, plus its imported `Relation` and `Baseline` implementations, for actual label leakage, split errors and source OOF mistakes. No actionable errors were found in the reviewed implementation and current data.

The real `split_data` function was executed with its output directory redirected to `readout/training_review`, without editing the parent file or fitting any model. The six outer folds have nonempty train/validation/test roles covering all three classes, zero acquisition or MMSI overlap between roles, and exactly one outer test assignment per target. All 24 source OOF splits exclude both held bins and their MMSI, cover each source training target exactly once, and retain all three fit classes; minimum inner fit size is 1,596. Each source role projection has at least 32 unique valid fitting nodes.

Static review confirms that baseline preprocessing/ridge, geometry normalization and dedicated regional projection fitting use outer train data; alpha and epoch choices use source validation; test labels are only used for reporting. Source OOF scores come from refitted inner baselines. Query and regional representations use outer-source projections, so the implementation is score-level OOF rather than a fully inner-cross-fitted representation pipeline. No optimization or saved classifier score replay was run in this branch.

Evidence: `summary.json`, `verification.json`, `whole_replay_summary.json`, `training_review/checks.json`, coverage CSVs and review panels. `native_targets/`, feature archives and product shards are local data artifacts, not Git deliverables.
