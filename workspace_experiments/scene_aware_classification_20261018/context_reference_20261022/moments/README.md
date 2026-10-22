# Native background feature-map moments

Completed 2026-10-22. `moments.npz` is ready for the background-reference experiment. All implementation and generated files for this task are confined to this directory. The expanded readout remains frozen; no training, detection, PCA, dependency installation or Git operation was performed.

## Input and alignment

The input is the final `context_expansion_20261022/readout/{targets.csv,neighbors.csv,features.npz}`: 5,695 targets, 2,397 edges, 1,165 targets with context and 178 products. Target order, instance IDs, node masks and edge indices are unchanged. Labels, identities, splits, original feature arrays, target windows and neighbor windows are not edited.

All pixels come from bounded GDAL reads of the recorded original 8-bit VV/VH raster windows. No full scene is decoded. VV and VH retain their separate native windows. Detection source paths come from `data_image_review/ports24/fullscene_candidate_catalogs.csv`; `window_stats.csv` records the exact audit CSV, raster, window, detection ID and background counts used.

## Arrays

Polarization axis order is `[VV, VH]`; feature channels are the 64 frozen R50 stem output channels.

| Key | Shape | Meaning |
| --- | --- | --- |
| `target_mean`, `target_second` | 5695 x 2 x 64 | float32 first and raw second background moments |
| `neighbor_mean`, `neighbor_second` | 5695 x 16 x 2 x 64 | The same moments, aligned to frozen neighbor slots |
| `target_valid`, `target_bg_pixels` | 5695 x 2 | Per-polarization validity and native background pixel count |
| `neighbor_valid`, `neighbor_bg_pixels` | 5695 x 16 x 2 | Per-polarization validity and native background pixel count |
| `target_weight_sum` | 5695 x 2 x 4 | Feature-map area-weight sums for each exact rotation |
| `neighbor_weight_sum` | 5695 x 16 x 2 x 4 | The corresponding neighbor weight sums |
| `instance_ids`, `mask`, `edge_index` | Same as frozen input | Exact identity and edge alignment |
| `neighbor_cache_index` | 5695 x 16 | Row of `unique_neighbors.csv`, or -1 for padding |

`*_excluded_obb_count` records the number of intersecting candidate bounding boxes considered for mask construction, including the dilation fringe; it is not a count of independently verified ships.

For each window, all same-product VV detections with `conf >= 0.25` and `pixel_status == valid` are eligible. Bounding boxes intersecting the crop plus the dilation fringe are queried. Their OBB masks are rasterized on a padded local canvas, unioned and dilated by square/Chebyshev radius 3 native pixels, then cropped to the actual read window. BG is the complement. Thus adjacent OBBs and OBBs immediately outside the crop can also remove background pixels. No confidence or mask threshold was selected using predictions.

For each polarization and exact rotation k, let z be the conv1/bn1/relu map from the unchanged native image, and w be the rotated BG mask area-resampled onto that map. The saved quantities are:

`mean = (1/4) sum_k [sum(w_k * z_k) / sum(w_k)]`

`second = (1/4) sum_k [sum(w_k * z_k**2) / sum(w_k)]`

These are moments of feature-map pixels, not moments or variance of previously pooled node means. No original image pixels are zeroed or resized before the stem. A polarization is valid only with at least 64 native BG pixels and finite moments. Invalid results and padded neighbor slots are exact zeros with false flags.

## Coverage, cache and timing

All 5,695 targets and 2,397 active edges are valid in both polarizations. Minimum BG counts are 2,896/2,912 for target VV/VH and 2,507/2,507 for neighbor VV/VH. The fixed 64-pixel threshold was not changed.

2,101 unique product/detection/dual-window neighbor inputs were encoded; 296 repeated edges reuse cached numeric moments without repeated GPU encoding. `unique_neighbors.csv` records the keys. The first full invocation took 73.948092 seconds, including 69.805880 seconds summed across product extraction. Verification took 2.366074 seconds, excluding interpreter/import startup.

Relative to the frozen old own-OBB background counts, all-OBB exclusion reduces background pixels in 538 VV and 545 VH target windows, and 888 VV/VH neighbor edges. This describes changed evidence sampling, not a gain or a causal mechanism.

## Verification and native panels

`verification.json` confirms finite saved arrays, exact input IDs/masks/edge indices and exact equality on all 296 repeated cached edges. Four target cases and the specified neighbor were independently replayed in both polarizations. A reference rasterizes every eligible catalog OBB on a larger bounded canvas, without the bounding-box candidate filter; its BG masks exactly match extraction and saved PNG masks for all 10 polarization cases.

An explicit NumPy float64 weighted computation over stem feature-map pixels reproduces the saved moments: maximum absolute difference 2.439495e-6 for mean and 8.907870e-5 for raw second, within the declared float32 tolerances. The fixed threshold was exercised at 63 and 64 native pixels; nonfinite input returns zero/false. Details are in `moment_replay.csv`.

All five panels were actually opened with `view_image`, showing VV/VH native pixels, every eligible OBB, the actual dilated exclusion mask, and k=0 feature-map area weights:

- Mombasa `pair_0031576`: two bright detected objects; both legal OBBs are excluded. BG is 3,150 pixels in each polarization.
- Shanghai `pair_0069192`: the main detection is excluded; another visible bright spot remains outside the eligible catalog mask. No visual relabeling or new detection was introduced.
- Shanghai neighbor `det_01443` of `pair_0069192`: its OBB is excluded, but substantial bright shoreline/scene structure remains in the retained background. BG is 5,883 pixels in each polarization.
- Melbourne `pair_0030918`: VV/VH have different 64/72 native side lengths; masks follow each recorded window, with BG counts 3,478/4,566.
- Richards Bay `pair_0065308`: the recorded polarization offsets are preserved and the shared global OBB maps into both windows. BG counts are 6,859/6,859.

The sample IDs were fixed from earlier image review and the requested `01443` case, not selected using the new moments or classification outcomes. Exact source windows and excluded detection IDs are saved in `review_cases.csv`; native binary BG masks are retained beside the panels.

## Interpretation limits

Background means outside the eligible detection OBBs. It can include undetected objects, shoreline, clutter, sidelobe responses and other scene content; it is not pure sea or a physical component. A 7 x 7 convolutional receptive field mixes evidence near mask boundaries. The four rotations repeatedly observe the same native pixels and are not four independent samples. `second - mean**2`, if used downstream, is a feature-response dispersion that also combines rotation responses; it is not a named physical variance. No claim about the proposed 16-group correction's effectiveness is made here.
