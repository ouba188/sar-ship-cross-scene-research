# Independent Region Audit

Scope: native OSS2 localization, transformed pixels, cached score replay and metric indexing. CPU only. The auditor owns `audit_region_experiment.py` and `AUDIT*` files only.

## Mapping Evidence

- Native source: `D:/Documents/project_root/project_data/OpenSARShip_2/<scene>/Ship.xml`, `SARShipInformation/{Head_x,Head_y,Tail_x,Tail_y,Center_x,Center_y,UpperLeft_x,UpperLeft_y,LowerRight_x,LowerRight_y}`.
- Join using scene and numeric `Center_x,Center_y` parsed from `source_path_vv`; verify MMSI. Do not join only on MMSI or row positions from a different cohort.
- `D:/Documents/project_root/project_data/scripts/build_public_opensarship2.py` imports `Patch_Uint8` TIFF arrays directly to PNG without resize or rotation. Existing `public_metadata.csv` contains scene coordinates, not chip coordinates, and omits the four chip boundary fields.
- In the 3745-row experiment manifest, all widths satisfy `W=abs(UpperLeft_x-LowerRight_x)+1`; 3744 heights satisfy the analogous formula. The names are reversed numerically in 3744 records. Normalize each axis with min/max.
- Ordinary mapping: `x_chip=x_scene-min_x`, `y_chip=y_scene-min_y`, zero-based array indexing `[y_chip,x_chip]`. The XML-defined center matches `(W-1)/2,(H-1)/2` in those 3744 records, but this is a verified consequence, not a universal assumption.
- `pub2_opensarship2_003922`: native and PNG size 45x34; XML x=24414..24458, y=-10..34, center=(24436,12). Raw `Patch` and `Patch_Cal` are also 34x45 and have MATLAB software tags. A scene-boundary crop to rows 1..34 implies `(x0,y0)=(24414,1)` and head/tail `(21,7),(22,18)`. This origin is an explicit inference from extent and producer evidence; upstream MATLAB crop code was not located. Parent excludes this record from every regional metric scope, preserving its original image.
- `pub2_opensarship2_006067`: 29x29, origin=(15528,12582), head/tail=(6,-1),(11,6); interference=1 and multi_ship=1. The head is outside the native chip. Mark invalid for strict endpoint-complete regions, rather than silently treating clipped endpoints as complete localization.
- `pub2_opensarship2_006393`: 1x1 with coincident head/tail=(-10,11) relative to chip. Invalid despite interference=0. No useful background sham exists.
- Example `pub2_opensarship2_000003`: 73x73, origin=(12648,10038), head/tail=(27,32),(45,39). XML file is in scene `S1A_IW_GRDH_1SDV_20150612T221710_20150612T221738_006346_0085A2_39EE`.

## Admissible Interpretation

Head/tail give integer image-axis localization with possible 180-degree head/tail ambiguity. Their padded axis-aligned bounding rectangle is a coarse annotated target region; it is not a hull segmentation. The XML upper/lower corners bound the entire chip, not the vessel. No native ship segmentation masks were found in this local release. AIS width and length are physical metadata and do not supply image-space hull boundaries.

Padding `max(3 px, 0.20*headtail_length)` and sensitivity `0.35*headtail_length` are diagnostic region choices. They include background and possibly other vessels; pixels outside them are only the complementary region, not guaranteed pure background. The same pixel padding has different physical meaning on GRDH and anisotropic SLC. A shifted sham must preserve the actual clipped rectangle's integer width/height and have zero overlap; otherwise mark sham invalid. Compare target and sham on the same sham-valid query set.

## Verification Status

PASS within the bounded scope below; no outstanding actionable implementation finding in the checked paths. No parent files were edited, and no GPU or model fitting was used.

- Full metadata: all 3745 native XML joins and all 7490 PNG sizes checked. Strict region-valid records are 3742, excluding the three named records. The metadata receipt was updated from the earlier endpoint-only count of 3743 to this strict rule after the independent all-row pixel/mapping pass.
- Full region geometry: all 3745 rows checked against native XML; default/wide sham-valid counts 2976/2391. Same integer rectangle dimensions, equal area and zero overlap verified; all default saved region coordinates checked. Mask interpretation remains a coarse annotated target region.
- Native pixels: 24 predetermined product/class examples, size extremes and the three exceptions, all 11 conditions, 264 checks. Unedited pixels preserved exactly, flat values match border-median fill, shuffling preserves joint VV/VH pixel pairs, invalid rows and replay_original preserve all pixels. Display checks cover the edit boundary, not an independent full photometric reconstruction. See `AUDIT_pixels.json` for exact IDs.
- Independent metric recomputation: fold 2, draws 0 and 4, original/target_flat/outer_flat/target_shuffle/target_display/replay_original, all six scopes and all six paths. 432 path-effect, 144 utility and 3456 class rows passed; maximum absolute metric error 1.784683512084939e-14. All rows of the three context-mean summary tables match aggregation of their detail tables. Other condition/fold/draw metric cells were not independently recomputed.
- Old-cache raw baseline: both folds and both R/T arms, maximum probability error 8.940696716308594e-08, predictions identical. Replay-original negative control is independently checked on fold 2, draws 0/4, all six paths, with no prediction changes. The parent's wider negative-control result is not presented as independently recomputed here.
- Actual query denominator: each of five draws has 3681 row-disjoint queries before validity filtering and 3678 after it. All three invalid records are absent from every support set. This side experiment does not use the main task's 3440-query partition. Each scope computes BA over the classes present in that scope; class-wise and rescue/harm recomputation agrees.
- The extractor reencodes invalid rows, but a direct audit of the resulting caches confirms all three invalid raw/extra rows are exactly equal to original caches for all ten extracted conditions. The earlier potential drift concern did not materialize in the actual outputs.

Run with `D:/Program_files/anaconda3/python.exe -B audit_region_experiment.py metadata`, `pixels` or `scores`. Receipts are `AUDIT_metadata.json`, `AUDIT_pixels.json` and `AUDIT_scores.json`. The auditor deliberately does not import the parent's model/evaluator modules; their common-module name collision cannot affect these independent checks.
