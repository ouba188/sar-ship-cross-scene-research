# AIS Dimensions And Native Ground Grid

Input: `E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv`.
Output: `ais_metadata_sidecar.csv`, exactly 38,091 rows, one per input instance, preserving input order.
The earlier 2,713-image set was an experimental subset, not the full local-image inventory.

## Main Interface

- `global_row`, `instance_id`, `mmsi`, `product`, `port`: joins to the original manifest. `global_row` is zero-based.
- `ais_length_m`, `ais_width_m`, `ais_size_valid`, `ais_size_status`, `ais_size_source`: selected size reference and provenance.
- `ais_size_scope`: mandatory provenance stratum for dimension-agreement reporting.
- `pixel_x_m_vv`, `pixel_y_m_vv`, `pixel_x_m_vh`, `pixel_y_m_vh`: native PNG projected-ground grid sampling in metres, only when corresponding source georeference and original crop dimensions are verified.
- `georef_status`: `both_valid`, `vv_only`, `vh_only`, or `unavailable`. `ground_scale_status_vv/vh` gives the detailed reason.
- `scale_valid_vv/vh` and `chip_position_valid_vv/vh`: separate native linear-scale validity and absolute chip-georeference validity. Missing offsets alone do not invalidate scale when native no-resize is established. In this final result all recoverable offsets were found in original server manifests, so no offset-only gap remains.
- `affine_xx_vv`, `affine_xy_vv`, `affine_yx_vv`, `affine_yy_vv`, and matching `_vh` fields: linear transform from native chip pixel displacement to projected metre displacement.
- `epsg_vv/vh`, `georef_source_vv/vh`, `georef_header_path_vv/vh`, `georef_link_method_vv/vh`: actual header sources and association method.
- `source_transform_json_vv/vh`, `chip_transform_json_vv/vh`: GDAL-order transforms `[x0,xx,xy,y0,yx,yy]`.
- These JSON arrays are **not Rasterio order** `[a,b,c,d,e,f]`. To construct a Rasterio/Affine object use `[xx,xy,x0,yx,yy,y0]`. `transform_json_order` records the GDAL convention.
- `identity_link_quality`, `ais_record_match_quality`, `position_link_quality`, `position_methods_json`, `position_uncertainty_m_max`: distinct identity, record, and position evidence.
- `vessel_static_conflict`, `vessel_static_variants_json`, `same_product_static_variants_json`, `expected_imo_variants_json`, `same_product_imo_conflict`: conflicts remain explicit.
- `mmsi_valid`: 198 rows have empty MMSI; the correct distinct nonempty nine-digit MMSI count is 18,766. They remain in the 38,091-row sidecar with missing AIS dimensions.

For an OBB corner-to-corner pixel vector `(dx,dy)`, use
`dX=affine_xx*dx+affine_xy*dy`, `dY=affine_yx*dx+affine_yy*dy`,
then `sqrt(dX*dX+dY*dY)`. Transform the two adjacent OBB edges and sort their metric lengths.
Do not convert an axis-aligned enclosing box into ship length. Use the affine for the same polarization.
If detection was performed in a padded canvas, a translation cancels in edge vectors; scaling would require reversing that scale first.

## AIS Reference Semantics

Priority and `ais_size_status` values:

1. `valid_exact_record`: exact JSON object UID/product/MMSI, one object AIS timestamp across available views, and the same product/MMSI/timestamp in the raw 18-column export. Source file and one-based line are retained. Conflicting positions/motion or dimensions are not silently selected.
2. `valid_uid_metadata`: an exact UID/product/MMSI object whose `meta.source` is `raw_ais_review_only`; its raw timestamp record was unavailable. This is explicitly weaker than reopening the export row.
3. `valid_same_product_static`: all valid size pairs for that MMSI in the selected same-product export agree. This is a static attribute association, not an exact dynamic observation or position match.
4. `valid_cross_observation_static`: one consistent valid size pair across the inspected cohort-product raw/object evidence for the MMSI, with no conflicting available IMO. This is explicitly cross-observation static reuse, not dynamic record matching.
5. `missing_or_conflicting_static` or `conflicting_exact_evidence`: no chosen dimensions; do not fill from detector dimensions.

`ais_size_valid` means a positive, associated export/reference pair with length >= width under the stated provenance scope. It does not establish surveyed hull truth or correctness of the SAR-to-AIS target association. Analyze exact-record and static-reference strata separately. `ais_size_source` is a raw TXT path or JSON object reference; `ais_source_line` is one-based where present.

The local export stores length/beam directly in zero-based columns 14/15 (human columns 15/16). The checked export parser and JSON builder preserve these fields independently of the SAR detector: `register_sarais_txt_to_local_tiffs.py` reads them, `auto_v5_after_geocode.py` copies them, and `build_raw_ais_review_json.py` retains them in `meta.length_m/beam_m`. All are under `E:/临时会话/v5_server_toolkit_20260912/scripts/`.

No decomposed A/B/C/D fields were present in the inspected 18-column export or object metadata schemas. `ais_A_m/B_m/C_m/D_m` remain empty. We do not fabricate components, claim a newly verified A+B/C+D reconstruction, or back-calculate AIS dimensions from a detection box. The SARAIS local entity definitions likewise expose total length/width or length/breadth. Static update times are unknown; dynamic timestamps are not static-update timestamps.

The current raw-record matches have no independently stored object AIS coordinates. A product/MMSI/time match therefore verifies the record chain, not the original geometric association. The SAR crop target centre is not substituted for an AIS coordinate. Existing interpolation method and uncertainty are retained for assessment.
`ais_position_match_verified` is consequently false throughout; `chip_position_valid_*` describes image georeferencing, not AIS target correctness.

## Ground Grid Semantics

Spacing comes from actual cached GeoTIFF affines and projected metre CRSs, including all available `geometry_source_recovery_20261016/source_georefs.json` records, all synced server heads, the older exact header caches, and documented original 4x4/256-overlap tiling where a corresponding full-scene header exists. This pass is over all 38,091 manifest rows; it is not restricted to the earlier 8,622-row recovery cohort.

Original legacy crop manifests and returned server manifests restore crop offsets and side lengths by exact instance/source/UID/image identity. PNG headers are checked against those original crop side lengths before marking the native chip linear transform valid. VV and VH remain separate. No default 10 m is supplied to missing records. The reported metre sampling is a projected ground-grid spacing, not effective SAR spatial resolution or a geodesic survey measurement.

## Evidence And Limits

- `summary.json`, `coverage_by_port.csv`: final coverage.
- `view_metadata.csv`: 76,182 view records with crop links, JSON references, and header evidence.
- `dimension_evidence.csv`: each valid dimension variant per product/MMSI and a raw source row witnessing it.
- `dimension_conflicts.csv`: affected instances with conflicting static variants or exact evidence.
- `read_ais_sources.csv`: chosen raw file, bytes read, relevant records, and alternative local copies.
- `object_metadata_keys.json`: keys encountered in the opened object catalogs.
- `verification.json`: focused join/conflict cases, full manifest alignment, raw source spot checks, and affine-field checks.

One existing export copy per product is read, preferring the synchronized original collection and then the largest available file. Alternative paths are recorded but not exhaustively compared. Cross-observation consistency means consistency in these inspected cohort-product exports and linked JSON objects, not in all AIS messages ever received. Repeated exports/polarizations are not independent vessel evidence. Missing sources remain missing; this task does not download imagery, modify raw data, run detection/training, or edit the main report.

Run in the existing environment:

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'E:/临时会话/chip_quality_20261017/ais_metadata/build_sidecar.py'
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'E:/临时会话/chip_quality_20261017/ais_metadata/verify_sidecar.py'
```
