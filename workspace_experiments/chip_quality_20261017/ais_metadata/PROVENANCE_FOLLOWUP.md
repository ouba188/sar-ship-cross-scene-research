# Detector And AIS Dimension Provenance Follow-up

Date: 2026-10-17. Scope: read-only investigation of existing files, followed by this documentation handoff. No detection, training, network requests, exclusions, or sidecar changes were performed during this follow-up.

## Conclusions

1. The archived `550k.pt` detector was already used in the historical V5 annotation/matching and chip-construction workflow. The current re-detection is a same-source detector consistency check, not independent detection ground truth. Different canvas sizes, confidence thresholds, or corrected geometry do not establish independence.
2. All 245 selected anomalous-size sidecar rows agree with their cited raw SARAIS TXT size fields. The check reopened 126 files; there were zero mismatches, and all 245 cited records have export source `BLM`.
3. This proves traceability to exported values, not physical correctness of vessel dimensions. No local default assignment or A+B/C+D calculation was found in the inspected export/parser/JSON path. The upstream service may have its own decoding, aggregation, defaults, or saturation behavior; its implementation and original AIS components/messages were unavailable.
4. Nothing was removed or corrected. `ais_size_valid=True` remains a numeric/provenance validity flag under the documented association scope, not a physical-plausibility or surveyed-truth flag.

## Detector Evidence

All line numbers below are one-based.

| Evidence | Exact file and line | What it establishes |
|---|---|---|
| Actual historical run | [run_config.json](E:/临时会话/sea_ship_redetect_full_20260806_v5_direct_publish/local_F_remaining_v1/run/run_config.json:5) | The model is `E:\Docms\SARlabelimg\550k.pt`; the same file records 1024-pixel windows, confidence 0.08, active publication, and `axis_aligned_for_chip_crop`. |
| Archived model identity in project documentation | [README_zh.md](E:/临时会话/v5_server_toolkit_20260912/README_zh.md:20) | Explicitly identifies `models/550k.pt` as the historical V5 single-class Ship OBB weights, distinct from the alternative 51-class model. |
| Detector updates annotation geometry | [sea_ship_redetect_preview.py](E:/临时会话/v5_server_toolkit_20260912/scripts/sea_ship_redetect_preview.py:569) | `preview_object` converts the detection to an axis-aligned crop box, replaces `pts` and `cxcywh_angle`, and stores `redetect_status`/`redetect_model_conf` (lines 570, 573, 579, 583-585). |
| Detection-based selection for exported chips | [export_v5_matched_chips.py](E:/临时会话/v5_server_toolkit_20260912/scripts/export_v5_matched_chips.py:238) | The filter at line 241 accepts `matched_recentered`; lines 256-258 derive bounds and read that raster window. |
| Dynamic context crops inherit those boxes | [build_v5_context_chips.py](E:/临时会话/v5_server_toolkit_20260912/scripts/build_v5_context_chips.py:268) | Lines 269-275 use the prior box centre and crop-side rule to crop native pixels; lines 289-298 write crop coordinates and metadata. |
| Pipeline order | [auto_v5_after_geocode.py](E:/临时会话/v5_server_toolkit_20260912/scripts/auto_v5_after_geocode.py:263) | Redetection/publication runs first, then `export_v5_matched_chips.py` at line 275. |

The conclusion rests on existing execution records, documented model provenance, and the actual construction code. This follow-up did not freshly compare weight-file bytes or establish the detector's pretraining data. It also does not assert that every archived chip was produced by one identical invocation. Suggested report wording: same-source detector reinspection with separately sourced AIS dimension references, not independent ground-truth validation.

## Dimension Field Evidence

| Stage | Exact file and line | Operation |
|---|---|---|
| Request server export | [sar_ais_batch.py](E:/Hermes/scripts/sar_ais_batch.py:480) | Calls `exportAreaTrackToMat`. |
| Store server file | [sar_ais_batch.py](E:/Hermes/scripts/sar_ais_batch.py:508) | Saves the downloaded `file_bytes` directly; no local size decoding or component addition in this step. |
| Alternative export implementation | [sarais_export_safe_list_fast.py](E:/Hermes/scripts/sarais_export_safe_list_fast.py:124) | Also writes downloaded bytes directly. |
| Parse length and beam | [register_sarais_txt_to_local_tiffs.py](E:/临时会话/v5_server_toolkit_20260912/scripts/register_sarais_txt_to_local_tiffs.py:82) | Copies zero-based columns 14/15 into `length`/`beam`; these are human columns 15/16 of the headerless 18-column TXT. |
| Transfer to annotation candidates | [auto_v5_after_geocode.py](E:/临时会话/v5_server_toolkit_20260912/scripts/auto_v5_after_geocode.py:165) | Copies `row['length']` and `row['beam']` into `length_m` and `beam_m` at lines 165-166. |
| Preserve in object metadata | [build_raw_ais_review_json.py](E:/临时会话/v5_server_toolkit_20260912/scripts/build_raw_ais_review_json.py:373) | Copies candidate size values directly into `meta.length_m`/`meta.beam_m`; line 375 marks source `raw_ais_review_only`. |
| Separate visual box minimum | [build_raw_ais_review_json.py](E:/临时会话/v5_server_toolkit_20260912/scripts/build_raw_ais_review_json.py:330) | Lines 331-332 enforce a 6-pixel minimum for display/annotation geometry, but line 373 still copies the original size metadata. That pixel minimum does not explain the anomalous metre values. |
| Existing client data model | [AreaTrackResult.cs](E:/临时会话/SARAIS_decompiled/SARAISApplication.entity/AreaTrackResult.cs:24) | Copies service `length`/`width`; [ShipStaticResult.cs](E:/临时会话/SARAIS_decompiled/SARAISApplication.entity/ShipStaticResult.cs:24) copies `length`/`breadth`. These are client mappings, not the upstream decoder. |

The available TXT and object metadata contain total-length/width-like fields, not their A/B/C/D components. Arithmetic facts `1022 = 2 * 511` and `126 = 2 * 63` make component/default/saturation behavior a candidate explanation, but do not prove it. In particular, do not label these records confirmed AIS sentinels, true kilometre-long ships, or errors created by the local detector without additional evidence.

## Existing Readback Result

The prior read-only command selected sidecar rows where length is 511 or 1022, plus rows with length=width=2. It followed `ais_size_source` and the one-based `ais_source_line`, checked the raw field count (18), MMSI, and numeric equality of fields 14/15 to the sidecar length/width. It read until the largest requested line in each file. This check did not newly prove all rows were exact dynamic observations: the sidecar's exact-record, same-product-static, and cross-observation-static scopes still apply.

| Exported length x width | Sidecar rows | Distinct nonempty MMSI |
|---|---:|---:|
| 2 x 2 | 3 | 3 |
| 511 x 4 | 1 | 1 |
| 511 x 15 | 5 | 3 |
| 511 x 20 | 1 | 1 |
| 511 x 30 | 7 | 2 |
| 511 x 63 | 192 | 99 |
| 511 x 126 | 1 | 1 |
| 1022 x 6 | 12 | 1 |
| 1022 x 110 | 1 | 1 |
| 1022 x 126 | 22 | 12 |
| Total selected rows | 245 | Not summed across size groups |

The saved [raw_field_readback_245.json](E:/临时会话/chip_quality_20261017/ais_metadata/raw_field_readback_245.json) preserves the earlier command's aggregate result and all ten representative records, including exact paths and raw line numbers. It is transcribed from that completed tool output, not a new raw-file scan. The original command did not emit a per-instance 245-row audit log, so none is represented as having been saved. Existing per-instance source references remain in `ais_metadata_sidecar.csv`.

Representative raw evidence includes:

- 511 x 63, MMSI 235103397: [TXT line 8442](E:/Hermes/paper_tracking/port_ais_2024_2025_from_zip/products_by_primary_port/Antwerp-Bruges/S1A_IW_GRDH_1SDV_20250304T173328_20250304T173353_058158_072F3A_33A2.SAFE/S1A_IW_GRDH_1SDV_20250304T173328_20250304T173353_058158_072F3A_33A2_AIS_pm3min.txt:8442).
- 1022 x 126, MMSI 200063741: [TXT line 4672](E:/Hermes/round5_safe_collect_light_new_only/S1A_IW_GRDH_1SDV_20250609T095438_20250609T095503_059568_07653B_733A.SAFE/S1A_IW_GRDH_1SDV_20250609T095438_20250609T095503_059568_07653B_733A_AIS_pm3min.txt:4672).
- 2 x 2, MMSI 412440188: [TXT line 180224](E:/Hermes/mfais27/s/S1A_IW_GRDH_1SDV_20250901T095436_20250901T095501_060793_079158_DE89.SAFE/S1A_IW_GRDH_1SDV_20250901T095436_20250901T095501_060793_079158_DE89_AIS_pm10min.txt:180224).

Separately, the previously reported 1,016 static-conflict instance rows involve 290 distinct nonempty MMSIs and zero empty-MMSI rows. That conflict set is not synonymous with the 245-row anomalous-value set.

## Interpretation Boundary

**Established:** the values occur in existing service-exported records and were carried through the inspected local metadata path; the detector's historical use creates dependence between old crop selection and current re-detection.

**Not established:** physical hull dimensions, correct SAR-to-AIS association for every chip, upstream component/default/sentinel semantics, or independent detector ground truth. Retain the original values and these limitations; this follow-up performs no exclusions or corrections.
