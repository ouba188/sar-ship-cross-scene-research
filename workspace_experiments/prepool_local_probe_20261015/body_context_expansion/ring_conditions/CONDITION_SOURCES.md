# Ring condition sources

Coverage: 4289/4289 unique required samples; 20 frozen episodes.
Frozen query/support union equals required_samples.sample_id; manifest row_index alignment checked.
No predictions, correctness labels, performance outcomes, fitted thresholds or plots are used.

## Sources and definitions

- Inventory: `E:\临时会话\prepool_local_probe_20261015\body_context_expansion\required_samples.csv` and `E:\临时会话\prepool_local_probe_20261015\body_context_expansion\frozen.json`.
- Class names and native bbox geometry: `E:\临时会话\region_cue_replication_20261013\ports\frame.csv`; exact sample_id join.
- Cache row identity: `E:\临时会话\visual_reliable_baseline\artifacts\eight_class_adaptive_20260916\manifest.csv`, instance_id at row_index equals sample_id.
- Pixels: `D:\SAR_completion_cache_20261010\source_inputs128.npy`, uint8 [N,2,128,128], channels VV then VH.
- Masks: `E:\临时会话\prepool_local_probe_20261015\localization.py` spatial_masks(rows), default 56x56.
  Box is the union of normalized VV/VH native axis-aligned bboxes sampled at cell centers.
  Ring is dilation(box,8) AND NOT dilation(box,2), using the existing scipy default structure.
  Both masks are resized to 128x128 with torch nearest interpolation, exactly as the existing raw readout.
  Background sampling fallback count: 0; sampled backgrounds are unused.
- box_occupancy_vv/vh: native bbox width*height / native image width*height; fraction.
- box_aspect_vv/vh: max(native bbox width,height)/min(width,height); dimensionless >=1.
  These are detector-box image-space proxies, NOT true vessel length, physical aspect or segmentation occupancy.
- native_h/w and bbox coordinates: native image pixels; box/ring_pixels_128: resized mask pixel counts.
- raw_box/ring_vv/vh_mean/std: original stretched uint8 divided by 255, range [0,1]; population std (ddof=0).
  No radiometric calibration: NOT sigma0, dB, physical sea state or a causal sea-condition measurement.
- standardized_contrast_vv/vh: (box_mean-ring_mean)/max(ring_std,1/255); dimensionless.
- saturation_fraction: fraction of mask pixels equal to uint8 255 (upper endpoint only).
  zero_fraction separately records pixels equal to 0; endpoints alone do not establish sensor clipping.

## Bounded optional-condition search

Two candidate table headers inspected (limit three):
- context_condition_study_20261015/geometry_by_sample.csv: no incidence, relative angle or speed fields.
- same_vessel_condition_study_20261010/observations.csv: instance_id, incidence_deg, relative_signed_deg and validity statuses.
  instance_id is joined exactly to required sample_id; only finite values with source status 'ok' are retained.
  Angles are in degrees, relative_signed_deg preserves the source signed convention without reinterpretation.
  Missing joins and unavailable/rejected source statuses are retained explicitly in the CSV.
- Linked check_motion_teacher_feasibility.py uses motion_knowledge_20261011/attributes.npz
  via fit/held_obs_ids and separate SOG validity masks, not a direct sample_id speed table.
  A reliable target-sample speed join was not established within this bounded search; speed_knots is entirely missing.
  No source-split speed is extrapolated to target samples. No network search performed.

## Nonmissing coverage

| Field | Available | Total |
| --- | ---: | ---: |
| class_name | 4289 | 4289 |
| box_occupancy_vv | 4289 | 4289 |
| box_occupancy_vh | 4289 | 4289 |
| box_aspect_vv | 4289 | 4289 |
| box_aspect_vh | 4289 | 4289 |
| box_pixels_128 | 4289 | 4289 |
| raw_box_vv_mean | 4289 | 4289 |
| raw_box_vv_std | 4289 | 4289 |
| raw_box_vv_saturation_fraction | 4289 | 4289 |
| raw_box_vv_zero_fraction | 4289 | 4289 |
| raw_box_vh_mean | 4289 | 4289 |
| raw_box_vh_std | 4289 | 4289 |
| raw_box_vh_saturation_fraction | 4289 | 4289 |
| raw_box_vh_zero_fraction | 4289 | 4289 |
| ring_pixels_128 | 4289 | 4289 |
| raw_ring_vv_mean | 4289 | 4289 |
| raw_ring_vv_std | 4289 | 4289 |
| raw_ring_vv_saturation_fraction | 4289 | 4289 |
| raw_ring_vv_zero_fraction | 4289 | 4289 |
| raw_ring_vh_mean | 4289 | 4289 |
| raw_ring_vh_std | 4289 | 4289 |
| raw_ring_vh_saturation_fraction | 4289 | 4289 |
| raw_ring_vh_zero_fraction | 4289 | 4289 |
| standardized_contrast_vv | 4289 | 4289 |
| standardized_contrast_vh | 4289 | 4289 |
| incidence_deg | 3543 | 4289 |
| relative_signed_deg | 341 | 4289 |
| speed_knots | 0 | 4289 |

incidence_source_status: ok=3543; unsupported_coordinate_provenance=746

relative_angle_source_status: body_axis_unavailable=3807; ok=341; invalid_grid=141
