# Input-stratum diagnosis, 2026-10-08

The matched input correction recovers discriminative signal before target adaptation. It does not identify a single physical cause or explain all remaining cross-domain errors.

| Arm | Old OA | New OA | Delta OA pp | Delta fixed-8 BA pp |
| --- | ---: | ---: | ---: | ---: |
| original_base | 17.920356 | 24.207575 | +6.287219 | +1.467595 |
| base64 | 21.433259 | 28.937087 | +7.503828 | +4.459808 |
| full5 | 21.834150 | 29.633543 | +7.799392 | +4.265153 |

## Changed inputs versus fallback

| Arm / group | Query rows (draw mean) | Group OA delta pp | Group present-class BA delta pp | Overall OA contribution pp | Fixed-8 BA contribution pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| original_base / changed | 76526.8 | +10.140238 | -1.593469 | +6.287219 | +1.467595 |
| original_base / fallback | 46898.2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| base64 / changed | 76526.8 | +12.493699 | +5.086430 | +7.746405 | +4.327315 |
| base64 / fallback | 46898.2 | -0.638453 | -0.346058 | -0.242576 | +0.132492 |
| full5 / changed | 76526.8 | +13.073626 | +5.462704 | +8.105975 | +4.477808 |
| full5 / fallback | 46898.2 | -0.806923 | -0.820765 | -0.306583 | -0.212655 |

Source-only fallback prediction flips: 0.0 per draw. The same frozen source head therefore provides a direct input-pipeline contrast on changed rows. On fallback rows the base64 difference reflects changed support features and their fitted correction; full5 also includes expert adaptation and competition for the global handoff cap. These fallback gains/losses are downstream spillover, not newly recovered pixels.

Contributions use the same full-query denominators in each draw and sum exactly to overall changes. Present-class subgroup BA cannot be summed or compared as if class composition were fixed.

## Where the gains concentrate

On changed rows, source-only OA rises 10.140 pp, but their own eight-class BA changes -1.593 pp (21.337% to 19.743%). This differs from their positive contribution to full-cohort fixed-8 BA because each class has a different changed/fallback share. The gain is class-selective, not a uniform recovery. Class IDs: 0=bulk_carrier, 1=fishing_vessel, 2=general_cargo, 3=product_chemical_tanker, 4=container_ship, 5=crude_oil_tanker, 6=tug_towing, 7=offshore_supply.

- Source-only status: detector_dynamic: OA contribution +6.287 pp, fixed-8 BA contribution +1.468 pp; fallback_ambiguous_mmsi: OA contribution +0.000 pp, fixed-8 BA contribution +0.000 pp; fallback_missing_mmsi: OA contribution +0.000 pp, fixed-8 BA contribution +0.000 pp; fallback_no_detection: OA contribution +0.000 pp, fixed-8 BA contribution +0.000 pp.
- Source-only identity_join: same_fid_unique_mmsi: OA contribution +3.598 pp, fixed-8 BA contribution +1.704 pp; remapped_unique_mmsi: OA contribution +2.317 pp, fixed-8 BA contribution -0.370 pp; mmsi_and_location: OA contribution +0.372 pp, fixed-8 BA contribution +0.134 pp; ambiguous_mmsi: OA contribution +0.000 pp, fixed-8 BA contribution +0.000 pp; missing_mmsi: OA contribution +0.000 pp, fixed-8 BA contribution +0.000 pp.
- Source-only requested_side_bin: 64: OA contribution +5.703 pp, fixed-8 BA contribution +2.029 pp; 72-128: OA contribution +0.586 pp, fixed-8 BA contribution -0.558 pp; fallback_old_input: OA contribution +0.000 pp, fixed-8 BA contribution +0.000 pp; >128: OA contribution -0.002 pp, fixed-8 BA contribution -0.003 pp.
- Source-only det_conf_bin: [0.75,+inf): OA contribution +3.441 pp, fixed-8 BA contribution +1.065 pp; [0.5,0.75): OA contribution +2.512 pp, fixed-8 BA contribution +0.535 pp; [0.25,0.5): OA contribution +0.335 pp, fixed-8 BA contribution -0.133 pp; <0.25: OA contribution +0.000 pp, fixed-8 BA contribution +0.000 pp.
- Source-only class_id: 0: OA contribution +8.553 pp, fixed-8 BA contribution +4.571 pp; 2: OA contribution +1.397 pp, fixed-8 BA contribution +2.851 pp; 3: OA contribution +0.084 pp, fixed-8 BA contribution +0.056 pp; 7: OA contribution -0.168 pp, fixed-8 BA contribution -1.247 pp; 4: OA contribution -0.251 pp, fixed-8 BA contribution -0.126 pp; 6: OA contribution -0.271 pp, fixed-8 BA contribution -0.313 pp; 1: OA contribution -0.318 pp, fixed-8 BA contribution -1.293 pp; 5: OA contribution -2.739 pp, fixed-8 BA contribution -3.031 pp.

Source-only scene OA change: 2557 positive, 1044 negative, 1985 zero. scene_summary.csv includes scene size, class composition, MMSIs, crop coverage and metadata quantiles; small or class-sparse scenes are descriptive, not reliable standalone effect estimates.

## Identity and input limits

123,489 rows; 5,586 scenes; 15,550 distinct recorded MMSIs; 116,264 scene/MMSI pairs. 51 MMSIs have multiple class IDs (548 rows); 0 scene/MMSI pairs have multiple labels. These are metadata consistency counts, not label error rates or verified vessel counts.

A unique MMSI match in current DET records is source uniqueness, not ground-truth identity. MMSI-plus-location is a heuristic. Detection confidence does not establish semantic foreground. len_m is privileged AIS metadata, not visual evidence and not used as a predictive feature here. len_m and det_len_px are recorded separately with missing/nonpositive flags; no meter/pixel ratio or physical disagreement is inferred. Fallback requested_side=224 is excluded from dynamic >128.

Changed support rows by draw: [37, 38, 44, 39, 48]. Query rows sharing support MMSI: [1356, 1065, 999, 1248, 1240]. The split is row-level and the five draws overlap.

## Radiometric trace and interpretation

The named PIL and GDAL V5 crop producers copy existing 8-bit TIFF pixels into PNG crops. The source manifest names UTM_8bit tiles, but the raw-to-8bit recipe remains unresolved after a bounded search. The target independently stretches each chip/band at 2/98 percentiles. Changing the window changes both content and stretch statistics: centering, identity association, spatial scale and radiometry are not isolated. The source-only changed-row gain locates a demonstrated loss at input construction; remaining loss cannot be assigned to raw scaling, encoders or heads from these strata alone. Feature/head probes are separate.

Checks passed: original-row joins, sample/label alignment, identical supports/query masks, published metric reproduction, and additive OA/fixed-8 BA contribution reconciliation for every partition. No training, feature-matrix reads, GPU use, new model modules or visual adjudication.

Outputs: input_diagnosis.json; strata_per_draw.csv; strata_means.csv; strata_metadata.csv; scene_summary.csv; support_summary.csv; mmsi_label_summary.csv; input_flags.csv.
