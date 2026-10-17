# Metadata handoff

Owner writes only this directory. Primary: `oss1_factor_metadata.csv`, exactly
1349 rows in `geometry.csv` order, joined by both `global_row` and `sample_id`.
Optional: `oss2_factor_metadata.csv`, 3745 rows in target manifest order.

Stable fields: `global_row`, `sample_id`, `scene_id`, `product`, `product_type`,
`mmsi`, `class_id`, `class_name`, `ais_sog_knots`, `ais_sog_mps`, `ais_cog_deg`,
`ais_true_heading_deg`, `ais_nav_status`, `ais_draught_m`, `ais_length_m`,
`ais_width_m`, `sar_manual_length_m`, `marine_length_overall_m`,
`ais_time_utc`, `sar_time_utc`, `ais_minus_sar_seconds`, `ais_abs_timegap_s`,
`orbit_los_horizontal_east_unit`, `orbit_los_horizontal_north_unit`,
`orbit_flight_east_unit`, `orbit_flight_north_unit`, `orbit_incidence_deg`,
`orbit_incidence_minus_table_deg`, `ground_los_horizontal_signed_mps`,
`ground_los_horizontal_abs_mps`, `ground_los_3d_signed_mps`,
`ground_los_3d_abs_mps`, `ground_alongflight_signed_mps`,
`ground_alongflight_abs_mps`, `ground_projection_valid`.

LOS points from target toward satellite. Positive horizontal LOS motion is
toward its horizontal projection; positive alongflight is satellite ECEF
velocity projected onto the target tangent plane. Three-dimensional LOS uses
horizontal ground velocity with zero vertical component. All are AIS snapshot
proxies evaluated with product-time geometry, not observed motion at SAR time.

Raw codes, censor flags, validity flags, XML paths and product metadata accompany
these columns. SAR XML timestamps are constant within each product; no precise
per-pixel azimuth-time recovery or full chip georeference is claimed. Imported
heading/length are provenance-only fields, never independent AIS truth.

Final receipts: `coverage.csv`, `product_coverage.csv`, `PROTOCOL.json`,
`FINDINGS_ZH.md`, and `test_results.txt`.

## Final status

Completed: OSS1 1349 rows / 22 GRDH products / 1110 MMSIs; OSS2 3745 rows /
38 products / 2698 MMSIs (2746 GRDH + 999 SLC). Nine focused tests pass,
including nine separately parsed raw XML ship checks.

Primary usable projection mask: `ground_projection_valid`. OSS1: 1347;
OSS2: 3455 (2458 GRDH + 997 SLC). `orbit_geometry_valid` alone only indicates
computable geometry at the reported timestamp. All 60 products have vectors.

OSS2 product `S1A_IW_GRDH_1SDV_20160116T095421_20160116T095446_009518_00DD1D_3A0D`
has 282 selected rows with reported SAR time 09:55:00Z outside its acquisition
09:54:21.324792Z to 09:54:46.323740Z. All 282 have
`sar_time_within_product_span=False` and no usable motion projections; 201 also
fail the 1-degree incidence consistency check. Raw times and gaps are retained,
not silently corrected. Use the product-span flag also for AIS/SAR timing analyses.

Both datasets are dominated by low SOG: 1123/1349 and 2333/3745 are below 1 knot.
COG code-validity does not certify useful direction at low speed. Draught >255
is invalid under the documented AIS encoding, not automatically divided by 100.
