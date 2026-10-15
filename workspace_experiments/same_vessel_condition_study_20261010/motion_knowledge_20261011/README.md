# Source motion knowledge

Run ../extract_motion_knowledge.py with the existing ClearSAR Python environment.
fit means sourcefit; held is held source vessels, never a target dataset.
Main consumer: attributes.npz; required 1D keys fit_obs_ids, held_obs_ids, fit_sog, held_sog, fit_sog_valid, held_sog_valid.
sog aliases sog_knots (knots); validity masks are boolean. attributes.npz and motion_knowledge.npz have identical content.
motion_knowledge.npz uses fit_ and held_ prefixes, in original cohort order.
Keys: obs_ids, sog_knots, cog_deg, true_heading_deg; each motion field has a _valid mask.
Additional keys: ais_epoch, abs_time_delta_seconds, raw_lon, raw_lat, match_status, mmsi, product, coordinate_check.
Missing/invalid/ambiguous motion values are NaN. Zero speed is valid.
motion_knowledge.csv contains one row per observation and provenance; uid_view_audit.csv preserves JSON evidence.
exact_raw_records.jsonl retains all exact-key candidates and source line numbers.
read_sources.csv bounds the TXT reads; summary.json records masks and limitations.
COG is not true heading. No navigation-status field is produced.
Use fit masks for auxiliary training only; held metadata is audit-only. SAR-only inference needs none of these fields.
