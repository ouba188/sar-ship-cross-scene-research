# Facility/context audit, 2026-10-07

The existing facility alignment is usable for a bounded context gate without a
map rebuild. All 38,091 current manifest rows are retained. No model was trained,
no data downloaded, and no query labels, scores, rankings or class fields were
loaded for this audit. All writes are confined to this directory.

Handoff status: READY / FROZEN. `facility_features.npz` is the immutable input
for the main agent's bounded pilot; no further regeneration is planned. Use only
`values` and `known` for that pilot. Three focused tests passed (0.310 seconds),
in addition to the successful raw/cache and query-index checks during export.

## Immediate input

`facility_features.npz` is aligned to the exact current manifest order:
`E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv`.

```python
with np.load("E:/Hermes/paper_tracking/sensitivity/port_context_audit_20261007/facility_features.npz",
             allow_pickle=False) as z:
    X = z["values"]             # float32, (38091, 60), masked zeros
    known = z["known"]          # bool, same shape, MUST accompany values
    row_ids = z["instance_id"]  # identifiers for alignment, not model inputs
    row_index = z["row_index"]
    feature_names = z["feature_names"]
```

Use the existing cache `spos` and `qpos` to slice these arrays; no rows were
reordered or selected by outcomes. Missing query context must retain the baseline
prediction. Fit scaling and any gate only on the authorized source partition.
Supported zero and missing zero differ through `known`, not through the value.

Additional arrays are `role_values`/`role_known` (38091 x 8 x 4), `role_quality`
(38091 x 8), and `raw_values`/`raw_known` (38091 x 42). Raw fields contain each
of 14 layers' capped distance and log1p counts at 1 and 3 km. The role mapping
and scaling reproduce `run_crca_development.py`. Names accompany every block.
`historical_values`/`historical_known` and `historical_role_values`/
`historical_role_known` preserve the less restrictive old support for audit;
they are NOT interchangeable with the default bounded mask.

## Verified counts

| Quantity | Rows |
| --- | ---: |
| Current full manifest, 24 ports | 38,091 |
| Historical OSM support, raw tables exactly reproduced | 33,082 |
| Default support with existing full 10.1 km footprint evidence | 18,098 |
| Default supported rows with mapped physical infrastructure within 3 km | 3,158 |
| Historical support lacking full-footprint evidence | 14,984 |
| No historical OSM support | 5,009 |
| Current separately recorded georeferenced center available | 35,014 |
| Exact overlap with old ten-port manifest by product and VV/VH filenames | 3,618 |
| Those overlap rows supported by old ten-port role cache | 156 |

Raw source/cache support and value disagreement: zero rows. Source key collisions:
zero. Measured zero cells in the default 60-feature array: 498,926. This count
includes zero contrasts and is not a count of areas with no real facilities.
Seamark routes are excluded from the physical-infrastructure-nearby count.

All 23 active query caches match the current manifest's port ordering and exact
query indices. Melbourne remains in the 38,091-row export; its outer cache is
marked skipped. The union of active query rows has 38,027 rows, including 18,092
with bounded context. Cached source partitions have 11,265 to 18,083 supported
rows per outer fold. These are the cache's existing `spos` partitions, not a new
claim of identity independence.

| Outer port | Query rows | Bounded query context | Infrastructure within 3 km |
| --- | ---: | ---: | ---: |
| Antwerp-Bruges | 886 | 848 | 219 |
| Busan | 871 | 856 | 103 |
| Callao | 476 | 203 | 71 |
| Fujairah | 1,158 | 1,047 | 274 |
| Hamburg | 1,619 | 517 | 14 |
| Houston | 804 | 282 | 140 |
| Jebel Ali | 4,602 | 1,224 | 170 |
| Los Angeles | 583 | 124 | 121 |
| Mombasa | 149 | 7 | 5 |
| New York | 470 | 1 | 0 |
| Newcastle | 299 | 4 | 2 |
| Ningbo-Zhoushan | 5,756 | 5,745 | 871 |
| Port Hedland | 859 | 0 | 0 |
| Port Klang | 3,425 | 554 | 159 |
| Port Said | 1,895 | 709 | 5 |
| Qingdao | 1,981 | 1,899 | 76 |
| Richards Bay | 744 | 193 | 0 |
| Rotterdam | 573 | 552 | 173 |
| Santos | 1,658 | 49 | 11 |
| Shanghai | 2,390 | 2,323 | 513 |
| Singapore | 5,259 | 741 | 218 |
| Sydney Botany | 304 | 6 | 4 |
| Tanger Med | 1,266 | 207 | 4 |

## Exact assets and joins

The cache is `artifacts/eight_class_adaptive_20260916/knowledge_real_recovered_20260917_pm5/relations.npz`,
with `relation_features.json` and `relation_sources_private.csv` in that directory.
Only its 60 `osm_` columns are reused. The pure `derive_relations` function in
`local_relation_gate_features.py` recomputed them from all 12 selected source
tables; values and masks match the cached representation. Detector/AIS feature
blocks, port aggregates, ship types and all label fields are excluded.

The complete absolute paths are in `source_inventory.csv`. Selected source counts:

| Source directory | Selected rows | Bounded rows |
| --- | ---: | ---: |
| Returned metadata/local_knowledge_v4 | 13,998 | 0 |
| Merged shared_assets_local_knowledge_v6 | 986 | 0 |
| class_neutral_osm_v1 | 4,583 | 4,583 |
| pilot_local_knowledge_recovery_20260914_v2 | 518 | 518 |
| v6_relation_observation_recovery_expanded_20260916 | 858 | 858 |
| real_k_asset_audit_20260916 | 1,189 | 1,189 |
| real_k_asset_audit_20260916/extra_map_recovery | 1,544 | 1,544 |
| real_k_asset_audit_20260916/extra_map_recovery2 | 2 | 2 |
| real_k_asset_audit_20260916/extra_map_recovery3 | 6,557 | 6,557 |
| real_k_asset_audit_20260916/extra_map_recovery4 | 23 | 23 |
| real_k_asset_audit_20260916/extra_map_recovery5 | 188 | 188 |
| real_k_asset_audit_20260917/osm_recovery | 2,636 | 2,636 |

Returned rows join by exact product, both source TIFF paths and both object UIDs.
The class-containing `return_pair_key` is not even loaded. Legacy/shared rows
use source-recorded IDs with product checks and, for the shared manifest, both
exact filenames. Old ten-port rows require product plus both exact chip filenames.
Every side of any duplicate join key is excluded and recorded, never averaged.
No fallback to port centroids, approximate names, nearby ships, or majority classes
exists in this script.

## Provenance limits

The default mask verifies recorded map-footprint evidence, not perfect spatial
truth. Of its 18,098 supported rows, 965 use recorded AIS positions/proxies:
895 within 180 seconds, 4 within 300 seconds, and 66 older <=500 m heuristic
position proxies. Another 6,438 inherit verified historical sample centers.
The remaining 10,695 have crop-center/georeferencing coordinate kinds. The
per-row `coordinate_kind` records the position actually used by the map source;
`current_coordinate_kind` separately records newer coordinate availability.
Do not describe all default support as exact contemporaneous chip locations.

The 13,998 returned-source rows were extracted at per-chip coordinates, but the
inspected extractor did not verify the full map footprint. The 986 shared-source
rows also lack that evidence in the followed source sidecars. Both remain usable
historical descriptors with a caveat, but are excluded from default `known`.
This is an evidence gap, not proof those rows lie outside the map.

Snapshots mix 2025-01-01 historical PBF and 2026 map queries. Some old sidecars
provide the map path but no row-level map date: blanks in `row_provenance.csv`
are intentionally retained. Temporal matching to SAR acquisition remains unresolved;
these maps must not be described as contemporaneous facility observations.
Provider boundaries are current. Missing/empty layers
do not establish absence; queried count-zero means no mapped object in that
supported layer/radius. Counts include map geometries, not a deduplicated census.
Distances cap at 10 km. Functional role names are geometric associations, not
observed ship types, berth occupancy, cargo or trajectories. No berth/trajectory
branch was established by this audit.

The 5,009 remaining map gaps preserve the existing reasons: 3,765 pending PBF
extraction; 1,012 outside available regions; 148 lacking full-radius edge coverage;
44 with no compatible semantic layer; 31 without verified coordinates; 9 with an
incomplete download. No attempt was made to rebuild or download these assets.

A bounded context gate is feasible now with the exported mask and unchanged
missing-context fallback. Coverage is highly uneven: Port Hedland has no bounded
query context, and New York/Newcastle/Mombasa/Sydney Botany have very few rows.
This audit establishes input alignment and support, not classifier improvement.

## Reproduction and files

Run `D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe -B audit_context.py`
in this directory. The script reads only selected sources and their named sidecars.
It writes `audit_summary.json`, `source_inventory.csv`, `row_provenance.csv`,
`coverage_by_port.csv`, `coverage_by_port_source.csv`, `query_source_coverage.csv`,
`query_row_ids.csv`, `old_role_mapping.csv`, `key_collisions.csv` and the NPZ.
Raw/cache discrepancies stop export and produce `cache_mismatches.csv`.

`python -B -m unittest test_audit_context -v` checks duplicate-key exclusion,
known zero versus missing with no fallback, and full exported row alignment,
finite values, masks and feature allowlist. These checks specifically catch
accidental join averaging, lost missingness, removed targets or label fields in
the export; no scientific model reruns are involved.
