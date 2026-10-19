# Geometry recovery coverage

Valid: 240/269; previous: 132/269; added: 108.
Ownership eligible: 240; missing: 29.

| Date | Rows | Previous | Eligible | Recovered |
| --- | ---: | ---: | ---: | ---: |
| 20250202 | 36 | 12 | 16 | 16 |
| 20250211 | 6 | 0 | 6 | 6 |
| 20250315 | 130 | 49 | 121 | 121 |
| 20250401 | 66 | 66 | 66 | 66 |
| 20251126 | 31 | 5 | 31 | 31 |

Missing reasons:

- ownership_gate:multiple_or_nonmatching_product_footprints: 29

The 108 newly eligible rows previously had incidence_only metadata from old caches without native pixel/line.
Mixed/nonmatching product windows remain missing. No default angles or pass-derived signs are assigned.
All original orbit CSV columns are retained. axis_difference_deg remains NaN on newly valid rows because the old reference angle is missing.
The old 132 valid rows are recomputed from full XML and replay-checked; see receipt.json for per-column differences.
This is metadata coverage only, not an evaluation of the learner or independent evidence of classification gains.
Public XML downloads this run: 4 files, 6893470 bytes. No imagery or credentials read.
Writes are confined to recovery/. The old experiments and external raw cache are read-only. No Git operations.
