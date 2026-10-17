# Product annotation handoff

## Current full-cohort delivery

Extension COMPLETE: `manifest.csv` now has 87 successful exact-product records,
covering all 5145 radar-invalid rows in the original fixed 8622-row cohort.
`extended_products.csv` contains the 83 added products (4358 rows);
`extended_target_rows.csv` maps those original row IDs to products.
All 83 complete VV annotations were available locally (144007584 bytes), so the
extension made zero network requests. All 87 grids passed the existing consumer
validation. The original four manifest rows are preserved exactly. Manifest was
published via same-directory temporary file and atomic replace.

Use `manifest.csv.local_grid_path` for exact-product lookup. The original
`summary.json` and `target_rows.csv` describe only the initial 787-row delivery;
use `extended_summary.json` and `EXTENDED_RECOVERY_ZH.md` for current coverage.
Georef association and pixel-angle applicability remain main's responsibility.

## Initial delivery record

All 787 missing-grid rows map to exactly four products. Their complete original
VV annotations already exist under
`E:/临时会话/visual_reliable_baseline/artifacts/safe841_metadata_20260917/products/<product>.SAFE/annotation/iw-vv.xml`.

Previous `radar_fetch_status.csv` records show timeouts / interrupted connections,
not missing archive objects. Product counts: 18D1=354, B38D=305, 9D79=99, C0C6=29.

Recovery COMPLETE: `manifest.csv`, `target_rows.csv`, `annotations/<product>.xml`,
and `radar_grid/<product>.json` (existing consumer schema: status=ok, product,
points with line/pixel/latitude/longitude/incidenceAngle). Read final report and
manifest before consumption. This work does not recover georef or convert angles.

All four manifest rows have status `recovered_local_exact_product`, covering 787/787
target rows. Every grid has 210 nodes and passed the existing
`relative_radar_geometry._validate` consumer check. Use `local_grid_path` directly;
the old global cache and old CSV were not modified. All four public source URLs
returned HTTP 206 to anonymous 4096-byte range requests, matching the local XML
prefix. Full annotations were reused locally, not re-downloaded.

See `RECOVERY_REPORT_ZH.md` for the initial scope and official source checks.
The subsequent extension above no longer needs to wait for a sourcebridge list.
