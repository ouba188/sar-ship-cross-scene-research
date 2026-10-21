# Radiometry Trace Reproduction

Scope: 100 crude-oil-tanker pairs / 200 endpoints / 400 VV-VH views from the existing 24-port cohort. All outputs are written beside these scripts. No classifier imports, training, scene decoding, or dataset changes.

Run from PowerShell with the existing environment:

```powershell
$traceDir = 'E:/临时会话/same_vessel_gallery_20261019/crude_input_controls_20261021/radiometry_trace'
$tracePython = 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe'
& $tracePython -X utf8 -B "$traceDir/trace_radiometry.py" prepare
& $tracePython -X utf8 -B "$traceDir/trace_radiometry.py" inventory
& $tracePython -X utf8 -B "$traceDir/trace_radiometry.py" local-windows
& $tracePython -X utf8 -B "$traceDir/finalize_trace.py"
```

The first stage verifies cohort identity against the original manifest, exact returned crop records, image dimensions, actual PNG alias pixels and existing target/rest pixel statistics. Failure stops rather than silently assigning a different source.

The inventory stage enumerates only the data roots listed in `trace_radiometry.py`; it reads ZIP central directories only if a matching archive exists. The recorded search domain is `local_search_roots.json`, not a claim to have audited every drive or remote machine.

The window stage reads raster headers and bounded windows. GDAL PAM writes are disabled. The raw measurement rectangle is capped at 256x256; current maximum is 113x112. Source strip decoding can exceed requested window size; the output records both requested pixels and the estimated strip bytes. `local_raster_headers.json` contains headers, not global raster statistics. No new dependencies are required.

Remote checks are optional and are **not** part of the default replay. The already-completed connection attempts are retained in `remote_connection_status.json`; all failed before source data access. `remote_trace.py` uses the existing named SSH key, never reads or emits key contents, and writes nothing on servers. Do not rerun or expand remote operations to reproduce the local report.

Main outputs:

- `FINDINGS_ZH.md`: conclusions, actual evidence, focus cases and unresolved gaps.
- `SUMMARY.json`: machine-readable counts and scope.
- `view_coverage_400.csv`, `endpoint_coverage_200.csv`, `pair_coverage_100.csv`, `product_coverage_43.csv`: coverage at each denominator.
- `view_trace.csv`, `focus_provenance.json`: original crop windows and full provenance.
- `png_statistics.csv`, `focus_statistics.csv`: unchanged PNG statistics; values in 0..255, not physical units.
- `source_pixel_comparisons.csv`, `sparse_pixel_mismatches.csv`: pixel comparisons, including overlap duplicates and the two unmatched pixels.
- `source_window_statistics.csv`, `raw_window_mappings.json`: source window statistics and raw GCP mapping qualifications.
- `safe_metadata_coverage.csv`, `source_json_radiometry_search.csv`, `CODE_EVIDENCE.md`: metadata and actual stored code evidence.
- `focus_native_pixels.png`: fixed 0..255 display of the four focus endpoints, nearest 3x.
- `raw_candidate_pixels.png`: two raw-window examples; newly computed local display limits are explicitly labeled, not historical parameters.
- `recovered_windows/*.npy`: small derived local windows for replay; exclude these from research-repository sync under the no-raw-data rule.

`exact_pixels` applies only to direct UInt8 comparisons. It is false for raw DN candidates because neither their radiometry nor the historical scene warp is exactly reproduced. `fitted_display_*` values are descriptive comparisons fitted against PNG; never use them as recovered historical stretch limits.

The scripts preserve unknown limits as empty fields, distinguish metadata-only SAFE directories from actual measurement TIFFs, and do not call any original pipeline's processing entry point.
