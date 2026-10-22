# F0921 target-to-scene linkage preparation

All new links remain candidates. Confirmed training targets: 0.

Run: `D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe -B build_linkage.py`

Inputs are read-only; raster access reads metadata only. Outputs are confined to this script directory.

| Scene | Objects | Queue targets | Unique candidates | MMSI | With external neighbors | Complete 1km |
|---|---:|---:|---:|---:|---:|---:|
| 2EC7 | 87 | 6 | 1 | 1 | 0 | 1 |
| 45FE | 37 | 6 | 4 | 4 | 1 | 4 |
| 31CC | 159 | 5 | 1 | 1 | 1 | 1 |
| 5FFE | 363 | 45 | 43 | 43 | 12 | 43 |
| B529 | 344 | 26 | 22 | 22 | 13 | 22 |

## Target disposition

```json
{
  "unresolved_product_ownership": 29,
  "outside_five_scene_inventory": 161,
  "geometry_one_to_one_candidate": 71,
  "no_qualifying_box_overlap": 7,
  "box_candidates_pending": 1
}
```

## Evidence and limits

- Inherited pixel replay is crop-to-mosaic, not an independently verified vessel label.
- Record product may be outside these five scenes; overlapping mosaic ownership stays unresolved.
- Box IoU >= 0.25 plus reciprocal uniqueness yields a candidate only; no nearest-point or MMSI assignment.
- 224-coordinate detection boxes are geometrically mapped to native pixels; no resized image is used as native input.
- Neighbor counts are retained local detections, not verified vessels or full-scene density.
- The 3604 inherited edges are inventoried; 1km external-chip neighbors are recomputed from retained object geometry.
- Coverage is union of existing VV detection windows; it is not detector recall or dual-pol completeness.
- No native pixel equality between newly linked object and target is asserted. No new visual confirmation.

Next: inspect native VV/VH target and object windows for the candidate list, preserving competing boxes and source identity uncertainty. Freeze evaluation inputs only after that evidence supports the specific target association.

No training, download, shared-file edits, credential access, or Git synchronization was performed. The current directory-only scope takes precedence over the standing archive workflow.
