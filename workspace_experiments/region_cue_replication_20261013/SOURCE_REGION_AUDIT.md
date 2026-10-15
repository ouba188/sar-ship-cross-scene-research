# Local24 Source Region Audit

## Result and contract

- 38,091 source samples, 24 ports; one CSV row per original instance_id, original order and all original columns preserved.
- All localization is detection bbox metadata. No head/tail columns are present in the localization manifests: head_x/head_y/tail_x/tail_y remain empty and headtail_valid is false.
- region_valid means BOTH native polarization bboxes pass extent checks; it does not mean head/tail validity. bbox_valid is an explicit alias. Never pass these rows directly to the old head-tail region_boxes.
- Unsuffixed geometry is VV. All actual intervention geometry and native dimensions also have _vv and _vh versions. center_x/y is W//2,H//2 for the coordinate-adapter contract, not an assumed object center. bbox_center_x/y preserves the actual object center.

## Mapping evidence

Legacy rows join the preserved base manifest by instance_id. Port, product, class, MMSI, chip basename and nonempty geometry are cross-checked. The existing crop generator build_v5_context_chips.py computes target_center=(left+right)/2 and (top+bottom)/2. Thus legacy scene bounds are exactly center +/- recorded width/height / 2, then the recorded crop origin is subtracted. No chip-center assumption is used. The unavailable historical F: location is not required for this derivation.

Returned rows join original server manifests by (computer,obj_uid,polarization), with identity/product/port/class/chip checks. The 12,435 later local-knowledge rows store the server in the original server column instead of return_server_vv/vh; resolved_server_vv/vh records that recovery. Original left/top/right/bottom, crop bounds, source JSON/TIFF paths and object IDs are retained. Returned reconstructed edges are checked against original edges after subtracting crop origin. The earlier source_returned_metadata.csv contains condition summaries, not the localization edges, so it is not used as a localization substitute.

Every actual PNG header is read and compared with the descriptor cache joined by sample_id. No image features, GPU, model fitting, old-file edits or shared-memory updates occur. Cached dimension agreement is a mapping check; no pixel-content identity claim is made.

## Bbox mask definition and limits

source_region_metadata.py exposes bbox_regions(bounds, (H,W), padding). Native bbox bounds are pixel edges with exclusive right/bottom. Each side is padded by max(3 native pixels, padding * bbox long side), with padding .20 or .35. Integer lower edges use floor, upper edges ceil, and boxes are clipped to native extent. Unlike the old endpoint rule, no +1 is added to an already exclusive bbox edge. Region and equal-area, nonoverlapping farthest-corner sham boxes are included for both widths/polarizations. A zero sham box means sham_valid=false, not a valid control.

These are bbox-neighborhood interventions, not an exact replication of OSS2 head/tail-length neighborhoods. No endpoint is invented from a diagonal, center, heading or bbox. Detection boxes may include sea, sidelobes or wake and do not establish physical target/background separation. Bounding-box validity does not establish detector accuracy. Head/tail availability outside the traced manifests was not exhaustively searched.

VV/VH must be transformed in their own native coordinate frames before the existing model resize. Their boxes and extents can differ. Pixel-pair shuffle cannot be defined directly on unequal native lattices; the parent must explicitly choose its common-grid implementation and report the changed intervention semantics. This sidecar does not implement features or evaluator adaptation.

## Archived task coverage

The 23 existing target.npz archives were checked by both rows and ids, with five 64-sample supports and disjoint query masks. They cover 38,027 samples. Melbourne's 64 rows remain in this full sidecar but have b64_task_eligible=false: B64 leaves zero queries. Exact excluded IDs, manifest row indices, classes and identities are in source_region_metadata.json / excluded_b64_samples and marked in the CSV. This is a target-task exclusion, not a universal training exclusion.

## Measured checks

Additional bounded verification passed: exact CSV round-trip preservation of every original source column and sample order; empty head/tail fields; hand-calculated shifted and clipped bbox cases for .20/.35 padding and exclusive edges; invalid bounds; absence of an equal-area nonoverlapping sham for a full-image box. There are 198 rows with unknown identity, retained as empty rather than assigned invented vessel IDs.

- Valid paired bboxes: 38091 / 38,091.
- Rows with missing original crop fields recovered: 12435.
- Unequal native polarization shapes: 6871.
- VV: {"bbox_valid": 38091, "maximum_center_offset": 28.5, "native_shape_counts": {"104x104": 325, "112x112": 156, "120x120": 57, "128x128": 46, "136x136": 1, "152x152": 1, "184x184": 1, "64x64": 27784, "72x72": 4403, "80x80": 2393, "88x88": 1791, "96x96": 1133}, "native_size_differs_from_requested_side": 0, "non_chip_center_rows": 28592, "sham_valid": 6700, "wide_sham_valid": 1533}
- VH: {"bbox_valid": 38091, "maximum_center_offset": 28.50438562747845, "native_shape_counts": {"104x104": 453, "112x112": 282, "120x120": 123, "128x128": 203, "64x64": 27248, "72x72": 4246, "80x80": 2378, "88x88": 1942, "96x96": 1216}, "native_size_differs_from_requested_side": 0, "non_chip_center_rows": 28676, "sham_valid": 6441, "wide_sham_valid": 1324}

Run: `D:/Program_files/anaconda3/python.exe -B source_region_metadata.py`. Generated files are only source_region_metadata.csv, source_region_metadata.json and this report next to the script. The JSON is a coverage/provenance receipt; sample records are in CSV. No next study is proposed or launched.
