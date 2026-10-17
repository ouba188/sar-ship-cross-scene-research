# Independent crop/export code review

Date: 2026-10-16. Scope: read-only review of the three requested exporters and necessary local geometry/context references. No source edits, training, inference, web access, or inspection of the target objects' original JSON/TIFF metadata. This report is the only written artifact.

## Result and attribution

No additional explicit image rotation, transpose, x/y exchange, or width/height exchange was found in the requested crop/export implementations. Their array and window coordinate conventions are internally consistent.

The upstream detector geometry defect is separate: `obb.xywhr` width/height are sorted without adjusting the angle. The main agent reports verifying that the actual local Ultralytics `regularize_rboxes` permits `w<h`, establishing reachability. The main agent owns the regression test, root-script fix, and original-chip detector probe. This review does not independently claim to have inspected that Ultralytics installation or attributed the defect to either target object.

- Root detector location supplied by the main agent: `E:/临时会话/sea_ship_redetect_preview.py:299-310`.
- Independently inspected toolkit location: `E:/临时会话/v5_server_toolkit_20260912/scripts/sea_ship_redetect_preview.py:300-311`. Line 305 sorts the dimensions; lines 309-310 retain the original angle.
- Independent numerical check using the toolkit's actual `axis_aligned_box` function: raw `(w=10,h=20,angle=80 degrees)` projects to padded `(26.790796,16.651301)`; sorting to `(20,10)` at the same angle projects to `(16.651301,26.790796)`. This establishes the geometric consequence, not the target samples' provenance.

## Matched-chip exporters: exact field origins

For this section, A is `E:/临时会话/scripts/export_v5_matched_chips.py`; B is `E:/临时会话/v5_server_toolkit_20260912/scripts/export_v5_matched_chips.py`.

A direct file diff found only the `canonical_port` implementation differs (A:93-95; B:103-105 lies within the expanded block B:93-105). All geometry/export logic is identical, with B's later line numbers offset by +10.

| Operation | A lines | B lines | Meaning |
| --- | --- | --- | --- |
| Pair TIFF with JSON | 125-126, 217-219 | 135-136, 227-229 | Same filename stem; JSON `image` must match TIFF filename. |
| Select objects | 228-232 | 238-242 | Only `meta.redetect_status == matched_recentered`. |
| Read geometry | 156-160 | 166-170 | `cx,cy,width,height = cxcywh_angle[:4]`; no reading of `pts`, AIS dimensions, or `sar_center_refine`. |
| Compute left/top | 165-166 | 175-176 | `max(0,floor(cx-width/2))`, `max(0,floor(cy-height/2))`. |
| Compute right/bottom | 167-168 | 177-178 | `min(dataset.width,ceil(cx+width/2))`, `min(dataset.height,ceil(cy+height/2))`. |
| Read crop | 246-248 | 256-258 | `Window(left,top,right-left,bottom-top)`; first offset is column/x, second is row/y. |
| Save PNG | 250-252 | 260-262 | `Image.fromarray(data).save(...)`, followed by rename. No resize, rotation, transpose, or reprojection call. |
| Write manifest geometry | 269-274 | 279-284 | Bounds are the same crop bounds; `width=right-left`, `height=bottom-top`. |

Thus manifest width/height are the integer, image-clipped axis-aligned crop extents, not necessarily the original floating-point dimensions. Right/bottom are exclusive crop endpoints. A fresh export with manifest width 14 and height 26 produces a PNG of width 14 and height 26, not a 64x64 PNG.

Ignoring angle is not an additional demonstrated defect in the supported matched-object path: toolkit `sea_ship_redetect_preview.py:227-232` converts detection OBB extents to a padded axis-aligned box and returns angle zero; `:569-585` writes that box into `pts` and `cxcywh_angle`, with `redetect_angle_policy=axis_aligned_for_chip_crop`; `:657-660` uses this function for matched objects. The upstream dimension-sorting defect can therefore already be embedded in an angle-zero JSON box before export.

The local tiling helper also does not exchange axes: `E:/临时会话/v5_server_toolkit_20260912/scripts/tile_sarlabelimg_quadrants.py:200-207` subtracts x offset from x and y offset from y, retaining width/height/angle. This is a code-path observation, not confirmation that these target samples passed through that helper.

## Requested pilot context exporter

File: `E:/临时会话/sarais_pilot_20260904/export_context_v5.py`.

- Lines 54-56 load the per-chip JSON's first object and select source TIFF from `chip_meta.source_image_path`, falling back to the CSV's `source_image`.
- Lines 57-59 use `meta.sar_center_refine.selected_candidate.global_cx/global_cy`; if either is absent, both coordinates are reconstructed as `crop_rect_xyxy[:2] + cxcywh_angle[:2]`. The fallback adds x to x and y to y.
- Lines 32-40 crop a square in source pixels: `(x0,y0) = round((cx,cy)-side/2)`, with `Window(sx0,sy0,sx1-sx0,sy1-sy0)` and destination indexing `[y_slice,x_slice]`. Outside-image regions are zero-padded. No axis exchange or interpolation occurs.
- Lines 63-66 request fixed side lengths 64 and 128 and save `Image.fromarray(chip)`. The uint8 cast at line 39 changes dtype, not geometry.
- Line 67 writes `context_px`, `center_xy`, and `crop_rect_xyxy`; it does not write the matched exporter's `left/top/right/bottom/width/height` columns. The returned context rectangle is the requested square, including any out-of-image padded extent.

This is a different schema and center-selection path from the matched exporter. Code inspection alone does not show it generated the supplied target PNGs. Correctness of its source-image coordinate frame and selected center depends on upstream metadata, which the main agent is examining.

## Related manifest-driven context builder

Necessary local reference: `E:/临时会话/v5_server_toolkit_20260912/scripts/build_v5_context_chips.py`.

- Lines 29-34 and 286 preserve the input manifest's original bbox bounds and width/height in the context manifest.
- Lines 70-72 set defaults `min_side=64`, `scale=1.8`, `max_side=128`. Lines 159-166 compute a square side from the longer original box side and round upward to a multiple of 8. A 14x26 bbox produces side 64 under these defaults.
- Lines 269-274 compute center from the original bbox endpoints and request a square source-image window. Lines 169-180 shift the square inward near image edges; they do not rotate it or swap axes.
- Lines 265 and 275 read the original TIFF using Pillow and call `image.crop((left,top,right,bottom))`. Lines 283-284 save the crop without explicit resizing or rotation.
- Lines 290-296 write separate `crop_left/top/right/bottom`, `crop_side`, and `target_center_x/y` fields.

Consequently, a 64x64 context PNG and manifest bbox width 14 / height 26 are compatible by construction. The bbox describes the original target box; `crop_*` describes the context window. This compatibility does not prove the original target box fits the visible ship.

### TIFF decoding boundary

Absence of explicit rotation in the context builder is not proof against all decoder-driven orientation changes. The currently available local Pillow source at `C:/Users/25791/AppData/Roaming/Python/Python314/site-packages/PIL/TiffImagePlugin.py:1328-1330` calls `ImageOps.exif_transpose` while finishing TIFF loading; lines 1468-1472 swap reported dimensions for Orientation values 5-8.

No target TIFF Orientation tag or historical Pillow version was inspected here. This is an attribution boundary, not a second confirmed reachable project defect. It matters specifically to the Pillow TIFF reader in the related builder; the requested three exporters read source pixels with Rasterio and create new Pillow images from arrays.

## Existing-output boundary

The matched exporter reuses completed outputs by manifest path (A:180-184, 247, 280-284; B:190-194, 257, 290-294). Re-running after changing source geometry does not refresh those completed PNGs or manifest rows. The related context builder likewise skips completed UIDs (:236-240) and existing destination PNGs (:283-284). Therefore, current source code/JSON alone cannot establish the generation history of an existing image. This is not evidence of a rotation and was not shown to occur for these objects.

## Verification and limits

Read-only, memory-only checks used AST extraction of the actual functions, without importing their export entry points or writing bytecode/output files:

1. Both `bounds` implementations returned `(33,23,47,49)` for center `(40,36)` and dimensions `(14,26)` in a 96x80 image. An asymmetric coordinate-pattern array saved/reopened through Pillow PNG retained shape `(26,14)` and exact pixel order.
2. The pilot `crop` function preserved pixel order for a centered 64x64 window and correctly placed pixels/zero padding for a top-left-edge window. This used an in-memory dataset/Window stand-in with the expected row/column contract.
3. The upstream numerical projection comparison is recorded above. No detector model was run.

The default interpreter lacks Rasterio and Ultralytics; no packages were installed. The checks therefore do not constitute real Rasterio/TIFF integration tests or validation of historical library behavior.

For tanker 353491000, VV obj_uid 40341c4507ef0b68 and VH obj_uid 32e92f49fbefd327, the supplied observation of horizontal bright structure versus bbox width 14 / height 26 does not, by itself, prove a crop-axis bug. The review identifies no additional explicit crop/export axis bug. The verified upstream sorting defect provides a concrete mechanism, but object-specific causality still requires the main agent's raw geometry, metadata, and probe evidence.
