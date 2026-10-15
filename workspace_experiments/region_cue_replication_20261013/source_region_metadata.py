"""Read-only local24 localization recovery; writes only this sidecar's outputs.

No head/tail annotations exist in the input manifests. Bboxes remain bboxes.
Unsuffixed geometry is VV; use the suffixed fields for each native polarization.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from PIL import Image

OUT = Path(__file__).resolve().parent
WORK = OUT.parent
SOURCE = WORK / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv"
DATA = Path("E:/SAR_AIS_T_shipchip_dataset")
BASE = DATA / "v5_context_dynamic_64_128_merged_20260915/metadata/base_manifest_order.csv"
RETURN = DATA / "v5_context_dynamic_64_128_return_20260915/metadata"
ARCHIVE = Path("E:/Hermes/paper_tracking/sensitivity")
DESC = ARCHIVE / "c4_condition_diagnosis_20261009/descriptors/self24ports.csv"
TASKS = ARCHIVE / "learned_frequency_residual_20261009/tasks/self24ports"
GEOMETRY = ("crop_left", "crop_top", "crop_side", "target_center_x",
            "target_center_y", "width", "height")


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def same(a, b, numeric=False):
    if numeric:
        np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=1e-7)
    else:
        np.testing.assert_array_equal(np.asarray(a, str), np.asarray(b, str))


def bbox_regions(bounds, shape, padding):
    """BBox-only rule: pad each edge by max(3, padding * bbox long side).

    Bounds are native pixel edges, with exclusive right/bottom (no extra +1).
    Returns target box, equal-area corner sham, validity, sham validity.
    """
    h, w = shape
    l, t, r, b = map(float, bounds)
    valid = all(map(math.isfinite, (l, t, r, b))) and 0 <= l < r <= w and 0 <= t < b <= h
    if not valid:
        return (0, 0, 0, 0), (0, 0, 0, 0), False, False
    pad = max(3., padding * max(r-l, b-t))
    box = (max(0, math.floor(l-pad)), max(0, math.floor(t-pad)),
           min(w, math.ceil(r+pad)), min(h, math.ceil(b+pad)))
    x0, y0, x1, y1 = box
    bw, bh = x1-x0, y1-y0
    candidates = [(0, 0, bw, bh), (w-bw, 0, w, bh),
                  (0, h-bh, bw, h), (w-bw, h-bh, w, h)]
    cx, cy = (x0+x1)/2, (y0+y1)/2
    candidates.sort(key=lambda p: -(((p[0]+p[2])/2-cx)**2 + ((p[1]+p[3])/2-cy)**2))
    for p in candidates:
        if min(x1, p[2]) <= max(x0, p[0]) or min(y1, p[3]) <= max(y0, p[1]):
            return box, p, True, True
    return box, (0, 0, 0, 0), True, False


def native_shape(path):
    with Image.open(path) as image:
        return image.height, image.width


def recover(source):
    base = read(BASE).set_index("instance_id", verify_integrity=True)
    returned = []
    for path in sorted(RETURN.glob("server*/manifest.csv")):
        part = read(path)
        part["metadata_path"] = str(path)
        returned.append(part)
    returned = pd.concat(returned, ignore_index=True)
    returned = returned.set_index(["computer", "obj_uid", "polarization"], verify_integrity=True)
    output = source.copy()
    output["sample_id"] = source.instance_id
    output["manifest_row"] = np.arange(len(source))
    output["identity"] = source.mmsi.str.replace(r"\.0$", "", regex=True)
    output["localization_kind"] = "detection_bbox"
    output["headtail_valid"] = False
    diagnostics = {}
    legacy = source.data_origin.eq("legacy")
    for pol in ("vv", "vh"):
        recovered = pd.DataFrame(index=source.index)
        original = base.loc[source.loc[legacy, "instance_id"]].reset_index(drop=True)
        original.index = source.index[legacy]
        for field in ("port", "product", "class_name"):
            same(source.loc[legacy, field], original[field])
        same(source.loc[legacy, "mmsi"].str.replace(r"\.0$", "", regex=True),
             original.mmsi.str.replace(r"\.0$", "", regex=True))
        same(source.loc[legacy, "image_relpath_"+pol].map(lambda p: Path(p).name),
             original["image_relpath_"+pol].map(lambda p: Path(p).name))
        for field in GEOMETRY:
            recovered.loc[legacy, field] = original[field+"_"+pol].astype(float)
        servers = source["return_server_"+pol].where(source["return_server_"+pol].ne(""), source.server)
        output["resolved_server_"+pol] = servers
        keys = pd.MultiIndex.from_arrays([servers.loc[~legacy],
                                         source.loc[~legacy, "obj_uid_"+pol],
                                         np.repeat(pol.upper(), (~legacy).sum())])
        original_return = returned.loc[keys].reset_index()
        original_return.index = source.index[~legacy]
        for field in ("port", "product", "class_name"):
            same(source.loc[~legacy, field], original_return[field])
        same(source.loc[~legacy, "mmsi"].str.replace(r"\.0$", "", regex=True),
             original_return.mmsi.str.replace(r"\.0$", "", regex=True))
        same(source.loc[~legacy, "image_relpath_"+pol].map(lambda p: Path(p).name),
             original_return.context_output_png.map(lambda p: Path(p).name))
        for field in GEOMETRY:
            recovered.loc[~legacy, field] = original_return[field].astype(float)
            present = source[field+"_"+pol].ne("")
            same(source.loc[present, field+"_"+pol], recovered.loc[present, field], numeric=True)
            output["resolved_"+field+"_"+pol] = recovered[field]
        for field in ("source_json", "source_tiff", "source_object_index", "obj_uid",
                      "left", "top", "right", "bottom", "crop_right", "crop_bottom",
                      "position_method", "redetect_status", "geometry_confidence"):
            output["returned_"+field+"_"+pol] = ""
            output.loc[~legacy, "returned_"+field+"_"+pol] = original_return[field]
        cx = recovered.target_center_x-recovered.crop_left
        cy = recovered.target_center_y-recovered.crop_top
        bounds = pd.DataFrame(dict(x0=cx-recovered.width/2, y0=cy-recovered.height/2,
                                   x1=cx+recovered.width/2, y1=cy+recovered.height/2))
        for local_field, raw_field, offset in (("x0", "left", "crop_left"), ("y0", "top", "crop_top"),
                                              ("x1", "right", "crop_left"), ("y1", "bottom", "crop_top")):
            same(bounds.loc[~legacy, local_field],
                 original_return[raw_field].astype(float)-original_return[offset].astype(float), numeric=True)
            output["bbox_"+local_field+"_"+pol] = bounds[local_field]
            output["scene_bbox_"+local_field+"_"+pol] = bounds[local_field]+recovered[offset]
        for axis, values in (("x", cx), ("y", cy)):
            present = source["target_chip_center_"+axis+"_"+pol].ne("")
            same(source.loc[present, "target_chip_center_"+axis+"_"+pol], values[present], numeric=True)
            output["bbox_center_"+axis+"_"+pol] = values
        output["provenance_"+pol] = str(BASE)+"; join=instance_id; bbox=center+width/height; subtract=crop_origin"
        output.loc[~legacy, "provenance_"+pol] = (original_return.metadata_path+
            "; join=(computer,obj_uid,polarization); bbox=original_left/top/right/bottom; subtract=crop_origin")
        output["path_"+pol] = source["image_relpath_"+pol]
        print("Reading native PNG headers:", pol, len(source), flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            shapes = np.asarray(list(pool.map(native_shape, output["path_"+pol])))
        output["native_h_"+pol], output["native_w_"+pol] = shapes[:, 0], shapes[:, 1]
        # Compare cached image-header dimensions by sample ID, never by row position alone.
        desc = read(DESC).set_index("sample_id", verify_integrity=True).loc[output.sample_id]
        same(shapes[:, 0], desc["native_height_px_"+pol], numeric=True)
        same(shapes[:, 1], desc["native_width_px_"+pol], numeric=True)
        output["center_x_"+pol], output["center_y_"+pol] = shapes[:, 1]//2, shapes[:, 0]//2
        for name in ("head_x", "head_y", "tail_x", "tail_y"):
            output[name+"_"+pol] = np.nan
        for padding, label in ((.20, "region"), (.35, "wide_region")):
            regions = [bbox_regions(b, s, padding) for b, s in zip(bounds.to_numpy(), shapes)]
            for pos, field in enumerate(("x0", "y0", "x1", "y1")):
                output[label+"_"+field+"_"+pol] = [r[0][pos] for r in regions]
                output[label.replace("region", "sham")+"_"+field+"_"+pol] = [r[1][pos] for r in regions]
            output[label+"_valid_"+pol] = [r[2] for r in regions]
            output[label.replace("region", "sham")+"_valid_"+pol] = [r[3] for r in regions]
        output["bbox_valid_"+pol] = output["region_valid_"+pol]
        diagnostics[pol] = dict(
            non_chip_center_rows=int(((cx != shapes[:, 1]//2) | (cy != shapes[:, 0]//2)).sum()),
            maximum_center_offset=float(np.hypot(cx-shapes[:, 1]//2, cy-shapes[:, 0]//2).max()),
            native_size_differs_from_requested_side=int(((shapes[:, 0] != recovered.crop_side) |
                                                        (shapes[:, 1] != recovered.crop_side)).sum()),
            bbox_valid=int(output["bbox_valid_"+pol].sum()),
            sham_valid=int(output["sham_valid_"+pol].sum()),
            wide_sham_valid=int(output["wide_sham_valid_"+pol].sum()),
            native_shape_counts={f"{h}x{w}": int(n) for (h,w), n in
                                 output.groupby(["native_h_"+pol, "native_w_"+pol]).size().items()})
        output = output.copy()
    for field in ("head_x", "head_y", "tail_x", "tail_y", "center_x", "center_y", "native_h", "native_w",
                  "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "provenance"):
        output[field] = output[field+"_vv"]
    output["region_valid"] = output.bbox_valid_vv & output.bbox_valid_vh
    output["bbox_valid"] = output.region_valid
    output["native_pair_same_shape"] = (output.native_h_vv.eq(output.native_h_vh) &
                                         output.native_w_vv.eq(output.native_w_vh))
    return output, diagnostics


def task_coverage(frame):
    tasks = []
    for path in sorted(TASKS.glob("*/target.npz")):
        port = path.parent.name
        with np.load(path, allow_pickle=False) as z:
            expected = frame.loc[frame.port.eq(port)]
            same(z["ids"], frame.iloc[z["rows"]].sample_id)
            same(np.sort(z["rows"]), expected.manifest_row.to_numpy())
            assert z["support"].shape == (5, 64), path
            assert z["eval_masks"].shape == (5, len(expected)), path
            for draw, support in enumerate(z["support"]):
                assert len(np.unique(support)) == 64 and not z["eval_masks"][draw, support].any(), path
            tasks.append(dict(port=port, rows=len(expected), archive=str(path),
                              query_rows=[int(a.sum()) for a in z["eval_masks"]]))
    excluded = frame.loc[~frame.port.isin([t["port"] for t in tasks])]
    assert set(excluded.port) == {"Melbourne"} and len(excluded) == 64
    assert len(tasks) == 23 and sum(t["rows"] for t in tasks) == 38027
    return tasks, excluded[["sample_id", "manifest_row", "port", "class_name", "identity"]].to_dict("records")


def main():
    source = read(SOURCE)
    assert source.instance_id.is_unique and len(source) == 38091
    frame, diagnostics = recover(source)
    tasks, excluded = task_coverage(frame)
    frame["b64_task_eligible"] = frame.port.isin([t["port"] for t in tasks])
    frame["b64_exclusion_reason"] = np.where(frame.b64_task_eligible, "", "Melbourne has 64 rows: B64 leaves zero queries")
    same(frame.sample_id, source.instance_id)
    receipt = dict(status="complete", rows=len(frame), ports=frame.port.nunique(),
        source_manifest=str(SOURCE), source_columns_preserved=source.columns.tolist(),
        headtail_valid=0, bbox_pair_valid=int(frame.region_valid.sum()),
        native_shape_mismatch_pairs=int((~frame.native_pair_same_shape).sum()),
        original_missing_crop_rows=int(source.crop_left_vv.eq("").sum()),
        joins={"legacy": "instance_id plus port/product/class/MMSI/path checks",
               "returned": "(computer,obj_uid,polarization) plus port/product/class/MMSI/path checks",
               "native_headers": "all 76182 PNG headers read; cached dimensions joined by sample_id",
               "tasks": "archive row indices AND sample IDs agree for all 23 tasks"},
        geometry=diagnostics, b64_tasks=tasks, excluded_b64_samples=excluded,
        region_rule="bbox pixel-edge bounds padded max(3,.20*max(width,height)); wide .35; clipped to native extent",
        comparison_limit="Detection bboxes, not head/tail annotations or vessel masks. Bbox long side is not head-tail length.",
        unsuffixed_geometry="VV native frame; center_x/y = native_w/h // 2 are coordinate adapter values, not object center",
        runner_requirement="Use bbox_regions or generated per-polarization boxes. Existing head-tail region_boxes cannot consume missing endpoints. Process polarizations independently before model resize; do not stack unequal native shapes.",
        source_bbox_derivation="Legacy center+width/height in base manifest; crop generator computes center=(left+right)/2. Returned original edges exactly cross-checked.",
        changed_original_files=False, gpu_used=False)
    ordered = ["sample_id", "region_valid", "head_x", "head_y", "tail_x", "tail_y", "center_x", "center_y",
               "native_h", "native_w", "identity", "port", "provenance"]
    frame = frame[ordered+[c for c in frame.columns if c not in ordered]]
    frame.to_csv(OUT / "source_region_metadata.csv", index=False, encoding="utf-8-sig")
    (OUT / "source_region_metadata.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    report = """# Local24 Source Region Audit

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

"""
    report += f"- Valid paired bboxes: {receipt['bbox_pair_valid']} / 38,091.\n"
    report += f"- Rows with missing original crop fields recovered: {receipt['original_missing_crop_rows']}.\n"
    report += f"- Unequal native polarization shapes: {receipt['native_shape_mismatch_pairs']}.\n"
    for pol, values in diagnostics.items():
        report += f"- {pol.upper()}: {json.dumps(values, sort_keys=True)}\n"
    report += "\nRun: `D:/Program_files/anaconda3/python.exe -B source_region_metadata.py`. Generated files are only source_region_metadata.csv, source_region_metadata.json and this report next to the script. The JSON is a coverage/provenance receipt; sample records are in CSV. No next study is proposed or launched.\n"
    (OUT / "SOURCE_REGION_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("status", "rows", "ports", "bbox_pair_valid", "headtail_valid",
                                            "native_shape_mismatch_pairs", "original_missing_crop_rows")}), flush=True)


if __name__ == "__main__":
    main()
