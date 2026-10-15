"""Use independently recovered bboxes, preserving source manifest order."""

from replication_context import *


def main():
    frame = pd.read_csv(OUT / "source_region_metadata.csv", low_memory=False, keep_default_na=False)
    source = pd.read_csv(SOURCE / "manifest.csv", keep_default_na=False, low_memory=False)
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str), source.instance_id.to_numpy(str))
    fields = ["sample_id", "class_id", "class_name", "identity", "port", "product", "region_valid",
              "native_pair_same_shape", "path_vv", "path_vh", "native_h", "native_w"]
    fields += [f"{key}_{pol}" for pol in ("vv", "vh") for key in
               ("native_h", "native_w", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")]
    frame = frame[fields].copy()
    frame["scene_id"] = frame["product"]
    frame["noise_xml"] = ""
    frame["dataset_name"] = "ports"
    folder = OUT / "ports"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "frame.csv", index=False)
    write(folder / "LOCALIZATION.json", dict(rows=len(frame), valid=int(frame.region_valid.sum()),
        native_pair_unequal=int((~frame.native_pair_same_shape).sum()),
        definition="Actual per-polarization detection bbox, pad max3px/.20longside; wider .35",
        headtail_annotations_available=False, strict_same_mask_replication=False))
    print("PORTS_LOCALIZATION", len(frame), flush=True)


if __name__ == "__main__":
    main()
