"""Reuse exact encoders/transforms, loading native pairs once per batch."""

from replication_context import *
import time
import argparse
import torch
from PIL import Image, ImageDraw
import extract_regions as old_extract
from transforms_regions import region_boxes
import port_transforms


def archive_features(dataset):
    if dataset == "ports":
        return dict(raw=np.load(SOURCE / "pooled_features.npy", mmap_mode="r"),
            extra=np.load(CROSS.ROOT / "neural_complement_pilot_20261007/frozen_replay/source_extra.npy", mmap_mode="r"))
    return rc.run.target(dataset, "original")


def metadata(frame, folder, dataset):
    records = []
    for row in frame.itertuples():
        if dataset == "ports":
            regions = port_transforms.boxes(row, .20)
            box = regions[1][0]
            areas = [(b[2]-b[0])*(b[3]-b[1])/(getattr(row, "native_h_"+pol)*getattr(row, "native_w_"+pol))
                     for pol, (b, _, _, _) in zip(("vv", "vh"), regions)]
            records.append(dict(sample_id=row.sample_id, region_valid=all(r[2] for r in regions),
                sham_valid=all(r[3] for r in regions), region_area=float(np.mean(areas)),
                region_area_vv=areas[0], region_area_vh=areas[1],
                native_h=row.native_h_vh, native_w=row.native_w_vh,
                x0=box[0], y0=box[1], x1=box[2], y1=box[3]))
            continue
        box, sham, valid, sham_valid, length, points = region_boxes(row, (int(row.native_h), int(row.native_w)), .20)
        if bool(row.region_valid) != valid:
            raise ValueError("Localization adapter/transform disagree: " + row.sample_id)
        records.append(dict(sample_id=row.sample_id, region_valid=valid, sham_valid=sham_valid,
            region_area=(box[2]-box[0])*(box[3]-box[1])/(row.native_h*row.native_w),
            native_h=row.native_h, native_w=row.native_w, headtail_length=length,
            x0=box[0], y0=box[1], x1=box[2], y1=box[3],
            hx=points[0, 0], hy=points[0, 1], tx=points[1, 0], ty=points[1, 1]))
    table = pd.DataFrame(records)
    table.to_csv(folder / "region_metadata.csv", index=False)
    return table


def preview(frame, folder, dataset):
    chosen = frame[frame.region_valid].groupby("class_id", sort=True).head(1).index.to_list()
    conditions = ("original", "target_flat", "sham_flat", "target_shuffle", "outer_shuffle", "target_display", "outer_display")
    canvas = Image.new("RGB", (140*len(conditions), 158*len(chosen)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, i in enumerate(chosen):
        row = frame.iloc[i]
        pair = port_transforms.read_images(row) if dataset == "ports" else read_pair(row)
        for c, condition in enumerate(conditions):
            a, info = (port_transforms.transform_images if dataset == "ports" else transform)(pair, row, condition, i)
            im = Image.fromarray(a[1]).convert("RGB").resize((132, 132), Image.Resampling.NEAREST)
            if condition == "original":
                d = ImageDraw.Draw(im)
                s = 132/a[1].shape[1]
                d.rectangle(tuple(int(info[k]*s) for k in ("x0", "y0", "x1", "y1")), outline="red")
            canvas.paste(im, (140*c, 158*r+23))
            draw.text((140*c+2, 158*r+2), f"{row.class_id}:{condition}", fill="black")
    canvas.save(folder / "preview.png")
    frame.iloc[chosen].to_csv(folder / "preview_rows.csv", index=False)


def extract(dataset):
    folder = OUT / dataset
    features = feature_root(dataset)
    frame = frame_for(dataset)
    table = metadata(frame, folder, dataset)
    preview(frame, folder, dataset)
    reader = port_transforms.read_images if dataset == "ports" else read_pair
    intervention = port_transforms.transform_images if dataset == "ports" else transform
    prepare = (lambda pair: port_transforms.prepared(pair, old_extract.legacy.TF)) if dataset == "ports" else old_extract.prepared
    old = archive_features(dataset)
    models = old_extract.legacy.encoders()
    take = np.linspace(0, len(frame)-1, 32, dtype=int)
    with torch.inference_mode():
        output = old_extract.forward([prepare(reader(frame.iloc[i])) for i in take], *models)
    errors = {k: old_extract.legacy.differences(v, old[k][take]) for k, v in output.items()}
    write(folder / "SPARSE_REPLAY.json", errors)
    if any(v["max_relative_l2"] > (.001 if k == "extra" else .02) for k, v in errors.items()):
        raise ValueError("Wrong preprocessing/encoder in sparse replay")
    print("SPARSE_REPLAY", dataset, errors, flush=True)
    pending = [c for c in ("replay_original", *CONDITIONS[1:-1])
               if not (features / c / "COMPLETE.json").exists()]
    arrays, records = {}, {c: [] for c in pending}
    for c in pending:
        dst = features / c
        dst.mkdir(parents=True, exist_ok=True)
        arrays[c] = {k: np.lib.format.open_memmap(dst / (k+".npy"), mode="w+", dtype="float32", shape=(len(frame), dim))
                     for k, dim in (("raw", 2048), ("extra", 512))}
    started = time.perf_counter()
    if pending:
        for lo in range(0, len(frame), 32):
            rows = list(frame.iloc[lo:lo+32].itertuples())
            pairs = [reader(row) for row in rows]
            for c in pending:
                batch = []
                for j, (row, pair) in enumerate(zip(rows, pairs)):
                    a, info = intervention(pair, row, c, lo+j)
                    batch.append(prepare(a))
                    records[c].append(dict(sample_id=row.sample_id, **info))
                with torch.inference_mode():
                    values = old_extract.forward(batch, *models)
                for key, value in values.items():
                    if not np.isfinite(value).all():
                        raise ValueError("Nonfinite region features")
                    arrays[c][key][lo:lo+len(rows)] = value
            if lo % 512 == 0 or lo+32 >= len(frame):
                print("EXTRACT", dataset, min(lo+32, len(frame)), len(frame),
                      len(pending), round(time.perf_counter()-started, 1), flush=True)
        for c in pending:
            dst = features / c
            for a in arrays[c].values():
                a.flush()
            np.save(dst / "ids.npy", frame.sample_id.to_numpy(str))
            pd.DataFrame(records[c]).to_csv(dst / "controls.csv", index=False)
            write(dst / "COMPLETE.json", dict(rows=len(frame), native_read_shared_across_conditions=True))
    replay = {k: old_extract.legacy.differences(
        np.load(features / "replay_original" / (k+".npy"), mmap_mode="r"), old[k]) for k in ("raw", "extra")}
    write(folder / "FEATURES_COMPLETE.json", dict(rows=len(frame), valid=int(table.region_valid.sum()),
        sham_valid=int(table.sham_valid.sum()), conditions=CONDITIONS, full_feature_replay=replay,
        seconds=time.perf_counter()-started, neural_training=False, feature_root=str(features),
        primary_reference="Same-pipeline reextracted original; archived original separately replayed"))
    print("FEATURES_COMPLETE", dataset, replay, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("oss1", "ports"))
    extract(parser.parse_args().dataset)
