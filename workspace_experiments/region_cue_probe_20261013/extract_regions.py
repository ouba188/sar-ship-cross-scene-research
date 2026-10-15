"""Extract only raw R50 and R18 used by the expert, leaving C4/main files intact."""

from region_context import *
import time
import torch
from PIL import Image, ImageDraw
import extract_inputs as legacy
from transforms_regions import transform, read_pair, region_boxes, box_mask


def prepared(pair):
    images = [Image.fromarray(a) for a in pair]
    raw = np.stack([np.asarray(im.resize((128, 128), Image.Resampling.BILINEAR)) for im in images])
    return raw, legacy.TF(images[1].convert("RGB"))


def forward(batch, paired, ssl):
    x = torch.from_numpy(np.stack([b[0] for b in batch])).cuda().float()
    x = (x/255.-.5)/.25
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        spatial = paired(x).flatten(2).transpose(1, 2)
    raw = spatial.float().cpu().numpy().astype(np.float16).astype(np.float32).mean(1)
    groups = {}
    for i, b in enumerate(batch):
        groups.setdefault(tuple(b[1].shape), []).append(i)
    extra = np.empty((len(batch), 512), np.float32)
    for indices in groups.values():
        extra[indices] = ssl(torch.stack([batch[i][1] for i in indices]).cuda()).flatten(1).cpu().numpy()
    return dict(raw=raw, extra=extra)


def prepare_metadata(frame):
    records = []
    for i, row in enumerate(frame.itertuples()):
        pair = read_pair(row)
        _, info = transform(pair, row, "original", i)
        info["sample_id"] = row.sample_id
        info["native_h"], info["native_w"] = pair.shape[1:]
        records.append(info)
    table = pd.DataFrame(records)
    table.to_csv(OUT / "region_metadata.csv", index=False)
    return table


def preview(frame):
    selected = []
    for product in ("GRDH", "SLC"):
        part = frame[frame.scene_id.str.contains("_"+product+"_", regex=False)]
        selected.extend(part.groupby("class_id", sort=True).head(1).index.tolist())
    columns = ("original", "target_flat", "outer_flat", "sham_flat", "target_shuffle", "outer_shuffle", "target_display", "outer_display")
    canvas = Image.new("RGB", (len(columns)*140, len(selected)*158), "white")
    draw = ImageDraw.Draw(canvas)
    for r, i in enumerate(selected):
        row = frame.iloc[i]
        pair = read_pair(row)
        for c, condition in enumerate(columns):
            result, info = transform(pair, row, condition, i)
            im = Image.fromarray(result[1]).convert("RGB").resize((132, 132), Image.Resampling.NEAREST)
            if condition == "original" and info["region_valid"]:
                local = ImageDraw.Draw(im)
                s = 132/pair.shape[-1]
                local.rectangle(tuple(int(info[k]*s) for k in ("x0", "y0", "x1", "y1")), outline="red", width=1)
            canvas.paste(im, (c*140, r*158+23))
            draw.text((c*140+2, r*158+2), f"{r}:{condition}", fill="black")
    canvas.save(OUT / "region_preview.png")
    frame.iloc[selected].to_csv(OUT / "preview_rows.csv", index=False)


def extract():
    data, frame = cohort()
    table = prepare_metadata(frame)
    preview(frame)
    models = legacy.encoders()
    take = np.linspace(0, len(frame)-1, 32, dtype=int)
    with torch.inference_mode():
        values = forward([prepared(read_pair(frame.iloc[i])) for i in take], *models)
    replay = {k: legacy.differences(v, data[k][take]) for k, v in values.items()}
    if any(v["max_relative_l2"] > (.001 if k == "extra" else .02) for k, v in replay.items()):
        write("FEATURE_REPLAY_FAILED.json", replay)
        raise ValueError("Original encoder preprocessing replay failed")
    write("FEATURE_REPLAY.json", dict(indices=take.tolist(), errors=replay))
    print("REPLAY", json.dumps(replay), flush=True)
    for condition in CONDITIONS[1:]:
        folder = OUT / "features" / condition
        if (folder / "COMPLETE.json").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.lib.format.open_memmap(folder / (k+".npy"), mode="w+", dtype="float32", shape=(len(frame), d))
                  for k, d in (("raw", 2048), ("extra", 512))}
        records = []
        started = time.perf_counter()
        for lo in range(0, len(frame), 32):
            batch = []
            for i in range(lo, min(lo+32, len(frame))):
                row = frame.iloc[i]
                result, info = transform(read_pair(row), row, condition, i)
                batch.append(prepared(result))
                records.append(dict(sample_id=row.sample_id, **info))
            with torch.inference_mode():
                values = forward(batch, *models)
            for k, v in values.items():
                if not np.isfinite(v).all():
                    raise ValueError("Nonfinite encoder output")
                arrays[k][lo:lo+len(batch)] = v
            if lo % 512 == 0 or lo+32 >= len(frame):
                print("FEATURES", condition, min(lo+32, len(frame)), len(frame), round(time.perf_counter()-started, 1), flush=True)
        for value in arrays.values():
            value.flush()
        np.save(folder / "ids.npy", data["ids"])
        pd.DataFrame(records).to_csv(folder / "controls.csv", index=False)
        write(str(Path("features") / condition / "COMPLETE.json"), dict(rows=len(frame), seconds=time.perf_counter()-started))
        del arrays
    write("FEATURES_COMPLETE.json", dict(rows=len(frame), region_valid=int(table.region_valid.sum()),
        sham_valid=int(table.sham_valid.sum()), conditions=CONDITIONS, new_neural_training=False))


if __name__ == "__main__":
    extract()
