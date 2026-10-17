"""Detect every vessel candidate in native 24-port chips, without AIS inputs."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = ROOT / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv"
WEIGHTS = ROOT / "v5_server_toolkit_20260912/models/550k.pt"


def decode_result(result, width, height, ox, oy):
    raw = result.obb.xywhr.cpu().numpy().copy()
    corners = result.obb.xyxyxyxy.cpu().numpy().copy()
    confidences = result.obb.conf.cpu().numpy()
    raw[:, :2] -= [ox, oy]
    corners -= [ox, oy]
    boundary = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    records = []
    for box, polygon, confidence in zip(raw, corners, confidences):
        if not np.isfinite(box).all() or min(box[2:4]) <= 0:
            continue
        area, _ = cv2.intersectConvexConvex(polygon.astype(np.float32), boundary)
        fraction = float(area / (box[2] * box[3]))
        center_inside = 0 <= box[0] < width and 0 <= box[1] < height
        if not center_inside and (area < 4 or fraction < .1):
            continue
        length, beam = sorted(map(float, box[2:4]), reverse=True)
        edges = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        # Official corners independently catch incorrect width/height-angle coupling.
        np.testing.assert_allclose(sorted(edges[:2]), [beam, length], atol=1e-3, rtol=1e-4)
        records.append(dict(xywhr=box.tolist(), corners=polygon.tolist(), confidence=float(confidence),
                            length_px=length, beam_px=beam, center_inside=bool(center_inside),
                            visible_fraction=min(1., max(0., fraction)),
                            shape_usable=bool(length >= 4 and beam >= 1.5 and length / beam <= 25),
                            touches_edge=bool((polygon[:, 0] < 1).any() or (polygon[:, 0] > width-1).any()
                                              or (polygon[:, 1] < 1).any() or (polygon[:, 1] > height-1).any())))
    return sorted(records, key=lambda r: -r["confidence"])


def run(args):
    torch.set_num_threads(2)
    frame = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    frame.insert(0, "global_row", np.arange(len(frame)))
    if args.rows_file:
        selected = set(pd.read_csv(args.rows_file)["global_row"].astype(int))
        frame = frame[frame.global_row.isin(selected)]
    if args.pilot:
        # Metadata-only, fixed coverage across all ports, independent of detections.
        frame = pd.concat([g.iloc[np.unique(np.linspace(0, len(g)-1, 4, dtype=int))]
                           for _, g in frame.groupby("port", sort=True)])
    stage = "pilot" if args.pilot else (Path(args.rows_file).stem if args.rows_file else "full")
    outdir = HERE / stage / f"canvas{args.canvas}"
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "detections.jsonl"
    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            done = {(r["instance_id"], r["pol"]) for r in map(json.loads, stream)}
    jobs = [(row, pol) for row in frame.to_dict("records") for pol in ("vv", "vh")
            if (row["instance_id"], pol) not in done]
    config = dict(manifest=str(MANIFEST), weights=str(WEIGHTS), canvas=args.canvas,
                  confidence=.08, nms_iou=.30, native_pixels_unscaled=True,
                  target_labels_or_AIS_used=False, padding="zero, native chip centered",
                  full_scene_context=False, model_names={0: "Ship"})
    (outdir / "PROTOCOL.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = YOLO(str(WEIGHTS), task="obb")
    started = time.perf_counter()
    with output.open("a", encoding="utf-8") as stream:
        for start in range(0, len(jobs), args.batch):
            batch = jobs[start:start + args.batch]
            images, contexts = [], []
            for row, pol in batch:
                with Image.open(row["image_relpath_" + pol]) as im:
                    native = np.asarray(im.convert("L"))
                h, w = native.shape
                if max(h, w) > args.canvas:
                    raise ValueError("Native chip larger than selected detector canvas")
                ox, oy = (args.canvas-w)//2, (args.canvas-h)//2
                canvas = np.zeros((args.canvas, args.canvas, 3), np.uint8)
                canvas[oy:oy+h, ox:ox+w] = native[:, :, None]
                images.append(canvas)
                contexts.append((w, h, ox, oy))
            results = model.predict(source=images, imgsz=args.canvas, conf=.08, iou=.30,
                                    device=0, verbose=False, save=False, max_det=300)
            for (row, pol), result, (w, h, ox, oy) in zip(batch, results, contexts):
                record = dict(global_row=int(row["global_row"]), instance_id=row["instance_id"],
                              pol=pol, native_w=w, native_h=h, image=row["image_relpath_"+pol],
                              detections=decode_result(result, w, h, ox, oy))
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            if start % (args.batch*20) == 0 or start + args.batch >= len(jobs):
                processed = min(start+args.batch, len(jobs))
                elapsed = time.perf_counter()-started
                print(json.dumps(dict(processed=processed, total=len(jobs), seconds=round(elapsed, 1),
                                      remaining_seconds=round(elapsed/processed*(len(jobs)-processed), 1))), flush=True)
    summary = dict(images_total=len(frame)*2, images_this_run=len(jobs),
                   seconds_this_run=time.perf_counter()-started, **config)
    (outdir / "DONE.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--canvas", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--rows-file")
    run(parser.parse_args())
