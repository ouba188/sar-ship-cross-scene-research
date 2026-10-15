"""Small strict source-only R18 SSL pilot for a frozen F_0921 water holdout."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "E:/Hermes/paper_tracking/sensitivity/oof_reliability_check")
from general_domain_ssl import SSL, augment, nt_xent, normalize, rgb  # noqa: E402


PIXELS = Path("E:/Hermes/paper_tracking/sensitivity/f0921_detector_recrop_20261007/inputs128.npy")
METADATA = ROOT.parents[1] / "scene_port_recovery_20261014" / "f0921_water_regions.csv"
SEED = 20261014
PAIRS = 64
EPOCHS = 2
BATCH = 64


@torch.inference_mode()
def encode(backbone, pixels, rows):
    backbone.eval()
    values = np.empty((len(rows), 512), dtype=np.float32)
    for start in range(0, len(rows), BATCH):
        batch_rows = rows[start:start + BATCH]
        pair = torch.from_numpy(np.array(pixels[batch_rows], dtype=np.float32)).cuda() / 255.
        pair = F.interpolate(pair, size=(112, 112), mode="bilinear", align_corners=False, antialias=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = backbone(normalize(rgb(pair)))
        values[start:start + len(batch_rows)] = features.float().cpu().numpy()
    return values


def source_ba(train_features, val_features, train_labels, val_labels):
    mean = train_features.mean(0)
    scale = train_features.std(0).clip(1e-5)
    x = np.column_stack(((train_features - mean) / scale, np.ones(len(train_features))))
    v = np.column_stack(((val_features - mean) / scale, np.ones(len(val_features))))
    counts = np.bincount(train_labels, minlength=8)
    weights = 1. / counts[train_labels]
    weights /= weights.mean()
    target = np.eye(8)[train_labels]
    gram = x.T @ (weights[:, None] * x)
    gram.flat[::len(gram) + 1] += 1.
    coef = np.linalg.solve(gram, x.T @ (weights[:, None] * target))
    predicted = (v @ coef).argmax(1)
    return float(np.mean([
        (predicted[val_labels == cls] == cls).mean()
        for cls in range(8) if np.any(val_labels == cls)
    ]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=("Gulf_of_Mexico", "California_Baja_Pacific"), required=True)
    args = parser.parse_args()
    folder = ROOT / args.region
    split = np.load(folder / "split.npz")
    train_rows, val_rows = split["ssl_train"], split["source_validation"]
    pixels = np.load(PIXELS, mmap_mode="r")
    import pandas as pd
    metadata = pd.read_csv(METADATA, usecols=["class_id", "acquisition", "mmsi", "water_region"], dtype={"mmsi": str})
    target_rows = split["target_acquisition"]
    assert not set(metadata.acquisition.iloc[train_rows]) & set(metadata.acquisition.iloc[target_rows])
    assert not set(metadata.mmsi.iloc[train_rows]) & set(metadata.mmsi.iloc[target_rows])
    assert not set(metadata.mmsi.iloc[train_rows]) & set(metadata.mmsi.iloc[val_rows])
    labels = metadata.class_id.to_numpy(int)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    model = SSL("r18").cuda()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 3e-5},
        {"params": model.projection.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    steps = len(train_rows) // PAIRS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps * EPOCHS)
    records = []
    best_ba, best_epoch = -1., None
    started = time.perf_counter()
    for epoch in range(EPOCHS + 1):
        if epoch:
            model.train()
            shuffled = np.random.default_rng(SEED + epoch).permutation(train_rows)[:steps * PAIRS]
            losses = []
            for step in range(steps):
                rows = shuffled[step * PAIRS:(step + 1) * PAIRS]
                pair = torch.from_numpy(np.array(pixels[rows], dtype=np.float32)).cuda() / 255.
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model(torch.cat((augment(pair), augment(pair))))
                loss = nt_xent(z)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite SSL loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
                if (step + 1) % 50 == 0:
                    print(f"epoch={epoch} step={step + 1}/{steps} loss={losses[-1]:.4f}", flush=True)
        train_features = encode(model.backbone, pixels, train_rows)
        val_features = encode(model.backbone, pixels, val_rows)
        ba = source_ba(train_features, val_features, labels[train_rows], labels[val_rows])
        record = {"epoch": epoch, "source_val_ba": ba,
                  "ssl_loss": float(np.mean(losses)) if epoch else None,
                  "seconds_elapsed": round(time.perf_counter() - started, 1)}
        records.append(record)
        print(json.dumps(record), flush=True)
        if epoch and ba > best_ba:
            best_ba, best_epoch = ba, epoch
            torch.save({"backbone": {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
                        "epoch": epoch, "source_validation_ba": ba,
                        "source_train_rows": len(train_rows), "target_ssl_rows": 0},
                       folder / "source_only_r18_best.pt")
        (folder / "training_curve.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (folder / "training_complete.json").write_text(json.dumps({
        "epochs": EPOCHS, "steps_per_epoch": steps, "best_positive_epoch": best_epoch,
        "best_source_validation_ba": best_ba, "image_size": 112,
        "input": "paired VV/VH -> RGB(VV,VH,mean)", "target_ssl_rows": 0,
        "target_query_labels_used_for_selection": False,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
