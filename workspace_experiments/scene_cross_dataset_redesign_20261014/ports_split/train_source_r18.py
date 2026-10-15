"""Two-epoch source-port-only R18 SSL pilot for frozen 24-port splits."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "f0921_split"))
from train_source_r18 import SSL, augment, encode, nt_xent, source_ba  # noqa: E402


MANIFEST = Path("E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv")
PIXELS = Path("D:/SAR_completion_cache_20261010/source_inputs128.npy")
SEED = 20261014
PAIRS = 64
EPOCHS = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", choices=("Rotterdam", "Singapore"), required=True)
    args = parser.parse_args()
    folder = ROOT / args.port
    with np.load(folder / "split.npz") as z:
        train, validation, target = z["ssl_train"], z["source_validation"], z["target_acquisition"]
    metadata = pd.read_csv(MANIFEST, usecols=["class_id", "port", "product", "mmsi"], dtype={"mmsi": str})
    parts = metadata["product"].str.split("_")
    acquisition = parts.map(lambda x: "_".join((x[0], x[6], x[7])))
    assert not set(acquisition.iloc[train]) & set(acquisition.iloc[target])
    assert not set(metadata.mmsi.iloc[train]) & set(metadata.mmsi.iloc[target])
    assert not set(metadata.mmsi.iloc[train]) & set(metadata.mmsi.iloc[validation])
    assert not metadata.port.iloc[train].eq(args.port).any()
    pixels = np.load(PIXELS, mmap_mode="r")
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
    steps = len(train) // PAIRS
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps * EPOCHS)
    started = time.perf_counter()
    records = []
    best = (-1., None)
    for epoch in range(EPOCHS + 1):
        if epoch:
            model.train()
            order = np.random.default_rng(SEED + epoch).permutation(train)[:steps * PAIRS]
            losses = []
            for step in range(steps):
                rows = order[step * PAIRS:(step + 1) * PAIRS]
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
                schedule.step()
                losses.append(float(loss.detach()))
        a = encode(model.backbone, pixels, train)
        b = encode(model.backbone, pixels, validation)
        ba = source_ba(a, b, labels[train], labels[validation])
        record = dict(epoch=epoch, source_val_ba=ba,
                      loss=float(np.mean(losses)) if epoch else None,
                      elapsed_seconds=round(time.perf_counter() - started, 1))
        records.append(record)
        print(json.dumps(record), flush=True)
        if epoch and ba > best[0]:
            best = (ba, epoch)
            torch.save({"backbone": {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
                        "epoch": epoch, "source_val_ba": ba, "target_ssl_rows": 0},
                       folder / "source_only_r18_best.pt")
        (folder / "training_curve.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (folder / "training_complete.json").write_text(json.dumps({
        "selected_epoch": best[1], "selected_source_val_ba": best[0],
        "source_train_rows": len(train), "source_validation_rows": len(validation),
        "source_val_class_counts": np.bincount(labels[validation], minlength=8).tolist(),
        "target_ssl_rows": 0, "target_query_labels_used_for_selection": False,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
