"""Source-only SimCLR R18 training for the strict OSS1 pilot."""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "f0921_split"))
from train_source_r18 import SSL, augment, encode, nt_xent, source_ba  # noqa: E402

SEED = 20261014
PAIRS = 64
EPOCHS = 2


def main():
    train = pd.read_csv(HERE / "source_train.csv", dtype={"mmsi": str})
    val = pd.read_csv(HERE / "source_validation.csv", dtype={"mmsi": str})
    ssl = pd.read_csv(HERE / "ssl.csv", dtype={"mmsi": str})
    query = pd.read_csv(ROOT / "public_split/oss1_query.csv", dtype={"mmsi": str})
    support = pd.read_csv(ROOT / "public_split/oss1_support_candidates.csv", dtype={"mmsi": str})
    assert set(ssl.sample_id) == set(train.sample_id)
    assert "class_id" not in ssl
    assert not (set(query.acquisition) | set(support.acquisition)) & set(ssl.acquisition)
    assert not (set(query.mmsi) | set(support.mmsi)) & set(ssl.mmsi)
    assert not set(train.mmsi) & set(val.mmsi)
    assert set(val.class_id) == set(range(8))
    index = pd.read_csv(HERE / "pixel_index.csv", usecols=["sample_id"])
    positions = pd.Series(np.arange(len(index)), index=index.sample_id)
    train_rows = positions.loc[ssl.sample_id].to_numpy(np.int64)
    val_rows = positions.loc[val.sample_id].to_numpy(np.int64)
    labels = np.full(len(index), -1, dtype=int)
    labels[train_rows] = train.set_index("sample_id").loc[ssl.sample_id, "class_id"].to_numpy(int)
    labels[val_rows] = val.class_id.to_numpy(int)
    pixels = np.load(HERE / "pixels128.npy", mmap_mode="r")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = SSL("r18").cuda()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 3e-5},
        {"params": model.projection.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    steps = len(train_rows) // PAIRS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * steps)
    started = time.perf_counter()
    records = []
    best = (-1., None)
    for epoch in range(EPOCHS + 1):
        if epoch:
            model.train()
            order = np.random.default_rng(SEED + epoch).permutation(train_rows)[:steps * PAIRS]
            losses = []
            for step in range(steps):
                batch = order[step * PAIRS:(step + 1) * PAIRS]
                pair = torch.from_numpy(np.array(pixels[batch], dtype=np.float32)).cuda() / 255.
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
        a = encode(model.backbone, pixels, train_rows)
        b = encode(model.backbone, pixels, val_rows)
        ba = source_ba(a, b, labels[train_rows], labels[val_rows])
        record = dict(epoch=epoch, source_val_ba=ba,
                      loss=float(np.mean(losses)) if epoch else None,
                      elapsed_seconds=round(time.perf_counter() - started, 1))
        records.append(record)
        print(json.dumps(record), flush=True)
        if epoch and ba > best[0]:
            best = (ba, epoch)
            torch.save({"backbone": {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
                        "epoch": epoch, "source_val_ba": ba, "target_ssl_rows": 0},
                       HERE / "source_only_r18_best.pt")
        (HERE / "training_curve.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (HERE / "training_complete.json").write_text(json.dumps({
        "selected_epoch": best[1], "selected_source_val_ba": best[0],
        "source_train_rows": len(train_rows), "source_validation_rows": len(val_rows),
        "source_val_class_counts": np.bincount(labels[val_rows], minlength=8).tolist(),
        "target_ssl_rows": 0, "target_query_labels_used_for_selection": False,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
