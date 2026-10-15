"""Train one 24-port-only R18 for the frozen F_0921/OSS2 external episodes."""

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "f0921_split"))
from train_source_r18 import SSL, augment, encode, nt_xent, source_ba  # noqa: E402


SOURCE = ROOT.parent / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv"
PIXELS = Path("D:/SAR_completion_cache_20261010/source_inputs128.npy")
F_META = ROOT.parent / "scene_port_recovery_20261014/f0921_water_regions.csv"
OSS2_META = ROOT.parent / "external_scene_transfer_20261013/oss2_metadata.csv"
SOURCE_ROWS = ROOT.parent / "external_scene_transfer_20261013/source_rows.npy"
SEED = 20261014
EPOCHS = 2
PAIRS = 64


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(SOURCE, usecols=["mmsi", "product", "class_id"], dtype={"mmsi": str})
    pixels = np.load(PIXELS, mmap_mode="r")
    assert len(metadata) == len(pixels)
    source = np.load(SOURCE_ROWS).astype(np.int64)
    assert len(source) == len(np.unique(source))
    acquisition = metadata["product"].str.split("_").map(
        lambda x: "|".join((x[0], x[6], x[7])))
    f = pd.read_csv(F_META, usecols=["mmsi", "acquisition"], dtype=str)
    o = pd.read_csv(OSS2_META, usecols=["identity", "acquisition"], dtype=str)
    external_mmsi = set(f.mmsi.dropna()) | set(o.identity.dropna())
    external_acq = set(f.acquisition.dropna()) | set(o.acquisition.dropna())
    assert not set(metadata.mmsi.iloc[source]) & external_mmsi
    assert not set(acquisition.iloc[source]) & external_acq

    source_frame = metadata.iloc[source]
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(folds.split(source_frame, source_frame.class_id, acquisition.iloc[source]))
    validation = source[va]
    train = source[tr]
    train = train[~metadata.mmsi.iloc[train].isin(set(metadata.mmsi.iloc[validation]))]
    assert set(metadata.class_id.iloc[validation]) == set(range(8))
    assert not set(acquisition.iloc[train]) & set(acquisition.iloc[validation])
    assert not set(metadata.mmsi.iloc[train]) & set(metadata.mmsi.iloc[validation])
    np.savez_compressed(OUT / "source_split.npz", ssl_train=train, source_validation=validation,
                        source_fit=source)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps * EPOCHS)
    started = time.perf_counter()
    records = []
    best = (-1., None)
    for epoch in range(EPOCHS + 1):
        if epoch:
            model.train()
            order = np.random.default_rng(SEED + epoch).permutation(train)[:steps * PAIRS]
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
        train_features = encode(model.backbone, pixels, train)
        val_features = encode(model.backbone, pixels, validation)
        ba = source_ba(train_features, val_features, labels[train], labels[validation])
        record = {"epoch": epoch, "source_val_ba": ba,
                  "loss": float(np.mean(losses)) if epoch else None,
                  "seconds_elapsed": round(time.perf_counter() - started, 1)}
        records.append(record)
        print(json.dumps(record), flush=True)
        if epoch and ba > best[0]:
            best = (ba, epoch)
            torch.save({"backbone": {k: v.detach().cpu()
                                      for k, v in model.backbone.state_dict().items()},
                        "epoch": epoch, "source_val_ba": ba, "target_ssl_rows": 0},
                       OUT / "source_only_r18_best.pt")
        (OUT / "training_curve.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (OUT / "training_complete.json").write_text(json.dumps({
        "source_rows": len(source), "source_train_rows": len(train),
        "source_validation_rows": len(validation),
        "source_validation_classes": np.bincount(labels[validation], minlength=8).tolist(),
        "selected_epoch": best[1], "selected_source_val_ba": best[0],
        "target_ssl_rows": 0, "target_query_labels_used_for_selection": False,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
