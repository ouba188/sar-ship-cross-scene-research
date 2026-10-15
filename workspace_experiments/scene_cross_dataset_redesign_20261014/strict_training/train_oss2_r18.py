"""Train a source-only R18 on the frozen OSS2 acquisition-held pilot."""

import json
from pathlib import Path
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "f0921_split"))
from train_source_r18 import SSL, augment, encode, nt_xent, source_ba  # noqa: E402


SPLIT = ROOT / "public_split"
MANIFEST = Path("E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/target_manifest.csv")
PIXELS = Path("D:/SAR_completion_cache_20261010/oss2_inputs128.npy")
SEED = 20261014
PAIRS = 64
EPOCHS = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fold", choices=("original", "balanced"), default="original")
    parser.add_argument("--episode-dir", type=Path)
    parser.add_argument("--dataset", choices=("oss1", "oss2"), default="oss2")
    args = parser.parse_args()
    if args.episode_dir and args.source_fold != "original":
        parser.error("--source-fold balanced is only available for the original pilot")
    out = (args.episode_dir / "fit" if args.episode_dir else
           ROOT / "strict_training" / ("oss2_v2" if args.source_fold == "balanced" else "oss2"))
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = (Path(f"E:/Hermes/paper_tracking/sensitivity/opensarship{args.dataset[-1]}_transfer_20261007/target_manifest.csv")
                     if args.episode_dir else MANIFEST)
    pixel_path = (ROOT / "public_expansion" / "oss1_inputs128.npy"
                  if args.episode_dir and args.dataset == "oss1" else PIXELS)
    ids_path = (ROOT / "public_expansion" / "oss1_ids.npy"
                if args.episode_dir and args.dataset == "oss1" else
                Path("D:/SAR_completion_cache_20261010/oss2_ids.npy"))
    manifest = pd.read_csv(manifest_path, usecols=["sample_id", "class_id"])
    ids = manifest.sample_id.to_numpy(str)
    np.testing.assert_array_equal(ids, np.load(ids_path).astype(str))
    index = pd.Series(np.arange(len(ids)), index=ids)
    split = args.episode_dir if args.episode_dir else SPLIT
    prefix = "" if args.episode_dir else "oss2_"
    train = pd.read_csv(split / f"{prefix}source_train.csv", dtype={"mmsi": str})
    validation = pd.read_csv(split / f"{prefix}source_validation.csv", dtype={"mmsi": str})
    if args.source_fold == "balanced":
        pool = pd.concat((train, validation), ignore_index=True)
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_index, val_index = next(folds.split(pool, pool.class_id, pool.acquisition))
        validation = pool.iloc[val_index].copy()
        train = pool.iloc[train_index].copy()
        train = train[~train.mmsi.isin(set(validation.mmsi))]
        train.to_csv(out / "source_train.csv", index=False)
        validation.to_csv(out / "source_validation.csv", index=False)
    query = pd.read_csv(split / f"{prefix}query.csv", dtype={"mmsi": str})
    candidates = pd.read_csv(split / f"{prefix}support_candidates.csv", dtype={"mmsi": str})
    source_rows = index.loc[train.sample_id].to_numpy(np.int64)
    val_rows = index.loc[validation.sample_id].to_numpy(np.int64)
    held_acq = set(query.acquisition) | set(candidates.acquisition)
    assert not held_acq & (set(train.acquisition) | set(validation.acquisition))
    assert not (set(query.mmsi) | set(candidates.mmsi)) & (set(train.mmsi) | set(validation.mmsi))
    assert not set(train.mmsi) & set(validation.mmsi)
    pixels = np.load(pixel_path, mmap_mode="r")
    labels = manifest.class_id.to_numpy(int)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    model = SSL("r18").cuda()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 3e-5},
        {"params": model.projection.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    steps = len(source_rows) // PAIRS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * steps)
    started = time.perf_counter()
    records = []
    best = (-1., None)
    for epoch in range(EPOCHS + 1):
        if epoch:
            model.train()
            order = np.random.default_rng(SEED + epoch).permutation(source_rows)[:steps * PAIRS]
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
        a = encode(model.backbone, pixels, source_rows)
        b = encode(model.backbone, pixels, val_rows)
        ba = source_ba(a, b, labels[source_rows], labels[val_rows])
        record = dict(epoch=epoch, source_val_ba=ba,
                      loss=float(np.mean(losses)) if epoch else None,
                      elapsed_seconds=round(time.perf_counter() - started, 1))
        records.append(record)
        print(json.dumps(record), flush=True)
        if epoch and ba > best[0]:
            best = (ba, epoch)
            torch.save({"backbone": {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
                        "epoch": epoch, "source_val_ba": ba, "target_ssl_rows": 0},
                       out / "source_only_r18_best.pt")
        (out / "training_curve.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out / "training_complete.json").write_text(json.dumps({
        "selected_epoch": best[1], "selected_source_val_ba": best[0],
        "source_train_rows": len(source_rows), "source_validation_rows": len(val_rows),
        "source_val_class_counts": np.bincount(labels[val_rows], minlength=8).tolist(),
        "source_fold": args.source_fold,
        "episode_dir": str(args.episode_dir) if args.episode_dir else None,
        "dataset": args.dataset,
        "target_ssl_rows": 0, "target_query_labels_used_for_selection": False,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
