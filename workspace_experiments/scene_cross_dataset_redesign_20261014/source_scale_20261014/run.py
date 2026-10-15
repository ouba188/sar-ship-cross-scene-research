"""Separate legal OSS2 SSL data exposure from optimization steps on one fixed episode."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
EPISODE = ROOT / "public_expansion" / "oss2" / "S1A_009518_00DD1D"
FIT = EPISODE / "fit"
PIXELS = Path("D:/SAR_completion_cache_20261010/oss2_inputs128.npy")
MANIFEST = Path("E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/target_manifest.csv")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "f0921_split"))
from train_source_r18 import SSL, augment, encode, nt_xent, source_ba  # noqa: E402
from evaluate_pilot import support_score  # noqa: E402
from general_domain_readout import fit_head, metrics  # noqa: E402


SEEDS = (17, 29, 43)
FRACTIONS = (.25, .5, 1.)
SHORT_STEPS = 70
LONG_STEPS = 280
BATCH = 64


def nested_subsets(train, index):
    ships = train.groupby("mmsi", sort=False).class_id.nunique()
    assert ships.eq(1).all()
    by_class = train.groupby("class_id").mmsi.unique()
    ordered = {}
    for class_id, values in by_class.items():
        values = np.sort(values.astype(str))
        ordered[class_id] = np.random.default_rng(20261014 + int(class_id)).permutation(values)
    subsets = {}
    details = []
    for fraction in FRACTIONS:
        selected = set()
        for values in ordered.values():
            selected.update(values[:max(1, round(len(values) * fraction))])
        frame = train[train.mmsi.isin(selected)]
        rows = index.loc[frame.sample_id].to_numpy(np.int64)
        subsets[fraction] = rows
        details.append({
            "fraction": fraction, "rows": len(rows), "ships": frame.mmsi.nunique(),
            "acquisitions": frame.acquisition.nunique(),
            "classes": frame.class_id.nunique(),
            "class_rows": json.dumps(frame.class_id.value_counts().sort_index().to_dict()),
            "product_rows": json.dumps(frame["product"].value_counts().sort_index().to_dict()),
        })
    assert set(subsets[.25]) <= set(subsets[.5]) <= set(subsets[1.])
    return subsets, pd.DataFrame(details)


def load_episode():
    metadata = pd.read_csv(MANIFEST, dtype={"mmsi_or_ship_id": str})
    ids = metadata.sample_id.to_numpy(str)
    np.testing.assert_array_equal(ids, np.load("D:/SAR_completion_cache_20261010/oss2_ids.npy").astype(str))
    index = pd.Series(np.arange(len(ids)), index=ids)
    train = pd.read_csv(EPISODE / "source_train.csv", dtype={"mmsi": str})
    validation = pd.read_csv(EPISODE / "source_validation.csv", dtype={"mmsi": str})
    query = pd.read_csv(EPISODE / "query.csv", dtype={"mmsi": str})
    candidates = pd.read_csv(EPISODE / "support_candidates.csv", dtype={"mmsi": str})
    draws = pd.read_csv(EPISODE / "support_draws.csv", dtype={"mmsi": str})
    assert set(train.class_id) == set(validation.class_id) == set(range(8))
    assert not set(train.acquisition) & set(validation.acquisition)
    assert not (set(train.acquisition) | set(validation.acquisition)) & (
        set(query.acquisition) | set(candidates.acquisition))
    assert not (set(train.mmsi) | set(validation.mmsi)) & (
        set(query.mmsi) | set(candidates.mmsi))
    assert draws.groupby(["draw", "support_origin"]).size().eq(16).all()
    with np.load(FIT / "pilot_features.npz") as cached:
        rows = cached["rows"]
        g50 = cached["G50"]
        g18 = cached["G18"]
    required = np.r_[index.loc[train.sample_id], index.loc[validation.sample_id],
                     index.loc[query.sample_id], index.loc[draws.sample_id]]
    assert set(required) <= set(rows)
    local = lambda sample_ids: np.searchsorted(rows, index.loc[sample_ids].to_numpy(np.int64))
    source = local(pd.concat((train.sample_id, validation.sample_id), ignore_index=True))
    source_train = local(train.sample_id)
    source_val = local(validation.sample_id)
    target = local(query.sample_id)
    labels = metadata.class_id.to_numpy(int)[rows]
    support = {(draw, origin): local(draws[(draws.draw == draw)
                                         & (draws.support_origin == origin)].sample_id)
               for draw in range(5) for origin in ("same", "cross")}
    assert all(len(values) == 16 for values in support.values())
    return metadata, index, train, rows, source, source_train, source_val, target, labels, support, g50, g18


def pca_features(raw, source):
    projector = PCA(n_components=128, svd_solver="randomized", random_state=20261014)
    return projector.fit(raw[source]).transform(raw).astype(np.float32)


def controls(g50_raw, g18_raw, source, target, support, labels):
    g50 = pca_features(g50_raw, source)
    g18 = pca_features(g18_raw, source)
    base_head = fit_head(g50, labels, source)
    imagenet = (g50, g18)
    imagenet_head = fit_head(imagenet, labels, source)
    result = {}
    for key, indices in {("none", -1): None, **{
            (origin, draw): values for (draw, origin), values in support.items()}}.items():
        result[("G50", *key)] = support_score(g50, base_head, target, indices, labels)
        result[("G50_G18", *key)] = support_score(imagenet, imagenet_head, target, indices, labels)
    return g50, result


def evaluate(model, pixels, rows, source, source_train, source_val, target,
             labels, support, g50, control, seed, fraction, step):
    raw = encode(model.backbone, pixels, rows)
    source_validation = source_ba(raw[source_train], raw[source_val],
                                  labels[source_train], labels[source_val])
    d18 = pca_features(raw, source)
    joint = (g50, d18)
    joint_head = fit_head(joint, labels, source)
    records = []
    for origin, draw in [("none", -1)] + [(origin, draw) for origin in ("same", "cross")
                                           for draw in range(5)]:
        indices = None if origin == "none" else support[(draw, origin)]
        score = support_score(joint, joint_head, target, indices, labels)
        fused = (.5 * softmax(control[("G50", origin, draw)], axis=1)
                 + .5 * softmax(score, axis=1))
        for mode, candidate in (("G50_D18", score), ("fused_G50_D18", fused)):
            records.append({"seed": seed, "fraction": fraction, "steps": step,
                            "support_origin": origin, "draw": draw, "mode": mode,
                            "source_val_ba": 100 * source_validation,
                            "query_rows": len(target),
                            **metrics(labels[target], candidate.argmax(1))})
    return records


def train_one(seed, fraction, subset, pixels, rows, source, source_train,
              source_val, target, labels, support, g50, control, extend_half=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SSL("r18").cuda()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 3e-5},
        {"params": model.projection.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LONG_STEPS)
    rng = np.random.default_rng(seed + 20261014)
    queue = np.empty(0, dtype=np.int64)
    records = []
    losses = []
    maximum = LONG_STEPS if fraction == 1. or extend_half else SHORT_STEPS
    started = time.perf_counter()
    for step in range(1, maximum + 1):
        while len(queue) < BATCH:
            queue = np.r_[queue, rng.permutation(subset)]
        batch, queue = queue[:BATCH], queue[BATCH:]
        pair = torch.from_numpy(np.array(pixels[batch], dtype=np.float32)).cuda() / 255.
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(torch.cat((augment(pair), augment(pair))))
        loss = nt_xent(z)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at seed={seed} fraction={fraction} step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if step == SHORT_STEPS or (maximum == LONG_STEPS and step in (140, LONG_STEPS)):
            checkpoint = evaluate(model, pixels, rows, source, source_train, source_val,
                                  target, labels, support, g50, control,
                                  seed, fraction, step)
            records.extend(checkpoint)
            print(json.dumps({"seed": seed, "fraction": fraction, "steps": step,
                              "loss_recent": float(np.mean(losses[-SHORT_STEPS:])),
                              "source_val_ba": checkpoint[0]["source_val_ba"],
                              "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--half-long", action="store_true")
    args = parser.parse_args()
    (metadata, index, train, rows, source, source_train, source_val, target,
     labels, support, g50_raw, g18_raw) = load_episode()
    subsets, details = nested_subsets(train, index)
    HERE.mkdir(parents=True, exist_ok=True)
    details.to_csv(HERE / "subset_details.csv", index=False)
    print(details.to_string(index=False), flush=True)
    if args.preflight:
        return
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    pixels = np.load(PIXELS, mmap_mode="r")
    g50, control = controls(g50_raw, g18_raw, source, target, support, labels)
    baseline = []
    for (mode, origin, draw), score in control.items():
        baseline.append({"mode": mode, "support_origin": origin, "draw": draw,
                         "query_rows": len(target), **metrics(labels[target], score.argmax(1))})
    pd.DataFrame(baseline).to_csv(HERE / "fixed_controls.csv", index=False)
    results = []
    for seed in SEEDS:
        for fraction in ((.5,) if args.half_long else FRACTIONS):
            results.extend(train_one(seed, fraction, subsets[fraction], pixels, rows,
                                     source, source_train, source_val, target, labels,
                                     support, g50, control, extend_half=args.half_long))
            pd.DataFrame(results).to_csv(HERE / ("half_long_results.csv" if args.half_long else "results.csv"), index=False)
    (HERE / ("half_long_protocol.json" if args.half_long else "protocol.json")).write_text(json.dumps({
        "episode": str(EPISODE), "seeds": SEEDS, "fractions": FRACTIONS,
        "short_steps": SHORT_STEPS, "long_steps": LONG_STEPS, "batch": BATCH,
        "lr_schedule": "cosine T_max=280 for every run; no LR restart",
        "half_long": args.half_long,
        "subset_unit": "nested source MMSI within source class; all observations kept",
        "source_fit": "same full 2252 train + 528 validation rows for every readout",
        "selection": "fixed checkpoints 70/140/280; no target-label model selection",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
