"""Freeze train-only readouts and disjoint context-gate development episodes."""

import argparse
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import torch

from full_readout import CONTEXT, LOCAL, MANIFEST, build_model, load, source_gallery
from run import class_map, source_tasks, learn_relation

HERE = Path(__file__).resolve().parent
OUT = HERE / "body_context"
SEED = 20261015


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def balanced_rows(frame, rng, size):
    shuffled = frame.iloc[rng.permutation(len(frame))].drop_duplicates("mmsi")
    pools = [g.index.to_list() for _, g in shuffled.groupby("class_id", sort=True)]
    selected = []
    while pools and len(selected) < size:
        for pool in pools:
            if pool and len(selected) < size:
                selected.append(pool.pop())
        pools = [pool for pool in pools if pool]
    return selected


def make_tasks(frame, seed, draws=5):
    rng = np.random.default_rng(seed)
    tasks, skipped = [], []
    for (port, acq), group in frame.groupby(["port", "acquisition"], sort=True):
        cross = frame.loc[frame.port.eq(port) & frame.acquisition.ne(acq)]
        if group.mmsi.nunique() <= 16 or cross.mmsi.nunique() < 16:
            skipped.append(dict(port=port, acquisition=acq, ships=group.mmsi.nunique(),
                                cross_ships=cross.mmsi.nunique()))
            continue
        for draw in range(draws):
            same = balanced_rows(group, rng, 16)
            candidates = group.loc[~group.mmsi.isin(frame.loc[same, "mmsi"])]
            query = candidates.iloc[rng.permutation(len(candidates))].drop_duplicates("mmsi")
            query = query.iloc[:32]
            other = balanced_rows(cross.loc[~cross.mmsi.isin(query.mmsi)], rng, 16)
            if len(other) < 16 or not len(query):
                continue
            for origin, support in (("same", same), ("cross", other)):
                assert not set(query.mmsi) & set(frame.loc[support, "mmsi"])
                tasks.append(dict(port=port, acquisition=acq, draw=draw, origin=origin,
                                  q=query.index.to_list(), s=support))
    return tasks, skipped


def prepare(port, episodes, manifest):
    start = time.perf_counter()
    slug = port.replace(" ", "_")
    out = OUT / slug
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prepared.json").exists():
        print("cached", slug, flush=True)
        return
    first = episodes[0]
    folder = Path(first["folder"])
    data = load(port, first["fold"], folder.parent)
    train_frame = pd.read_csv(folder / "source_train.csv", dtype={"mmsi": str})
    val_frame = pd.read_csv(folder / "source_validation.csv", dtype={"mmsi": str})
    assert not set(train_frame.mmsi) & set(val_frame.mmsi)
    allowed = ~train_frame.acquisition.isin(val_frame.acquisition)
    removed = int((~allowed).sum())
    data["train"] = data["train"][allowed.to_numpy()]
    data["source_train"] = data["source_train"].loc[allowed.to_numpy()].reset_index(drop=True)
    train_frame = train_frame.loc[allowed].reset_index(drop=True)
    assert not set(train_frame.acquisition) & set(val_frame.acquisition)
    original_val = data["val"].copy()
    data["val"] = np.empty(0, dtype=int)
    all_positions = np.arange(len(data["labels"]))
    labels, base, expert, prior_b, prior_e = build_model(data, all_positions)
    relation = class_map(expert[:, :-1], labels, data["train"])
    print("train-only heads ready", slug, len(data["train"]), flush=True)
    weights, relation_count = learn_relation(source_tasks(data, base, expert, relation))
    ids = manifest.instance_id.to_numpy(str)[data["rows"]]
    lookup = pd.Series(all_positions, index=ids)
    target_ids = sorted({item for e in episodes for item in e["query"] + [
        x for draw in e["draws"] for origin in ("same", "cross") for x in draw[origin]]})
    held = lookup.loc[target_ids].to_numpy(int)
    selected = np.sort(np.unique(np.r_[original_val, held]))
    assert not set(data["train"]) & set(selected)
    global_selected = data["rows"][selected]

    merged = []
    for e in episodes:
        with np.load(LOCAL / f"features_{e['fold']}.npz") as cache:
            merged.append((cache["rows"].copy(), cache["ship_peak"].copy(), cache["raw_ring"].copy()))
    cache_rows = np.concatenate([v[0] for v in merged])
    unique, first_indices = np.unique(cache_rows, return_index=True)
    ship = np.concatenate([v[1] for v in merged])[first_indices]
    ring = np.concatenate([v[2] for v in merged])[first_indices]
    gallery = source_gallery(folder)
    gallery = gallery[np.isin(gallery, train_frame.row_index)]
    pos_gallery = np.searchsorted(unique, gallery)
    pos_selected = np.searchsorted(unique, global_selected)
    np.testing.assert_array_equal(unique[pos_gallery], gallery)
    np.testing.assert_array_equal(unique[pos_selected], global_selected)
    local = {}
    for name, raw, width in (("ship", ship, 64), ("ring", ring, 4)):
        pca = PCA(n_components=width, svd_solver="randomized", random_state=SEED)
        transformed = pca.fit(raw[pos_gallery]).transform(raw[pos_selected]).astype(np.float64)
        transformed /= np.linalg.norm(transformed, axis=1, keepdims=True).clip(1e-12)
        local[name] = transformed
    np.savez_compressed(out / "readout.npz", labels=labels[selected], base=base[selected],
                        expert=expert[selected], prior_b=prior_b[selected], prior_e=prior_e[selected],
                        relation=relation[selected], weights=weights, **local,
                        global_rows=global_selected, ids=ids[selected])

    meta = manifest.iloc[global_selected].copy().reset_index(drop=True)
    meta["sample_id"] = ids[selected]
    meta["global_row"] = global_selected
    meta["acquisition"] = meta["product"].str.split("_").map(
        lambda p: "_".join((p[0], p[6], p[7])))
    meta["position"] = np.arange(len(selected))
    meta[["position", "sample_id", "global_row", "port", "acquisition", "mmsi", "class_id"]].to_csv(
        out / "metadata.csv", index=False)
    val = meta.loc[np.isin(global_selected, val_frame.row_index)].copy()
    rng = np.random.default_rng(SEED)
    acqs = np.asarray(sorted(val.acquisition.unique()))
    selection_acqs = set(rng.permutation(acqs)[:int(np.ceil(.30 * len(acqs)))])
    selection = val.loc[val.acquisition.isin(selection_acqs)]
    fit = val.loc[~val.acquisition.isin(selection_acqs) & ~val.mmsi.isin(selection.mmsi)]
    assert not set(fit.mmsi) & set(selection.mmsi)
    assert not set(fit.acquisition) & set(selection.acquisition)
    split_rows, counts = [], []
    for name, frame in (("module_fit", fit), ("selection", selection)):
        split_rows.extend(dict(position=int(i), split=name) for i in frame.index)
        counts.append(dict(split=name, rows=len(frame), ships=frame.mmsi.nunique(),
                           acquisitions=frame.acquisition.nunique(),
                           class_counts=frame.class_id.value_counts().sort_index().to_dict()))
    pd.DataFrame(split_rows).to_csv(out / "source_split.csv", index=False)
    fit_tasks, fit_skipped = make_tasks(fit, SEED + 1)
    selection_tasks, selection_skipped = make_tasks(selection, SEED + 2)
    for task in fit_tasks + selection_tasks:
        assert max(task["q"] + task["s"]) < len(selected)
    if not fit_tasks or not selection_tasks:
        raise RuntimeError(f"No same-port paired source tasks: {slug}")
    target_tasks = []
    compact_lookup = pd.Series(np.arange(len(selected)), index=ids[selected])
    for episode in episodes:
        for draw, supports in enumerate(episode["draws"]):
            for origin in ("same", "cross"):
                target_tasks.append(dict(fold=episode["fold"], port=port, draw=draw, origin=origin,
                    acquisition=episode["fold"].split("__", 1)[1],
                    q=compact_lookup.loc[episode["query"]].to_list(),
                    s=compact_lookup.loc[supports[origin]].to_list()))
    save_json(out / "episodes.json", dict(fit=fit_tasks, selection=selection_tasks, target=target_tasks))
    detail = dict(port=port, source_train=len(data["train"]), removed_source_acquisitions_rows=removed,
                  gallery=len(gallery), relation_tasks=relation_count, source_counts=counts,
                  fit_tasks=len(fit_tasks), selection_tasks=len(selection_tasks),
                  fit_skipped=fit_skipped, selection_skipped=selection_skipped,
                  build_seconds=time.perf_counter()-start,
                  protocol="source_train_only_all_arms; no_target_query_selection")
    save_json(out / "prepared.json", detail)
    print("prepared", slug, "tasks", len(fit_tasks), len(selection_tasks),
          "seconds", round(detail["build_seconds"], 1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port")
    args = parser.parse_args()
    torch.set_num_threads(2)
    episodes = json.loads((CONTEXT / "target_episodes.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST, dtype={"mmsi": str}, usecols=[
        "instance_id", "class_id", "port", "mmsi", "product"])
    with threadpool_limits(limits=2):
        for port in dict.fromkeys(e["port"] for e in episodes):
            if args.port and args.port != port:
                continue
            prepare(port, [e for e in episodes if e["port"] == port], manifest)


if __name__ == "__main__":
    main()
