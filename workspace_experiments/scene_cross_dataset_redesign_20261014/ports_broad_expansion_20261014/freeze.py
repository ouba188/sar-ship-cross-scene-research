"""Freeze broader source-port-held scenes for the 24-port eight-output task."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


HERE = Path(__file__).resolve().parent
MANIFEST = Path("E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv")
PIXELS = Path("D:/SAR_completion_cache_20261010/source_inputs128.npy")
PIXEL_IDS = PIXELS.with_name("source_ids.npy")
REQUESTED_PORTS = ("Ningbo-Zhoushan", "Shanghai", "Jebel Ali", "Qingdao")
FALLBACK_PORTS = ("Santos", "Port Said", "Tanger Med", "Houston")
SEED = 20261014
BUDGET = 16
CLASSES = tuple(range(8))
FIELDS = ("row_index", "sample_id", "port", "acquisition", "product", "mmsi", "class_id", "class_name")


def load_data():
    data = pd.read_csv(MANIFEST, usecols=["instance_id", "port", "product", "mmsi", "class_id", "class_name"],
                       dtype={"instance_id": str, "product": str, "mmsi": str})
    np.testing.assert_array_equal(data.instance_id.to_numpy(str), np.load(PIXEL_IDS).astype(str))
    assert len(data) == len(np.load(PIXELS, mmap_mode="r"))
    parts = data["product"].str.split("_")
    assert parts.str.len().eq(9).all()
    data["acquisition"] = parts.map(lambda p: "_".join((p[0], p[6], p[7])))
    data["sample_id"] = data.instance_id
    data["row_index"] = np.arange(len(data), dtype=np.int64)
    return data[data.mmsi.fillna("").str.fullmatch("[1-9][0-9]{8}")].copy()


def one_per_ship(frame):
    return frame.sort_values("sample_id").drop_duplicates("mmsi")


def counts(frame):
    return frame.class_id.value_counts().reindex(CLASSES, fill_value=0).astype(int).tolist()


def source_split(legal):
    folds = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    options = []
    for tr, va in folds.split(legal, legal.class_id, legal.acquisition):
        validation = legal.iloc[va]
        train = legal.iloc[tr]
        train = train[~train.mmsi.isin(validation.mmsi)]
        if (len(train) >= 20000 and len(validation) >= 1000
                and set(train.class_id) == set(validation.class_id) == set(CLASSES)):
            options.append((len(train), len(validation), train, validation))
    if not options:
        raise ValueError("No source split with >=20000 train, >=1000 validation and eight classes each")
    _, _, train, validation = max(options, key=lambda item: item[:2])
    return train, validation


def split_query(frame, seed):
    one = one_per_ship(frame)
    rng = np.random.default_rng(seed)
    selected = []
    for cls, part in one.groupby("class_id"):
        rows = part.row_index.to_numpy(np.int64)
        selected.extend(rng.choice(rows, size=(len(rows) + 1) // 2, replace=False))
    query = one[one.row_index.isin(selected)]
    same = one[~one.mmsi.isin(query.mmsi)]
    return query, same


def matched_quota(same, cross, query):
    wanted = set(query.class_id)
    capacity = np.minimum(counts(same), counts(cross))
    classes = [cls for cls in CLASSES if cls in wanted and capacity[cls] > 0]
    if len(classes) < 3 or sum(capacity[cls] for cls in classes) < BUDGET:
        return None, capacity
    quota = np.zeros(8, dtype=np.int64)
    quota[classes] = 1
    while quota.sum() < BUDGET:
        available = [cls for cls in classes if quota[cls] < capacity[cls]]
        cls = min(available, key=lambda c: (quota[c], -capacity[c], c))
        quota[cls] += 1
    return quota, capacity


def choose_scene(target):
    groups = [(acq, one_per_ship(frame)) for acq, frame in target.groupby("acquisition")]
    candidates = []
    for rank, (query_acq, frame) in enumerate(groups):
        query, same = split_query(frame, SEED + rank)
        if len(query) < 40 or query.class_id.nunique() < 3:
            continue
        cross_candidates = []
        for cross_acq, cross_frame in groups:
            if cross_acq == query_acq:
                continue
            cross = cross_frame[~cross_frame.mmsi.isin(query.mmsi)]
            quota, capacity = matched_quota(same, cross, query)
            if quota is None:
                continue
            matched = int((quota > 0).sum())
            cross_candidates.append((matched, sum(capacity), len(cross), cross_acq,
                                     cross, quota, capacity))
        if cross_candidates:
            matched, capacity_total, _, cross_acq, cross, quota, capacity = max(
                cross_candidates, key=lambda item: item[:4])
            candidates.append((query.class_id.nunique(), int(query.class_id.value_counts().min()),
                               len(query), matched, capacity_total, query_acq, cross_acq,
                               query, same, cross, quota, capacity))
    if not candidates:
        raise ValueError("No >=40-ship, >=3-class query with matched same/cross B16")
    return max(candidates, key=lambda item: item[:7])


def draw_support(same, cross, quota, seed):
    tokens = np.concatenate([np.repeat(cls, quota[cls]) for cls in CLASSES]).astype(np.int64)
    selected = []
    for draw in range(5):
        for origin, pool in (("same", same), ("cross", cross)):
            rng = np.random.default_rng(seed + 100 * draw + (0 if origin == "same" else 10000))
            rows = []
            for cls in CLASSES:
                if quota[cls]:
                    options = pool.loc[pool.class_id.eq(cls), "row_index"].to_numpy(np.int64)
                    rows.extend(rng.choice(options, size=quota[cls], replace=False))
            chosen = pool.set_index("row_index").loc[rows].reset_index()
            selected.append(chosen.assign(budget=BUDGET, draw=draw, support_origin=origin))
    return pd.concat(selected, ignore_index=True), tokens


def freeze_port(data, port):
    target = data[data.port.eq(port)]
    legal = data[~data.port.eq(port) & ~data.acquisition.isin(target.acquisition)
                 & ~data.mmsi.isin(target.mmsi)]
    train, validation = source_split(legal)
    source_fit = pd.concat([train, validation]).sort_values("row_index")
    (query_classes, _, _, matched_classes, _, query_acq, cross_acq,
     query, same, cross, quota, capacity) = choose_scene(target)
    draws, tokens = draw_support(same, cross, quota, SEED)
    folder = HERE / f"{port.replace(' ', '_')}__{query_acq}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in (("source_train", train), ("source_validation", validation),
                        ("source_fit", source_fit), ("support_candidates", pd.concat([
                            same.assign(support_origin="same"), cross.assign(support_origin="cross")]))):
        columns = list(FIELDS)
        if name == "support_candidates":
            columns = ["support_origin"] + columns
        frame[columns].to_csv(folder / f"{name}.csv", index=False)
    query[[f for f in FIELDS if f not in ("class_id", "class_name")]].to_csv(folder / "query.csv", index=False)
    query[["row_index", "sample_id", "class_id", "class_name"]].to_csv(folder / "query_labels.csv", index=False)
    draws[["budget", "draw", "support_origin"] + list(FIELDS)].to_csv(folder / "support_draws.csv", index=False)
    composition = []
    for role, frame in (("source_train", train), ("source_validation", validation),
                        ("query", query), ("same_pool", same), ("cross_pool", cross)):
        grouped = frame.assign(product_family=frame["product"].str.split("_").str[2]).groupby(
            ["acquisition", "product_family", "class_id"], as_index=False).agg(
                rows=("row_index", "size"), ships=("mmsi", "nunique"))
        composition.append(grouped.assign(role=role))
    pd.concat(composition, ignore_index=True)[
        ["role", "acquisition", "product_family", "class_id", "rows", "ships"]
    ].to_csv(folder / "composition.csv", index=False)
    np.savez_compressed(folder / "split.npz", ssl_train=train.row_index.to_numpy(np.int64),
                        source_validation=validation.row_index.to_numpy(np.int64),
                        source_fit=source_fit.row_index.to_numpy(np.int64),
                        query=query.row_index.to_numpy(np.int64),
                        support_same=np.stack([draws[draws.draw.eq(i) & draws.support_origin.eq("same")]
                                               .row_index.to_numpy(np.int64) for i in range(5)]),
                        support_cross=np.stack([draws[draws.draw.eq(i) & draws.support_origin.eq("cross")]
                                                .row_index.to_numpy(np.int64) for i in range(5)]),
                        support_classes=tokens)
    summary = {
        "fold": folder.name, "target_port": port, "query_acquisition": query_acq,
        "cross_acquisition": cross_acq, "source_train_rows": len(train),
        "source_validation_rows": len(validation), "source_fit_rows": len(source_fit),
        "source_train_ships": train.mmsi.nunique(), "source_validation_ships": validation.mmsi.nunique(),
        "source_train_acquisitions": train.acquisition.nunique(),
        "source_validation_acquisitions": validation.acquisition.nunique(),
        "target_port_rows": len(target), "target_port_acquisitions": target.acquisition.nunique(),
        "query_rows": len(query), "query_ships": query.mmsi.nunique(), "query_classes": query_classes,
        "same_pool_rows": len(same), "cross_pool_rows": len(cross), "support_classes": matched_classes,
        "source_train_class_counts": counts(train), "source_validation_class_counts": counts(validation),
        "query_class_counts": counts(query), "same_pool_class_counts": counts(same),
        "cross_pool_class_counts": counts(cross), "support_quota": quota.tolist(),
        "matched_capacity": capacity.tolist(), "support_budget": BUDGET, "support_draws": 5,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    data = load_data()
    summaries, availability = [], []
    for port in REQUESTED_PORTS:
        try:
            summary = freeze_port(data, port)
        except ValueError as error:
            availability.append({"port": port, "status": "infeasible", "reason": str(error)})
        else:
            summaries.append(summary)
            availability.append({"port": port, "status": "frozen", "reason": ""})
    for port in FALLBACK_PORTS:
        if len(summaries) == len(REQUESTED_PORTS):
            break
        try:
            summary = freeze_port(data, port)
        except ValueError as error:
            availability.append({"port": port, "status": "infeasible", "reason": str(error)})
        else:
            summaries.append(summary)
            availability.append({"port": port, "status": "fallback_frozen", "reason": ""})
    pd.DataFrame(summaries).to_csv(HERE / "fold_summary.csv", index=False)
    pd.DataFrame(availability).to_csv(HERE / "availability.csv", index=False)
    (HERE / "protocol.json").write_text(json.dumps({
        "manifest": str(MANIFEST), "pixel_cache": str(PIXELS), "pixel_ids": str(PIXEL_IDS),
        "target_unit": "entire port, all its acquisitions, all its MMSI",
        "source_validation": "identity-clean best retained source train among ten stratified acquisition-grouped splits",
        "query": "one row per ship, stratified half from one acquisition; >=40 ships and >=3 present classes",
        "support": "same/cross acquisition matched class quota, B16, five draws",
        "selection": "query class coverage, minimum per-class query ships, total ships, then matched support capacity; no model outcomes",
        "requested_ports": REQUESTED_PORTS, "fallback_ports": FALLBACK_PORTS,
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries)[["fold", "source_train_rows", "source_validation_rows",
                                   "query_rows", "query_classes", "support_classes"]].to_string(index=False), flush=True)
    print(pd.DataFrame(availability).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
