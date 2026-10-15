"""Freeze source-port-held, eight-class 24-port scene episodes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


HERE = Path(__file__).resolve().parent
MANIFEST = Path("E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv")
PIXEL_IDS = Path("D:/SAR_completion_cache_20261010/source_ids.npy")
PIXELS = Path("D:/SAR_completion_cache_20261010/source_inputs128.npy")
PORTS = ("Singapore", "Port Klang")
SEED = 20261014
CLASSES = tuple(range(8))
BUDGET = 16
FIELDS = ("row_index", "sample_id", "port", "acquisition", "product", "mmsi", "class_id", "class_name")


def load_data():
    data = pd.read_csv(MANIFEST, usecols=["instance_id", "port", "product", "mmsi", "class_id", "class_name"],
                       dtype={"instance_id": str, "mmsi": str, "product": str})
    ids = np.load(PIXEL_IDS).astype(str)
    pixels = np.load(PIXELS, mmap_mode="r")
    np.testing.assert_array_equal(data.instance_id.to_numpy(str), ids)
    assert len(data) == len(pixels)
    parts = data["product"].str.split("_")
    assert parts.str.len().eq(9).all()
    data["acquisition"] = parts.map(lambda p: "_".join((p[0], p[6], p[7])))
    data["sample_id"] = data.instance_id
    data["row_index"] = np.arange(len(data))
    return data[data.mmsi.fillna("").str.fullmatch("[1-9][0-9]{8}")].copy()


def one_per_ship(frame):
    return frame.sort_values("sample_id").drop_duplicates("mmsi")


def class_counts(frame):
    return frame.class_id.value_counts().reindex(CLASSES, fill_value=0).astype(int).tolist()


def choose_query_acquisitions(target):
    counts = target.groupby(["acquisition", "class_id"]).mmsi.nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=CLASSES, fill_value=0)
    eligible = counts[(counts.min(axis=1) >= 2) & (counts.sum(axis=1) >= 100)]
    return eligible.assign(minimum=eligible.min(axis=1), total=eligible.sum(axis=1)).sort_values(
        ["minimum", "total"], ascending=False).head(2).index.tolist()


def source_split(legal):
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    options = []
    for train_indices, val_indices in splitter.split(legal, legal.class_id, legal.acquisition):
        validation = legal.iloc[val_indices]
        train = legal.iloc[train_indices]
        train = train[~train.mmsi.isin(validation.mmsi)]
        if (len(train) >= 20000 and len(validation) >= 1000
                and set(train.class_id) == set(validation.class_id) == set(CLASSES)):
            options.append((len(train), len(validation), train, validation))
    if options:
        _, _, train, validation = max(options, key=lambda item: item[:2])
        return train, validation
    raise ValueError("No sufficiently large eight-class source split")


def split_query(acquisition_frame, seed):
    one = one_per_ship(acquisition_frame)
    rng = np.random.default_rng(seed)
    query_rows = []
    for cls in CLASSES:
        rows = one.loc[one.class_id.eq(cls), "row_index"].to_numpy(np.int64)
        if len(rows) < 2:
            raise ValueError(f"Cannot split class {cls} across query/same support")
        query_rows.extend(rng.choice(rows, size=(len(rows) + 1) // 2, replace=False))
    query = one[one.row_index.isin(query_rows)]
    same = one[~one.mmsi.isin(query.mmsi)]
    return query, same


def quota_for(same, cross):
    capacity = np.minimum(class_counts(same), class_counts(cross))
    if min(capacity) < 1 or sum(capacity) < BUDGET:
        return None, capacity
    quota = np.ones(8, dtype=np.int64)
    while quota.sum() < BUDGET:
        available = [cls for cls in CLASSES if quota[cls] < capacity[cls]]
        cls = min(available, key=lambda c: (quota[c], -capacity[c], c))
        quota[cls] += 1
    return quota, capacity


def choose_cross(target, query_acq, query, same):
    candidates = []
    for acq, frame in target[target.acquisition.ne(query_acq)].groupby("acquisition"):
        cross = one_per_ship(frame)
        cross = cross[~cross.mmsi.isin(query.mmsi)]
        quota, capacity = quota_for(same, cross)
        if quota is not None:
            candidates.append((min(capacity), sum(capacity), len(cross), acq, cross, quota))
    if not candidates:
        raise ValueError(f"No matched B16 eight-class cross acquisition for {query_acq}")
    _, _, _, acq, cross, quota = max(candidates, key=lambda x: x[:4])
    return acq, cross, quota


def draw_support(same, cross, quota, seed):
    tokens = np.concatenate([np.repeat(cls, quota[cls]) for cls in CLASSES]).astype(np.int64)
    rows = []
    for draw in range(5):
        for origin, pool in (("same", same), ("cross", cross)):
            rng = np.random.default_rng(seed + 100 * draw + (0 if origin == "same" else 10000))
            chosen = []
            for cls in CLASSES:
                options = pool.loc[pool.class_id.eq(cls), "row_index"].to_numpy(np.int64)
                chosen.extend(rng.choice(options, size=quota[cls], replace=False))
            selected = pool.set_index("row_index").loc[chosen].reset_index()
            selected = selected.assign(budget=BUDGET, draw=draw, support_origin=origin)
            rows.append(selected)
    return pd.concat(rows, ignore_index=True), tokens


def save_fold(data, port, acq, rank, earlier_query_ships):
    target = data[data.port.eq(port)]
    target_acqs = set(target.acquisition)
    target_ships = set(target.mmsi)
    legal = data[~data.port.eq(port) & ~data.acquisition.isin(target_acqs)
                 & ~data.mmsi.isin(target_ships)]
    train, validation = source_split(legal)
    source_fit = pd.concat([train, validation]).sort_values("row_index")
    query_frame = target[target.acquisition.eq(acq) & ~target.mmsi.isin(earlier_query_ships)]
    query, same = split_query(query_frame, SEED + rank)
    cross_acq, cross, quota = choose_cross(target, acq, query, same)
    draws, tokens = draw_support(same, cross, quota, SEED + rank)
    folder = HERE / f"{port.replace(' ', '_')}__{acq}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in (("source_train", train), ("source_validation", validation),
                        ("source_fit", source_fit), ("support_candidates", pd.concat([
                            same.assign(support_origin="same"), cross.assign(support_origin="cross")]))):
        columns = list(FIELDS)
        if name == "support_candidates":
            columns = ["support_origin"] + columns
        frame[columns].to_csv(folder / f"{name}.csv", index=False)
    query[[field for field in FIELDS if field not in ("class_id", "class_name")]].to_csv(
        folder / "query.csv", index=False)
    query[["row_index", "sample_id", "class_id", "class_name"]].to_csv(folder / "query_labels.csv", index=False)
    draws[["budget", "draw", "support_origin"] + list(FIELDS)].to_csv(
        folder / "support_draws.csv", index=False)
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
                        support_same=np.stack([draws[(draws.draw.eq(i)) & (draws.support_origin.eq("same"))]
                                               .row_index.to_numpy(np.int64) for i in range(5)]),
                        support_cross=np.stack([draws[(draws.draw.eq(i)) & (draws.support_origin.eq("cross"))]
                                                .row_index.to_numpy(np.int64) for i in range(5)]),
                        support_classes=tokens)
    summary = {
        "fold": folder.name, "target_port": port, "query_acquisition": acq,
        "cross_acquisition": cross_acq, "source_train_rows": len(train),
        "source_validation_rows": len(validation), "source_fit_rows": len(source_fit),
        "source_train_ships": train.mmsi.nunique(), "source_validation_ships": validation.mmsi.nunique(),
        "source_train_acquisitions": train.acquisition.nunique(),
        "source_validation_acquisitions": validation.acquisition.nunique(),
        "target_port_rows": len(target), "target_port_acquisitions": target.acquisition.nunique(),
        "query_rows": len(query), "query_ships": query.mmsi.nunique(),
        "earlier_fold_query_ships_excluded": len(set(earlier_query_ships) &
                                                set(target[target.acquisition.eq(acq)].mmsi)),
        "same_pool_rows": len(same), "cross_pool_rows": len(cross),
        "source_train_class_counts": class_counts(train),
        "source_validation_class_counts": class_counts(validation),
        "query_class_counts": class_counts(query),
        "same_pool_class_counts": class_counts(same),
        "cross_pool_class_counts": class_counts(cross),
        "support_quota": quota.tolist(), "support_budget": BUDGET, "support_draws": 5,
        "target_port_excluded_from_source": True,
        "target_acquisitions_excluded_from_source": True,
        "target_ships_excluded_from_source": True,
    }
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, set(query.mmsi)


def main():
    data = load_data()
    summaries = []
    for port in PORTS:
        target = data[data.port.eq(port)]
        acquisitions = choose_query_acquisitions(target)
        if len(acquisitions) != 2:
            raise ValueError(f"Expected two eight-class acquisitions for {port}")
        earlier_query_ships = set()
        for rank, acq in enumerate(acquisitions):
            summary, query_ships = save_fold(data, port, acq, rank, earlier_query_ships)
            summaries.append(summary)
            earlier_query_ships.update(query_ships)
    pd.DataFrame(summaries).to_csv(HERE / "fold_summary.csv", index=False)
    (HERE / "protocol.json").write_text(json.dumps({
        "manifest": str(MANIFEST), "pixel_cache": str(PIXELS), "pixel_ids": str(PIXEL_IDS),
        "split_unit": "entire target port and every associated acquisition; target MMSI removed globally",
        "query": "one observation per ship; stratified half per class from target acquisition; second fold excludes prior query ships",
        "cross": "one different acquisition in the held target port, selected by eight-class B16 capacity",
        "source_validation": "largest identity-clean source train among eligible stratified acquisition-grouped ten-fold partitions; source fit=train+validation",
        "support": "same/cross matched eight-class quota, B16 five draws; target labels only for episode freezing",
        "selection": "metadata and class capacity only; no model scores or target predictions",
        "ports": list(PORTS), "query_acquisitions_per_port": 2,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries)[["fold", "source_train_rows", "source_validation_rows",
                                   "query_rows", "support_budget"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
