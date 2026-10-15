"""Freeze a few acquisition-held public SAR episodes before fitting models."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "public_split"))
from freeze_pilots import load_metadata, matched_quota, representative_rows  # noqa: E402

MANIFESTS = {
    name: Path(f"E:/Hermes/paper_tracking/sensitivity/opensarship{number}_transfer_20261007/target_manifest.csv")
    for name, number in (("oss1", "1"), ("oss2", "2"))
}
SEED = 20261014
FIELDS = ["dataset", "sample_id", "acquisition", "scene_id", "product", "mmsi",
          "chip_path_vv", "chip_path_vh", "class_id", "class_name"]


def split_query(rows, labels, seed):
    by_ship = defaultdict(list)
    for row in rows:
        by_ship[row["mmsi"]].append(row)
    by_class = defaultdict(list)
    ambiguous_ships = set()
    for ship, observations in by_ship.items():
        classes = {labels[item["sample_id"]]["class_id"] for item in observations}
        if len(classes) != 1:
            ambiguous_ships.add(ship)
            continue
        by_class[classes.pop()].append(ship)
    query_ships = set()
    for class_id, ships in sorted(by_class.items(), key=lambda pair: int(pair[0])):
        random.Random(seed + int(class_id)).shuffle(ships)
        query_ships.update(ships[:(len(ships) + 1) // 2])
    query = [row for row in rows if row["mmsi"] in query_ships]
    same = [row for row in rows if row["mmsi"] not in query_ships]
    return query, same, ambiguous_ships


def source_split(source, labels, minimum):
    frame = pd.DataFrame(source)
    frame["class_id"] = frame.sample_id.map(lambda sample_id: labels[sample_id]["class_id"]).astype(int)
    frame["class_name"] = frame.sample_id.map(lambda sample_id: labels[sample_id]["class_name"])
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    options = []
    for train_idx, val_idx in folds.split(frame, frame.class_id, frame.acquisition):
        validation = frame.iloc[val_idx]
        train = frame.iloc[train_idx]
        train = train[~train.mmsi.isin(set(validation.mmsi))]
        if (len(train) >= minimum and len(validation) >= 50
                and set(train.class_id) == set(validation.class_id) == set(range(8))):
            options.append((len(train), train, validation))
    if not options:
        return None
    _, train, validation = max(options, key=lambda item: item[0])
    return train, validation


def labeled(rows, labels):
    return [{**row, **labels[row["sample_id"]]} for row in rows]


def freeze(dataset, query_key, excluded_query_paths):
    metadata, labels = load_metadata(dataset, MANIFESTS[dataset])
    by_acq = defaultdict(list)
    for row in metadata:
        by_acq[row["acquisition"]].append(row)
    if query_key not in by_acq:
        raise ValueError(f"Unknown acquisition: {query_key}")
    query, same, ambiguous_ships = split_query(by_acq[query_key], labels, SEED + len(query_key))
    prior_ships = set()
    for path in excluded_query_paths:
        prior_ships.update(pd.read_csv(path, dtype={"mmsi": str}).mmsi)
    moved = [row for row in query if row["mmsi"] in prior_ships]
    query = [row for row in query if row["mmsi"] not in prior_ships]
    same.extend(moved)
    minimum = 650 if dataset == "oss1" else 2000
    candidates = []
    for cross_key, cross_rows in by_acq.items():
        if cross_key == query_key:
            continue
        query_ships = {row["mmsi"] for row in query}
        cross = [row for row in cross_rows if row["mmsi"] not in query_ships]
        quota, capacity, same_by_class, cross_by_class = matched_quota(same, cross, labels, 16)
        if sum(quota.values()) < 16:
            continue
        held_ships = {row["mmsi"] for row in query + same + cross}
        source = [row for row in metadata if row["acquisition"] not in (query_key, cross_key)
                  and row["mmsi"] not in held_ships]
        split = source_split(source, labels, minimum)
        if split is None:
            continue
        train, validation = split
        candidates.append((len(train), capacity, cross_key, cross, quota,
                           same_by_class, cross_by_class, train, validation))
    if not candidates:
        raise ValueError(f"No B16 episode with >= {minimum} source train rows: {query_key}")
    # Predefine cross support by capacity, then retain the most source training data.
    selected = max(candidates, key=lambda item: (item[1], item[0], item[2]))
    train_count, capacity, cross_key, cross, quota, same_by_class, cross_by_class, train, validation = selected
    out = Path(__file__).resolve().parent / dataset / query_key
    out.mkdir(parents=True, exist_ok=True)
    train[FIELDS].to_csv(out / "source_train.csv", index=False)
    validation[FIELDS].to_csv(out / "source_validation.csv", index=False)
    pd.DataFrame(query)[[field for field in FIELDS if field not in ("class_id", "class_name")]].to_csv(out / "query.csv", index=False)
    pd.DataFrame(labeled(query, labels))[["sample_id", "class_id", "class_name"]].to_csv(out / "query_labels.csv", index=False)
    pd.DataFrame([{"support_origin": origin, **item} for origin, rows in
                  (("same", same), ("cross", cross)) for item in labeled(rows, labels)])[
                      ["support_origin"] + FIELDS].to_csv(out / "support_candidates.csv", index=False)
    draws = []
    for draw in range(5):
        for origin, by_class in (("same", same_by_class), ("cross", cross_by_class)):
            for class_id, count in quota.items():
                options = sorted(by_class[class_id], key=lambda row: row["sample_id"])
                random.Random(SEED + draw * 100 + int(class_id) + (1000 if origin == "cross" else 0)).shuffle(options)
                draws.extend({"budget": 16, "draw": draw, "support_origin": origin,
                              **item, **labels[item["sample_id"]]} for item in options[:count])
    pd.DataFrame(draws)[["budget", "draw", "support_origin"] + FIELDS].to_csv(out / "support_draws.csv", index=False)
    assert not ({row["mmsi"] for row in query} & {row["mmsi"] for row in same + cross})
    assert not (set(train.acquisition) | set(validation.acquisition)) & {query_key, cross_key}
    assert not (set(train.mmsi) | set(validation.mmsi)) & {row["mmsi"] for row in query + same + cross}
    summary = {
        "dataset": dataset, "query_acquisition": query_key, "cross_acquisition": cross_key,
        "query_rows": len(query), "query_ships": len({row["mmsi"] for row in query}),
        "query_acquisition_ambiguous_ships_excluded": len(ambiguous_ships),
        "query_ships_removed_as_prior_queries": len({row["mmsi"] for row in moved}),
        "query_classes": len({labels[row["sample_id"]]["class_id"] for row in query}),
        "same_support_ships": len({row["mmsi"] for row in same}),
        "cross_support_ships": len({row["mmsi"] for row in cross}),
        "matched_b16_capacity": capacity,
        "matched_quota": {str(key): value for key, value in quota.items()},
        "source_train_rows": train_count, "source_validation_rows": len(validation),
        "source_train_ships": int(train.mmsi.nunique()),
        "source_val_ships": int(validation.mmsi.nunique()),
        "source_train_classes": int(train.class_id.nunique()),
        "source_val_classes": int(validation.class_id.nunique()),
        "selection": "metadata/class-capacity only; no model outcomes",
    }
    (out / "episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(MANIFESTS), required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--exclude-query-csv", type=Path, action="append", default=[])
    args = parser.parse_args()
    freeze(args.dataset, args.query, args.exclude_query_csv)
