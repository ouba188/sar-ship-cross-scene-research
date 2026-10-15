"""Metadata-only capacity audit for a few OSS1 held-acquisition folds."""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


HERE = Path(__file__).resolve().parent
SCENE_ROOT = HERE.parent
MANIFEST = Path("E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/target_manifest.csv")
PILOT = "S1A_010568_00FB64"
PILOT_CROSS = "S1A_009496_00DC88"
SEED = 20261014
CLASSES = set(range(8))


def acquisition(scene):
    bits = scene.split("_")
    return "_".join((bits[0], bits[-3], bits[-2]))


def read_manifest():
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        raw = list(csv.DictReader(stream))
    assert all(row["label_included"] == "True" for row in raw)
    assert len({row["sample_id"] for row in raw}) == len(raw)
    rows = [
        {
            "sample_id": row["sample_id"],
            "acquisition": acquisition(row["scene_id"]),
            "mmsi": row["mmsi_or_ship_id"],
            "class_id": int(row["class_id"]),
        }
        for row in raw
    ]
    assert all(row["mmsi"] for row in rows)
    assert all(len({r["class_id"] for r in rows if r["mmsi"] == ship}) == 1
               for ship in {r["mmsi"] for r in rows})
    return rows


def read_ids(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["sample_id"] for row in csv.DictReader(stream)}


def split_query(rows, rank):
    ships = sorted({row["mmsi"] for row in rows})
    random.Random(SEED + rank).shuffle(ships)
    query_ships = set(ships[1::2])
    query = [row for row in rows if row["mmsi"] in query_ships]
    same = [row for row in rows if row["mmsi"] not in query_ships]
    return query, same


def one_per_ship(rows):
    return list({row["mmsi"]: row for row in sorted(rows, key=lambda r: r["sample_id"])}.values())


def support_capacity(same, cross):
    a = Counter(row["class_id"] for row in one_per_ship(same))
    b = Counter(row["class_id"] for row in one_per_ship(cross))
    matched = {cls: min(a[cls], b[cls]) for cls in CLASSES}
    return sum(matched.values()), sum(value > 0 for value in matched.values()), matched


def source_split(pool):
    labels = np.array([row["class_id"] for row in pool])
    groups = np.array([row["acquisition"] for row in pool])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    options = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(pool)), labels, groups)):
        val = [pool[i] for i in val_idx]
        val_ships = {row["mmsi"] for row in val}
        train = [pool[i] for i in train_idx if pool[i]["mmsi"] not in val_ships]
        train_classes = Counter(row["class_id"] for row in train)
        val_classes = Counter(row["class_id"] for row in val)
        if set(train_classes) != CLASSES or set(val_classes) != CLASSES:
            continue
        options.append((min(val_classes.values()), len(train), -fold, train, val))
    if not options:
        return None
    _, _, minus_fold, train, val = max(options, key=lambda option: option[:3])
    return -minus_fold, train, val


def audit_pair(source_base, by_acq, query_key, cross_key, rank):
    query, same = split_query(by_acq[query_key], rank)
    query_ships = {row["mmsi"] for row in query}
    cross = [row for row in by_acq[cross_key] if row["mmsi"] not in query_ships]
    held_ships = {row["mmsi"] for row in by_acq[query_key] + by_acq[cross_key]}
    source = [row for row in source_base if row["acquisition"] not in (query_key, cross_key)
              and row["mmsi"] not in held_ships]
    split = source_split(source)
    if query_key == PILOT and cross_key == PILOT_CROSS:
        completed = SCENE_ROOT / "strict_training" / "oss1"
        actual_train = read_ids(completed / "source_train.csv")
        actual_val = read_ids(completed / "source_validation.csv")
        train = [row for row in source if row["sample_id"] in actual_train]
        val = [row for row in source if row["sample_id"] in actual_val]
        assert len(train) == len(actual_train) == 720
        assert len(val) == len(actual_val) == 202
        split = ("completed", train, val)
    capacity, matched_classes, quota = support_capacity(same, cross)
    result = {
        "query_acquisition": query_key,
        "cross_acquisition": cross_key,
        "source_pool_rows": len(source),
        "source_pool_ships": len({row["mmsi"] for row in source}),
        "source_pool_classes": len({row["class_id"] for row in source}),
        "source_train_rows": len(split[1]) if split else 0,
        "source_train_ships": len({row["mmsi"] for row in split[1]}) if split else 0,
        "source_validation_rows": len(split[2]) if split else 0,
        "source_validation_ships": len({row["mmsi"] for row in split[2]}) if split else 0,
        "source_split_fold": split[0] if split else "",
        "source_cv_purged_rows": len(source) - len(split[1]) - len(split[2]) if split else "",
        "source_train_class_counts": str(dict(sorted(Counter(r["class_id"] for r in split[1]).items()))) if split else "",
        "source_validation_class_counts": str(dict(sorted(Counter(r["class_id"] for r in split[2]).items()))) if split else "",
        "target_rows": len(by_acq[query_key]),
        "target_ships": len({row["mmsi"] for row in by_acq[query_key]}),
        "target_classes": len({row["class_id"] for row in by_acq[query_key]}),
        "query_rows": len(query),
        "query_ships": len(query_ships),
        "query_classes": len({row["class_id"] for row in query}),
        "query_classes_ge2": sum(v >= 2 for v in Counter(r["class_id"] for r in query).values()),
        "same_support_ships": len({row["mmsi"] for row in same}),
        "cross_support_ships": len({row["mmsi"] for row in cross}),
        "matched_class_capacity": capacity,
        "matched_support_classes": matched_classes,
        "B16_paired": capacity >= 16,
        "B64_paired": capacity >= 64,
        "query_same_ship_overlap": len(query_ships & {row["mmsi"] for row in same}),
        "query_cross_ship_overlap": len(query_ships & {row["mmsi"] for row in cross}),
        "source_held_ship_overlap": len({row["mmsi"] for row in source} & held_ships),
        "source_held_acq_overlap": len({row["acquisition"] for row in source} & {query_key, cross_key}),
        "query_class_counts": str(dict(sorted(Counter(r["class_id"] for r in query).items()))),
        "matched_class_counts": str(dict(sorted(quota.items()))),
    }
    assert not any(result[key] for key in ("query_same_ship_overlap", "query_cross_ship_overlap",
                                       "source_held_ship_overlap", "source_held_acq_overlap"))
    if split:
        assert {row["mmsi"] for row in split[1]}.isdisjoint({row["mmsi"] for row in split[2]})
        assert {row["acquisition"] for row in split[1]}.isdisjoint({row["acquisition"] for row in split[2]})
    return result


def main():
    rows = read_manifest()
    frozen = SCENE_ROOT / "public_split"
    source_ids = read_ids(frozen / "oss1_source_train.csv") | read_ids(
        frozen / "oss1_source_validation.csv")
    source_base = [row for row in rows if row["sample_id"] in source_ids]
    assert len(source_base) == len(source_ids) == 1002
    by_acq = defaultdict(list)
    for row in rows:
        by_acq[row["acquisition"]].append(row)
    ranked = sorted(by_acq, key=lambda key: (-len({r["mmsi"] for r in by_acq[key]}), key))
    audit = []
    for rank, query_key in enumerate(ranked):
        if query_key == PILOT:
            options = [audit_pair(source_base, by_acq, query_key, PILOT_CROSS, rank)]
        else:
            options = [audit_pair(source_base, by_acq, query_key, cross_key, rank)
                       for cross_key in ranked if cross_key != query_key]
            options = [option for option in options if option["matched_class_capacity"] >= 24
                       and option["matched_support_classes"] >= 4
                       and option["source_train_rows"] >= 600]
            if not options:
                continue
            options.sort(key=lambda option: (-option["source_train_rows"],
                                             -option["matched_support_classes"],
                                             -option["matched_class_capacity"],
                                             option["cross_acquisition"]))
            options = options[:1]
        audit.extend(options)

    eligible = [item for item in audit if item["query_acquisition"] != PILOT
                and item["target_ships"] >= 70 and item["target_classes"] >= 6
                and item["query_ships"] >= 30 and item["query_classes"] >= 4
                and item["source_train_rows"] >= 600 and item["B16_paired"]]
    eligible.sort(key=lambda item: (-item["query_classes_ge2"], -item["query_classes"],
                                    -item["target_classes"], -item["target_ships"],
                                    item["query_acquisition"]))
    chosen = [next(item for item in audit if item["query_acquisition"] == PILOT)] + eligible[:3]
    for name, data in (("all_candidate_folds.csv", audit), ("recommended_folds.csv", chosen)):
        with (HERE / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(chosen[0]))
            writer.writeheader()
            writer.writerows(data)
    print("eligible", len(eligible), "chosen", len(chosen))
    for item in chosen:
        print(item["query_acquisition"], "cross", item["cross_acquisition"],
              "source", item["source_pool_rows"], "train", item["source_train_rows"],
              "query", item["query_rows"], "classes", item["query_classes"],
              "capacity", item["matched_class_capacity"])


if __name__ == "__main__":
    main()
