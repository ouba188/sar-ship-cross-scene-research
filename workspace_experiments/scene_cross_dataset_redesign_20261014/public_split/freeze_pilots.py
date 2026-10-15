"""Freeze metadata-selected OpenSARShip pilot episodes without model outputs."""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


INPUTS = {
    "oss1": Path("E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/target_manifest.csv"),
    "oss2": Path("E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/target_manifest.csv"),
}
OUT = Path(__file__).resolve().parent
SEED = 20261014
BUDGETS = (16, 64)
DRAW_COUNT = 5
BASE_FIELDS = [
    "dataset", "sample_id", "acquisition", "scene_id", "product", "mmsi",
    "chip_path_vv", "chip_path_vh",
]
LABELED_FIELDS = BASE_FIELDS + ["class_id", "class_name"]


def acquisition(scene_id):
    parts = scene_id.split("_")
    assert len(parts) in (9, 10) and parts[0].startswith("S1")
    return "_".join((parts[0], parts[-3], parts[-2]))


def load_metadata(dataset, path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        original = list(csv.DictReader(handle))
    assert all(row["label_included"] == "True" for row in original)
    assert len({row["sample_id"] for row in original}) == len(original)
    metadata = []
    labels = {}
    for row in original:
        scene = row["scene_id"]
        item = {
            "dataset": dataset,
            "sample_id": row["sample_id"],
            "acquisition": acquisition(scene),
            "scene_id": scene,
            "product": scene.split("_")[2],
            "mmsi": row["mmsi_or_ship_id"],
            "chip_path_vv": row["chip_path_vv"],
            "chip_path_vh": row["chip_path_vh"],
        }
        assert item["mmsi"] and item["chip_path_vv"] and item["chip_path_vh"]
        metadata.append(item)
        labels[item["sample_id"]] = {
            "class_id": row["class_id"], "class_name": row["class_name"]
        }
    return metadata, labels


def groups(rows):
    result = defaultdict(list)
    for row in rows:
        result[row["acquisition"]].append(row)
    return result


def ranked_keys(grouped, allowed=None, mixed=False):
    keys = allowed if allowed is not None else grouped.keys()
    if mixed:
        keys = [key for key in keys if {row["product"] for row in grouped[key]} == {"GRDH", "SLC"}]
    return sorted(keys, key=lambda key: (-len({row["mmsi"] for row in grouped[key]}), key))


def representative_rows(rows, labels):
    by_ship = defaultdict(list)
    for row in rows:
        by_ship[row["mmsi"]].append(row)
    chosen = [min(ship_rows, key=lambda row: row["sample_id"]) for ship_rows in by_ship.values()]
    by_class = defaultdict(list)
    for row in chosen:
        by_class[labels[row["sample_id"]]["class_id"]].append(row)
    return by_class


def matched_quota(same, cross, labels, budget):
    a = representative_rows(same, labels)
    b = representative_rows(cross, labels)
    capacities = {cls: min(len(a[cls]), len(b[cls])) for cls in sorted(set(a) | set(b), key=int)}
    quota = {cls: 0 for cls in capacities}
    for _ in range(budget):
        available = [cls for cls in capacities if quota[cls] < capacities[cls]]
        if not available:
            break
        cls = min(available, key=lambda name: (quota[name], int(name)))
        quota[cls] += 1
    return {cls: count for cls, count in quota.items() if count}, sum(capacities.values()), a, b


def write_csv(name, fields, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def with_labels(rows, labels):
    return [{**row, **labels[row["sample_id"]]} for row in rows]


def counts(rows):
    return {
        "rows": len(rows),
        "mmsi": len({row["mmsi"] for row in rows}),
        "acquisitions": len({row["acquisition"] for row in rows}),
        "products": dict(sorted(Counter(row["product"] for row in rows).items())),
    }


def main():
    data = {name: load_metadata(name, path) for name, path in INPUTS.items()}
    grouped = {name: groups(data[name][0]) for name in INPUTS}

    # The acquisition and query-ship decisions use metadata only.
    oss1_rank = ranked_keys(grouped["oss1"])
    held = {"oss1": {"query": oss1_rank[0], "cross": oss1_rank[1]}}
    oss2_rank = ranked_keys(
        grouped["oss2"],
        set(grouped["oss2"]) - set(held["oss1"].values()),
        mixed=True,
    )
    held["oss2"] = {"query": oss2_rank[0], "cross": oss2_rank[1]}
    all_held_keys = set(held["oss1"].values()) | set(held["oss2"].values())
    all_held_mmsi = {
        row["mmsi"] for name in INPUTS for row in data[name][0]
        if row["acquisition"] in all_held_keys
    }

    summary = {
        "protocol": "2026-10-14 metadata-frozen OSS pilot; no model outcomes used",
        "inputs": {name: str(path) for name, path in INPUTS.items()},
        "selection": "OSS1 largest query and next-largest cross acquisition by distinct MMSI; OSS2 largest and next-largest mixed GRDH/SLC acquisitions outside OSS1 held acquisitions; ties by acquisition key. Query ships are the odd positions of a fixed-seed shuffle of sorted MMSIs.",
        "seed": SEED,
        "draw_count": DRAW_COUNT,
        "datasets": {},
    }
    for offset, name in enumerate(INPUTS):
        metadata, labels = data[name]
        by_acq = grouped[name]
        query_key = held[name]["query"]
        cross_key = held[name]["cross"]
        ships = sorted({row["mmsi"] for row in by_acq[query_key]})
        random.Random(SEED + offset).shuffle(ships)
        query_mmsi = set(ships[1::2])
        same_mmsi = set(ships[0::2])
        query = sorted(
            (row for row in by_acq[query_key] if row["mmsi"] in query_mmsi),
            key=lambda row: row["sample_id"],
        )
        same = sorted(
            (row for row in by_acq[query_key] if row["mmsi"] in same_mmsi),
            key=lambda row: row["sample_id"],
        )
        cross = sorted(
            (row for row in by_acq[cross_key] if row["mmsi"] not in query_mmsi),
            key=lambda row: row["sample_id"],
        )
        assert {row["mmsi"] for row in query}.isdisjoint({row["mmsi"] for row in same + cross})

        source_keys = set(by_acq) - all_held_keys
        source_rank = ranked_keys(by_acq, source_keys)
        validation_key = source_rank[min(4, len(source_rank) - 1)]
        validation = sorted(
            (row for row in by_acq[validation_key] if row["mmsi"] not in all_held_mmsi),
            key=lambda row: row["sample_id"],
        )
        validation_mmsi = {row["mmsi"] for row in validation}
        train = sorted(
            (row for row in metadata if row["acquisition"] in source_keys
             and row["acquisition"] != validation_key
             and row["mmsi"] not in all_held_mmsi
             and row["mmsi"] not in validation_mmsi),
            key=lambda row: row["sample_id"],
        )

        write_csv(f"{name}_source_train.csv", LABELED_FIELDS, with_labels(train, labels))
        write_csv(f"{name}_source_validation.csv", LABELED_FIELDS, with_labels(validation, labels))
        write_csv(f"{name}_ssl.csv", BASE_FIELDS, train)
        write_csv(f"{name}_query.csv", BASE_FIELDS, query)
        write_csv(
            f"{name}_query_labels.csv", ["sample_id", "class_id", "class_name"],
            [{"sample_id": row["sample_id"], **labels[row["sample_id"]]} for row in query],
        )
        candidate_fields = ["support_origin"] + LABELED_FIELDS
        write_csv(
            f"{name}_support_candidates.csv", candidate_fields,
            [{"support_origin": origin, **row, **labels[row["sample_id"]]}
             for origin, pool in (("same", same), ("cross", cross)) for row in pool],
        )

        budget_summary = {"B0": {"feasible": bool(query), "support_rows": 0}}
        draws = []
        for budget in BUDGETS:
            quota, matched_capacity, same_classes, cross_classes = matched_quota(same, cross, labels, budget)
            feasible = sum(quota.values()) == budget and bool(query)
            budget_summary[f"B{budget}"] = {
                "feasible": feasible,
                "raw_unique_mmsi_feasible": (
                    len({row["mmsi"] for row in same}) >= budget
                    and len({row["mmsi"] for row in cross}) >= budget
                ),
                "matched_unique_mmsi_capacity": matched_capacity,
                "class_quota": {labels[rows[0]["sample_id"]]["class_name"]: quota[cls]
                                for cls, rows in same_classes.items() if cls in quota},
            }
            if not feasible:
                continue
            for draw in range(DRAW_COUNT):
                for origin, by_class in (("same", same_classes), ("cross", cross_classes)):
                    for cls, needed in quota.items():
                        options = sorted(by_class[cls], key=lambda row: row["sample_id"])
                        random.Random(SEED + offset * 10000 + budget * 100 + draw * 10
                                      + int(cls) + (1000 if origin == "cross" else 0)).shuffle(options)
                        draws.extend(
                            {"budget": budget, "draw": draw, "support_origin": origin,
                             **row, **labels[row["sample_id"]]}
                            for row in options[:needed]
                        )
        write_csv(f"{name}_support_draws.csv", ["budget", "draw"] + candidate_fields, draws)

        product_feasibility = {}
        for product in sorted({row["product"] for row in query}):
            same_product = [row for row in same if row["product"] == product]
            cross_product = [row for row in cross if row["product"] == product]
            product_feasibility[product] = {
                "query_rows": sum(row["product"] == product for row in query),
                "same_unique_mmsi": len({row["mmsi"] for row in same_product}),
                "cross_unique_mmsi": len({row["mmsi"] for row in cross_product}),
                "matched_class_capacity": matched_quota(
                    same_product, cross_product, labels, 64
                )[1],
            }

        summary["datasets"][name] = {
            "query_acquisition": query_key,
            "cross_support_acquisition": cross_key,
            "source_validation_acquisition": validation_key,
            "source_train": counts(train),
            "source_validation": counts(validation),
            "query": counts(query),
            "same_support_candidates": counts(same),
            "cross_support_candidates": counts(cross),
            "excluded_held_acquisitions_global": sorted(all_held_keys),
            "excluded_held_mmsi_global": len(all_held_mmsi),
            "budgets": budget_summary,
            "product_feasibility": product_feasibility,
        }

    with (OUT / "pilot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
