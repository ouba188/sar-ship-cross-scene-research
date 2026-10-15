"""Check the frozen pilot manifests against the original metadata and split rules."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from freeze_pilots import BASE_FIELDS, INPUTS, acquisition


ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ships(rows):
    return {row["mmsi"] for row in rows}


def main():
    summary = json.loads((ROOT / "pilot_summary.json").read_text(encoding="utf-8"))
    originals = {}
    for dataset, path in INPUTS.items():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            originals[dataset] = {row["sample_id"]: row for row in csv.DictReader(handle)}

    held_keys = set()
    held_ships = set()
    for dataset, detail in summary["datasets"].items():
        held_keys.update((detail["query_acquisition"], detail["cross_support_acquisition"]))
        for raw in originals.values():
            held_ships.update(
                row["mmsi_or_ship_id"] for row in raw.values()
                if acquisition(row["scene_id"]) in held_keys
            )

    for dataset, detail in summary["datasets"].items():
        raw = originals[dataset]
        train = read(f"{dataset}_source_train.csv")
        validation = read(f"{dataset}_source_validation.csv")
        ssl = read(f"{dataset}_ssl.csv")
        query = read(f"{dataset}_query.csv")
        query_labels = read(f"{dataset}_query_labels.csv")
        candidates = read(f"{dataset}_support_candidates.csv")
        draws = read(f"{dataset}_support_draws.csv")
        same = [row for row in candidates if row["support_origin"] == "same"]
        cross = [row for row in candidates if row["support_origin"] == "cross"]

        assert len(train) == detail["source_train"]["rows"]
        assert len(validation) == detail["source_validation"]["rows"]
        assert len(query) == detail["query"]["rows"]
        assert len(same) == detail["same_support_candidates"]["rows"]
        assert len(cross) == detail["cross_support_candidates"]["rows"]
        assert set(query[0]) == set(BASE_FIELDS)
        assert [{field: row[field] for field in BASE_FIELDS} for row in train] == ssl
        assert not (ships(train + validation) & held_ships)
        assert not (ships(train) & ships(validation))
        assert not (ships(query) & ships(candidates))
        assert {row["acquisition"] for row in train + validation}.isdisjoint(held_keys)
        assert {row["acquisition"] for row in query + same} == {detail["query_acquisition"]}
        assert {row["acquisition"] for row in cross} == {detail["cross_support_acquisition"]}
        assert {row["acquisition"] for row in validation} == {detail["source_validation_acquisition"]}
        assert {row["sample_id"] for row in query} == {row["sample_id"] for row in query_labels}
        for row in query_labels:
            original = raw[row["sample_id"]]
            assert (row["class_id"], row["class_name"]) == (
                original["class_id"], original["class_name"]
            )

        for row in train + validation + ssl + query + candidates:
            original = raw[row["sample_id"]]
            assert row["scene_id"] == original["scene_id"]
            assert row["acquisition"] == acquisition(original["scene_id"])
            assert row["mmsi"] == original["mmsi_or_ship_id"]
            assert row["product"] == original["scene_id"].split("_")[2]
            if "class_id" in row:
                assert row["class_id"] == original["class_id"]
            assert Path(row["chip_path_vv"]).is_file()
            assert Path(row["chip_path_vh"]).is_file()

        by_draw = defaultdict(list)
        candidate_ids = {(row["support_origin"], row["sample_id"]) for row in candidates}
        for row in draws:
            assert (row["support_origin"], row["sample_id"]) in candidate_ids
            by_draw[(int(row["budget"]), int(row["draw"]))].append(row)
        for budget in (16, 64):
            feasible = detail["budgets"][f"B{budget}"]["feasible"]
            expected = set(range(summary["draw_count"])) if feasible else set()
            assert {draw for size, draw in by_draw if size == budget} == expected
            for draw in expected:
                rows = by_draw[(budget, draw)]
                selected = {origin: [row for row in rows if row["support_origin"] == origin]
                            for origin in ("same", "cross")}
                assert all(len(pool) == budget and len(ships(pool)) == budget
                           for pool in selected.values())
                assert Counter(row["class_id"] for row in selected["same"]) == Counter(
                    row["class_id"] for row in selected["cross"]
                )
                assert ships(rows).isdisjoint(ships(query))

        if dataset == "oss2":
            for key in (detail["query_acquisition"], detail["cross_support_acquisition"]):
                products = {row["product"] for row in query + candidates if row["acquisition"] == key}
                assert products == {"GRDH", "SLC"}
        print(
            f"{dataset}: train={len(train)} validation={len(validation)} "
            f"query={len(query)} candidates={len(candidates)} draws={len(draws)} OK"
        )


if __name__ == "__main__":
    main()
