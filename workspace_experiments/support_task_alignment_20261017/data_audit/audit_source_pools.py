"""Source-only support-pool census; no sampler execution, training, or target labels."""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIAG = ROOT / "local_evidence_reliability_20261017/source_diagnostic"
SOURCE = ROOT / "scene_cross_dataset_redesign_20261014"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, records):
    with (HERE / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def distinct(records):
    seen, result = set(), []
    for record in sorted(records, key=lambda row: row["sample_id"]):
        if record["mmsi"] not in seen:
            seen.add(record["mmsi"])
            result.append(record)
    return result


def summarize_tasks(records, fold, kind, origin):
    def distribution(field):
        values = [row[field] for row in records]
        return {f"{field}_min": min(values),
                f"{field}_median": float(np.median(values)),
                f"{field}_max": max(values)}

    return dict(
        fold=fold, kind=kind, origin=origin, tasks=len(records),
        changeable_tasks=sum(row["changeable"] for row in records),
        fixed_membership_tasks=sum(not row["changeable"] for row in records),
        tasks_losing_candidates_to_prior=sum(row["prior_excluded_ships"] > 0 for row in records),
        original_support_rows=len(records) * 16,
        forced_original_slots=sum(row["forced_original_slots"] for row in records),
        maximum_replaced_slots=sum(row["max_replaced_slots"] for row in records),
        **distribution("raw_pool_ships"), **distribution("legal_pool_ships"),
        **distribution("quota_eligible_ships"), **distribution("extra_quota_eligible_ships"),
        **distribution("expected_new_ships_one_redraw"),
        **distribution("expected_bank_union_original_plus_four"),
    )


def main():
    episodes = json.loads((DIAG / "source_task_rows.json").read_text(encoding="utf-8"))
    metadata = {(row["fold"], row["kind"], int(row["task_number"])): row
                for row in read_csv(DIAG / "source_tasks.csv")}
    with np.load(ROOT / "all_factor_probe_20261017/spatial_descriptors.npz") as cache:
        sample_ids = cache["sample_ids"].astype(str)
    global_index = {value: index for index, value in enumerate(sample_ids)}
    with np.load(ROOT / "local_evidence_reliability_20261017/quality.npz") as cache:
        assert np.array_equal(cache["sample_ids"].astype(str), sample_ids), "Quality row-axis mismatch"
        quality_valid = cache["valid"].astype(bool)

    sources, masks = {}, {}
    for fold in sorted({episode["fold"] for episode in episodes}):
        folder = (SOURCE / "strict_training/oss1" if fold == "pilot"
                  else SOURCE / "public_expansion/oss1" / fold)
        splits = {}
        for kind, filename in (("fit", "source_train.csv"), ("selection", "source_validation.csv")):
            records = read_csv(folder / filename)
            for row in records:
                row["class_id"] = int(row["class_id"])
                row["global_row"] = global_index[row["sample_id"]]
            splits[kind] = records
        sources[fold] = splits
        with np.load(ROOT / "oss1_joint_geometry_20261017" / fold / "local_features.npz") as cache:
            assert np.array_equal(cache["sample_ids"].astype(str), sample_ids), "Local row-axis mismatch"
            masks[fold] = cache["valid"].astype(bool) & quality_valid

    task_rows, class_rows, query_rows, candidate_rows = [], [], [], []
    counters = Counter()
    for episode in episodes:
        fold, kind = episode["fold"], episode["kind"]
        number = counters[fold, kind]
        counters[fold, kind] += 1
        meta = metadata[fold, kind, number]
        context = dict(task_id=meta["task_id"], fold=fold, kind=kind, origin=meta["origin"])
        frame = sources[fold][kind]
        index = {row["global_row"]: row for split in sources[fold].values() for row in split}
        q, s, prior = [[index[row] for row in episode[key]] for key in ("q", "s", "prior")]
        valid = masks[fold]
        qships, pships = {row["mmsi"] for row in q}, {row["mmsi"] for row in prior}
        original = {row["global_row"] for row in s}
        acquisition_rows = [row for row in frame if row["acquisition"] == meta["support_acquisition"]]
        raw_pool = distinct(acquisition_rows)
        no_query = [row for row in raw_pool if row["mmsi"] not in qships]
        pool = [row for row in no_query if row["mmsi"] not in pships]
        assert len(s) == 16 and len({row["mmsi"] for row in s}) == 16
        assert original <= {row["global_row"] for row in pool}, "Original support lost by source reconstruction"
        assert len(q) == int(meta["queries"]) and len(prior) == int(meta["prior_rows"])
        counts = Counter(row["class_id"] for row in s)
        combinations, forced, max_replaced = 1, 0, 0
        expected_new, expected_union = 0.0, 0.0
        eligible = [row for row in pool if counts[row["class_id"]] > 0]
        per_class = {}
        for cls in range(8):
            candidates = [row for row in pool if row["class_id"] == cls]
            old = [row for row in s if row["class_id"] == cls]
            n, k = len(candidates), counts[cls]
            nv = sum(bool(valid[row["global_row"]]) for row in candidates)
            ov = sum(bool(valid[row["global_row"]]) for row in old)
            ways = math.comb(n, k)
            combinations *= ways
            forced += k if n == k else 0
            max_replaced += min(k, n - k)
            if k:
                expected_new += k * (1 - k / n)
                expected_union += k + (n - k) * (1 - (1 - k / n) ** 4)
            per_class[cls] = (k, nv, ov)
            class_name = next(row["class_name"] for row in index.values() if row["class_id"] == cls)
            class_rows.append(dict(**context, class_id=cls, class_name=class_name,
                original_quota=k, legal_pool_ships=n, valid_pool_ships=nv,
                original_valid_supports=ov, changeable=int(k > 0 and n > k),
                forced_original_class=int(k > 0 and n == k),
                unused_due_to_zero_quota=n if k == 0 else 0,
                support_subsets=ways,
                chance_valid_sameclass_one_redraw=(1 - math.comb(n - nv, k) / ways) if k else 0.0))
        for row in pool:
            candidate_rows.append(dict(**context, global_row=row["global_row"],
                sample_id=row["sample_id"], mmsi=row["mmsi"], acquisition=row["acquisition"],
                class_id=row["class_id"], class_name=row["class_name"],
                original_support=int(row["global_row"] in original),
                quota_eligible=int(counts[row["class_id"]] > 0),
                local_valid=int(valid[row["global_row"]])))
        task_rows.append(dict(**context, support_acquisition=meta["support_acquisition"],
            query_acquisition=meta["acquisition"], queries=len(q), prior_rows=len(prior),
            raw_acquisition_rows=len(acquisition_rows), raw_pool_ships=len(raw_pool),
            query_excluded_ships=len(raw_pool) - len(no_query),
            prior_excluded_ships=len(no_query) - len(pool), legal_pool_ships=len(pool),
            quota_eligible_ships=len(eligible), extra_quota_eligible_ships=len(eligible) - 16,
            unused_zero_quota_ships=len(pool) - len(eligible),
            support_subsets=combinations, changeable=int(combinations > 1),
            forced_original_slots=forced, max_replaced_slots=max_replaced,
            expected_new_ships_one_redraw=expected_new,
            expected_bank_union_original_plus_four=expected_union,
            original_valid_supports=sum(bool(valid[row["global_row"]]) for row in s),
            original_valid_classes=len({row["class_id"] for row in s if valid[row["global_row"]]})))
        for row in q:
            k, nv, ov = per_class[row["class_id"]]
            qvalid = bool(valid[row["global_row"]])
            possible = qvalid and k > 0 and nv > 0
            reason = ("invalid_query" if not qvalid else "zero_original_class_quota" if not k
                      else "no_valid_sameclass_candidate" if not nv else "eligible")
            query_rows.append(dict(**context, global_row=row["global_row"],
                sample_id=row["sample_id"], mmsi=row["mmsi"], acquisition=row["acquisition"],
                class_id=row["class_id"], class_name=row["class_name"], local_valid=int(qvalid),
                original_sameclass_count=k, original_valid_sameclass_count=ov,
                legal_valid_sameclass_candidates=nv,
                original_aux_eligible=int(qvalid and ov > 0),
                possible_aux_eligible=int(possible), blocker=reason))

    unique = defaultdict(list)
    for row in query_rows:
        for origin in ("pooled", row["origin"]):
            unique[row["kind"], origin, row["global_row"]].append(row)
    unique_rows = []
    for (_, origin, _), records in unique.items():
        row = records[0]
        assert len({r["local_valid"] for r in records}) == 1, "Fold-dependent validity needs separate reporting"
        old, possible = max(r["original_aux_eligible"] for r in records), max(r["possible_aux_eligible"] for r in records)
        unique_rows.append({**{key: row[key] for key in ("kind", "global_row", "sample_id", "mmsi", "acquisition", "class_id", "class_name", "local_valid")},
            "origin": origin,
            "task_contexts": len(records), "folds": len({r["fold"] for r in records}),
            "original_aux_eligible": old, "possible_aux_eligible": possible,
            "newly_aux_eligible_possible": int(possible and not old),
            "unfixable_by_support_resampling": int(not possible),
            "blockers": ";".join(sorted({r["blocker"] for r in records}))})

    summaries = []
    for kind in ("fit", "selection"):
        for fold in ["ALL"] + sorted(sources):
            for origin in ("ALL", "same", "cross"):
                group = [row for row in task_rows if row["kind"] == kind
                         and (fold == "ALL" or row["fold"] == fold)
                         and (origin == "ALL" or row["origin"] == origin)]
                if group:
                    summaries.append(summarize_tasks(group, fold, kind, origin))
    class_summary = []
    for kind in ("fit", "selection"):
        for origin in ("pooled", "same", "cross"):
            for cls in range(8):
                group = [row for row in unique_rows if row["kind"] == kind and row["class_id"] == cls and row["origin"] == origin]
                if not group:
                    continue
                class_summary.append(dict(kind=kind, origin=origin, class_id=cls, class_name=group[0]["class_name"],
                    unique_queries=len(group), unique_ships=len({row["mmsi"] for row in group}),
                    valid_queries=sum(row["local_valid"] for row in group),
                    invalid_queries=sum(not row["local_valid"] for row in group),
                    original_aux_eligible=sum(row["original_aux_eligible"] for row in group),
                    possible_aux_eligible=sum(row["possible_aux_eligible"] for row in group),
                    newly_aux_eligible_possible=sum(row["newly_aux_eligible_possible"] for row in group),
                    valid_but_still_ineligible=sum(row["local_valid"] and not row["possible_aux_eligible"] for row in group)))
    support_census = []
    for kind in ("fit", "selection"):
        for cls in ["ALL"] + list(range(8)):
            group = [row for row in candidate_rows if row["kind"] == kind and (cls == "ALL" or row["class_id"] == cls)]
            eligible = [row for row in group if row["quota_eligible"]]
            old = [row for row in group if row["original_support"]]
            class_group = [row for row in class_rows if row["kind"] == kind and (cls == "ALL" or row["class_id"] == cls)]
            original_rows, original_ships = {r["global_row"] for r in old}, {r["mmsi"] for r in old}
            eligible_rows, eligible_ships = {r["global_row"] for r in eligible}, {r["mmsi"] for r in eligible}
            support_census.append(dict(kind=kind, class_id=cls,
                class_name="ALL" if cls == "ALL" else class_group[0]["class_name"],
                original_unique_rows=len(original_rows), original_unique_ships=len(original_ships),
                eligible_unique_rows=len(eligible_rows), eligible_unique_ships=len(eligible_ships),
                additional_rows_outside_original_union=len(eligible_rows - original_rows),
                additional_ships_outside_original_union=len(eligible_ships - original_ships),
                original_slots=sum(r["original_quota"] for r in class_group),
                forced_slots=sum(r["original_quota"] for r in class_group if r["forced_original_class"]),
                supported_task_classes=sum(r["original_quota"] > 0 for r in class_group),
                changeable_task_classes=sum(r["changeable"] for r in class_group),
                unused_zero_quota_candidates=sum(r["unused_due_to_zero_quota"] for r in class_group)))
    for name, records in (("task_pools.csv", task_rows), ("class_pools.csv", class_rows),
                          ("pool_candidates.csv", candidate_rows), ("query_contexts.csv", query_rows),
                          ("unique_queries.csv", unique_rows), ("pool_summary.csv", summaries),
                          ("query_class_summary.csv", class_summary), ("support_census.csv", support_census)):
        write_csv(name, records)
    print(json.dumps(dict(
        overall=[row for row in summaries if row["fold"] == row["origin"] == "ALL"],
        cross_rare_classes=[row for row in class_summary if row["origin"] == "cross" and row["class_id"] in (1, 5, 6, 7)],
        support_census=support_census,
        minimum_original_valid_classes=min(row["original_valid_classes"] for row in task_rows),
        source_only=True, sampler_executed=False, training_executed=False), indent=2))


if __name__ == "__main__":
    main()
