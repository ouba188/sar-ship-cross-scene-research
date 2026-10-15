"""Audit native-pixel context availability in the frozen 24-port episodes."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGION = ROOT / "region_cue_replication_20261013"
SCENE = ROOT / "scene_cross_dataset_redesign_20261014"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REGION))
from source_region_metadata import bbox_regions  # noqa: E402


def region_area(row, pol, padding):
    h, w = int(row[f"native_h_{pol}"]), int(row[f"native_w_{pol}"])
    bounds = [row[f"bbox_{axis}_{pol}"] for axis in ("x0", "y0", "x1", "y1")]
    box, _, valid, _ = bbox_regions(bounds, (h, w), padding)
    return (box[2] - box[0]) * (box[3] - box[1]) if valid else np.nan


def summarize(rows, keys):
    return (rows.groupby(keys, dropna=False).agg(
        chips=("sample_id", "size"), ships=("identity", "nunique"),
        scenes=("scene_id", "nunique"), valid=("valid", "sum"),
        near_ok=("near_ok", "sum"), far_ok=("far_ok", "sum"),
        both_ok=("both_ok", "sum"),
        near_p10=("near_frac", lambda x: x.quantile(.10)),
        near_median=("near_frac", "median"),
        far_p10=("far_frac", lambda x: x.quantile(.10)),
        far_median=("far_frac", "median"),
    ).reset_index())


def intersects(a, b):
    return (a[0] < b[2] and b[0] < a[2]
            and a[1] < b[3] and b[1] < a[3])


def chip_overlap(a, b):
    if a.scene_id != b.scene_id:
        return False
    for pol in ("vv", "vh"):
        x, y = a[f"resolved_crop_left_{pol}"], a[f"resolved_crop_top_{pol}"]
        u, v = b[f"resolved_crop_left_{pol}"], b[f"resolved_crop_top_{pol}"]
        if intersects((x, y, x + a[f"native_w_{pol}"], y + a[f"native_h_{pol}"]),
                      (u, v, u + b[f"native_w_{pol}"], v + b[f"native_h_{pol}"])):
            return True
    return False


def main():
    frame = pd.read_csv(REGION / "ports/frame.csv", dtype={"identity": str})
    reference = pd.read_csv(REGION / "ports/region_metadata.csv",
                            usecols=["sample_id", "region_area"])
    data = frame.merge(reference, on="sample_id", validate="one_to_one")
    for pol in ("vv", "vh"):
        size = data[f"native_h_{pol}"] * data[f"native_w_{pol}"]
        inner = data.apply(region_area, axis=1, args=(pol, .20))
        outer = data.apply(region_area, axis=1, args=(pol, .35))
        data[f"inner_{pol}"] = inner
        data[f"near_{pol}"] = outer - inner
        data[f"far_{pol}"] = size - outer
        data[f"near_frac_{pol}"] = data[f"near_{pol}"] / size
        data[f"far_frac_{pol}"] = data[f"far_{pol}"] / size
    actual = .5 * (data.inner_vv / (data.native_h_vv * data.native_w_vv)
                   + data.inner_vh / (data.native_h_vh * data.native_w_vh))
    np.testing.assert_allclose(actual[data.region_valid],
                               data.loc[data.region_valid, "region_area"], atol=1e-12)
    data["valid"] = data.region_valid
    data["near_frac"] = data[["near_frac_vv", "near_frac_vh"]].min(axis=1)
    data["far_frac"] = data[["far_frac_vv", "far_frac_vh"]].min(axis=1)
    data["near_ok"] = data.valid & (data.near_frac >= .05) & (data[["near_vv", "near_vh"]].min(axis=1) >= 32)
    data["far_ok"] = data.valid & (data.far_frac >= .20) & (data[["far_vv", "far_vh"]].min(axis=1) >= 64)
    data["both_ok"] = data.near_ok & data.far_ok
    data[["sample_id", "identity", "port", "scene_id", "class_id", "class_name",
          "near_frac", "far_frac", "near_ok", "far_ok", "both_ok"]].to_csv(
        HERE / "geometry_by_sample.csv", index=False)
    summarize(data, ["port"]).to_csv(HERE / "availability_by_port.csv", index=False)
    summarize(data, ["class_id", "class_name"]).to_csv(HERE / "availability_by_class.csv", index=False)

    positions = pd.read_csv(REGION / "source_region_metadata.csv", usecols=[
        "sample_id", "resolved_crop_left_vv", "resolved_crop_top_vv",
        "resolved_crop_left_vh", "resolved_crop_top_vh"])
    positions = positions.merge(data[["sample_id", "scene_id", "native_w_vv", "native_h_vv",
                                      "native_w_vh", "native_h_vh"]],
                                on="sample_id", validate="one_to_one").set_index("sample_id")
    by_id = data.set_index("sample_id", verify_integrity=True)
    folds = []
    overlaps = []
    overlap_ids = []
    queries = []
    quotas = []
    support_draws = []
    candidate_conflicts = []
    for parent in (SCENE / "ports_full_expansion", SCENE / "ports_broad_expansion_20261014"):
        for folder in sorted(parent.iterdir()):
            query_path = folder / "query.csv"
            support_path = folder / "support_draws.csv"
            if not query_path.exists() or not support_path.exists():
                continue
            query = pd.read_csv(query_path, dtype={"mmsi": str})
            support = pd.read_csv(support_path, dtype={"mmsi": str})
            candidates = pd.read_csv(folder / "support_candidates.csv", dtype={"mmsi": str})
            assert query.sample_id.is_unique
            assert support.groupby(["draw", "support_origin"]).size().eq(16).all()
            class_quotas = support.groupby(["draw", "support_origin", "class_id"]).size()
            assert all(class_quotas.loc[draw, origin].to_dict()
                       == class_quotas.loc[0, "same"].to_dict()
                       for draw in support.draw.unique() for origin in ("same", "cross"))
            assert set(query.sample_id).issubset(by_id.index)
            assert set(support.sample_id).issubset(by_id.index)
            query_pos = positions.loc[query.sample_id]
            same_candidate_pos = positions.loc[candidates.loc[
                candidates.support_origin == "same", "sample_id"]]
            conflict = {sid for sid, sr in same_candidate_pos.iterrows()
                        if any(chip_overlap(qr, sr) for _, qr in query_pos.iterrows())}
            candidate_conflicts.append({"fold": folder.name,
                                        "same_candidates": len(same_candidate_pos),
                                        "overlap_any_query": len(conflict)})
            for row in by_id.loc[query.sample_id].itertuples():
                queries.append({"fold": folder.name, "sample_id": row.Index,
                                "class_id": row.class_id, "class_name": row.class_name,
                                "near_ok": row.near_ok, "far_ok": row.far_ok,
                                "both_ok": row.both_ok})
            for role, samples in [("query", query.sample_id),
                                  ("same_support", support.loc[support.support_origin == "same", "sample_id"]),
                                  ("cross_support", support.loc[support.support_origin == "cross", "sample_id"])]:
                rows = by_id.loc[samples.unique()].reset_index()
                result = {
                    "fold": folder.name, "role": role, "chips": len(rows),
                    "ships": rows.identity.nunique(), "classes": rows.class_id.nunique(),
                    "scenes": rows.scene_id.nunique(), "near_ok": int(rows.near_ok.sum()),
                    "far_ok": int(rows.far_ok.sum()), "both_ok": int(rows.both_ok.sum()),
                    "near_median": float(rows.near_frac.median()),
                    "far_median": float(rows.far_frac.median()),
                }
                folds.append(result)
            assert not set(query.mmsi).intersection(set(support.mmsi))
            draw0 = support.loc[(support.draw == 0) & (support.support_origin == "same")]
            for class_id, quota in draw0.class_id.value_counts().sort_index().items():
                counts = {}
                for origin in ("same", "cross"):
                    pool = candidates.loc[(candidates.support_origin == origin)
                                          & (candidates.class_id == class_id)].copy()
                    pool["both_ok"] = by_id.loc[pool.sample_id, "both_ok"].to_numpy()
                    counts[f"{origin}_ships"] = pool.mmsi.nunique()
                    counts[f"{origin}_eligible_ships"] = pool.loc[pool.both_ok, "mmsi"].nunique()
                    counts[f"{origin}_eligible_no_overlap_ships"] = pool.loc[
                        pool.both_ok & ~pool.sample_id.isin(conflict), "mmsi"].nunique()
                    pool["far_ok"] = by_id.loc[pool.sample_id, "far_ok"].to_numpy()
                    counts[f"{origin}_far_no_overlap_ships"] = pool.loc[
                        pool.far_ok & ~pool.sample_id.isin(conflict), "mmsi"].nunique()
                quotas.append({"fold": folder.name, "class_id": class_id, "quota": quota,
                               **counts, "eligible_quota_feasible": all(
                                   counts[f"{origin}_eligible_ships"] >= quota
                                   for origin in ("same", "cross")),
                               "no_overlap_quota_feasible": all(
                                   counts[f"{origin}_eligible_no_overlap_ships"] >= quota
                                   for origin in ("same", "cross")),
                               "far_no_overlap_quota_feasible": all(
                                   counts[f"{origin}_far_no_overlap_ships"] >= quota
                                   for origin in ("same", "cross"))})
            for (draw, origin), group in support.groupby(["draw", "support_origin"]):
                support_draws.append({"fold": folder.name, "draw": draw,
                                      "support_origin": origin, "ships": group.mmsi.nunique(),
                                      "eligible_ships": int(by_id.loc[group.sample_id, "both_ok"].sum())})
            for draw in sorted(support.draw.unique()):
                same_ids = support.loc[(support.draw == draw) & (support.support_origin == "same"), "sample_id"]
                same_pos = positions.loc[same_ids]
                affected = set()
                pairs = 0
                for qid, qr in query_pos.iterrows():
                    for sid, sr in same_pos.iterrows():
                        if chip_overlap(qr, sr):
                            affected.add(qid)
                            pairs += 1
                overlap_ids.extend({"fold": folder.name, "draw": draw, "sample_id": qid}
                                   for qid in sorted(affected))
                overlaps.append({"fold": folder.name, "draw": draw,
                                 "query_chips": len(query), "overlap_query_chips": len(affected),
                                 "overlap_pairs": pairs})
    pd.DataFrame(folds).to_csv(HERE / "availability_by_fold.csv", index=False)
    pd.DataFrame(quotas).to_csv(HERE / "matched_support_quota.csv", index=False)
    pd.DataFrame(support_draws).to_csv(HERE / "support_draw_eligibility.csv", index=False)
    pd.DataFrame(candidate_conflicts).to_csv(HERE / "support_candidate_overlap.csv", index=False)
    pd.DataFrame(overlaps).to_csv(HERE / "same_scene_crop_overlap.csv", index=False)
    pd.DataFrame(overlap_ids, columns=["fold", "draw", "sample_id"]).to_csv(
        HERE / "same_scene_overlap_ids.csv", index=False)
    eligible = pd.DataFrame(queries)
    touched = {(item["fold"], item["sample_id"]) for item in overlap_ids}
    eligible["no_overlap_any_draw"] = [
        (row.fold, row.sample_id) not in touched for row in eligible.itertuples()]
    eligible["both_region_old_draw_eligible"] = eligible.both_ok & eligible.no_overlap_any_draw
    eligible["far_eligible_old_draws"] = eligible.far_ok & eligible.no_overlap_any_draw
    eligible.to_csv(HERE / "query_eligibility.csv", index=False)
    summary = {"rows": len(data), "valid": int(data.valid.sum()),
               "near_ok": int(data.near_ok.sum()), "far_ok": int(data.far_ok.sum()),
               "both_ok": int(data.both_ok.sum()), "folds": len(overlaps) // 5,
               "query_overlap_draws": sum(item["overlap_query_chips"] > 0 for item in overlaps),
               "queries": len(eligible), "both_region_old_draw_queries": int(eligible.both_region_old_draw_eligible.sum()),
               "far_eligible_queries": int(eligible.far_ok.sum()),
               "far_eligible_old_draw_queries": int(eligible.far_eligible_old_draws.sum())}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
