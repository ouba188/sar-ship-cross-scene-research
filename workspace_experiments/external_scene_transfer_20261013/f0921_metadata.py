"""Build F0921 acquisition metadata and deterministic capacity bounds only."""
import json
import math
import re
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
SAR = Path("E:/Hermes/paper_tracking/sensitivity")
MANIFEST = SAR / "f0921_transfer_20261007/target_manifest.csv"
INCIDENCE = SAR / "f0921_incidence_recovery_20261010/f0921_incidence.csv"
PORTS = OUT.parent / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv"
FIELDS = ["sample_id", "class_id", "class_name", "identity", "known_ship",
          "product", "acquisition", "acquisition_known", "scene", "status"]
LOCAL = "tile+rounded_ship_lonlat -> sensor/start/code -> product grid; convex interpolation"
ONLINE = "Exact sensor/start/suffix online product; native annotation grid; convex interpolation"
REMOTE = "ASF exact product identified; native grid unavailable without authenticated access"
METHODS = {"ok": {LOCAL, ONLINE}, "outside_grid": {ONLINE},
           "missing_remote_grid": {REMOTE}, "ambiguous_acquisition": {LOCAL}}
PRODUCT_RE = re.compile(
    r"(S1[ABC])_IW_GRDH_1SDV_\d{8}T\d{6}_\d{8}T\d{6}_(\d{6})_([0-9A-F]{6})_[0-9A-F]{4}"
)


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def counts(series):
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def identity(series):
    # The actual inputs contain plain numeric MMSIs, no embedded or float IDs.
    value = series.str.strip()
    return value.where(value.str.fullmatch(r"[2-7][0-9]{8}"), "")


def conflicts(frame, id_col):
    known = frame[frame[id_col].ne("")]
    n = known.groupby(id_col).class_id.nunique()
    bad = set(n[n.gt(1)].index)
    records = []
    for ship, group in known[known[id_col].isin(bad)].groupby(id_col, sort=True):
        records.append({"identity": ship, "rows": len(group),
                        "class_rows": counts(group.class_id)})
    return bad, records


def acquisition(product):
    match = PRODUCT_RE.fullmatch(product)
    if match is None:
        raise ValueError("Recovered product does not match the observed GRDH schema: " + product)
    return "|".join(match.groups())


def distribution(values):
    s = pd.Series(values, dtype=float)
    return {str(q): float(s.quantile(q)) for q in (0, .25, .5, .75, .9, .95, .99, 1)}


def records_by_group(frame, key, clean):
    records = []
    clean_groups = {k: g for k, g in clean.groupby(key, sort=False)}
    empty = clean.iloc[:0]
    for name, group in frame.groupby(key, sort=True):
        good = clean_groups.get(name, empty)
        records.append({
            key: name, "rows": len(group),
            "known_ships": int(group.loc[group.known_ship, "identity"].nunique()),
            "products": int(group.loc[group["product"].ne(""), "product"].nunique()),
            "acquisitions": int(group.loc[group.acquisition_known, "acquisition"].nunique()),
            "class_rows": counts(group.class_id),
            "consistent_known_ships": int(good.identity.nunique()),
            "consistent_ship_capacity_by_class": {
                str(k): int(v) for k, v in good.groupby("class_id").identity.nunique().items()
            },
        })
    return records


def main():
    manifest, side, ports = read(MANIFEST), read(INCIDENCE), read(PORTS)
    # These checks prevent wrong cache order, row multiplication, or silent unmapped IDs.
    if manifest.sample_id.duplicated().any() or side.sample_id.duplicated().any():
        raise ValueError("Duplicate sample_id prevents a one-to-one join")
    joined = manifest.merge(side, on="sample_id", how="left", sort=False,
                            validate="one_to_one", suffixes=("", "_incidence"), indicator=True)
    if joined.sample_id.tolist() != manifest.sample_id.tolist() or not joined["_merge"].eq("both").all():
        raise ValueError("Join lost target rows or changed manifest order")
    if not joined.scene.eq(joined.scene_incidence).all():
        raise ValueError("Manifest and incidence scene disagree")
    for status, group in joined.groupby("status"):
        if status not in METHODS or not set(group.method).issubset(METHODS[status]):
            raise ValueError("Unreviewed status/method combination: " + status)

    known_acq = joined.status.ne("ambiguous_acquisition")
    if not joined.loc[known_acq, "acquisition_candidates"].eq("1").all():
        raise ValueError("A supposedly recovered product has nonunique acquisition candidates")
    if not joined.loc[~known_acq, "product"].eq("").all():
        raise ValueError("Ambiguous rows unexpectedly carry a product; review provenance")
    product_map = {p: acquisition(p) for p in joined.loc[known_acq, "product"].unique()}
    data = pd.DataFrame({
        "sample_id": joined.sample_id,
        "class_id": joined.class_id.astype(int),
        "class_name": joined.get("class_name", pd.Series("", index=joined.index)),
        "identity": identity(joined.mmsi),
        "product": joined["product"].where(known_acq, ""),
        "acquisition": joined["product"].map(product_map).fillna(""),
        "acquisition_known": known_acq,
        "scene": joined.scene, "status": joined.status,
    })
    data["known_ship"] = data.identity.ne("")
    data = data[FIELDS]
    ports["identity"] = identity(ports.mmsi)
    ports["class_id"] = ports.class_id.astype(int)
    bad_f, conflict_f = conflicts(data, "identity")
    bad_p, conflict_p = conflicts(ports, "identity")
    known = data[data.acquisition_known]
    clean = known[known.known_ship & ~known.identity.isin(bad_f)]
    scene_records = records_by_group(known, "acquisition", clean)
    tile_records = records_by_group(data, "scene", clean)

    # Distinct ships outside a group include ships also observed inside it.
    # No query identity has yet been removed: the resulting bounds are optimistic.
    ship_acqs = clean[["identity", "class_id", "acquisition"]].drop_duplicates()
    global_class = clean.groupby("class_id").identity.nunique().to_dict()
    n_acqs = ship_acqs.groupby("identity").acquisition.nunique()
    exclusive = ship_acqs[ship_acqs.identity.map(n_acqs).eq(1)]
    exclusive_counts = exclusive.groupby(["acquisition", "class_id"]).size().to_dict()
    scene_names = known.groupby("acquisition").scene.unique().to_dict()
    for item in scene_records:
        name = item["acquisition"]
        inside = item["consistent_ship_capacity_by_class"]
        outside = {str(c): int(n - exclusive_counts.get((name, c), 0))
                   for c, n in global_class.items()}
        matched = {str(c): min(inside.get(str(c), 0), outside[str(c)]) for c in global_class}
        n = item["consistent_known_ships"]
        shared_classes = sum(v > 0 for v in matched.values())
        reserve = max(5, math.ceil(.3 * n))
        item.update(
            other_acquisition_ship_capacity_by_class=outside,
            matched_class_capacity_before_query=matched,
            matched_capacity_before_query=sum(matched.values()),
            query_reserve_30pct_min5=reserve,
            original_scene_names=sorted(scene_names[name]),
        )
        for budget in (16, 64):
            item["B" + str(budget)] = {
                "support_only_upper_eligible": shared_classes >= 2 and sum(matched.values()) >= budget,
                "min5_query_upper_eligible": shared_classes >= 2 and n >= budget + 5
                    and sum(matched.values()) >= budget,
                "reserve30pct_query_upper_eligible": shared_classes >= 2 and n - reserve >= budget
                    and sum(matched.values()) >= budget,
                "optimistic_query_ship_upper_at_budget": max(0, n - budget)
                    if shared_classes >= 2 and sum(matched.values()) >= budget else 0,
            }

    fships = set(data.loc[data.known_ship, "identity"])
    pships = set(ports.loc[ports.identity.ne(""), "identity"])
    shared = fships & pships
    fgroups = {k: g for k, g in data[data.identity.isin(shared)].groupby("identity")}
    pgroups = {k: g for k, g in ports[ports.identity.isin(shared)].groupby("identity")}
    overlaps = []
    for ship in sorted(shared):
        f, p = fgroups[ship], pgroups[ship]
        fc, pc = set(f.class_id), set(p.class_id)
        overlaps.append({
            "identity": ship, "f0921_rows": len(f), "ports24_rows": len(p),
            "f0921_classes": sorted(fc), "ports24_classes": sorted(pc),
            "same_class_set": fc == pc, "union_class_conflict": len(fc | pc) > 1,
            "ports": sorted(p.port.unique()),
            "f0921_known_acquisition_rows": int(f.acquisition_known.sum()),
        })
    coverage = []
    for c, group in data.groupby("class_id", sort=True):
        valid = group[group.acquisition_known]
        consistent = clean[clean.class_id.eq(c)]
        per_acq = consistent.groupby("acquisition").identity.nunique()
        coverage.append({
            "class_id": int(c), "class_names": sorted(group.class_name.unique()),
            "rows": len(group), "known_ships": int(group.identity.nunique()),
            "known_acquisition_rows": len(valid), "known_acquisitions": int(valid.acquisition.nunique()),
            "known_acquisition_fraction": len(valid) / len(group),
            "unknown_acquisition_rows": len(group) - len(valid),
            "known_products": int(valid["product"].nunique()),
            "consistent_known_acquisition_ships": int(consistent.identity.nunique()),
            "conflicting_identity_rows": int(group.identity.isin(bad_f).sum()),
            "shared_ports24_ships": int(group.loc[group.identity.isin(shared), "identity"].nunique()),
            "single_class_acquisitions_at_least_B16_ships": int(per_acq.ge(16).sum()),
            "single_class_acquisitions_at_least_B64_ships": int(per_acq.ge(64).sum()),
        })
    budget_summary = {}
    for b in (16, 64):
        key = "B" + str(b)
        budget_summary[key] = {
            flag: sum(item[key][flag] for item in scene_records)
            for flag in ("support_only_upper_eligible", "min5_query_upper_eligible",
                         "reserve30pct_query_upper_eligible")
        }
        budget_summary[key]["sum_optimistic_query_ship_slots_min5_upper"] = sum(
            item[key]["optimistic_query_ship_upper_at_budget"]
            for item in scene_records if item[key]["min5_query_upper_eligible"])
    semantic_counts = []
    for (status, method, candidates), group in joined.groupby(
            ["status", "method", "acquisition_candidates"], sort=True):
        semantic_counts.append(dict(status=status, method=method, acquisition_candidates=int(candidates),
                                    rows=len(group), unique_nonempty_products=int(
                                        group.loc[group["product"].ne(""), "product"].nunique())))
    report = {
        "date": "2026-10-13",
        "scope": "Metadata only; no fitting, sampling, prediction or feature extraction.",
        "inputs": {"target_manifest": str(MANIFEST), "incidence": str(INCIDENCE), "ports24_manifest": str(PORTS)},
        "schema": FIELDS,
        "definitions": {
            "identity": "Plain 9-digit ship MMSI matching [2-7][0-9]{8}; otherwise empty. Format validity is not registry verification. All observed F MMSIs also match the historical project [1-9] rule.",
            "known_ship": "identity is nonempty; label conflicts do not change known_ship.",
            "acquisition": "platform|absoluteorbit|datatake parsed only from uniquely recovered product; leading zeroes preserved.",
            "scene": "Original manifest date/UTM tile-collection label, NOT acquisition or port.",
            "status": "Original incidence recovery status; not a generalized acquisition-availability status.",
            "product": "Actual uniquely recovered full product only; never old_tile_product or tile-first fallback.",
            "outside_grid": "Unique acquisition known; angle missing because ship position is outside native grid convex hull.",
            "missing_remote_grid": "Unique exact ASF product known; angle unavailable because native grid was not retrieved.",
            "ambiguous_acquisition": "Two candidate acquisitions; blank product/acquisition, acquisition_known=False.",
            "conflicts": "Preserve source labels and all CSV rows. Exclude globally F0921 multi-class known identities only from clean capacity counts.",
            "capacity": "One observation per distinct consistent known MMSI per support/query arm; group by acquisition.",
            "bounds": "Optimistic necessary conditions, not actual enrollments. min5 or 30pct query reserve subtracts ship count from inside pool only; other pool still includes future query identities. Require >=2 shared support classes; class-specific query reservations/identities and >=2 actual query classes must be checked by main experiment.",
            "independence": "Ships may recur across acquisitions; summed scene slots and multiple draws are not independent ships or datasets.",
        },
        "join": {"manifest_rows": len(manifest), "sidecar_rows": len(side),
                 "extra_sidecar_ids": len(set(side.sample_id) - set(manifest.sample_id)),
                 "matched_rows": len(data), "unique_ids": int(data.sample_id.nunique()),
                 "original_order_preserved": True, "scene_agreement": True},
        "totals": {
            "rows": len(data), "known_ship_rows": int(data.known_ship.sum()), "known_ships": len(fships),
            "known_acquisition_rows": len(known), "unknown_acquisition_rows": len(data) - len(known),
            "known_acquisitions": int(known.acquisition.nunique()), "known_products": int(known["product"].nunique()),
            "original_scene_labels": int(data.scene.nunique()),
            "known_acquisition_rows_missing_angle": int((known.status != "ok").sum()),
            "status_counts": counts(data.status),
            "consistent_known_acquisition_rows": len(clean),
            "consistent_known_acquisition_ships": int(clean.identity.nunique()),
        },
        "status_method_semantics": semantic_counts, "class_coverage": coverage,
        "identity_conflicts": {
            "f0921_ships": len(bad_f), "f0921_rows": int(data.identity.isin(bad_f).sum()),
            "f0921_records": conflict_f, "ports24_ships": len(bad_p),
            "ports24_rows": int(ports.identity.isin(bad_p).sum()), "ports24_records": conflict_p,
        },
        "ports24_overlap": {
            "ports24_rows": len(ports), "ports24_ports": int(ports.port.nunique()),
            "ports24_known_ships": len(pships), "ports24_unknown_identity_rows": int(ports.identity.eq("").sum()),
            "shared_known_ships": len(shared), "f0921_rows": int(data.identity.isin(shared).sum()),
            "ports24_overlap_rows": int(ports.identity.isin(shared).sum()),
            "different_class_set_ships": sum(not x["same_class_set"] for x in overlaps),
            "union_class_conflict_ships": sum(x["union_class_conflict"] for x in overlaps),
            "by_port": [
                dict(port=p, shared_ships=int(g.loc[g.identity.isin(shared), "identity"].nunique()),
                     overlap_rows=int(g.identity.isin(shared).sum()))
                for p, g in ports.groupby("port", sort=True)
            ],
            "records": overlaps,
        },
        "capacity_quantiles": {
            "raw_known_ships_per_acquisition": distribution([x["known_ships"] for x in scene_records]),
            "consistent_known_ships_per_acquisition": distribution([x["consistent_known_ships"] for x in scene_records]),
            "matched_before_query": distribution([x["matched_capacity_before_query"] for x in scene_records]),
        },
        "budget_upper_bounds": budget_summary,
        "acquisition_inventory": scene_records,
        "original_scene_inventory": tile_records,
        "limitations": [
            "Acquisition identity does not establish equal incidence, background, port, or independently collected evidence.",
            "nearest24 is not a verified port assignment. Original scene labels are date/UTM collections.",
            "No angle matching is implemented; 400 known-acquisition rows have missing angle.",
            "F0921 is historically reused and has recorded unlabeled encoder pretraining exposure; this preparation does not establish an untouched test dataset.",
            "F0921 public release/license status is unestablished in project records.",
            "Cross-dataset source fitting must remove shared known MMSIs when claiming vessel disjointness; this metadata task does not perform that purge or erase historical encoder exposure.",
            "Capacity upper bounds exclude F0921 internal label-conflicting identities but do not remove ports24 overlap; a protocol requiring that exclusion has smaller or equal capacity.",
            "All bounds precede actual feature availability checks and fixed query selection; no eligible episodes or performance claims have been produced.",
        ],
    }
    data.to_csv(OUT / "f0921_metadata.csv", index=False, encoding="utf-8")
    (OUT / "F0921_METADATA.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    t = report["totals"]
    ov = report["ports24_overlap"]
    lines = [
        "# F0921 \u91c7\u96c6\u5143\u6570\u636e\u4e0e\u5bb9\u91cf\u4e0a\u9650",
        "",
        "\u65e5\u671f\uff1a2026-10-13\u3002\u4ec5\u5143\u6570\u636e\u51c6\u5907\uff1b\u6ca1\u6709\u62bd\u6837\u3001\u62df\u5408\u3001\u8bad\u7ec3\u3001\u9884\u6d4b\u6216\u7279\u5f81\u63d0\u53d6\u3002",
        "",
        "## \u7ed3\u679c\u4e0e\u8fde\u63a5",
        "",
        f"\u539f manifest \u5171 {len(data):,} \u884c\uff0c\u6309 sample_id \u4e00\u5bf9\u4e00\u8fde\u63a5\uff0c\u539f\u987a\u5e8f\u5b8c\u6574\u4fdd\u7559\u3002"
        f"\u5168\u90e8 {t['known_ship_rows']:,} \u884c\u5177\u6709\u683c\u5f0f\u6709\u6548\u8239\u8236 MMSI\uff0c\u5171 {t['known_ships']:,} \u4e2a\u4e0d\u540c\u8eab\u4efd\u3002",
        f"\u552f\u4e00\u6062\u590d\u91c7\u96c6 {t['known_acquisition_rows']:,} \u884c\uff0c\u8986\u76d6 {t['known_acquisitions']:,} \u6b21\u91c7\u96c6\u3001"
        f"{t['known_products']:,} \u4e2a\u4ea7\u54c1\uff1b{t['unknown_acquisition_rows']} \u884c\u91c7\u96c6\u6b67\u4e49\u4fdd\u7559\u7a7a\u503c\u3002",
        "\u8f93\u51fa\u5b57\u6bb5\uff1a" + ", ".join(FIELDS) + "\u3002",
        "",
        "## \u72b6\u6001\u8bed\u4e49",
        "",
        "- acquisition = platform|absoluteorbit|datatake\uff0c\u4fdd\u7559\u8f68\u9053\u524d\u5bfc\u96f6\uff1b\u53ea\u89e3\u6790\u5b9e\u9645\u552f\u4e00\u6062\u590d\u4ea7\u54c1\u3002",
        "- scene \u539f\u6837\u4fdd\u7559\u65e5\u671f/UTM\u96c6\u5408\u540d\uff0c\u4e0d\u80fd\u5f53\u4f5c\u91c7\u96c6\u6216\u6e2f\u53e3\u3002\u7edf\u8ba1\u4e2d\u7684\u201c\u6bcf\u666f\u201d\u4ee5\u4e0b\u5747\u6307 acquisition\u3002",
        "- status \u539f\u6837\u4fdd\u7559\u89d2\u5ea6\u6062\u590d\u72b6\u6001\u3002outside_grid=389 \u4e0e missing_remote_grid=11 \u5747\u6709\u5df2\u77e5\u91c7\u96c6\uff0c\u89d2\u5ea6\u7f3a\u5931\u4e0d\u7b49\u4e8e\u91c7\u96c6\u672a\u77e5\u3002",
        "- ambiguous_acquisition=139\uff0c\u5747\u6709\u4e24\u4e2a\u5019\u9009\uff1b\u4e0d\u4f7f\u7528 old_tile_product\uff0c\u4e0d\u9ed8\u8ba4\u5206\u914d\u4ea7\u54c1\u3002",
        "- known_ship \u4ec5\u8868\u793a\u6709\u6548\u683c\u5f0f\u8eab\u4efd\uff1b\u4e0d\u8868\u793a\u6807\u7b7e\u4e00\u81f4\uff0c\u4e5f\u4e0d\u662f\u6ce8\u518c\u6570\u636e\u5e93\u6838\u9a8c\u3002identity \u4ec5\u63a5\u53d7 [2-7][0-9]{8}\uff1b\u672c\u6b21\u4e24\u6570\u636e\u6e90\u7684\u6709\u6548\u8eab\u4efd\u4e0e\u9879\u76ee\u65e7 [1-9] \u53e3\u5f84\u5b8c\u5168\u4e00\u81f4\u3002",
        "- status/method \u4e0e\u539f\u6062\u590d\u811a\u672c\u9010\u9879\u6838\u5bf9\uff0c\u7ec4\u5408\u8ba1\u6570\u89c1 JSON \u7684 status_method_semantics\u3002",
        "",
        "## \u6bcf\u7c7b\u8986\u76d6",
        "",
        "| \u7c7b\u522b | \u6837\u672c | \u4e0d\u540c\u8239 | \u5df2\u77e5\u91c7\u96c6\u6837\u672c | \u91c7\u96c6\u6570 | \u6b67\u4e49\u6837\u672c | \u51b2\u7a81\u8eab\u4efd\u6837\u672c |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in coverage:
        lines.append(f"| {c['class_id']} {','.join(c['class_names'])} | {c['rows']} | {c['known_ships']} | "
                     f"{c['known_acquisition_rows']} | {c['known_acquisitions']} | "
                     f"{c['unknown_acquisition_rows']} | {c['conflicting_identity_rows']} |")
    lines += [
        "",
        "## \u540c\u666f\u5bb9\u91cf\u53ca B16/B64",
        "",
        f"\u5bb9\u91cf\u7edf\u8ba1\u5254\u9664 F0921 \u5185\u90e8\u591a\u7c7b\u8eab\u4efd\uff0c\u4fdd\u7559 {len(clean):,} \u884c\u3001"
        f"{clean.identity.nunique():,} \u4e2a\u4e0d\u540c\u8239\u3002\u539f CSV \u4e0d\u5220\u884c\u3001\u4e0d\u6539\u6807\u7b7e\u3002",
        "\u6bcf\u4e2a\u91c7\u96c6\u7684\u6837\u672c\u6570\u3001\u4e0d\u540c\u8239\u6570\u3001\u7c7b\u522b\u8239\u6570\u3001\u4ea7\u54c1\u6570\u3001\u5176\u4ed6\u91c7\u96c6\u8239\u5bb9\u91cf\u4e0e\u5339\u914d\u4e0a\u9650\uff0c"
        "\u5b8c\u6574\u4fdd\u5b58\u5728 JSON acquisition_inventory\uff1b\u539f scene \u96c6\u5408\u7684\u5bb9\u91cf\u53e6\u5b58 original_scene_inventory\u3002",
        "\u6bcf\u7c7b\u5339\u914d\u5bb9\u91cf = min(\u540c\u91c7\u96c6\u4e0d\u540c\u8239\u6570, \u5176\u4ed6\u91c7\u96c6\u4e0d\u540c\u8239\u6570)\uff0c\u518d\u8de8\u7c7b\u6c42\u548c\uff1b\u81f3\u5c11\u4e24\u7c7b\u53ef\u7528\u3002"
        "\u4e0d\u9884\u7559\u67e5\u8be2\u8239\u65f6\u662f\u6700\u5bbd\u677e\u4e0a\u9650\uff1b\u53e6\u62a5\u544a\u9884\u7559\u81f3\u5c115\u8239\uff0c\u4ee5\u53ca max(5,ceil(30%\u8239\u6570)) \u7684\u4e0a\u9650\u3002",
        "\u8fd9\u4e9b\u4ec5\u4e3a\u5fc5\u8981\u6761\u4ef6\uff1a\u5c1a\u672a\u6309\u5b9e\u9645\u67e5\u8be2\u8eab\u4efd\u4ece\u4e24\u4fa7\u540c\u65f6\u6392\u9664\uff0c\u5c1a\u672a\u5206\u914d\u67e5\u8be2\u7c7b\u522b\u3001\u56fa\u5b9a\u652f\u6301\u914d\u989d"
        "\u6216\u68c0\u67e5\u7279\u5f81\u53ef\u7528\u6027\u3002\u56e0\u6b64\u4e0d\u80fd\u628a\u4e0b\u8868\u79f0\u4e3a\u5df2\u786e\u8ba4\u53ef\u6267\u884c\u573a\u666f\u6570\u3002",
        "",
        "| B | \u672a\u7559\u67e5\u8be2\u7684\u91c7\u96c6\u6570\u4e0a\u9650 | \u81f3\u5c11\u75595\u8239\u4e0a\u9650 | \u755930%\u4e14\u81f3\u5c115\u8239\u4e0a\u9650 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for b, stats in budget_summary.items():
        lines.append(f"| {b} | {stats['support_only_upper_eligible']} | "
                     f"{stats['min5_query_upper_eligible']} | {stats['reserve30pct_query_upper_eligible']} |")
    q = report["capacity_quantiles"]["consistent_known_ships_per_acquisition"]
    lines += [
        "",
        f"\u53bb\u51b2\u7a81\u540e\u6bcf\u91c7\u96c6\u4e0d\u540c\u8239\u6570\uff1a\u4e2d\u4f4d\u6570 {q['0.5']:g}\uff0cP90 {q['0.9']:g}\uff0c"
        f"P95 {q['0.95']:g}\uff0c\u6700\u5927 {q['1']:g}\u3002\u5b8c\u6574\u5206\u4f4d\u6570\u53ca\u9010\u666f\u6e05\u5355\u89c1 JSON\u3002",
        "\u6bcf\u7c7b\u8239\u5bb9\u91cf\u4e0d\u53ef\u628a\u540c\u8239\u591a\u6b21\u89c2\u6d4b\u76f8\u52a0\uff1b\u540c\u8239\u8de8\u91c7\u96c6\u4ecd\u53ef\u91cd\u590d\u51fa\u73b0\uff0c\u5404\u666f\u5bb9\u91cf\u4e4b\u548c\u4e0d\u662f\u72ec\u7acb\u8239\u603b\u6570\u3002"
        "\u5bb9\u91cf\u5c1a\u672a\u6392\u9664\u4e0e24\u6e2f\u91cd\u53e0\u8eab\u4efd\uff0c\u82e5\u534f\u8bae\u8981\u6c42\u6392\u9664\uff0c\u5b9e\u9645\u5bb9\u91cf\u53ea\u80fd\u66f4\u5c0f\u6216\u76f8\u7b49\u3002",
        "",
        "## \u8eab\u4efd\u6807\u7b7e\u51b2\u7a81\u4e0e24\u6e2f\u4ea4\u96c6",
        "",
        f"F0921 \u5185\u90e8\u591a\u7c7b\u522b MMSI\uff1a{len(bad_f)} \u8239\u3001{report['identity_conflicts']['f0921_rows']} \u884c\uff1b"
        f"24\u6e2f\u5185\u90e8\uff1a{len(bad_p)} \u8239\u3001{report['identity_conflicts']['ports24_rows']} \u884c\u3002\u539f\u59cb\u7c7b\u522b\u5747\u4fdd\u7559\u3002",
        f"24\u6e2f\u6e05\u5355 {len(ports):,} \u884c\u3001{ports.port.nunique()} \u6e2f\u3001{len(pships):,} \u4e2a\u5df2\u77e5\u8239\uff0c"
        f"{ov['ports24_unknown_identity_rows']} \u884c\u8eab\u4efd\u672a\u77e5\u3002\u4e24\u6570\u636e\u96c6\u5171\u4eab {len(shared):,} \u4e2a\u5df2\u77e5 MMSI\uff0c"
        f"\u5bf9\u5e94 F0921 {ov['f0921_rows']:,} \u884c\u300124\u6e2f {ov['ports24_overlap_rows']:,} \u884c\u3002",
        f"\u4ea4\u96c6\u4e2d\u7c7b\u522b\u96c6\u5408\u4e0d\u540c {ov['different_class_set_ships']} \u8239\uff0c\u5408\u5e76\u540e\u5177\u6709\u591a\u7c7b\u522b "
        f"{ov['union_class_conflict_ships']} \u8239\u3002\u5b8c\u6574\u8eab\u4efd\u3001\u4e24\u4fa7\u6807\u7b7e/\u884c\u6570\u548c\u6e2f\u53e3\u5b58\u4e8e "
        "ports24_overlap.records\uff1b\u5185\u90e8\u51b2\u7a81\u6e05\u5355\u5b58\u4e8e identity_conflicts\u3002",
        "\u58f0\u79f0\u8de8\u6570\u636e\u96c6\u8239\u53ea\u9694\u79bb\u65f6\uff0c\u4e3b\u5b9e\u9a8c\u987b\u4ece\u5b9e\u9645\u6e90\u62df\u5408\u6570\u636e\u6392\u9664\u5171\u4eab MMSI\uff1b"
        "\u4ec5\u51c6\u5907\u5143\u6570\u636e\u4e0d\u4f1a\u6267\u884c\u6392\u9664\uff0c\u4e5f\u4e0d\u4f1a\u6d88\u9664\u5386\u53f2\u7f16\u7801\u5668\u9884\u8bad\u7ec3\u66b4\u9732\u3002",
        "",
        "## \u4f7f\u7528\u8fb9\u754c",
        "",
        "F0921 \u5df2\u88ab\u5386\u53f2\u5f00\u53d1\u4f7f\u7528\uff0c\u4e14\u6709\u65e0\u6807\u7b7e\u7f16\u7801\u5668\u9884\u8bad\u7ec3\u66b4\u9732\u8bb0\u5f55\uff0c\u4e0d\u80fd\u79f0\u4e3a\u5168\u65b0\u72ec\u7acb\u786e\u8ba4\uff1b"
        "\u516c\u5f00\u53d1\u5e03\u4e0e\u8bb8\u53ef\u72b6\u6001\u5728\u9879\u76ee\u8bb0\u5f55\u4e2d\u672a\u786e\u7acb\u3002nearest24 \u4e0d\u80fd\u4f5c\u4e3a\u771f\u5b9e\u6e2f\u53e3\u3002",
        "\u540c\u4e00\u6b21\u91c7\u96c6\u4e0d\u4fdd\u8bc1\u76f8\u540c\u5165\u5c04\u89d2\u3001\u80cc\u666f\u6216\u6e2f\u53e3\uff1b\u672c\u6b21\u6ca1\u6709\u89d2\u5ea6\u5339\u914d\u6216\u7269\u7406\u539f\u56e0\u8bc6\u522b\u3002"
        "\u8fd9\u91cc\u53ea\u62a5\u544a\u91c7\u96c6\u5f52\u5c5e\u4e0e\u6570\u636e\u5bb9\u91cf\u9650\u5236\uff0c\u4e0d\u5c06\u65e7\u8f93\u5165\u7ba1\u7ebf\u95ee\u9898\u91cd\u8ff0\u4e3a\u9886\u57df\u7814\u7a76\u539f\u56e0\u3002",
        "JSON \u4fdd\u7559\u6240\u6709\u7edf\u8ba1\u53e3\u5f84\u3001\u9010\u91c7\u96c6\u6e05\u5355\u3001\u8eab\u4efd\u51b2\u7a81\u548c\u4ea4\u96c6\u3002\u5e03\u5c14\u5217\u91c7\u7528 True/False\uff1b"
        "\u8bfb\u53d6 CSV \u5efa\u8bae identity/sample_id/acquisition \u6309\u5b57\u7b26\u4e32\u3001keep_default_na=False\uff0c\u7a7a\u8eab\u4efd\u4e0d\u5f97\u586b\u6210\u4f2a\u8239\u53ea\u3002",
        "",
        "## \u6765\u6e90\u4e0e\u590d\u73b0",
        "",
        *[f"- {key}: `{value}`" for key, value in report["inputs"].items()],
        "- \u8bed\u4e49\u6765\u6e90\uff1aoof_reliability_check/external_knowledge_metadata.py \u4e0e recover_f0921_incidence_online.py\uff1b\u8bfb\u53d6\u65e2\u6709\u6062\u590d\u62a5\u544a\uff0c\u672a\u91cd\u65b0\u6062\u590d\u6216\u4e0b\u8f7d\u3002",
        "- \u8fd0\u884c\uff1a\u73b0\u6709 ClearSAR Python\uff0c\u53c2\u6570 -X utf8 -B\uff0c\u6267\u884c\u672c\u76ee\u5f55 f0921_metadata.py\u3002",
        "",
    ]
    (OUT / "F0921_METADATA_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"totals": t, "bounds": budget_summary,
                      "conflicts": {"f0921_ships": len(bad_f), "ports24_ships": len(bad_p)},
                      "overlap": {k: v for k, v in ov.items() if k not in ("records", "by_port")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

