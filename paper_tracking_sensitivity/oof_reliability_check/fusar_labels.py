"""Explicit FUSAR fine-label alignment, fixed before inspecting outcomes.

Only official_subcategory supplies a fine label. A populated official_category
must agree with its expected coarse family; unknown coarse families also fail
this check. Missing coarse labels provide no contradictory evidence. The five
official families contain no offshore-supply family, so OffshoreSupplyShip under
Cargo or Tug is excluded, as are BulkCarrier under Fishing or Tug. A standalone
OffshoreSupplyShip fine label with no coarse label can map to source ID 7.

Eligibility is mandatory: only True (or the CSV string "True") is included.
Existing eligibility exclusions take precedence over label decisions. There is
no fallback from CargoShip/Tanker to a fine class, no prediction-based repair,
and no quality, identity, dimension or image filtering beyond given eligibility.
The original five-class IDs/names and all input rows remain available for audit.
"""

from collections import Counter

import pandas as pd


CLASSES = (
    "bulk_carrier", "fishing_vessel", "general_cargo",
    "product_chemical_tanker", "container_ship", "crude_oil_tanker",
    "tug_towing", "offshore_supply",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}

# Fine spelling -> (source ID, compatible official coarse family).
# None means there is no compatible family in the official five-class cohort.
FINE_LABELS = {
    "BulkCarrier": (0, "Cargo"),
    "SelfDischargingBulkCarrier": (0, "Cargo"),
    "Fishing": (1, "Fishing"),
    "Trawler": (1, "Fishing"),
    "GeneralCargo": (2, "Cargo"),
    "ChemicalTanker": (3, "Tanker"),
    "OilProductsTanker": (3, "Tanker"),
    "Oil_ChemicalTanker": (3, "Tanker"),
    "ContainerShip": (4, "Cargo"),
    "CrudeOilTanker": (5, "Tanker"),
    "Tug": (6, "Tug"),
    "OffshoreSupplyShip": (7, None),
}
AMBIGUOUS_LABELS = {
    "SupplyVessel", "FishCarrier", "HeavyLoadCarrier",
    "Asphalt_BitumenTanker", "BunkeringTanker",
}


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def _eligible(value):
    # Never use bool(value): the CSV string "False" is truthy.
    return _text(value).lower() == "true"


def _map_row(row):
    fine = _text(row["official_subcategory"])
    coarse = _text(row["official_category"])
    entry = FINE_LABELS.get(fine)
    if not _eligible(row["eligible"]):
        reason = "ineligible"
    elif not fine:
        reason = "missing_label"
    elif entry is not None:
        reason = "coarse_fine_conflict" if coarse and coarse != entry[1] else "mapped"
    elif fine in {"Cargo", "CargoShip", "Tanker"} or fine.startswith(
        ("Cargo-Hazard", "Tanker-Hazard")
    ):
        reason = "coarse_label"
    elif fine in AMBIGUOUS_LABELS:
        reason = "ambiguous_label"
    else:
        reason = "unsupported_label"
    class_id = entry[0] if reason == "mapped" else None
    return {
        "class_id": class_id,
        "class_name": CLASSES[class_id] if class_id is not None else None,
        "label_included": reason == "mapped",
        "label_mapping_reason": reason,
    }


def map_fusar_labels(frame):
    """Copy the cohort without dropping/reordering rows or changing its index.

    Required columns are the official fine/coarse labels, original class_id and
    class_name, and eligible. Original exclusion_reason is preserved when present.
    Excluded rows receive null source labels; class_id uses nullable Int64.
    target_manifest_row is positional 0..N-1, independent of the input index.
    """
    required = {
        "official_category", "official_subcategory", "class_id", "class_name",
        "eligible",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing FUSAR cohort columns: {sorted(missing)}")
    result = frame.copy()
    result["official_five_class_id"] = frame["class_id"].to_numpy()
    result["official_five_class_name"] = frame["class_name"].to_numpy()
    result["target_manifest_row"] = range(len(frame))
    columns = ("class_id", "class_name", "label_included", "label_mapping_reason")
    labels = pd.DataFrame(
        [_map_row(row) for row in frame.to_dict("records")], columns=columns,
    )
    for column in columns:
        result[column] = labels[column].to_numpy()
    result["class_id"] = pd.array(result["class_id"], dtype="Int64")
    result["label_included"] = result["label_included"].astype(bool)
    return result


def summarize_fusar_mapping(mapped):
    """Return JSON-ready counts and provenance, never model statistics.

    class_counts is always eight integer counts in source ID order 0..7, counting
    only eligible, label-compatible rows. mapped_count equals included_count.
    Eligibility reasons count original ineligible rows; missing reasons are
    reported as "unspecified". Provenance retains both coarse and fine labels.
    """
    eligible = mapped["eligible"].map(_eligible).astype(bool)
    included = mapped.loc[mapped["label_included"]]
    excluded = mapped.loc[~mapped["label_included"]]
    counts = Counter(included["class_id"])
    class_counts = [int(counts[index]) for index in range(len(CLASSES))]
    eligibility_reasons = Counter(
        _text(row.get("exclusion_reason")) or "unspecified"
        for row in mapped.loc[~eligible].to_dict("records")
    )
    provenance_counts = Counter(
        (
            _text(row["official_category"]), _text(row["official_subcategory"]),
            _eligible(row["eligible"]), _text(row.get("exclusion_reason")),
            bool(row["label_included"]), row["label_mapping_reason"],
            _text(row["class_name"]),
        )
        for row in mapped.to_dict("records")
    )
    provenance = [
        {
            "official_category": coarse, "official_subcategory": fine,
            "eligible": is_eligible, "exclusion_reason": eligibility_reason,
            "label_included": is_included, "label_mapping_reason": reason,
            "class_name": name or None,
            "class_id": CLASS_TO_ID.get(name), "count": count,
        }
        for (coarse, fine, is_eligible, eligibility_reason, is_included, reason, name), count
        in sorted(provenance_counts.items())
    ]
    return {
        "original_count": len(mapped),
        "eligible_count": int(eligible.sum()),
        "ineligible_count": int((~eligible).sum()),
        "mapped_count": len(included),
        "included_count": len(included),
        "excluded_count": len(excluded),
        "eligible_unmapped_count": int((eligible & ~mapped["label_included"]).sum()),
        "class_names": list(CLASSES),
        "class_counts": class_counts,
        "present_class_ids": [i for i, count in enumerate(class_counts) if count],
        "absent_class_ids": [i for i, count in enumerate(class_counts) if not count],
        "excluded_by_reason": dict(Counter(excluded["label_mapping_reason"])),
        "eligibility_reasons": dict(eligibility_reasons),
        "inclusion_provenance": [row for row in provenance if row["label_included"]],
        "exclusion_provenance": [row for row in provenance if not row["label_included"]],
        "quality_filter_applied": False,
        "dimension_filter_applied": False,
    }
