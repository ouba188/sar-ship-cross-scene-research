"""Bounded label alignment for current-source -> public OpenSARShip2.

Source IDs: visual_reliable_baseline/prepare_eight_class_adaptive.py, CLASSES,
confirmed against artifacts/eight_class_adaptive_20260916/manifest.csv.
Source aliases: b1_clean_expert.py, ALIAS; mmsi_fill_from_fdrive.py, TYPE2FINE.
The latter explicitly groups oil-products and chemical-oil-products tankers.
class_map_unified.csv also places Chemical_Tanker in product_chemical_tanker,
but differs on several other subtypes. Do not import its whole mapping:
Cargo/Cargo_Ship, Cement Carrier, Heavy Load Carrier, Fish Carrier and mixed
offshore/tug labels are not unambiguous aliases for this bridge.

Historical e181_opensarship2.py describes a 2,469-row interference0/shared3/
sizeclean, five-region experiment. Its selection and coarse labels are NOT
used here. All quality/interference values are retained; image eligibility
belongs to the caller. This module only aligns labels and reports coverage.
It neither selects support nor fits models: the runner must limit target
fitting labels to the authorized 64 support examples.
"""

from collections import Counter
import re

import pandas as pd


CLASSES = (
    "bulk_carrier", "fishing_vessel", "general_cargo",
    "product_chemical_tanker", "container_ship", "crude_oil_tanker",
    "tug_towing", "offshore_supply",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
LABEL_FIELDS = ("label_elaborated", "label_primary", "label_rough")

# Explicit equivalents only; normalization does not infer vessel function.
ALIASES = {name: name for name in CLASSES}
ALIASES.update({
    "bulk": "bulk_carrier",
    "self_discharging_bulk_carrier": "bulk_carrier",
    "fishing": "fishing_vessel",
    "commercial_fishing_vessel": "fishing_vessel",
    "general_cargo_ship": "general_cargo",
    "product_chemical": "product_chemical_tanker",
    "chemical_oil_products_tanker": "product_chemical_tanker",
    "oil_chemical_tanker": "product_chemical_tanker",
    "oil_products_tanker": "product_chemical_tanker",
    "chemical_tanker": "product_chemical_tanker",
    "container": "container_ship",
    "crude_oil": "crude_oil_tanker",
    "tug": "tug_towing",
    "pusher_tug": "tug_towing",
    "towing": "tug_towing",
    "towing_vessel": "tug_towing",
    "offshore": "offshore_supply",
    "offshore_supply_ship": "offshore_supply",
    "offshore_supply_vessel": "offshore_supply",
    "offshore_support_vessel": "offshore_supply",
})

AMBIGUOUS_LABELS = {
    "cargo_containership", "tug_supply_vessel", "tug_ice_breaker",
    "multi_purpose_offshore_vessel", "supply_vessel", "offshore_tug",
    "cement_carrier", "heavy_load_carrier", "fish_carrier",
    "fruit_juice_tanker", "bunkering_tanker", "asphalt_bitumen_tanker",
}


def normalize_label(value):
    """Normalize case and observed separators; preserve meaningful words."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[\s/_-]+", "_", str(value).strip().lower()).strip("_")


def map_label(value):
    """Return a canonical source class, or None for an unsupported label."""
    return ALIASES.get(normalize_label(value))


def map_target_row(row):
    """Use the first populated field, including populated unmappable labels.

    A missing fine field permits primary/rough fallback. An unknown, coarse,
    ambiguous or contradictory populated fine field does not. label_raw,
    AIS codes, vessel identity and quality are never label rescue sources.
    """
    field, value, normalized = "", "", ""
    for candidate in LABEL_FIELDS:
        raw = row.get(candidate)
        key = normalize_label(raw)
        if key:
            field, value, normalized = candidate, str(raw), key
            break
    name = ALIASES.get(normalized)
    if name is not None:
        reason = "mapped"
    elif not normalized:
        reason = "missing_label"
    elif normalized in {"cargo", "tanker", "cargo_coarse", "tanker_coarse"} or normalized.startswith(
        ("cargo_hazard_", "tanker_hazard_")
    ):
        reason = "coarse_label"
    elif normalized in AMBIGUOUS_LABELS:
        reason = "ambiguous_label"
    else:
        reason = "unsupported_label"
    return {
        "class_name": name,
        "class_id": CLASS_TO_ID[name] if name is not None else None,
        "label_included": name is not None,
        "label_field": field,
        "label_value": value,
        "label_normalized": normalized,
        "label_mapping_reason": reason,
    }


def map_opensarship2_labels(frame):
    """Return every input row, in input order/index, with mapping provenance.

    class_id is nullable Int64, always in the source's original 0..7 space.
    Select label_included for a label-compatible subset; no other eligibility
    filter is applied. Original target label/quality/interference columns stay.
    """
    columns = list(map_target_row({}))
    mapped = pd.DataFrame(
        [map_target_row(row) for row in frame.to_dict("records")],
        columns=columns,
    )
    result = frame.copy()
    for column in columns:
        result[column] = mapped[column].to_numpy()
    result["class_id"] = pd.array(result["class_id"], dtype="Int64")
    result["label_included"] = result["label_included"].astype(bool)
    return result


def validate_source_taxonomy(source):
    """Reject class-ID drift before a runner can attach labels to predictions."""
    pairs = source[["class_name", "class_id"]].drop_duplicates()
    actual = set(pairs.itertuples(index=False, name=None))
    expected = set(CLASS_TO_ID.items())
    if actual != expected:
        raise ValueError(f"Source taxonomy differs from eight-class IDs: {actual!r}")


def summarize_mapping(mapped, source=None):
    """JSON-ready alignment counts only, never performance/model statistics."""
    if source is not None:
        validate_source_taxonomy(source)
    included = mapped.loc[mapped.label_included]
    excluded = mapped.loc[~mapped.label_included]
    counts = Counter(included.class_name)
    source_counts = Counter(source.class_name) if source is not None else None
    by_class = {
        name: {
            "class_id": CLASS_TO_ID[name],
            "target_count": counts[name],
            "target_fraction": counts[name] / len(mapped) if len(mapped) else 0.0,
            "field_provenance": dict(Counter(
                included.loc[included.class_name.eq(name), "label_field"]
            )),
        }
        for name in CLASSES
    }
    if source_counts is not None:
        for name in CLASSES:
            by_class[name]["source_count"] = source_counts[name]
    provenance = []
    for (field, label, reason), group in mapped.groupby(
        ["label_field", "label_value", "label_mapping_reason"], sort=True
    ):
        provenance.append({
            "field": field, "label": label, "reason": reason,
            "count": len(group),
            "class_name": group.class_name.iloc[0] if reason == "mapped" else None,
        })
    return {
        "total": len(mapped), "included": len(included), "excluded": len(excluded),
        "coverage": len(included) / len(mapped) if len(mapped) else 0.0,
        "by_class": by_class,
        "included_by_field": dict(Counter(included.label_field)),
        "excluded_by_reason": dict(Counter(excluded.label_mapping_reason)),
        "label_provenance": provenance,
        "quality_filter_applied": False,
        "interference_filter_applied": False,
    }
