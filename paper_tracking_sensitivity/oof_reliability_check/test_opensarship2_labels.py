import pandas as pd
import pytest

from opensarship2_labels import (
    CLASS_TO_ID, map_label, map_opensarship2_labels, map_target_row,
    summarize_mapping, validate_source_taxonomy,
)


@pytest.mark.parametrize("label, expected", [
    ("  BULK  Carrier ", "bulk_carrier"),
    ("Fishing", "fishing_vessel"),
    ("General_Cargo_Ship", "general_cargo"),
    ("Oil/Chemical Tanker", "product_chemical_tanker"),
    ("Oil Products Tanker", "product_chemical_tanker"),
    ("Chemical Tanker", "product_chemical_tanker"),
    ("Chemical_Oil_Products_Tanker", "product_chemical_tanker"),
    ("Container Ship", "container_ship"),
    ("Crude Oil Tanker", "crude_oil_tanker"),
    ("Pusher-Tug", "tug_towing"),
    ("Towing", "tug_towing"),
    ("Offshore Supply Ship", "offshore_supply"),
])
def test_explicit_aliases(label, expected):
    assert map_label(label) == expected


@pytest.mark.parametrize("label", [
    "Cargo", "Tanker", "Cargo Ship", "LPG Tanker", "LNG Tanker",
    "Tanker - Hazard A (Major)", "Cargo - Hazard B", "Tug/Supply Vessel",
    "Cargo/Containership", "Fish Carrier", "Cement Carrier", "Heavy Load Carrier",
    "Bunkering Tanker", "Fruit Juice Tanker", "Asphalt/Bitumen Tanker",
    "Multi Purpose Offshore Vessel", "Supply Vessel", "Tug/Ice Breaker",
])
def test_coarse_conflicting_or_unsupported_labels_are_not_fabricated(label):
    assert map_label(label) is None


@pytest.mark.parametrize("fine", ["LPG Tanker", "Tug/Supply Vessel", "Unspecified", "Cargo"])
def test_populated_unmappable_fine_field_blocks_fallback(fine):
    result = map_target_row({
        "label_elaborated": fine, "label_primary": "General Cargo", "label_rough": "Tug",
    })
    assert not result["label_included"]
    assert result["class_id"] is None
    assert result["label_field"] == "label_elaborated"


def test_fine_field_wins_over_conflicting_primary_and_rough():
    result = map_target_row({
        "label_elaborated": "Crude Oil Tanker", "label_primary": "Oil Products Tanker",
        "label_rough": "Cargo",
    })
    assert result["class_id"] == 5
    assert result["label_field"] == "label_elaborated"


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, "", "  "])
def test_only_missing_fields_allow_fallback(missing):
    result = map_target_row({"label_elaborated": missing, "label_primary": "Fishing"})
    assert result["class_id"] == 1
    assert result["label_field"] == "label_primary"
    result = map_target_row({"label_elaborated": missing, "label_primary": missing, "label_rough": "Towing"})
    assert result["class_id"] == 6
    assert result["label_field"] == "label_rough"


def test_no_rescue_from_identity_ais_or_raw_and_no_coarse_primary_fallback():
    result = map_target_row({"label_raw": "bulk_carrier", "ais_ship_type": 70})
    assert result["label_mapping_reason"] == "missing_label"
    result = map_target_row({"label_primary": "Tanker", "label_rough": "Fishing"})
    assert result["label_mapping_reason"] == "coarse_label"


def test_preserves_rows_quality_interference_and_source_ids_in_report():
    source = pd.DataFrame({
        "class_name": list(CLASS_TO_ID), "class_id": list(range(8)),
    })
    frame = pd.DataFrame({
        "label_elaborated": ["Offshore Supply Ship", "Cargo", None, "Oil Products Tanker"],
        "label_primary": ["Other", "Cargo", "Fishing", "Tanker"],
        "quality_flag": ["review", "ok", "bad", "ok"],
        "interference": [1, 0, 1, 1], "has_vh": [0, 1, 1, 1],
    }, index=[9, 2, 9, 0])
    result = map_opensarship2_labels(frame)
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    assert result.class_id.dropna().tolist() == [7, 1, 3]
    assert str(result.class_id.dtype) == "Int64"
    report = summarize_mapping(result, source)
    assert (report["total"], report["included"], report["excluded"]) == (4, 3, 1)
    assert report["included_by_field"] == {"label_elaborated": 2, "label_primary": 1}
    assert report["by_class"]["offshore_supply"]["target_count"] == 1
    assert report["by_class"]["bulk_carrier"]["target_count"] == 0
    assert report["excluded_by_reason"] == {"coarse_label": 1}
    assert not report["quality_filter_applied"]
    assert not report["interference_filter_applied"]
    source.loc[0, "class_id"] = 1
    with pytest.raises(ValueError, match="Source taxonomy"):
        validate_source_taxonomy(source)
