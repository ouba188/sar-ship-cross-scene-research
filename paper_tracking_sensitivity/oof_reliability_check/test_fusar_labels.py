import json
from pathlib import Path

import pandas as pd
import pytest

from fusar_labels import map_fusar_labels, summarize_fusar_mapping


def cohort(rows):
    return pd.DataFrame([
        {
            "class_id": 0, "class_name": "cargo", "eligible": True,
            "exclusion_reason": "", "official_category": "Cargo",
            "official_subcategory": "BulkCarrier", **row,
        }
        for row in rows
    ])


@pytest.mark.parametrize("coarse,fine,class_id,name", [
    ("Cargo", "BulkCarrier", 0, "bulk_carrier"),
    ("Cargo", "SelfDischargingBulkCarrier", 0, "bulk_carrier"),
    ("Fishing", "Fishing", 1, "fishing_vessel"),
    ("Fishing", "Trawler", 1, "fishing_vessel"),
    ("Cargo", "GeneralCargo", 2, "general_cargo"),
    ("Tanker", "ChemicalTanker", 3, "product_chemical_tanker"),
    ("Tanker", "OilProductsTanker", 3, "product_chemical_tanker"),
    ("Tanker", "Oil_ChemicalTanker", 3, "product_chemical_tanker"),
    ("Cargo", "ContainerShip", 4, "container_ship"),
    ("Tanker", "CrudeOilTanker", 5, "crude_oil_tanker"),
    ("Tug", "Tug", 6, "tug_towing"),
    (None, "OffshoreSupplyShip", 7, "offshore_supply"),
])
def test_explicit_equivalences(coarse, fine, class_id, name):
    result = map_fusar_labels(cohort([
        {"official_category": coarse, "official_subcategory": fine},
    ]))
    assert result.class_id.tolist() == [class_id]
    assert result.class_name.tolist() == [name]
    assert result.label_included.tolist() == [True]
    assert result.label_mapping_reason.tolist() == ["mapped"]


@pytest.mark.parametrize("coarse,fine", [
    ("Fishing", "BulkCarrier"), ("Tug", "BulkCarrier"),
    ("Cargo", "OffshoreSupplyShip"), ("Tug", "OffshoreSupplyShip"),
    ("Dredger", "Tug"), ("Cargo", "CrudeOilTanker"),
    ("Unspecified", "BulkCarrier"),
])
def test_coarse_fine_conflicts_are_excluded(coarse, fine):
    result = map_fusar_labels(cohort([
        {"official_category": coarse, "official_subcategory": fine},
    ]))
    assert result.label_mapping_reason.tolist() == ["coarse_fine_conflict"]
    assert not result.label_included.any()
    assert result.class_id.isna().all()
    assert result.class_name.isna().all()


@pytest.mark.parametrize("fine,reason", [
    ("CargoShip", "coarse_label"), ("Tanker", "coarse_label"),
    ("Cargo-HazardA", "coarse_label"), ("Tanker-HazardD", "coarse_label"),
    ("SupplyVessel", "ambiguous_label"), ("FishCarrier", "ambiguous_label"),
    ("LPGTanker", "unsupported_label"), ("OreCarrier", "unsupported_label"),
    ("Dredger", "unsupported_label"), ("Tug/SupplyVessel", "unsupported_label"),
    (None, "missing_label"), (pd.NA, "missing_label"), ("  ", "missing_label"),
])
def test_no_coarse_identity_or_prediction_rescue(fine, reason):
    result = map_fusar_labels(cohort([{
        "official_category": "Tug", "official_subcategory": fine,
        "class_id": 3, "class_name": "tug", "predicted_class_id": 6,
        "mmsi": "123456789", "ais_ship_type": 52,
    }]))
    assert result.label_mapping_reason.tolist() == [reason]
    assert not result.label_included.any()
    assert result.class_id.isna().all()


@pytest.mark.parametrize("eligible", [False, "False", pd.NA, None, ""])
def test_requires_explicit_true_eligibility(eligible):
    result = map_fusar_labels(cohort([{
        "eligible": eligible, "exclusion_reason": "shared_main_mmsi",
    }]))
    assert result.label_mapping_reason.tolist() == ["ineligible"]
    assert result.class_id.isna().all()
    assert result.exclusion_reason.tolist() == ["shared_main_mmsi"]


def test_preserves_rows_original_labels_and_metadata_without_quality_filters():
    frame = cohort([
        {"eligible": "True", "quality_flag": "bad", "width": 0},
        {"eligible": "False", "exclusion_reason": "shared_main_mmsi"},
        {"eligible": False, "exclusion_reason": "invalid_mmsi"},
        {"official_subcategory": "GeneralCargo", "width": None},
    ])
    frame.index = [9, 2, 9, 0]
    original = frame.copy(deep=True)
    result = map_fusar_labels(frame)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(
        result[frame.columns.drop(["class_id", "class_name"])],
        frame.drop(columns=["class_id", "class_name"]),
    )
    assert result.official_five_class_id.tolist() == frame.class_id.tolist()
    assert result.official_five_class_name.tolist() == frame.class_name.tolist()
    assert result.target_manifest_row.tolist() == [0, 1, 2, 3]
    assert result.label_included.tolist() == [True, False, False, True]
    assert result.class_id.dropna().tolist() == [0, 2]
    assert str(result.class_id.dtype) == "Int64"
    report = summarize_fusar_mapping(result)
    assert report["original_count"] == 4
    assert report["eligible_count"] == report["mapped_count"] == 2
    assert report["class_counts"] == [1, 0, 1, 0, 0, 0, 0, 0]
    assert report["present_class_ids"] == [0, 2]
    assert report["eligibility_reasons"] == {"shared_main_mmsi": 1, "invalid_mmsi": 1}
    assert sum(row["count"] for row in report["inclusion_provenance"]) == 2
    assert sum(row["count"] for row in report["exclusion_provenance"]) == 2
    assert not report["quality_filter_applied"]
    assert not report["dimension_filter_applied"]
    json.dumps(report, allow_nan=False)


def test_missing_eligibility_is_rejected_and_empty_cohort_is_supported():
    frame = cohort([{}])
    with pytest.raises(ValueError, match="eligible"):
        map_fusar_labels(frame.drop(columns="eligible"))
    result = map_fusar_labels(frame.iloc[:0])
    report = summarize_fusar_mapping(result)
    assert report["original_count"] == report["eligible_count"] == report["mapped_count"] == 0
    assert report["class_counts"] == [0] * 8
    assert report["inclusion_provenance"] == report["exclusion_provenance"] == []
    json.dumps(report, allow_nan=False)


def test_prepared_fusar_cohort_counts_and_provenance():
    path = Path(
        "E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/"
        "bdh_fusar_preparation_20261003/experiment_cohort.csv"
    )
    if not path.exists():
        pytest.skip("Local prepared FUSAR cohort is unavailable")
    frame = pd.read_csv(path)
    mapped = map_fusar_labels(frame)
    report = summarize_fusar_mapping(mapped)
    assert len(mapped) == len(frame) == report["original_count"] == 3276
    assert report["eligible_count"] == 3166
    assert report["ineligible_count"] == 110
    assert report["eligibility_reasons"] == {"shared_main_mmsi": 96, "invalid_mmsi": 14}
    assert report["class_counts"] == [247, 731, 33, 28, 57, 8, 54, 0]
    assert report["mapped_count"] == report["included_count"] == 1158
    assert report["excluded_count"] == 2118
    assert report["eligible_unmapped_count"] == 2008
    assert report["present_class_ids"] == [0, 1, 2, 3, 4, 5, 6]
    assert report["absent_class_ids"] == [7]
    assert report["excluded_by_reason"]["coarse_fine_conflict"] == 9
    assert mapped.target_manifest_row.tolist() == list(range(len(frame)))
    assert not mapped.loc[~frame.eligible, "label_included"].any()
    assert sum(row["count"] for row in report["inclusion_provenance"]) == 1158
    assert sum(row["count"] for row in report["exclusion_provenance"]) == 2118
    json.dumps(report, allow_nan=False)
