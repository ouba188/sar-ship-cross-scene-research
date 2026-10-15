import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import f0921_cohort as cohort


def product(start="20250102T122317", code="EB07"):
    return f"S1A_IW_GRDH_1SDV_{start}_20250102T122347_057265_070B6D_{code}"


def test_acquisition_stream_handles_records_crossing_read_boundaries(tmp_path):
    records = [
        {"tile": "clean", "sensor": "S1A", "t": "20250102T122317", "code": "EB07"},
    ] * 1800
    path = tmp_path / "acquisitions.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    assert list(cohort._iter_acquisitions(path)) == records


def test_join_uses_chip_key_and_preserves_archive_order_and_metadata():
    index = pd.DataFrame({
        "row": [40, 2, 9], "mmsi": ["001234567", "200", "900"],
        "class_raw": ["tug", "cargo_ship", "bulk_carrier"],
        "class8": ["old_tug", "old_general", "old_bulk"], "lon": [1., 2., 3.],
    })
    clean = cohort.join_clean_rows(index, np.array([9, 40]))
    assert clean.row.tolist() == [9, 40]
    assert clean.clean_feature_row.tolist() == [0, 1]
    assert clean.mmsi.tolist() == ["900", "001234567"]
    assert clean.class8.tolist() == ["old_bulk", "old_tug"]
    assert clean.lon.tolist() == [3., 1.]
    with pytest.raises(ValueError, match="missing from chip index"):
        cohort.join_clean_rows(index, np.array([77]))
    with pytest.raises(ValueError, match="not a one-to-one merge"):
        cohort.join_clean_rows(pd.concat([index, index.iloc[:1]]), np.array([40]))
    with pytest.raises(ValueError, match="not a one-to-one merge"):
        cohort.join_clean_rows(index, np.array([9, 9]))


@pytest.mark.parametrize("variant", ["aligned", "duplicate", "reordered", "ssl_mismatch"])
def test_only_row_members_are_read_and_alignment_is_required(monkeypatch, variant):
    row_arrays = [np.array([9, 40]), np.array([9, 40]), np.array([9, 40])]
    if variant == "duplicate":
        row_arrays[0] = np.array([9, 9])
    elif variant == "reordered":
        row_arrays[1] = np.array([40, 9])
    elif variant == "ssl_mismatch":
        row_arrays[2] = np.array([9, 41])
    calls = []

    class RowArchive:
        def __init__(self, rows):
            self.rows = rows

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __getitem__(self, key):
            assert key == "row", "Cohort preparation must not decompress feature arrays"
            return self.rows

    def load(path, **kwargs):
        calls.append(path)
        return RowArchive(row_arrays[len(calls) - 1])

    monkeypatch.setattr(cohort.np, "load", load)
    if variant == "aligned":
        np.testing.assert_array_equal(cohort.load_clean_rows(), [9, 40])
        assert calls == list(cohort.CLEAN_ARCHIVES)
    else:
        with pytest.raises(ValueError, match="Duplicate|alignment"):
            cohort.load_clean_rows()


@pytest.mark.parametrize("raw,expected", [
    ("self_discharging_bulk_carrier", 0), ("commercial_fishing_vessel", 1),
    ("general_cargo_ship", 2), ("chemical_oil_products_tanker", 3),
    ("chemical_tanker", 3), ("oil_products_tanker", 3), ("container_ship", 4),
    ("crude_oil_tanker", 5), ("pusher_tug", 6), ("towing_vessel", 6),
    ("offshore_support_vessel", 7), ("offshore_supply_vessel", 7),
    ("CargoShip", None), ("cargo_ship", None), ("tanker", None),
    ("lpg_tanker", None), ("lng_tanker", None), ("offshore_tug", None),
    ("drilling_ship", None), ("supply_tender", None), ("anchor_hoy", None),
    ("fish_factory_ship", None), ("fishing_support_vessel", None),
    ("heavy_load_carrier", None), ("cement_carrier", None), ("", None),
])
def test_bounded_aliases_do_not_inherit_class8(raw, expected):
    frame = pd.DataFrame({"class_raw": [raw], "class8": ["general_cargo"], "mmsi": ["007"]})
    result = cohort.map_f0921_labels(frame)
    assert result.class8.tolist() == ["general_cargo"]
    assert result.mmsi.tolist() == ["007"]
    assert result.label_field.tolist() == ["class_raw"]
    if expected is None:
        assert result.class_id.isna().all()
        assert not result.label_included.any()
    else:
        assert result.class_id.tolist() == [expected]
        assert result.class_name.tolist() == [cohort.CLASSES[expected]]


def test_product_candidates_include_nonfirst_products_without_day_only_matches():
    first, second = product(), product("20250102T130000", "ABCD")
    unrelated = product("20250102T140000", "ABCD")
    records = [
        {"tile": "clean", "sensor": "S1A", "t": "20250102T122317", "code": "EB07"},
        {"tile": "clean", "sensor": "S1A", "t": "20250102T130000", "code": "ABCD"},
        {"tile": "dirty", "sensor": "S1A", "t": "20250102T140000", "code": "ABCD"},
    ]
    shared, provenance = cohort.resolve_product_overlap(
        ["clean"], [first, second, unrelated], records, {"clean": {"product": first}},
    )
    assert set(shared) == {first, second}
    assert provenance["additional_products_beyond_first"] == [second]
    assert provenance["multi_product_clean_tiles"] == 1
    assert not provenance["chip_level_product_assignment"]
    with pytest.raises(ValueError, match="Missing acquisition provenance"):
        cohort.resolve_product_overlap(["absent"], [first], records, {})


def test_prepare_and_purge_use_all_clean_rows_before_label_filtering(tmp_path, monkeypatch):
    first, second, other = product(), product("20250102T130000", "ABCD"), product("20250103T130000", "AAAA")
    index = pd.DataFrame({
        "row": [30, 2, 9, 15], "scene": ["unmapped", "dirty", "mapped", "mapped"],
        "mmsi": ["001", "dirty-only", "002", ""],
        "class_raw": ["cargo_ship", "tug", "container_ship", "tug"],
        "class8": ["general_cargo", "tug_towing", "container_ship", "tug_towing"],
        "lon": [1., 2., 3., 4.], "lat": [5., 6., 7., 8.],
    })
    index_path = tmp_path / "index.csv"
    index.to_csv(index_path, index=False)
    monkeypatch.setattr(cohort, "INDEX", index_path)
    monkeypatch.setattr(cohort, "load_clean_rows", lambda: np.array([9, 30, 15]))
    records = [
        {"tile": "unmapped", "sensor": "S1A", "t": "20250102T122317", "code": "EB07"},
        {"tile": "unmapped", "sensor": "S1A", "t": "20250102T130000", "code": "ABCD"},
        {"tile": "mapped", "sensor": "S1A", "t": "20250104T120000", "code": "BBBB"},
    ]
    acq = tmp_path / "acq.json"
    acq.write_text(json.dumps(records))
    tile_map = tmp_path / "tile.json"
    tile_map.write_text(json.dumps({"unmapped": {"product": first}}))
    monkeypatch.setattr(cohort, "ACQUISITIONS", acq)
    monkeypatch.setattr(cohort, "TILE_PRODUCTS", tile_map)
    out = tmp_path / "out"
    target = cohort.prepare_target(out)
    assert target.row.tolist() == [9, 15]
    assert target.clean_feature_row.tolist() == [0, 2]
    assert target.target_manifest_row.tolist() == [9, 15]
    assert target.sample_id.tolist() == ["f0921_9", "f0921_15"]
    mapping = pd.read_csv(out / "label_mapping.csv", dtype={"mmsi": str}, keep_default_na=False)
    assert mapping.row.tolist() == [9, 30, 15]
    assert mapping.mmsi.tolist() == ["002", "001", ""]
    assert set(index.columns) <= set(target.columns)
    manifest = {
        "labels": np.array([0, 1, 2, 3, 4]), "ports": np.array(["a"] * 5),
        "mmsi": np.array(["001", "new", "001", "dirty-only", ""]),
        "product": np.array([other, second, first, other, ""]),
    }
    keep = cohort.prepare_source_split(out, manifest)
    np.testing.assert_array_equal(keep, [3, 4])
    summary = json.loads((out / "source_split.json").read_text())
    assert summary["target_clean_rows"] == 3
    assert summary["overlaps"]["mmsi"]["source_rows"] == 2
    assert summary["overlaps"]["product"]["source_rows"] == 2
    assert summary["overlaps"]["both_source_rows"] == 1
    assert summary["missing_source_classes"] == [cohort.CLASSES[i] for i in [0, 1, 2, 5, 6, 7]]
    with np.load(out / "source_split.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_rows"], keep)
        np.testing.assert_array_equal(saved["excluded_source_rows"], [0, 1, 2])
        assert json.loads(str(saved["metadata_json"])) == summary


def test_actual_cohort_counts_and_source_purge(tmp_path):
    source_path = Path("E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/"
                       "eight_class_adaptive_20260916/manifest.csv")
    if not cohort.INDEX.exists() or not source_path.exists():
        pytest.skip("Local F_0921 cohort and source manifest are required")
    target = cohort.prepare_target(tmp_path)
    summary = json.loads((tmp_path / "label_mapping_summary.json").read_text())
    assert (summary["index_rows"], summary["clean_rows"], summary["included"], summary["excluded"]) == (
        385650, 165880, 123489, 42391,
    )
    assert target.class_id.value_counts().sort_index().tolist() == [
        28880, 3800, 7564, 23229, 30638, 13947, 13357, 2074,
    ]
    land = pd.read_csv(cohort.ROOT / "f0921_land_stats.csv", usecols=["row", "land_px_224"])
    clean_rows = pd.read_csv(tmp_path / "label_mapping.csv", usecols=["row"])
    checked = clean_rows.merge(land, on="row", validate="one_to_one", how="left")
    assert checked.land_px_224.eq(0).all()
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    manifest = {"labels": source.class_id.to_numpy(dtype=int), "ports": source.port.to_numpy(),
                "mmsi": source.mmsi.to_numpy(), "product": source["product"].to_numpy()}
    retained = cohort.prepare_source_split(tmp_path, manifest)
    split = json.loads((tmp_path / "source_split.json").read_text())
    assert len(retained) == 29717
    assert split["excluded_source_rows"] == 8374
    assert split["overlaps"]["mmsi"]["source_rows"] == 7899
    assert split["overlaps"]["product"] == {"source_rows": 1839, "shared_ids": 146}
    assert split["overlaps"]["both_source_rows"] == 1364
    assert split["provenance"]["historical_first_product_shared"] == 89
    assert split["missing_source_classes"] == []
    assert list(split["source_class_counts"].values()) == [7237, 8237, 5019, 4755, 1525, 1006, 852, 1086]
