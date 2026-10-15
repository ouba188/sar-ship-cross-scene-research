"""Verify the four broad 24-port folds without fitting any model."""

import json

import numpy as np
import pandas as pd

from freeze import BUDGET, CLASSES, HERE, REQUESTED_PORTS, load_data


def read(folder, name):
    return pd.read_csv(folder / f"{name}.csv", dtype={"mmsi": str})


def main():
    data = load_data().set_index("row_index")
    summary = pd.read_csv(HERE / "fold_summary.csv")
    availability = pd.read_csv(HERE / "availability.csv")
    assert len(summary) == len(REQUESTED_PORTS) == 4
    assert set(summary.target_port) == set(REQUESTED_PORTS)
    assert set(availability.port) == set(REQUESTED_PORTS)
    assert availability.status.eq("frozen").all()
    folders = {path.parent.name for path in HERE.glob("*__*/split.npz")}
    assert folders == set(summary.fold), (folders, set(summary.fold))
    for fold in summary.itertuples():
        folder = HERE / fold.fold
        train = read(folder, "source_train")
        val = read(folder, "source_validation")
        fit = read(folder, "source_fit")
        query = read(folder, "query")
        query_labels = read(folder, "query_labels")
        pools = read(folder, "support_candidates")
        draws = read(folder, "support_draws")
        target = data[data.port.eq(fold.target_port)]
        for role, frame in (("train", train), ("validation", val), ("fit", fit),
                            ("query", query), ("support", pools)):
            assert frame.row_index.is_unique or role == "support"
            np.testing.assert_array_equal(data.loc[frame.row_index, "sample_id"].to_numpy(),
                                          frame.sample_id.to_numpy())
        assert len(train) >= 20000 and len(val) >= 1000
        assert set(train.class_id) == set(val.class_id) == set(CLASSES)
        assert set(fit.row_index) == set(train.row_index) | set(val.row_index)
        assert not set(train.acquisition) & set(val.acquisition)
        assert not set(train.mmsi) & set(val.mmsi)
        assert not set(fit.port) & {fold.target_port}
        assert not set(fit.acquisition) & set(target.acquisition)
        assert not set(fit.mmsi) & set(target.mmsi)
        assert len(query) >= 40 and query.mmsi.nunique() == len(query)
        assert query_labels.class_id.nunique() >= 3
        assert set(query.row_index) == set(query_labels.row_index)
        assert set(query.acquisition) == {fold.query_acquisition}
        assert set(query.port) == {fold.target_port}
        assert not set(query.mmsi) & set(pools.mmsi)
        quota = np.asarray(json.loads(fold.support_quota), dtype=int)
        assert len(quota) == 8 and quota.sum() == BUDGET
        assert set(np.flatnonzero(quota)).issubset(set(query_labels.class_id))
        assert (quota > 0).sum() >= 3
        for origin, acquisition in (("same", fold.query_acquisition),
                                    ("cross", fold.cross_acquisition)):
            pool = pools[pools.support_origin.eq(origin)]
            assert pool.mmsi.is_unique
            assert set(pool.port) == {fold.target_port}
            assert set(pool.acquisition) == {acquisition}
            for draw in range(5):
                support = draws[draws.draw.eq(draw) & draws.support_origin.eq(origin)]
                assert len(support) == BUDGET and support.mmsi.nunique() == BUDGET
                np.testing.assert_array_equal(
                    support.class_id.value_counts().reindex(CLASSES, fill_value=0).to_numpy(), quota)
                assert set(support.row_index).issubset(set(pool.row_index))
                assert not set(support.mmsi) & set(query.mmsi)
        with np.load(folder / "split.npz") as arrays:
            assert set(arrays.files) == {"ssl_train", "source_validation", "source_fit", "query",
                                         "support_same", "support_cross", "support_classes"}
            assert set(arrays["ssl_train"]) == set(train.row_index)
            assert set(arrays["source_validation"]) == set(val.row_index)
            assert set(arrays["source_fit"]) == set(fit.row_index)
            np.testing.assert_array_equal(arrays["query"], query.row_index.to_numpy())
            assert arrays["support_same"].shape == arrays["support_cross"].shape == (5, BUDGET)
            np.testing.assert_array_equal(np.bincount(arrays["support_classes"], minlength=8), quota)
            for draw in range(5):
                for key, origin in (("support_same", "same"), ("support_cross", "cross")):
                    expected = draws[draws.draw.eq(draw) & draws.support_origin.eq(origin)].row_index.to_numpy()
                    np.testing.assert_array_equal(arrays[key][draw], expected)
                    np.testing.assert_array_equal(data.loc[expected, "class_id"].to_numpy(),
                                                  arrays["support_classes"])
        composition = read(folder, "composition")
        for role, frame in (("source_train", train), ("source_validation", val), ("query", query)):
            assert composition.loc[composition.role.eq(role), "rows"].sum() == len(frame)
        print(f"{fold.target_port}: train={len(train)} val={len(val)} query={len(query)} "
              f"query_classes={query_labels.class_id.nunique()} support_classes={(quota > 0).sum()}")
    print("Four frozen folds verified; no obsolete fold under --root.")


if __name__ == "__main__":
    main()
