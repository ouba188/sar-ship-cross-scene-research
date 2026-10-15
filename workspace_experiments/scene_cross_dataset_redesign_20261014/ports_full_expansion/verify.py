"""Verify isolation and paired support draws in the frozen 24-port episodes."""

from pathlib import Path

import numpy as np
import pandas as pd

from freeze import CLASSES, HERE, load_data


def read(folder, name):
    return pd.read_csv(folder / f"{name}.csv", dtype={"mmsi": str})


def main():
    data = load_data()
    folds = pd.read_csv(HERE / "fold_summary.csv")
    report = []
    query_ships_by_port = {}
    for fold in folds.itertuples():
        folder = HERE / fold.fold
        train = read(folder, "source_train")
        validation = read(folder, "source_validation")
        source_fit = read(folder, "source_fit")
        query = read(folder, "query")
        query_labels = read(folder, "query_labels")
        candidates = read(folder, "support_candidates")
        draws = read(folder, "support_draws")
        target = data[data.port.eq(fold.target_port)]
        target_acqs, target_ships = set(target.acquisition), set(target.mmsi)
        assert len(train) >= 20000 and len(validation) >= 1000
        assert set(train.class_id) == set(validation.class_id) == set(CLASSES)
        assert not set(train.acquisition) & set(validation.acquisition)
        assert not set(train.mmsi) & set(validation.mmsi)
        assert set(source_fit.row_index) == set(train.row_index) | set(validation.row_index)
        assert not set(source_fit.acquisition) & target_acqs
        assert not set(source_fit.mmsi) & target_ships
        assert set(source_fit.class_id) == set(CLASSES)
        assert len(query) == query.mmsi.nunique()
        assert set(query.acquisition) == {fold.query_acquisition}
        assert set(query.port) == {fold.target_port}
        assert set(query_labels.class_id) == set(CLASSES)
        assert set(query.row_index) == set(query_labels.row_index)
        assert not set(query.mmsi) & set(candidates.mmsi)
        for origin, acq in (("same", fold.query_acquisition), ("cross", fold.cross_acquisition)):
            pool = candidates[candidates.support_origin.eq(origin)]
            assert set(pool.acquisition) == {acq}
            assert set(pool.port) == {fold.target_port}
            assert set(pool.class_id) == set(CLASSES)
            for draw in range(5):
                support = draws[draws.draw.eq(draw) & draws.support_origin.eq(origin)]
                assert len(support) == 16 and support.mmsi.nunique() == 16
                assert support.class_id.value_counts().reindex(CLASSES, fill_value=0).eq(2).all()
                assert set(support.row_index).issubset(set(pool.row_index))
                assert not set(support.mmsi) & set(query.mmsi)
        with np.load(folder / "split.npz") as split:
            assert np.array_equal(split["query"], query.row_index.to_numpy())
            assert set(split["source_fit"]) == set(source_fit.row_index)
            assert split["support_same"].shape == split["support_cross"].shape == (5, 16)
            for draw in range(5):
                for key, origin in (("support_same", "same"), ("support_cross", "cross")):
                    expected = draws[draws.draw.eq(draw) & draws.support_origin.eq(origin)].row_index.to_numpy()
                    assert np.array_equal(split[key][draw], expected)
        query_ships_by_port.setdefault(fold.target_port, []).append(set(query.mmsi))
        report.append({"fold": fold.fold, "train": len(train), "validation": len(validation),
                       "fit": len(source_fit), "query": len(query), "classes": query_labels.class_id.nunique(),
                       "same_acquisition": fold.query_acquisition, "cross_acquisition": fold.cross_acquisition})
    for port, query_sets in query_ships_by_port.items():
        overlap = len(query_sets[0] & query_sets[1])
        assert overlap == 0
        print(f"{port} cross-fold repeated query ships: {overlap}")
    print(pd.DataFrame(report).to_string(index=False))
    print("Isolation and five paired B16 draws verified in all folds.")


if __name__ == "__main__":
    main()
