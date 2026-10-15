"""Check held-acquisition/public-support contracts for the new episodes."""

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def main():
    checked = 0
    for dataset in ("oss1", "oss2"):
        for folder in (HERE / dataset).iterdir():
            if not folder.is_dir():
                continue
            train = pd.read_csv(folder / "source_train.csv", dtype={"mmsi": str})
            val = pd.read_csv(folder / "source_validation.csv", dtype={"mmsi": str})
            query = pd.read_csv(folder / "query.csv", dtype={"mmsi": str})
            support = pd.read_csv(folder / "support_candidates.csv", dtype={"mmsi": str})
            draws = pd.read_csv(folder / "support_draws.csv", dtype={"mmsi": str})
            source = pd.concat((train, val), ignore_index=True)
            assert set(train.class_id) == set(val.class_id) == set(range(8))
            assert not set(train.acquisition) & set(val.acquisition)
            assert not set(train.mmsi) & set(val.mmsi)
            assert not set(source.acquisition) & (set(query.acquisition) | set(support.acquisition))
            assert not set(source.mmsi) & (set(query.mmsi) | set(support.mmsi))
            assert not set(query.mmsi) & set(support.mmsi)
            assert not set(query.sample_id) & set(support.sample_id)
            assert set(draws.draw) == set(range(5))
            assert set(draws.support_origin) == {"same", "cross"}
            assert draws.groupby(["draw", "support_origin"]).size().eq(16).all()
            assert draws.groupby(["draw", "support_origin"]).mmsi.nunique().eq(16).all()
            for draw in range(5):
                a = draws[(draws.draw == draw) & (draws.support_origin == "same")]
                b = draws[(draws.draw == draw) & (draws.support_origin == "cross")]
                assert a.class_id.value_counts().sort_index().equals(b.class_id.value_counts().sort_index())
            fit = folder / "fit"
            assert (fit / "training_complete.json").exists()
            result = pd.read_csv(fit / "pilot_metrics.csv")
            assert set(result.draw) == {-1, 0, 1, 2, 3, 4}
            assert result.groupby(["support_origin", "mode"]).size().loc["same"].eq(5).all()
            assert result.groupby(["support_origin", "mode"]).size().loc["cross"].eq(5).all()
            checked += 1
    assert checked == 6
    print(f"verified {checked} new public acquisition-held episodes")


if __name__ == "__main__":
    main()
