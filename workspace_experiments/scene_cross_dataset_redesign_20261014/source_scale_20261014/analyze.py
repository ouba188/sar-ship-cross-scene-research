"""Summarize the fixed-fold OSS2 source-size and training-step comparison."""

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def seed_means(frame):
    return (frame.groupby(["mode", "fraction", "steps", "support_origin", "seed"], as_index=False)
            [["oa", "ba", "source_val_ba"]].mean())


def paired(left, right, mode="G50_D18", origin="cross"):
    keys = ["seed", "support_origin", "mode"]
    a = left[(left["mode"] == mode) & (left.support_origin == origin)]
    b = right[(right["mode"] == mode) & (right.support_origin == origin)]
    merged = a.merge(b, on=keys, validate="one_to_one", suffixes=("_a", "_b"))
    assert len(merged) == 3
    return merged[["seed"]].assign(
        oa_delta=merged.oa_a - merged.oa_b,
        ba_delta=merged.ba_a - merged.ba_b,
        source_val_ba_delta=merged.source_val_ba_a - merged.source_val_ba_b,
    )


def select(frame, fraction, steps):
    return frame[(frame.fraction == fraction) & (frame.steps == steps)]


def main():
    primary = pd.read_csv(HERE / "results.csv")
    half_long = pd.read_csv(HERE / "half_long_results.csv")
    controls = pd.read_csv(HERE / "fixed_controls.csv")
    assert len(primary) == 330 and len(half_long) == 198 and len(controls) == 22
    assert set(primary.seed) == set(half_long.seed) == {17, 29, 43}
    for frame in (primary, half_long):
        counts = frame.groupby(["seed", "fraction", "steps", "support_origin", "mode"]).size()
        assert (counts[counts.index.get_level_values("support_origin") == "none"] == 1).all()
        assert (counts[counts.index.get_level_values("support_origin") != "none"] == 5).all()
    a = seed_means(primary)
    b = seed_means(half_long)
    summary = pd.concat((a.assign(run="primary"), b.assign(run="half_long")))
    summary.to_csv(HERE / "seed_means.csv", index=False)
    sections = [
        ("half_minus_full_70", a, .5, 70, a, 1., 70),
        ("full_280_minus_70", a, 1., 280, a, 1., 70),
        ("half_minus_full_280", b, .5, 280, a, 1., 280),
        ("half_replay_70_minus_initial", b, .5, 70, a, .5, 70),
    ]
    differences = []
    for name, left, left_fraction, left_step, right, right_fraction, right_step in sections:
        for origin in ("cross", "same", "none"):
            result = paired(select(left, left_fraction, left_step),
                            select(right, right_fraction, right_step), origin=origin)
            differences.append(result.assign(comparison=name, support_origin=origin))
    pd.concat(differences, ignore_index=True).to_csv(HERE / "paired_differences.csv", index=False)
    print("seed means:")
    print(a[a["mode"] == "G50_D18"].groupby(
        ["fraction", "steps", "support_origin"])[["oa", "ba", "source_val_ba"]]
          .agg(["mean", "std"]).round(2).to_string())
    print("half-long 280 means:")
    print(b[(b["mode"] == "G50_D18") & (b.steps == 280)].groupby(
        ["fraction", "support_origin"])[["oa", "ba", "source_val_ba"]]
          .agg(["mean", "std"]).round(2).to_string())


if __name__ == "__main__":
    main()
