"""Prespecified paired contrasts and cohort/dose accounting, without new fits."""

from region_context import OUT, write
import numpy as np
import pandas as pd


def main():
    paths = pd.read_csv(OUT / "path_effects.csv")
    utility = pd.read_csv(OUT / "utility.csv")
    contrasts = []
    for scope, table in paths.groupby("scope"):
        for path, group in table.groupby("path"):
            values = group.pivot(index=["fold", "draw"], columns="condition", values="delta_ba")
            for a, b in (("target_shuffle", "outer_shuffle"), ("outer_display", "target_display"),
                         ("target_flat", "sham_flat"), ("wide_target_flat", "target_flat")):
                d = values[a]-values[b]
                contrasts.append(dict(scope=scope, path=path, contrast=a+"-"+b,
                    mean=float(d.mean()), minimum=float(d.min()), maximum=float(d.max()),
                    positive=int((d > 0).sum()), negative=int((d < 0).sum()), contexts=len(d)))
    pd.DataFrame(contrasts).to_csv(OUT / "paired_contrasts.csv", index=False)
    joint = utility[utility['mode'].eq("joint")].copy()
    joint["net_change"] = joint.net-joint.original_net
    joint.groupby(["scope", "condition"]).agg(mean=("net_change", "mean"),
        minimum=("net_change", "min"), maximum=("net_change", "max"),
        positive=("net_change", lambda a: int((a > 1e-10).sum())),
        negative=("net_change", lambda a: int((a < -1e-10).sum()))).reset_index().to_csv(
            OUT / "joint_net_change.csv", index=False)
    frame = pd.read_csv(OUT / "cohort.csv", keep_default_na=False)
    region = pd.read_csv(OUT / "region_metadata.csv")
    valid = region.region_valid.to_numpy(bool)
    with np.load(OUT / "cohort.npz") as z:
        support = z["support"]
        masks = z["eval_masks"]
    doses = []
    for folder in sorted((OUT / "features").iterdir()):
        f = pd.read_csv(folder / "controls.csv")
        for scope, selected in (("valid", valid), ("sham_valid", region.sham_valid.to_numpy(bool))):
            a = f[selected]
            doses.append(dict(condition=folder.name, scope=scope, rows=len(a),
                area_mean=a.edited_area.mean(), area_median=a.edited_area.median(),
                rms_mean=a.change_rms.mean(), changed_fraction_mean=a.changed_fraction.mean()))
    pd.DataFrame(doses).to_csv(OUT / "dose_summary.csv", index=False)
    coverage = dict(manifest_rows=len(frame), valid_rows=int(valid.sum()),
        invalid_samples=frame.sample_id[~valid].tolist(),
        identities=int(frame.loc[valid, "identity"].nunique()),
        datatakes=int(frame.loc[valid, "datatake_key"].nunique()),
        products=int(frame.loc[valid, "scene_id"].nunique()),
        class_counts=frame.loc[valid, "class_name"].value_counts().to_dict(),
        sham_valid=int(region.sham_valid.sum()),
        valid_queries_per_draw=(masks & valid).sum(1).tolist(),
        common_valid_queries=int((masks.all(0) & valid).sum()),
        invalid_support_counts=[int((~valid)[s].sum()) for s in support],
        target_area_median=float(region.loc[valid, "region_area"].median()),
        target_area_mean=float(region.loc[valid, "region_area"].mean()),
        full_area_regions=int(np.sum(region.region_area.eq(1))),
        target_and_sham_area_equal=bool(np.array_equal(
            pd.read_csv(OUT / "features/target_flat/controls.csv").loc[region.sham_valid, "edited_area"],
            pd.read_csv(OUT / "features/sham_flat/controls.csv").loc[region.sham_valid, "edited_area"])))
    write("COVERAGE.json", coverage)
    print(pd.DataFrame(contrasts).query("scope == 'full' and path in ['T_Aonly','T_Bonly']").to_string(index=False))
    print(joint.groupby(["scope", "condition"])["net_change"].mean().loc["full"].to_string())


if __name__ == "__main__":
    main()
