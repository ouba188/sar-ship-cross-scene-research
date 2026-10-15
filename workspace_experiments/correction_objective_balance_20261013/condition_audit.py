"""Describe frozen selector losses using source-fixed condition strata only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
REBUILD = ROOT / "correction_episode_rebuild_20261013"
STUDY = ROOT / "same_vessel_condition_study_20261010"
METADATA = ROOT / "ports_public_data_study_20261012/audit/metadata.csv"
INCIDENCE = Path("E:/Hermes/paper_tracking/pixel_correction_review/chip_incidence_v1/incidence_by_instance.csv")
ARM = "expanded__contextual"
CONDITIONS = ["contrast_proxy", "visible_footprint_proxy_fraction", "incidence_deg"]
CLASSES = [0, 5]
PORTS = ["Los Angeles", "Port Klang", "Santos"]


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(frame, name):
    frame.to_csv(OUT / ("condition_" + name + ".csv"), index=False, encoding="utf-8-sig")


def describe(group, scope, condition, threshold):
    values = group[condition]
    known = values.dropna()
    return dict(scope=scope, class_id=int(group.class_id.iloc[0]), condition=condition,
                n=len(group), ships=group.mmsi.nunique(), known_n=len(known),
                missing_n=int(values.isna().sum()), low_n=int((values <= threshold).sum()),
                high_n=int((values > threshold).sum()), source_median=threshold,
                low_ships=group.loc[values <= threshold, "mmsi"].nunique(),
                high_ships=group.loc[values > threshold, "mmsi"].nunique(),
                missing_ships=group.loc[values.isna(), "mmsi"].nunique(),
                minimum=known.min(), p10=known.quantile(.1), median=known.median(),
                p90=known.quantile(.9), maximum=known.max())


def summarize_cell(frame, mask, unit, cls, condition, level):
    cell = frame.loc[mask]
    n = len(cell)
    class_n = int((frame.class_id == cls).sum())
    lost = int(cell.rescue_lost.sum())
    avoided = int(cell.harm_avoided.sum())
    net = avoided - lost
    return dict(task=unit["task"], uid=unit["uid"], draw=int(unit["unit"]), class_id=cls,
                condition=condition, level=level, query_n=n, query_ships=cell.mmsi.nunique(),
                class_query_n=class_n, class_query_ships=frame.loc[frame.class_id == cls, "mmsi"].nunique(),
                class_mass_pct=100 * n / class_n, raw_recall=100 * cell.raw_correct.mean(),
                initial_recall=100 * cell.initial_correct.mean(), selector_recall=100 * cell.selector_correct.mean(),
                delta_recall_pp=100 * net / n if n else np.nan,
                class_recall_contribution_pp=100 * net / class_n,
                port_oa_contribution_pp=100 * net / len(frame),
                port_ba_contribution_pp=100 * net / class_n / frame.class_id.nunique(),
                original_rescue_n=int(cell.original_rescue.sum()), original_harm_n=int(cell.original_harm.sum()),
                rescue_kept_n=int(cell.rescue_kept.sum()), rescue_lost_n=lost,
                harm_avoided_n=avoided, harm_kept_n=int(cell.harm_kept.sum()), net_correct_n=net,
                rescue_lost_class_pp=100 * lost / class_n, harm_avoided_class_pp=100 * avoided / class_n,
                rescue_lost_rate_pct=100 * lost / cell.original_rescue.sum() if cell.original_rescue.sum() else np.nan,
                harm_avoided_rate_pct=100 * avoided / cell.original_harm.sum() if cell.original_harm.sum() else np.nan)


def main():
    units = [u for u in read_json(REBUILD / "PREDICTIONS_FROZEN.json")["units"] if u["family"] == "held_port"]
    assert len(units) == 15
    columns = ["instance_id", "port", "class_name", "class_id", "mmsi", "product", "image_relpath_vv", "image_relpath_vh"]
    manifest = pd.read_csv(units[0]["label_path"], usecols=columns, dtype={"mmsi": str})
    manifest["manifest_row"] = np.arange(len(manifest))
    metadata = pd.read_csv(METADATA, usecols=["dataset", "sample_id", "class_id", "actual_port", "product"] + CONDITIONS[:2])
    metadata = metadata.loc[metadata.dataset == "self24ports"].rename(columns={"sample_id": "instance_id", "class_id": "metadata_class_id", "actual_port": "metadata_port", "product": "metadata_product"})
    angle = pd.read_csv(INCIDENCE, usecols=["instance_id", "product", "port", "status", "incidence_deg"])
    angle.loc[angle.status != "ok", "incidence_deg"] = np.nan
    coverage = []

    def enrich(frame, source=False):
        result = frame.merge(metadata, on="instance_id", how="left", validate="one_to_one", indicator="metadata_join")
        assert result.metadata_join.eq("both").all()
        np.testing.assert_array_equal(result.class_id, result.metadata_class_id)
        np.testing.assert_array_equal(result.port, result.metadata_port)
        np.testing.assert_array_equal(result["product"], result.metadata_product)
        if source:
            result.loc[result.status != "ok", "incidence_deg"] = np.nan
            angle_matches = np.nan
        else:
            result = result.merge(angle, on=["instance_id", "product", "port"], how="left", validate="one_to_one", indicator="angle_join")
            angle_matches = int(result.angle_join.eq("both").sum())
            result = result.drop(columns="angle_join")
        coverage.append(dict(scope="sourceFIT" if source else frame.port.iloc[0], observations=len(frame),
                             metadata_matched_n=int(result.metadata_join.eq("both").sum()),
                             incidence_table_matched_n=angle_matches,
                             valid_incidence_n=int(result.incidence_deg.notna().sum())))
        return result.drop(columns=["metadata_class_id", "metadata_port", "metadata_product", "metadata_join"])

    obs = pd.read_csv(STUDY / "observations.csv", dtype={"mmsi": str})
    fit_ids = load(STUDY / "crossfit_evidence_20261011/source_folds.npz")["fit_obs_ids"]
    np.testing.assert_array_equal(fit_ids, read_json(REBUILD / "EPISODES_FROZEN.json")["fit_obs_ids"])
    source = obs.iloc[fit_ids].copy()
    assert len(source) == 1296 and source.instance_id.nunique() == 1296
    np.testing.assert_array_equal(manifest.iloc[source.manifest_row].instance_id, source.instance_id)
    source = enrich(source, source=True)
    thresholds = {(cls, key): source.loc[source.class_id == cls, key].median() for cls in CLASSES for key in CONDITIONS}
    source["scope"] = "sourceFIT"
    catalogs = [source.loc[source.class_id.isin(CLASSES)]]
    catalog_query_ids = {}
    for port in PORTS:
        unit = next(u for u in units if u["task"] == port)
        ids = load(unit["path"])["query_ids"]
        catalog_query_ids[port] = np.sort(ids)
        frame = enrich(manifest.iloc[ids].copy())
        frame["scope"] = port
        catalogs.append(frame.loc[frame.class_id.isin(CLASSES)])
    catalog = pd.concat(catalogs, ignore_index=True)
    for key in CONDITIONS:
        t = catalog.class_id.map({cls: thresholds[cls, key] for cls in CLASSES})
        catalog[key + "_bin"] = np.where(catalog[key].isna(), "missing", np.where(catalog[key] <= t, "low", "high"))
    catalog_columns = ["scope", "instance_id", "class_id", "class_name", "mmsi", "product", "port", "manifest_row", "status"] + CONDITIONS + [key + "_bin" for key in CONDITIONS] + ["image_relpath_vv", "image_relpath_vh"]
    save(catalog[catalog_columns], "observations")
    distribution = pd.DataFrame([describe(g, scope, key, thresholds[cls, key])
                                 for (scope, cls), g in catalog.groupby(["scope", "class_id"])
                                 for key in CONDITIONS])
    save(distribution, "distributions")
    save(distribution.loc[distribution.scope == "sourceFIT"], "source_thresholds")
    save(pd.DataFrame(coverage), "metadata_coverage")

    refs = pd.read_csv(REBUILD / "metrics.csv").set_index(["uid", "arm"])
    class_refs = pd.read_csv(REBUILD / "classes.csv").set_index(["uid", "arm", "class_id"])
    strata, details, checks = [], [], []
    for unit in units:
        z = load(unit["path"])
        d = load(REBUILD / "decisions" / (unit["uid"] + ".npz"))
        np.testing.assert_array_equal(z["query_ids"], d["query_ids"])
        np.testing.assert_array_equal(np.sort(z["query_ids"]), catalog_query_ids[unit["task"]])
        frame = manifest.iloc[z["query_ids"]].copy().reset_index(drop=True)
        y = frame.class_id.to_numpy()
        r, i = z["raw"].argmax(1), z["initial"].argmax(1)
        choose = d[ARM] > .5
        selected = np.where(choose, i, r)
        frame["raw_class"], frame["initial_class"], frame["selector_class"] = r, i, selected
        frame["raw_correct"], frame["initial_correct"], frame["selector_correct"] = r == y, i == y, selected == y
        frame["selector_probability"] = d[ARM]
        frame["choose_initial"] = choose
        frame["original_rescue"] = (i == y) & (r != y)
        frame["original_harm"] = (i != y) & (r == y)
        frame["rescue_lost"] = frame.original_rescue & ~choose
        frame["rescue_kept"] = frame.original_rescue & choose
        frame["harm_avoided"] = frame.original_harm & ~choose
        frame["harm_kept"] = frame.original_harm & choose
        frame["delta_correct"] = (selected == y).astype(int) - (i == y).astype(int)
        actual = 100 * frame.delta_correct.mean()
        np.testing.assert_allclose(actual, refs.loc[(unit["uid"], ARM), "versus_initial_oa"], atol=1e-10)
        checks.append(dict(uid=unit["uid"], check="port_oa", actual=actual, reference=refs.loc[(unit["uid"], ARM), "versus_initial_oa"]))
        target = catalog.loc[catalog.scope == unit["task"], ["instance_id"] + CONDITIONS + [key + "_bin" for key in CONDITIONS]]
        frame = frame.merge(target, on="instance_id", how="left", validate="one_to_one", sort=False)
        for cls in CLASSES:
            mask = frame.class_id == cls
            total = summarize_cell(frame, mask, unit, cls, "all", "all")
            strata.append(total)
            ref = class_refs.loc[(unit["uid"], ARM, cls)]
            np.testing.assert_allclose(total["selector_recall"], ref.recall, atol=1e-10)
            np.testing.assert_allclose(total["port_ba_contribution_pp"], ref.ba_contribution_vs_initial, atol=1e-10)
            checks.append(dict(uid=unit["uid"], check="class_" + str(cls) + "_ba", actual=total["port_ba_contribution_pp"], reference=ref.ba_contribution_vs_initial))
            for key in CONDITIONS:
                cells = [summarize_cell(frame, mask & frame[key + "_bin"].eq(level), unit, cls, key, level) for level in ("low", "high", "missing")]
                assert sum(c["query_n"] for c in cells) == total["query_n"]
                for field in ("net_correct_n", "rescue_lost_n", "harm_avoided_n"):
                    assert sum(c[field] for c in cells) == total[field]
                strata.extend(cells)
        part = frame.loc[frame.class_id.isin(CLASSES)].copy()
        part["uid"], part["draw"], part["task"] = unit["uid"], int(unit["unit"]), unit["task"]
        details.append(part)
    per_draw = pd.DataFrame(strata)
    save(per_draw, "draws")
    group = ["task", "class_id", "condition", "level"]
    numeric = per_draw.select_dtypes(include=np.number).columns.difference(["class_id", "draw"])
    summary = per_draw.groupby(group)[numeric].mean().reset_index()
    variation = per_draw.groupby(group).agg(draws=("draw", "nunique"), delta_sd=("delta_recall_pp", "std"), delta_min=("delta_recall_pp", "min"), delta_max=("delta_recall_pp", "max"), negative_draws=("delta_recall_pp", lambda x: int((x < -1e-10).sum()))).reset_index()
    summary = summary.merge(variation, on=group, validate="one_to_one")
    assert summary.draws.eq(5).all()
    save(summary, "summary")
    detail = pd.concat(details, ignore_index=True)
    save(detail, "sample_draws")
    sample_keys = ["task", "class_id", "instance_id", "mmsi", "product"]
    stability = detail.groupby(sample_keys, dropna=False).agg(draws=("draw", "nunique"), lost_draws=("rescue_lost", "sum"), avoided_draws=("harm_avoided", "sum"), original_rescue_draws=("original_rescue", "sum"), original_harm_draws=("original_harm", "sum"), mean_delta=("delta_correct", "mean"), mean_selector_probability=("selector_probability", "mean")).reset_index()
    assert len(stability) == len(detail[["task", "instance_id"]].drop_duplicates())
    assert stability.draws.eq(5).all()
    stable_metadata = detail.drop_duplicates(["task", "instance_id"])[["task", "instance_id", "manifest_row"] + CONDITIONS + [key + "_bin" for key in CONDITIONS] + ["image_relpath_vv", "image_relpath_vh"]]
    stability = stability.merge(stable_metadata, on=["task", "instance_id"], validate="one_to_one")
    stability["consistency"] = np.select([stability.lost_draws == 5, stability.avoided_draws == 5, (stability.lost_draws > 0) & (stability.avoided_draws == 0), (stability.avoided_draws > 0) & (stability.lost_draws == 0), (stability.lost_draws > 0) & (stability.avoided_draws > 0)], ["lost_5of5", "avoided_5of5", "loss_only", "gain_only", "mixed"], default="unchanged")
    save(stability.sort_values(["task", "class_id", "consistency", "instance_id"]), "sample_consistency")
    checks = pd.DataFrame(checks)
    checks["abs_error"] = (checks.actual - checks.reference).abs()
    save(checks, "verification")
    print("SOURCE", len(source), source.mmsi.nunique())
    print(pd.DataFrame(coverage).to_string(index=False))
    print("MAX_REFERENCE_ERROR", checks.abs_error.max())
    print(summary.loc[summary.condition == "all", ["task", "class_id", "query_n", "query_ships", "initial_recall", "delta_recall_pp", "rescue_lost_n", "harm_avoided_n", "negative_draws"]].to_string(index=False))


if __name__ == "__main__":
    main()
