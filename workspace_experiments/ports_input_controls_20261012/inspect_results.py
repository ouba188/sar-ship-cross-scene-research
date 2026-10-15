"""Compact, paired tables for the fixed-input pilot closeout."""

from context import *


def main():
    if not (OUT / "READOUT_COMPLETE.json").exists():
        raise RuntimeError("Finish the frozen readout first")
    tables = {}
    effects = pd.read_csv(OUT / "input_effects.csv")
    primary = effects[(effects.kind == "mlp") & (effects.scope == "original")]
    group = ["dataset", "condition", "head", "arm"]
    tables["input_primary"] = primary.groupby(group).agg(
        delta_ba=("delta_ba", "mean"), risk_gain=("risk_gain_macro", "mean"),
        ba_positive=("delta_ba", lambda x: int((x > 0).sum())),
        risk_positive=("risk_gain_macro", lambda x: int((x > 0).sum())),
        paired_context_draws=("delta_ba", "size")).reset_index()
    arms = pd.read_csv(OUT / "arm_effect_summary.csv")
    tables["cue_primary"] = arms[(arms.kind == "mlp") & (arms.scope == "original") &
        (arms["head"] == "expert") & arms.contrast.isin(["P-R", "PE-P", "T-R", "DD-T"])][
        ["dataset", "condition", "contrast", "delta_ba", "risk_gain_macro"]]
    assert len(tables["cue_primary"]) == 64
    features = pd.read_csv(OUT / "feature_diagnostics.csv")
    tables["feature_primary"] = features[features.kind == "mlp"].groupby(["dataset", "condition"])[
        ["target_r2", "input_squared_z", "source_scaled_mse"]].mean().reset_index()
    replay = pd.read_csv(OUT / "baseline_replay.csv")
    tables["baseline_replay_summary"] = replay.groupby(["dataset", "head", "arm"]).agg(
        mean_delta_ba=("delta_ba", "mean"), max_abs_delta_ba=("delta_ba", lambda x: x.abs().max()),
        mean_delta_brier=("delta_brier_macro", "mean"),
        max_abs_delta_brier=("delta_brier_macro", lambda x: x.abs().max()),
        max_probability_error=("max_probability_error", "max")).reset_index()
    classes = pd.read_csv(OUT / "classes.csv")
    classes = classes[(classes.kind == "mlp") & (classes.scope == "original")]
    keys = ["fold", "task", "dataset", "draw", "head", "arm", "class_id"]
    paired = classes[classes.condition != "original"].merge(classes[classes.condition == "original"],
        on=keys, suffixes=("_new", "_old"), validate="many_to_one")
    paired["delta_recall"] = paired.recall_new-paired.recall_old
    paired["risk_gain"] = paired.brier_old-paired.brier_new
    tables["class_input_effects"] = paired.groupby(["dataset", "condition_new", "head", "arm", "class_id"])[
        ["delta_recall", "risk_gain"]].mean().reset_index()
    tables["task_input_effects"] = primary.groupby(["fold", "task", "dataset", "condition", "head", "arm"])[
        ["delta_ba", "risk_gain_macro"]].mean().reset_index()
    clean = pd.read_csv(OUT / "proxy_retained_no_support_risk_summary.csv")
    original = pd.read_csv(OUT / "input_effect_summary.csv")
    original = original[original.scope == "original"]
    sensitivity_keys = ["condition", "dataset", "kind", "head", "arm"]
    tables["geometry_sensitivity"] = original.merge(clean, on=sensitivity_keys, suffixes=("_all", "_clean"),
        validate="one_to_one")[sensitivity_keys+["delta_ba_all", "delta_ba_clean", "risk_gain_macro_all", "risk_gain_macro_clean"]]
    # This detects a redirected source fit, not merely a matching experiment label.
    contexts = 0
    for fold in FOLDS:
        fits = PREVIOUS / f"fold_{fold}/fits"
        for name in tasks(fold):
            for condition in CONDITIONS:
                receipt = OUT / "predictions" / condition / f"fold_{fold}/targets/shared_frozen" / name / "COMPLETE.json"
                value = json.loads(receipt.read_text(encoding="utf-8"))
                assert Path(value["shared_geometry_path"]) == fits / "geometry.pkl"
                assert Path(value["mapper_path"]) == fits / "mapper.npz"
                contexts += 1
    for name, table in tables.items():
        table.to_csv(OUT / (name+".csv"), index=False)
    print("INPUTS", tables["input_primary"].query("head == 'c4_late' and arm in ['R', 'T', 'DD']").to_string(index=False), sep="\n")
    print("CUES", tables["cue_primary"].to_string(index=False), sep="\n")
    print("FEATURES", tables["feature_primary"].to_string(index=False), sep="\n")
    print("REPLAY", tables["baseline_replay_summary"].query("arm in ['R', 'T', 'DD']").to_string(index=False), sep="\n")
    write(OUT / "INTEGRATION_CHECK.json", dict(contexts=contexts, original_source_fit_paths=True,
        no_model_promotion=True, reported_table_count=len(tables)))


if __name__ == "__main__":
    main()
