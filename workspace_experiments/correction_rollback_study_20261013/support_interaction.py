"""Read frozen caches: support interactions, without extraction or model fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
REBUILD = ROOT / "correction_episode_rebuild_20261013"
CONDITION = ROOT / "correction_objective_balance_20261013/condition_sample_draws.csv"
SOURCE_LABELS = ROOT / "same_vessel_condition_study_20261010/observations.csv"
ARM = "expanded__contextual"
SCOPE = ["family", "task", "cohort"]
IDENTITY = ["task", "class_id", "instance_id", "mmsi"]
METRICS = ["raw_support_count", "initial_support_count",
           "raw_oof_rescue", "initial_oof_rescue", "raw_oof_harm", "initial_oof_harm",
           "raw_oof_net", "initial_oof_net", "raw_profile_mean", "initial_profile_mean",
           "raw_margin", "initial_margin", "base_agrees_raw", "base_agrees_initial",
           "initial_zero_positive", "initial_zero_negative", "initial_zero_both",
           "selector_probability"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save(frame, name):
    frame.to_csv(OUT / ("support_" + name + ".csv"), index=False, encoding="utf-8-sig")


def labels(path):
    columns = pd.read_csv(path, nrows=0).columns
    wanted = [c for c in ("class_id", "instance_id", "sample_id", "mmsi",
                          "mmsi_or_ship_id", "port", "manifest_row") if c in columns]
    df = pd.read_csv(path, usecols=wanted, dtype={"mmsi": str, "mmsi_or_ship_id": str})
    if "instance_id" not in df:
        df["instance_id"] = df["sample_id"]
    if "mmsi" not in df:
        df["mmsi"] = df["mmsi_or_ship_id"]
    if "port" not in df:
        df["port"] = "not_in_manifest"
    df["label_row"] = np.arange(len(df))
    return df


def summarize_zero(events):
    rows = []
    # Port-class rows use true labels only for retrospective reporting.
    levels = [([], "scope"), (["class_id"], "class"),
              (["port", "class_id"], "port_class")]
    for extra, level in levels:
        for keys, cell in events.groupby(SCOPE + extra + ["status"], dropna=False):
            record = dict(zip(SCOPE + extra + ["status"], keys))
            record.update(level=level, exposures=len(cell),
                          observations=cell.instance_id.nunique(), ships=cell.mmsi.nunique(),
                          units=cell.uid.nunique())
            for role in ("raw", "initial"):
                for event in ("positive", "negative", "both"):
                    flag = role + "_zero_" + event
                    hit = cell[flag]
                    record[flag + "_n"] = int(hit.sum())
                    record[flag + "_exposure_pct"] = 100 * hit.mean()
                    per_obs = cell.groupby(["instance_id", "mmsi"], dropna=False)[flag].mean()
                    record[flag + "_obs_pct"] = 100 * per_obs.mean()
                    record[flag + "_ship_pct"] = 100 * per_obs.groupby("mmsi").mean().mean()
                    record[flag + "_observations"] = cell.loc[hit, "instance_id"].nunique()
                    record[flag + "_ships"] = cell.loc[hit, "mmsi"].nunique()
                for field in ("support_count", "oof_rescue", "oof_harm"):
                    record[role + "_" + field + "_mean"] = cell[role + "_" + field].mean()
            rows.append(record)
    return pd.DataFrame(rows)


def make_stability(focus):
    rows = []
    for keys, cell in focus.groupby(IDENTITY, dropna=False):
        cell = cell.sort_values("draw")
        rescue = cell.original_rescue
        kept, lost = cell.rescue_kept, cell.rescue_lost
        n_rescue, n_kept, n_lost = int(rescue.sum()), int(kept.sum()), int(lost.sum())
        assert len(cell) == cell.draw.nunique() == 5
        assert n_rescue == n_kept + n_lost
        status = ("no_rescue" if not n_rescue else "mixed_accept_rollback" if n_kept and n_lost
                  else "all_available_rescues_lost" if n_lost else "all_available_rescues_kept")
        r = dict(zip(IDENTITY, keys))
        r.update(draws=5, rescue_draws=n_rescue, kept_draws=n_kept, lost_draws=n_lost,
                 no_opportunity_draws=5-n_rescue, raw_already_correct_draws=int(cell.raw_correct.sum()),
                 both_wrong_draws=int((~cell.raw_correct & ~cell.initial_correct).sum()),
                 opportunity_varies=0 < n_rescue < 5,
                 choice_varies=cell.choose_initial.nunique() > 1,
                 choice_varies_among_rescues=bool(n_kept and n_lost), status=status,
                 choose_initial_draws=int(cell.choose_initial.sum()),
                 rescue_draw_ids="|".join(cell.loc[rescue, "draw"].astype(str)),
                 kept_draw_ids="|".join(cell.loc[kept, "draw"].astype(str)),
                 lost_draw_ids="|".join(cell.loc[lost, "draw"].astype(str)),
                 raw_class_by_draw="|".join(cell.raw_class.astype(str)),
                 initial_class_by_draw="|".join(cell.initial_class.astype(str)),
                 label_row=int(cell.label_row.iloc[0]))
        rows.append(r)
    obs = pd.DataFrame(rows)
    save(obs, "observation_stability")
    summaries = []
    for keys, g in list(obs.groupby(["task", "class_id"])) + [(('ALL_3_PORTS', -1), obs)]:
        row = dict(task=keys[0], class_id=keys[1], observations=len(g), ships=g.mmsi.nunique(),
                   unknown_ship_obs=int(g.mmsi.isna().sum()),
                   draw_exposures=int(g.draws.sum()), rescue_exposures=int(g.rescue_draws.sum()),
                   kept_exposures=int(g.kept_draws.sum()), lost_exposures=int(g.lost_draws.sum()))
        for name, mask in {
            "any_rescue": g.rescue_draws > 0, "any_loss": g.lost_draws > 0,
            "no_rescue": g.rescue_draws == 0, "opportunity_varies": g.opportunity_varies,
            "rescue_all_5": g.rescue_draws == 5,
            "choice_varies": g.choice_varies,
            "mixed_accept_rollback": g.choice_varies_among_rescues,
            "all_available_rescues_lost": g.status == "all_available_rescues_lost",
            "all_available_rescues_kept": g.status == "all_available_rescues_kept",
        }.items():
            row[name + "_obs"] = int(mask.sum())
            row[name + "_ships"] = g.loc[mask, "mmsi"].nunique()
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    save(summary, "stability_summary")
    hist = obs.groupby(["task", "class_id", "rescue_draws", "lost_draws", "kept_draws"]).agg(
        observations=("instance_id", "size"), ships=("mmsi", "nunique")).reset_index()
    save(hist, "frequency_histogram")
    return obs, summary


def paired_analysis(focus, obs):
    rows = []
    mixed = obs.loc[obs.choice_varies_among_rescues, IDENTITY]
    paired = focus.merge(mixed, on=IDENTITY, validate="many_to_one")
    for keys, g in paired.groupby(IDENTITY, dropna=False):
        accepted, rollback = g.loc[g.rescue_kept], g.loc[g.rescue_lost]
        for metric in METRICS:
            a, b = accepted[metric].mean(), rollback[metric].mean()
            row = dict(zip(IDENTITY, keys))
            row.update(metric=metric, accepted=a, rollback=b, delta=a-b,
                       accepted_draws=len(accepted), rollback_draws=len(rollback),
                       descriptive_drawpair_n=len(accepted)*len(rollback),
                       accepted_uids="|".join(accepted.uid), rollback_uids="|".join(rollback.uid),
                       raw_class_varies=g.loc[g.original_rescue, "raw_class"].nunique() > 1)
            rows.append(row)
    result = pd.DataFrame(rows)
    save(result, "pair_observations")
    ships = result.groupby(["task", "class_id", "mmsi", "metric"]).agg(
        observations=("instance_id", "nunique"), accepted=("accepted", "mean"),
        rollback=("rollback", "mean"), delta=("delta", "mean")).reset_index()
    save(ships, "pair_ships")
    out = []
    for keys, g in result.groupby(["task", "class_id", "metric"]):
        vessel = ships.loc[(ships.task == keys[0]) & (ships.class_id == keys[1]) & (ships.metric == keys[2])]
        out.append(dict(task=keys[0], class_id=keys[1], metric=keys[2],
                        observations=len(g), ships=g.mmsi.nunique(),
                        descriptive_drawpair_n=int(g.descriptive_drawpair_n.sum()),
                        obs_mean_accepted=g.accepted.mean(), obs_mean_rollback=g.rollback.mean(),
                        obs_mean_delta=g.delta.mean(), ship_mean_accepted=vessel.accepted.mean(),
                        ship_mean_rollback=vessel.rollback.mean(), ship_mean_delta=vessel.delta.mean(),
                        obs_raw_class_varies=int(g.raw_class_varies.sum())))
    summary = pd.DataFrame(out)
    save(summary, "pair_summary")
    return summary


def main():
    manifests = {}
    profile_rows, support_rows, event_frames, focus_frames = [], [], [], []
    denom_frames, checks, unit_rows, frequency_rows = [], [], [], []
    condition = pd.read_csv(CONDITION, dtype={"mmsi": str})
    eval_units = read_json(REBUILD / "PREDICTIONS_FROZEN.json")["units"]
    source_units = [dict(u, family="source_fit", task=u["context"], cohort="expanded",
                         label_path=str(SOURCE_LABELS), unit=u["key"])
                    for u in read_json(REBUILD / "SOURCE_READY.json")["units"] if u["kind"] == "expanded"]
    references = pd.read_csv(REBUILD / "metrics.csv").set_index(["uid", "arm"])
    for unit in source_units + eval_units:
        if unit["label_path"] not in manifests:
            manifests[unit["label_path"]] = labels(unit["label_path"])
        manifest = manifests[unit["label_path"]]
        z = load(unit["path"])
        support = manifest.iloc[z["support_ids"]].copy()
        sy, gain = support.class_id.to_numpy(int), z["profile_oof_gain"]
        assert len(support) == len(gain) == 16
        # A wrong iloc namespace would invalidate every role-based comparison.
        for name, mask in (("counts", np.ones(16, dtype=bool)), ("rescue", gain == 1), ("harm", gain == -1)):
            actual = np.bincount(sy[mask], minlength=8)
            np.testing.assert_array_equal(actual, z["profile_" + name])
        checks.append(dict(uid=unit["uid"], check="support_labels_counts_rescue_harm", value=16))
        identity = {key: unit[key] for key in SCOPE}
        unit_rows.append(dict(**identity, uid=unit["uid"], unit=unit["unit"],
                              input_path=unit["path"], label_path=unit["label_path"],
                              query_n=len(z["query_ids"]), support_n=16,
                              support_obs=support.instance_id.nunique(), support_ships=support.mmsi.nunique(),
                              oof_rescue_n=int((gain == 1).sum()), oof_harm_n=int((gain == -1).sum()),
                              support_rows="|".join(map(str, z["support_ids"]))))
        for pos, (_, s) in enumerate(support.iterrows()):
            support_rows.append(dict(**identity, uid=unit["uid"], support_position=pos,
                                     label_row=int(s.label_row), instance_id=s.instance_id,
                                     mmsi=s.mmsi, class_id=int(s.class_id), port=s.port,
                                     oof_gain=int(gain[pos])))
        for cls in range(8):
            profile_rows.append(dict(**identity, uid=unit["uid"], class_id=cls,
                                     count=int(z["profile_counts"][cls]),
                                     rescue=int(z["profile_rescue"][cls]), harm=int(z["profile_harm"][cls]),
                                     posterior_mean=float(z["profile_mean"][cls])))
        q = manifest.iloc[z["query_ids"]].copy().reset_index(drop=True)
        q["uid"] = unit["uid"]
        q["draw"] = unit["unit"]
        for key, value in identity.items():
            q[key] = value
        y = q.class_id.to_numpy(int)
        r, i = z["raw"].argmax(1), z["initial"].argmax(1)
        q["raw_class"], q["initial_class"] = r, i
        q["raw_correct"], q["initial_correct"] = r == y, i == y
        q["original_rescue"] = (i == y) & (r != y)
        q["original_harm"] = (r == y) & (i != y)
        for role, pred in (("raw", r), ("initial", i)):
            for output, field in (("support_count", "counts"), ("oof_rescue", "rescue"),
                                  ("oof_harm", "harm"), ("profile_mean", "mean")):
                q[role + "_" + output] = z["profile_" + field][pred]
            q[role + "_oof_net"] = q[role + "_oof_rescue"] - q[role + "_oof_harm"]
            q[role + "_zero_positive"] = q[role + "_oof_rescue"] == 0
            q[role + "_zero_negative"] = q[role + "_oof_harm"] == 0
            q[role + "_zero_both"] = q[role + "_zero_positive"] & q[role + "_zero_negative"]
        q["raw_margin"], q["initial_margin"] = z["x"][:, 2], z["x"][:, 3]
        q["base_agrees_raw"], q["base_agrees_initial"] = z["x"][:, 6], z["x"][:, 7]
        q["initial_count_gt_raw"] = q.initial_support_count > q.raw_support_count
        np.testing.assert_allclose(q.raw_support_count / 16, z["x"][:, 12], atol=0, rtol=0)
        np.testing.assert_allclose(q.initial_support_count / 16, z["x"][:, 13], atol=0, rtol=0)
        if unit["family"] == "source_fit":
            q["status"] = np.select([q.original_rescue, q.original_harm],
                                    ["original_rescue", "original_harm"], default="neutral")
        else:
            decision = load(REBUILD / "decisions" / (unit["uid"] + ".npz"))
            np.testing.assert_array_equal(decision["query_ids"], z["query_ids"])
            q["selector_probability"] = decision[ARM]
            q["choose_initial"] = decision[ARM] > .5
            q["rescue_kept"] = q.original_rescue & q.choose_initial
            q["rescue_lost"] = q.original_rescue & ~q.choose_initial
            q["harm_avoided"] = q.original_harm & ~q.choose_initial
            q["harm_kept"] = q.original_harm & q.choose_initial
            q["status"] = np.select([q.rescue_kept, q.rescue_lost, q.harm_avoided, q.harm_kept],
                                    ["rescue_kept", "rescue_lost", "harm_avoided", "harm_kept"], default="neutral")
            actual = 100 * (q.harm_avoided.sum() - q.rescue_lost.sum()) / len(q)
            reference = references.loc[(unit["uid"], ARM), "versus_initial_oa"]
            np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=0)
            checks.append(dict(uid=unit["uid"], check="published_delta_oa_abs_error", value=abs(actual-reference)))
        # Compute BA on the full query set, before filtering count groups or events.
        _, inverse, class_n = np.unique(y, return_inverse=True, return_counts=True)
        weights = {"micro": np.full(len(q), 1 / len(q)), "ba": 1 / (len(class_n) * class_n[inverse])}
        for greater in (False, True):
            mask = q.initial_count_gt_raw.to_numpy() == greater
            row = dict(**identity, uid=unit["uid"], initial_count_gt_raw=greater,
                       full_query_n=len(q), group_query_n=int(mask.sum()),
                       rescue_n=int((mask & q.original_rescue).sum()),
                       harm_n=int((mask & q.original_harm).sum()))
            for weighting, w in weights.items():
                row[weighting + "_query_mass_pp"] = 100 * w[mask].sum()
                row[weighting + "_rescue_mass_pp"] = 100 * w[mask & q.original_rescue].sum()
                row[weighting + "_harm_mass_pp"] = 100 * w[mask & q.original_harm].sum()
            frequency_rows.append(row)
        denom_frames.append(q[SCOPE + ["uid", "instance_id", "mmsi", "class_id", "port"]].copy())
        event_frames.append(q.loc[q.original_rescue | q.original_harm].copy())
        if unit["family"] == "held_port":
            focus = q.loc[q.class_id.isin([0, 5])].copy()
            focus["draw"] = focus.draw.astype(int)
            old = condition.loc[condition.uid == unit["uid"]].set_index("instance_id")
            old = old.loc[focus.instance_id]
            np.testing.assert_array_equal(focus.label_row, old.manifest_row)
            for field in ("class_id", "mmsi", "raw_class", "initial_class", "original_rescue", "rescue_lost", "rescue_kept"):
                np.testing.assert_array_equal(focus[field].fillna("MISSING"), old[field].fillna("MISSING"))
            np.testing.assert_allclose(focus.selector_probability, old.selector_probability, atol=1e-15, rtol=0)
            checks.append(dict(uid=unit["uid"], check="condition_rows_exact_replay", value=len(focus)))
            focus_frames.append(focus)
    units = pd.DataFrame(unit_rows)
    save(units, "units")
    save(pd.DataFrame(profile_rows), "unit_classes")
    profiles = pd.DataFrame(profile_rows)
    profiles["zero_support"] = profiles["count"] == 0
    profiles["zero_both"] = (profiles.rescue == 0) & (profiles.harm == 0)
    profiles["present_zero_both"] = (profiles["count"] > 0) & profiles.zero_both
    profiles["positive"] = profiles.rescue > 0
    profiles["negative"] = profiles.harm > 0
    class_profiles = profiles.groupby(SCOPE + ["class_id"]).agg(
        units=("uid", "size"), mean_count=("count", "mean"), zero_support_units=("zero_support", "sum"),
        zero_both_units=("zero_both", "sum"), present_zero_both_units=("present_zero_both", "sum"),
        positive_units=("positive", "sum"), negative_units=("negative", "sum")).reset_index()
    save(class_profiles, "oof_class_summary")
    support_table = pd.DataFrame(support_rows)
    save(support_table, "members")
    profile_summary = []
    for key, g in units.groupby(SCOPE):
        s = support_table.loc[support_table.uid.isin(g.uid)]
        profile_summary.append(dict(zip(SCOPE, key), units=len(g),
                                    distinct_support_row_sets=g.support_rows.nunique(),
                                    support_observations=s.instance_id.nunique(), support_ships=s.mmsi.nunique(),
                                    mean_oof_events=(g.oof_rescue_n + g.oof_harm_n).mean(),
                                    mean_oof_rescue=g.oof_rescue_n.mean(), mean_oof_harm=g.oof_harm_n.mean(),
                                    zero_event_units=int(((g.oof_rescue_n + g.oof_harm_n) == 0).sum()),
                                    max_oof_events=int((g.oof_rescue_n + g.oof_harm_n).max())))
    save(pd.DataFrame(profile_summary), "profile_summary")
    denominator = pd.concat(denom_frames, ignore_index=True)
    denominators = []
    for extra, level in (([], "scope"), (["port", "class_id"], "port_class")):
        result = denominator.groupby(SCOPE + extra).agg(
            query_exposures=("instance_id", "size"), observations=("instance_id", "nunique"),
            ships=("mmsi", "nunique"), units=("uid", "nunique")).reset_index()
        result["level"] = level
        denominators.append(result)
    save(pd.concat(denominators, ignore_index=True), "denominators")
    events = pd.concat(event_frames, ignore_index=True)
    save(events, "event_draws")
    zero = summarize_zero(events)
    save(zero, "zero_event_summary")
    frequency = pd.DataFrame(frequency_rows)
    save(frequency, "frequency_role_units")
    frequency_summaries = []
    for key, cell in frequency.groupby(SCOPE + ["initial_count_gt_raw"]):
        row = dict(zip(SCOPE + ["initial_count_gt_raw"], key))
        row.update(units=len(cell), query_exposures=int(cell.group_query_n.sum()),
                   rescue_exposures=int(cell.rescue_n.sum()), harm_exposures=int(cell.harm_n.sum()))
        for weighting in weights:
            query, rescue, harm = [cell[weighting + "_" + name + "_mass_pp"].mean()
                                   for name in ("query", "rescue", "harm")]
            row.update({weighting + "_" + name: value for name, value in {
                "query_mass_pp": query, "rescue_mass_pp": rescue, "harm_mass_pp": harm,
                "net_mass_pp": rescue-harm,
                "event_rescue_pct": 100 * rescue/(rescue+harm) if rescue+harm else np.nan,
                "group_rescue_pct": 100 * rescue/query if query else np.nan,
                "group_harm_pct": 100 * harm/query if query else np.nan}.items()})
        frequency_summaries.append(row)
    save(pd.DataFrame(frequency_summaries), "frequency_role_summary")
    focus = pd.concat(focus_frames, ignore_index=True)
    assert len(focus) == len(condition) == 10025
    assert focus.instance_id.nunique() == 2005
    save(focus, "sample_draws")
    obs, stability = make_stability(focus)
    assert len(obs) == 2005
    pairs = paired_analysis(focus, obs)
    draw_summary = focus.groupby(["task", "class_id", "uid", "draw"]).agg(
        query_n=("instance_id", "size"), query_ships=("mmsi", "nunique"),
        original_rescue=("original_rescue", "sum"), rescue_kept=("rescue_kept", "sum"),
        rescue_lost=("rescue_lost", "sum"), harm_avoided=("harm_avoided", "sum")).reset_index()
    save(draw_summary, "focus_draw_summary")
    save(pd.DataFrame(checks), "verification")
    print("STABILITY\n" + stability.to_string(index=False))
    print("PAIR SUMMARY\n" + pairs.to_string(index=False))
    print("PROFILE SUMMARY\n" + pd.DataFrame(profile_summary).to_string(index=False))
    print("ZERO FOCUS\n" + zero.loc[(zero.family == "held_port") & (zero.level == "class")
          & zero.class_id.isin([0, 5]) & zero.status.isin(["rescue_kept", "rescue_lost"]),
          ["task", "class_id", "status", "exposures", "observations", "ships",
           "initial_zero_positive_n", "initial_zero_negative_n", "initial_zero_both_n"]].to_string(index=False))
    print("CHECKS", len(checks), "EVAL_UNITS", len(eval_units), "SOURCE_UNITS", len(source_units))


if __name__ == "__main__":
    main()
