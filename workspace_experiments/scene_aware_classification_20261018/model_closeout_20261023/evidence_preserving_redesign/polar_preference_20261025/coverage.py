"""Source-only endpoint coverage; no fitting or development-set statistics."""
import sys
sys.dont_write_bytecode = True

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "query_polar_gate_20261025"
HYBRID = HERE.parent / "hybrid_polar_fusion_20261025"
SEEDS = (17, 29, 43)
OUTCOMES = ("both_correct", "vv_only_correct", "vh_only_correct", "both_wrong")


def counts(frame):
    result = dict(n=len(frame), ships=frame.mmsi.nunique(),
                  passes=frame.pass_group.nunique(), ports=frame.port.nunique(),
                  classes=frame.class_id.nunique())
    for name in OUTCOMES:
        sub = frame[frame.outcome.eq(name)]
        result[name] = len(sub)
        result[name + "_ships"] = sub.mmsi.nunique()
        result[name + "_passes"] = sub.pass_group.nunique()
    result["conflict_n"] = result["vv_only_correct"] + result["vh_only_correct"]
    result["conflict_pct"] = 100 * result["conflict_n"] / len(frame)
    result["prediction_disagreement_n"] = int(frame.pred_vv.ne(frame.pred_vh).sum())
    for direction in ("VV", "VH", "tie"):
        result["ce_" + direction] = int(frame.ce_preference.eq(direction).sum())
    result["ce_gap_abs_le_1e_6"] = int(frame.ce_gap_vh_minus_vv.abs().le(1e-6).sum())
    result["ce_gap_mean"] = frame.ce_gap_vh_minus_vv.mean()
    result["ce_gap_macro_mean"] = frame.groupby("class_id").ce_gap_vh_minus_vv.mean().mean()
    result["ce_gap_half_micro_half_macro"] = .5 * (result["ce_gap_mean"] + result["ce_gap_macro_mean"])
    result["vv_OA_pct"] = 100 * frame.correct_vv.mean()
    result["vh_OA_pct"] = 100 * frame.correct_vh.mean()
    result["equal_OA_pct"] = 100 * frame.correct_equal.mean()
    result["base_OA_pct"] = 100 * frame.correct_base.mean()
    conflict = frame[frame.outcome.isin(("vv_only_correct", "vh_only_correct"))]
    result["conflict_ce_prefers_wrong_endpoint"] = int((
        (conflict.outcome.eq("vv_only_correct") & conflict.ce_preference.eq("VH")) |
        (conflict.outcome.eq("vh_only_correct") & conflict.ce_preference.eq("VV"))).sum())
    return result


def main():
    folds = pd.read_csv(PREVIOUS / "source_folds.csv", dtype={"mmsi": str})
    table = pd.read_csv(HERE.parent / "native_validation/table.csv", dtype={"mmsi": str})
    source = table.loc[table.role.eq("selection")].copy()
    assert len(source) == len(folds) == 612
    assert source.up_index.is_unique and folds.up_index.is_unique
    source = source.drop(columns="fold").merge(folds.rename(columns={"fold": "source_fold"}),
        on=["up_index", "obs_id", "mmsi", "pass_group", "class_id", "class_name"],
        how="inner", validate="one_to_one")
    assert len(source) == 612
    assert source.source_fold.value_counts().to_dict() == {"A": 432, "B": 180}
    for key in ("mmsi", "pass_group"):
        assert not set(source.loc[source.source_fold.eq("A"), key]) & set(
            source.loc[source.source_fold.eq("B"), key])

    samples, checks = [], []
    for seed in SEEDS:
        with np.load(PREVIOUS / "results" / f"features_{seed}.npz") as z:
            lookup = pd.Index(z["up_index"])
            assert lookup.is_unique
            ids = lookup.get_indexer(source.up_index)
            assert (ids >= 0).all()
            base, dv, dh = (z[k][ids] for k in ("base", "delta_vv", "delta_vh"))
            available = z["available"][ids].astype(bool)
        vv, vh = base + dv, base + dh
        equal = base + .5 * dv + .5 * dh
        replay = {}
        for arm, score in (("prior_vv", vv), ("center_vh", vh)):
            with np.load(HYBRID / "results" / f"{arm}_{seed}.npz") as z:
                ix = pd.Index(z["up_index"]).get_indexer(source.up_index)
                assert (ix >= 0).all()
                original = z["score"][ix]
            replay[arm + "_max_error"] = float(abs(original - score).max())
            assert replay[arm + "_max_error"] < 3e-6
            np.testing.assert_array_equal(original.argmax(1), score.argmax(1))
        assert np.isfinite(np.stack((base, vv, vh))).all()
        # Missing VH is a base-versus-VV comparison, not two observed views.
        assert np.all(dh[~available[:, 1]] == 0)
        frame = source.copy()
        frame["seed"] = seed
        frame["available_vv"], frame["available_vh"] = available.T
        frame["availability"] = np.select(
            (available.all(1), available[:, 0], available[:, 1]),
            ("both", "VV_only", "VH_only"), default="neither")
        y = frame.class_id.to_numpy(int)
        for name, score in (("base", base), ("vv", vv), ("vh", vh), ("equal", equal)):
            frame["pred_" + name] = score.argmax(1)
            frame["correct_" + name] = frame["pred_" + name].eq(y)
            score64 = score.astype(np.float64)
            frame["ce_" + name] = logsumexp(score64, axis=1) - score64[np.arange(len(y)), y]
        frame["ce_gap_vh_minus_vv"] = frame.ce_vh - frame.ce_vv
        frame["ce_preference"] = np.select(
            (frame.ce_gap_vh_minus_vv.gt(0), frame.ce_gap_vh_minus_vv.lt(0)),
            ("VV", "VH"), default="tie")
        frame["outcome"] = np.select(
            (frame.correct_vv & frame.correct_vh, frame.correct_vv, frame.correct_vh),
            OUTCOMES[:3], default="both_wrong")
        samples.append(frame)
        checks.append(dict(seed=seed, source_n=len(frame), **replay))
    samples = pd.concat(samples, ignore_index=True)

    summary, composition, overlap, ce_cross = [], [], [], []
    for seed, seed_frame in samples.groupby("seed"):
        for fold in ("ALL", "A", "B"):
            fold_frame = seed_frame if fold == "ALL" else seed_frame[seed_frame.source_fold.eq(fold)]
            for availability in ("ALL", *sorted(fold_frame.availability.unique())):
                frame = fold_frame if availability == "ALL" else fold_frame[fold_frame.availability.eq(availability)]
                context = dict(seed=int(seed), fold=fold, availability=availability)
                row = context | counts(frame)
                assert sum(row[k] for k in OUTCOMES) == row["n"]
                summary.append(row)
                for (outcome, preference), sub in frame.groupby(["outcome", "ce_preference"]):
                    ce_cross.append(context | dict(outcome=outcome, ce_preference=preference,
                        n=len(sub), ships=sub.mmsi.nunique(), passes=sub.pass_group.nunique(),
                        mean_ce_gap=sub.ce_gap_vh_minus_vv.mean()))
                for dimension in ("class_name", "mmsi", "pass_group", "port"):
                    groups = list(frame.groupby(dimension))
                    for value, sub in groups:
                        composition.append(context | dict(dimension=dimension, value=str(value)) | counts(sub))
                    for kind in ("top1", "ce"):
                        left = "vv_only_correct" if kind == "top1" else "VV"
                        right = "vh_only_correct" if kind == "top1" else "VH"
                        col = "outcome" if kind == "top1" else "ce_preference"
                        left_groups = {str(k) for k, g in groups if g[col].eq(left).any()}
                        right_groups = {str(k) for k, g in groups if g[col].eq(right).any()}
                        overlap.append(context | dict(dimension=dimension, kind=kind,
                            n_groups=len(groups), vv_groups=len(left_groups), vh_groups=len(right_groups),
                            bidirectional_groups=len(left_groups & right_groups),
                            bidirectional_values=json.dumps(sorted(left_groups & right_groups), ensure_ascii=False)))

    repeated = []
    for obs_id, group in samples.groupby("obs_id"):
        first = group.iloc[0]
        assert group.seed.nunique() == len(SEEDS)
        repeated.append(dict(obs_id=int(obs_id), up_index=int(first.up_index), fold=first.source_fold,
            mmsi=first.mmsi, pass_group=first.pass_group, class_name=first.class_name,
            availability=first.availability,
            vv_only_seeds=int(group.outcome.eq("vv_only_correct").sum()),
            vh_only_seeds=int(group.outcome.eq("vh_only_correct").sum()),
            ce_vv_seeds=int(group.ce_preference.eq("VV").sum()),
            ce_vh_seeds=int(group.ce_preference.eq("VH").sum()),
            top1_direction_reversal=bool(group.outcome.eq("vv_only_correct").any() and
                group.outcome.eq("vh_only_correct").any()),
            ce_direction_reversal=bool(group.ce_preference.eq("VV").any() and group.ce_preference.eq("VH").any())))

    frames = {"samples": samples, "summary": pd.DataFrame(summary),
              "composition": pd.DataFrame(composition), "group_overlap": pd.DataFrame(overlap),
              "ce_top1": pd.DataFrame(ce_cross), "cross_seed": pd.DataFrame(repeated)}
    for name, frame in frames.items():
        frame.to_csv(HERE / f"coverage_{name}.csv", index=False, encoding="utf-8-sig")
    receipt = dict(source_only=True, development_metrics_computed=False, fits=0,
        seeds=list(SEEDS), unique_source_observations=612, repeated_seed_rows=len(samples),
        fold_n={"A": 432, "B": 180}, source_folds=str(PREVIOUS / "source_folds.csv"),
        cache_pattern=str(PREVIOUS / "results" / "features_{seed}.npz"), checks=checks,
        ce_definition="CE = logsumexp(float64 cached endpoint logits) - true-class logit; strict sign, no chosen threshold",
        ce_gap_definition="CE_VH - CE_VV: positive favors VV; 1e-6 count is numerical context only",
        top1_definition="Exactly one of base+deltaVV and base+deltaVH predicts the true class",
        endpoint_definition="Old VV correction versus centered VH correction, on an already dual-polar base",
        limitation="Historical selection612 used upstream early stopping; grouped gate coverage is not clean full-pipeline OOF. Coverage and oracle possibility do not establish learnability or deployment benefit.",
        outputs=[f"coverage_{name}.csv" for name in frames])
    (HERE / "coverage_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["seed", "fold", "availability", "n", *OUTCOMES, "ce_VV", "ce_VH",
               "ce_gap_half_micro_half_macro", "conflict_ce_prefers_wrong_endpoint"]
    print(frames["summary"][columns].to_string(index=False))


if __name__ == "__main__":
    main()
