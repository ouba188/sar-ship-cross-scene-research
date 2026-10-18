"""Bounded paired review of saved ROI controls; no inference or fitting."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
CANDIDATE = "target_only"
COMPARATORS = ("roi_global", "roi_repeat_target", "fg_bg_global")
PINNED = (("20250315", 32, 0, 80), ("20250401", 32, 1, 236))
CAVEATS = (
    "All readouts pool existing feature maps; these are NOT separately cropped ROI inference. "
    "Masks are target/other candidates, not verified physical segmentation; rest is not verified background. "
    "Stored224 VV/VH are shown separately at 1:1 with no extra stretch; stored224 is a resized input, "
    "not native source spatial resolution. Record acquisition is NOT per-pixel product ownership; "
    "supports may share query acquisition. Source-mosaic replay establishes lineage, not independent AIS truth. "
    "Repeated queries and support draws are not independent evidence; selected cases are not population rates. "
    "Image appearance, masks and saved scores do not establish a physical mechanism."
)


def load_helpers():
    spec = importlib.util.spec_from_file_location("roi_parent_review", PARENT / "review_cases.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def category_for(own, base):
    correct, was_correct = own.prediction == own.truth, base.prediction == base.truth
    if correct != was_correct:
        return "rescue" if correct else "harm"
    suffix = "improve" if own.margin > base.margin else (
        "worsen" if own.margin < base.margin else "unchanged")
    return "persistent_" + ("correct_" if correct else "wrong_") + suffix


def select_cases(helpers, predictions, cohort, comparator, requested=()):
    events, counts = helpers.select_events(predictions, cohort, CANDIDATE, comparator)
    indexed = predictions.set_index(helpers.KEYS + ["arm"])
    own = predictions.loc[predictions.arm.eq(CANDIDATE)].set_index(helpers.KEYS)
    base = predictions.loc[predictions.arm.eq(comparator)].set_index(helpers.KEYS).loc[own.index]
    rules = {event[1]: helpers.RULE for event in events}

    # Preserve persistent evidence even when every corresponding margin is unchanged.
    for correct in (True, False):
        prefix = "persistent_" + ("correct" if correct else "wrong")
        unchanged = (own.prediction.eq(own.truth).eq(correct)
                     & base.prediction.eq(base.truth).eq(correct) & own.margin.eq(base.margin))
        for budget in sorted(own.index.get_level_values("budget").unique()):
            pool = own.loc[unchanged & (own.index.get_level_values("budget") == budget)]
            pool = pool.reset_index().sort_values(["fold", "row", "draw"], kind="stable")
            covered = any(category.startswith(prefix) and key[1] == budget
                          for category, key, _, _ in events)
            counts.append(dict(budget=int(budget), category=prefix + "_unchanged",
                               eligible_events=len(pool), fallback_selected=bool(len(pool) and not covered)))
            if pool.empty or covered:
                continue
            chosen = pool.iloc[(len(pool) - 1) // 2]
            key = (str(chosen.fold), int(chosen.budget), int(chosen.draw), int(chosen.row))
            events.append((prefix + "_unchanged", key,
                           indexed.loc[key + (CANDIDATE,)], indexed.loc[key + (comparator,)]))
            rules[key] = ("Unchanged persistent fallback where no changed persistent case of this correctness "
                          "was selected for this budget; lower median, ties fold,row,draw")

    for key in (*PINNED, *requested):
        if key not in rules:
            candidate, baseline = (indexed.loc[key + (arm,)] for arm in (CANDIDATE, comparator))
            events.append((category_for(candidate, baseline), key, candidate, baseline))
            rules[key] = "Explicit explanatory comparison; not lower-median selection or independent evidence"
    return events, counts, rules


def native_unavailable(helpers, item, enabled):
    if not enabled:
        return "disabled by --no-native-query"
    if not hasattr(helpers, "native_query"):
        return "parent helper has no native_query"
    try:
        from osgeo import gdal  # noqa: F401
    except (ImportError, OSError) as exc:
        return f"GDAL unavailable: {exc}"
    for field in ("source_tif", "left", "top", "side"):
        if field not in item or pd.isna(item[field]):
            return f"missing source-window field: {field}"
    if not Path(item.source_tif).is_file():
        return f"source mosaic unavailable: {item.source_tif}"
    return ""


def review_comparison(helpers, predictions, supports, cohort, regions, comparator, native, requested=()):
    events, counts, rules = select_cases(helpers, predictions, cohort, comparator, requested)
    prepared = [(category, key, own, base,
                 helpers.actual_panels(key, own, base, supports, cohort, CANDIDATE, comparator))
                for category, key, own, base in events]
    destination = HERE / "case_review" / f"{CANDIDATE}_vs_{comparator}"
    destination.mkdir(parents=True, exist_ok=True)
    rows, selected = [], []
    for category, key, own, base, panels in prepared:
        name = f"B{key[1]}_{category}_{key[0]}_draw{key[2]}_row{key[3]}"
        canvas, records = helpers.render_case(
            category, key, own, base, panels, cohort, regions, CANDIDATE, comparator,
            selection_rule=rules[key] + ". " + CAVEATS)
        figure = destination / f"{name}.png"
        canvas.save(figure)
        for record in records:
            record.update(figure=str(figure), candidate=CANDIDATE, comparator=comparator)
        rows.extend(records)
        actual = supports.loc[supports.fold.eq(key[0]) & supports.budget.eq(key[1])
                              & supports.draw.eq(key[2])]
        event = dict(category=category, fold=key[0], budget=key[1], draw=key[2], row=key[3],
                     delta_margin=float(own.margin - base.margin), selection_rule=rules[key],
                     pinned_prior_comparison=key in PINNED, figure=str(figure),
                     requested_explanatory_comparison=key in requested,
                     actual_support_rows=json.dumps(actual.support_row.astype(int).tolist()),
                     actual_support_class_counts=json.dumps(
                         {int(label): int(count) for label, count in actual.label.value_counts().items()}))
        for prefix, record in (("candidate", own), ("comparator", base)):
            event.update({f"{prefix}_{field}": value for field, value in record.items()})
        reason = native_unavailable(helpers, cohort.loc[key[3]], native)
        if reason:
            event["native_query_status"] = reason
        else:
            path = destination / f"{name}_native_query.png"
            # A replay mismatch must fail rather than silently present unrelated native pixels.
            event["native_display_limits"] = json.dumps(helpers.native_query(cohort.loc[key[3]], path))
            event["native_query_figure"] = str(path)
            event["native_query_status"] = "rendered; stored224 replay verified by parent helper"
        selected.append(event)
    pd.DataFrame(rows).to_csv(destination / "case_index.csv", index=False)
    pd.DataFrame(selected).to_csv(destination / "selected_events.csv", index=False)
    report = dict(
        candidate=CANDIDATE, comparator=comparator, selection_rule=helpers.RULE,
        unchanged_policy="One lower-median unchanged fallback per budget/correctness only when changed persistent cases are absent",
        selection_randomness="None; existing support draw IDs retained; no resampling or new seed",
        cases=len(events), counts=counts, pinned_prior_comparisons=[list(key) for key in PINNED],
        requested_explanatory_comparisons=[list(key) for key in requested],
        predictions=str(HERE / "predictions.csv"), supports=str(PARENT / "supports.csv"),
        cohort=str(PARENT / "cohort.csv"), masks=str(PARENT / "masks.npz"),
        episode_plan=str(PARENT / "episodes.json"), helper=str(PARENT / "review_cases.py"),
        native_query_requested=native, inference_or_fitting=False, limitations=CAVEATS,
        native_query_status_counts=pd.Series([event["native_query_status"] for event in selected]).value_counts().to_dict(),
        interpretation="Visual evidence package only; record visible facts separately from model evidence and untested mechanisms. No inherited contribution traces are attributed to these new comparisons.")
    (destination / "REVIEW.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(events)} paired cases to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparators", nargs="+", choices=COMPARATORS, default=COMPARATORS,
                        help="Saved comparators to review; default: all three")
    parser.add_argument("--case", action="append", default=[], metavar="FOLD,BUDGET,DRAW,ROW",
                        help="Include an additional explanatory event without replacing lower-median cases")
    parser.add_argument("--no-native-query", action="store_true",
                        help="Omit source-native query windows; otherwise render when GDAL/source are available")
    args = parser.parse_args()
    requested = []
    for value in args.case:
        fields = value.split(",")
        try:
            if len(fields) != 4:
                raise ValueError
            requested.append((fields[0], *(int(part) for part in fields[1:])))
        except ValueError:
            parser.error("--case requires FOLD,BUDGET,DRAW,ROW")
    for path in (HERE / "predictions.csv", PARENT / "cohort.csv", PARENT / "supports.csv", PARENT / "masks.npz"):
        if not path.is_file():
            parser.error(f"Required saved input missing: {path}. Run after controls results exist; no polling performed.")
    helpers = load_helpers()
    cohort = pd.read_csv(PARENT / "cohort.csv", dtype={"mmsi": str, "obs_time": str})
    helpers.require_columns(cohort, ["row", "label", "clean_row", "png_vv", "png_vh", "mmsi", "class_raw"], "cohort.csv")
    if len(cohort) != 269 or not np.array_equal(cohort.row.to_numpy(), np.arange(len(cohort))):
        raise ValueError("Expected parent 269 cohort in masks row order")
    cohort = cohort.set_index("row", drop=False)
    predictions = pd.read_csv(HERE / "predictions.csv", dtype={"fold": str})
    supports = pd.read_csv(PARENT / "supports.csv", dtype={"fold": str})
    helpers.require_columns(supports, ["fold", "budget", "draw", "support_row", "label"], "supports.csv")
    with np.load(PARENT / "masks.npz", allow_pickle=False) as saved:
        regions = saved["regions"]
    if regions.shape != (len(cohort), 3, 128, 128) or not np.isfinite(regions).all():
        raise ValueError("Expected finite parent masks regions[N,3,128,128]")
    for comparator in dict.fromkeys(args.comparators):
        review_comparison(helpers, predictions, supports, cohort, regions, comparator,
                          not args.no_native_query, requested)


if __name__ == "__main__":
    main()
