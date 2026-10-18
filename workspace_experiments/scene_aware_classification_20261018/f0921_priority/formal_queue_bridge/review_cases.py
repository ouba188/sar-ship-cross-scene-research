"""Review saved final-system outcomes and actual supports, without inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
KEYS = ["fold", "budget", "draw", "row"]
CATEGORIES = ("rescue", "harm", "persistent_correct_improve",
              "persistent_correct_worsen", "persistent_wrong_improve",
              "persistent_wrong_worsen")
SUPPORT_FIELDS = ("top_same_row", "top_wrong_row", "top_predicted_row")
PROBS = [f"p{i}" for i in range(7)]
RULE = "Lower median of signed candidate-minus-comparator final margin; ties fold,row,draw; per budget/category"
COLORS = ((230, 65, 75), (20, 175, 210))


def require_columns(frame, names, filename):
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"{filename}: missing columns {sorted(missing)}")


def select_events(predictions, cohort, candidate, comparator):
    require_columns(predictions, KEYS + ["arm", "truth", "prediction", "margin",
                    "ap", "brier"] + PROBS + list(SUPPORT_FIELDS), "predictions.csv")
    parts = []
    for arm in (candidate, comparator):
        part = predictions.loc[predictions.arm.eq(arm)].copy()
        if part.empty or part.duplicated(KEYS).any():
            raise ValueError(f"Missing or duplicate paired events for {arm}")
        parts.append(part.set_index(KEYS).sort_index())
    own, base = parts
    if not own.index.equals(base.index) or not own.truth.equals(base.truth):
        raise ValueError("Candidate/comparator event keys or truth differ")
    rows = own.index.get_level_values("row").to_numpy()
    if not np.isin(rows, cohort.index).all():
        raise ValueError("Prediction row absent from cohort")
    if not np.array_equal(own.truth.to_numpy(), cohort.loc[rows, "label"].to_numpy()):
        raise ValueError("Prediction truth differs from cohort label")
    for arm, frame in zip((candidate, comparator), parts):
        if not np.isfinite(frame[["margin", "brier"] + PROBS].to_numpy(float)).all():
            raise ValueError(f"Nonfinite saved outcomes for {arm}")
        if np.isinf(frame.ap.to_numpy(float)).any():
            raise ValueError(f"Infinite AP for {arm}")
        if not frame.prediction.isin(range(7)).all():
            raise ValueError(f"Invalid class prediction for {arm}")
    change = own.margin - base.margin
    correct, was_correct = own.prediction.eq(own.truth), base.prediction.eq(base.truth)
    masks = (correct & ~was_correct, ~correct & was_correct,
             correct & was_correct & change.gt(0), correct & was_correct & change.lt(0),
             ~correct & ~was_correct & change.gt(0), ~correct & ~was_correct & change.lt(0))
    paired = pd.DataFrame({"delta_margin": change})
    events, counts = [], []
    for category, mask in zip(CATEGORIES, masks):
        for budget in sorted(set(own.index.get_level_values("budget"))):
            pool = paired.loc[mask & (paired.index.get_level_values("budget") == budget)]
            pool = pool.reset_index().sort_values(["delta_margin", "fold", "row", "draw"], kind="stable")
            counts.append(dict(budget=int(budget), category=category, eligible_events=len(pool)))
            if pool.empty:
                continue
            chosen = pool.iloc[(len(pool) - 1) // 2]
            key = (str(chosen.fold), int(chosen.budget), int(chosen.draw), int(chosen.row))
            events.append((category, key, own.loc[key], base.loc[key]))
    return events, counts


def actual_panels(key, own, base, supports, cohort, candidate, comparator):
    selected = supports.loc[supports.fold.eq(key[0]) & supports.budget.eq(key[1])
                            & supports.draw.eq(key[2])]
    if len(selected) != key[1] or selected.support_row.duplicated().any():
        raise ValueError(f"Missing, duplicated or wrong-size actual support draw: {key[:3]}")
    support_rows = selected.support_row.to_numpy()
    if not np.isin(support_rows, cohort.index).all():
        raise ValueError("Actual support row absent from cohort")
    if not np.array_equal(cohort.loc[support_rows, "label"].to_numpy(), selected.label.to_numpy()):
        raise ValueError("Actual support labels differ from cohort")
    if key[3] in support_rows or cohort.loc[key[3], "mmsi"] in set(cohort.loc[support_rows, "mmsi"]):
        raise ValueError("Query ship enters its actual support draw")
    panels = [dict(role="QUERY", arm="", row=key[3], status="query")]
    for tag, arm, record in (("CAND", candidate, own), ("COMP", comparator, base)):
        for field, role in zip(SUPPORT_FIELDS, ("same class", "wrong class", "predicted class")):
            if field == "top_wrong_row":
                eligible = selected.loc[selected.label.ne(record.truth), "support_row"]
            else:
                label = record.prediction if field == "top_predicted_row" else record.truth
                eligible = selected.loc[selected.label.eq(label), "support_row"]
            value = record[field]
            if pd.isna(value) or value == -1:
                if len(eligible):
                    raise ValueError(f"{arm}/{field} missing despite eligible actual supports: {key}")
                panels.append(dict(role=f"{tag} {role}", arm=arm, row=None,
                                   status=f"NO {role.upper()} SUPPORT IN THIS DRAW"))
                continue
            if not np.isfinite(float(value)) or float(value) != int(value) or int(value) not in set(eligible):
                raise ValueError(f"{arm}/{field}={value} is not an eligible actual support: {key}")
            panels.append(dict(role=f"{tag} {role}", arm=arm, row=int(value), status="saved_ranked_support"))
    return panels


def font():
    try:
        return ImageFont.truetype("arial.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def wrapped(draw, text, xy, width, face):
    x, y = xy
    line = ""
    for word in str(text).split():
        trial = f"{line} {word}".strip()
        if line and draw.textbbox((0, 0), trial, font=face)[2] > width:
            draw.text((x, y), line, fill="black", font=face)
            line, y = word, y + 19
        else:
            line = trial
    draw.text((x, y), line, fill="black", font=face)
    return y + 19


def stored_pair(item):
    images = []
    for field in ("png_vv", "png_vh"):
        with Image.open(item[field]) as source:
            pixels = np.array(source)
        if pixels.shape != (224, 224) or pixels.dtype != np.uint8:
            raise ValueError(f"Expected stored224 grayscale uint8: {item[field]}")
        images.append(Image.fromarray(pixels).convert("RGB"))
    return images


def overlay(image, regions):
    result = image.copy()
    for mask, color in zip(regions[:2], COLORS):
        alpha = Image.fromarray(np.rint(np.clip(mask, 0, 1) * 95).astype(np.uint8))
        alpha = alpha.resize(image.size, Image.Resampling.NEAREST)
        result.paste(Image.new("RGB", image.size, color), (0, 0), alpha)
    return result


def acquisition(item):
    value = item.get("acquisition", "")
    return "unknown" if pd.isna(value) or not str(value).strip() else str(value)


def contribution_panels(key, own, supports, cohort, decisions, contributions):
    selected = decisions
    terms = contributions
    for field, value in zip(KEYS, key):
        selected = selected.loc[selected[field].eq(value)]
        terms = terms.loc[terms[field].eq(value)]
    if selected.empty:
        return [], {}
    if len(selected) != 1 or len(terms) != key[1] or terms.support_row.duplicated().any():
        raise ValueError(f"Incomplete saved contribution trace: {key}")
    decision = selected.iloc[0]
    if int(decision.true) != int(own.truth) or int(decision.candidate_prediction) != int(own.prediction):
        raise ValueError(f"Contribution trace does not match candidate: {key}")
    actual = supports.loc[supports.fold.eq(key[0]) & supports.budget.eq(key[1])
                          & supports.draw.eq(key[2]), "support_row"]
    if set(terms.support_row) != set(actual):
        raise ValueError(f"Contribution trace does not use this actual support draw: {key}")
    if not np.isclose(terms.total_term.sum(), decision.r50_margin_change, atol=1e-8):
        raise ValueError(f"Contribution sum differs from saved fixed-rival change: {key}")
    panels = [dict(role="QUERY", arm="", row=key[3], status="query")]
    chosen = [(int(decision[field]), role) for field, role in (
        ("strongest_helping_support", "Strongest helping support"),
        ("strongest_harming_support", "Strongest harming support"))]
    for same, role in ((True, "Helping same-class support"), (False, "Helping other-class support")):
        group = terms.loc[terms.support_label.eq(own.truth).eq(same) & terms.total_term.gt(0)]
        if not group.empty:
            row = int(group.sort_values(["total_term", "support_row"], ascending=[False, True]).iloc[0].support_row)
            if row not in [r for r, _ in chosen]:
                chosen.append((row, role))
    for row, role in chosen:
        term = terms.set_index("support_row").loc[row]
        if int(term.support_label) != int(cohort.loc[row, "label"]):
            raise ValueError(f"Contribution support label mismatch: {key}/{row}")
        panels.append(dict(role=role, arm="target_only", row=row, status="saved_fixed_rival_contribution",
                           kernel_term=float(term.kernel_term), refit_term=float(term.refit_term),
                           total_term=float(term.total_term), old_similarity=float(term.old_similarity),
                           new_similarity=float(term.new_similarity)))
    result = decision.to_dict()
    result.update(same_class_terms_sum=float(terms.loc[terms.support_label.eq(own.truth), "total_term"].sum()),
                  other_class_terms_sum=float(terms.loc[terms.support_label.ne(own.truth), "total_term"].sum()))
    return panels, result


def render_case(category, key, own, base, panels, cohort, regions, candidate, comparator, selection_rule=RULE):
    column, face = 270, font()
    canvas = Image.new("RGB", (column * len(panels), 1900), "white")
    draw = ImageDraw.Draw(canvas)
    lines = [f"{category} | fold {key[0]} B{key[1]} draw {key[2]} query row {key[3]}",
             f"CAND {candidate}: pred={int(own.prediction)} margin={own.margin:+.6f}; "
             f"COMP {comparator}: pred={int(base.prediction)} margin={base.margin:+.6f}; truth={int(own.truth)}",
             f"Final margin change={own.margin-base.margin:+.6f}; AP {own.ap:.5f}/{base.ap:.5f}; "
             f"Brier {own.brier:.5f}/{base.brier:.5f} (candidate/comparator)",
             selection_rule,
             "Stored224 VV/VH: 1:1, no extra stretch. Overlays: red target candidates, cyan other candidates; rest unchanged.",
             "Saved ranked supports from this exact draw. Wrong-class support need not belong to the predicted class. AP=nan means undefined retrieval.",
             "Source-held-date few-shot: target supports may share an acquisition. Record acquisition is NOT per-pixel product ownership.",
             "Pixel lineage is to the source mosaic, not independent AIS truth. Masks are candidates, not physical segmentation. Cases are not population rates."]
    y = 8
    for line in lines:
        y = wrapped(draw, line, (10, y), canvas.width - 20, face)
    title_top = max(170, y + 12)
    pixel_top = title_top + 260
    records = []
    for j, panel in enumerate(panels):
        x = j * column + 12
        y = wrapped(draw, panel["role"], (x, title_top), column - 24, face)
        row = panel["row"]
        record = dict(panel, category=category, fold=key[0], budget=key[1], draw=key[2], query_row=key[3])
        if row is None:
            wrapped(draw, panel["status"], (x, y + 25), column - 24, face)
        else:
            item = cohort.loc[row]
            y = wrapped(draw, f"row {row} | clean {item.clean_row} | class {item.label}", (x, y), column - 24, face)
            y = wrapped(draw, item.class_raw.replace("_", " "), (x, y), column - 24, face)
            y = wrapped(draw, f"MMSI {item.mmsi}", (x, y), column - 24, face)
            current_acq, query_acq = acquisition(item), acquisition(cohort.loc[key[3]])
            relation = "unknown" if "unknown" in (current_acq, query_acq) else (
                "same" if current_acq == query_acq else "different")
            y = wrapped(draw, f"Record acq: {current_acq}", (x, y), column - 24, face)
            y = wrapped(draw, f"Record acq vs query: {relation}", (x, y), column - 24, face)
            if "total_term" in panel:
                y = wrapped(draw, f"Similarity {panel['old_similarity']:+.6f} -> {panel['new_similarity']:+.6f}",
                            (x, y), column - 24, face)
                wrapped(draw, f"Kernel {panel['kernel_term']:+.6f}; refit {panel['refit_term']:+.6f}; sum {panel['total_term']:+.6f}",
                        (x, y), column - 24, face)
            pair = stored_pair(item)
            for index, (pol, image) in enumerate(zip(("VV", "VH"), pair)):
                top = pixel_top + index * 250
                draw.text((x, top - 20), f"{pol} stored224", fill="black", font=face)
                canvas.paste(image, (x, top))
                top = pixel_top + 500 + index * 250
                draw.text((x, top - 20), f"{pol} + target/other mask", fill="black", font=face)
                canvas.paste(overlay(image, regions[row]), (x, top))
            fractions = regions[row].mean(axis=(1, 2))
            draw.text((x, pixel_top + 983), "T/O/rest: " + "/".join(f"{v:.3f}" for v in fractions), fill="black", font=face)
            record.update(item.to_dict())
            record.update(display_acquisition=current_acq, acquisition_vs_query=relation)
            record.update({f"mask_{name}_fraction": float(v) for name, v in zip(("target", "other", "rest"), fractions)})
        records.append(record)
    return canvas.crop((0, 0, canvas.width, pixel_top + 1020)), records


def native_query(item, destination):
    """Display a source-mosaic native window with the documented stored-chip transfer."""
    from osgeo import gdal

    gdal.UseExceptions()
    gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")
    dataset = gdal.Open(str(item.source_tif), gdal.GA_ReadOnly)
    left, top, side = (int(item[k]) for k in ("left", "top", "side"))
    algorithm = gdal.GRIORA_Average if side > 224 else gdal.GRIORA_Bilinear
    face, width = font(), max(280, side + 24)
    canvas = Image.new("RGB", (2 * width, side + 240), "white")
    draw = ImageDraw.Draw(canvas)
    wrapped(draw, "Source mosaic native window, 1:1; NOT model input or verified single-acquisition pixels. Display uses stored224 per-band 2/98 limits (flat crop: min/max), no extra stretch.",
            (10, 8), canvas.width - 20, face)
    limits = {}
    for i, pol in enumerate(("vv", "vh")):
        band = dataset.GetRasterBand(i + 1)
        raw = band.ReadAsArray(left, top, side, side)
        resized = band.ReadAsArray(left, top, side, side, 224, 224, resample_alg=algorithm)
        values = resized[np.isfinite(resized)]
        if not len(values):
            raise ValueError("No finite source samples for native review")
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            low, high = float(values.min()), float(max(values.max(), values.min() + 1e-6))
        def display(values):
            with np.errstate(invalid="ignore"):
                return np.clip((values - low) / (high - low) * 255, 0, 255).astype(np.uint8)
        with Image.open(item["png_" + pol]) as original:
            if not np.array_equal(display(resized), np.asarray(original)):
                raise ValueError(f"Source replay differs from stored {pol}: clean row {item.clean_row}")
        canvas.paste(Image.fromarray(display(raw)).convert("RGB"), (i * width + 12, 210))
        wrapped(draw, f"{pol.upper()} {side}x{side}; original limits {low:.7g}, {high:.7g}",
                (i * width + 12, 125), width - 24, face)
        limits[pol] = dict(low=float(low), high=float(high), source_dtype=str(raw.dtype),
                          stored224_replay_exact=True, native_is_model_input=False,
                          source_kind="daily first-nonzero mosaic; per-pixel product ownership unverified")
    dataset = None
    canvas.save(destination)
    return limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="target_only")
    parser.add_argument("--comparator", default="global_grid")
    parser.add_argument("--predictions", type=Path, default=HERE / "predictions.csv")
    parser.add_argument("--supports", type=Path, default=HERE / "supports.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case", action="append", default=[], metavar="FOLD,BUDGET,DRAW,ROW",
                        help="Also trace this actual paired event; repeat for additional cases")
    parser.add_argument("--native-query", action="store_true", help="Add GDAL source-mosaic native query windows with documented crop display limits")
    args = parser.parse_args()
    if args.candidate == args.comparator:
        parser.error("Candidate and comparator must differ")
    for path in (args.predictions, args.supports, HERE / "masks.npz"):
        if not path.is_file():
            parser.exit(0, f"Waiting for saved integration outputs: {path}\nNo review outputs written.\n")
    cohort = pd.read_csv(HERE / "cohort.csv", dtype={"mmsi": str, "obs_time": str})
    require_columns(cohort, ["row", "label", "clean_row", "png_vv", "png_vh", "mmsi", "class_raw", "observation_key"], "cohort.csv")
    if not np.array_equal(cohort.row.to_numpy(), np.arange(len(cohort))):
        raise ValueError("cohort row order does not match masks row convention")
    cohort = cohort.set_index("row", drop=False)
    predictions = pd.read_csv(args.predictions, dtype={"fold": str})
    supports = pd.read_csv(args.supports, dtype={"fold": str})
    require_columns(supports, ["fold", "budget", "draw", "support_row", "label"], "supports.csv")
    with np.load(HERE / "masks.npz", allow_pickle=False) as saved:
        regions = saved["regions"]
    if regions.shape != (len(cohort), 3, 128, 128) or not np.isfinite(regions).all():
        raise ValueError("Expected finite masks regions[N,3,128,128]")
    events, counts = select_events(predictions, cohort, args.candidate, args.comparator)
    requested = set()
    indexed = predictions.set_index(KEYS + ["arm"])
    for value in args.case:
        fields = value.split(",")
        if len(fields) != 4:
            parser.error("--case requires FOLD,BUDGET,DRAW,ROW")
        key = (fields[0], *(int(v) for v in fields[1:]))
        own, base = (indexed.loc[key + (arm,)] for arm in (args.candidate, args.comparator))
        requested.add(key)
        if any(event[1] == key for event in events):
            continue
        correct, was_correct = own.prediction == own.truth, base.prediction == base.truth
        if correct != was_correct:
            category = "rescue" if correct else "harm"
        elif own.margin != base.margin:
            category = "persistent_" + ("correct" if correct else "wrong") + (
                "_improve" if own.margin > base.margin else "_worsen")
        else:
            parser.error(f"Requested unchanged event has no improve/worsen category: {value}")
        events.append((category, key, own, base))
    prepared = [(category, key, own, base, actual_panels(key, own, base, supports, cohort,
                 args.candidate, args.comparator)) for category, key, own, base in events]
    destination = args.output_dir or HERE / "case_review" / f"{args.candidate}_vs_{args.comparator}"
    destination.mkdir(parents=True, exist_ok=True)
    traces = []
    if (args.candidate, args.comparator) == ("target_only", "global_grid"):
        for baseline, prefix in (("global_grid", ""), ("postpool", "postpool_")):
            dpath, cpath = HERE / f"{prefix}decision_trace.csv", HERE / f"{prefix}case_support_contributions.csv"
            if dpath.is_file() and cpath.is_file():
                traces.append((baseline, pd.read_csv(dpath, dtype={"fold": str}),
                               pd.read_csv(cpath, dtype={"fold": str})))
    rows, selected = [], []
    for category, key, own, base, panels in prepared:
        name = f"B{key[1]}_{category}_{key[0]}_draw{key[2]}_row{key[3]}"
        rule = "Explicit requested trace of this saved paired event; not an additional independent result" if key in requested else RULE
        canvas, records = render_case(category, key, own, base, panels, cohort, regions, args.candidate, args.comparator, rule)
        figure = destination / f"{name}.png"
        canvas.save(figure)
        for record in records:
            record.update(figure=str(figure), candidate=args.candidate, comparator=args.comparator)
        rows.extend(records)
        event = dict(category=category, fold=key[0], budget=key[1], draw=key[2], row=key[3],
                     delta_margin=float(own.margin - base.margin), selection_rule=rule, figure=str(figure))
        for prefix, record in (("candidate", own), ("comparator", base)):
            event.update({f"{prefix}_{field}": value for field, value in record.items()})
        if args.native_query:
            path = destination / f"{name}_native_query.png"
            event["native_display_limits"] = json.dumps(native_query(cohort.loc[key[3]], path))
            event["native_query_figure"] = str(path)
        for trace_base, decisions, contributions in traces:
            trace_panels, decision = contribution_panels(key, own, supports, cohort, decisions, contributions)
            if trace_panels:
                traced_base = indexed.loc[key + (trace_base,)]
                trace_note = (f"Saved exact R50 fixed-rival path vs {trace_base} (rival={int(decision['fixed_final_rival'])}): "
                              f"kernel {decision['direct_kernel']:+.6f}, refit {decision['support_refit']:+.6f}, "
                              f"sum {decision['r50_margin_change']:+.6f}; same-class sum {decision['same_class_terms_sum']:+.6f}, "
                              f"other-class sum {decision['other_class_terms_sum']:+.6f}. Not final-probability or physical attribution.")
                supplemental, extra = render_case(f"Trace of {category} vs {args.comparator}", key, own, traced_base,
                                                  trace_panels, cohort, regions, args.candidate, trace_base, trace_note)
                suffix = "" if trace_base == args.comparator else f"_{trace_base}"
                path = destination / f"{name}{suffix}_support_contributions.png"
                supplemental.save(path)
                for record in extra:
                    record.update(figure=str(path), candidate=args.candidate, comparator=trace_base)
                rows.extend(extra)
                event[f"{trace_base}_support_contribution_figure"] = str(path)
                event.update({f"trace_{trace_base}_{field}": value for field, value in decision.items()})
        selected.append(event)
    pd.DataFrame(rows).to_csv(destination / "case_index.csv", index=False)
    pd.DataFrame(selected).to_csv(destination / "selected_events.csv", index=False)
    report = dict(candidate=args.candidate, comparator=args.comparator, selection_rule=RULE,
                  selection_randomness="None; actual support draw IDs retained", cases=len(events), counts=counts,
                  predictions=str(args.predictions), supports=str(args.supports), cohort=str(HERE / "cohort.csv"),
                  masks=str(HERE / "masks.npz"), episode_plan=str(HERE / "episodes.json"),
                  native_query=args.native_query, inference_or_fitting=False,
                  requested_cases=args.case,
                  contribution_comparators=[baseline for baseline, _, _ in traces],
                  limitations="Repeated queries/draws are not independent cases. Zero-change persistent events are not called improve/worsen. No physical mechanism inferred from images. Source mosaic lineage does not prove original-product pixel ownership; acquisition labels refer to records. Supports may share query acquisition.")
    (destination / "REVIEW.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(events)} paired cases to {destination}")


if __name__ == "__main__":
    main()
