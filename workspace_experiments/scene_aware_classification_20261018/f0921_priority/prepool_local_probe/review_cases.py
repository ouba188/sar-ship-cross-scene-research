"""Review saved paired predictions and native chips; no inference or fitting."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "region_relation_probe"
OWN = "prepool"
GRID = "grid12"
KEYS = ["fold", "draw", "row"]
CATEGORIES = ("rescue", "harm", "bothcorrect_improve", "bothcorrect_worsen",
              "bothwrong_improve", "bothwrong_worsen", "changed_prediction")
COLORS = np.array([[230, 70, 70], [240, 170, 40], [140, 140, 140]], dtype=np.uint8)
SUPPORT_NOTE = "Saved nearest wrong-class support; not necessarily the predicted-class support"
PREDICTED_NOTE = "Saved highest-scoring support within this arm's predicted class (top_predicted_row)"


def selection_rule(category, candidate=OWN, comparator=GRID):
    if category == "changed_prediction":
        return "Wrong-class switch only, no correctness flips; max absolute delta_margin; ties fold,row,draw"
    extreme = "min" if category in ("harm", "bothcorrect_worsen", "bothwrong_worsen") else "max"
    return f"{extreme} signed ({candidate} - {comparator}) relation_margin within {category}; ties fold,row,draw"


def ownership_regions(regions):
    # Merge the old water/rest masks into one background before display.
    return np.stack((regions[:, 0], regions[:, 1], regions[:, 2] + regions[:, 3]), axis=1)


def select_events(pred, meta, candidate=OWN, comparator=GRID):
    parts = {}
    for arm in (candidate, comparator):
        part = pred.loc[pred.arm.eq(arm)].copy()
        if part.empty:
            raise ValueError(f"No saved predictions for {arm}")
        if part.duplicated(KEYS).any():
            raise ValueError(f"Duplicate fold/draw/row predictions for {arm}")
        parts[arm] = part.set_index(KEYS).sort_index()
    own, grid = parts[candidate], parts[comparator]
    if not own.index.equals(grid.index):
        raise ValueError("Arms must contain identical fold/draw/row events")
    for column in ("truth", "object_id", "mmsi"):
        if not own[column].equals(grid[column]):
            raise ValueError(f"Paired {column} differs between arms")
    rows = own.index.get_level_values("row").to_numpy(dtype=int)
    if np.any((rows < 0) | (rows >= len(meta))):
        raise ValueError("Prediction row is outside sibling cohort.csv")
    for column, metadata_column in (("object_id", "object_id"), ("truth", "label"),
                                    ("mmsi", "mmsi")):
        if not np.array_equal(own[column].to_numpy(), meta.iloc[rows][metadata_column].to_numpy()):
            raise ValueError(f"Prediction {column} does not match sibling cohort row")
    # Derive correctness from saved labels to avoid string-to-bool CSV ambiguity.
    correct_own = own.prediction.eq(own.truth)
    correct_grid = grid.prediction.eq(grid.truth)
    paired = pd.DataFrame({"delta_margin": own.relation_margin - grid.relation_margin})
    if not np.isfinite(paired.delta_margin).all():
        raise ValueError("Nonfinite relation margins cannot rank review events")
    paired["abs_delta_margin"] = paired.delta_margin.abs()
    rescue = correct_own & ~correct_grid
    harm = ~correct_own & correct_grid
    bothcorrect = correct_own & correct_grid
    bothwrong = ~correct_own & ~correct_grid
    wrong_switch = bothwrong & own.prediction.ne(grid.prediction)
    if rescue.any() or harm.any():
        wrong_switch = wrong_switch & False
    groups = (rescue, harm, bothcorrect, bothcorrect, bothwrong, bothwrong, wrong_switch)
    events = []
    for category, mask in zip(CATEGORIES, groups):
        ascending = category == "harm" or category.endswith("worsen")
        ranking = "abs_delta_margin" if category == "changed_prediction" else "delta_margin"
        candidates = paired.loc[mask].reset_index().sort_values(
            [ranking, "fold", "row", "draw"],
            ascending=[ascending, True, True, True], kind="stable")
        if candidates.empty:
            continue
        event = candidates.iloc[0]
        key = (event.fold, int(event.draw), int(event.row))
        events.append((category, key, own.loc[key], grid.loc[key]))
    return events


def support_panels(query_row, own, grid, meta, candidate=OWN, comparator=GRID):
    panels = [("QUERY", query_row, "query", "")]
    for tag, arm, record in (("CAND", candidate, own), ("COMP", comparator, grid)):
        for field, kind in (("top_same_row", "same"), ("top_wrong_row", "nearest_wrong"),
                            ("top_predicted_row", "predicted_class")):
            value = record[field]
            if pd.isna(value) or float(value) != int(value):
                raise ValueError(f"Invalid saved support row: {field}={value}")
            row = int(value)
            if row < 0 or row >= len(meta):
                raise ValueError(f"Support row outside sibling cohort: {row}")
            same = int(meta.iloc[row].label) == int(record.truth)
            if kind != "predicted_class" and same != (kind == "same"):
                raise ValueError(f"Saved {field} has an inconsistent class at row {row}")
            if kind == "predicted_class" and int(meta.iloc[row].label) != int(record.prediction):
                raise ValueError(f"Saved {field} does not belong to the predicted class at row {row}")
            if meta.iloc[row]["product"] == meta.iloc[query_row]["product"]:
                raise ValueError("Saved cross-acquisition support belongs to query product")
            panels.append((f"{tag} {kind}", row, kind, arm))
    return panels


def native_path(value):
    path = Path(value)
    return path if path.is_absolute() else SOURCE / path


def wrapped(draw, text, x, y, width, font):
    line = ""
    for word in str(text).split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=font) > width:
            draw.text((x, y), line, fill="black", font=font)
            y += 17
            line = word
        else:
            line = candidate
    draw.text((x, y), line, fill="black", font=font)
    return y + 17


def render_case(category, key, own, grid, meta, regions, candidate=OWN, comparator=GRID):
    panels = support_panels(key[2], own, grid, meta, candidate, comparator)
    font = ImageFont.load_default(size=14)
    images = {}
    for _, row, _, _ in panels:
        for pol in ("vv", "vh"):
            path = native_path(meta.iloc[row]["native_" + pol])
            with Image.open(path) as im:
                images[row, pol] = im.convert("RGB")
    max_width = max(im.width for im in images.values())
    max_height = max(im.height for im in images.values())
    column = max(280, max_width + 24)
    slot = max(150, max_height)
    vv_y, vh_y = 270, 300 + slot
    mask_y = vh_y + slot + 45
    footer_y = mask_y + 260
    canvas = Image.new("RGB", (column * len(panels), footer_y + 140), "white")
    draw = ImageDraw.Draw(canvas)
    delta = float(own.relation_margin - grid.relation_margin)
    header = [
        f"{category} | fold {key[0]} draw {key[1]} query row {key[2]} | cohort label {int(own.truth)}",
        f"CAND={candidate} prediction {int(own.prediction)} vs COMP={comparator} {int(grid.prediction)} | relation margin {own.relation_margin:+.5f} vs {grid.relation_margin:+.5f} | delta {delta:+.5f}",
        f"AP {own.ap:.4f} vs {grid.ap:.4f}; Brier {own.brier:.4f} vs {grid.brier:.4f}; NLL {own.nll:.4f} vs {grid.nll:.4f}",
        "Selection: " + selection_rule(category, candidate, comparator) + ". Descriptive extremes, not population proportions.",
        "Native VV/VH are shown at 1:1 pixels without restretching; masks are a separate resized schematic.",
        "Predicted-class columns use top_predicted_row; nearest-wrong columns may belong to another class. Repeated chips retain their actual roles.",
    ]
    y = 8
    for text in header:
        y = wrapped(draw, text, 10, y, canvas.width - 20, font)
    records = []
    for j, (role, row, kind, arm) in enumerate(panels):
        item = meta.iloc[row]
        x = j * column + 10
        draw.text((x, 122), f"{role} | row {row}", fill="black", font=font)
        draw.text((x, 141), f"class {int(item.label)} | {item['product'][-4:]} | MMSI {item.mmsi}", fill="black", font=font)
        wrapped(draw, item.class_raw.replace("_", " "), x, 160, column - 20, font)
        match_grid = int(item.label) == int(grid.prediction)
        match_own = int(item.label) == int(own.prediction)
        draw.text((x, 201), f"Class = COMP prediction: {match_grid}", fill="black", font=font)
        draw.text((x, 218), f"Class = CAND prediction: {match_own}", fill="black", font=font)
        for pol, top in (("vv", vv_y), ("vh", vh_y)):
            im = images[row, pol]
            draw.text((x, top - 20), f"{pol.upper()} native {im.width}x{im.height} (1:1)", fill="black", font=font)
            canvas.paste(im, (j * column + (column - im.width) // 2, top))
        mask = Image.fromarray(COLORS[regions[row].argmax(0)])
        mask = mask.resize((224, 224), Image.Resampling.NEAREST)
        lines = ImageDraw.Draw(mask)
        lines.line((112, 0, 112, 223), fill="black", width=1)
        lines.line((0, 112, 223, 112), fill="black", width=1)
        draw.text((x, mask_y - 20), "Three ownership masks + 2x2 grid", fill="black", font=font)
        canvas.paste(mask, (x, mask_y))
        area = regions[row].mean(axis=(1, 2))
        draw.text((x, mask_y + 230), "Area T/O/B: " + "/".join(f"{v:.2f}" for v in area), fill="black", font=font)
        record = dict(category=category, role=role, support_kind=kind, support_arm=arm,
                      fold=key[0], draw=key[1], query_row=key[2], row=row,
                      object_id=item.object_id, mmsi=item.mmsi, product=item["product"],
                      label=int(item.label), class_raw=item.class_raw,
                      native_vv=str(native_path(item.native_vv)), native_vh=str(native_path(item.native_vh)),
                      native_vv_width=images[row, "vv"].width, native_vv_height=images[row, "vv"].height,
                      native_vh_width=images[row, "vh"].width, native_vh_height=images[row, "vh"].height,
                      truth_verified=item.get("truth_verified", False),
                      source_key_json=item.get("source_key_json", ""),
                      cohort_path=str(SOURCE / "cohort.csv"), mask_path=str(SOURCE / "inputs.npz"), mask_row=row,
                      query_truth=int(own.truth), candidate_arm=candidate, comparator_arm=comparator,
                      delta_margin=delta,
                      delta_brier=float(own.brier - grid.brier), delta_ap=float(own.ap - grid.ap),
                      delta_nll=float(own.nll - grid.nll),
                      abs_delta_margin=abs(delta),
                      selection_rule=selection_rule(category, candidate, comparator),
                      support_evidence={"nearest_wrong": SUPPORT_NOTE, "predicted_class": PREDICTED_NOTE,
                                        "same": "Saved highest-scoring same-class support", "query": "Query"}[kind],
                      figure=f"case_{category}.png")
        record.update({f"{candidate}_prediction": int(own.prediction),
                       f"{comparator}_prediction": int(grid.prediction),
                       f"class_matches_{candidate}_prediction": match_own,
                       f"class_matches_{comparator}_prediction": match_grid})
        for name, value in zip(("target", "other", "background"), area):
            record[f"mask_{name}_fraction"] = float(value)
        for prefix, prediction in ((candidate, own), (comparator, grid)):
            for field in ("ap", "relation_margin", "brier", "nll", "top_support_row", "top_same_row", "top_wrong_row", "top_predicted_row", *[f"p{k}" for k in range(7)]):
                record[f"{prefix}_{field}"] = prediction[field]
        records.append(record)
    method_note = f"Saved prediction comparison: {candidate} vs {comparator}. No inference or fitting in this review."
    if (candidate, comparator) == (OWN, GRID):
        method_note = "PREPOOL: 2x2 spatial cells x 3 ownership groups before pooling (12 tokens); GRID12: 3x4 spatial grid (12 tokens). No training."
    elif (candidate, comparator) == ("target_only", "postpool"):
        method_note = "TARGET_ONLY changes the target slot before pooling; other slots retain POSTPOOL. No training."
    notes = [
        "Masks: red target candidate; orange other candidates; gray all remaining background. No map water/land distinction. Black lines: 2x2 grid.",
        method_note,
        "Labels and spatial associations remain exploratory proposals, not human-verified identity truth. Correct/wrong refer to cohort labels.",
        "Predicted-class supports come from each arm's saved top_predicted_row; rankings are not recomputed here.",
    ]
    for text in notes:
        footer_y = wrapped(draw, text, 10, footer_y, canvas.width - 20, font) + 5
    return canvas, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=HERE / "predictions.csv")
    parser.add_argument("--candidate", default=OWN)
    parser.add_argument("--comparator", default=GRID)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    if args.candidate == args.comparator:
        parser.error("Candidate and comparator must be different arms")
    if not args.predictions.is_file():
        parser.error(f"Saved predictions do not exist yet: {args.predictions}")
    meta = pd.read_csv(SOURCE / "cohort.csv", dtype={"mmsi": str})
    pred = pd.read_csv(args.predictions, dtype={"fold": str, "mmsi": str})
    with np.load(SOURCE / "inputs.npz", allow_pickle=False) as data:
        regions = ownership_regions(data["regions"])
        if not np.array_equal(data["labels"], meta.label.to_numpy()):
            raise ValueError("Sibling inputs.npz labels do not match cohort order")
        if not np.array_equal(data["mmsi"].astype(str), meta.mmsi.to_numpy()):
            raise ValueError("Sibling inputs.npz MMSIs do not match cohort order")
    events = select_events(pred, meta, args.candidate, args.comparator)
    support_path = HERE / "supports.csv"
    supports = pd.read_csv(support_path, dtype={"fold": str}) if support_path.is_file() else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for category, key, own, grid in events:
        if supports is not None:
            allowed = set(supports.loc[supports.fold.eq(key[0]) & supports.draw.eq(key[1]), "support_row"])
            shown = support_panels(key[2], own, grid, meta, args.candidate, args.comparator)
            if any(row not in allowed for _, row, kind, _ in shown if kind != "query"):
                raise ValueError(f"A displayed support is absent from the actual saved support draw {key[:2]}")
        canvas, rows = render_case(category, key, own, grid, meta, regions, args.candidate, args.comparator)
        canvas.save(args.output_dir / f"case_{category}.png")
        for row in rows:
            row["predictions_path"] = str(args.predictions.resolve())
            row["support_membership_checked"] = supports is not None
        records.extend(rows)
    pd.DataFrame(records).to_csv(args.output_dir / "case_index.csv", index=False)
    missing = sorted(set(CATEGORIES) - {event[0] for event in events})
    print(f"Wrote {len(events)} panels and case_index.csv to {args.output_dir}")
    print(f"Absent categories: {', '.join(missing) or 'none'}; case_index.csv lists this run's panels.")
    print(PREDICTED_NOTE)
    if not records:
        return
    queries = pd.DataFrame(records).query("support_kind == 'query'")
    print(queries[["category", "fold", "draw", "query_row", "query_truth",
                   f"{args.candidate}_prediction", f"{args.comparator}_prediction", "delta_margin", "delta_brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
