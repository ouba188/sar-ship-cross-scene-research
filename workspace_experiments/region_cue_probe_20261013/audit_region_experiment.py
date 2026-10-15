"""Independent CPU checks; writes only AUDIT receipts beside this script."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from lxml import etree
from PIL import Image
from scipy.special import softmax


OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / "ports_root_cause_20261013"
MANIFEST = Path("E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/target_manifest.csv")
FIELDS = ("Head_x", "Head_y", "Tail_x", "Tail_y", "Center_x", "Center_y",
          "UpperLeft_x", "UpperLeft_y", "LowerRight_x", "LowerRight_y")
PATHS = ("R_query", "T_Aonly", "T_Bonly", "T_query", "R_joint", "T_joint")


def receipt(stage, result):
    (OUT / ("AUDIT_" + stage + ".json")).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def xml_rows():
    frame = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    records = []
    for scene, rows in frame.groupby("scene_id", sort=False):
        path = Path(rows.iloc[0].source_path_vv).parent.parent / "Ship.xml"
        raw = path.read_bytes().decode("utf-8", errors="replace").encode("utf-8")
        root = etree.fromstring(raw, etree.XMLParser(recover=True))
        for ship in root.findall(".//ship"):
            row = {k: int(ship.findtext("SARShipInformation/" + k)) for k in FIELDS}
            row.update(scene_id=scene, xml_mmsi=ship.findtext("AISShipInformation/MMSI"))
            records.append(row)
    frame[["Center_x", "Center_y"]] = frame.source_path_vv.str.extract(
        r"_x(-?\d+)_y(-?\d+)_vv\.tif$").astype(int)
    result = frame.merge(pd.DataFrame(records), on=["scene_id", "Center_x", "Center_y"],
                         how="left", validate="one_to_one")
    assert result[list(FIELDS)].notna().all().all()
    np.testing.assert_array_equal(result.mmsi_or_ship_id, result.xml_mmsi)
    return result


def mapped(row):
    w, h = int(row.chip_width_px), int(row.chip_height_px)
    x0, y0 = min(row.UpperLeft_x, row.LowerRight_x), min(row.UpperLeft_y, row.LowerRight_y)
    full_w = abs(row.UpperLeft_x - row.LowerRight_x) + 1
    full_h = abs(row.UpperLeft_y - row.LowerRight_y) + 1
    boundary = (w, h) != (full_w, full_h)
    if boundary:
        # This one native MATLAB chip has scene rows 1..34, not -10..34.
        assert row.sample_id == "pub2_opensarship2_003922"
        assert (w, h, y0, max(row.UpperLeft_y, row.LowerRight_y)) == (45, 34, -10, 34)
        y0 = 1
    points = np.array([[row.Head_x - x0, row.Head_y - y0],
                       [row.Tail_x - x0, row.Tail_y - y0]], dtype=float)
    valid = bool(not boundary and np.any(points[0] != points[1]) and
                 ((points >= 0) & (points < [w, h])).all())
    return w, h, x0, y0, points, valid, boundary


def metadata_audit():
    frame = xml_rows()
    anomalies, sizes = [], []
    exact_extents = 0
    valid_count = 0
    for row in frame.itertuples(index=False):
        w, h, x0, y0, points, valid, boundary = mapped(row)
        exact_extents += not boundary
        for pol in ("vv", "vh"):
            with Image.open(getattr(row, "chip_path_" + pol)) as image:
                assert image.size == (w, h), row.sample_id
        valid_count += valid
        sizes.append((w, h))
        if boundary or not valid:
            anomalies.append(dict(sample_id=row.sample_id, size=[w, h], origin=[x0, y0],
                                  points=points.tolist(), region_valid=valid,
                                  boundary_origin_inferred=boundary))
    receipt("metadata", dict(rows=len(frame), products=frame.scene_id.nunique(),
        exact_xml_extent_rows=exact_extents, region_valid=valid_count,
        both_png_sizes_verified=True, anomalies=anomalies,
        provenance="Native Ship.xml + unchanged Patch_Uint8 import; no upstream MATLAB crop source found",
        precision="Integer scene pixels; boundary origin is an explicit inference; not a ship segmentation mask"))


def mean_classes(values, labels):
    return float(np.mean([np.asarray(values)[labels == c].mean() for c in np.unique(labels)]))


def margin(scores, labels):
    true = scores[np.arange(len(labels)), labels]
    wrong = scores.copy()
    wrong[np.arange(len(labels)), labels] = 0
    return true - wrong.sum(axis=1) / (scores.shape[1] - 1)


def selected_rows(frame, control, data, draw, name):
    selected = np.asarray(data["eval_masks"][draw], dtype=bool).copy()
    selected &= control.region_valid.astype(str).str.lower().isin(["true", "1"]).to_numpy()
    if name in ("GRDH", "SLC"):
        selected &= frame.scene_id.str.contains("_" + name + "_", regex=False).to_numpy()
    elif name == "clean":
        selected &= frame.noise_xml.astype(str).eq("0").to_numpy()
    elif name == "unseen_support_vessel":
        known = set(frame.identity.iloc[data["support"][draw]].astype(str))
        selected &= ~frame.identity.astype(str).isin(known).to_numpy()
    elif name in ("sham_valid", "region_valid"):
        values = control[name].astype(str).str.lower()
        assert values.isin(["true", "false", "1", "0"]).all()
        selected &= values.isin(["true", "1"]).to_numpy()
    else:
        assert name == "full", name
    return selected


def score_audit():
    needed = ["cohort.npz", "cohort.csv", "region_metadata.csv", "READOUT_COMPLETE.json"]
    missing = [name for name in needed if not (OUT / name).exists()]
    if missing:
        receipt("scores", dict(status="pending", missing=missing))
        return
    with np.load(OUT / "cohort.npz") as archive:
        data = {k: archive[k] for k in archive.files}
    frame = pd.read_csv(OUT / "cohort.csv", keep_default_na=False)
    control = pd.read_csv(OUT / "region_metadata.csv", keep_default_na=False)
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str), data["ids"].astype(str))
    np.testing.assert_array_equal(control.sample_id.to_numpy(str), data["ids"].astype(str))
    original_frame = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False).set_index("sample_id")
    np.testing.assert_array_equal(frame.class_id.to_numpy(int),
                                  original_frame.loc[data["ids"], "class_id"].to_numpy(int))
    yall = frame.class_id.to_numpy(int)
    for draw, support in enumerate(data["support"]):
        assert not data["eval_masks"][draw, support].any()
        assert len(np.unique(support)) == len(support)
        np.testing.assert_array_equal(yall[support], data["support_labels"][draw])
    baseline, cached = {}, {}
    replay_error = 0.0
    for fold in (2, 3):
        baseline[fold] = np.load(OUT / f"baseline_fold_{fold}.npy", mmap_mode="r")
        old_dir = PREVIOUS / f"predictions/original/original/fold_{fold}/oss2"
        with np.load(old_dir / "predictions.npz") as old:
            for key in ("ids", "support", "eval_masks"):
                np.testing.assert_array_equal(old[key], data[key])
            arm_ids = [list(old["arms"]).index(arm) for arm in ("R", "T")]
            expert = list(old["heads"]).index("expert")
        old_p = np.load(old_dir / "probabilities.npy", mmap_mode="r")
        for arm, ai in enumerate(arm_ids):
            p = softmax(baseline[fold][:, arm], axis=-1).astype(np.float32)
            expected = old_p[:, ai, expert]
            np.testing.assert_allclose(p, expected, atol=2e-7, rtol=0)
            np.testing.assert_array_equal(p.argmax(-1), expected.argmax(-1))
            replay_error = max(replay_error, float(np.max(np.abs(p - expected))))

    def scores(fold, condition):
        key = fold, condition
        if key not in cached:
            cached[key] = np.load(OUT / f"scores_fold_{fold}_{condition}.npy", mmap_mode="r")
        return cached[key]

    numerical_control = []
    for draw in (0, 4):
        selected = data["eval_masks"][draw]
        for i, base_i in enumerate((0, 1, 1, 1, 0, 1)):
            now = scores(2, "replay_original")[draw, i][selected]
            before = baseline[2][draw, base_i][selected]
            np.testing.assert_array_equal(now.argmax(1), before.argmax(1))
            error = float(np.max(abs(softmax(now, axis=1)-softmax(before, axis=1))))
            assert error < 1e-12
            numerical_control.append(dict(fold=2, draw=draw, path=PATHS[i],
                                          prediction_flips=0, probability_max_error=error))

    invalid = ~control.region_valid.to_numpy(bool)
    original_features = OUT.parent / "ports_input_controls_20261012/features/oss2/original"
    old_indices = pd.Index(np.load(original_features / "ids.npy")).get_indexer(data["ids"][invalid])
    assert (old_indices >= 0).all()
    invalid_cache_checks = 0
    for folder in sorted((OUT / "features").iterdir()):
        if not (folder / "COMPLETE.json").exists():
            continue
        np.testing.assert_array_equal(np.load(folder / "ids.npy"), data["ids"])
        for key in ("raw", "extra"):
            current = np.load(folder / (key + ".npy"), mmap_mode="r")[invalid]
            old = np.load(original_features / (key + ".npy"), mmap_mode="r")[old_indices]
            np.testing.assert_array_equal(current, old)
            invalid_cache_checks += 1

    checked = {"path_effects": 0, "utility": 0, "classes": 0}
    maximum_error = 0.0
    audit_conditions = ("original", "target_flat", "outer_flat", "target_shuffle", "target_display", "replay_original")

    def compare(row, values):
        nonlocal maximum_error
        for name, expected in values.items():
            error = abs(float(getattr(row, name)) - float(expected))
            maximum_error = max(maximum_error, error)
            assert error < 1e-8, (row.fold, row.draw, row.condition, row.scope, name, error)

    for filename in checked:
        table = pd.read_csv(OUT / (filename + ".csv"))
        subset = table[(table.fold == 2) & table.draw.isin([0, 4]) & table.condition.isin(audit_conditions)]
        for row in subset.itertuples(index=False):
            ix = selected_rows(frame, control, data, row.draw, row.scope)
            y = yall[ix]
            assert row.n == len(y)
            base = baseline[row.fold][row.draw][:, ix]
            zz = scores(row.fold, row.condition)[row.draw][:, ix]
            r0, t0 = base[0].argmax(1), base[1].argmax(1)
            rescue0, harm0 = (r0 != y) & (t0 == y), (r0 == y) & (t0 != y)
            pct = lambda v: 100 * mean_classes(v, y)
            if filename == "utility":
                ri, ti = (0, 3) if row.mode == "query" else (4, 5)
                rp, tp = zz[ri].argmax(1), zz[ti].argmax(1)
                rescue, harm = (rp != y) & (tp == y), (rp == y) & (tp != y)
                values = dict(r_ba=pct(rp == y), t_ba=pct(tp == y),
                    net=pct(tp == y)-pct(rp == y), rescue=pct(rescue), harm=pct(harm),
                    original_net=pct(t0 == y)-pct(r0 == y), retained_rescue=pct(rescue0 & rescue),
                    old_rescue_both_correct=pct(rescue0 & (rp == y) & (tp == y)),
                    old_rescue_t_wrong=pct(rescue0 & (tp != y)), old_harm_avoided=pct(harm0 & ~harm))
            else:
                z = zz[PATHS.index(row.path)]
                b = base[0 if row.path.startswith("R_") else 1]
                p, bp = z.argmax(1), b.argmax(1)
                md = margin(z, y) - margin(b, y)
                if filename == "classes":
                    cm = y == row.class_id
                    values = dict(count=cm.sum(), recall=100*(p[cm] == y[cm]).mean(),
                        delta_recall=100*((p[cm] == y[cm]).mean()-(bp[cm] == y[cm]).mean()),
                        margin_delta=md[cm].mean())
                else:
                    values = dict(ba=pct(p == y), oa=100*(p == y).mean(),
                        delta_ba=pct(p == y)-pct(bp == y), flips_macro=pct(p != bp),
                        flips_micro=100*(p != bp).mean(), margin_delta=mean_classes(md, y),
                        old_rescue_mass=pct(rescue0), old_rescue_lost=pct(rescue0 & (p != y)),
                        old_harm_corrected=pct(harm0 & (p == y)))
            compare(row, values)
            checked[filename] += 1
        keys = ["condition", "scope", "mode"] if filename == "utility" else ["condition", "scope", "path"]
        if filename == "classes":
            keys += ["class_id"]
        numeric = [c for c in table if c not in keys + ["fold", "draw"]]
        expected = table.groupby(keys)[numeric].mean().sort_index()
        saved = pd.read_csv(OUT / (filename + "_summary.csv")).set_index(keys).sort_index()
        assert expected.index.equals(saved.index)
        np.testing.assert_allclose(expected[numeric], saved[numeric], atol=1e-8, rtol=0)
    receipt("scores", dict(status="passed", rows=len(frame),
        query_counts=data["eval_masks"].sum(axis=1).tolist(), checked=checked,
        valid_query_counts=(data["eval_masks"] & ~invalid).sum(axis=1).tolist(),
        invalid_support_counts=invalid[data["support"]].sum(axis=1).tolist(),
        invalid_feature_cache_checks=invalid_cache_checks, numerical_control=numerical_control,
        strict_region_valid=int(control.region_valid.sum()),
        audited_folds=[2], audited_draws=[0, 4], audited_conditions=list(audit_conditions),
        all_scopes=True, summary_aggregation="all rows, all conditions, folds and draws",
        baseline_probability_max_error=replay_error, metric_max_error=maximum_error,
        denominator="Each scope: unweighted mean of per-present-class recalls; query rows only",
        scope="Cached scores and raw replay, no GPU, no feature or coefficient refit"))


def pixel_audit():
    import transforms_regions as transforms

    frame = xml_rows()
    control = pd.read_csv(OUT / "region_metadata.csv", keep_default_na=False).set_index("sample_id")
    chosen = set(frame.groupby(frame.scene_id.str.contains("_SLC_")).head(1).index)
    strata = frame.assign(product=frame.scene_id.str.contains("_SLC_"))
    chosen.update(strata.groupby(["product", "class_id"]).head(1).index)
    chosen.update(frame.chip_width_px.astype(int).nsmallest(3).index)
    chosen.update(frame.chip_width_px.astype(int).nlargest(3).index)
    expected_valid, sham_counts, checked_pixels = [], {"0.2": 0, "0.35": 0}, 0
    conditions = ("original", "replay_original", "target_flat", "outer_flat", "sham_flat",
                  "target_shuffle", "outer_shuffle", "target_display", "outer_display",
                  "wide_target_flat", "wide_outer_flat")
    for index, row in enumerate(frame.itertuples(index=False)):
        w, h, x0, y0, points, valid, boundary = mapped(row)
        expected_valid.append(valid)
        if not valid:
            chosen.add(index)
        native = SimpleNamespace(**{k.lower(): getattr(row, k) for k in FIELDS})
        masks = {}
        for padding in (.20, .35):
            box, sham, actual_valid, sham_valid, length, actual_points = transforms.region_boxes(native, (h, w), padding)
            assert actual_valid == valid, row.sample_id
            if not valid:
                assert not sham_valid
                masks[padding] = (np.zeros((h, w), bool), np.zeros((h, w), bool))
                continue
            np.testing.assert_array_equal(actual_points, points)
            distance = np.linalg.norm(points[0] - points[1])
            np.testing.assert_allclose(length, distance, atol=0, rtol=1e-14)
            radius = max(3, padding * distance)
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            expected = ((xx >= np.floor(points[:, 0].min()-radius)) &
                        (xx <= np.ceil(points[:, 0].max()+radius)) &
                        (yy >= np.floor(points[:, 1].min()-radius)) &
                        (yy <= np.ceil(points[:, 1].max()+radius)))
            np.testing.assert_array_equal(transforms.box_mask((h, w), box), expected)
            sham_mask = transforms.box_mask((h, w), sham)
            if sham_valid:
                assert expected.sum() == sham_mask.sum()
                assert not (expected & sham_mask).any()
                assert box[2]-box[0] == sham[2]-sham[0] and box[3]-box[1] == sham[3]-sham[1]
            else:
                bw, bh = box[2]-box[0], box[3]-box[1]
                assert all(expected[cy:cy+bh, cx:cx+bw].any()
                           for cx, cy in ((0, 0), (w-bw, 0), (0, h-bh), (w-bw, h-bh)))
            sham_counts[str(padding)] += sham_valid
            masks[padding] = expected, sham_mask
            if padding == .20:
                saved = control.loc[row.sample_id]
                np.testing.assert_array_equal([saved[k] for k in ("x0", "y0", "x1", "y1")], box)
                np.testing.assert_array_equal([saved.hx, saved.hy, saved.tx, saved.ty], points.ravel())
                assert bool(saved.sham_valid) == sham_valid
        assert bool(control.loc[row.sample_id].region_valid) == valid
        if index not in chosen:
            continue
        with Image.open(row.chip_path_vv) as vv, Image.open(row.chip_path_vh) as vh:
            pair = np.stack([np.asarray(vv), np.asarray(vh)])
        for condition in conditions:
            result, info = transforms.transform(pair, native, condition, index)
            target, sham = masks[.35 if condition.startswith("wide_") else .20]
            zone = condition.removeprefix("wide_").split("_")[0]
            if condition in ("original", "replay_original") or not valid:
                mask = np.zeros((h, w), bool)
            else:
                mask = target if zone == "target" else ~target if zone == "outer" else sham
            np.testing.assert_array_equal(result[:, ~mask], pair[:, ~mask])
            assert result.dtype == pair.dtype and result.shape == pair.shape
            if mask.any() and condition.endswith("flat"):
                for pol in range(2):
                    border = np.concatenate([pair[pol, 0], pair[pol, -1], pair[pol, :, 0], pair[pol, :, -1]])
                    assert np.all(result[pol, mask] == np.rint(np.median(border)))
            if mask.any() and condition.endswith("shuffle"):
                # Encode both channels together: detects independent-polarization permutations.
                before = pair[0, mask].astype(np.uint16)*256+pair[1, mask]
                after = result[0, mask].astype(np.uint16)*256+result[1, mask]
                np.testing.assert_array_equal(np.sort(before), np.sort(after))
            checked_pixels += 1
    receipt("pixels", dict(status="passed", mapping_rows=len(frame), strict_region_valid=sum(expected_valid),
        sham_valid_by_padding=sham_counts, native_pixel_samples=len(chosen), pixel_condition_checks=checked_pixels,
        sample_ids=frame.iloc[sorted(chosen)].sample_id.tolist(), conditions=list(conditions),
        checks="XML mapping; padded rectangle; same-shape equal-area disjoint sham; region metadata; unchanged complement; flat fill; paired-channel shuffle multiset; invalid and replay unchanged"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("metadata", "pixels", "scores"))
    args = parser.parse_args()
    {"metadata": metadata_audit, "pixels": pixel_audit, "scores": score_audit}[args.stage]()
