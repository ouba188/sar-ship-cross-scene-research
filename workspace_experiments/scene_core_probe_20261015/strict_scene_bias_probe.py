"""Held-port B16 comparison of scene bias and pair-dependent support kernels.

The scene cohort uses extra unlabeled SAR chips, not query labels or identities.
Its membership is frozen by the existing non-overlap episode, not by outcomes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.special import softmax
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SCENE = PROJECT / "scene_cross_dataset_redesign_20261014"
CONTEXT = PROJECT / "context_condition_study_20261015"
REGION = PROJECT / "region_cue_replication_20261013"
FOLD = "Singapore__S1A_060968_079865"
PORT = "Singapore"
SOURCE = SCENE / "ports_full_expansion" / PORT
EXTRA = HERE / "singapore_extra"
sys.path.insert(0, str(SCENE / "ports_full_expansion"))
sys.path.insert(0, str(CONTEXT))
sys.path.insert(0, str(PROJECT / "scene_cross_dataset_redesign_20261014/public_full_20261014"))
from evaluate import load  # noqa: E402
from source_context_pilot import design_block, fit_head, coefficients  # noqa: E402
from audit_ports_context import chip_overlap  # noqa: E402


def frozen_episode():
    episodes = json.loads((CONTEXT / "target_episodes.json").read_text(encoding="utf-8"))
    return next(item for item in episodes if item["fold"] == FOLD)


def clean_cohorts(episode):
    folder = Path(episode["folder"])
    query = pd.read_csv(folder / "query.csv", dtype={"mmsi": str})
    query = query.loc[query.sample_id.isin(episode["query"])]
    candidates = pd.read_csv(folder / "support_candidates.csv", dtype={"mmsi": str})
    far_ok = set(pd.read_csv(CONTEXT / "geometry_by_sample.csv",
                             usecols=["sample_id", "far_ok"]).query("far_ok").sample_id)
    candidates = candidates.loc[candidates.sample_id.isin(far_ok)]
    positions = pd.read_csv(REGION / "source_region_metadata.csv", usecols=[
        "sample_id", "resolved_crop_left_vv", "resolved_crop_top_vv",
        "resolved_crop_left_vh", "resolved_crop_top_vh"])
    shape = pd.read_csv(REGION / "ports/frame.csv", usecols=[
        "sample_id", "scene_id", "native_h_vv", "native_w_vv", "native_h_vh", "native_w_vh"])
    coords = positions.merge(shape, on="sample_id", validate="one_to_one").set_index("sample_id")
    qpos = coords.loc[query.sample_id]
    same = candidates.support_origin.eq("same")
    overlap = [origin and any(chip_overlap(coords.loc[row.sample_id], q)
                              for _, q in qpos.iterrows())
               for origin, row in zip(same, candidates.itertuples())]
    candidates = candidates.loc[~np.asarray(overlap)]
    candidates = candidates.loc[~candidates.mmsi.isin(query.mmsi)]
    candidates = candidates.sort_values("sample_id").drop_duplicates(["support_origin", "mmsi"])
    for draw in episode["draws"]:
        for origin in ("same", "cross"):
            assert set(draw[origin]).issubset(set(candidates.sample_id))
    assert not set(query.mmsi) & set(candidates.mmsi)
    print("clean unlabeled cohort", candidates.groupby("support_origin").mmsi.nunique().to_dict(), flush=True)
    return candidates


def encode_missing(rows):
    EXTRA.mkdir(exist_ok=True)
    row_file = EXTRA / "rows.npy"
    if row_file.exists():
        np.testing.assert_array_equal(np.load(row_file), rows)
        if all((EXTRA / (name + ".npy")).exists() for name in ("c4", "zero", "G18", "D18")):
            return
    from encode import PIXELS, encode_r50, encode_r18  # noqa: E402
    from general_domain_ssl import backbone  # noqa: E402
    np.save(row_file, rows)
    pixels = np.load(PIXELS, mmap_mode="r")
    torch.set_num_threads(2)
    r50, _ = backbone("r50")
    encode_r50(r50.cuda().eval(), pixels, rows, EXTRA)
    del r50
    torch.cuda.empty_cache()
    r18, _ = backbone("r18")
    encode_r18(r18.cuda().eval(), pixels, rows, EXTRA, "G18")
    checkpoint = torch.load(SOURCE / "source_only_r18_best.pt", map_location="cpu", weights_only=False)
    r18.load_state_dict(checkpoint["backbone"], strict=True)
    encode_r18(r18.cuda().eval(), pixels, rows, EXTRA, "D18")


def merged_raw(original, old_rows, new_rows, selected):
    width = original.shape[1]
    result = np.empty((len(selected), width), np.float32)
    old_pos = np.searchsorted(old_rows, selected)
    old_good = (old_pos < len(old_rows)) & (old_rows[np.minimum(old_pos, len(old_rows)-1)] == selected)
    result[old_good] = np.asarray(original[old_pos[old_good]], np.float32)
    extra_pos = np.searchsorted(new_rows, selected[~old_good])
    assert (extra_pos < len(new_rows)).all()
    np.testing.assert_array_equal(new_rows[extra_pos], selected[~old_good])
    return result, old_good, extra_pos


def feature_model(data, selected):
    old_rows = data["rows"]
    extra_rows = np.load(EXTRA / "rows.npy")
    source_global = old_rows[np.r_[data["train"], data["val"]]]
    source = np.searchsorted(selected, source_global)
    np.testing.assert_array_equal(selected[source], source_global)
    manifest = pd.read_csv(PROJECT / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv",
                           usecols=["class_id"])
    labels = manifest.class_id.to_numpy(int)[selected]
    components = {}
    for name, original, extra_name in (("base", data["c4"], "c4"),
                                       ("general", data["raw"]["G50"], "zero"),
                                       ("domain", data["raw"]["D18"], "D18")):
        values, old_good, extra_pos = merged_raw(original, old_rows, extra_rows, selected)
        values[~old_good] = np.load(EXTRA / (extra_name + ".npy"), mmap_mode="r")[extra_pos]
        pca = PCA(n_components=128, svd_solver="randomized", random_state=20261014)
        with threadpool_limits(limits=2):
            pca.fit(values[source])
            components[name] = pca.transform(values).astype(np.float32)
        del values
    idx = np.arange(len(selected))
    base = design_block(components["base"], idx)
    expert = design_block((components["general"], components["domain"]), idx)
    with threadpool_limits(limits=2):
        prior_b = base @ fit_head(components["base"], labels, source)
        prior_e = expert @ fit_head((components["general"], components["domain"]), labels, source)
    return labels, base, expert, prior_b, prior_e, source


def score(y, prob):
    pred = prob.argmax(1)
    oa = 100 * np.mean(pred == y)
    ba = 100 * np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])
    brier = np.mean(np.sum((prob - np.eye(8)[y]) ** 2, axis=1))
    return oa, ba, brier, pred


def prediction(q, s, y, base, expert, prior_b, prior_e, delta, mode, width):
    qb, sb = base[q], base[s]
    qz, sz = expert[q], expert[s]
    base_score = prior_b[q] + (qb @ sb.T) @ coefficients(sb @ sb.T, prior_b[s], y[s], 8.)
    if mode.startswith("rbf"):
        ss = 32 * np.exp(-cdist(sz, sz, "sqeuclidean") / width)
        moved = qz - delta if mode != "rbf_raw" else qz
        qs = 32 * np.exp(-cdist(moved, sz, "sqeuclidean") / width)
    else:
        ss = sz @ sz.T
        qs = (qz - delta) @ sz.T if mode != "dot_raw" else qz @ sz.T
    expert_score = prior_e[q] + qs @ coefficients(ss, prior_e[s], y[s], 8.)
    return .5 * (softmax(base_score, axis=1) + softmax(expert_score, axis=1)), expert_score


def map_at_r(sim, query_labels, gallery_labels):
    results = []
    for i, truth in enumerate(query_labels):
        relevant = gallery_labels == truth
        n = int(relevant.sum())
        if n == 0:
            continue
        hits = relevant[np.argsort(-sim[i])[:n]]
        results.append((int(truth), float(np.sum(np.cumsum(hits) / np.arange(1, n + 1) * hits) / n)))
    frame = pd.DataFrame(results, columns=["class_id", "ap_at_r"])
    return float(frame.ap_at_r.mean()), float(frame.groupby("class_id").ap_at_r.mean().mean())


def balanced_offset(expert, a, b, ay, by):
    classes = [c for c in range(8) if np.sum(ay == c) >= 3 and np.sum(by == c) >= 3]
    if not classes:
        return np.r_[expert[b, :-1].mean(0) - expert[a, :-1].mean(0), 0.], 0
    pieces = [expert[b[by == c], :-1].mean(0) - expert[a[ay == c], :-1].mean(0)
              for c in classes]
    return np.r_[np.mean(pieces, axis=0), 0.], len(classes)


def main():
    global FOLD, PORT, SOURCE, EXTRA
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default=FOLD)
    args = parser.parse_args()
    FOLD = args.fold
    episode = frozen_episode()
    PORT = episode["port"]
    root = Path(episode["folder"]).parent
    SOURCE = root / PORT.replace(" ", "_")
    EXTRA = HERE / ("singapore_extra" if FOLD == "Singapore__S1A_060968_079865"
                    else FOLD + "_extra")
    print("strict fold", FOLD, flush=True)
    candidates = clean_cohorts(episode)
    data = load(PORT, FOLD, root)
    query_rows = pd.read_csv(Path(episode["folder"]) / "query.csv").set_index("sample_id")
    query_global = query_rows.loc[episode["query"], "row_index"].to_numpy(int)
    source_global = data["rows"][np.r_[data["train"], data["val"]]]
    selected = np.unique(np.r_[source_global, query_global, candidates.row_index.to_numpy(int)])
    old_rows = data["rows"]
    missing = np.setdiff1d(selected, old_rows, assume_unique=True)
    known = old_rows[:4]
    extra_rows = np.unique(np.r_[missing, known])
    print("missing candidate features", len(missing), "reference rows", len(known), flush=True)
    encode_missing(extra_rows)
    for name, original in (("zero", data["zero"]), ("c4", data["c4"]),
                           ("G18", data["raw"]["G18"]), ("D18", data["raw"]["D18"])):
        observed = np.load(EXTRA / (name + ".npy"), mmap_mode="r")
        pos = np.searchsorted(extra_rows, known)
        baseline = np.asarray(original[np.searchsorted(old_rows, known)])
        max_error = float(np.max(np.abs(observed[pos] - baseline)))
        print("encoder replay", name, max_error, flush=True)
        assert max_error < .05, (name, max_error)
    labels, base, expert, prior_b, prior_e, source = feature_model(data, selected)
    local = lambda rows: np.searchsorted(selected, rows)
    q = local(query_global)
    assert np.array_equal(selected[q], query_global)
    a_global = candidates.loc[candidates.support_origin.eq("cross"), "row_index"].to_numpy(int)
    b_global = candidates.loc[candidates.support_origin.eq("same"), "row_index"].to_numpy(int)
    a, b = local(a_global), local(b_global)
    assert np.array_equal(selected[a], a_global) and np.array_equal(selected[b], b_global)
    offset = np.r_[expert[b, :-1].mean(0) - expert[a, :-1].mean(0), 0.]
    by_origin = {name: frame.set_index("mmsi") for name, frame in
                 candidates.groupby("support_origin")}
    shared = sorted(set(by_origin["same"].index) & set(by_origin["cross"].index))
    if shared:
        matched_a = local(by_origin["cross"].loc[shared, "row_index"].to_numpy(int))
        matched_b = local(by_origin["same"].loc[shared, "row_index"].to_numpy(int))
        matched_offset = expert[matched_b, :-1].mean(0) - expert[matched_a, :-1].mean(0)
        offset_cosine = float(np.dot(offset[:-1], matched_offset) /
                              (np.linalg.norm(offset[:-1]) * np.linalg.norm(matched_offset)))
    else:
        matched_offset = np.zeros(expert.shape[1] - 1)
        offset_cosine = None
    print("matched diagnostic", len(shared), "offset cosine",
          round(offset_cosine, 4) if offset_cosine is not None else None,
          "norms", round(np.linalg.norm(offset[:-1]), 3),
          round(np.linalg.norm(matched_offset), 3), flush=True)
    pseudo_a, pseudo_b = prior_b[a].argmax(1), prior_b[b].argmax(1)
    pseudo_offset, pseudo_classes = balanced_offset(expert, a, b, pseudo_a, pseudo_b)
    true_offset, true_classes = balanced_offset(expert, a, b, labels[a], labels[b])
    print("cohort pseudo accuracy", round(np.mean(pseudo_a == labels[a]), 3),
          round(np.mean(pseudo_b == labels[b]), 3), "balanced classes", pseudo_classes,
          true_classes, flush=True)
    source_sample = np.random.default_rng(20261015).choice(source, min(512, len(source)), replace=False)
    width = float(np.median(pdist(expert[source_sample], "sqeuclidean")))
    assert width > 0
    print("RBF source-only width", round(width, 4), flush=True)
    gallery_labels = labels[a]
    raw_retrieval = map_at_r(expert[q] @ expert[a].T, labels[q], gallery_labels)
    moved_retrieval = map_at_r((expert[q] - offset) @ expert[a].T, labels[q], gallery_labels)
    print("cross-scene MAP@R micro/macro", raw_retrieval, moved_retrieval, flush=True)
    rows, predictions = [], []
    for draw, pair in enumerate(episode["draws"]):
        candidate_ids = candidates.set_index("sample_id")
        s_global = candidate_ids.loc[pair["cross"], "row_index"].to_numpy(int)
        s = local(s_global)
        assert np.array_equal(selected[s], s_global) and len(s) == 16
        modes = ["dot_raw", "dot_bias", "dot_half", "dot_pseudo", "dot_truebal",
                 "rbf_raw", "rbf_shift", "rbf_half"]
        if len(shared) >= 3:
            modes += ["dot_matched", "rbf_matched"]
        for mode in modes:
            shift = (pseudo_offset if mode == "dot_pseudo" else
                     true_offset if mode == "dot_truebal" else
                     np.r_[matched_offset, 0.] if mode.endswith("matched") else
                     .5 * offset if mode.endswith("half") else offset)
            prob, expert_score = prediction(q, s, labels, base, expert, prior_b, prior_e,
                                             shift, mode, width)
            oa, ba, brier, pred = score(labels[q], prob)
            rows.append(dict(fold=FOLD, draw=draw, origin="cross", mode=mode, n=len(q),
                             cohort_a=len(a), cohort_b=len(b), oa=oa, ba=ba,
                             brier=brier, map_at_r_raw=raw_retrieval[0],
                             map_at_r_shift=moved_retrieval[0]))
            predictions.extend(dict(draw=draw, origin="cross", mode=mode, sample_id=sid, truth=int(truth),
                                    pred=int(value), true_probability=float(prob[i, truth]))
                               for i, (sid, truth, value) in enumerate(zip(episode["query"], labels[q], pred)))
            print(draw, mode, round(oa, 3), round(ba, 3), round(brier, 4), flush=True)
            if mode == "rbf_raw":
                raw_expert = expert_score
            elif mode == "rbf_shift":
                print("RBF shift score query-varying std", float(np.mean(np.std(expert_score - raw_expert, axis=0))), flush=True)
        if draw == 0:
            reference = pd.read_csv(CONTEXT / "target_context_pilot.csv")
            line = reference.loc[(reference.fold == FOLD) & (reference.origin == "cross") &
                                 (reference.draw == 0) & (reference.arm == "plain")].iloc[0]
            actual = rows[0]
            np.testing.assert_allclose([actual["oa"], actual["ba"], actual["brier"]],
                                       [line.oa, line.ba, line.brier], atol=1e-6)
        same_global = candidate_ids.loc[pair["same"], "row_index"].to_numpy(int)
        same_support = local(same_global)
        for mode in ("dot_raw", "rbf_raw"):
            prob, _ = prediction(q, same_support, labels, base, expert, prior_b, prior_e,
                                 offset, mode, width)
            oa, ba, brier, pred = score(labels[q], prob)
            rows.append(dict(fold=FOLD, draw=draw, origin="same", mode=mode, n=len(q),
                             cohort_a=len(a), cohort_b=len(b), oa=oa, ba=ba, brier=brier,
                             map_at_r_raw=raw_retrieval[0], map_at_r_shift=moved_retrieval[0]))
            predictions.extend(dict(draw=draw, origin="same", mode=mode, sample_id=sid,
                                    truth=int(truth), pred=int(value),
                                    true_probability=float(prob[i, truth]))
                               for i, (sid, truth, value) in enumerate(zip(episode["query"], labels[q], pred)))
            print(draw, "same", mode, round(oa, 3), round(ba, 3), round(brier, 4), flush=True)
    pd.DataFrame(rows).to_csv(HERE / f"strict_scene_metrics_{FOLD}.csv", index=False)
    pd.DataFrame(predictions).to_csv(HERE / f"strict_scene_predictions_{FOLD}.csv", index=False)
    (HERE / f"strict_scene_inventory_{FOLD}.json").write_text(json.dumps(dict(
        fold=FOLD, source_rows=len(source), query_rows=len(q), cohort_a=len(a), cohort_b=len(b),
        missing_encoded=len(missing), matched_cohort_ships=len(shared),
        unpaired_matched_shift_cosine=offset_cosine, rbf_width=width,
        pseudo_accuracy_a=float(np.mean(pseudo_a == labels[a])),
        pseudo_accuracy_b=float(np.mean(pseudo_b == labels[b])),
        pseudo_balanced_classes=pseudo_classes, true_balanced_classes=true_classes,
        retrieval_raw=raw_retrieval,
        retrieval_shift=moved_retrieval), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
