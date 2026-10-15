"""Re-evaluate the frozen C4 dual readout with early-layer support evidence."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
CONTEXT = HERE.parent / "context_condition_study_20261015"
sys.path.insert(0, str(CONTEXT))
from target_context_pilot import build_model, class_map, infer, load, score  # noqa: E402
from source_context_pilot import coefficients  # noqa: E402

MANIFEST = (HERE.parent / "visual_reliable_baseline/artifacts/"
            "eight_class_adaptive_20260916/manifest.csv")
LOCAL = HERE / "localization"


def source_gallery(folder):
    train = pd.read_csv(folder / "source_train.csv")
    rng = np.random.default_rng(20261015)
    rows = []
    for _, group in train.groupby(["port", "class_id"], sort=True):
        rows.extend(rng.choice(group.row_index.to_numpy(int),
                               min(8, len(group)), replace=False).tolist())
    return np.asarray(rows, int)


def local_vectors(fold, folder, selected, data):
    with np.load(LOCAL / f"features_{fold}.npz") as cache:
        rows = cache["rows"].astype(int)
        gallery = source_gallery(folder)
        gallery_local = np.searchsorted(rows, gallery)
        np.testing.assert_array_equal(rows[gallery_local], gallery)
        selected_global = data["rows"][selected]
        present = np.isin(selected_global, rows)
        local = np.searchsorted(rows, selected_global[present])
        np.testing.assert_array_equal(rows[local], selected_global[present])
        vectors = {}
        names = ["gap", "stem_gap", "layer2_peak", "layer2_ship_peak",
                 "full_peak", "ship_peak", "near_ring_mean", "near_contrast_peak",
                 "far_contrast_peak", "raw_ship", "raw_ring", "raw_contrast"] + [f"background_peak_{i}" for i in range(3)] + [
                 f"layer2_background_peak_{i}" for i in range(3)]
        with threadpool_limits(limits=2):
            for name in names:
                width = min(64, cache[name].shape[1])
                pca = PCA(n_components=width, svd_solver="randomized", random_state=20261015)
                pca.fit(cache[name][gallery_local])
                value = pca.transform(cache[name][local]).astype(np.float32)
                value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
                aligned = np.zeros((len(selected), width), dtype=np.float32)
                aligned[present] = value
                vectors[name] = aligned
    return vectors, present


def shuffled_vectors(local, held_ids, held_positions, folder):
    acquisition = pd.concat([
        pd.read_csv(folder / "query.csv", usecols=["sample_id", "acquisition"]),
        pd.read_csv(folder / "support_candidates.csv", usecols=["sample_id", "acquisition"]),
    ]).drop_duplicates("sample_id").set_index("sample_id").loc[held_ids]
    shuffled = {}
    for seed in range(3):
        rng = np.random.default_rng(20261020 + seed)
        copies = {name: local[name].copy()
                  for name in ("full_peak", "ship_peak", "layer2_peak",
                               "layer2_ship_peak", "near_ring_mean", "raw_ring")}
        for _, group in acquisition.groupby("acquisition", sort=True):
            positions = held_positions[acquisition.index.get_indexer(group.index)]
            assigned = rng.permutation(positions)
            for name, value in copies.items():
                value[positions] = local[name][assigned]
        for name, value in copies.items():
            shuffled[f"{name}_shuffled_{seed}"] = value
    return shuffled


def infer_with_local(prior_b, prior_e, base, expert, q, s, labels,
                     local=None, fraction=0., relation=None, weights=None, placement="both",
                     return_parts=False):
    sb, qb = base[s], base[q]
    sz, qz = expert[s], expert[q]
    base_score = prior_b[q] + (qb @ sb.T) @ coefficients(
        sb @ sb.T, prior_b[s], labels[s], 8.)
    ss, qs = sz @ sz.T, qz @ sz.T
    if local is not None and fraction:
        assert expert.shape[1] == 257
        old_support = sz[:, 128:256]
        old_query = qz[:, 128:256]
        if placement in ("both", "support_only"):
            ss += fraction * (16. * (local[s] @ local[s].T) -
                              old_support @ old_support.T)
        if placement in ("both", "query_only"):
            qs += fraction * (16. * (local[q] @ local[s].T) -
                              old_query @ old_support.T)
    if relation is not None:
        sr, qr = relation[s], relation[q]
        multiplier = np.ones(sr.shape[1]) if weights is None else weights
        ss = .75 * ss + .25 * ((sr * multiplier) @ sr.T)
        qs = .75 * qs + .25 * ((qr * multiplier) @ sr.T)
    expert_score = prior_e[q] + qs @ coefficients(ss, prior_e[s], labels[s], 8.)
    pb, pe = softmax(base_score, axis=1), softmax(expert_score, axis=1)
    result = .5 * (pb + pe)
    return (result, pb, pe) if return_parts else result


def support_vote(baseline, local, q, s, labels, fraction):
    classes = np.unique(labels[s])
    similarity = local[q] @ local[s].T
    class_mean = np.stack([similarity[:, labels[s] == cls].mean(1)
                           for cls in classes], axis=1)
    vote = softmax(8. * class_mean, axis=1)
    probability = baseline.copy()
    supported_mass = baseline[:, classes].sum(axis=1, keepdims=True)
    probability[:, classes] = ((1. - fraction) * baseline[:, classes] +
                               fraction * supported_mass * vote)
    return probability


def main():
    episodes = json.loads((CONTEXT / "target_episodes.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST, usecols=["instance_id"])
    archive = pd.read_csv(CONTEXT / "target_context_pilot.csv")
    rows, predictions = [], []
    for episode in episodes:
        fold, port = episode["fold"], episode["port"]
        folder = Path(episode["folder"])
        data = load(port, fold, folder.parent)
        feature_ids = manifest.instance_id.to_numpy(str)[data["rows"]]
        lookup = pd.Series(np.arange(len(feature_ids)), index=feature_ids)
        held_ids = sorted({sample for sample in episode["query"] + [
            item for draw in episode["draws"] for origin in ("same", "cross")
            for item in draw[origin]]})
        held = lookup.loc[held_ids].to_numpy(int)
        source = np.r_[data["train"], data["val"]]
        assert not set(source) & set(held)
        selected = np.sort(np.unique(np.r_[source, held, data["query"],
                                      data["support"][("same", 0)]]))
        labels, base, expert, prior_b, prior_e = build_model(data, selected)
        local, present = local_vectors(fold, folder, selected, data)
        held_positions = np.searchsorted(selected, held)
        np.testing.assert_array_equal(selected[held_positions], held)
        local.update(shuffled_vectors(local, held_ids, held_positions, folder))
        local_train = np.searchsorted(selected, data["train"])
        relation = class_map(expert[:, :-1], labels, local_train)
        relation_weights = np.asarray(json.loads((folder / "full_fit.json").read_text(
            encoding="utf-8"))["relation_weights"])
        query_ids = episode["query"]
        query = np.searchsorted(selected, lookup.loc[query_ids].to_numpy(int))
        assert np.array_equal(selected[query], lookup.loc[query_ids].to_numpy(int))
        for draw_id, draw in enumerate(episode["draws"]):
            for origin in ("same", "cross"):
                support = np.searchsorted(selected, lookup.loc[draw[origin]].to_numpy(int))
                assert len(support) == 16 and present[np.r_[query, support]].all()
                for relation_name, relation_map, weights in (
                    ("plain", None, None), ("class_fixed", relation, None),
                    ("class_learned", relation, relation_weights)):
                    reference = infer(prior_b, prior_e, base, expert, None, 1.,
                                      query, support, labels, 0., relation_map, weights)
                    replay, baseline_pb, baseline_pe = infer_with_local(
                        prior_b, prior_e, base, expert, query, support, labels,
                        relation=relation_map, weights=weights, return_parts=True)
                    np.testing.assert_allclose(replay, reference, rtol=1e-12, atol=1e-12)
                    truth = labels[query]
                    baseline_pred = replay.argmax(1)
                    arms = [
                        ("original", None, 0., "both"),
                        ("late_gap64_025", local["gap"], .25, "both"),
                        ("stem_gap_025", local["stem_gap"], .25, "both"),
                        ("layer2_peak_025", local["layer2_peak"], .25, "both"),
                        ("layer2_ship_peak_025", local["layer2_ship_peak"], .25, "both"),
                        ("full_peak_025", local["full_peak"], .25, "both"),
                        ("full_peak_050", local["full_peak"], .50, "both"),
                        ("full_peak_query_only_025", local["full_peak"], .25, "query_only"),
                        ("full_peak_support_only_025", local["full_peak"], .25, "support_only"),
                        ("ship_peak_025", local["ship_peak"], .25, "both"),
                        ("ship_peak_050", local["ship_peak"], .50, "both"),
                        ("ship_peak_query_only_025", local["ship_peak"], .25, "query_only"),
                        ("ship_peak_support_only_025", local["ship_peak"], .25, "support_only"),
                        ("near_ring_mean_025", local["near_ring_mean"], .25, "both"),
                        ("near_contrast_peak_025", local["near_contrast_peak"], .25, "both"),
                        ("far_contrast_peak_025", local["far_contrast_peak"], .25, "both"),
                        ("raw_ship_025", local["raw_ship"], .25, "both"),
                        ("raw_ring_025", local["raw_ring"], .25, "both"),
                        ("raw_contrast_025", local["raw_contrast"], .25, "both"),
                        *((f"background_peak_{i}_025", local[f"background_peak_{i}"],
                           .25, "both")
                          for i in range(3)),
                        *((f"layer2_background_peak_{i}_025",
                           local[f"layer2_background_peak_{i}"], .25, "both")
                          for i in range(3))]
                    for kind in ("full_peak", "ship_peak", "gap", "background_peak_0",
                                 "background_peak_1", "background_peak_2"):
                        for fraction in (.10, .25):
                            arms.append((f"vote_{kind}_{int(100 * fraction):03d}",
                                         local[kind], fraction, "vote"))
                    for kind in ("full_peak", "ship_peak", "layer2_peak",
                                 "layer2_ship_peak", "near_ring_mean", "raw_ring"):
                        for seed in range(3):
                            name = f"{kind}_shuffled_{seed}"
                            arms.append((name + "_025", local[name], .25, "both"))
                    for name, vector, fraction, placement in arms:
                        if placement == "vote":
                            prob = support_vote(replay, vector, query, support, labels, fraction)
                            base_prob, expert_prob = baseline_pb, baseline_pe
                        else:
                            prob, base_prob, expert_prob = infer_with_local(
                                prior_b, prior_e, base, expert, query, support, labels,
                                vector, fraction, relation_map, weights, placement, True)
                        pred = prob.argmax(1)
                        oa, ba, brier = score(truth, prob)
                        if vector is None:
                            match = archive.loc[
                                archive.fold.eq(fold) & archive.draw.eq(draw_id) &
                                archive.origin.eq(origin) & archive.arm.eq(relation_name)].iloc[0]
                            np.testing.assert_allclose([oa, ba, brier],
                                                       [match.oa, match.ba, match.brier],
                                                       rtol=1e-10, atol=1e-10)
                        arm = relation_name + "_" + name
                        rows.append(dict(fold=fold, port=port, draw=draw_id,
                                         origin=origin, arm=arm, n=len(truth),
                                         query_classes=len(np.unique(truth)),
                                         oa=oa, ba=ba, brier=brier,
                                         rescued=int(np.sum((baseline_pred != truth) & (pred == truth))),
                                         harmed=int(np.sum((baseline_pred == truth) & (pred != truth)))))
                        predictions.extend(dict(fold=fold, port=port, draw=draw_id,
                                                origin=origin, arm=arm, sample_id=sample_id,
                                                truth=int(y), pred=int(p), correct=bool(y == p),
                                                true_probability=float(prob[i, y]),
                                                base_pred=int(base_prob[i].argmax()),
                                                expert_pred=int(expert_prob[i].argmax()),
                                                expert_true_probability=float(expert_prob[i, y]))
                                           for i, (sample_id, y, p) in enumerate(
                                               zip(query_ids, truth, pred)))
        print("completed", fold, flush=True)
        out = HERE / "full_readout"
        out.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
    result = pd.DataFrame(rows)
    print(result.groupby(["origin", "arm"])[["oa", "ba", "brier", "rescued", "harmed"]]
          .mean().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
