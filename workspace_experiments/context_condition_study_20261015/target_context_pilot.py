"""Paired six-fold held-port evaluation on newly frozen far-eligible B16 episodes."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent / "scene_cross_dataset_redesign_20261014"
REGION = HERE.parent / "region_cue_replication_20261013"
sys.path.insert(0, str(SCENE / "ports_full_expansion"))
from evaluate import load  # noqa: E402
from source_context_pilot import get_descriptors, prep_context, infer  # noqa: E402
from source_context_pilot import design_block, fit_head  # noqa: E402
from run import class_map  # noqa: E402


def build_model(data, selected):
    source = np.r_[data["train"], data["val"]]
    local_source = np.searchsorted(selected, source)
    assert np.array_equal(selected[local_source], source)
    labels = data["labels"][selected]
    features = {}
    for name, raw in (("base", data["c4"]), ("general", data["raw"]["G50"]),
                      ("domain", data["raw"]["D18"])):
        values = np.asarray(raw[selected], dtype=np.float32)
        pca = PCA(n_components=128, svd_solver="randomized", random_state=20261014)
        with threadpool_limits(limits=2):
            pca.fit(values[local_source])
            features[name] = pca.transform(values).astype(np.float32)
        del values
    ids = np.arange(len(selected))
    base = design_block(features["base"], ids)
    expert = design_block((features["general"], features["domain"]), ids)
    with threadpool_limits(limits=2):
        prior_b = base @ fit_head(features["base"], labels, local_source)
        prior_e = expert @ fit_head((features["general"], features["domain"]), labels, local_source)
    return labels, base, expert, prior_b, prior_e


def score(y, prob):
    pred = prob.argmax(1)
    return (100 * np.mean(pred == y),
            100 * np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]),
            np.mean(np.sum((prob - np.eye(8)[y]) ** 2, axis=1)))


def main():
    episodes = json.loads((HERE / "target_episodes.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(
        HERE.parent / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv",
        usecols=["instance_id"])
    geometry = pd.read_csv(HERE / "geometry_by_sample.csv", usecols=["sample_id", "far_ok"])
    far_ok = set(geometry.loc[geometry.far_ok, "sample_id"].astype(str))
    cols = ["sample_id"] + [f"path_{p}" for p in ("vv", "vh")]
    for pol in ("vv", "vh"):
        cols += [f"bbox_{axis}_{pol}" for axis in ("x0", "y0", "x1", "y1")]
        cols += [f"native_h_{pol}", f"native_w_{pol}", f"resolved_width_{pol}"]
    metadata = pd.read_csv(REGION / "source_region_metadata.csv", usecols=cols)
    metadata = metadata.set_index("sample_id", verify_integrity=True)
    rows, predictions = [], []
    for episode in episodes:
        folder = Path(episode["folder"])
        fold, port = episode["fold"], episode["port"]
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
        local_train = np.searchsorted(selected, data["train"])
        relation = class_map(expert[:, :-1], labels, local_train)
        weights = np.asarray(json.loads((folder / "full_fit.json").read_text(
            encoding="utf-8"))["relation_weights"])
        query_old = np.searchsorted(selected, data["query"])
        support_old = np.searchsorted(selected, data["support"][("same", 0)])
        original = infer(prior_b, prior_e, base, expert, None, 1., query_old,
                         support_old, labels, 0.)
        old_oa, old_ba, _ = score(labels[query_old], original)
        old_table = pd.read_csv(folder / "full_metrics.csv")
        reference = old_table.loc[(old_table.origin == "same") & (old_table.draw == 0)
                                  & (old_table.arm == "c4_dual_plain")].iloc[0]
        np.testing.assert_allclose([old_oa, old_ba],
                                   [reference.chip_oa, reference.chip_ba], atol=1e-6)
        for arm, use_weights in (("c4_class_fixed", None),
                                 ("c4_class_learned", weights)):
            check = infer(prior_b, prior_e, base, expert, None, 1., query_old,
                          support_old, labels, 0., relation, use_weights)
            reference = old_table.loc[(old_table.origin == "same") & (old_table.draw == 0)
                                      & (old_table.arm == arm)].iloc[0]
            np.testing.assert_allclose(score(labels[query_old], check)[:2],
                                       [reference.chip_oa, reference.chip_ba], atol=1e-6)
        print("baseline reproduced", fold, round(old_oa, 3), round(old_ba, 3), flush=True)

        candidates = np.asarray([sample for sample in feature_ids[data["train"]]
                                 if sample in far_ok])
        rng = np.random.default_rng(20261017)
        train_context_ids = rng.choice(candidates, min(768, len(candidates)), replace=False).tolist()
        descriptor_ids = train_context_ids + held_ids
        assert len(descriptor_ids) == len(set(descriptor_ids))
        far_raw, geo_raw = get_descriptors(
            metadata, descriptor_ids, HERE / f"target_far_descriptors_{fold}.npz")
        train_mask = np.arange(len(descriptor_ids)) < len(train_context_ids)
        contexts = {
            "far": prep_context(far_raw, train_mask),
            "geometry": prep_context(geo_raw, train_mask),
            "crop": prep_context(geo_raw[:, [0, 1, 8, 9]], train_mask),
            "box": prep_context(geo_raw[:, [2, 3, 6, 7, 10, 11, 14, 15]], train_mask),
            "position": prep_context(geo_raw[:, [4, 5, 12, 13]], train_mask),
        }
        acquisition = pd.concat([
            pd.read_csv(folder / "query.csv", usecols=["sample_id", "acquisition"]),
            pd.read_csv(folder / "support_candidates.csv", usecols=["sample_id", "acquisition"]),
        ]).drop_duplicates("sample_id").set_index("sample_id").loc[held_ids]
        shuffled_far = contexts["far"][0].copy()
        shuffle = np.random.default_rng(20261018)
        held_positions = pd.Series(np.arange(len(held_ids)) + len(train_context_ids), index=held_ids)
        for _, group in acquisition.groupby("acquisition", sort=True):
            position = held_positions.loc[group.index].to_numpy(int)
            shuffled_far[position] = contexts["far"][0][shuffle.permutation(position)]
        contexts["far_shuffled"] = (shuffled_far, contexts["far"][1])
        context_lookup = pd.Series(np.arange(len(descriptor_ids)), index=descriptor_ids)
        query_ids = episode["query"]
        query = np.searchsorted(selected, lookup.loc[query_ids].to_numpy(int))
        assert np.array_equal(selected[query], lookup.loc[query_ids].to_numpy(int))
        for draw_id, draw in enumerate(episode["draws"]):
            for origin in ("same", "cross"):
                support_ids = draw[origin]
                support = np.searchsorted(selected, lookup.loc[support_ids].to_numpy(int))
                assert len(support) == 16
                positions = np.r_[query, support]
                descriptor_positions = context_lookup.loc[
                    feature_ids[selected[positions]]].to_numpy(int)
                plain = None
                for arm, kind, alpha in (("plain", None, 0.), ("far_025", "far", .25),
                                         ("geometry_025", "geometry", .25),
                                         ("crop_025", "crop", .25),
                                         ("box_025", "box", .25),
                                         ("class_fixed", None, 0.),
                                         ("class_fixed_far_025", "far", .25),
                                         ("class_fixed_geometry_025", "geometry", .25),
                                         ("class_learned", None, 0.),
                                         ("class_learned_far_025", "far", .25),
                                         ("class_learned_far_shuffled_025", "far_shuffled", .25),
                                         ("class_learned_geometry_025", "geometry", .25),
                                         ("class_learned_crop_025", "crop", .25),
                                         ("class_learned_box_025", "box", .25),
                                         ("class_learned_position_025", "position", .25)):
                    if kind is None:
                        context, width = None, 1.
                    else:
                        descriptor, width = contexts[kind]
                        context = np.zeros((len(selected), descriptor.shape[1]), dtype=np.float32)
                        context[positions] = descriptor[descriptor_positions]
                    use_relation = relation if arm.startswith("class_") else None
                    use_weights = weights if arm.startswith("class_learned") else None
                    prob = infer(prior_b, prior_e, base, expert, context, width,
                                 query, support, labels, alpha, use_relation, use_weights)
                    truth = labels[query]
                    pred = prob.argmax(1)
                    if plain is None:
                        plain = pred
                    oa, ba, brier = score(truth, prob)
                    rows.append(dict(fold=fold, port=port, draw=draw_id, origin=origin,
                                     arm=arm, n=len(query), query_classes=len(np.unique(truth)),
                                     oa=oa, ba=ba, brier=brier,
                                     rescued=int(np.sum((plain != truth) & (pred == truth))),
                                     harmed=int(np.sum((plain == truth) & (pred != truth)))))
                    predictions.extend(dict(fold=fold, port=port, draw=draw_id,
                                            origin=origin, arm=arm, sample_id=sample_id,
                                            truth=int(y), pred=int(p), correct=bool(y == p),
                                            true_probability=float(prob[i, y]))
                                       for i, (sample_id, y, p) in enumerate(zip(query_ids, truth, pred)))
        print("completed", fold, flush=True)
        pd.DataFrame(rows).to_csv(HERE / "target_context_pilot.csv", index=False)
        pd.DataFrame(predictions).to_csv(HERE / "target_context_predictions.csv", index=False)
    result = pd.DataFrame(rows)
    print(result.groupby(["origin", "arm"])[["oa", "ba", "brier", "rescued", "harmed"]]
          .mean().round(4).to_string(), flush=True)
    print(result.groupby(["port", "origin", "arm"])[["oa", "ba", "brier"]]
          .mean().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
