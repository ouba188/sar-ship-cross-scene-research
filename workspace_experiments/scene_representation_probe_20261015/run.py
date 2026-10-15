"""Source-only pair-metric probe for cross-acquisition ship evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.special import softmax
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from sys import path as sys_path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SCENE = PROJECT / "scene_cross_dataset_redesign_20261014"
CONTEXT = PROJECT / "context_condition_study_20261015"
sys_path.insert(0, str(SCENE / "ports_full_expansion"))
sys_path.insert(0, "E:/Hermes/paper_tracking/sensitivity/oof_reliability_check")
from evaluate import load  # noqa: E402
from general_domain_readout import design_block, fit_head  # noqa: E402
from unified_head_core import coefficients  # noqa: E402


def local_rows(data, frame):
    rows = frame.row_index.to_numpy(int)
    loc = np.searchsorted(data["rows"], rows)
    np.testing.assert_array_equal(data["rows"][loc], rows)
    return loc


def make_features(data):
    source = np.r_[data["train"], data["val"]]
    blocks = {}
    with threadpool_limits(limits=2):
        for name in ("C4", "G50", "D18"):
            raw = data["c4"] if name == "C4" else data["raw"][name]
            pca = PCA(n_components=128, svd_solver="randomized", random_state=20261014)
            pca.fit(raw[source])
            blocks[name] = pca.transform(raw).astype(np.float32)
    expert = (blocks["G50"], blocks["D18"])
    return blocks["C4"], expert, design_block(expert, np.arange(len(data["rows"])))


def sample_pairs(frame, count, scene_aware, within_port, rng):
    def values(rows):
        return rows["local"].to_numpy(int), rows["mmsi"].to_numpy(str)

    by_class = {int(c): values(group) for c, group in frame.groupby("class_id")}
    by_class_scene = {int(c): {name: values(group) for name, group in rows.groupby("acquisition")}
                      for c, rows in frame.groupby("class_id")}
    by_port_class = {int(c): {port: {name: values(group) for name, group in port_rows.groupby("acquisition")}
                               for port, port_rows in rows.groupby("port")
                               if port_rows.acquisition.nunique() >= 2 and port_rows.mmsi.nunique() >= 2}
                     for c, rows in frame.groupby("class_id")}
    by_scene = {name: {int(c): values(group) for c, group in rows.groupby("class_id")}
                for name, rows in frame.groupby("acquisition") if rows.class_id.nunique() >= 2}
    scene_names = list(by_scene)
    classes = np.array(sorted(by_class))
    pos, neg = [], []
    for _ in range(count):
        c = int(rng.choice(classes))
        while True:
            if scene_aware:
                groups = (by_port_class[c][rng.choice(list(by_port_class[c]))]
                          if within_port else by_class_scene[c])
                scenes = list(groups)
                first, second = rng.choice(scenes, size=2, replace=False)
                left, right = groups[first], groups[second]
            else:
                left = right = by_class[c]
            ia, ib = rng.integers(len(left[0])), rng.integers(len(right[0]))
            if left[1][ia] != right[1][ib]:
                pos.append((left[0][ia], right[0][ib]))
                break
        if scene_aware:
            scene = by_scene[scene_names[rng.integers(len(scene_names))]]
            first, second = rng.choice(list(scene), size=2, replace=False)
            left, right = scene[int(first)], scene[int(second)]
        else:
            first, second = rng.choice(classes, size=2, replace=False)
            left, right = by_class[int(first)], by_class[int(second)]
        neg.append((left[0][rng.integers(len(left[0]))],
                    right[0][rng.integers(len(right[0]))]))
    return np.asarray(pos), np.asarray(neg)


def metric(z, frame, scene_aware, within_port, seed):
    rng = np.random.default_rng(seed)
    positive, negative = sample_pairs(frame, 4096, scene_aware, within_port, rng)
    p = z[positive[:, 0]] - z[positive[:, 1]]
    n = z[negative[:, 0]] - z[negative[:, 1]]
    positive_cov = p.T @ p / len(p)
    negative_cov = n.T @ n / len(n)
    ridge = 0.5 * np.trace(positive_cov) / z.shape[1]
    with threadpool_limits(limits=2):
        _, vectors = eigh(negative_cov, positive_cov + ridge * np.eye(z.shape[1]),
                          subset_by_index=(z.shape[1] - 128, z.shape[1] - 1))
    update = vectors @ vectors.T
    update *= z.shape[1] / np.trace(update)
    transform = 0.5 * (np.eye(z.shape[1]) + update)
    return transform.astype(np.float32), float(ridge)


def map_at_r(query, gallery, query_labels, gallery_labels):
    similarity = query @ gallery.T
    scores = []
    for i, truth in enumerate(query_labels):
        relevant = gallery_labels == truth
        n = int(relevant.sum())
        if n == 0:
            continue
        hits = relevant[np.argsort(-similarity[i])[:n]]
        scores.append((int(truth), float(np.sum(np.cumsum(hits) /
                         np.arange(1, n + 1) * hits) / n)))
    per_class = pd.DataFrame(scores, columns=["class_id", "ap"])
    return float(per_class.ap.mean()), float(per_class.groupby("class_id").ap.mean().mean())


def port_map_at_r(representation, train, validation, labels):
    scores = []
    for port in sorted(set(train.port) & set(validation.port)):
        q = validation.loc[validation.port.eq(port), "local"].to_numpy(int)
        s = train.loc[train.port.eq(port), "local"].to_numpy(int)
        if len(q) and len(s):
            score = map_at_r(representation[q, :-1], representation[s, :-1],
                             labels[q], labels[s])[0]
            if np.isfinite(score):
                scores.append(score)
    return float(np.mean(scores))


def evaluate(data, episode, c4, components, representation, gallery, validation,
             source_train, source_val, reference=None):
    labels = data["labels"]
    source = np.r_[data["train"], data["val"]]
    base = design_block(c4, np.arange(len(labels)))
    prior_b = base @ fit_head(c4, labels, source)
    prior_e = representation @ fit_head(components, labels, source)
    q = local_rows(data, pd.read_csv(episode["folder"] + "/query.csv"))
    eligible = pd.Index(pd.read_csv(episode["folder"] + "/query.csv").sample_id)
    q = q[eligible.isin(episode["query"])]
    support_ids = pd.read_csv(episode["folder"] + "/support_candidates.csv").set_index("sample_id")
    cross_ids = sorted(set(sum((draw["cross"] for draw in episode["draws"]), [])))
    cross = local_rows(data, support_ids.loc[cross_ids])
    retrieval = map_at_r(representation[q, :-1], representation[cross, :-1],
                         labels[q], labels[cross])
    source_retrieval = map_at_r(representation[validation, :-1],
                                representation[gallery, :-1], labels[validation], labels[gallery])
    source_port_map = port_map_at_r(representation, source_train, source_val, labels)
    rows = []
    for draw, pair in enumerate(episode["draws"]):
        for origin in ("same", "cross"):
            s = local_rows(data, support_ids.loc[pair[origin]])
            qb, sb = base[q], base[s]
            b = prior_b[q] + (qb @ sb.T) @ coefficients(sb @ sb.T, prior_b[s], labels[s], 8.)
            variants = [("full", representation, prior_e)]
            if reference is not None:
                old_representation, old_prior = reference
                variants += [("prior_only", old_representation, prior_e),
                             ("kernel_only", representation, old_prior)]
            for mechanism, features, prior in variants:
                ze, se = features[q], features[s]
                e = prior[q] + (ze @ se.T) @ coefficients(se @ se.T, prior[s], labels[s], 8.)
                pred = (0.5 * (softmax(b, axis=1) + softmax(e, axis=1))).argmax(1)
                oa = 100 * np.mean(pred == labels[q])
                classes = np.unique(labels[q])
                class_accuracy = {int(c): float(np.mean(pred[labels[q] == c] == c))
                                  for c in classes}
                ba = 100 * np.mean(list(class_accuracy.values()))
                rows.append(dict(fold=episode["fold"], draw=draw, origin=origin,
                                 mechanism=mechanism, oa=oa, ba=ba,
                                 class_accuracy=json.dumps(class_accuracy, sort_keys=True),
                                 source_map=source_retrieval[0], source_macro_map=source_retrieval[1],
                                 source_port_map=source_port_map,
                                 target_map=retrieval[0], target_macro_map=retrieval[1], n=len(q)))
    return rows


def run(fold):
    episode = next(item for item in json.loads((CONTEXT / "target_episodes.json").read_text())
                   if item["fold"] == fold)
    root = Path(episode["folder"]).parent
    data = load(episode["port"], fold, root)
    c4, blocks, original = make_features(data)
    train = pd.read_csv(Path(episode["folder"]) / "source_train.csv", dtype={"mmsi": str})
    train["local"] = local_rows(data, train)
    train["mmsi"] = train.mmsi.astype(str)
    source_val = pd.read_csv(Path(episode["folder"]) / "source_validation.csv", dtype={"mmsi": str})
    source_val["local"] = local_rows(data, source_val)
    z = original[:, :-1].astype(np.float32)
    rng = np.random.default_rng(20261015)
    gallery = np.concatenate([rng.choice(data["train"][data["labels"][data["train"]] == c],
                                         min(128, np.sum(data["labels"][data["train"]] == c)),
                                         replace=False) for c in range(8)])
    validation = np.concatenate([rng.choice(data["val"][data["labels"][data["val"]] == c],
                                            min(128, np.sum(data["labels"][data["val"]] == c)),
                                            replace=False) for c in range(8)])
    output = []
    baseline = evaluate(data, episode, c4, blocks, original, gallery, validation,
                        train, source_val)
    source = np.r_[data["train"], data["val"]]
    original_prior = original @ fit_head(blocks, data["labels"], source)
    archive = pd.read_csv(PROJECT / "scene_core_probe_20261015" /
                          f"strict_scene_metrics_{fold}.csv")
    archive = archive.loc[archive["mode"].eq("dot_raw")].set_index(["draw", "origin"])
    for row in baseline:
        expected = archive.loc[(row["draw"], row["origin"])]
        np.testing.assert_allclose([row["oa"], row["ba"]],
                                   [expected.oa, expected.ba], atol=1e-6)
        output.append(dict(model="original", pair_ridge=np.nan, **row))
    print("archived baseline replay exact", fold, flush=True)
    for name, aware, within_port in (("generic", False, False),
                                     ("cross_scene", True, False),
                                     ("within_port_scene", True, True)):
        matrix, ridge = metric(z, train, aware, within_port, 20261015)
        projected = z @ matrix
        representation = design_block(projected, np.arange(len(projected)))
        for row in evaluate(data, episode, c4, projected, representation, gallery, validation,
                            train, source_val,
                            (original, original_prior) if name == "within_port_scene" else None):
            output.append(dict(model=name, pair_ridge=ridge, **row))
    HERE.mkdir(exist_ok=True)
    pd.DataFrame(output).to_csv(HERE / f"results_{fold}.csv", index=False)
    print(pd.DataFrame(output).query("mechanism == 'full'").groupby(["model", "origin"])[
          ["oa", "ba", "source_port_map", "target_map"]]
          .mean().round(3).to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fold")
    run(parser.parse_args().fold)
