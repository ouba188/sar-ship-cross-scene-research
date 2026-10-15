"""Source-validation feasibility test for native far-context support relations.

This is a development diagnostic, not an independent held-target result.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent / "scene_cross_dataset_redesign_20261014"
REGION = HERE.parent / "region_cue_replication_20261013"
sys.path.insert(0, str(SCENE / "ports_full_expansion"))
sys.path.insert(0, str(SCENE / "public_full_20261014"))
sys.path.insert(0, str(REGION))
from evaluate import load  # noqa: E402
from run import design_block, fit_head, coefficients  # noqa: E402
from source_region_metadata import bbox_regions  # noqa: E402

FOLD = "Singapore__S1A_060968_079865"
ROOT = SCENE / "ports_full_expansion"
SOURCE_META = REGION / "source_region_metadata.csv"


def descriptors(row):
    appearance, geometry = [], []
    for pol in ("vv", "vh"):
        image = np.asarray(Image.open(row[f"path_{pol}"]).convert("L"), dtype=np.float32)
        h, w = image.shape
        bounds = [row[f"bbox_{axis}_{pol}"] for axis in ("x0", "y0", "x1", "y1")]
        box, _, valid, _ = bbox_regions(bounds, (h, w), .35)
        assert valid and h == row[f"native_h_{pol}"] and w == row[f"native_w_{pol}"]
        mask = np.ones((h, w), dtype=bool)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = False
        pixels = image[mask]
        assert len(pixels) >= 64
        q = np.quantile(pixels, [.1, .5, .9])
        dx = np.abs(np.diff(image, axis=1))[mask[:, :-1] & mask[:, 1:]]
        dy = np.abs(np.diff(image, axis=0))[mask[:-1, :] & mask[1:, :]]
        appearance.extend([pixels.mean(), pixels.std(), *q, (pixels > 200).mean(),
                           dx.mean() if len(dx) else 0., dy.mean() if len(dy) else 0.])
        bx0, by0, bx1, by1 = bounds
        geometry.extend([h / 256, w / 256, (bx1-bx0) / w, (by1-by0) / h,
                         (bx0+bx1) / (2*w), (by0+by1) / (2*h), mask.mean(),
                         float(row[f"resolved_width_{pol}"]) / max(h, w)])
    return appearance, geometry


def get_descriptors(meta, ids, cache=None):
    cache = HERE / "source_far_descriptors.npz" if cache is None else Path(cache)
    if cache.exists():
        with np.load(cache) as saved:
            if set(ids).issubset(set(saved["ids"].astype(str))):
                index = pd.Index(saved["ids"].astype(str))
                pos = index.get_indexer(ids)
                return saved["far"][pos], saved["geometry"][pos]
    far, geo = [], []
    for i, sample_id in enumerate(ids):
        a, g = descriptors(meta.loc[sample_id])
        far.append(a)
        geo.append(g)
        if (i + 1) % 250 == 0:
            print("native descriptors", i + 1, "/", len(ids), flush=True)
    far, geo = np.asarray(far, np.float32), np.asarray(geo, np.float32)
    np.savez_compressed(cache, ids=np.asarray(ids), far=far, geometry=geo)
    return far, geo


def prep_context(x, train_mask):
    reference = x[train_mask]
    center = np.median(reference, axis=0)
    scale = np.quantile(reference, .75, axis=0) - np.quantile(reference, .25, axis=0)
    scaled = np.clip((x - center) / np.maximum(scale, .02), -5, 5)
    rng = np.random.default_rng(20261015)
    subset = scaled[rng.choice(np.flatnonzero(train_mask), min(512, train_mask.sum()), replace=False)]
    distances = euclidean_distances(subset, squared=True)
    pairs = distances[np.triu_indices(len(subset), 1)]
    nonzero = pairs[pairs > 1e-8]
    width = np.median(nonzero) if len(nonzero) else 1.
    if not len(nonzero):
        print("constant source-context descriptor; kernel will be all ones", flush=True)
    return scaled, width


def infer(prior_b, prior_e, base, expert, ctx, width, q, s, y, alpha,
          relation=None, weights=None):
    sb, qb = base[s], base[q]
    sz, qz = expert[s], expert[q]
    base_score = prior_b[q] + (qb @ sb.T) @ coefficients(sb @ sb.T, prior_b[s], y[s], 8.)
    ss, qs = sz @ sz.T, qz @ sz.T
    if relation is not None:
        sr, qr = relation[s], relation[q]
        multiplier = np.ones(sr.shape[1]) if weights is None else weights
        ss = .75 * ss + .25 * ((sr * multiplier) @ sr.T)
        qs = .75 * qs + .25 * ((qr * multiplier) @ sr.T)
    if ctx is not None and alpha:
        ks = np.exp(-euclidean_distances(ctx[s], squared=True) / width)
        kq = np.exp(-euclidean_distances(ctx[q], ctx[s], squared=True) / width)
        ss *= 1 - alpha + alpha * ks
        qs *= 1 - alpha + alpha * kq
    expert_score = prior_e[q] + qs @ coefficients(ss, prior_e[s], y[s], 8.)
    return .5 * (softmax(base_score, axis=1) + softmax(expert_score, axis=1))


def main():
    data = load("Singapore", FOLD, ROOT)
    episodes = json.loads((HERE / "source_episodes.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(
        HERE.parent / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv",
        usecols=["instance_id"])
    feature_ids = manifest.instance_id.to_numpy(str)[data["rows"]]
    lookup = pd.Series(np.arange(len(feature_ids)), index=feature_ids)
    train = data["train"]
    held_ids = sorted({sample for e in episodes for sample in e["query"] + [
        item for draw in e["draws"] for origin in ("same", "cross") for item in draw[origin]]})
    held = lookup.loc[held_ids].to_numpy(int)
    assert not set(train) & set(held)
    selected = np.sort(np.r_[train, held])
    local_train = np.searchsorted(selected, train)
    local_held = np.searchsorted(selected, held)
    assert np.array_equal(selected[local_held], held)
    labels = data["labels"][selected]
    print("source train", len(train), "held", len(held), flush=True)
    features = {}
    for name, raw in (("base", data["c4"]), ("general", data["raw"]["G50"]),
                      ("domain", data["raw"]["D18"])):
        values = np.asarray(raw[selected], dtype=np.float32)
        pca = PCA(n_components=128, svd_solver="randomized", random_state=20261014)
        with threadpool_limits(limits=2):
            pca.fit(values[local_train])
            features[name] = pca.transform(values).astype(np.float32)
        del values
        print("PCA", name, flush=True)
    indices = np.arange(len(selected))
    base = design_block(features["base"], indices)
    expert = design_block((features["general"], features["domain"]), indices)
    with threadpool_limits(limits=2):
        prior_b = base @ fit_head(features["base"], labels, local_train)
        prior_e = expert @ fit_head((features["general"], features["domain"]), labels, local_train)

    geometry = pd.read_csv(HERE / "geometry_by_sample.csv", usecols=["sample_id", "far_ok"])
    good = set(geometry.loc[geometry.far_ok, "sample_id"].astype(str))
    train_ids = feature_ids[train]
    candidates = np.asarray([x for x in train_ids if x in good])
    rng = np.random.default_rng(20261015)
    train_context_ids = rng.choice(candidates, min(768, len(candidates)), replace=False).tolist()
    descriptor_ids = train_context_ids + held_ids
    assert len(descriptor_ids) == len(set(descriptor_ids))
    cols = ["sample_id"] + [f"path_{p}" for p in ("vv", "vh")]
    for pol in ("vv", "vh"):
        cols += [f"bbox_{axis}_{pol}" for axis in ("x0", "y0", "x1", "y1")]
        cols += [f"native_h_{pol}", f"native_w_{pol}", f"resolved_width_{pol}"]
    metadata = pd.read_csv(SOURCE_META, usecols=cols).set_index("sample_id", verify_integrity=True)
    far_raw, geo_raw = get_descriptors(metadata, descriptor_ids)
    train_mask = np.arange(len(descriptor_ids)) < len(train_context_ids)
    far, far_width = prep_context(far_raw, train_mask)
    geo, geo_width = prep_context(geo_raw, train_mask)
    far_residual_raw = far - Ridge(alpha=10.).fit(geo[train_mask], far[train_mask]).predict(geo)
    far_residual, far_residual_width = prep_context(far_residual_raw, train_mask)
    geometry_parts = {}
    for name, offsets in (("crop", (0, 1)), ("box", (2, 3, 6, 7)),
                          ("position", (4, 5))):
        columns = [8 * pol + item for pol in range(2) for item in offsets]
        geometry_parts[name] = prep_context(geo_raw[:, columns], train_mask)
    held_frame = pd.read_csv(ROOT / FOLD / "source_validation.csv", usecols=["sample_id", "acquisition"])
    held_frame = held_frame.set_index("sample_id").loc[held_ids]
    far_shuffled = far.copy()
    shuffle = np.random.default_rng(20261016)
    for _, group in held_frame.groupby("acquisition", sort=True):
        position = len(train_context_ids) + np.asarray([held_ids.index(x) for x in group.index])
        far_shuffled[position] = far[shuffle.permutation(position)]
    far_map = pd.Series(np.arange(len(descriptor_ids)), index=descriptor_ids)
    context = {"far": (far, far_width), "geometry": (geo, geo_width),
               "far_residual": (far_residual, far_residual_width),
               "far_shuffled": (far_shuffled, far_width), **geometry_parts}
    rows = []
    predictions = []
    for episode_id, episode in enumerate(episodes):
        q_id = episode["query"]
        q = np.searchsorted(selected, lookup.loc[q_id].to_numpy(int))
        for draw_id, draw in enumerate(episode["draws"]):
            for origin in ("same", "cross"):
                s_id = draw[origin]
                s = np.searchsorted(selected, lookup.loc[s_id].to_numpy(int))
                assert len(s) == 16 and np.array_equal(selected[s], lookup.loc[s_id].to_numpy(int))
                old = None
                for arm, kind, alpha in (("plain", None, 0.), ("far_025", "far", .25),
                                         ("far_050", "far", .50),
                                         ("geometry_025", "geometry", .25),
                                         ("crop_025", "crop", .25),
                                         ("box_025", "box", .25),
                                         ("position_025", "position", .25),
                                         ("far_residual_025", "far_residual", .25),
                                         ("far_shuffled_025", "far_shuffled", .25)):
                    ctx, width = context.get(kind, (None, 1.))
                    if ctx is not None:
                        use = np.r_[far_map.loc[feature_ids[selected[np.r_[q, s]]]].to_numpy(int)]
                        full_ctx = np.zeros((len(selected), ctx.shape[1]), dtype=np.float32)
                        full_ctx[np.r_[q, s]] = ctx[use]
                    else:
                        full_ctx = None
                    prob = infer(prior_b, prior_e, base, expert, full_ctx, width, q, s, labels, alpha)
                    pred = prob.argmax(axis=1)
                    truth = labels[q]
                    if old is None:
                        old = pred
                    present = np.unique(truth)
                    rows.append(dict(port=episode["port"], episode=episode_id,
                                     query_acquisition=episode["query_acquisition"],
                                     cross_acquisition=episode["cross_acquisition"],
                                     draw=draw_id, origin=origin, arm=arm,
                                     n=len(q), support_classes=sum(np.asarray(episode["quota"]) > 0),
                                     oa=100*np.mean(pred == truth),
                                     ba=100*np.mean([np.mean(pred[truth == c] == c) for c in present]),
                                     brier=np.mean(np.sum((prob - np.eye(8)[truth])**2, axis=1)),
                                     rescued=int(np.sum((old != truth) & (pred == truth))),
                                     harmed=int(np.sum((old == truth) & (pred != truth)))))
                    predictions.extend(dict(port=episode["port"], episode=episode_id,
                                            query_acquisition=episode["query_acquisition"],
                                            origin=origin, draw=draw_id, arm=arm,
                                            sample_id=sample_id, truth=int(y), prediction=int(p),
                                            correct=bool(y == p), true_probability=float(prob[i, y]))
                                       for i, (sample_id, y, p) in enumerate(zip(q_id, truth, pred)))
    result = pd.DataFrame(rows)
    result.to_csv(HERE / "source_context_pilot.csv", index=False)
    pd.DataFrame(predictions).to_csv(HERE / "source_context_predictions.csv", index=False)
    print(result.groupby(["origin", "arm"])[["oa", "ba", "brier", "rescued", "harmed"]].mean().round(4).to_string())
    print("By port", flush=True)
    print(result.groupby(["port", "origin", "arm"])[["oa", "ba", "brier"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
