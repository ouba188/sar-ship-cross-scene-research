"""Frozen SAR-R18 pre-pool evidence probe on held-port acquisition episodes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CONTEXT = PROJECT / "context_condition_study_20261015"
PIXELS = Path("D:/SAR_completion_cache_20261010/source_inputs128.npy")
sys.path.insert(0, "E:/Hermes/paper_tracking/sensitivity/oof_reliability_check")
from general_domain_ssl import backbone, normalize, rgb  # noqa: E402


def fixed_rows(episode):
    folder = Path(episode["folder"])
    train = pd.read_csv(folder / "source_train.csv", dtype={"mmsi": str})
    validation = pd.read_csv(folder / "source_validation.csv", dtype={"mmsi": str})
    query = pd.read_csv(folder / "query.csv", dtype={"mmsi": str})
    query = query.set_index("sample_id").loc[episode["query"]].reset_index()
    support = pd.read_csv(folder / "support_candidates.csv", dtype={"mmsi": str})
    support = support.set_index("sample_id")
    rng = np.random.default_rng(20261015)
    gallery = []
    for _, group in train.groupby(["port", "class_id"], sort=True):
        gallery.extend(rng.choice(group.row_index.to_numpy(int),
                                  min(8, len(group)), replace=False).tolist())
    gallery = train.set_index("row_index").loc[gallery].reset_index()
    support_ids = {origin: sorted(set(sum((draw[origin] for draw in episode["draws"]), [])))
                   for origin in ("same", "cross")}
    all_rows = np.unique(np.r_[gallery.row_index.to_numpy(int),
                               validation.row_index.to_numpy(int),
                               query.row_index.to_numpy(int),
                               support.loc[support_ids["same"]].row_index.to_numpy(int),
                               support.loc[support_ids["cross"]].row_index.to_numpy(int)])
    local = lambda frame: np.searchsorted(all_rows, frame.row_index.to_numpy(int))
    groups = dict(gallery=gallery, validation=validation, query=query,
                  same=support.loc[support_ids["same"]].reset_index(),
                  cross=support.loc[support_ids["cross"]].reset_index())
    for frame in groups.values():
        np.testing.assert_array_equal(all_rows[local(frame)], frame.row_index.to_numpy(int))
    assert not set(query.mmsi) & set(groups["same"].mmsi)
    assert not set(query.mmsi) & set(groups["cross"].mmsi)
    return all_rows, groups


@torch.inference_mode()
def encode(rows, checkpoint):
    torch.set_num_threads(2)
    model, _ = backbone("r18")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(weights["backbone"], strict=True)
    model.cuda().eval()
    pixels = np.load(PIXELS, mmap_mode="r")
    captured = {name: [] for name in ("stem", "layer1", "layer2", "layer3", "layer4")}
    hooks = [model.relu.register_forward_hook(
                 lambda _, __, result: captured["stem"].append(result)),
             model.layer1.register_forward_hook(
                 lambda _, __, result: captured["layer1"].append(result)),
             model.layer2.register_forward_hook(
                 lambda _, __, result: captured["layer2"].append(result)),
             model.layer3.register_forward_hook(
                 lambda _, __, result: captured["layer3"].append(result)),
             model.layer4.register_forward_hook(
                 lambda _, __, result: captured["layer4"].append(result))]
    prefixes = ("early_stem", "early1", "early2", "mid")
    parts = {name: [] for name in ("gap", "peak", "mean_std", "quadrants")}
    for prefix in prefixes:
        parts.update({prefix + "_" + suffix: []
                      for suffix in ("gap", "peak", "mean_std", "quadrants")})
    for start in range(0, len(rows), 32):
        batch = rows[start:start + 32]
        pair = torch.from_numpy(np.array(pixels[batch], dtype=np.float32)).cuda() / 255.
        image = normalize(rgb(F.interpolate(pair, (112, 112), mode="bilinear",
                                            align_corners=False, antialias=True)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pooled = model(image).float()
        spatial = captured["layer4"].pop().float()
        spatial_mean = spatial.mean((2, 3))
        torch.testing.assert_close(spatial_mean, pooled, atol=0.03, rtol=0.01)
        gap = pooled
        values = dict(gap=gap, peak=spatial.amax((2, 3)),
                      mean_std=torch.cat((gap, spatial.std((2, 3), unbiased=False)), 1),
                      quadrants=F.adaptive_avg_pool2d(spatial, (2, 2)).flatten(1))
        for prefix, layer in (("early_stem", "stem"), ("early1", "layer1"),
                              ("early2", "layer2"), ("mid", "layer3")):
            feature_map = captured[layer].pop().float()
            mean = feature_map.mean((2, 3))
            values[prefix + "_gap"] = mean
            values[prefix + "_peak"] = feature_map.amax((2, 3))
            values[prefix + "_mean_std"] = torch.cat(
                (mean, feature_map.std((2, 3), unbiased=False)), 1)
            values[prefix + "_quadrants"] = F.adaptive_avg_pool2d(
                feature_map, (2, 2)).flatten(1)
        for name, value in values.items():
            parts[name].append(value.cpu().numpy().astype(np.float32))
    for hook in hooks:
        hook.remove()
    return {name: np.concatenate(chunks) for name, chunks in parts.items()}


def cached_replay(rows, pooled, source):
    known = np.load(source / "rows.npy")
    position = np.searchsorted(known, rows)
    np.testing.assert_array_equal(known[position], rows)
    cache = np.load(source / "D18.npy", mmap_mode="r")
    reference = np.asarray(cache[position], np.float32)
    difference = pooled - reference
    maximum = float(np.abs(difference).max())
    relative_rmse = float(np.linalg.norm(difference) / np.linalg.norm(reference))
    print("pooled cache replay", source.name, "max", maximum,
          "relative_rmse", relative_rmse, flush=True)
    assert maximum < 0.2 and relative_rmse < 0.01
    return maximum, relative_rmse


def unit_normalize(array):
    array = np.asarray(array, np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def representations(raw, gallery):
    results = {"gap_raw": unit_normalize(raw["gap"])}
    with threadpool_limits(limits=2):
        for name in raw:
            if name.startswith(("early_stem_", "early1_")):
                width = 64
            elif name.startswith("early2_"):
                width = 128
            else:
                width = 256
            pca = PCA(n_components=width, svd_solver="randomized", random_state=20261015)
            pca.fit(raw[name][gallery])
            results[name + "_pca" + str(width)] = unit_normalize(pca.transform(raw[name]))
    final = results["gap_pca256"]
    for stage, width in (("early_stem", 64), ("early1", 64),
                         ("early2", 128), ("mid", 256)):
        for statistic in ("gap", "peak"):
            local = results[stage + "_" + statistic + "_pca" + str(width)]
            results["fusion_" + stage + "_" + statistic] = np.concatenate(
                (final, local), axis=1) / np.sqrt(2.)
    return results


def map_at_r(query, gallery, qlabel, glabel):
    similarity = query @ gallery.T
    scores = []
    for i, cls in enumerate(qlabel):
        relevant = glabel == cls
        count = int(relevant.sum())
        if not count:
            continue
        hits = relevant[np.argsort(-similarity[i])[:count]]
        scores.append((int(cls), float(np.sum(np.cumsum(hits) /
                      np.arange(1, count + 1) * hits) / count)))
    frame = pd.DataFrame(scores, columns=["class_id", "ap"])
    by_class = frame.groupby("class_id").ap.mean()
    return (float(frame.ap.mean()), float(by_class.mean()),
            {int(cls): float(value) for cls, value in by_class.items()})


def evaluate(features, groups, all_rows, episode):
    local = lambda name: np.searchsorted(all_rows, groups[name].row_index.to_numpy(int))
    gallery = groups["gallery"]
    validation = groups["validation"]
    source_scores = []
    for port in sorted(set(gallery.port) & set(validation.port)):
        q = validation.loc[validation.port.eq(port)]
        s = gallery.loc[gallery.port.eq(port)]
        value = map_at_r(features[np.searchsorted(all_rows, q.row_index.to_numpy(int))],
                         features[np.searchsorted(all_rows, s.row_index.to_numpy(int))],
                         q.class_id.to_numpy(int), s.class_id.to_numpy(int))
        if np.isfinite(value[0]):
            source_scores.append(value[:2])
    target = {}
    target_b16 = {}
    truth = pd.read_csv(Path(groups["query"].attrs["folder"]) /
                        "query_labels.csv").set_index("sample_id")
    qlabel = truth.loc[groups["query"].sample_id, "class_id"].to_numpy(int)
    for origin in ("same", "cross"):
        target[origin] = map_at_r(features[local("query")], features[local(origin)],
                                  qlabel,
                                  groups[origin].class_id.to_numpy(int))
        support = groups[origin].set_index("sample_id")
        draws = []
        for draw in episode["draws"]:
            selected = support.loc[draw[origin]]
            s = np.searchsorted(all_rows, selected.row_index.to_numpy(int))
            draws.append(map_at_r(features[local("query")], features[s],
                                  qlabel, selected.class_id.to_numpy(int))[:2])
        target_b16[origin] = np.mean(draws, axis=0)
    return float(np.mean(source_scores, axis=0)[0]), target, target_b16


def run(fold):
    episodes = json.loads((CONTEXT / "target_episodes.json").read_text(encoding="utf-8"))
    episode = next(item for item in episodes if item["fold"] == fold)
    folder = Path(episode["folder"])
    source = folder.parent / episode["port"].replace(" ", "_")
    rows, groups = fixed_rows(episode)
    groups["query"].attrs["folder"] = str(folder)
    raw = encode(rows, source / "source_only_r18_best.pt")
    error, relative_rmse = cached_replay(rows, raw["gap"], source)
    g = np.searchsorted(rows, groups["gallery"].row_index.to_numpy(int))
    features = representations(raw, g)
    output = []
    for name, values in features.items():
        source_map, target, target_b16 = evaluate(values, groups, rows, episode)
        for origin, (micro, macro, by_class) in target.items():
            output.append(dict(fold=fold, representation=name, origin=origin,
                               source_port_map=source_map, target_map=micro,
                               target_macro_map=macro,
                               target_b16_map=target_b16[origin][0],
                               target_b16_macro_map=target_b16[origin][1],
                               target_by_class=json.dumps(by_class, sort_keys=True),
                               encoded_rows=len(rows),
                               pooled_cache_max_error=error,
                               pooled_cache_relative_rmse=relative_rmse))
    HERE.mkdir(exist_ok=True)
    frame = pd.DataFrame(output)
    frame.to_csv(HERE / f"results_{fold}.csv", index=False)
    print(frame.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fold")
    run(parser.parse_args().fold)
