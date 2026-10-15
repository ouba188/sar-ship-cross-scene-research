"""Test source-confusability-selected class-pair relations on frozen SAR features."""

import argparse
import gc
from itertools import chain, islice
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "public_full_20261014"))
sys.path.insert(0, str(ROOT / "ports_full_expansion"))
from run import class_map, episode_paths, infer, load_fold, pca, ship_metrics  # noqa: E402
from evaluate import load as load_port  # noqa: E402
from general_domain_readout import design_block, fit_head, metrics  # noqa: E402


PAIRS = np.asarray([(a, b) for a in range(8) for b in range(a + 1, 8)])
ARMS = ("plain", "all28", "confusing14", "easy14", "random14")
PUBLIC = ROOT / "public_full_20261014"
PORT_ROOTS = (ROOT / "ports_full_expansion", ROOT / "ports_broad_expansion_20261014")


def public_folds(dataset):
    manifest = pd.read_csv(
        f"E:/Hermes/paper_tracking/sensitivity/opensarship{dataset[-1]}_transfer_20261007/target_manifest.csv"
    )
    c4 = np.load(PUBLIC / f"{dataset}_c4.npy", mmap_mode="r")
    zero = np.load(PUBLIC / f"{dataset}_zero.npy", mmap_mode="r")
    for fold, cache, episode, prefix in episode_paths(dataset):
        data = load_fold(dataset, fold, cache, episode, prefix, manifest, c4, zero)
        data["dataset"] = dataset
        yield data, PUBLIC / dataset / f"{fold}_metrics.csv"


def port_folds():
    for root in PORT_ROOTS:
        for split in sorted(root.glob("*__*/split.npz")):
            fold = split.parent.name
            port = fold.split("__", 1)[0].replace("_", " ")
            data = load_port(port, fold, root)
            yield data, split.parent / "full_metrics.csv"


def class_confusions(z, labels, train, val):
    source_head = fit_head(z[:, :-1], labels, train)
    probabilities = softmax(z[val] @ source_head, axis=1)
    counts = np.bincount(labels[val], minlength=8)
    by_class = np.stack([probabilities[labels[val] == c].mean(0) for c in range(8)])
    score = by_class[PAIRS[:, 0], PAIRS[:, 1]] + by_class[PAIRS[:, 1], PAIRS[:, 0]]
    order = np.argsort(-score, kind="stable")
    return counts, score, order[:14], order[14:]


def relation_mask(relation, source, chosen):
    full_energy = np.mean(np.sum(relation[source] ** 2, axis=1))
    part_energy = np.mean(np.sum(relation[source][:, chosen] ** 2, axis=1))
    weights = np.zeros(len(PAIRS))
    weights[chosen] = full_energy / part_energy
    return weights


def evaluate_fold(data, reference):
    labels, train, val, query = (data[key] for key in ("labels", "train", "val", "query"))
    source = np.r_[train, val]
    c = pca(data["c4"], source)
    g = pca(data["raw"]["G50"], source)
    d = pca(data["raw"]["D18"], source)
    rows = np.arange(len(labels))
    base = design_block(c, rows)
    expert = design_block((g, d), rows)
    relation = class_map(expert[:, :-1], labels, train)
    base_prior = base @ fit_head(c, labels, source)
    expert_prior = expert @ fit_head((g, d), labels, source)
    val_counts, confusion, top, bottom = class_confusions(expert, labels, train, val)
    random = np.sort(np.random.default_rng(20261015).choice(len(PAIRS), size=14, replace=False))
    masks = {"confusing14": relation_mask(relation, train, top),
             "easy14": relation_mask(relation, train, bottom),
             "random14": relation_mask(relation, train, random)}

    fold = data["fold"]
    dataset = data["dataset"]
    port = data.get("port", "")
    truth = labels[query]
    frame = data["query_frame"].copy()
    frame["class_id"] = truth
    raw_rows = data["rows"][query]
    recorded = pd.read_csv(reference)
    summary, cases = [], []
    for (origin, draw), support in data["support"].items():
        _, _, plain = infer(base, expert, None, base_prior, expert_prior, query, support, labels)
        _, _, all28 = infer(base, expert, relation, base_prior, expert_prior, query, support, labels)
        outputs = {"plain": plain, "all28": all28}
        for name, mask in masks.items():
            _, _, outputs[name] = infer(base, expert, relation, base_prior, expert_prior,
                                        query, support, labels, mask)
        for arm, scores in outputs.items():
            chip = metrics(truth, scores.argmax(1))
            ship = ship_metrics(frame, scores)
            summary.append(dict(dataset=dataset, port=port, fold=fold, origin=origin,
                                draw=draw, arm=arm, query_rows=len(query),
                                query_classes=len(np.unique(truth)), source_val_min=int(val_counts.min()),
                                chip_oa=chip["oa"], chip_ba=chip["ba"],
                                ship_oa=ship["oa"], ship_ba=ship["ba"]))
            if arm in ("plain", "all28"):
                original_arm = "c4_dual_plain" if arm == "plain" else "c4_class_fixed"
                old = recorded[(recorded.origin == origin) & (recorded.draw == draw)
                               & (recorded.arm == original_arm)].iloc[0]
                np.testing.assert_allclose([chip["oa"], chip["ba"], ship["oa"], ship["ba"]],
                                           old[["chip_oa", "chip_ba", "ship_oa", "ship_ba"]].to_numpy(float),
                                           atol=1e-8, rtol=0)
        for i, (row, mmsi, target) in enumerate(zip(raw_rows, frame.mmsi, truth)):
            item = dict(dataset=dataset, port=port, fold=fold, origin=origin,
                        draw=draw, query_row=int(row), mmsi=str(mmsi), truth=int(target))
            for arm, scores in outputs.items():
                item[f"pred_{arm}"] = int(np.argmax(scores[i]))
                item[f"ptrue_{arm}"] = float(scores[i, target])
            cases.append(item)
    fit = dict(dataset=dataset, port=port, fold=fold, source_train=len(train),
               source_val=len(val), source_val_class_counts=val_counts.tolist(),
               pairs=PAIRS.tolist(), confusion_scores=confusion.tolist(),
               confusing14=top.tolist(), easy14=bottom.tolist(), random14=random.tolist(),
               mask_scales={name: float(mask.max()) for name, mask in masks.items()},
               selection="14 largest symmetric soft confusion scores from source validation; source-train expert head")
    return pd.DataFrame(summary), pd.DataFrame(cases), fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("oss1", "oss2", "ports", "all"), default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    groups = []
    if args.scope in ("oss1", "all"):
        groups.append(public_folds("oss1"))
    if args.scope in ("oss2", "all"):
        groups.append(public_folds("oss2"))
    if args.scope in ("ports", "all"):
        groups.append(port_folds())
    folds = chain.from_iterable(groups)
    if args.limit is not None:
        folds = islice(folds, args.limit)
    for data, reference in folds:
        print(f"FOLD {data['dataset']} {data.get('port', '')} {data['fold']}", flush=True)
        with threadpool_limits(limits=2):
            summary, cases, fit = evaluate_fold(data, reference)
        target = HERE / data["dataset"] / data["fold"]
        target.mkdir(parents=True, exist_ok=True)
        summary.to_csv(target / "metrics.csv", index=False)
        cases.to_csv(target / "cases.csv", index=False)
        (target / "source_selection.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
        print(summary.groupby(["origin", "arm"])[["chip_oa", "chip_ba"]].mean().round(2).to_string(), flush=True)
        del data, summary, cases
        gc.collect()


if __name__ == "__main__":
    main()
