"""Check source-held discriminability of selected relation coordinates."""

from itertools import chain
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from experiment import HERE, class_map, design_block, pca, port_folds, public_folds


def graph_gap(pairs, chosen):
    laplacian = np.zeros((8, 8))
    for index in chosen:
        a, b = pairs[index]
        laplacian[a, a] += 1
        laplacian[b, b] += 1
        laplacian[a, b] -= 1
        laplacian[b, a] -= 1
    return float(np.linalg.eigvalsh(laplacian)[1])


def main():
    result = []
    folds = chain(public_folds("oss1"), public_folds("oss2"), port_folds())
    for data, _ in folds:
        labels, train, val = (data[key] for key in ("labels", "train", "val"))
        source = np.r_[train, val]
        with threadpool_limits(limits=2):
            g = pca(data["raw"]["G50"], source)
            d = pca(data["raw"]["D18"], source)
            z = design_block((g, d), np.arange(len(labels)))
            relation = class_map(z[:, :-1], labels, train)
        selection = json.loads((HERE / data["dataset"] / data["fold"] / "source_selection.json").read_text(encoding="utf-8"))
        auc = []
        for index, (a, b) in enumerate(selection["pairs"]):
            aa = relation[val[labels[val] == a], index]
            bb = relation[val[labels[val] == b], index]
            auc.append(roc_auc_score(np.r_[np.ones(len(aa)), np.zeros(len(bb))], np.r_[aa, bb]))
        for group in ("confusing14", "easy14", "random14"):
            chosen = selection[group]
            result.append(dict(dataset=data["dataset"], port=data.get("port", ""), fold=data["fold"],
                               group=group, mean_source_pair_auc=float(np.mean(np.asarray(auc)[chosen])),
                               graph_gap=graph_gap(selection["pairs"], chosen)))
        print("DONE", data["dataset"], data["fold"], flush=True)
        del data, g, d, z, relation
    out = pd.DataFrame(result)
    out.to_csv(HERE / "SOURCE_PAIR_DISCRIMINABILITY.csv", index=False)
    print(out.groupby(["dataset", "group"])[["mean_source_pair_auc", "graph_gap"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
