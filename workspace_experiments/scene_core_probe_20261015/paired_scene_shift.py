"""Probe whether a scene-pair shift transfers to unseen vessels and support use.

Source-development diagnostic. Paired MMSIs estimate shifts; target inference
must not rely on MMSI or labels. Oracle class shifts are an upper bound only.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "scene_cross_dataset_redesign_20261014/ports_full_expansion"
FOLD = "Singapore__S1A_060968_079865"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "scene_cross_dataset_redesign_20261014/public_full_20261014"))
from evaluate import load  # noqa: E402
from run import coefficients, design_block, fit_head  # noqa: E402


def select_pairs(frame):
    options = []
    unique = frame.drop_duplicates(["port", "acquisition", "mmsi"])
    for port, port_rows in unique.groupby("port", sort=True):
        scenes = {acq: rows.set_index("mmsi") for acq, rows in
                  port_rows.groupby("acquisition", sort=True)}
        for a, b in itertools.combinations(scenes, 2):
            shared = sorted(set(scenes[a].index) & set(scenes[b].index))
            if len(shared) < 24:
                continue
            labels_a = scenes[a].loc[shared, "class_id"].to_numpy(int)
            labels_b = scenes[b].loc[shared, "class_id"].to_numpy(int)
            valid = labels_a == labels_b
            shared = np.asarray(shared)[valid]
            if len(shared) >= 24 and len(set(labels_a[valid])) >= 2:
                options.append(dict(port=port, a=a, b=b, ships=len(shared),
                                    classes=len(set(labels_a[valid]))))
    chosen, used = [], {}
    for item in sorted(options, key=lambda v: (-v["ships"], v["port"], v["a"], v["b"])):
        port = item["port"]
        if item["a"] in used.get(port, set()) or item["b"] in used.get(port, set()):
            continue
        if sum(x["port"] == port for x in chosen) >= 3:
            continue
        chosen.append(item)
        used.setdefault(port, set()).update([item["a"], item["b"]])
    return sorted(chosen, key=lambda v: (v["port"], v["a"], v["b"]))


def scene_vectors(frame, base, expert, port, acquisition):
    subset = frame.loc[frame.port.eq(port) & frame.acquisition.eq(acquisition)]
    result = {}
    for mmsi, rows in subset.groupby("mmsi", sort=True):
        if rows.class_id.nunique() != 1:
            continue
        ids = rows.index.to_numpy(int)
        result[mmsi] = (int(rows.class_id.iloc[0]), base[ids].mean(0), expert[ids].mean(0))
    return result


def split_ships(shared, classes, rng):
    train, test = [], []
    for c in np.unique(classes):
        members = rng.permutation(np.flatnonzero(classes == c))
        if len(members) == 1:
            train.extend(members)
        else:
            n = min(len(members) - 1, max(1, int(np.ceil(.7 * len(members)))))
            train.extend(members[:n])
            test.extend(members[n:])
    return np.asarray(train, int), np.asarray(test, int)


def balanced_support(train, classes, rng):
    groups = {c: list(rng.permutation(train[classes[train] == c]))
              for c in np.unique(classes[train])}
    selected = []
    while len(selected) < 16:
        advanced = False
        for c in groups:
            if groups[c] and len(selected) < 16:
                selected.append(groups[c].pop())
                advanced = True
        if not advanced:
            break
    assert len(selected) == 16
    return np.asarray(selected, int)


def metrics(y, prediction):
    return (100 * np.mean(y == prediction),
            100 * np.mean([np.mean(prediction[y == c] == c) for c in np.unique(y)]))


def risk(y, prob):
    return float(np.mean(np.sum((prob - np.eye(8)[y]) ** 2, axis=1)))


def support_infer(qb, qz, sb, sz, sy, head_b, head_e, offset):
    base_prior_s, expert_prior_s = sb @ head_b, sz @ head_e
    base_score = qb @ head_b + (qb @ sb.T) @ coefficients(
        sb @ sb.T, base_prior_s, sy, 8.)
    expert_coef = coefficients(sz @ sz.T, expert_prior_s, sy, 8.)
    expert_score = qz @ head_e + ((qz - offset) @ sz.T) @ expert_coef
    return .5 * (softmax(base_score, axis=1) + softmax(expert_score, axis=1))


def cohort_shift(a, b, excluded, head_b, balance=None):
    aa = [value for mmsi, value in a.items() if mmsi not in excluded]
    bb = [value for mmsi, value in b.items() if mmsi not in excluded]
    az, bz = np.stack([v[2][:-1] for v in aa]), np.stack([v[2][:-1] for v in bb])
    if balance is None:
        return bz.mean(0) - az.mean(0), 0, len(aa), len(bb)
    if balance == "predicted":
        ay = (np.stack([v[1] for v in aa]) @ head_b).argmax(1)
        by = (np.stack([v[1] for v in bb]) @ head_b).argmax(1)
    else:
        ay, by = np.asarray([v[0] for v in aa]), np.asarray([v[0] for v in bb])
    common = [c for c in range(8) if np.sum(ay == c) >= 3 and np.sum(by == c) >= 3]
    if not common:
        return bz.mean(0) - az.mean(0), 0, len(aa), len(bb)
    offset = np.mean([bz[by == c].mean(0) - az[ay == c].mean(0) for c in common], axis=0)
    return offset, len(common), len(aa), len(bb)


def main():
    data = load("Singapore", FOLD, ROOT)
    frame = pd.read_csv(ROOT / FOLD / "source_train.csv", dtype={"mmsi": str})
    assert np.array_equal(frame.row_index.to_numpy(int), data["rows"][data["train"]])
    pairs = select_pairs(frame)
    assert len(pairs) >= 6
    pair_ships = set()
    for pair in pairs:
        a = frame.loc[frame.port.eq(pair["port"]) & frame.acquisition.eq(pair["a"]), "mmsi"]
        b = frame.loc[frame.port.eq(pair["port"]) & frame.acquisition.eq(pair["b"]), "mmsi"]
        pair_ships.update(set(a) & set(b))
    source_fit = np.flatnonzero(~frame.mmsi.isin(pair_ships).to_numpy())
    print("pairs", len(pairs), "ports", len(set(x["port"] for x in pairs)),
          "held identities", len(pair_ships), "head fit rows", len(source_fit), flush=True)

    features = {}
    for name, raw in (("c4", data["c4"]), ("g50", data["raw"]["G50"]),
                      ("d18", data["raw"]["D18"])):
        values = np.asarray(raw[data["train"]], dtype=np.float32)
        pca = PCA(n_components=128, svd_solver="randomized", random_state=20261014)
        with threadpool_limits(limits=2):
            features[name] = pca.fit_transform(values).astype(np.float32)
        del values
        print("PCA", name, flush=True)
    idx = np.arange(len(frame))
    base = design_block(features["c4"], idx)
    expert = design_block((features["g50"], features["d18"]), idx)
    labels = frame.class_id.to_numpy(int)
    with threadpool_limits(limits=2):
        head_b = fit_head(features["c4"], labels, source_fit)
        head_e = fit_head((features["g50"], features["d18"]), labels, source_fit)
    inventory, rows, decisions = [], [], []
    for pair_id, pair in enumerate(pairs):
        a = scene_vectors(frame, base, expert, pair["port"], pair["a"])
        b = scene_vectors(frame, base, expert, pair["port"], pair["b"])
        shared = sorted(m for m in set(a) & set(b) if a[m][0] == b[m][0])
        y = np.asarray([a[m][0] for m in shared], int)
        a_b = np.stack([a[m][1] for m in shared])
        b_b = np.stack([b[m][1] for m in shared])
        a_z = np.stack([a[m][2] for m in shared])
        b_z = np.stack([b[m][2] for m in shared])
        delta = b_z[:, :-1] - a_z[:, :-1]
        assert len(shared) == pair["ships"]
        inventory.append({**pair, "pair_id": pair_id,
                          "class_counts": {str(c): int(n) for c, n in
                                           zip(*np.unique(y, return_counts=True))}})
        for split in range(3):
            rng = np.random.default_rng(20261015 + pair_id * 100 + split)
            train, test = split_ships(shared, y, rng)
            assert len(train) >= 16 and len(test) >= 4
            global_delta = delta[train].mean(0)
            excluded = {shared[j] for j in test}
            cohort_delta, _, cohort_a, cohort_b = cohort_shift(a, b, excluded, head_b)
            pseudo_delta, pseudo_classes, _, _ = cohort_shift(
                a, b, excluded, head_b, "predicted")
            true_bal_delta, true_bal_classes, _, _ = cohort_shift(
                a, b, excluded, head_b, "true")
            class_delta = {c: delta[train[y[train] == c]].mean(0)
                           for c in np.unique(y[train])
                           if np.sum(y[train] == c) >= 3}
            oracle_delta = np.stack([class_delta.get(c, global_delta) for c in y[test]])
            base_prior = b_b[test] @ head_b
            predicted_class = base_prior.argmax(1)
            predicted_delta = np.stack([class_delta.get(c, global_delta)
                                        for c in predicted_class])
            class_options = np.stack([class_delta.get(c, global_delta) for c in range(8)])
            soft_delta = softmax(base_prior, axis=1) @ class_options
            observed = delta[test]
            denominator = float(np.sum(observed ** 2))
            assert denominator > 0
            repeated_r2 = {"zero": 0.,
                           "global": 1 - float(np.sum((observed - global_delta) ** 2)) / denominator,
                           "class_oracle": 1 - float(np.sum((observed - oracle_delta) ** 2)) / denominator,
                           "class_predicted": 1 - float(np.sum((observed - predicted_delta) ** 2)) / denominator,
                           "class_soft": 1 - float(np.sum((observed - soft_delta) ** 2)) / denominator,
                           "cohort_unlabeled": 1 - float(np.sum((observed - cohort_delta) ** 2)) / denominator,
                           "cohort_pseudobal": 1 - float(np.sum((observed - pseudo_delta) ** 2)) / denominator,
                           "cohort_truebal": 1 - float(np.sum((observed - true_bal_delta) ** 2)) / denominator,
                           "reverse": 1 - float(np.sum((observed + global_delta) ** 2)) / denominator}
            score_delta = observed @ head_e[:-1]
            score_den = float(np.sum(score_delta ** 2))
            score_r2 = {"global": 1 - float(np.sum((score_delta - global_delta @ head_e[:-1]) ** 2)) / score_den,
                        "class_oracle": 1 - float(np.sum((score_delta - oracle_delta @ head_e[:-1]) ** 2)) / score_den,
                        "class_predicted": 1 - float(np.sum((score_delta - predicted_delta @ head_e[:-1]) ** 2)) / score_den,
                        "class_soft": 1 - float(np.sum((score_delta - soft_delta @ head_e[:-1]) ** 2)) / score_den,
                        "cohort_unlabeled": 1 - float(np.sum((score_delta - cohort_delta @ head_e[:-1]) ** 2)) / score_den,
                        "cohort_pseudobal": 1 - float(np.sum((score_delta - pseudo_delta @ head_e[:-1]) ** 2)) / score_den,
                        "cohort_truebal": 1 - float(np.sum((score_delta - true_bal_delta @ head_e[:-1]) ** 2)) / score_den,
                        "reverse": 1 - float(np.sum((score_delta + global_delta @ head_e[:-1]) ** 2)) / score_den}
            for draw in range(3):
                support = balanced_support(train, y, np.random.default_rng(
                    20261016 + pair_id * 100 + split * 10 + draw))
                modes = {"raw": np.zeros_like(b_z[test, :-1]),
                         "global": np.broadcast_to(global_delta, (len(test), len(global_delta))),
                         "reverse": np.broadcast_to(-global_delta, (len(test), len(global_delta))),
                         "class_predicted": predicted_delta,
                         "class_soft": soft_delta,
                         "cohort_unlabeled": np.broadcast_to(cohort_delta, (len(test), len(cohort_delta))),
                         "cohort_pseudobal": np.broadcast_to(pseudo_delta, (len(test), len(pseudo_delta))),
                         "cohort_truebal": np.broadcast_to(true_bal_delta, (len(test), len(true_bal_delta))),
                         "class_oracle": oracle_delta}
                raw_prediction = None
                for mode, shift in modes.items():
                    offset = np.c_[shift, np.zeros(len(shift))]
                    prob = support_infer(b_b[test], b_z[test], a_b[support], a_z[support],
                                         y[support], head_b, head_e, offset)
                    pred = prob.argmax(1)
                    if raw_prediction is None:
                        raw_prediction = pred
                    oa, ba = metrics(y[test], pred)
                    rows.append({**pair, "pair_id": pair_id, "split": split,
                                 "draw": draw, "mode": mode, "train_ships": len(train),
                                 "test_ships": len(test), "query_classes": len(np.unique(y[test])),
                                 "cohort_a_ships": cohort_a, "cohort_b_ships": cohort_b,
                                 "pseudo_balanced_classes": pseudo_classes,
                                 "true_balanced_classes": true_bal_classes,
                                 "base_class_accuracy": float(np.mean(predicted_class == y[test])),
                                 "oa": oa, "ba": ba, "brier": risk(y[test], prob),
                                 "rescued": int(np.sum((raw_prediction != y[test]) & (pred == y[test]))),
                                 "harmed": int(np.sum((raw_prediction == y[test]) & (pred != y[test]))),
                                 "r2_feature": repeated_r2.get(mode, np.nan),
                                 "r2_score": score_r2.get(mode, np.nan)})
                    decisions.extend(dict(pair_id=pair_id, split=split, draw=draw,
                                          mode=mode, mmsi=shared[j], truth=int(y[j]),
                                          pred=int(pred[k]))
                                     for k, j in enumerate(test))
            print("pair", pair_id, pair["port"], "ships", len(shared),
                  "R2", round(repeated_r2["global"], 3),
                  "scoreR2", round(score_r2["global"], 3), flush=True)
    (HERE / "pair_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "paired_scene_metrics.csv", index=False)
    pd.DataFrame(decisions).to_csv(HERE / "paired_scene_decisions.csv", index=False)
    print(table.groupby("mode")[["oa", "ba", "brier", "r2_feature", "r2_score"]].mean().round(4).to_string())
    print(table.groupby(["port", "mode"])[["oa", "ba", "r2_feature"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
