"""Source-only C4, class-contrast, and budgeted-handoff public SAR readout."""

import argparse
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "E:/Hermes/paper_tracking/sensitivity/oof_reliability_check")
from general_domain_readout import design_block, fit_head, metrics  # noqa: E402
from unified_head_core import coefficients  # noqa: E402
from adaptive_fusion_handoff_policy import features as gate_features, fit as fit_gate, probability as gate_probability  # noqa: E402


def episode_paths(dataset):
    pilot = ROOT / "strict_training" / ("oss1" if dataset == "oss1" else "oss2_v2")
    public = ROOT / "public_split"
    result = [("pilot", pilot, public, dataset + "_")]
    for folder in sorted((ROOT / "public_expansion" / dataset).iterdir()):
        if folder.is_dir() and (folder / "fit" / "pilot_features.npz").exists():
            result.append((folder.name, folder / "fit", folder, ""))
    return result


def load_fold(dataset, fold, cache, episode, prefix, manifest, c4, zero):
    ids = manifest.sample_id.astype(str)
    global_index = pd.Series(np.arange(len(ids)), index=ids)
    with np.load(cache / "pilot_features.npz") as saved:
        rows = (global_index.loc[saved["sample_ids"].astype(str)].to_numpy(int)
                if "sample_ids" in saved else saved["rows"].astype(int))
        raw = {name: saved[name] for name in ("G50", "G18", "D18")}
    assert len(rows) == len(set(rows))
    labels = manifest.class_id.to_numpy(int)[rows]
    local_index = pd.Series(np.arange(len(rows)), index=ids.iloc[rows])
    local = lambda frame: local_index.loc[frame.sample_id.astype(str)].to_numpy(int)
    source_folder = cache if (cache / "source_train.csv").exists() else episode
    source_train = pd.read_csv(source_folder / "source_train.csv", dtype={"mmsi": str})
    source_val = pd.read_csv(source_folder / "source_validation.csv", dtype={"mmsi": str})
    query_frame = pd.read_csv(episode / f"{prefix}query.csv", dtype={"mmsi": str})
    draws = pd.read_csv(episode / f"{prefix}support_draws.csv", dtype={"mmsi": str})
    draws = draws[draws.budget == 16].copy()
    train, val, query = map(local, (source_train, source_val, query_frame))
    support = {(origin, draw): local(draws[(draws.support_origin == origin) & (draws.draw == draw)])
               for origin in ("same", "cross") for draw in range(5)}
    assert all(len(s) == 16 for s in support.values())
    assert set(source_train.class_id) == set(source_val.class_id) == set(range(8))
    assert not set(source_train.acquisition) & set(source_val.acquisition)
    target_acq = set(query_frame.acquisition) | set(draws.acquisition)
    target_ship = set(query_frame.mmsi) | set(draws.mmsi)
    assert not (set(source_train.acquisition) | set(source_val.acquisition)) & target_acq
    assert not (set(source_train.mmsi) | set(source_val.mmsi)) & target_ship
    np.testing.assert_array_equal(manifest.sample_id.to_numpy(str)[rows[query]], query_frame.sample_id.to_numpy(str))
    return dict(fold=fold, train=train, val=val, query=query, support=support, labels=labels,
                raw=raw, c4=np.asarray(c4[rows]), zero=np.asarray(zero[rows]), source_train=source_train, source_val=source_val,
                query_frame=query_frame, rows=rows)


def pca(raw, source):
    fit = PCA(n_components=128, svd_solver="randomized", random_state=20261014)
    return fit.fit(raw[source]).transform(raw).astype(np.float32)


def balanced_support(pool, labels, rng):
    choices = []
    by_class = {c: list(rng.permutation(pool[labels[pool] == c])) for c in range(8)}
    while len(choices) < 16:
        available = False
        for c in range(8):
            if by_class[c] and len(choices) < 16:
                choices.append(by_class[c].pop())
                available = True
        if not available:
            break
    return np.asarray(choices, dtype=int)


def class_map(z, labels, train):
    centers = np.stack([z[train[labels[train] == c]].mean(0) for c in range(8)])
    pair = np.array([(a, b) for a in range(8) for b in range(a + 1, 8)])
    value = z @ (centers[pair[:, 0]] - centers[pair[:, 1]]).T
    mean = value[train].mean(0)
    scale = value[train].std(0).clip(1e-6)
    value = (value - mean) / scale
    energy = np.mean(np.sum(z[train] ** 2, axis=1) + 1)
    value *= np.sqrt(energy / np.mean(np.sum(value[train] ** 2, axis=1)))
    return value


def infer(base, z, relation, base_prior, expert_prior, q, s, labels, weights=None):
    sb, qb = base[s], base[q]
    sz, qz = z[s], z[q]
    sy = labels[s]
    base_score = base_prior[q] + (qb @ sb.T) @ coefficients(
        sb @ sb.T, base_prior[s], sy, 8.)
    ss = sz @ sz.T
    qs = qz @ sz.T
    if relation is not None:
        sr, qr = relation[s], relation[q]
        multiplier = np.ones(sr.shape[1]) if weights is None else weights
        ss = .75 * ss + .25 * ((sr * multiplier) @ sr.T)
        qs = .75 * qs + .25 * ((qr * multiplier) @ sr.T)
    expert_score = expert_prior[q] + qs @ coefficients(ss, expert_prior[s], sy, 8.)
    pb, pe = softmax(base_score, axis=1), softmax(expert_score, axis=1)
    return pb, pe, .5 * (pb + pe)


def source_tasks(data, base, z, relation):
    rng = np.random.default_rng(20261014)
    frame, labels = data["source_train"], data["labels"]
    train = data["train"]
    tasks = []
    grouped_acquisitions = list(frame.groupby("acquisition", sort=True))
    cap = data.get("relation_task_cap")
    if cap is not None and len(grouped_acquisitions) > cap:
        grouped_acquisitions = sorted(grouped_acquisitions, key=lambda item: (-len(item[1]), item[0]))[:cap]
    for acquisition, grouped in grouped_acquisitions:
        q = train[grouped.index.to_numpy()]
        if len(q) < 8:
            continue
        q = rng.choice(q, size=min(64, len(q)), replace=False)
        ship = set(frame.iloc[grouped.index].mmsi)
        pool = train[~frame.acquisition.eq(acquisition).to_numpy() & ~frame.mmsi.isin(ship).to_numpy()]
        if set(labels[pool]) != set(range(8)):
            continue
        s = balanced_support(pool, labels, rng)
        if len(s) != 16:
            continue
        bh, eh = fit_head(base[:, :-1], labels, pool), fit_head(z[:, :-1], labels, pool)
        pb, pe = base @ bh, z @ eh
        sb, qb, sz, qz, sr, qr = base[s], base[q], z[s], z[q], relation[s], relation[q]
        sy, y = labels[s], labels[q]
        base_score = pb[q] + (qb @ sb.T) @ coefficients(sb @ sb.T, pb[s], sy, 8.)
        tasks.append(dict(kss=torch.tensor(.75 * sz @ sz.T, dtype=torch.float64),
                          kqs=torch.tensor(.75 * qz @ sz.T, dtype=torch.float64),
                          vss=torch.tensor(sr, dtype=torch.float64),
                          vqs=torch.tensor(qr, dtype=torch.float64),
                          prior=torch.tensor(pe[q], dtype=torch.float64),
                          support_prior=torch.tensor(pe[s], dtype=torch.float64),
                          base=torch.tensor(softmax(base_score, axis=1), dtype=torch.float64),
                          target=torch.tensor(2 * np.eye(8)[sy] - 1, dtype=torch.float64),
                          truth=torch.tensor(np.eye(8)[y], dtype=torch.float64),
                          query_weight=torch.tensor(1. / (len(np.unique(y)) * np.bincount(y, minlength=8)[y]), dtype=torch.float64),
                          root=torch.tensor(np.sqrt(len(s) / (len(np.unique(sy)) * np.bincount(sy, minlength=8)[sy])), dtype=torch.float64)))
    return tasks


def learn_relation(tasks):
    if not tasks:
        raise RuntimeError("No source-held scene can train the class relation")
    theta = torch.nn.Parameter(torch.zeros(28, dtype=torch.float64))
    optimizer = torch.optim.Adam([theta], lr=.03)
    for _ in range(40):
        weights = 28 * torch.softmax(theta, dim=0)
        risks = []
        for task in tasks:
            s = task["kss"] + .25 * ((task["vss"] * weights) @ task["vss"].T)
            q = task["kqs"] + .25 * ((task["vqs"] * weights) @ task["vss"].T)
            root = task["root"]
            target = task["target"] - task["support_prior"]
            coef = root[:, None] * torch.linalg.solve(root[:, None] * s * root[None, :] + 8 * torch.eye(len(root)),
                                                      root[:, None] * target)
            full = .5 * (task["base"] + torch.softmax(task["prior"] + q @ coef, dim=1))
            risks.append(torch.sum(task["query_weight"] * torch.sum((full - task["truth"]) ** 2, dim=1)))
        loss = torch.stack(risks).mean() + .001 * torch.mean((weights - 1) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (28 * torch.softmax(theta.detach(), dim=0)).numpy(), len(tasks)


def source_gate(data, base, z, relation, weights):
    train, labels = data["train"], data["labels"]
    bh, eh = fit_head(base[:, :-1], labels, train), fit_head(z[:, :-1], labels, train)
    pb, pe = base @ bh, z @ eh
    rng = np.random.default_rng(20261015)
    probs, truths, groups = [], [], []
    for acquisition, grouped in data["source_val"].groupby("acquisition", sort=True):
        q = data["val"][grouped.index.to_numpy()]
        pool = train[~data["source_train"].mmsi.isin(set(grouped.mmsi)).to_numpy()]
        s = balanced_support(pool, labels, rng)
        if len(s) < 16:
            continue
        b, _, full = infer(base, z, relation, pb, pe, q, s, labels, weights)
        probs.append((b, full))
        truths.append(labels[q])
        groups.extend([str(acquisition)] * len(q))
    if not probs:
        return None, 0
    b = np.concatenate([item[0] for item in probs])
    f = np.concatenate([item[1] for item in probs])
    y = np.concatenate(truths)
    x = gate_features(b, 2 * f - b)
    gate = fit_gate(x, b.argmax(1), f.argmax(1), y, np.asarray(groups))
    return gate, len(y)


def ship_metrics(frame, scores):
    items = pd.DataFrame({"ship": frame.mmsi.to_numpy(str), "truth": frame.class_id.to_numpy(int),
                          "scores": list(scores)})
    grouped = items.groupby("ship", sort=False).agg(truth=("truth", "first"),
        scores=("scores", lambda values: np.stack(values).mean(0)))
    return metrics(grouped.truth.to_numpy(int), np.stack(grouped.scores).argmax(1))


def evaluate(data, manifest, source_only=False):
    labels = data["labels"]
    source = np.r_[data["train"], data["val"]]
    g = pca(data["raw"]["G50"], source)
    d = pca(data["raw"]["D18"], source)
    h = pca(data["raw"]["G18"], source)
    c = pca(data["c4"], source)
    r = pca(data["zero"], source)
    base, original = design_block(c, np.arange(len(labels))), design_block(r, np.arange(len(labels)))
    z = design_block((g, d), np.arange(len(labels)))
    g18 = design_block((g, h), np.arange(len(labels)))
    relation = class_map(z[:, :-1], labels, data["train"])
    tasks = source_tasks(data, base, z, relation)
    weights, task_count = learn_relation(tasks)
    gate, gate_rows = source_gate(data, base, z, relation, weights)
    if source_only:
        return dict(relation_tasks=task_count, gate_rows=gate_rows)
    bh, rh = fit_head(c, labels, source), fit_head(r, labels, source)
    eh = fit_head((g, d), labels, source)
    gh = fit_head((g, h), labels, source)
    bp, rp, ep, gp = base @ bh, original @ rh, z @ eh, g18 @ gh
    query = data["query"]
    truth = labels[query]
    query_frame = data["query_frame"].copy()
    query_frame["class_id"] = truth
    rows = []
    for (origin, draw), support in data["support"].items():
        b, e, plain = infer(base, z, None, bp, ep, query, support, labels)
        _, _, fixed = infer(base, z, relation, bp, ep, query, support, labels)
        _, _, learned = infer(base, z, relation, bp, ep, query, support, labels, weights)
        _, _, raw_base = infer(original, z, None, rp, ep, query, support, labels)
        _, _, imagenet = infer(base, g18, None, bp, gp, query, support, labels)
        outputs = {"raw_base_plain": raw_base, "c4_base": b, "expert_plain": e,
                   "c4_g18_plain": imagenet, "c4_dual_plain": plain,
                   "c4_class_fixed": fixed, "c4_class_learned": learned}
        if gate is not None:
            x = gate_features(b, 2 * learned - b)
            p = gate_probability(gate, x)
            disagreement = b.argmax(1) != learned.argmax(1)
            adaptive = disagreement & (p >= .5)
            selected = np.flatnonzero(disagreement)
            rank = selected[np.argsort(-p[selected], kind="stable")[:int(np.floor(.05 * len(query)))]]
            budget5 = np.zeros(len(query), bool)
            budget5[rank] = True
            outputs["c4_adaptive"] = np.where(adaptive[:, None], learned, b)
            outputs["c4_budget5"] = np.where(budget5[:, None], learned, b)
        for arm, score in outputs.items():
            chip = metrics(truth, score.argmax(1))
            ship = ship_metrics(query_frame, score)
            rows.append(dict(dataset=data["dataset"], fold=data["fold"], origin=origin, draw=draw,
                             arm=arm, query_rows=len(query), query_ships=query_frame.mmsi.nunique(),
                             query_classes=len(np.unique(truth)), support_labels=len(support),
                             source_train_rows=len(data["train"]), source_val_rows=len(data["val"]),
                             relation_tasks=task_count, gate_source_rows=gate_rows,
                             selected_fraction=(float(adaptive.mean()) if arm == "c4_adaptive" else
                                                float(budget5.mean()) if arm == "c4_budget5" else np.nan),
                             chip_oa=chip["oa"], chip_ba=chip["ba"],
                             ship_oa=ship["oa"], ship_ba=ship["ba"]))
    return rows, dict(relation_weights=weights.tolist(), relation_tasks=task_count,
                      gate_source_rows=gate_rows, gate_events=None if gate is None else gate["events"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("oss1", "oss2"), required=True)
    parser.add_argument("--fold")
    args = parser.parse_args()
    dataset = args.dataset
    manifest = pd.read_csv(f"E:/Hermes/paper_tracking/sensitivity/opensarship{dataset[-1]}_transfer_20261007/target_manifest.csv")
    c4 = np.load(HERE / f"{dataset}_c4.npy", mmap_mode="r")
    zero = np.load(HERE / f"{dataset}_zero.npy", mmap_mode="r")
    assert len(manifest) == len(c4)
    output = HERE / dataset
    output.mkdir(exist_ok=True)
    for fold, cache, episode, prefix in episode_paths(dataset):
        if args.fold and fold != args.fold:
            continue
        print("FOLD", dataset, fold, flush=True)
        data = load_fold(dataset, fold, cache, episode, prefix, manifest, c4, zero)
        data["dataset"] = dataset
        with threadpool_limits(limits=2):
            rows, detail = evaluate(data, manifest)
        pd.DataFrame(rows).to_csv(output / f"{fold}_metrics.csv", index=False)
        (output / f"{fold}_fit.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
        print(pd.DataFrame(rows).groupby(["origin", "arm"])[["chip_oa", "chip_ba"]].mean().round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
