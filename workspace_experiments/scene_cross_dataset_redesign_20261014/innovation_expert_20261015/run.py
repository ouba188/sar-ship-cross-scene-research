"""Frozen-feature test of R18 evidence not reconstructible from R50."""

import argparse
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

PUBLIC = Path(__file__).resolve().parents[1] / "public_full_20261014"
sys.path.insert(0, str(PUBLIC))
import run as protocol  # noqa: E402


HERE = Path(__file__).resolve().parent
ARMS = ("full", "reconstructed", "innovation", "single_backbone")


def ridge_predict(x_train, y_train, x_test):
    x_mean, y_mean = x_train.mean(0), y_train.mean(0)
    x_scale = x_train.std(0).clip(1e-6)
    y_scale = y_train.std(0).clip(1e-6)
    model = Ridge(alpha=.01 * len(x_train), fit_intercept=False, solver="cholesky")
    model.fit((x_train - x_mean) / x_scale, (y_train - y_mean) / y_scale)
    return (model.predict((x_test - x_mean) / x_scale) * y_scale + y_mean).astype(np.float32)


def mlp_predict(x_train, y_train, x_test):
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    device = "cuda" if x_train.shape[1] > 512 and torch.cuda.is_available() else "cpu"
    width, epochs = (512, 8) if x_train.shape[1] > 512 else (256, 25)
    x_mean, y_mean = x_train.mean(0), y_train.mean(0)
    raw_scale = x_train.std(0)
    x_scale = raw_scale.clip(max(1e-6, .1 * np.median(raw_scale)))
    y_scale = y_train.std(0).clip(1e-6)
    x = torch.from_numpy(((x_train - x_mean) / x_scale).astype(np.float32)).to(device)
    y = torch.from_numpy(((y_train - y_mean) / y_scale).astype(np.float32)).to(device)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], width), torch.nn.GELU(),
                                torch.nn.Linear(width, y.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    for _ in range(epochs):
        for batch in torch.randperm(len(x)).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x[batch]) - y[batch]) ** 2)
            loss.backward()
            optimizer.step()
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in np.array_split(np.arange(len(x_test)), max(1, int(np.ceil(len(x_test) / 1024)))):
            normalized = torch.from_numpy(((x_test[batch] - x_mean) / x_scale).astype(np.float32))
            predictions.append(model(normalized.to(device)).cpu().numpy())
    return (np.concatenate(predictions) * y_scale + y_mean).astype(np.float32)


def reconstruct_source_held(g, d, source, acquisitions, mapper, raw_input=None):
    predict = ridge_predict if mapper == "linear" else mlp_predict
    inputs = g if raw_input is None else raw_input
    predicted = predict(inputs[source], d[source], inputs)
    oof = np.empty_like(d[source])
    for fit, hold in GroupKFold(n_splits=3).split(source, groups=acquisitions):
        oof[hold] = predict(inputs[source[fit]], d[source[fit]], inputs[source[hold]])
    predicted[source] = oof
    residual = d - predicted
    residual_energy = np.sum((d - d.mean(0)) ** 2, axis=1).mean()
    return predicted, residual, {
        "source_oof_mse": float(np.mean(residual[source] ** 2)),
        "source_oof_r2": float(1 - np.sum(residual[source] ** 2) /
                               np.sum((d[source] - d[source].mean(0)) ** 2)),
        "target_diagnostic_r2": None,
        "residual_energy_fraction_all_rows": float(np.mean(np.sum(residual ** 2, axis=1)) / residual_energy),
    }


def fixed_rival_margin(probability, truth, rival):
    index = np.arange(len(truth))
    return probability[index, truth] - probability[index, rival]


def evaluate_fold(dataset, fold, cache, episode, prefix, manifest, c4, zero, mapper):
    data = protocol.load_fold(dataset, fold, cache, episode, prefix, manifest, c4, zero)
    labels = data["labels"]
    source = np.r_[data["train"], data["val"]]
    acquisitions = np.r_[data["source_train"].acquisition.to_numpy(str),
                         data["source_val"].acquisition.to_numpy(str)]
    g = protocol.pca(data["raw"]["G50"], source)
    d = protocol.pca(data["raw"]["D18"], source)
    c = protocol.pca(data["c4"], source)
    raw_input = None
    if mapper == "raw_mlp":
        raw_input = np.concatenate((data["raw"]["G50"], data["c4"]), axis=1)
    reconstructed, innovation, diagnostics = reconstruct_source_held(
        g, d, source, acquisitions, mapper, raw_input)
    diagnostics["target_diagnostic_r2"] = float(
        1 - np.sum(innovation[data["query"]] ** 2) /
        np.sum((d[data["query"]] - d[source].mean(0)) ** 2))
    base = protocol.design_block(c, np.arange(len(labels)))
    base_prior = base @ protocol.fit_head(c, labels, source)
    expert = {}
    for name, second in (("full", d), ("reconstructed", reconstructed),
                         ("innovation", innovation), ("single_backbone", c)):
        z = protocol.design_block((g, second), np.arange(len(labels)))
        prior = z @ protocol.fit_head((g, second), labels, source)
        expert[name] = z, prior

    query = data["query"]
    truth = labels[query]
    frame = data["query_frame"].copy()
    frame["class_id"] = truth
    archived = pd.read_csv(PUBLIC / dataset / f"{fold}_metrics.csv")
    metrics, cases = [], []
    for (origin, draw), support in data["support"].items():
        scores = {}
        pb = None
        for name in ARMS:
            z, prior = expert[name]
            base_probability, _, fused = protocol.infer(
                base, z, None, base_prior, prior, query, support, labels)
            pb = base_probability
            scores[name] = fused
        reference = archived[(archived.origin == origin) & (archived.draw == draw) &
                             (archived.arm == "c4_dual_plain")].iloc[0]
        measured = protocol.metrics(truth, scores["full"].argmax(1))
        np.testing.assert_allclose(
            [measured["oa"], measured["ba"]],
            [reference.chip_oa, reference.chip_ba], rtol=0, atol=1e-8)
        support_count = np.bincount(labels[support], minlength=8)
        rival_scores = pb.copy()
        rival_scores[np.arange(len(truth)), truth] = -np.inf
        rival = rival_scores.argmax(1)
        for name, probability in (("base", pb), *scores.items()):
            pred = probability.argmax(1)
            chip = protocol.metrics(truth, pred)
            ship = protocol.ship_metrics(frame, probability)
            metrics.append(dict(dataset=dataset, fold=fold, origin=origin, draw=draw,
                                arm=name, chip_oa=chip["oa"], chip_ba=chip["ba"],
                                ship_oa=ship["oa"], ship_ba=ship["ba"],
                                query_rows=len(query), query_ships=frame.mmsi.nunique(),
                                source_rows=len(source), support_missing_classes=int((support_count == 0).sum())))
            margin = fixed_rival_margin(probability, truth, rival)
            cases.append(pd.DataFrame(dict(dataset=dataset, fold=fold, origin=origin,
                                           draw=draw, arm=name,
                                           sample_id=frame.sample_id.to_numpy(str),
                                           mmsi=frame.mmsi.to_numpy(str),
                                           acquisition=frame.acquisition.to_numpy(str),
                                           truth=truth, prediction=pred, rival=rival,
                                           true_rival_margin=margin,
                                           true_class_support_count=support_count[truth])))
    root = HERE if mapper == "linear" else HERE / ("raw_mlp_scalefix_single" if mapper == "raw_mlp" else mapper)
    fold_dir = root / dataset / fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(fold_dir / "metrics.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(fold_dir / "cases.csv", index=False)
    pd.Series(diagnostics).to_json(fold_dir / "mapper_diagnostics.json", indent=2)
    print(dataset, fold, pd.DataFrame(metrics).groupby(["origin", "arm"])
          [["chip_oa", "chip_ba"]].mean().round(2).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("oss1", "oss2"))
    parser.add_argument("--fold")
    parser.add_argument("--mapper", choices=("linear", "mlp", "raw_mlp"), default="linear")
    args = parser.parse_args()
    datasets = (args.dataset,) if args.dataset else ("oss1", "oss2")
    for dataset in datasets:
        manifest = pd.read_csv(
            f"E:/Hermes/paper_tracking/sensitivity/opensarship{dataset[-1]}_transfer_20261007/target_manifest.csv")
        c4 = np.load(PUBLIC / f"{dataset}_c4.npy", mmap_mode="r")
        zero = np.load(PUBLIC / f"{dataset}_zero.npy", mmap_mode="r")
        for fold, cache, episode, prefix in protocol.episode_paths(dataset):
            if args.fold and args.fold != fold:
                continue
            with threadpool_limits(limits=2):
                evaluate_fold(dataset, fold, cache, episode, prefix, manifest, c4, zero, args.mapper)


if __name__ == "__main__":
    main()
