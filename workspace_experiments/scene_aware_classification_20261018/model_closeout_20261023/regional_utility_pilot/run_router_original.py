"""Matched region-strategy routing on source-only held-out predictions."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import json
import time
from pathlib import Path
from fractions import Fraction

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import softmax, logsumexp

HERE = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
STRATEGIES = ("base", "body", "full")
ROLES = ("utility_train", "selection", "evaluation")


def center_scores(scores):
    return scores - scores.mean(axis=-1, keepdims=True)


def utilities(scores, y):
    z = center_scores(scores)
    anchor = z.mean(axis=1)
    changed = np.broadcast_to(anchor[:, None, None, :], (len(z), 8, 3, 8)).copy()
    # Evaluate the actual class-coordinate replacement, including softmax coupling.
    for c in range(8):
        changed[:, c, :, c] = z[:, :, c]
    prob = softmax(changed, axis=-1)
    err = ((prob - np.eye(8)[y, None, None, :]) ** 2).sum(-1)
    return err[..., 0:1] - err


def inputs(scores, regional):
    z = center_scores(scores)
    # Neither labels nor identities enter the class-shared coordinator.
    own = np.moveaxis(z, 1, 2)
    competitor = np.stack([
        np.max(np.delete(z, c, axis=2), axis=2) for c in range(8)
    ], axis=1)
    return np.concatenate((regional, own, own - competitor), axis=-1)


def blend(scores, weights):
    z = center_scores(scores)
    delta = np.moveaxis(z - z[:, 0:1], 1, 2)
    return z[:, 0] + (weights * delta).sum(-1)


def class_weights(y):
    labels, counts = np.unique(y, return_counts=True)
    weights = np.zeros(8)
    weights[labels] = len(y) / (len(labels) * counts)
    return .5 + .5 * weights[y]


def metrics(y, s):
    pred = s.argmax(1)
    lp = s - logsumexp(s, axis=1, keepdims=True)
    recalls = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    loss = -lp[np.arange(len(y)), y]
    brier = ((np.exp(lp) - np.eye(8)[y]) ** 2).sum(1)
    return dict(n=len(y), classes=len(recalls), OA=100 * np.mean(pred == y),
                BA=100 * np.mean(recalls), J=50 * (np.mean(pred == y) + np.mean(recalls)),
                NLL=float(loss.mean()), Brier=float(brier.mean()),
                balanced_Brier=float(np.mean([brier[y == c].mean() for c in np.unique(y)])))


def exact_j(y, pred):
    labels = np.unique(y)
    return Fraction(int(np.sum(y == pred)), 2 * len(y)) + sum(
        Fraction(int(np.sum(pred[y == c] == c)), 2 * len(labels) * int(np.sum(y == c)))
        for c in labels)


class Router(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 12), nn.GELU(), nn.Linear(12, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        u = self.net(x)
        anchored = torch.cat((torch.zeros_like(u[..., :1]), u), dim=-1)
        return anchored, anchored.softmax(-1)


def evaluate_model(model, x, z):
    with torch.inference_mode():
        u, w = model(x)
        score = z[:, 0] + (w * (z - z[:, 0:1]).transpose(1, 2)).sum(-1)
    return score.numpy().astype(float), w.numpy().astype(float), u.numpy().astype(float)


def utility_metrics(y, score, weight, estimated):
    u = utilities(score, y)
    chosen = weight.argmax(-1)
    picked = np.take_along_axis(u, chosen[..., None], axis=-1)[..., 0]
    err = (estimated - u) ** 2
    return dict(utility_mse=float(err[..., 1:].mean()),
                utility_proxy_regret=float((u.max(-1) - picked).mean()),
                proxy_best_strategy_fraction=float((picked >= u.max(-1) - 1e-12).mean()))


def load_data(directory, seed):
    table = pd.read_csv(directory / "table.csv", dtype={"mmsi": str})
    with np.load(directory / f"strategies_seed{seed}.npz") as f:
        scores = f["logits"].astype(float)
        raw = f["regional_raw"].astype(float)
        indices = f["indices"]
        role = f["role"].astype(str)
    assert scores.shape == (len(table), 3, 8)
    assert raw.shape[:2] == (len(table), 8)
    np.testing.assert_array_equal(table.row_index.to_numpy(), indices)
    np.testing.assert_array_equal(table.role.to_numpy(), role)
    assert table.instance_id.is_unique
    assert np.isfinite(scores).all() and np.isfinite(raw).all()
    split = {r: np.flatnonzero(role == r) for r in ROLES}
    assert all(len(ix) for ix in split.values()), {r: len(ix) for r, ix in split.items()}
    for a, b in ((ROLES[0], ROLES[1]), (ROLES[0], ROLES[2]), (ROLES[1], ROLES[2])):
        for key in ("mmsi", "pass_group"):
            assert not set(table.iloc[split[a]][key]) & set(table.iloc[split[b]][key]), (a, b, key)
    return table, scores, raw, split


def fixed_candidates():
    yield np.array([1/3, 1/3, 1/3], float)
    for a in range(5):
        for b in range(5 - a):
            yield np.array([a, b, 4 - a - b], float) / 4


def prediction_frame(table, ix, seed, arm, scores, weights, epoch):
    f = table.iloc[ix].copy()
    f["seed"], f["arm"], f["epoch"] = seed, arm, epoch
    f["prediction"] = scores.argmax(1)
    for c in range(8):
        f[f"s{c}"] = scores[:, c]
        for k, name in enumerate(STRATEGIES):
            f[f"w_{name}_{c}"] = weights[:, c, k]
    return f


def run(directory):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out = HERE / "router_results"
    out.mkdir(exist_ok=True)
    if (out / "complete.json").exists():
        raise RuntimeError("Completed pilot exists; preserve its original results.")
    tick = time.perf_counter()
    curves, selected, all_metrics, predictions, checks, opportunity = [], [], [], [], [], []
    for seed in SEEDS:
        table, score, regional, split = load_data(directory, seed)
        y = table.class_id.to_numpy(int)
        tr, va = split[ROLES[0]], split[ROLES[1]]
        raw = inputs(score, regional)
        mu = raw[tr].mean((0, 1), keepdims=True)
        scale = np.maximum(raw[tr].std((0, 1), keepdims=True), 1e-6)
        x = torch.tensor((raw - mu) / scale, dtype=torch.float32)
        z = torch.tensor(center_scores(score), dtype=torch.float32)
        truth = utilities(score[tr], y[tr])
        u_scale = max(float(np.sqrt(np.mean(truth[..., 1:] ** 2))), 1e-3)
        target = torch.tensor(truth / u_scale, dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.long)
        loss_weights = torch.tensor(class_weights(y[tr]), dtype=torch.float32)

        best_fixed, best_j = None, Fraction(-1)
        for w in fixed_candidates():
            s = blend(score[va], np.broadcast_to(w, (len(va), 8, 3)))
            j = exact_j(y[va], s.argmax(1))
            if j > best_j:
                best_j, best_fixed = j, w.copy()
        selected.append(dict(seed=seed, arm="fixed", epoch=-1,
                             weights=best_fixed.tolist(), selection_J=float(best_j)*100))
        static = [(name, np.eye(3)[k]) for k, name in enumerate(STRATEGIES)]
        static += [("equal", np.ones(3)/3), ("fixed", best_fixed)]
        initial_state = None
        trained_models = {}
        for arm, lam in (("ce_only", 0.), ("utility", .1)):
            torch.manual_seed(seed)
            model = Router(raw.shape[-1])
            if initial_state is not None:
                assert all(torch.equal(v, initial_state[k]) for k, v in model.state_dict().items())
            initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_state = {k: v.clone() for k, v in initial_state.items()}
            s, w, _ = evaluate_model(model, x[va], z[va])
            np.testing.assert_allclose(w, 1/3, atol=1e-7)
            best_j, best_epoch = exact_j(y[va], s.argmax(1)), 0
            curves.append(dict(seed=seed, arm=arm, epoch=0, loss=np.nan,
                               ce=np.nan, utility_loss=np.nan, **metrics(y[va], s)))
            opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
            for epoch in range(1, 201):
                opt.zero_grad()
                predicted, weights = model(x[tr])
                s = z[tr, 0] + (weights * (z[tr] - z[tr, 0:1]).transpose(1, 2)).sum(-1)
                ce = (nn.functional.cross_entropy(s, yt, reduction="none") * loss_weights).mean()
                aux = (((predicted[..., 1:] - target[..., 1:]) ** 2).mean((1, 2))*loss_weights).mean()
                # A fixed warm-up teaches utility without moving any strategy classifier.
                loss = aux if arm == "utility" and epoch <= 20 else ce + lam * aux
                assert torch.isfinite(loss)
                loss.backward()
                opt.step()
                if epoch % 5:
                    continue
                sv, _, _ = evaluate_model(model, x[va], z[va])
                j = exact_j(y[va], sv.argmax(1))
                curves.append(dict(seed=seed, arm=arm, epoch=epoch, loss=float(loss.detach()),
                                   ce=float(ce.detach()), utility_loss=float(aux.detach()), **metrics(y[va], sv)))
                if j > best_j:
                    best_j, best_epoch = j, epoch
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            trained_models[arm] = (model, best_epoch)
            torch.save(dict(state=best_state, mu=mu, scale=scale, u_scale=u_scale,
                            seed=seed, arm=arm, epoch=best_epoch, input_dim=raw.shape[-1]),
                       out / f"{arm}_seed{seed}.pt")
            selected.append(dict(seed=seed, arm=arm, epoch=best_epoch,
                                 parameters=sum(p.numel() for p in model.parameters()),
                                 selection_J=float(best_j)*100, u_scale=u_scale))
            for role, ix in split.items():
                if role == "evaluation":
                    continue
                s, weights, estimated = evaluate_model(model, x[ix], z[ix])
                all_metrics.append(dict(seed=seed, role=role, arm=arm, epoch=best_epoch,
                                        **metrics(y[ix], s),
                                        **utility_metrics(y[ix], score[ix], weights, estimated*u_scale)))
                predictions.append(prediction_frame(table, ix, seed, arm, s, weights, best_epoch))
            checks.append(dict(seed=seed, arm=arm, same_initialization=True,
                               input_no_labels=True, fit_scaler_role="utility_train",
                               candidate_fits=0, evaluation_used_for_selection=False))
            print(seed, arm, "epoch", best_epoch, "validation_J", float(best_j)*100, flush=True)
        # Evaluate the final source holdout only after both selectors are frozen.
        for role, ix in split.items():
            correct = score[ix].argmax(2) == y[ix, None]
            opportunity.append(dict(seed=seed, role=role, n=len(ix),
                                    oracle_OA=100*correct.any(1).mean(),
                                    best_single_OA=100*correct.mean(0).max(),
                                    strategies_disagree=int(np.sum(np.ptp(score[ix].argmax(2), axis=1)>0)),
                                    body_only_correct=int(np.sum(correct[:, 1] & ~correct[:, 2])),
                                    full_only_correct=int(np.sum(correct[:, 2] & ~correct[:, 1]))))
            for name, w in static:
                weights = np.broadcast_to(w, (len(ix), 8, 3))
                s = blend(score[ix], weights)
                all_metrics.append(dict(seed=seed, role=role, arm=name, epoch=-1, **metrics(y[ix], s)))
                predictions.append(prediction_frame(table, ix, seed, name, s, weights, -1))
            if role == "evaluation":
                for arm, (model, epoch) in trained_models.items():
                    s, weights, estimated = evaluate_model(model, x[ix], z[ix])
                    all_metrics.append(dict(seed=seed, role=role, arm=arm, epoch=epoch,
                                            **metrics(y[ix], s),
                                            **utility_metrics(y[ix], score[ix], weights, estimated*u_scale)))
                    predictions.append(prediction_frame(table, ix, seed, arm, s, weights, epoch))
        for name, rows in (("metrics", all_metrics), ("selections", selected),
                           ("curves", curves), ("checks", checks), ("opportunity", opportunity)):
            pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    (out / "complete.json").write_text(json.dumps(dict(seconds=time.perf_counter()-tick,
        seeds=list(SEEDS), trained_heads=6, steps=1200, target_test_used=False,
        lambda_utility=.1, warmup_steps=20, total_steps=200,
        utility_definition="full_vector_brier_of_one_class_replacement_at_equal_anchor"), indent=2), encoding="utf-8")
    print(pd.DataFrame(all_metrics).query("role == 'evaluation'").to_string(index=False))


def self_test():
    rng = np.random.default_rng(18)
    score = rng.normal(size=(4, 3, 8))
    raw = rng.normal(size=(4, 8, 69))
    y = np.array([0, 3, 6, 7])
    weights = softmax(rng.normal(size=(4, 8, 3)), axis=-1)
    shifted = score + rng.normal(size=(4, 3, 1))*10
    np.testing.assert_allclose(blend(score, weights), blend(shifted, weights), atol=1e-12)
    np.testing.assert_allclose(inputs(score, raw), inputs(shifted, raw), atol=1e-12)
    for k in range(3):
        w = np.broadcast_to(np.eye(3)[k], (4, 8, 3))
        np.testing.assert_allclose(blend(score, w), center_scores(score)[:, k], atol=1e-12)
    np.testing.assert_allclose(blend(score, np.ones((4, 8, 3))/3), center_scores(score).mean(1))
    assert np.all(utilities(score, y)[:, :, 0] == 0)
    identical = np.repeat(score[:, :1], 3, axis=1)
    assert np.all(utilities(identical, y) == 0)
    assert np.isclose(class_weights(y).mean(), 1.)
    # A known correct strategy must have smaller Brier error than a wrong confident one.
    toy = np.zeros((1, 3, 8)); toy[0, 0, 1] = 8; toy[0, 1, 0] = 8; toy[0, 2, 1] = 8
    assert utilities(toy, np.array([0]))[0, 0, 1] > 0
    assert np.all(utilities(toy, np.array([0]))[0, :, 2] == 0)
    # Independently construct the affected complete decision vector, not a per-class error.
    for i in range(len(score)):
        for c in range(8):
            losses = []
            for k in range(3):
                v = center_scores(score[i]).mean(0)
                v[c] = center_scores(score[i])[k, c]
                p = softmax(v)
                losses.append(float(np.sum((p-np.eye(8)[y[i]])**2)))
            np.testing.assert_allclose(utilities(score, y)[i,c], np.array(losses[0])-losses, atol=1e-12)
    print("PASS: candidate-order endpoints, shift invariance, complete-decision intervention utility, neutral strategy, uniform fusion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data", type=Path, default=HERE/"upstream")
    args = parser.parse_args()
    self_test() if args.self_test else run(args.data)
