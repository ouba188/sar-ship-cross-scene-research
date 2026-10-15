"""Bounded paired training of body/ring readouts on source development tasks."""

import argparse
import copy
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from full_readout import infer_with_local as numpy_infer
from body_context_gate import Gate, readout

HERE = Path(__file__).resolve().parent
ROOT = HERE / "body_context"
FIELDS = ("base", "expert", "prior_b", "prior_e", "relation", "ship", "ring", "labels")


def balanced_loss(prob, y):
    losses = -torch.log(prob[torch.arange(len(y)), y].clamp_min(1e-12))
    return torch.stack([losses[y == c].mean() for c in torch.unique(y)]).mean()


def packed_task(arrays, task):
    q, s = np.asarray(task["q"]), np.asarray(task["s"])
    assert not set(q) & set(s)
    rows = np.r_[s, q]
    result = {key: torch.as_tensor(arrays[key][rows], dtype=(
        torch.long if key == "labels" else torch.float64)) for key in FIELDS}
    result["weights"] = torch.as_tensor(arrays["weights"], dtype=torch.float64)
    result["s"] = torch.arange(len(s))
    result["q"] = torch.arange(len(s), len(rows))
    return result


def predict(task, local):
    return readout(task["base"], task["expert"], task["prior_b"], task["prior_e"],
        task["labels"], task["q"], task["s"], local=local,
        relation=task["relation"], weights=task["weights"])


def additive(s, b):
    return torch.cat((np.sqrt(.5) * s, np.sqrt(.5) * b), dim=1)


def shuffle_gate_input(task, meta, arrays, seed):
    rows = np.r_[task["s"], task["q"]]
    b = arrays["ring"][rows].copy()
    rng = np.random.default_rng(seed)
    singletons = 0
    for offset, values in ((0, task["s"]), (len(task["s"]), task["q"])):
        frame = meta.iloc[values].reset_index(drop=True)
        for _, group in frame.groupby("acquisition", sort=True):
            pos = group.index.to_numpy(int) + offset
            singletons += int(len(pos) == 1)
            b[pos] = b[rng.permutation(pos)]
    return torch.as_tensor(b, dtype=torch.float64), singletons


def ap_scores(similarity, truth, support_y):
    ap = np.full(len(truth), np.nan)
    for i, cls in enumerate(truth):
        n = int(np.sum(support_y == cls))
        if n:
            hits = support_y[np.argsort(-similarity[i], kind="stable")[:n]] == cls
            ap[i] = np.sum(np.cumsum(hits) / np.arange(1, n + 1) * hits) / n
    return ap


def evaluate(arrays, tasks, packed, meta, arm, gate=None, seed=-1, shuffle=None, predictions=False):
    records, detailed = [], []
    with torch.no_grad():
        for i, (info, task) in enumerate(zip(tasks, packed)):
            s, b = task["ship"], task["ring"]
            singletons = 0
            if gate is not None:
                gate_b = None
                if shuffle is not None:
                    gate_b, singletons = shuffle_gate_input(info, meta, arrays, 20261030 + shuffle)
                local = gate(s, b, gate_b=gate_b)
            else:
                local = {"original": None, "ship": s, "ring": b,
                         "additive": additive(s, b)}[arm]
            probability = predict(task, local).numpy()
            y = task["labels"][task["q"]].numpy()
            pred = probability.argmax(1)
            classes = np.unique(y)
            correct = pred == y
            nll = -np.log(probability[np.arange(len(y)), y].clip(1e-12))
            brier = np.sum((probability - np.eye(8)[y]) ** 2, axis=1)
            expert = task["expert"].numpy()
            q, support = task["q"].numpy(), task["s"].numpy()
            similarity = expert[q] @ expert[support].T
            if local is not None:
                local_np = local.numpy()
                similarity += .25 * (16 * local_np[q] @ local_np[support].T -
                                     expert[q, 128:256] @ expert[support, 128:256].T)
            r, w = task["relation"].numpy(), arrays["weights"]
            similarity = .75 * similarity + .25 * (r[q] * w) @ r[support].T
            ap = ap_scores(similarity, y, task["labels"][support].numpy())
            row = {key: info[key] for key in ("port", "acquisition", "origin", "draw")}
            row.update(fold=info.get("fold", info["port"]), task=i, arm=arm, seed=seed,
                       shuffle=-1 if shuffle is None else shuffle, n=len(y),
                       oa=100 * correct.mean(), ba=100 * np.mean([correct[y == c].mean() for c in classes]),
                       nll=nll.mean(), macro_nll=np.mean([nll[y == c].mean() for c in classes]),
                       brier=brier.mean(), macro_brier=np.mean([brier[y == c].mean() for c in classes]),
                       map_r=np.nanmean(ap), macro_map_r=np.nanmean([
                           np.nanmean(ap[y == c]) for c in classes if np.isfinite(ap[y == c]).any()]),
                       retrieval_covered=int(np.isfinite(ap).sum()), shuffle_singletons=singletons)
            records.append(row)
            if predictions:
                margin_scores = probability.copy()
                margin_scores[np.arange(len(y)), y] = -np.inf
                margin = probability[np.arange(len(y)), y] - margin_scores.max(1)
                for k, pos in enumerate(info["q"]):
                    detailed.append(dict(fold=row["fold"], port=info["port"], acquisition=info["acquisition"],
                        task=i, origin=info["origin"], draw=info["draw"], arm=arm, seed=seed,
                        shuffle=row["shuffle"], sample_id=str(meta.iloc[pos].sample_id),
                        mmsi=str(meta.iloc[pos].mmsi), truth=int(y[k]), pred=int(pred[k]),
                        correct=bool(correct[k]), margin=float(margin[k]), ap=float(ap[k]),
                        nll=float(nll[k]), brier=float(brier[k]),
                        **{f"p{c}": float(probability[k,c]) for c in range(8)}))
    return pd.DataFrame(records), pd.DataFrame(detailed)


def check_real_parity(arrays, info, task):
    q, s = np.asarray(info["q"]), np.asarray(info["s"])
    local = np.concatenate((np.sqrt(.5) * arrays["ship"], np.sqrt(.5) * arrays["ring"]), axis=1)
    for candidate in (None, local):
        reference = numpy_infer(arrays["prior_b"], arrays["prior_e"], arrays["base"], arrays["expert"],
            q, s, arrays["labels"], candidate, 0. if candidate is None else .25,
            arrays["relation"], arrays["weights"])
        actual = predict(task, None if candidate is None else additive(task["ship"], task["ring"]))
        np.testing.assert_allclose(actual.detach().numpy(), reference, atol=1e-10, rtol=1e-10)


def train_one(folder, arrays, episodes, packed, meta, mode, seed):
    path = folder / f"{mode}_seed{seed}.pt"
    if path.exists():
        print("trained cache", folder.name, mode, seed, flush=True)
        return
    torch.manual_seed(seed)
    gate = Gate(mode=mode).double()
    optimizer = torch.optim.Adam(gate.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(20261015 + seed)
    best, best_step, best_state = -np.inf, 0, copy.deepcopy(gate.state_dict())
    curve = []
    started = time.perf_counter()
    with torch.no_grad():
        initial_fit_loss = np.mean([float(balanced_loss(predict(t, gate(t["ship"], t["ring"])),
            t["labels"][t["q"]])) for t in packed["fit"]])
    for step in range(601):
        if step in (0, 200, 400, 600):
            table, _ = evaluate(arrays, episodes["selection"], packed["selection"], meta, mode, gate, seed)
            result = table.loc[table.origin.eq("cross")].ba.mean()
            same = table.loc[table.origin.eq("same")].ba.mean()
            curve.append(dict(step=step, selection_cross_ba=result, selection_same_ba=same,
                              selection_macro_nll=table.loc[table.origin.eq("cross")].macro_nll.mean()))
            if result > best + 1e-12:
                best, best_step, best_state = result, step, copy.deepcopy(gate.state_dict())
            print("checkpoint", folder.name, mode, seed, step, "BA", round(result, 4), flush=True)
        if step == 600:
            break
        sampled = rng.choice(len(packed["fit"]), 4, replace=len(packed["fit"]) < 4)
        losses = []
        for index in sampled:
            task = packed["fit"][index]
            local = gate(task["ship"], task["ring"])
            losses.append(balanced_loss(predict(task, local), task["labels"][task["q"]]))
        loss = torch.stack(losses).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            curve.append(dict(step=step+1, train_loss=float(loss.detach())))
    last_state = copy.deepcopy(gate.state_dict())
    with torch.no_grad():
        last_fit_loss = np.mean([float(balanced_loss(predict(t, gate(t["ship"], t["ring"])),
            t["labels"][t["q"]])) for t in packed["fit"]])
        last_table, last_preds = evaluate(arrays, episodes["selection"], packed["selection"], meta,
                                          mode, gate, seed, predictions=True)
        all_ring = torch.as_tensor(arrays["ring"], dtype=torch.float64)
        logits = gate.c[None, :].expand(len(all_ring), -1)
        if mode == "conditional":
            logits = all_ring @ gate.W.T + gate.c
        adjustment = .25 * torch.tanh(logits)
        diagnostics = dict(initial_fit_nll=initial_fit_loss, last_fit_nll=last_fit_loss,
                           adjustment_abs_mean=float(adjustment.abs().mean()),
                           adjustment_abs_max=float(adjustment.abs().max()),
                           adjustment_across_sample_std=float(adjustment.std(dim=0).mean()),
                           parameter_norm=float(torch.cat([p.flatten() for p in gate.parameters()]).norm()))
    torch.save(dict(state=last_state, mode=mode, seed=seed, step=600, diagnostics=diagnostics),
               folder / f"{mode}_seed{seed}_last.pt")
    last_table.to_csv(folder / f"{mode}_seed{seed}_last_selection.csv", index=False)
    last_preds.to_csv(folder / f"{mode}_seed{seed}_last_predictions.csv", index=False)
    gate.load_state_dict(best_state)
    torch.save(dict(state=best_state, mode=mode, seed=seed, best_step=best_step,
                    selection_ba=best, seconds=time.perf_counter()-started), path)
    pd.DataFrame(curve).to_csv(folder / f"{mode}_seed{seed}_curve.csv", index=False)
    table, preds = evaluate(arrays, episodes["selection"], packed["selection"], meta,
                            mode, gate, seed, predictions=True)
    tables, details = [table], [preds]
    if mode == "conditional":
        for shuffle in range(3):
            t, p = evaluate(arrays, episodes["selection"], packed["selection"], meta,
                             mode, gate, seed, shuffle=shuffle, predictions=True)
            tables.append(t)
            details.append(p)
    pd.concat(tables).to_csv(folder / f"{mode}_seed{seed}_selection.csv", index=False)
    pd.concat(details).to_csv(folder / f"{mode}_seed{seed}_selection_predictions.csv", index=False)
    print("trained", folder.name, mode, seed, "best step", best_step,
          "seconds", round(time.perf_counter()-started, 1), flush=True)


def run(folder, seeds, target):
    assert (folder / "prepared.json").exists()
    with np.load(folder / "readout.npz") as cache:
        arrays = {key: cache[key] for key in (*FIELDS, "weights")}
    meta = pd.read_csv(folder / "metadata.csv", dtype={"mmsi": str})
    assert np.array_equal(meta.position.to_numpy(), np.arange(len(meta)))
    episodes = json.loads((folder / "episodes.json").read_text(encoding="utf-8"))
    packed = {name: [packed_task(arrays, t) for t in tasks] for name, tasks in episodes.items()}
    for group in episodes.values():
        for task in group:
            assert not set(meta.iloc[task["q"]].mmsi) & set(meta.iloc[task["s"]].mmsi)
    check_real_parity(arrays, episodes["selection"][0], packed["selection"][0])
    gate = Gate(mode="conditional").double()
    task = packed["selection"][0]
    np.testing.assert_allclose(predict(task, gate(task["ship"], task["ring"])).detach().numpy(),
                               predict(task, additive(task["ship"], task["ring"])).numpy(),
                               atol=1e-10, rtol=1e-10)
    if not target:
        tables, predictions = [], []
        for arm in ("original", "ship", "ring", "additive"):
            t, p = evaluate(arrays, episodes["selection"], packed["selection"], meta, arm, predictions=True)
            tables.append(t)
            predictions.append(p)
        pd.concat(tables).to_csv(folder / "baselines_selection.csv", index=False)
        pd.concat(predictions).to_csv(folder / "baselines_selection_predictions.csv", index=False)
        for seed in seeds:
            for mode in ("static", "conditional"):
                train_one(folder, arrays, episodes, packed, meta, mode, seed)
        return
    tables, predictions = [], []
    for arm in ("original", "ship", "ring", "additive"):
        t, p = evaluate(arrays, episodes["target"], packed["target"], meta, arm, predictions=True)
        tables.append(t)
        predictions.append(p)
    for seed in seeds:
        for mode in ("static", "conditional"):
            saved = torch.load(folder / f"{mode}_seed{seed}.pt", weights_only=False)
            gate = Gate(mode=mode).double()
            gate.load_state_dict(saved["state"])
            for shuffle in ([None, 0, 1, 2] if mode == "conditional" else [None]):
                t, p = evaluate(arrays, episodes["target"], packed["target"], meta,
                                 mode, gate, seed, shuffle=shuffle, predictions=True)
                tables.append(t)
                predictions.append(p)
    pd.concat(tables).to_csv(folder / "target_metrics.csv", index=False)
    pd.concat(predictions).to_csv(folder / "target_predictions.csv", index=False)
    print("target complete", folder.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--target", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for folder in sorted(ROOT.iterdir()):
            if folder.is_dir() and (not args.port or folder.name == args.port.replace(" ", "_")):
                run(folder, args.seeds, args.target)


if __name__ == "__main__":
    main()
