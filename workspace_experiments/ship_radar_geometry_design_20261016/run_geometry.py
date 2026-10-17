"""Train local readers through the existing full support solve, then freeze/evaluate."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "prepool_local_probe_20261015"
ROOT = OLD / "body_context_expansion"
sys.path.insert(0, str(OLD))
from body_context_gate import readout
from geometry_reader import GeometryReader

FIELDS = ("base", "expert", "prior_b", "prior_e", "relation", "ship", "labels")
GEO = ("ship_axis_rad", "radar_axis_rad", "incidence_rad", "ship_valid", "radar_valid", "incidence_valid", "ship_confidence")
DEVICE = "cuda"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def source_pools():
    return json.loads((ROOT / "ring_relaxed/query_coverage/frozen.json").read_text(encoding="utf-8"))["pools"]


def load_data(port, target=False):
    slug = port.replace(" ", "_")
    folder = (ROOT if target else OLD / "body_context") / slug
    meta = pd.read_csv(folder / "metadata.csv", dtype={"mmsi": str})
    with np.load(folder / "readout.npz") as z:
        arrays = {k: z[k] for k in (*FIELDS, "weights")}
    if target:
        tasks = {"target": json.loads((folder / "tasks.json").read_text(encoding="utf-8"))}
    else:
        pool = next(p for p in source_pools() if p["pool"] == port)
        tasks = {k: [t for t in v if t["budget"] == 32] for k, v in pool["splits"].items()}
    used = np.array(sorted({i for ts in tasks.values() for t in ts for i in t["q"] + t["s"]}))
    remap = {int(old): i for i, old in enumerate(used)}
    meta = meta.iloc[used].reset_index(drop=True)
    arrays = {k: (v if k == "weights" else v[used]) for k, v in arrays.items()}
    tasks = {k: [{**t, "q": [remap[i] for i in t["q"]], "s": [remap[i] for i in t["s"]]} for t in ts] for k, ts in tasks.items()}
    with np.load(HERE / "spatial_cache" / (slug + ".npz")) as z:
        pos = np.searchsorted(z["rows"], meta.global_row)
        np.testing.assert_array_equal(z["rows"][pos], meta.global_row)
        spatial = {k: z[k][pos] for k in ("tokens", "positions", "areas", "edge")}
    geom = pd.read_csv(HERE / "geometry_metadata.csv").set_index("global_row").loc[meta.global_row]
    geo = geom[list(GEO)].to_numpy(np.float32)
    geo = np.nan_to_num(geo)
    assert np.isfinite(geo).all()
    spatial["geometry"] = geo
    for ts in tasks.values():
        for t in ts:
            assert not set(meta.iloc[t["q"]].mmsi) & set(meta.iloc[t["s"]].mmsi)
    if not target:
        role = {k: sorted({i for t in ts for i in t["q"] + t["s"]}) for k, ts in tasks.items()}
        assert not set(meta.iloc[role["fit"]].mmsi) & set(meta.iloc[role["selection"]].mmsi)
        assert not set(meta.iloc[role["fit"]].acquisition) & set(meta.iloc[role["selection"]].acquisition)
        assert not meta.port.eq(port).any(), "Held port entered local module training"
    tensor = {k: torch.as_tensor(v, device=DEVICE, dtype=torch.long if k == "labels" else torch.float64) for k, v in arrays.items()}
    tensor.update({k: torch.as_tensor(v, device=DEVICE, dtype=torch.float32) for k, v in spatial.items() if k != "edge"})
    return dict(meta=meta, tasks=tasks, tensors=tensor, spatial=spatial)


def encode(model, d, rows):
    return model(*(d[k][rows] for k in ("tokens", "positions", "areas", "geometry")))


def prediction(d, info, context=None):
    q, s = torch.as_tensor(info["q"], device=DEVICE), torch.as_tensor(info["s"], device=DEVICE)
    rows = torch.cat((s, q))
    body = d["ship"][rows]
    local = body if context is None else torch.cat((body * np.sqrt(.5), context.double() * np.sqrt(.5)), 1)
    prob = readout(*(d[k][rows] for k in ("base", "expert", "prior_b", "prior_e", "labels")),
                   torch.arange(len(s), len(rows), device=DEVICE), torch.arange(len(s), device=DEVICE),
                   relation=d["relation"][rows], weights=d["weights"], local=local)
    return prob, local, rows


def loss_fn(prob, labels):
    nll = -prob[torch.arange(len(labels), device=DEVICE), labels].clamp_min(1e-12).log()
    return torch.stack([nll[labels == c].mean() for c in labels.unique()]).mean()


def task_metric(prob, y):
    correct = prob.argmax(1) == y
    classes = np.unique(y)
    nll = -np.log(prob[np.arange(len(y)), y].clip(1e-12))
    brier = ((prob - np.eye(8)[y]) ** 2).sum(1)
    return dict(oa=100 * correct.mean(), ba=100 * np.mean([correct[y == c].mean() for c in classes]),
                nll=nll.mean(), brier=brier.mean(), macro_nll=np.mean([nll[y == c].mean() for c in classes]),
                macro_brier=np.mean([brier[y == c].mean() for c in classes]))


def hierarchy(table):
    return table.groupby(["port", "acquisition"])[["oa", "ba", "macro_nll", "macro_brier"]].mean().groupby("port").mean().mean()


@torch.no_grad()
def contexts(model, data):
    d = data["tensors"]
    return torch.cat([encode(model, d, slice(i, i + 64)) for i in range(0, len(data["meta"]), 64)])


@torch.no_grad()
def evaluate(model, data, split, detailed=False):
    d = data["tensors"]
    context = contexts(model, data) if model is not None else None
    records, details = [], []
    for ti, t in enumerate(data["tasks"][split]):
        rows = t["s"] + t["q"]
        prob, local, _ = prediction(d, t, context[rows] if context is not None else None)
        prob = prob.cpu().numpy()
        y = d["labels"][t["q"]].cpu().numpy()
        common = {k: t[k] for k in ("port", "acquisition", "origin", "budget", "draw", "fold")}
        common["task"] = ti
        records.append({**common, **task_metric(prob, y), "n": len(y)})
        if detailed:
            n = len(t["s"])
            e = d["expert"][rows]
            r, w = d["relation"][rows], d["weights"]
            similarity = .75 * (e[n:] @ e[:n].T + .25 * (16 * local[n:] @ local[:n].T - e[n:, 128:256] @ e[:n, 128:256].T)) + .25 * ((r[n:] * w) @ r[:n].T)
            sim = similarity.cpu().numpy()
            sy = d["labels"][t["s"]].cpu().numpy()
            frame = data["meta"].iloc[t["q"]][["sample_id", "mmsi", "global_row", "class_id"]].copy().reset_index(drop=True)
            for k, v in common.items():
                frame[k] = v
            frame["truth"], frame["pred"] = y, prob.argmax(1)
            frame["correct"] = frame.truth.eq(frame.pred)
            frame["nll"] = -np.log(prob[np.arange(len(y)), y].clip(1e-12))
            frame["brier"] = ((prob - np.eye(8)[y]) ** 2).sum(1)
            geo = data["spatial"]["geometry"][t["q"]]
            frame["geometry_valid"] = (geo[:, 3:6] > 0).all(1)
            frame["edge"] = data["spatial"]["edge"][t["q"]]
            frame["support_geometry_fraction"] = (data["spatial"]["geometry"][t["s"], 3:6] > 0).all(1).mean()
            ap, margins = [], []
            for i, cls in enumerate(y):
                count = int((sy == cls).sum())
                if count:
                    hits = sy[np.argsort(-sim[i], kind="stable")[:count]] == cls
                    ap.append(np.sum(np.cumsum(hits) / np.arange(1, count + 1) * hits) / count)
                    margins.append(sim[i, sy == cls].mean() - sim[i, sy != cls].mean() if (sy != cls).any() else np.nan)
                else:
                    ap.append(np.nan)
                    margins.append(np.nan)
            frame["ap"], frame["class_kernel_margin"] = ap, margins
            for c in range(8):
                frame[f"p{c}"] = prob[:, c]
            details.append(frame)
    return pd.DataFrame(records), pd.concat(details, ignore_index=True) if details else None


def sampled_task(tasks, rng, groups):
    port = list(groups)[int(rng.integers(len(groups)))]
    scenes = groups[port]
    acq = list(scenes)[int(rng.integers(len(scenes)))]
    choices = scenes[acq]
    t = copy.deepcopy(tasks[int(rng.choice(choices))])
    if len(t["q"]) > 32:
        t["q"] = rng.choice(t["q"], 32, replace=False).tolist()
    return t


def train(port, mode, seed, steps, smoke=False):
    out = HERE / ("smoke" if smoke else "trained") / port.replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{mode}_seed{seed}.pt"
    if path.exists():
        print("TRAIN CACHE", path, flush=True)
        return
    data = load_data(port)
    d = data["tensors"]
    torch.manual_seed(20261016 + seed)
    model = GeometryReader(mode=mode).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(20261016 + seed)
    groups = {}
    for i, t in enumerate(data["tasks"]["fit"]):
        groups.setdefault(t["port"], {}).setdefault(t["acquisition"], []).append(i)
    curve, best_key, best_state = [], None, None
    checkpoints = sorted(set([0, steps] + ([50, 100] if not smoke else [])))
    started = time.perf_counter()
    first_grad = None
    exposed_queries, exposed_supports, exposed_tasks = set(), set(), set()
    for step in range(steps + 1):
        if step in checkpoints:
            model.eval()
            table, _ = evaluate(model, data, "selection")
            result = hierarchy(table)
            key = (float(result.ba), -float(result.macro_nll), -step)
            if best_key is None or key > best_key:
                best_key, best_step = key, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            curve.append(dict(step=step, **{"selection_" + k: float(v) for k, v in result.items()}))
            if step in (0, steps):
                fit_table, _ = evaluate(model, data, "fit")
                fit_result = hierarchy(fit_table)
                curve[-1].update({"fit_" + k: float(v) for k, v in fit_result.items()})
            print("SELECT", port, mode, seed, step, "BA", round(result.ba, 5), "NLL", round(result.macro_nll, 5), flush=True)
        if step == steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(2):
            t = sampled_task(data["tasks"]["fit"], rng, groups)
            exposed_queries.update(t["q"])
            exposed_supports.update(t["s"])
            exposed_tasks.add((t["port"], t["acquisition"], t["support_acquisition"], t["draw"]))
            rows = t["s"] + t["q"]
            c = encode(model, d, rows)
            probability, _, _ = prediction(d, t, c)
            loss = loss_fn(probability, d["labels"][t["q"]]) / 2
            assert torch.isfinite(loss)
            loss.backward()
            losses.append(float(loss.detach()) * 2)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        if first_grad is None:
            first_grad = float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
            assert first_grad > 0
        optimizer.step()
        if step % 25 == 0:
            curve.append(dict(step=step + 1, train_nll=np.mean(losses)))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, mode=mode, seed=seed, step=steps), out / f"{mode}_seed{seed}_last.pt")
    model.load_state_dict(best_state)
    table, _ = evaluate(model, data, "selection")
    table.to_csv(out / f"{mode}_seed{seed}_selection.csv", index=False)
    torch.save(dict(state=best_state, mode=mode, seed=seed, best_step=best_step, steps=steps,
                    selection_ba=best_key[0], seconds=seconds, parameters=sum(p.numel() for p in model.parameters()),
                    first_gradient_norm=first_grad, query_samples_exposed=len(exposed_queries),
                    support_samples_exposed=len(exposed_supports), tasks_exposed=len(exposed_tasks)), path)
    pd.DataFrame(curve).to_csv(out / f"{mode}_seed{seed}_curve.csv", index=False)
    print("TRAINED", port, mode, seed, "step", best_step, "seconds", round(seconds, 2), flush=True)


def test_integration(port):
    data = load_data(port)
    d = data["tensors"]
    info = data["tasks"]["fit"][0]
    info = {**info, "q": info["q"][:8]}
    torch.manual_seed(11)
    model = GeometryReader(mode="C").to(DEVICE)
    context = encode(model, d, info["s"] + info["q"])
    context.retain_grad()
    prob, _, _ = prediction(d, info, context)
    other = dict(d)
    other["labels"] = d["labels"].clone()
    other["labels"][info["q"]] = (other["labels"][info["q"]] + 1) % 8
    changed, _, _ = prediction(other, info, context)
    torch.testing.assert_close(changed, prob, atol=0, rtol=0)
    loss_fn(prob, d["labels"][info["q"]]).backward()
    assert context.grad[:32].norm() > 0 and context.grad[32:].norm() > 0
    result = dict(query_label_mutation_invariance=True,
                  support_context_gradient=float(context.grad[:32].norm()), query_context_gradient=float(context.grad[32:].norm()),
                  finite_gradient=bool(torch.isfinite(context.grad).all()))
    save_json(HERE / "INTEGRATION_CHECK.json", result)
    print(result, flush=True)


def target(port):
    slug = port.replace(" ", "_")
    out = HERE / "evaluation" / slug
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(port, target=True)
    for mode, seed in [("body", -1)] + [(m, s) for m in "ABC" for s in range(3)]:
        file = out / f"{mode}_seed{seed}_predictions.csv"
        if file.exists():
            continue
        model = None
        if mode != "body":
            saved = torch.load(HERE / "trained" / slug / f"{mode}_seed{seed}.pt", weights_only=False)
            model = GeometryReader(mode=mode).to(DEVICE)
            model.load_state_dict(saved["state"])
            model.eval()
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics, predictions = evaluate(model, data, "target", detailed=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        for frame in (metrics, predictions):
            frame["arm"], frame["seed"] = mode, seed
        metrics.to_csv(out / f"{mode}_seed{seed}_metrics.csv", index=False)
        predictions.to_csv(file, index=False)
        save_json(out / f"{mode}_seed{seed}_runtime.json", dict(seconds=seconds, tasks=len(metrics),
                  samples=len(data["meta"]), cached_spatial_extraction_excluded=True, includes_diagnostics=True))
        print("TARGET", port, mode, seed, "seconds", round(seconds, 2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["smoke", "train", "target", "check"])
    parser.add_argument("--port")
    parser.add_argument("--mode", nargs="+", default=list("ABC"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    torch.set_num_threads(2)
    ports = [p["pool"] for p in source_pools() if not args.port or p["pool"].replace(" ", "_") == args.port]
    if args.phase == "target":
        missing = [str(HERE / "trained" / p.replace(" ", "_") / f"{m}_seed{s}.pt")
                   for p in [x["pool"] for x in source_pools()] for m in "ABC" for s in range(3)
                   if not (HERE / "trained" / p.replace(" ", "_") / f"{m}_seed{s}.pt").exists()]
        assert not missing, f"Complete source freeze before target: {missing[:3]}"
        save_json(HERE / "SOURCE_FROZEN.json", dict(ports=[x["pool"] for x in source_pools()], modes=list("ABC"), seeds=[0, 1, 2], steps=200))
    for port in ports:
        if args.phase == "check":
            test_integration(port)
        elif args.phase == "target":
            target(port)
        else:
            for seed in ([0] if args.phase == "smoke" else args.seeds):
                for mode in args.mode:
                    train(port, mode, seed, 20 if args.phase == "smoke" else 200, args.phase == "smoke")


if __name__ == "__main__":
    main()
