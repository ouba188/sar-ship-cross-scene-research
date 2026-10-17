"""Train fixed-step source-only slot readers through the frozen full classifier."""

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

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "ship_radar_geometry_design_20261016"
sys.path.insert(0, str(PREV))
import run_geometry as old
from geometry_reader import GeometryReader
from body_context_gate import _coefficients
from slot_reader import SlotReader, centered_features, predict_increment

MODES = ("spatial", "conditional", "geometry")
PORTS = [p["pool"] for p in old.source_pools()]
DEVICE = "cuda"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load(port, target=False):
    data = old.load_data(port, target)
    data["tasks"] = {k: [t for t in ts if t["budget"] == 32] for k, ts in data["tasks"].items()}
    d = data["tensors"]
    with np.load(HERE / "slots_cache" / (port.replace(" ", "_") + ".npz")) as z:
        rows = data["meta"].global_row.to_numpy()
        idx = np.searchsorted(z["rows"], rows)
        np.testing.assert_array_equal(z["rows"][idx], rows)
        for key in ("img_tokens", "ship_tokens", "img_areas", "ship_areas"):
            d[key] = torch.as_tensor(z[key][idx], dtype=torch.float32, device=DEVICE)
    d["valid"] = (d["geometry"][:, 3:6] > 0).all(1)
    data["port"] = port
    return data


@torch.no_grad()
def prepare_baseline(data, seed):
    d = data["tensors"]
    ck = torch.load(PREV / "trained" / data["port"].replace(" ", "_") / f"B_seed{seed}.pt", weights_only=False, map_location=DEVICE)
    model = GeometryReader("B").to(DEVICE)
    model.load_state_dict(ck["state"])
    model.eval()
    c = old.contexts(model, data).double()
    local = torch.cat((np.sqrt(.5)*d["ship"], np.sqrt(.5)*c), -1)
    data["cache"] = {}
    for split, tasks in data["tasks"].items():
        items = []
        for t in tasks:
            s = torch.as_tensor(t["s"], device=DEVICE)
            q = torch.as_tensor(t["q"], device=DEVICE)
            e, b, r, w = (d[k] for k in ("expert", "base", "relation", "weights"))
            ss = e[s] @ e[s].T + .25*(16*(local[s] @ local[s].T) - e[s,128:256] @ e[s,128:256].T)
            qs = e[q] @ e[s].T + .25*(16*(local[q] @ local[s].T) - e[q,128:256] @ e[s,128:256].T)
            ss = .75*ss + .25*((r[s]*w) @ r[s].T)
            qs = .75*qs + .25*((r[q]*w) @ r[s].T)
            sy = d["labels"][s]
            bp = (d["prior_b"][q] + (b[q] @ b[s].T) @ _coefficients(b[s] @ b[s].T, d["prior_b"][s], sy)).softmax(-1)
            prob = .5*(bp + (d["prior_e"][q] + qs @ _coefficients(ss, d["prior_e"][s], sy)).softmax(-1))
            items.append(dict(t=t, s=s, q=q, ss=ss, qs=qs, sy=sy, prior_s=d["prior_e"][s],
                              prior_q=d["prior_e"][q], base_prob=bp, old_prob=prob))
        data["cache"][split] = items
    # One independent old readout replay catches replacement/relation scaling errors.
    first = next(iter(data["cache"].values()))[0]
    rows = torch.cat((first["s"], first["q"]))
    actual, _, _ = old.prediction(d, first["t"], c[rows])
    torch.testing.assert_close(actual, first["old_prob"], atol=1e-11, rtol=1e-11)
    data["old_best_step"] = int(ck["best_step"])


def descriptor(model, d, rows, perturb=None):
    prefix = "ship" if model.mode == "geometry" else "img"
    tokens, areas = d[prefix+"_tokens"][rows], d[prefix+"_areas"][rows]
    geo = d["geometry"][rows]
    if perturb:
        geo = geo.clone()
        if perturb == "ship45":
            geo[:, 0] += np.pi/4
            if model.mode == "geometry":
                tokens, areas = tokens.roll(-1, 2), areas.roll(-1, 2)
        elif perturb == "radar45":
            geo[:, 1] += np.pi/4
    return model(tokens, areas, geo)


def prediction(model, data, item, query_positions=None, raw_all=None, perturb=None):
    d = data["tensors"]
    positions = torch.arange(len(item["q"]), device=DEVICE) if query_positions is None else query_positions
    rows = torch.cat((item["s"], item["q"][positions]))
    raw = descriptor(model, d, rows, perturb) if raw_all is None else raw_all[rows]
    prob, kernel, psi = predict_increment(item["ss"], item["qs"][positions], item["prior_s"],
        item["prior_q"][positions], item["sy"], item["base_prob"][positions], item["old_prob"][positions],
        raw, d["valid"][rows], _coefficients)
    return prob, kernel, psi


def fit_inventory(data):
    d = data["tensors"]
    eligible, groups, queries, supports = [], {}, set(), set()
    for i, item in enumerate(data["cache"]["fit"]):
        qvalid = torch.where(d["valid"][item["q"]])[0]
        if int(d["valid"][item["s"]].sum()) < 2 or not len(qvalid):
            continue
        eligible.append(i)
        t = item["t"]
        groups.setdefault(t["port"], {}).setdefault(t["acquisition"], []).append(i)
        queries.update(item["q"][qvalid].cpu().tolist())
        supports.update(item["s"][d["valid"][item["s"]]].cpu().tolist())
    result = dict(tasks=len(eligible), query_samples=len(queries), support_samples=len(supports))
    for name, inds in (("query", queries), ("support", supports)):
        frame = data["meta"].iloc[sorted(inds)]
        result.update({name+"_ships": int(frame.mmsi.nunique()), name+"_acquisitions": int(frame.acquisition.nunique()),
                       name+"_classes": sorted(frame.class_id.unique().tolist())})
    return eligible, groups, sorted(queries | supports), result


@torch.no_grad()
def raw_descriptors(model, data, perturb=None):
    d = data["tensors"]
    return torch.cat([descriptor(model, d, slice(i, i+256), perturb) for i in range(0, len(data["meta"]), 256)])


@torch.no_grad()
def evaluate(model, data, split, details=False, perturb=None):
    d = data["tensors"]
    raw = raw_descriptors(model, data, perturb) if model is not None else None
    records, frames = [], []
    for ti, item in enumerate(data["cache"][split]):
        if model is None:
            prob, kernel = item["old_prob"], item["qs"]
        else:
            prob, kernel, _ = prediction(model, data, item, raw_all=raw)
        p = prob.cpu().numpy()
        y = d["labels"][item["q"]].cpu().numpy()
        sy = item["sy"].cpu().numpy()
        enabled = (d["valid"][item["q"]] & (d["valid"][item["s"]].sum() >= 2)).cpu().numpy()
        t = item["t"]
        common = {k: t[k] for k in ("port", "acquisition", "origin", "budget", "draw", "fold")}
        common["task"] = ti
        for scope, keep in (("full", np.ones(len(y), bool)), ("geometry", enabled)):
            if keep.any():
                records.append(dict(**common, scope=scope, **old.task_metric(p[keep], y[keep]), n=int(keep.sum())))
        if not details:
            continue
        frame = data["meta"].iloc[item["q"].cpu().numpy()][["sample_id", "global_row", "mmsi", "class_id"]].reset_index(drop=True)
        for key, value in common.items():
            frame[key] = value
        frame["truth"], frame["pred"], frame["enabled"] = y, p.argmax(1), enabled
        frame["correct"] = frame.truth.eq(frame.pred)
        frame["valid_supports"] = int(d["valid"][item["s"]].sum())
        frame["nll"] = -np.log(p[np.arange(len(y)), y].clip(1e-12))
        frame["brier"] = ((p - np.eye(8)[y])**2).sum(1)
        ap, margin = [], []
        k = kernel.cpu().numpy()
        # Disabled queries use B's actual kernel as well as B's prediction.
        k[~enabled] = item["qs"].cpu().numpy()[~enabled]
        for i, cls in enumerate(y):
            hit = sy == cls
            count = int(hit.sum())
            if count:
                relevant = hit[np.argsort(-k[i], kind="stable")[:count]]
                ap.append(float(np.sum(np.cumsum(relevant) / np.arange(1,count+1) * relevant) / count))
                margin.append(float(k[i,hit].mean() - k[i,~hit].mean()) if (~hit).any() else np.nan)
            else:
                ap.append(np.nan)
                margin.append(np.nan)
        frame["ap"], frame["kernel_margin"] = ap, margin
        for j in range(8):
            frame[f"p{j}"] = p[:,j]
        frames.append(frame)
    return pd.DataFrame(records), pd.concat(frames, ignore_index=True) if frames else None


def summary(table):
    result = {}
    for scope, group in table.groupby("scope"):
        for origin, g in group.groupby("origin"):
            v = old.hierarchy(g)
            result.update({scope+"_"+origin+"_"+k: float(x) for k,x in v.items()})
    return result


def train(data, mode, seed, steps=200, smoke=False):
    folder = HERE / ("smoke" if smoke else "trained") / data["port"].replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{mode}_seed{seed}.pt"
    if path.exists():
        return
    d = data["tensors"]
    eligible, groups, fit_rows, inventory = fit_inventory(data)
    dump(folder / "fit_inventory.json", inventory)
    if not eligible:
        torch.save(dict(unavailable=True, inventory=inventory), path)
        return
    torch.manual_seed(20261016+seed)
    model = SlotReader(mode).to(DEVICE)
    with torch.no_grad():
        raw = descriptor(model, d, fit_rows)
        energy = (raw-raw.mean(0)).square().sum(1).mean()
        assert energy > 1e-10, "No initial instance variation to scale"
        model.scale.copy_(energy.sqrt())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(20261016+seed)
    curve, trace, first_gradient = [], [], None
    started = time.perf_counter()
    for step in range(steps+1):
        if step in {0, 50, 100, steps}:
            model.eval()
            record = dict(step=step)
            for split in ("fit", "selection"):
                table, _ = evaluate(model, data, split)
                record.update({split+"_"+k:v for k,v in summary(table).items()})
            with torch.no_grad():
                values = descriptor(model, d, fit_rows)
                record["source_centered_energy"] = float((values-values.mean(0)).square().sum(1).mean())
            curve.append(record)
            print("CURVE", data["port"], mode, seed, step, "energy", round(record["source_centered_energy"],4), flush=True)
        if step == steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(2):
            ports = list(groups)
            port = ports[int(rng.integers(len(ports)))]
            scenes = list(groups[port])
            scene = scenes[int(rng.integers(len(scenes)))]
            ti = int(rng.choice(groups[port][scene]))
            item = data["cache"]["fit"][ti]
            available = torch.where(d["valid"][item["q"]])[0].cpu().numpy()
            pos = available if len(available)<=32 else rng.choice(available,32,replace=False)
            positions = torch.as_tensor(pos, device=DEVICE)
            probability, _, _ = prediction(model, data, item, positions)
            loss = old.loss_fn(probability, d["labels"][item["q"][positions]])
            assert torch.isfinite(loss)
            (loss/2).backward()
            losses.append(float(loss.detach()))
            trace.append(dict(step=step+1, task=ti, query_positions=pos.tolist()))
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        if first_gradient is None:
            first_gradient = float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
            assert first_gradient > 0
        optimizer.step()
        if step%25==0:
            curve.append(dict(step=step+1, train_nll=float(np.mean(losses))))
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()}, mode=mode, seed=seed,
                    steps=steps, seconds=seconds, inventory=inventory, first_gradient=first_gradient,
                    parameters=sum(p.numel() for p in model.parameters()), baseline_step=data["old_best_step"]), path)
    pd.DataFrame(curve).to_csv(folder / f"{mode}_seed{seed}_curve.csv", index=False)
    dump(folder / f"{mode}_seed{seed}_sampling.json", trace)
    print("TRAINED", data["port"], mode, seed, "seconds", round(seconds,2), flush=True)


def check(data):
    eligible, _, fit_rows, _ = fit_inventory(data)
    if not eligible:
        return dict(unavailable=True, reason="No fit task has both >=2 valid supports and a valid query", old_solver_replay=True)
    item = data["cache"]["fit"][eligible[0]]
    model = SlotReader("geometry").to(DEVICE)
    prob, _, psi = prediction(model, data, item)
    psi.retain_grad()
    old.loss_fn(prob, data["tensors"]["labels"][item["q"]]).backward()
    assert any(p.grad is not None and float(p.grad.norm()) > 0 for p in model.parameters())
    other = copy.copy(data)
    other["tensors"] = dict(data["tensors"])
    other["tensors"]["labels"] = data["tensors"]["labels"].clone()
    other["tensors"]["labels"][item["q"]] = (other["tensors"]["labels"][item["q"]]+1)%8
    changed, _, _ = prediction(model, other, item)
    torch.testing.assert_close(changed, prob, atol=0, rtol=0)
    rows = torch.cat((item["s"],item["q"]))
    valid = data["tensors"]["valid"][rows]
    constant = torch.ones((len(rows),192), device=DEVICE)
    p0, _, _ = predict_increment(item["ss"],item["qs"],item["prior_s"],item["prior_q"],item["sy"],
        item["base_prob"],item["old_prob"],constant,valid,_coefficients)
    torch.testing.assert_close(p0, item["old_prob"], atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(prob[~valid[len(item['s']):]], item["old_prob"][~valid[len(item['s']):]], atol=0,rtol=0)
    return dict(old_solver_replay=True, query_labels_unused=True, constant_increment_zero=True,
                missing_query_exact_fallback=True, gradient_nonzero=True)


def target(data, seed):
    folder = HERE / "evaluation" / data["port"].replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    for mode in ("B", *MODES):
        model = None
        if mode != "B":
            saved = torch.load(HERE / "trained" / data["port"].replace(" ","_") / f"{mode}_seed{seed}.pt", weights_only=False)
            if not saved.get("unavailable"):
                model = SlotReader(mode).to(DEVICE)
                model.load_state_dict(saved["state"])
                model.eval()
        for perturb in ([None,"ship45","radar45"] if mode=="geometry" else [None]):
            arm = mode if perturb is None else mode+"_"+perturb
            file = folder / f"{arm}_seed{seed}_predictions.csv"
            if file.exists():
                continue
            start=time.perf_counter()
            metrics, predictions = evaluate(model, data, "target", True, perturb)
            for frame in (metrics,predictions):
                frame["arm"],frame["seed"]=arm,seed
            metrics.to_csv(folder/f"{arm}_seed{seed}_metrics.csv",index=False)
            predictions.to_csv(file,index=False)
            dump(folder/f"{arm}_seed{seed}_runtime.json",dict(seconds=time.perf_counter()-start,tasks=len(data["tasks"]["target"]),includes_diagnostics=True,excludes_frozen_encoder=True))
            print("TARGET",data["port"],arm,seed,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["check","smoke","train","target"])
    parser.add_argument("--port")
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.phase=="target":
        expected=[HERE/"trained"/p.replace(" ","_")/f"{m}_seed{s}.pt" for p in PORTS for m in MODES for s in range(3)]
        assert all(p.exists() for p in expected), "Freeze all source fits before target"
        dump(HERE/"SOURCE_FROZEN.json",dict(models=len(expected),steps=200,seeds=[0,1,2],modes=MODES))
    checks={}
    for port in PORTS:
        if args.port and port.replace(" ","_")!=args.port:
            continue
        data=load(port,args.phase=="target")
        for seed in ([0] if args.phase in ("check","smoke") else range(3)):
            prepare_baseline(data,seed)
            if args.phase=="check":
                checks[port]=check(data)
            elif args.phase=="target":
                target(data,seed)
            else:
                for mode in MODES:
                    train(data,mode,seed,20 if args.phase=="smoke" else 200,args.phase=="smoke")
        del data
        torch.cuda.empty_cache()
    if checks:
        dump(HERE/"INTEGRATION_CHECK.json",checks)


if __name__=="__main__":
    main()
