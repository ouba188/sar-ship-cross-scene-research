"""Replay the original source seed-0 training and recover verified step-600 tails."""

import copy
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from run_body_context import FIELDS, Gate, balanced_loss, evaluate, packed_task, predict

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "body_context"
OUT = HERE / "body_context_expansion" / "recovered_tails"
POOLS = ("Jebel_Ali", "Ningbo-Zhoushan", "Qingdao", "Shanghai", "Singapore")
ATOL = 1e-10


def compare(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def load_source(folder):
    prepared = json.loads((folder / "prepared.json").read_text(encoding="utf-8"))
    with np.load(folder / "readout.npz") as cache:
        arrays = {key: cache[key] for key in (*FIELDS, "weights")}
    meta = pd.read_csv(folder / "metadata.csv", dtype={"mmsi": str})
    np.testing.assert_array_equal(meta.position.to_numpy(), np.arange(len(meta)))
    saved_episodes = json.loads((folder / "episodes.json").read_text(encoding="utf-8"))
    episodes = {name: saved_episodes[name] for name in ("fit", "selection")}
    split = pd.read_csv(folder / "source_split.csv")
    coverage = {}
    for name, tasks in episodes.items():
        assert len(tasks) == prepared[f"{name}_tasks"]
        split_name = "module_fit" if name == "fit" else name
        allowed = set(split.loc[split.split.eq(split_name), "position"])
        used = {row for task in tasks for row in task["s"] + task["q"]}
        assert used <= allowed, f"{folder.name}/{name}: non-source split rows"
        for task in tasks:
            assert not set(meta.iloc[task["q"]].mmsi) & set(meta.iloc[task["s"]].mmsi)
        frame = meta.iloc[sorted(used)]
        coverage[name] = dict(tasks=len(tasks), rows=len(frame), ships=int(frame.mmsi.nunique()),
                              ports=int(frame.port.nunique()),
                              acquisitions=int(frame.acquisition.nunique()),
                              class_counts={str(k): int(v) for k, v in
                                            frame.class_id.value_counts().sort_index().items()})
    packed = {name: [packed_task(arrays, task) for task in tasks]
              for name, tasks in episodes.items()}
    return arrays, meta, episodes, packed, dict(
        prepared_protocol=prepared["protocol"], prepared_source_counts=prepared["source_counts"],
        actual_task_coverage=coverage)


def recover(folder, arrays, meta, episodes, packed, mode):
    reference = torch.load(folder / f"{mode}_seed0.pt", weights_only=False, map_location="cpu")
    original_curve = pd.read_csv(folder / f"{mode}_seed0_curve.csv",
                                 float_precision="round_trip").set_index("step")
    assert reference["mode"] == mode and reference["seed"] == 0
    torch.manual_seed(0)
    gate = Gate(mode=mode).double()
    optimizer = torch.optim.Adam(gate.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(20261015 + 0)
    best, best_step, best_state = -np.inf, 0, copy.deepcopy(gate.state_dict())
    checkpoints, losses = [], []
    started = time.perf_counter()
    with torch.no_grad():
        initial_fit_loss = np.mean([float(balanced_loss(predict(t, gate(t["ship"], t["ring"])),
            t["labels"][t["q"]])) for t in packed["fit"]])
    for step in range(601):
        if step in (0, 200, 400, 600):
            table, _ = evaluate(arrays, episodes["selection"], packed["selection"],
                                meta, mode, gate, 0)
            result = table.loc[table.origin.eq("cross")].ba.mean()
            same = table.loc[table.origin.eq("same")].ba.mean()
            actual = dict(selection_cross_ba=float(result), selection_same_ba=float(same),
                          selection_macro_nll=float(
                              table.loc[table.origin.eq("cross")].macro_nll.mean()))
            expected = {key: float(original_curve.loc[step, key]) for key in actual}
            errors = {key: compare(value, expected[key]) for key, value in actual.items()}
            checkpoints.append(dict(step=step, actual=actual, expected=expected, abs_error=errors))
            if result > best + 1e-12:
                best, best_step, best_state = result, step, copy.deepcopy(gate.state_dict())
            print("checkpoint", folder.name, mode, "seed0", step, "BA", result,
                  "max_error", max(errors.values()), flush=True)
        if step == 600:
            break
        sampled = rng.choice(len(packed["fit"]), 4, replace=len(packed["fit"]) < 4)
        batch_losses = []
        for index in sampled:
            task = packed["fit"][index]
            local = gate(task["ship"], task["ring"])
            batch_losses.append(balanced_loss(predict(task, local), task["labels"][task["q"]]))
        loss = torch.stack(batch_losses).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            actual_loss = float(loss.detach())
            expected_loss = float(original_curve.loc[step + 1, "train_loss"])
            losses.append(dict(step=step + 1, actual=actual_loss, expected=expected_loss,
                               abs_error=compare(actual_loss, expected_loss)))
            print("train", folder.name, mode, step + 1, "loss", actual_loss, flush=True)

    assert best_step == reference["best_step"], (folder.name, mode, best_step, reference["best_step"])
    selected_ba_error = compare(best, reference["selection_ba"])
    assert best_state.keys() == reference["state"].keys()
    state_errors = {key: compare(value.numpy(), reference["state"][key].numpy())
                    for key, value in best_state.items()}
    last_state = copy.deepcopy(gate.state_dict())
    with torch.no_grad():
        last_fit_loss = np.mean([float(balanced_loss(predict(t, gate(t["ship"], t["ring"])),
            t["labels"][t["q"]])) for t in packed["fit"]])
        # Match the original unlabeled context diagnostics over the cached rows.
        all_ring = torch.as_tensor(arrays["ring"], dtype=torch.float64)
        logits = gate.c[None, :].expand(len(all_ring), -1)
        if mode == "conditional":
            logits = all_ring @ gate.W.T + gate.c
        adjustment = .25 * torch.tanh(logits)
        diagnostics = dict(initial_fit_nll=float(initial_fit_loss), last_fit_nll=float(last_fit_loss),
                           adjustment_abs_mean=float(adjustment.abs().mean()),
                           adjustment_abs_max=float(adjustment.abs().max()),
                           adjustment_across_sample_std=float(adjustment.std(dim=0).mean()),
                           parameter_norm=float(torch.cat([p.flatten() for p in gate.parameters()]).norm()))
    output = OUT / folder.name / f"{mode}_seed0_last.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state=last_state, mode=mode, seed=0, step=600, diagnostics=diagnostics)
    torch.save(payload, output)
    saved = torch.load(output, weights_only=False, map_location="cpu")
    assert saved.keys() == payload.keys()
    assert all(saved[key] == payload[key] for key in ("mode", "seed", "step", "diagnostics"))
    assert saved["state"].keys() == last_state.keys()
    for key, value in last_state.items():
        torch.testing.assert_close(saved["state"][key], value, atol=0, rtol=0)
        assert value.dtype == torch.float64 and torch.isfinite(value).all()
    result = dict(pool=folder.name, mode=mode, seed=0, step=600, output=str(output),
                  checkpoints=checkpoints, logged_train_losses=losses,
                  selected=dict(step=int(best_step), expected_step=int(reference["best_step"]),
                                ba=float(best), expected_ba=float(reference["selection_ba"]),
                                ba_abs_error=selected_ba_error,
                                state_abs_errors=state_errors),
                  diagnostics=diagnostics, serialization_exact=True, passed=True,
                  seconds=time.perf_counter() - started)
    print("recovered", folder.name, mode, "selected_step", best_step,
          "seconds", round(result["seconds"], 1), flush=True)
    return result


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    protected = [SOURCE / pool / f"{mode}_seed{seed}_last.pt"
                 for pool in POOLS for mode in ("static", "conditional") for seed in (1, 2)]
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in protected}
    report = dict(protocol="original_source_seed0_replay_no_target_evaluation",
                  configuration=dict(steps=600, batch_tasks=4, torch_seed=0,
                                     task_rng_seed=20261015, optimizer="Adam", lr=.001,
                                     weight_decay=.0001, dtype="torch.float64", threads=2,
                                     checkpoint_steps=[0, 200, 400, 600],
                                     selection="source cross BA; strict improvement > 1e-12",
                                     atol=ATOL, rtol=0),
                  environment=dict(python=sys.executable, torch=str(torch.__version__),
                                   numpy=np.__version__, pandas=pd.__version__),
                  source_pools={}, runs=[], complete=False)
    with threadpool_limits(limits=2):
        for pool in POOLS:
            folder = SOURCE / pool
            arrays, meta, episodes, packed, coverage = load_source(folder)
            report["source_pools"][pool] = coverage
            print("source", pool, "fit", len(packed["fit"]),
                  "selection", len(packed["selection"]), flush=True)
            for mode in ("static", "conditional"):
                report["runs"].append(recover(folder, arrays, meta, episodes, packed, mode))
    assert all((path.stat().st_size, path.stat().st_mtime_ns) == values
               for path, values in before.items()), "Existing seed1/2 tails changed during replay"
    report["verification"] = dict(
        runs=len(report["runs"]), checkpoints=sum(len(r["checkpoints"]) for r in report["runs"]),
        max_checkpoint_abs_error=max(error for r in report["runs"] for c in r["checkpoints"]
                                     for error in c["abs_error"].values()),
        max_train_loss_abs_error=max(v["abs_error"] for r in report["runs"]
                                    for v in r["logged_train_losses"]),
        max_selected_state_abs_error=max(error for r in report["runs"]
                                        for error in r["selected"]["state_abs_errors"].values()),
        selected_steps_match=True, saved_states_exact=True,
        existing_seed12_tails_unchanged=len(before), target_evaluations=0)
    report["complete"] = True
    report["seconds"] = time.perf_counter() - started
    report_path = OUT / "replay_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps(report["verification"], sort_keys=True), flush=True)
    print("report", report_path, "seconds", round(report["seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
