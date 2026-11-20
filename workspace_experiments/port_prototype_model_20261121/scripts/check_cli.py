"""Two optimizer steps on synthetic chips verify the complete CLI contract.

Artifacts stay under local_validation. This is an integration check, never a
scientific accuracy result or a SAR experiment. No external dataset is read.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def synthetic_records(directory):
    rng = np.random.default_rng(20261121)
    rows = []
    ports = ("S0", "S1", "S0", "S1", "V", "V", "T", "T")
    labels = (0, 0, 1, 1, 0, 1, 0, 1)
    yy, xx = np.mgrid[:72, :80]
    for i, (port, label) in enumerate(zip(ports, labels)):
        theta = .12 + i * .07
        center = np.asarray([40., 36.])
        rotation = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        corners = np.asarray([[-18, -5], [18, -5], [18, 5], [-18, 5]]) @ rotation.T + center
        row = dict(id=f"synthetic_{i}", label=label, vessel_id=f"vessel_{i}",
                   group_id=f"group_{i}", port=port)
        u = (xx - center[0]) * np.cos(theta) + (yy - center[1]) * np.sin(theta)
        v = -(xx - center[0]) * np.sin(theta) + (yy - center[1]) * np.cos(theta)
        for p, polarization in enumerate(("vv", "vh")):
            image = rng.normal(35 + 5 * p, 6, (72, 80))
            body = (np.abs(u) < 18) & (np.abs(v) < 5)
            image += body * (110 + 30 * np.cos(u * (.4 + .1 * label)) ** 2)
            image += 18 * np.exp(-v ** 2 / 8) * np.exp(-u ** 2 / 900)
            name = f"synthetic_{i}_{polarization}.png"
            Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(directory / name)
            row[polarization] = dict(image=name, target_index=0,
                                     detections=[dict(corners=corners.tolist(), confidence=.99)])
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    dest = args.output or ROOT / "local_validation" / ("cli_flow_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rows = synthetic_records(dest)
    manifest, split = dest / "manifest.json", dest / "split.json"
    random_bank, coverage_bank = dest / "random_bank.json", dest / "coverage_bank.json"
    write(manifest, rows)
    base = dict(classes=2, third_layer=True, gate=True, layer_matching="cross",
                port_aggregation="hierarchical", prior_weight=0., effect_weight=0.,
                consistency_weight=0., budget_weight=0., sinkhorn_steps=20)
    full = dict(base, prior_weight=.1, effect_weight=.1, consistency_weight=.1, budget_weight=.01)
    write(dest / "bootstrap_config.json", base)
    write(dest / "full_config.json", full)
    stages = []
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def run(name, arguments):
        tick = time.perf_counter()
        print("Running", name, flush=True)
        process = subprocess.run([sys.executable, "-m", "sar_port.cli", *map(str, arguments)],
                                 cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (dest / f"{name}.log").write_text(process.stdout + process.stderr, encoding="utf-8")
        stages.append(dict(stage=name, seconds=time.perf_counter() - tick, returncode=process.returncode))
        if process.returncode:
            print(process.stdout[-2500:] + process.stderr[-6000:])
            write(dest / "summary.json", dict(status="failed", stages=stages, synthetic_only=True))
            raise RuntimeError(f"{name} failed; see its log in {dest}")

    run("split", ["split", "--manifest", manifest, "--target-ports", "T", "--validation-ports", "V", "--output", split])
    run("random_bank", ["select-bank", "--manifest", manifest, "--split", split, "--per-class", 1,
                        "--method", "random", "--output", random_bank])
    common = ["--manifest", manifest, "--split", split, "--steps", 1, "--batch", 2,
              "--device", args.device, "--eval-every", 1]
    run("bootstrap", ["train", *common, "--bank", random_bank, "--config", dest / "bootstrap_config.json",
                      "--output", dest / "bootstrap"])
    teacher = dest / "bootstrap" / "best.pt"
    run("coverage_bank", ["select-bank", "--manifest", manifest, "--split", split, "--per-class", 1,
                          "--method", "coverage", "--checkpoint", teacher, "--device", args.device,
                          "--batch", 2, "--output", coverage_bank])
    run("full_train", ["train", *common, "--bank", coverage_bank, "--config", dest / "full_config.json",
                       "--initialize", teacher, "--teacher", teacher, "--output", dest / "full"])
    checkpoint = dest / "full" / "best.pt"
    run("evaluate", ["evaluate", "--checkpoint", checkpoint, "--manifest", manifest, "--split", split,
                     "--partition", "test", "--batch", 2, "--device", args.device, "--output", dest / "evaluate.json"])
    exported = dest / "deployment.pt"
    run("export", ["export", "--checkpoint", checkpoint, "--device", args.device, "--output", exported])
    unlabelled = [{key: row[key] for key in ("id", "vv", "vh")} for row in rows[-2:]]
    write(dest / "unlabelled.json", unlabelled)
    run("predict", ["predict", "--checkpoint", exported, "--manifest", dest / "unlabelled.json",
                    "--batch", 2, "--device", args.device, "--output", dest / "predict.json"])

    history = json.loads((dest / "full" / "history.json").read_text())
    for key in ("classification", "prior_kl", "effect_huber", "consistency", "budget", "loss"):
        if key not in history[0] or not np.isfinite(history[0][key]):
            raise AssertionError(f"Missing/nonfinite training term {key}")
    evaluated = json.loads((dest / "evaluate.json").read_text())["predictions"]
    predicted = json.loads((dest / "predict.json").read_text())
    expected = torch.tensor([item["logits"] for item in evaluated]).softmax(-1).numpy()
    actual = np.asarray([item["probabilities"] for item in predicted])
    difference = float(np.max(np.abs(expected - actual)))
    if difference > 1e-5:
        raise AssertionError(f"Exported prediction differs from live bank: {difference}")
    for a, b in zip(evaluated, predicted):
        if a["id"] != b["id"] or a["prediction"] != b["prediction"]:
            raise AssertionError("Prediction ordering/labels differ after export")
    artifact = torch.load(exported, map_location="cpu", weights_only=True)
    if any("vv" in row or "vh" in row for row in artifact["bank_records"]):
        raise AssertionError("Export unexpectedly retains reference image specifications")
    summary = dict(status="passed", synthetic_only=True, scientific_accuracy_claim=False,
                   source_rows=4, validation_rows=2, test_rows=2, classes=2, bank_size=2,
                   optimizer_steps=2, teacher_forwards_per_student_batch=25,
                   third_layer=True, gate=True, layer_matching="cross", port_aggregation="hierarchical",
                   training_terms=history[0], export_probability_max_error=difference, stages=stages)
    write(dest / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print("Local artifacts:", dest)


if __name__ == "__main__":
    main()
