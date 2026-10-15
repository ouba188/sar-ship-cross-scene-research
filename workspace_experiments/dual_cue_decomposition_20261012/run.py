"""Fixed-prior, source-mapper decomposition of the deployed R18 support kernel."""

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

HELPERS = Path("E:/Hermes/paper_tracking/sensitivity/oof_reliability_check")
sys.path.insert(0, str(HELPERS))
from dual_reconstruction_core import Inputs, reconstruct, folders, info_for, KINDS
from dual_evidence_core import EvidenceContext, load_task, base_cross, ROOT
from dual_reconstruction_predict import view
from projection_residual_expert import discarded_coordinates
from unified_head_core import coefficients
from backbone_ablation_report import ba
from compact_fusion_source import SOURCE
from prepare_learned_frequency import DATASETS

OUT = Path(__file__).resolve().parent
ARMS = ("R", "P", "E", "PE", "T")
READOUTS = ("expert", "late")
BATCH = 512


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def ssl_map(extra, model, stats, port=False):
    projected = model["pca_ssl"].transform(extra).astype(np.float32)
    # Port drift subtraction promotes the concatenated retained view to float64.
    if port:
        projected = projected.astype(float)
    retained = (projected - model["expert_center"][128:192]) / model["expert_scale"][128:192]
    complement = (discarded_coordinates(extra, model["pca_ssl"])
                  - stats["discarded_center"][2048:]) / stats["discarded_scale"][2048:]
    return np.c_[retained / np.sqrt(192), complement / np.sqrt(2560)] * np.sqrt(.5)


def teacher_map(features):
    return np.c_[features["dr"][:, 128:], features["dd"][:, 2048:]] * np.sqrt(.5)


def decomposed_kernels(kr, p, e, pref, eref):
    pp, ee = p @ pref.T, e @ eref.T
    return {"R": kr, "P": kr + pp, "E": kr + ee,
            "PE": kr + pp + ee, "T": kr + (p + e) @ (pref + eref).T}


def freeze_protocol():
    protocol = dict(
        date="2026-10-12", tasks=[info_for(f)[1] for f in folders()],
        estimand="Task-useful R18 kernel content not reproduced by two fixed source-only R50 mappers",
        mapper="Reuse BOTH linear and MLP from dual_reconstruction_20261010; no refit or target selection",
        decomposition="P=source-scaled deployed SSL map of reconstructed R18; E=actual deployed SSL map minus P",
        arms={"R": "KR", "P": "KR+PP'", "E": "KR+EE'",
              "PE": "KR+PP'+EE'", "T": "KR+(P+E)(P+E)'"},
        fixed="R50-only source prior for EVERY arm; original geometry/scales, base, five shared64 supports, balanced ridge8, equal probability fusion",
        interaction="T minus PE tests the cross terms PE'+EP'; cross terms alone need not be PSD and are not fitted",
        metrics="BA/OA, class-macro and micro Brier, NLL, paired rescue/harm; softmax temperature1, same as deployed fusion",
        aggregation="Five draw means within task, then equal-task means within dataset; class macro within task",
        support_budget=64, neural_training=False, new_source_fit=False,
        target_query_selection=False, independent_confirmation=False,
        boundaries=["Residual is mapper-relative, not information-theoretically absent from R50",
                    "P uses R50 original plus C4 pooling; E and T require R18 at inference",
                    "No separate recentering or renormalization; PE differs in kernel energy and geometry from T",
                    "Useful E may reflect input cues, learned encoding, extrapolation errors or their interaction",
                    "Prior-path advantages are held out of this diagnostic, not declared absent",
                    "Historical target SSL exposure remains; five draws are overlapping, not neural seeds",
                    "Existing context helpers read full source manifests; no new label fitting is performed"])
    path = OUT / "PROTOCOL.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Existing frozen protocol differs")
    else:
        write_json(path, protocol)


def predict(folder):
    info, suffix = info_for(folder)
    dst = OUT / "targets" / suffix
    if (dst / "COMPLETE.json").exists():
        print("REUSE", suffix, flush=True)
        return
    started = time.perf_counter()
    dst.mkdir(parents=True, exist_ok=True)
    data = load_task(folder)
    inputs = Inputs(folder)
    context = EvidenceContext(folder, data)
    with np.load(ROOT / "dual_reconstruction_20261010/targets" / suffix / "mapper.npz") as z:
        mapper = {k: z[k] for k in z.files}
    with np.load(ROOT / "dual_evidence_20261010/targets" / suffix / "predictions.npz") as z:
        for name in ("rows", "ids", "support", "eval_masks"):
            np.testing.assert_array_equal(z[name], data[name])
        old_modes = list(z["modes"])
        old_preds = z["predictions"]
    ref = np.unique(data["support"])
    locs = [np.searchsorted(ref, support) for support in data["support"]]
    fref = context.features(ref)
    kref = context.cross(fref, fref)
    tref = teacher_map(fref)
    kbref = base_cross(data, ref, ref, info["eta"])
    n, draws = len(data["rows"]), len(locs)
    probabilities = np.lib.format.open_memmap(dst / "probabilities.npy", mode="w+", dtype=np.float32,
        shape=(len(KINDS), draws, len(ARMS), len(READOUTS), n, 8))
    predictions = np.empty(probabilities.shape[:-1], np.int8)
    diagnostics, all_coefficients = [], {}
    max_kernel_error = 0.
    for ki, kind in enumerate(KINDS):
        reconstructed = reconstruct(inputs.target_x(ref), mapper, kind)
        pref = ssl_map(reconstructed, context.model, context.stats, inputs.dataset == "self24ports")
        eref = tref - pref
        # Validate the shortcut against the deployed full-view transform before scoring.
        bank = context.bank if inputs.dataset == "self24ports" else None
        angles = context.u if inputs.dataset == "self24ports" else None
        original_view = view(inputs, reconstructed, ref, context.fitted["dual"], bank, angles)
        expected = np.c_[original_view["ret"][:, 128:], original_view["comp"][:, 2048:]] * np.sqrt(.5)
        np.testing.assert_allclose(pref, expected, atol=1e-12, rtol=1e-12)
        kernels = decomposed_kernels(kref["R"], pref, eref, pref, eref)
        np.testing.assert_allclose(kernels["T"], kref["D"], atol=1e-10, rtol=1e-10)
        states = []
        for draw, loc in enumerate(locs):
            support, sy = data["support"][draw], data["support_labels"][draw]
            if len(np.unique(support)) != 64 or data["eval_masks"][draw, support].any():
                raise ValueError("Changed support budget or support/query overlap")
            ab = coefficients(kbref[np.ix_(loc, loc)], data["prior"][support], sy, 8.)
            alphas = {arm: coefficients(kernels[arm][np.ix_(loc, loc)], fref["priors"]["r50"][loc], sy, 8.)
                      for arm in ARMS}
            states.append((ab, alphas))
            for arm, alpha in alphas.items():
                all_coefficients[f"{kind}_{draw}_{arm}"] = alpha
        energy = np.zeros(4)
        for lo in range(0, n, BATCH):
            s = slice(lo, min(lo + BATCH, n))
            f = context.features(s)
            actual = context.cross(f, fref)
            t = teacher_map(f)
            p = ssl_map(reconstruct(inputs.target_x(s), mapper, kind), context.model, context.stats,
                        inputs.dataset == "self24ports")
            e = t - p
            kernels = decomposed_kernels(actual["R"], p, e, pref, eref)
            error = float(np.max(np.abs(kernels["T"] - actual["D"])))
            max_kernel_error = max(max_kernel_error, error)
            np.testing.assert_allclose(kernels["T"], actual["D"], atol=1e-9, rtol=1e-10)
            energy += [np.sum(t*t), np.sum(p*p), np.sum(e*e), 2*np.sum(p*e)]
            kb = base_cross(data, s, ref, info["eta"])
            for draw, loc in enumerate(locs):
                ab, alphas = states[draw]
                pb = softmax(data["prior"][s] + kb[:, loc] @ ab, axis=1)
                for ai, arm in enumerate(ARMS):
                    pe = softmax(f["priors"]["r50"] + kernels[arm][:, loc] @ alphas[arm], axis=1)
                    for hi, value in enumerate((pe, .5*(pb + pe))):
                        if not np.isfinite(value).all():
                            raise ValueError("Nonfinite probabilities")
                        predictions[ki, draw, ai, hi, s] = value.argmax(1)
                        probabilities[ki, draw, ai, hi, s] = value
            if n > 20000 and lo % (BATCH*40) == 0:
                print("CHUNK", suffix, kind, s.stop, n, flush=True)
        for arm, old_arm in (("R", "RR"), ("T", "RD")):
            for hi, head in enumerate(READOUTS):
                np.testing.assert_array_equal(predictions[ki, :, ARMS.index(arm), hi],
                                              old_preds[:, old_modes.index(old_arm + "_" + head)])
        diagnostics.append(dict(kind=kind, teacher_energy=energy[0]/n, predictable_energy=energy[1]/n,
                                residual_energy=energy[2]/n, cross_energy=energy[3]/n))
        print("PREDICTED", suffix, kind, round(time.perf_counter()-started, 2), flush=True)
    for arm in ("R", "T"):
        np.testing.assert_array_equal(predictions[0, :, ARMS.index(arm)], predictions[1, :, ARMS.index(arm)])
    probabilities.flush()
    del probabilities
    np.savez_compressed(dst / "predictions.npz", predictions=predictions, kinds=np.array(KINDS),
        arms=np.array(ARMS), readouts=np.array(READOUTS),
        **{k: data[k] for k in ("rows", "ids", "support", "eval_masks")})
    np.savez_compressed(dst / "coefficients.npz", **all_coefficients)
    write_json(dst / "COMPLETE.json", dict(dataset=info["dataset"], task=info["task"], n=n,
        seconds=time.perf_counter()-started, original_RR_RD_predictions_exact=True,
        maximum_kernel_decomposition_error=max_kernel_error, energies=diagnostics,
        query_classification_labels_evaluated=False))
    gc.collect()


def report():
    tasks = folders()
    if not all((OUT / "targets" / info_for(f)[1] / "COMPLETE.json").exists() for f in tasks):
        raise RuntimeError("Finish all frozen tasks before query evaluation")
    write_json(OUT / "ALL_PREDICTIONS_SAVED.json", dict(tasks=len(tasks), time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    records, contrasts, energies, classes = [], [], [], []
    for folder in tasks:
        info, suffix = info_for(folder)
        dst = OUT / "targets" / suffix
        with np.load(dst / "predictions.npz") as z:
            saved = {k: z[k] for k in z.files}
        probs = np.load(dst / "probabilities.npy", mmap_mode="r")
        manifest, idcol = ((SOURCE / "manifest.csv", "instance_id") if info["dataset"] == "self24ports"
                          else (ROOT / DATASETS[info["dataset"]][0] / "target_manifest.csv", "sample_id"))
        frame = pd.read_csv(manifest, usecols=[idcol, "class_id"], keep_default_na=False).iloc[saved["rows"]]
        np.testing.assert_array_equal(frame[idcol].to_numpy(str), saved["ids"])
        truth = frame.class_id.to_numpy(int)
        with np.load(folder / "target.npz") as z:
            np.testing.assert_array_equal(truth[saved["support"]], z["support_labels"])
        common = dict(dataset=info["dataset"], task=info["task"])
        energies.extend(dict(**common, **row) for row in json.loads((dst / "COMPLETE.json").read_text())["energies"])
        for ki, kind in enumerate(KINDS):
            for draw, ev in enumerate(saved["eval_masks"]):
                y = truth[ev]
                for hi, head in enumerate(READOUTS):
                    byarm = {}
                    for ai, arm in enumerate(ARMS):
                        pred = saved["predictions"][ki, draw, ai, hi, ev]
                        probability = np.asarray(probs[ki, draw, ai, hi, ev], float)
                        brier = np.sum((probability-np.eye(8)[y])**2, axis=1)
                        nll = -np.log(np.maximum(probability[np.arange(len(y)), y], 1e-15))
                        class_brier = [np.mean(brier[y == c]) for c in np.unique(y)]
                        row = dict(**common, kind=kind, draw=draw, head=head, arm=arm, n=len(y),
                                   oa=100*np.mean(pred == y), ba=ba(y, pred),
                                   brier_micro=float(brier.mean()), brier_macro=float(np.mean(class_brier)),
                                   nll_micro=float(nll.mean()),
                                   nll_macro=float(np.mean([nll[y == c].mean() for c in np.unique(y)])))
                        records.append(row)
                        byarm[arm] = (row, pred)
                        for c in np.unique(y):
                            mask = y == c
                            classes.append(dict(**common, kind=kind, draw=draw, head=head, arm=arm,
                                class_id=int(c), n=int(mask.sum()), recall=100*np.mean(pred[mask] == c),
                                brier=float(brier[mask].mean())))
                    for candidate, control in (("P", "R"), ("E", "R"), ("PE", "P"), ("T", "P"), ("T", "PE"), ("T", "R")):
                        a, pa = byarm[candidate]
                        b, pb = byarm[control]
                        contrasts.append(dict(**common, kind=kind, draw=draw, head=head,
                            contrast=candidate+"-"+control, delta_ba=a["ba"]-b["ba"],
                            risk_gain_micro=b["brier_micro"]-a["brier_micro"],
                            risk_gain_macro=b["brier_macro"]-a["brier_macro"],
                            rescues=int(np.sum((pa == y) & (pb != y))),
                            harms=int(np.sum((pa != y) & (pb == y)))))
    metrics = pd.DataFrame(records)
    effects = pd.DataFrame(contrasts)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    effects.to_csv(OUT / "effects.csv", index=False)
    pd.DataFrame(classes).to_csv(OUT / "classes.csv", index=False)
    pd.DataFrame(energies).to_csv(OUT / "energies.csv", index=False)
    values = ["oa", "ba", "brier_micro", "brier_macro", "nll_micro", "nll_macro"]
    task_means = metrics.groupby(["dataset", "task", "kind", "head", "arm"])[values].mean().reset_index()
    task_means.to_csv(OUT / "task_means.csv", index=False)
    summary = task_means.groupby(["dataset", "kind", "head", "arm"])[values].mean().reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    ev = ["delta_ba", "risk_gain_micro", "risk_gain_macro", "rescues", "harms"]
    task_effects = effects.groupby(["dataset", "task", "kind", "head", "contrast"])[ev].mean().reset_index()
    task_effects.to_csv(OUT / "task_effects.csv", index=False)
    task_effects.groupby(["dataset", "kind", "head", "contrast"])[ev].mean().reset_index().to_csv(OUT / "effect_summary.csv", index=False)
    effects.groupby(["dataset", "draw", "kind", "head", "contrast"])[ev].mean().reset_index().to_csv(OUT / "draw_effects.csv", index=False)
    write_json(OUT / "COMPLETE.json", dict(tasks=len(tasks), draws=5*len(tasks), metric_rows=len(metrics),
        original_RR_RD_predictions_exact=True, all_predictions_saved_before_query_evaluation=True,
        support_budget=64, no_new_training=True, no_target_tuning=True))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    freeze_protocol()
    with threadpool_limits(limits=2):
        if args.report:
            report()
        else:
            for folder in folders():
                if not args.tasks or info_for(folder)[0]["task"] in args.tasks:
                    predict(folder)
