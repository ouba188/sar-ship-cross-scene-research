"""Separate query feature changes from support-conditioned readout changes."""

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
from pathlib import Path
import sys
import pickle
import json
import time

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / "ports_root_cause_20261013"
sys.path.insert(0, str(PREVIOUS))
import run_factorial as run
import slc_geometry
from analyze_factorial import metadata, load, weighted_events
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

CONDITIONS = ("original", "range_geometry", "axis_sham")
ARMS = ("R", "T")


def predict():
    started = time.perf_counter()
    data = {c: run.target("oss2", c) if c == "original" else slc_geometry.data_for("oss2", c)
            for c in CONDITIONS}
    original = data["original"]
    for d in data.values():
        for key in ("ids", "support", "eval_masks", "support_labels"):
            np.testing.assert_array_equal(d[key], original[key])
    reference = np.unique(original["support"])
    locations = [np.searchsorted(reference, s) for s in original["support"]]
    np.savez_compressed(OUT / "cohort.npz", **{k: original[k] for k in
                        ("ids", "support", "eval_masks", "support_labels")})
    replays = []
    for fold in (2, 3):
        with (PREVIOUS / f"fits/original/fold_{fold}/geometry.pkl").open("rb") as f:
            bundle = pickle.load(f)
        refs = {c: run.encode(d, reference, bundle) for c, d in data.items()}
        coefficients = {}
        for c, f in refs.items():
            kr = f["r"] @ f["r"].T + 1.
            kt = kr + f["t"] @ f["t"].T
            for draw, loc in enumerate(locations):
                for arm, kernel in zip(ARMS, (kr, kt)):
                    coefficients[c, draw, arm] = run.evaluate.coefficients(
                        kernel[np.ix_(loc, loc)], f["pr"][loc], original["support_labels"][draw], 8.)
        n = len(original["ids"])
        folder = OUT / f"fold_{fold}"
        folder.mkdir(exist_ok=True)
        probs = np.lib.format.open_memmap(folder / "probabilities.npy", mode="w+", dtype=np.float32,
                                         shape=(3, 3, 5, 2, n, 8))
        for qi, qc in enumerate(CONDITIONS):
            for lo in range(0, n, 512):
                part = slice(lo, min(lo + 512, n))
                f = run.encode(data[qc], part, bundle)
                for si, sc in enumerate(CONDITIONS):
                    kr = f["r"] @ refs[sc]["r"].T + 1.
                    kt = kr + f["t"] @ refs[sc]["t"].T
                    for draw, loc in enumerate(locations):
                        for ai, (arm, kernel) in enumerate(zip(ARMS, (kr, kt))):
                            logits = f["pr"] + kernel[:, loc] @ coefficients[sc, draw, arm]
                            probs[si, qi, draw, ai, part] = softmax(logits, axis=1)
        probs.flush()
        for ci, c in enumerate(CONDITIONS):
            saved, oldp = load(fold, "oss2", "original", c)
            for k in ("ids", "support", "eval_masks"):
                np.testing.assert_array_equal(saved[k], original[k])
            for ai, arm in enumerate(ARMS):
                oldai = list(saved["arms"]).index(arm)
                old = np.asarray(oldp[:, oldai, 1])
                current = np.asarray(probs[ci, ci, :, ai])
                error = float(np.max(np.abs(current - old)))
                np.testing.assert_allclose(current, old, atol=2e-7, rtol=0.)
                np.testing.assert_array_equal(current.argmax(-1), old.argmax(-1))
                replays.append(dict(fold=fold, condition=c, arm=arm, max_probability_error=error))
        print("FROZEN", fold, round(time.perf_counter() - started, 2), flush=True)
    (OUT / "FROZEN.json").write_text(json.dumps(dict(replays=replays, seconds=time.perf_counter()-started,
        source_refit=False, neural_training=False, query_labels_used_for_fit=False), indent=2), encoding="utf-8")


def analyze():
    with np.load(OUT / "cohort.npz") as a:
        cohort = {k: a[k] for k in a.files}
    frame = metadata("oss2", cohort["ids"])
    yall = run.labels("oss2", cohort["ids"])
    slc = frame.scene_id.str.contains("_SLC_", regex=False).to_numpy()
    pairs = pd.read_csv(PREVIOUS / "dataset_audit/oss2_grdh_slc_pairs.csv", keep_default_na=False)
    pairs = pairs[pairs.same_label]
    index = pd.Index(cohort["ids"])
    gi, si = index.get_indexer(pairs.sample_id_grdh), index.get_indexer(pairs.sample_id_slc)
    rows, classes, changes = [], [], []
    for fold in (2, 3):
        probs = np.load(OUT / f"fold_{fold}/probabilities.npy", mmap_mode="r")
        for draw, ev in enumerate(cohort["eval_masks"]):
            paired = np.zeros(len(yall), bool)
            paired[si[ev[gi] & ev[si]]] = True
            scopes = dict(full=ev, SLC=ev & slc, GRDH=ev & ~slc, paired_SLC=paired)
            for scope, mask in scopes.items():
                y = yall[mask]
                for support_i, support in enumerate(CONDITIONS):
                    for query_i, query in enumerate(CONDITIONS):
                        for ai, arm in enumerate(ARMS):
                            p = np.asarray(probs[support_i, query_i, draw, ai, mask], float)
                            pred = p.argmax(1)
                            baseline = probs[0, 0, draw, ai, mask].argmax(1)
                            w, rescue, harm = weighted_events(y, baseline, pred)
                            brier = ((p - np.eye(8)[y])**2).sum(1)
                            meta = dict(fold=fold, draw=draw, scope=scope, arm=arm,
                                        support=support, query=query)
                            rows.append(dict(**meta, n=len(y), oa=100*np.mean(pred == y),
                                ba=w[pred == y].sum(), brier_macro=np.dot(w, brier)/100,
                                rescue=w[rescue].sum(), harm=w[harm].sum()))
                            for cls in np.unique(y):
                                cm = y == cls
                                classes.append(dict(**meta, class_id=int(cls), n=int(cm.sum()),
                                    recall=100*np.mean(pred[cm] == cls)))
                for condition in CONDITIONS:
                    ci = CONDITIONS.index(condition)
                    for ai, arm in enumerate(ARMS):
                        for kind, before, after in (
                            ("query_only", (0,0), (0,ci)),
                            ("support_only", (0,0), (ci,0)),
                            ("support_after_query", (0,ci), (ci,ci)),
                            ("query_after_support", (ci,0), (ci,ci)),
                            ("both", (0,0), (ci,ci))):
                            old = probs[before[0], before[1], draw, ai, mask].argmax(1)
                            new = probs[after[0], after[1], draw, ai, mask].argmax(1)
                            w, rescue, harm = weighted_events(y, old, new)
                            changes.append(dict(fold=fold, draw=draw, scope=scope, arm=arm,
                                condition=condition, contrast=kind, ba=w[rescue].sum()-w[harm].sum(),
                                oa=100*(np.mean(new == y)-np.mean(old == y)),
                                rescue=w[rescue].sum(), harm=w[harm].sum()))
    for filename, records, keys, measures in (
        ("metrics", rows, ["scope","arm","support","query"], ["oa","ba","brier_macro","rescue","harm"]),
        ("class_metrics", classes, ["scope","arm","support","query","class_id"], ["recall","n"]),
        ("effects", changes, ["scope","arm","condition","contrast"], ["oa","ba","rescue","harm"])):
        table = pd.DataFrame(records)
        table.to_csv(OUT / (filename + ".csv"), index=False)
        table.groupby(keys)[measures].mean().reset_index().to_csv(OUT / (filename + "_summary.csv"), index=False)
    print(pd.read_csv(OUT / "effects_summary.csv").query(
        "scope == 'SLC' and condition == 'range_geometry'").to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        predict()
        analyze()
