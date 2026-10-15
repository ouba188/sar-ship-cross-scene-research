"""Same fitted source models and interventions across public and held-port tasks."""

from replication_context import *
from scipy.special import softmax
from threadpoolctl import threadpool_limits
import argparse


def states(f, data):
    baseline = np.empty((5, 2, len(data["ids"]), 8), float)
    result = []
    for draw, s in enumerate(data["support"]):
        kr = f["r"] @ f["r"][s].T + 1.
        kb = f["t"] @ f["t"][s].T
        args = (f["pr"][s], data["support_labels"][draw], 8.)
        ar = rc.run.evaluate.coefficients(kr[s], *args)
        at = rc.run.evaluate.coefficients((kr+kb)[s], *args)
        a, b = f["pr"]+kr @ at, kb @ at
        baseline[draw, 0] = f["pr"] + kr @ ar
        baseline[draw, 1] = a+b
        result.append((ar, at, a, b))
    return baseline, result


def predict(name, fold, bundle):
    dst = OUT / "predictions" / name / f"fold_{fold}"
    if (dst / "COMPLETE.json").exists():
        return
    data, frame, controls, dataset, positions = task_data(name)
    dst.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dst / "cohort.csv", index=False)
    controls.to_csv(dst / "regions.csv", index=False)
    np.savez_compressed(dst / "cohort.npz", **{k: data[k] for k in ("ids", "support", "support_labels", "eval_masks")})
    archived, _ = states(rc.encode(data, bundle), data)
    oldpath = CROSS.OUT / f"fold_{fold}/targets/shared_frozen" / name
    saved = rc.run.archive(oldpath / "predictions.npz")
    oldp = np.load(oldpath / "probabilities.npy", mmap_mode="r")
    for k in ("ids", "support", "eval_masks"):
        np.testing.assert_array_equal(data[k], saved[k])
    replays = []
    for i, arm in enumerate(("R", "T")):
        ai = list(saved["arms"]).index(arm)
        expected = np.asarray(oldp[0, :, ai, 0])
        p = softmax(archived[:, i], axis=-1).astype(np.float32)
        np.testing.assert_allclose(p, expected, atol=2e-7, rtol=0.)
        np.testing.assert_array_equal(p.argmax(-1), expected.argmax(-1))
        replays.append(dict(arm=arm, max_probability_error=float(np.max(np.abs(p-expected)))))
    original = rc.encode(load_features(dataset, positions, "original"), bundle)
    baseline, base_states = states(original, data)
    np.save(dst / "archived_baseline.npy", archived)
    np.save(dst / "baseline.npy", baseline)
    numerical = []
    for i, arm in enumerate(("R", "T")):
        flips = (baseline[:, i].argmax(-1) != archived[:, i].argmax(-1)) & data["eval_masks"]
        numerical.append(dict(arm=arm, query_prediction_flips=int(flips.sum()),
            query_count=int(data["eval_masks"].sum()),
            probability_max_error=float(np.max(np.abs(softmax(baseline[:, i], axis=-1)-softmax(archived[:, i], axis=-1))))))
    for condition in CONDITIONS:
        f = original if condition in ("original", "replay_original") else rc.encode(load_features(dataset, positions, condition), bundle)
        scores = np.empty((5, len(PATHS), len(frame), 8), float)
        joint, _ = states(f, data)
        for draw, s in enumerate(data["support"]):
            ar, at, a0, b0 = base_states[draw]
            qr = f["r"] @ original["r"][s].T + 1.
            qt = f["t"] @ original["t"][s].T
            a, b = f["pr"] + qr @ at, qt @ at
            scores[draw, 0] = f["pr"] + qr @ ar
            scores[draw, 1] = a+b0
            scores[draw, 2] = a0+b
            scores[draw, 3] = a+b
            scores[draw, 4:] = joint[draw]
            np.testing.assert_allclose(scores[draw, 1]+scores[draw, 2]-baseline[draw, 1],
                                       scores[draw, 3], atol=1e-10, rtol=1e-10)
        if condition in ("original", "replay_original"):
            for i, j in enumerate((0, 1, 1, 1, 0, 1)):
                np.testing.assert_allclose(scores[:, i], baseline[:, j], atol=1e-11, rtol=0.)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite scores")
        np.save(dst / (condition + ".npy"), scores)
    write(dst / "COMPLETE.json", dict(task=name, dataset=dataset, fold=fold, rows=len(frame),
        legacy_replays=replays, numerical_control=numerical,
        valid_queries=(data["eval_masks"] & controls.region_valid.to_numpy(bool)).sum(1).tolist(),
        invalid_support_counts=[int((~controls.region_valid.to_numpy(bool))[s].sum()) for s in data["support"]],
        source_rows=len(bundle["source_rows"]), neural_training=False, source_refit=False))
    print("TASK_FROZEN", name, fold, len(frame), numerical, flush=True)


def scope_masks(frame, controls, data, draw):
    ev = data["eval_masks"][draw] & controls.region_valid.to_numpy(bool)
    identities = frame.identity.astype(str)
    support_ids = identities.iloc[data["support"][draw]]
    support_ids = support_ids[support_ids.str.fullmatch(r"[1-9]\d{8}")]
    values = dict(full=ev, sham_valid=ev & controls.sham_valid.to_numpy(bool),
                  unseen_support_vessel=ev & identities.str.fullmatch(r"[1-9]\d{8}").to_numpy() &
                  ~identities.isin(support_ids).to_numpy())
    if frame.dataset_name.iloc[0] != "ports":
        for product in ("GRDH", "SLC"):
            values[product] = ev & frame.scene_id.str.contains("_"+product+"_", regex=False).to_numpy()
        values["clean"] = ev & frame.noise_xml.astype(str).eq("0").to_numpy()
        values["no_other_release_vessel"] = ev & ~frame.other_release_overlap.to_numpy(bool)
    return {k: v for k, v in values.items() if v.any()}


def analyze(name, fold):
    dst = OUT / "predictions" / name / f"fold_{fold}"
    if (dst / "READOUT_COMPLETE.json").exists():
        return
    if not (dst / "COMPLETE.json").exists():
        raise RuntimeError("Unfinished predictions")
    data = rc.run.archive(dst / "cohort.npz")
    frame = pd.read_csv(dst / "cohort.csv", keep_default_na=False)
    controls = pd.read_csv(dst / "regions.csv")
    dataset = name if name in ("oss1", "oss2") else "ports"
    frame["dataset_name"] = dataset
    if dataset in ("oss1", "oss2"):
        other = frame_for("oss2" if dataset == "oss1" else "oss1")
        frame["other_release_overlap"] = frame.identity.astype(str).isin(other.identity.astype(str))
    yall = frame.class_id.to_numpy(int)
    baseline = np.load(dst / "baseline.npy", mmap_mode="r")
    rows, utility, classes = [], [], []
    for condition in CONDITIONS:
        scores = np.load(dst / (condition+".npy"), mmap_mode="r")
        for draw in range(5):
            for scope, selected in scope_masks(frame, controls, data, draw).items():
                y = yall[selected]
                w = rc.weights(y)
                r0 = baseline[draw, 0][selected].argmax(-1)
                t0 = baseline[draw, 1][selected].argmax(-1)
                rescue0, harm0 = (r0 != y) & (t0 == y), (r0 == y) & (t0 != y)
                meta = dict(task=name, dataset=dataset, fold=fold, draw=draw, scope=scope, condition=condition,
                            n=len(y), classes_present=len(np.unique(y)))
                for i, path in enumerate(PATHS):
                    z = scores[draw, i][selected]
                    p = z.argmax(-1)
                    b = baseline[draw, 0 if path.startswith("R_") else 1][selected]
                    bp = b.argmax(-1)
                    md = rc.class_margin(z, y)-rc.class_margin(b, y)
                    rows.append(dict(**meta, path=path, ba=w[p == y].sum(), oa=100*np.mean(p == y),
                        delta_ba=w[p == y].sum()-w[bp == y].sum(), flips_macro=w[p != bp].sum(),
                        old_rescue_mass=w[rescue0].sum(), old_rescue_lost=w[rescue0 & (p != y)].sum(),
                        old_harm_corrected=w[harm0 & (p == y)].sum(), margin_delta=rc.macro(md, y)))
                    for c in np.unique(y):
                        cm = y == c
                        classes.append(dict(**meta, path=path, class_id=int(c), count=int(cm.sum()),
                            recall=100*np.mean(p[cm] == c),
                            delta_recall=100*(np.mean(p[cm] == c)-np.mean(bp[cm] == c))))
                for mode, ri, ti in (("query", 0, 3), ("joint", 4, 5)):
                    r, t = scores[draw, ri][selected].argmax(-1), scores[draw, ti][selected].argmax(-1)
                    rescue, harm = (r != y) & (t == y), (r == y) & (t != y)
                    net = w[rescue].sum()-w[harm].sum()
                    np.testing.assert_allclose(net, w[t == y].sum()-w[r == y].sum(), atol=1e-10, rtol=0.)
                    utility.append(dict(**meta, mode=mode, r_ba=w[r == y].sum(), t_ba=w[t == y].sum(),
                        net=net, rescue=w[rescue].sum(), harm=w[harm].sum(),
                        original_net=w[rescue0].sum()-w[harm0].sum(),
                        retained_rescue=w[rescue0 & rescue].sum(),
                        old_rescue_both_correct=w[rescue0 & (r == y) & (t == y)].sum(),
                        old_rescue_t_wrong=w[rescue0 & (t != y)].sum()))
    for filename, records in (("path_effects", rows), ("utility", utility), ("classes", classes)):
        pd.DataFrame(records).to_csv(dst / (filename+".csv"), index=False)
    write(dst / "READOUT_COMPLETE.json", dict(path_rows=len(rows), utility_rows=len(utility), class_rows=len(classes)))
    print("TASK_ANALYZED", name, fold, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("predict", "analyze"))
    parser.add_argument("scope", choices=("public", "ports", "all"))
    args = parser.parse_args()
    tasks = [(name, fold) for name, fold in all_tasks() if args.scope == "all" or
             ((name in ("oss1", "oss2")) == (args.scope == "public"))]
    for fold in sorted(set(f for n, f in tasks)):
        bundle = bundle_for(fold) if args.stage == "predict" else None
        for name, task_fold in tasks:
            if task_fold == fold:
                predict(name, fold, bundle) if args.stage == "predict" else analyze(name, fold)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
