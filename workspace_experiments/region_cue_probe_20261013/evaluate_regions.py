"""Separate fixed-path reliance from matched support/query adaptation."""

from region_context import *
from scipy.special import softmax
from threadpoolctl import threadpool_limits


def predict():
    data, frame = cohort()
    np.savez_compressed(OUT / "cohort.npz", ids=data["ids"], support=data["support"],
                        support_labels=data["support_labels"], eval_masks=data["eval_masks"])
    frame.to_csv(OUT / "cohort.csv", index=False)
    n = len(frame)
    replays = []
    numerical_control = []
    for fold in (2, 3):
        with (PREVIOUS / f"fits/original/fold_{fold}/geometry.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
        original = encode(data, bundle)
        baseline = np.empty((5, 2, n, 8), float)
        original_states = []
        for draw, s in enumerate(data["support"]):
            kr = original["r"][s] @ original["r"][s].T + 1.
            kt = kr + original["t"][s] @ original["t"][s].T
            args = (original["pr"][s], data["support_labels"][draw], 8.)
            ar = run.evaluate.coefficients(kr, *args)
            at = run.evaluate.coefficients(kt, *args)
            qr = original["r"] @ original["r"][s].T + 1.
            qt = original["t"] @ original["t"][s].T
            a = original["pr"] + qr @ at
            b = qt @ at
            baseline[draw, 0] = original["pr"] + qr @ ar
            baseline[draw, 1] = a + b
            original_states.append((ar, at, a, b))
        saved, previous = load(fold, "oss2", "original", "original")
        for key in ("ids", "support", "eval_masks"):
            np.testing.assert_array_equal(saved[key], data[key])
        for i, arm in enumerate(("R", "T")):
            old = np.asarray(previous[:, list(saved["arms"]).index(arm), 1])
            p = softmax(baseline[:, i], axis=-1).astype(np.float32)
            np.testing.assert_allclose(p, old, atol=2e-7, rtol=0.)
            np.testing.assert_array_equal(p.argmax(-1), old.argmax(-1))
            replays.append(dict(fold=fold, arm=arm, max_error=float(np.max(np.abs(p-old)))))
        np.save(OUT / f"baseline_fold_{fold}.npy", baseline)
        for condition in CONDITIONS:
            f = original if condition == "original" else encode(features(condition, data), bundle)
            scores = np.empty((5, len(PATHS), n, 8), float)
            for draw, s in enumerate(data["support"]):
                ar, at, a0, b0 = original_states[draw]
                qr = f["r"] @ original["r"][s].T + 1.
                qt = f["t"] @ original["t"][s].T
                a = f["pr"] + qr @ at
                b = qt @ at
                scores[draw, 0] = f["pr"] + qr @ ar
                scores[draw, 1] = a + b0
                scores[draw, 2] = a0 + b
                scores[draw, 3] = a + b
                jr = f["r"] @ f["r"][s].T + 1.
                jt = jr + f["t"] @ f["t"][s].T
                args = (f["pr"][s], data["support_labels"][draw], 8.)
                cr = run.evaluate.coefficients(jr[s], *args)
                ct = run.evaluate.coefficients(jt[s], *args)
                scores[draw, 4] = f["pr"] + jr @ cr
                scores[draw, 5] = f["pr"] + jt @ ct
                np.testing.assert_allclose(scores[draw, 1]+scores[draw, 2]-baseline[draw, 1],
                                           scores[draw, 3], atol=1e-10, rtol=1e-10)
            if condition == "original":
                for i, base_i in enumerate((0, 1, 1, 1, 0, 1)):
                    np.testing.assert_allclose(scores[:, i], baseline[:, base_i], atol=1e-11, rtol=0.)
            if condition == "replay_original":
                for i, base_i in enumerate((0, 1, 1, 1, 0, 1)):
                    pred = scores[:, i].argmax(-1)
                    bp = baseline[:, base_i].argmax(-1)
                    numerical_control.append(dict(fold=fold, path=PATHS[i],
                        prediction_flips=int(np.sum((pred != bp) & data["eval_masks"])),
                        max_probability_error=float(np.max(np.abs(
                            softmax(scores[:, i], axis=-1)-softmax(baseline[:, base_i], axis=-1))))))
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite scores")
            np.save(OUT / f"scores_fold_{fold}_{condition}.npy", scores)
            print("FROZEN", fold, condition, flush=True)
    write("PREDICTIONS_FROZEN.json", dict(replays=replays, numerical_control=numerical_control,
        conditions=CONDITIONS, paths=PATHS,
        folds=[2, 3], support_draws=5, neural_training=False, source_refits=False,
        query_labels_used_in_predictor=False, query_outcomes_used_to_select_conditions=False))


def analyze():
    with np.load(OUT / "cohort.npz") as z:
        data = {k: z[k] for k in z.files}
    frame = pd.read_csv(OUT / "cohort.csv", keep_default_na=False)
    control = pd.read_csv(OUT / "region_metadata.csv")
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str), control.sample_id.to_numpy(str))
    yall = frame.class_id.to_numpy(int)
    rows, utility, classes = [], [], []
    for fold in (2, 3):
        baseline = np.load(OUT / f"baseline_fold_{fold}.npy", mmap_mode="r")
        for condition in CONDITIONS:
            score = np.load(OUT / f"scores_fold_{fold}_{condition}.npy", mmap_mode="r")
            for draw, ev in enumerate(data["eval_masks"]):
                ev = ev & control.region_valid.to_numpy(bool)
                support_ships = set(frame.identity.iloc[data["support"][draw]].astype(str))
                scopes = dict(full=ev,
                    GRDH=ev & frame.scene_id.str.contains("_GRDH_", regex=False).to_numpy(),
                    SLC=ev & frame.scene_id.str.contains("_SLC_", regex=False).to_numpy(),
                    clean=ev & frame.noise_xml.astype(str).eq("0").to_numpy(),
                    unseen_support_vessel=ev & ~frame.identity.astype(str).isin(support_ships).to_numpy(),
                    sham_valid=ev & control.sham_valid.to_numpy(bool))
                for scope, selected in scopes.items():
                    if not selected.any():
                        continue
                    y = yall[selected]
                    w = weights(y)
                    r0, t0 = (
                        baseline[draw, 0][selected].argmax(-1), baseline[draw, 1][selected].argmax(-1))
                    old_rescue, old_harm = (r0 != y) & (t0 == y), (r0 == y) & (t0 != y)
                    meta = dict(fold=fold, draw=draw, condition=condition, scope=scope, n=len(y))
                    for i, path in enumerate(PATHS):
                        z = np.asarray(score[draw, i][selected])
                        p = z.argmax(-1)
                        base_i = 0 if path.startswith("R_") else 1
                        base = baseline[draw, base_i][selected]
                        bp = base.argmax(-1)
                        margin_delta = class_margin(z, y)-class_margin(base, y)
                        row = dict(**meta, path=path, ba=w[p == y].sum(), oa=100*np.mean(p == y),
                            delta_ba=w[p == y].sum()-w[bp == y].sum(),
                            flips_macro=w[p != bp].sum(), flips_micro=100*np.mean(p != bp),
                            margin_delta=macro(margin_delta, y),
                            old_rescue_mass=w[old_rescue].sum(),
                            old_rescue_lost=w[old_rescue & (p != y)].sum(),
                            old_harm_corrected=w[old_harm & (p == y)].sum())
                        rows.append(row)
                        for c in np.unique(y):
                            cm = y == c
                            classes.append(dict(**meta, path=path, class_id=int(c), count=int(cm.sum()),
                                recall=100*np.mean(p[cm] == c),
                                delta_recall=100*np.mean((p[cm] == c).astype(float)-(bp[cm] == c)),
                                margin_delta=float(np.mean(margin_delta[cm]))))
                    for mode, ri, ti in (("query", 0, 3), ("joint", 4, 5)):
                        rp, tp = score[draw, ri][selected].argmax(-1), score[draw, ti][selected].argmax(-1)
                        rescue, harm = (rp != y) & (tp == y), (rp == y) & (tp != y)
                        net = w[rescue].sum()-w[harm].sum()
                        np.testing.assert_allclose(net, w[tp == y].sum()-w[rp == y].sum(), atol=1e-10)
                        utility.append(dict(**meta, mode=mode, r_ba=w[rp == y].sum(), t_ba=w[tp == y].sum(),
                            net=net, rescue=w[rescue].sum(), harm=w[harm].sum(),
                            original_net=w[old_rescue].sum()-w[old_harm].sum(),
                            retained_rescue=w[old_rescue & rescue].sum(),
                            old_rescue_both_correct=w[old_rescue & (rp == y) & (tp == y)].sum(),
                            old_rescue_t_wrong=w[old_rescue & (tp != y)].sum(),
                            old_harm_avoided=w[old_harm & ~harm].sum()))
            print("READOUT", fold, condition, flush=True)
    for name, records, keys in (
        ("path_effects", rows, ["condition", "scope", "path"]),
        ("utility", utility, ["condition", "scope", "mode"]),
        ("classes", classes, ["condition", "scope", "path", "class_id"])):
        df = pd.DataFrame(records)
        df.to_csv(OUT / (name + ".csv"), index=False)
        values = [c for c in df.columns if c not in keys + ["fold", "draw"]]
        df.groupby(keys)[values].mean().reset_index().to_csv(OUT / (name+"_summary.csv"), index=False)
    write("READOUT_COMPLETE.json", dict(path_rows=len(rows), utility_rows=len(utility),
        class_rows=len(classes), context_mean=True, independent_training_seeds=False))
    print(pd.read_csv(OUT / "utility_summary.csv").query("scope == 'full' and mode == 'joint'").to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["predict", "analyze"])
    with threadpool_limits(limits=2):
        {"predict": predict, "analyze": analyze}[parser.parse_args().stage]()
