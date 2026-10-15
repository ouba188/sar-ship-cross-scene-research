"""Matched public-target inference with per-task and identical-model controls."""

from common import *
import pickle
import time
import gc
import argparse
import pandas as pd
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits
from compact_fusion_source import _expert_inputs
from backbone_ablation_core import r50_views, r50_coordinates
from projection_residual_expert import fit_geometry_statistics, discarded_coordinates
from residual_expert import source_geometry
from unified_head_core import coefficients

sys.path.insert(0, str(OUT.parent / "dual_cue_decomposition_20261012"))
from run import ssl_map, decomposed_kernels


def pca_coordinates(raw, model):
    return ((model["pca"].transform(raw)-model["center"])/model["scale"]).astype(np.float32)


def prepare(regime, name):
    dst = fit_dir(regime, name)
    path = dst / "geometry.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    started = time.perf_counter()
    rows, raw, c4, extra = source_arrays(regime, name)
    labels = pd.read_csv(SOURCE / "manifest.csv", usecols=["class_id"]).class_id.to_numpy(int)[rows]
    if regime == "task_trained":
        with (joint(name) / "source_model.pkl").open("rb") as f:
            model = pickle.load(f)
        stats = archive(joint(name) / "deployment_complement.npz")
    else:
        pca = PCA(n_components=128, random_state=0).fit(raw[rows])
        x = pca.transform(raw[rows])
        center, scale = x.mean(0), x.std(0)+1e-6
        xs = (x-center)/scale
        ssl = PCA(n_components=64, random_state=0).fit(extra)
        joint_source = np.c_[xs, ssl.transform(extra)]
        _, ec, es = source_geometry(joint_source, np.arange(len(rows)))
        _, bc, bs = source_geometry(xs, np.arange(len(rows)))
        model = dict(pca=pca, pca_ssl=ssl, center=center, scale=scale,
                     expert_center=ec, expert_scale=es, residual_center=bc, residual_scale=bs)
        stats = fit_geometry_statistics(raw[rows], extra, np.arange(len(rows)), model)
    x = np.empty((len(rows), 192), float)
    x4 = np.empty((len(rows), 128), np.float32)
    for lo in range(0, len(rows), BATCH):
        s = slice(lo, min(lo+BATCH, len(rows)))
        x[s] = _expert_inputs(raw[rows[s]], extra[s], model)
        x4[s] = pca_coordinates(c4[rows[s]], model)
    heads = {"r50": RidgeClassifier(alpha=1., class_weight="balanced").fit(x[:, :128], labels),
             "dual": RidgeClassifier(alpha=1., class_weight="balanced").fit(x, labels),
             "base": RidgeClassifier(alpha=1., class_weight="balanced").fit(x4, labels)}
    for h in heads.values():
        np.testing.assert_array_equal(h.classes_, np.arange(8))
    bundle = dict(model=model, stats=stats, heads=heads, source_rows=rows)
    dst.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bundle, f)
    write(dst / "GEOMETRY_COMPLETE.json", dict(regime=regime, source_n=len(rows), seconds=time.perf_counter()-started,
        source_class_counts=np.bincount(labels, minlength=8).tolist(), target_rows_fitted=0,
        shared_across_both_targets=regime == "shared_frozen"))
    print("GEOMETRY", regime, name, len(rows), flush=True)
    return bundle


def target_arrays(regime, name, grdh_only=False):
    original_features = archive(original(name) / "features.npz")
    ids = original_features["sample_ids"]
    np.testing.assert_array_equal(np.load(ROOT / "cross_dataset_c4_20261009" / name / "sample_ids.npy"), ids)
    extra = np.load(joint(name) / ("joint" if regime == "task_trained" else "reference") / "target_extra.npy", mmap_mode="r")
    np.testing.assert_array_equal(np.load(joint(name) / "target_sample_ids.npy"), ids)
    if regime == "shared_frozen":
        np.testing.assert_array_equal(extra, original_features["vh_ssl"])
    policies = archive(original(name) / "policies.npz")
    np.testing.assert_array_equal(policies["sample_ids"], ids)
    # Only support labels are passed into the inference routine.
    manifest = pd.read_csv(original(name) / "target_manifest.csv", usecols=["sample_id", "class_id", "scene_id"], keep_default_na=False)
    np.testing.assert_array_equal(manifest.sample_id.to_numpy(str), ids)
    sy = manifest.class_id.to_numpy(int)[policies["support_rows"]]
    data = dict(raw=original_features["paired_r50"], extra=extra, ids=ids,
        c4=np.load(ROOT / "cross_dataset_c4_20261009" / name / "pooled_c4.npy", mmap_mode="r"),
        support=policies["support_rows"], support_labels=sy, eval_masks=policies["eval_masks"])
    if grdh_only:
        rows = np.flatnonzero(manifest.scene_id.str.contains("_GRDH_", regex=False).to_numpy())
        for key in ("raw", "extra", "ids", "c4"):
            data[key] = data[key][rows]
        data["support"] = np.stack([np.random.default_rng(seed).choice(len(rows), 64, replace=False) for seed in range(5)])
        data["support_labels"] = manifest.class_id.to_numpy(int)[rows[data["support"]]]
        data["eval_masks"] = np.stack([~np.isin(np.arange(len(rows)), s) for s in data["support"]])
    return data


def predict(regime, name, bundle, grdh_only=False):
    task = name + ("_grdh" if grdh_only else "")
    dst = OUT / "targets" / regime / task
    if (dst / "COMPLETE.json").exists():
        print("REUSE TARGET", regime, name, flush=True)
        return
    started = time.perf_counter()
    data = target_arrays(regime, name, grdh_only)
    mapper = archive(mapper_path(regime, name))
    model, stats, heads = (bundle[k] for k in ("model", "stats", "heads"))
    ref = np.unique(data["support"])
    locs = [np.searchsorted(ref, support) for support in data["support"]]
    n = len(data["ids"])
    dst.mkdir(parents=True, exist_ok=True)
    probabilities = np.lib.format.open_memmap(dst / "probabilities.npy", mode="w+", dtype=np.float32,
        shape=(2, 5, len(ARMS), 2, n, 8))
    predictions = np.empty(probabilities.shape[:-1], np.int8)

    def features(index):
        raw, extra = data["raw"][index], np.asarray(data["extra"][index])
        rr, rd = r50_views(raw, model, stats)
        x = _expert_inputs(raw, extra, model)
        x4 = pca_coordinates(data["c4"][index], model)
        phi = (x4-model["residual_center"])/model["residual_scale"]/np.sqrt(128)
        return dict(r=np.c_[rr, rd]*np.sqrt(.5), t=ssl_map(extra, model, stats), b=phi,
            pr=heads["r50"].decision_function(x[:, :128]).astype(np.float32).astype(float),
            pd=heads["dual"].decision_function(x).astype(np.float32).astype(float),
            pb=heads["base"].decision_function(x4).astype(np.float32).astype(float))

    fref = features(ref)
    kr = fref["r"]@fref["r"].T+1.
    kb = fref["b"]@fref["b"].T+1.
    saved_coef, diagnostics, gammas = {}, [], []
    for ki, kind in enumerate(KINDS):
        xref = np.c_[data["raw"][ref], data["c4"][ref]].astype(np.float32)
        pref = ssl_map(reconstruct(xref, mapper, kind), model, stats)
        eref = fref["t"]-pref
        kernels = decomposed_kernels(kr, pref, eref, pref, eref)
        np.testing.assert_allclose(kernels["T"], kr+fref["t"]@fref["t"].T, atol=1e-10, rtol=1e-10)
        states = []
        for draw, loc in enumerate(locs):
            support = data["support"][draw]
            if len(np.unique(support)) != 64 or data["eval_masks"][draw, support].any():
                raise ValueError("Changed support or leakage")
            sy = data["support_labels"][draw]
            gamma = float(np.sum(fref["t"][loc]**2)/(np.sum(pref[loc]**2)+np.sum(eref[loc]**2)))
            kss = {arm: kernels[arm][np.ix_(loc, loc)] for arm in ("R", "P", "E", "PE", "T")}
            kss["PE_scaled"] = kss["R"]+gamma*(kss["PE"]-kss["R"])
            kss["DD"] = kss["T"]
            alpha = {arm: coefficients(kss[arm], fref["pd" if arm == "DD" else "pr"][loc], sy, 8.) for arm in ARMS}
            ab = coefficients(kb[np.ix_(loc, loc)], fref["pb"][loc], sy, 8.)
            states.append((alpha, ab, gamma))
            gammas.append(dict(kind=kind, draw=draw, gamma=gamma))
            for arm, value in alpha.items():
                saved_coef[f"{kind}_{draw}_{arm}"] = value
        errors = energy = shifted = outside = 0.
        tsum, tsq = np.zeros(512), np.zeros(512)
        coord_energy = np.zeros(4)
        for lo in range(0, n, BATCH):
            s = slice(lo, min(lo+BATCH, n))
            f = features(s)
            x = np.c_[data["raw"][s], data["c4"][s]].astype(np.float32)
            yp = reconstruct(x, mapper, kind)
            p = ssl_map(yp, model, stats)
            e = f["t"]-p
            k = decomposed_kernels(f["r"]@fref["r"].T+1., p, e, pref, eref)
            b = f["b"]@fref["b"].T+1.
            for draw, loc in enumerate(locs):
                alphas, ab, gamma = states[draw]
                bp = softmax(f["pb"]+b[:, loc]@ab, axis=1)
                block = {a: k[a][:, loc] for a in ("R", "P", "E", "PE", "T")}
                block["DD"] = block["T"]
                block["PE_scaled"] = block["R"]+gamma*(block["PE"]-block["R"])
                for ai, arm in enumerate(ARMS):
                    ep = softmax(f["pd" if arm == "DD" else "pr"]+block[arm]@alphas[arm], axis=1)
                    for hi, value in enumerate((ep, .5*(bp+ep))):
                        if not np.isfinite(value).all():
                            raise ValueError("Nonfinite public-target probabilities")
                        probabilities[ki, draw, ai, hi, s] = value
                        predictions[ki, draw, ai, hi, s] = value.argmax(1)
            true = (np.asarray(data["extra"][s], float)-mapper["y_mean"])/mapper["y_std"]
            pred = (yp.astype(float)-mapper["y_mean"])/mapper["y_std"]
            errors += np.sum((true-pred)**2)
            tsum += true.sum(0)
            tsq += (true**2).sum(0)
            zx = (x-mapper["x_mean"])/mapper["x_std"]
            shifted += np.sum(zx.astype(float)**2)
            outside += np.sum(np.abs(zx)>3)
            coord_energy += [np.sum(f["t"]**2), np.sum(p*p), np.sum(e*e), 2*np.sum(p*e)]
        diagnostics.append(dict(kind=kind, source_scaled_mse=float(errors/(n*512)),
            target_r2=float(1-errors/np.sum(tsq-tsum**2/n)), input_squared_z=float(shifted/(n*4096)),
            input_fraction_outside3=float(outside/(n*4096)),
            **{k: float(v/n) for k,v in zip(("teacher_energy", "predictable_energy", "residual_energy", "cross_energy"), coord_energy)}))
    for arm in ("R", "T", "DD"):
        np.testing.assert_array_equal(predictions[0, :, ARMS.index(arm)], predictions[1, :, ARMS.index(arm)])
    replay = None
    if regime == "task_trained" and name == "oss2" and not grdh_only:
        oldroot = OUT.parent / "dual_cue_decomposition_20261012/targets/oss2/oss2"
        oldp = archive(oldroot / "predictions.npz")
        oldprobs = np.load(oldroot / "probabilities.npy", mmap_mode="r")
        np.testing.assert_array_equal(data["ids"], oldp["ids"])
        np.testing.assert_array_equal(data["support"], oldp["support"])
        replay = 0.
        for ai, arm in enumerate(("R", "P", "E", "PE", "T")):
            oldai = list(oldp["arms"]).index(arm)
            np.testing.assert_array_equal(predictions[:, :, ai, 0], oldp["predictions"][:, :, oldai, 0])
            delta = float(np.max(np.abs(probabilities[:, :, ai, 0]-oldprobs[:, :, oldai, 0])))
            replay = max(replay, delta)
            np.testing.assert_allclose(probabilities[:, :, ai, 0], oldprobs[:, :, oldai, 0], atol=2e-7, rtol=1e-6)
    probabilities.flush()
    del probabilities
    np.savez_compressed(dst / "predictions.npz", predictions=predictions, arms=np.array(ARMS), kinds=np.array(KINDS),
        ids=data["ids"], support=data["support"], eval_masks=data["eval_masks"], common_query=np.all(data["eval_masks"], axis=0))
    np.savez_compressed(dst / "coefficients.npz", **saved_coef)
    write(dst / "COMPLETE.json", dict(regime=regime, dataset=name, task=task, grdh_only=grdh_only, n=n, source_n=len(bundle["source_rows"]),
        seconds=time.perf_counter()-started, feature_diagnostics=diagnostics, support_gamma=gammas,
        previous_oss2_expert_prediction_exact=replay is not None, previous_oss2_probability_max_error=replay,
        target_query_metrics_evaluated=False, target_query_labels_used_for_fit=False,
        shared_geometry_path=str(fit_dir(regime, name) / "geometry.pkl"), mapper_path=str(mapper_path(regime, name))))
    print("PREDICTED", regime, name, round(time.perf_counter()-started, 2), flush=True)
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grdh-only", action="store_true")
    args = parser.parse_args()
    freeze()
    if args.grdh_only:
        write(OUT / "GRDH_PROTOCOL_ADDENDUM.json", dict(
            reason="Metadata-only audit found OSS1 entirely GRDH versus mixed GRDH/SLC in OSS2",
            defined_before_new_query_metrics=True, query_metric_readout_started=False,
            task="oss2_grdh", eligible="scene_id contains literal _GRDH_; no label-based filtering",
            support="Five uniform64 row draws, seeds0..4, within GRDH; query excludes selected support",
            source="Same fitted models/maps as complete OSS2 target; shared_frozen still uses both-public-purged source",
            comparison="OSS1 original already all GRDH; preserve OSS2 full-cohort results. Both product mix and support composition change",
            no_new_fit=True, no_search=True))
    with threadpool_limits(limits=2):
        for regime in REGIMES:
            for name in (("oss2",) if args.grdh_only else DATASETS):
                predict(regime, name, prepare(regime, name), args.grdh_only)
    marker = "GRDH_PREDICTIONS_FROZEN.json" if args.grdh_only else "PREDICTIONS_FROZEN.json"
    write(OUT / marker, dict(regimes=REGIMES, datasets=["oss2_grdh"] if args.grdh_only else DATASETS,
        all_requested_predictions_saved_before_query_metrics=True, seconds_since_epoch=time.time()))
