"""Fit coordinated source readouts and freeze all crossed target predictions."""

from settings import *
import argparse
import pickle
import time
from scipy.special import softmax
from threadpoolctl import threadpool_limits

evaluate = crossfit.configure_evaluation(2)


def bundle_for(fold, condition):
    path = (pilot.PREVIOUS / f"fold_{fold}/fits" if condition == "archive" else OUT / "fits" / condition / f"fold_{fold}")
    if condition != "archive":
        evaluate.fit_dir = lambda regime, name: path
        evaluate.source_arrays = lambda regime, name: source_arrays(fold, condition)
        bundle = evaluate.prepare("shared_frozen", str(fold))
    else:
        with (path / "geometry.pkl").open("rb") as f:
            bundle = pickle.load(f)
    np.testing.assert_array_equal(bundle["source_rows"], crossfit.permitted_rows(fold))
    return bundle


def encode(data, index, bundle):
    model, stats, heads = (bundle[k] for k in ("model", "stats", "heads"))
    raw, extra = data["raw"][index], np.asarray(data["extra"][index])
    rr, rd = evaluate.r50_views(raw, model, stats)
    x = evaluate._expert_inputs(raw, extra, model)
    x4 = evaluate.pca_coordinates(data["c4"][index], model)
    return dict(r=np.c_[rr,rd]*np.sqrt(.5), t=evaluate.ssl_map(extra,model,stats),
        b=(x4-model["residual_center"])/model["residual_scale"]/np.sqrt(128),
        pr=heads["r50"].decision_function(x[:,:128]).astype(np.float32).astype(float),
        pd=heads["dual"].decision_function(x).astype(np.float32).astype(float),
        pb=heads["base"].decision_function(x4).astype(np.float32).astype(float))


def predict(fold, source_condition, target_condition, name, bundle):
    dst = OUT / "predictions" / source_condition / target_condition / f"fold_{fold}" / name
    if (dst / "COMPLETE.json").exists():
        return
    started = time.perf_counter()
    data = target(name, target_condition)
    ref = np.unique(data["support"])
    locs = [np.searchsorted(ref,s) for s in data["support"]]
    fref = encode(data,ref,bundle)
    kr = fref["r"]@fref["r"].T+1.
    kt = kr+fref["t"]@fref["t"].T
    kb = fref["b"]@fref["b"].T+1.
    states = []
    for draw,loc in enumerate(locs):
        sy = data["support_labels"][draw]
        assert not data["eval_masks"][draw,data["support"][draw]].any()
        coef = {arm:evaluate.coefficients((kr if arm == "R" else kt)[np.ix_(loc,loc)],
            fref["pd" if arm == "DD" else "pr"][loc],sy,8.) for arm in ARMS}
        ab = evaluate.coefficients(kb[np.ix_(loc,loc)],fref["pb"][loc],sy,8.)
        states.append((coef,ab))
    dst.mkdir(parents=True,exist_ok=True)
    n = len(data["ids"])
    prob = np.lib.format.open_memmap(dst / "probabilities.npy",mode="w+",dtype=np.float32,shape=(5,3,3,n,8))
    pred = np.empty(prob.shape[:-1],np.int8)
    for lo in range(0,n,512):
        s = slice(lo,min(lo+512,n))
        f = encode(data,s,bundle)
        r = f["r"]@fref["r"].T+1.
        t = r+f["t"]@fref["t"].T
        b = f["b"]@fref["b"].T+1.
        for draw,loc in enumerate(locs):
            coef,ab = states[draw]
            bp = softmax(f["pb"]+b[:,loc]@ab,axis=1)
            for ai,arm in enumerate(ARMS):
                prior = f["pd" if arm == "DD" else "pr"]
                ep = softmax(prior+(r if arm == "R" else t)[:,loc]@coef[arm],axis=1)
                for hi,p in enumerate((softmax(prior,axis=1),ep,.5*(ep+bp))):
                    assert np.isfinite(p).all()
                    prob[draw,ai,hi,s] = p
                    pred[draw,ai,hi,s] = p.argmax(1)
    prob.flush()
    replay = None
    if source_condition == "archive":
        oldpath = INPUT / "predictions" / target_condition / f"fold_{fold}/targets/shared_frozen" / name
        saved = archive(oldpath / "predictions.npz")
        oldp = np.load(oldpath / "probabilities.npy",mmap_mode="r")
        for k in ("ids","support","eval_masks"):
            np.testing.assert_array_equal(saved[k],data[k])
        errors = []
        for ai,arm in enumerate(ARMS):
            oldai = list(saved["arms"]).index(arm)
            for hi in (1,2):
                np.testing.assert_array_equal(pred[:,ai,hi],saved["predictions"][1,:,oldai,hi-1])
                error = float(np.max(np.abs(prob[:,ai,hi]-oldp[1,:,oldai,hi-1])))
                if error > 2e-7:
                    raise ValueError("Legacy probability replay failed")
                errors.append(error)
        replay = max(errors)
    np.savez_compressed(dst / "predictions.npz",predictions=pred,ids=data["ids"],support=data["support"],
        eval_masks=data["eval_masks"],arms=np.array(ARMS),heads=np.array(HEADS))
    write(dst / "COMPLETE.json",dict(fold=fold,source_condition=source_condition,target_condition=target_condition,
        task=name,rows=n,source_rows=len(bundle["source_rows"]),legacy_probability_max_error=replay,
        query_labels_used_for_fit=False,query_metrics_evaluated=False,seconds=time.perf_counter()-started))
    print("FROZEN",source_condition,target_condition,fold,name,round(time.perf_counter()-started,1),flush=True)


def main(stage):
    if stage == "fit":
        for fold in FOLDS:
            for condition in CONDITIONS:
                bundle_for(fold,condition)
        write(OUT / "FITS_COMPLETE.json",dict(fits=4,neural_training_runs=0,mapper_training_runs=0))
    else:
        sources = ("archive",) if stage == "replay" else CONDITIONS
        if stage == "predict" and not (OUT / "FITS_COMPLETE.json").exists():
            raise RuntimeError("Fit source models before target prediction")
        for fold in FOLDS:
            for source in sources:
                bundle = bundle_for(fold,source)
                for condition in CONDITIONS:
                    for name in pilot.tasks(fold):
                        predict(fold,source,condition,name,bundle)
        write(OUT / ("REPLAY_COMPLETE.json" if stage == "replay" else "PREDICTIONS_FROZEN.json"),
            dict(contexts=18 if stage == "replay" else 36,source_conditions=sources,target_conditions=CONDITIONS,
                query_metrics_evaluated=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=["fit","replay","predict"])
    with threadpool_limits(limits=2):
        main(parser.parse_args().stage)
