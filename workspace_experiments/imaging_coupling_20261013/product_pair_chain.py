"""Matched product pairs: decision sensitivity, branch paths, and actual utility."""

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
from pathlib import Path
import sys
import json
import pickle
import time

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / "ports_root_cause_20261013"
sys.path.insert(0, str(PREVIOUS))
import run_factorial as run
from analyze_factorial import load, metadata
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits


def macro(values, labels):
    return float(np.mean([np.mean(values[labels == c]) for c in np.unique(labels)]))


def margin(z, y):
    true = z[np.arange(len(y)), y]
    return true - (z.sum(1) - true) / (z.shape[1] - 1)


def freeze():
    data = run.target("oss2", "original")
    pairs = pd.read_csv(PREVIOUS / "dataset_audit/oss2_grdh_slc_pairs.csv", keep_default_na=False)
    pairs = pairs[pairs.same_label & pairs.same_sar_datetime].copy().reset_index(drop=True)
    pairs.insert(0, "pair_id", np.arange(len(pairs)))
    index = pd.Index(data["ids"])
    g, s = index.get_indexer(pairs.sample_id_grdh), index.get_indexer(pairs.sample_id_slc)
    if (g < 0).any() or (s < 0).any():
        raise ValueError("Paired identity missing from frozen target cohort")
    pairs.to_csv(OUT / "product_pairs_frozen.csv", index=False)
    reference = np.unique(data["support"])
    locs = [np.searchsorted(reference, x) for x in data["support"]]
    np.savez_compressed(OUT / "product_cohort.npz", ids=data["ids"], support=data["support"],
                        eval_masks=data["eval_masks"], grdh=g, slc=s)
    replays = []
    started = time.perf_counter()
    for fold in (2,3):
        with (PREVIOUS / f"fits/original/fold_{fold}/geometry.pkl").open("rb") as f:
            bundle = pickle.load(f)
        ref = run.encode(data, reference, bundle)
        kr = ref["r"] @ ref["r"].T + 1.
        kt = kr + ref["t"] @ ref["t"].T
        coef = []
        for draw,loc in enumerate(locs):
            args = (ref["pr"][loc], data["support_labels"][draw], 8.)
            coef.append((run.evaluate.coefficients(kr[np.ix_(loc,loc)], *args),
                         run.evaluate.coefficients(kt[np.ix_(loc,loc)], *args)))
        n = len(data["ids"])
        scores_r = np.empty((5,n,8),float)
        a = np.empty_like(scores_r)
        b = np.empty_like(scores_r)
        for lo in range(0,n,512):
            part = slice(lo,min(lo+512,n))
            query = run.encode(data,part,bundle)
            r = query["r"] @ ref["r"].T + 1.
            t = query["t"] @ ref["t"].T
            for draw,loc in enumerate(locs):
                ar,at = coef[draw]
                scores_r[draw,part] = query["pr"] + r[:,loc] @ ar
                a[draw,part] = query["pr"] + r[:,loc] @ at
                b[draw,part] = t[:,loc] @ at
        saved, old = load(fold,"oss2","original","original")
        for key in ("ids","support","eval_masks"):
            np.testing.assert_array_equal(saved[key],data[key])
        for arm,scores in (("R",scores_r),("T",a+b)):
            ai = list(saved["arms"]).index(arm)
            p = softmax(scores,axis=-1).astype(np.float32)
            previous = np.asarray(old[:,ai,1])
            error = float(np.max(np.abs(p-previous)))
            np.testing.assert_allclose(p,previous,atol=2e-7,rtol=0.)
            np.testing.assert_array_equal(p.argmax(-1),previous.argmax(-1))
            replays.append(dict(fold=fold,arm=arm,max_error=error))
        np.savez_compressed(OUT / f"product_scores_fold_{fold}.npz", r=scores_r[:,np.stack([g,s])],
                            a=a[:,np.stack([g,s])], b=b[:,np.stack([g,s])])
        print("PRODUCT_FROZEN",fold,round(time.perf_counter()-started,2),flush=True)
    (OUT / "PRODUCT_FROZEN.json").write_text(json.dumps(dict(replays=replays,pairs=len(pairs),
        identities=pairs.identity.nunique(),datatakes=pairs.datatake_key.nunique(),
        source_refit=False,query_outcomes_used_to_select_pairs=False),indent=2),encoding="utf-8")


def analyze():
    pairs = pd.read_csv(OUT / "product_pairs_frozen.csv",keep_default_na=False)
    with np.load(OUT / "product_cohort.npz") as z:
        cohort = {k:z[k] for k in z.files}
    yall = run.labels("oss2",cohort["ids"])
    g,s = cohort["grdh"],cohort["slc"]
    y = yall[g]
    np.testing.assert_array_equal(y,yall[s])
    frame = metadata("oss2",cohort["ids"])
    clean = (pairs.noise_xml_grdh.astype(str) == "0") & (pairs.noise_xml_slc.astype(str) == "0")
    sufficient = pairs.class_id_grdh.map(pairs.class_id_grdh.value_counts()).to_numpy() >= 5
    records, responses, swaps = [],[],[]
    for fold in (2,3):
        with np.load(OUT / f"product_scores_fold_{fold}.npz") as z:
            rr,aa,bb = z["r"],z["a"],z["b"]
        for draw,ev in enumerate(cohort["eval_masks"]):
            eligible = ev[g] & ev[s]
            support_id = frame.identity.iloc[cohort["support"][draw]]
            unknown_ship = ~pairs.identity.astype(str).isin(support_id.astype(str)).to_numpy()
            scopes = dict(all=eligible, both_clean=eligible & clean.to_numpy(),
                          unseen_support_vessel=eligible & unknown_ship,
                          classes_ge5_pairs=eligible & sufficient)
            rg,rs = rr[draw].argmax(-1)
            tg,ts = (aa[draw]+bb[draw]).argmax(-1)
            a_delta = margin(aa[draw,1],y)-margin(aa[draw,0],y)
            b_delta = margin(bb[draw,1],y)-margin(bb[draw,0],y)
            total_delta = margin(aa[draw,1]+bb[draw,1],y)-margin(aa[draw,0]+bb[draw,0],y)
            np.testing.assert_allclose(a_delta+b_delta,total_delta,atol=1e-11,rtol=1e-10)
            for scope,mask in scopes.items():
                for i in np.flatnonzero(mask):
                    row = dict(fold=fold,draw=draw,scope=scope,pair_id=int(i),identity=str(pairs.identity.iloc[i]),
                               datatake=str(pairs.datatake_key.iloc[i]),class_id=int(y[i]))
                    for arm,p0,p1 in (("R",rg,rs),("T",tg,ts)):
                        c0,c1 = p0[i] == y[i],p1[i] == y[i]
                        records.append(dict(**row,arm=arm,grdh_correct=float(c0),slc_correct=float(c1),
                            mean_correct=.5*(float(c0)+float(c1)),both_correct=float(c0 and c1),
                            flip=float(p0[i] != p1[i]),both_wrong=float(not c0 and not c1),
                            stable_wrong=float(p0[i] == p1[i] and not c0),
                            lost_correct=float(c0 and not c1),gained_correct=float(not c0 and c1)))
                    responses.append(dict(**row,a_margin_delta=a_delta[i],b_margin_delta=b_delta[i],
                        total_margin_delta=total_delta[i],opposite=float(a_delta[i]*b_delta[i] < 0),
                        reinforce=float(a_delta[i]*b_delta[i] > 0),
                        cancel_fraction=1-abs(total_delta[i])/(abs(a_delta[i])+abs(b_delta[i])+1e-12)))
                for name,agrid,bgrid in (("GG",0,0),("SG",1,0),("GS",0,1),("SS",1,1)):
                    pred = (aa[draw,agrid]+bb[draw,bgrid]).argmax(-1)
                    swaps.append(dict(fold=fold,draw=draw,scope=scope,swap=name,pairs=int(mask.sum()),
                        ba=100*macro((pred[mask] == y[mask]).astype(float),y[mask]),
                        oa=100*np.mean(pred[mask] == y[mask]),
                        flip_vs_GG=100*np.mean(pred[mask] != tg[mask])))
    raw = pd.DataFrame(records)
    raw.to_csv(OUT / "product_pair_events.csv",index=False)
    response = pd.DataFrame(responses)
    response.to_csv(OUT / "product_path_events.csv",index=False)
    pd.DataFrame(swaps).to_csv(OUT / "product_swaps.csv",index=False)
    values = ["grdh_correct","slc_correct","mean_correct","both_correct","flip","both_wrong",
              "stable_wrong","lost_correct","gained_correct"]
    pairmean = raw.groupby(["scope","arm","pair_id","identity","datatake","class_id"])[values].mean().reset_index()
    pairmean.to_csv(OUT / "product_pair_means.csv",index=False)
    summary,contexts,clsrows = [],[],[]
    for (scope,arm),df in pairmean.groupby(["scope","arm"]):
        summary.append(dict(scope=scope,arm=arm,pairs=len(df),identities=df.identity.nunique(),
            datatakes=df.datatake.nunique(),classes=df.class_id.nunique(),
            **{k+"_micro":100*df[k].mean() for k in values},
            **{k+"_macro":100*df.groupby("class_id")[k].mean().mean() for k in values}))
        for cl,cdf in df.groupby("class_id"):
            clsrows.append(dict(scope=scope,arm=arm,class_id=cl,pairs=len(cdf),
                               **{k:100*cdf[k].mean() for k in values}))
    for (fold,draw,scope,arm),df in raw.groupby(["fold","draw","scope","arm"]):
        contexts.append(dict(fold=fold,draw=draw,scope=scope,arm=arm,pairs=len(df),classes=df.class_id.nunique(),
            **{k+"_macro":100*df.groupby("class_id")[k].mean().mean() for k in values}))
    pd.DataFrame(summary).to_csv(OUT / "product_summary.csv",index=False)
    pd.DataFrame(contexts).to_csv(OUT / "product_contexts.csv",index=False)
    pd.DataFrame(clsrows).to_csv(OUT / "product_classes.csv",index=False)
    scene_rows = []
    for (fold,draw,scene,arm),df in raw[raw.scope == "all"].groupby(["fold","draw","datatake","arm"]):
        scene_rows.append(dict(fold=fold,draw=draw,scene=scene,arm=arm,pairs=len(df),classes=df.class_id.nunique(),
            **{k:100*df.groupby("class_id")[k].mean().mean() for k in ("mean_correct","both_correct","flip")}))
    scenes = pd.DataFrame(scene_rows)
    scenes.to_csv(OUT / "product_scene_contexts.csv",index=False)
    scenes.groupby(["scene","arm"])[["pairs","classes","mean_correct","both_correct","flip"]].mean().reset_index().to_csv(
        OUT / "product_scene_summary.csv",index=False)
    response_summary = []
    for scope,df in response.groupby("scope"):
        means = df.groupby(["pair_id","class_id"])[["a_margin_delta","b_margin_delta","opposite","cancel_fraction"]].mean().reset_index()
        # Correlation within each context avoids treating repeated supports as new observations.
        corrs = [g.a_margin_delta.corr(g.b_margin_delta) for _,g in df.groupby(["fold","draw"])]
        response_summary.append(dict(scope=scope,pairs=len(means),a_b_correlation_mean=float(np.mean(corrs)),
            opposite_micro=100*means.opposite.mean(),opposite_macro=100*means.groupby("class_id").opposite.mean().mean(),
            cancel_fraction_micro=means.cancel_fraction.mean()))
    pd.DataFrame(response_summary).to_csv(OUT / "product_response_summary.csv",index=False)
    print(pd.DataFrame(summary).query("scope == 'all'").to_string(index=False),flush=True)
    print(pd.DataFrame(response_summary).to_string(index=False),flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        freeze()
        analyze()
