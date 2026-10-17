"""Source-only small probes for selecting frozen body/ring classifier endpoints."""

import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from threadpoolctl import threadpool_limits

from run_body_context import FIELDS, packed_task, predict

HERE = Path(__file__).resolve().parent
ROOT = HERE / "body_context_expansion"
OUT = ROOT / "ring_predictability"
FEATURES = [
    "body_gap", "mix_gap", "body_entropy", "mix_entropy", "probability_l1",
    "body_support_advantage_new", "ring_support_advantage_new", "ring_best_match",
    "ring_class_gap", "body_support_agrees_body", "ring_support_agrees_mix",
    "body_ring_support_agree", "body_class_support_fraction", "mix_class_support_fraction",
    "support_class_fraction", "ring_loo_stability", "body_loo_stability", "ring_loo_votes_mix",
]
UNIT = ["pool", "split", "port", "acquisition", "budget", "origin", "draw", "task"]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def class_scores(similarity, labels, remove=-1):
    score = np.full((len(similarity), 8), -1.0)
    for c in np.unique(labels):
        pos = np.flatnonzero(labels == c)
        pos = pos[pos != remove]
        if len(pos):
            values = np.sort(similarity[:, pos], axis=1)
            score[:, c] = values[:, -min(3, len(pos)):].mean(axis=1)
    return score


def available_features(task, pbody, pmix):
    """Only support labels and query-visible tensors enter these 18 features."""
    q, s = task["q"].numpy(), task["s"].numpy()
    ys = task["labels"][s].numpy()
    bsim = task["ship"][q].numpy() @ task["ship"][s].numpy().T
    rsim = task["ring"][q].numpy() @ task["ring"][s].numpy().T
    bs, rs = class_scores(bsim, ys), class_scores(rsim, ys)
    bp, mp = pbody.argmax(1), pmix.argmax(1)
    bc, rc = bs.argmax(1), rs.argmax(1)
    bvote = np.stack([class_scores(bsim, ys, i).argmax(1) for i in range(len(s))], axis=1)
    rvote = np.stack([class_scores(rsim, ys, i).argmax(1) for i in range(len(s))], axis=1)
    idx = np.arange(len(q))
    counts = np.bincount(ys, minlength=8) / len(s)
    gap = lambda p: np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2]
    entropy = lambda p: -(p * np.log(p.clip(1e-12))).sum(1)
    values = [gap(pbody), gap(pmix), entropy(pbody), entropy(pmix), abs(pmix-pbody).sum(1),
        bs[idx, mp]-bs[idx, bp], rs[idx, mp]-rs[idx, bp], rs.max(1), gap(rs),
        bc == bp, rc == mp, bc == rc, counts[bp], counts[mp],
        np.full(len(q), len(np.unique(ys))/8), (rvote == rc[:, None]).mean(1),
        (bvote == bc[:, None]).mean(1), (rvote == mp[:, None]).mean(1)]
    result = pd.DataFrame(dict(zip(FEATURES, values)), dtype=float)
    assert np.isfinite(result.to_numpy()).all()
    return result


def extract(folder, tasks, pool, split):
    with np.load(folder / "readout.npz") as z:
        arrays = {k: z[k] for k in (*FIELDS, "weights")}
    meta = pd.read_csv(folder / "metadata.csv", dtype={"mmsi": str})
    result = []
    for ti, info in enumerate(tasks):
        task = packed_task(arrays, info)
        local_mix = torch.cat((np.sqrt(.5)*task["ship"], np.sqrt(.5)*task["ring"]), dim=1)
        local_half = torch.cat((np.sqrt(.5)*task["ship"], np.sqrt(.5)*task["expert"][:,128:256]/4), dim=1)
        with torch.no_grad():
            pb = predict(task, task["ship"]).numpy()
            pm = predict(task, local_mix).numpy()
            ph = predict(task, local_half).numpy()
        f = available_features(task, pb, pm)
        # A real leakage test: query labels may change metrics but not predictions/features.
        if ti == 0:
            altered = dict(task)
            altered["labels"] = task["labels"].clone()
            altered["labels"][task["q"]] = (altered["labels"][task["q"]] + 3) % 8
            with torch.no_grad():
                pb2 = predict(altered, task["ship"]).numpy()
                pm2 = predict(altered, local_mix).numpy()
            np.testing.assert_array_equal(pb, pb2)
            np.testing.assert_array_equal(pm, pm2)
            np.testing.assert_array_equal(f.to_numpy(), available_features(altered, pb2, pm2).to_numpy())
        qmeta = meta.iloc[info["q"]]
        f["sample_id"] = qmeta.sample_id.to_numpy()
        f["mmsi"] = qmeta.mmsi.to_numpy()
        f["truth"] = task["labels"][task["q"]].numpy()
        for key in ("port", "acquisition", "origin", "draw"):
            f[key] = info[key]
        f["pool"], f["split"], f["task"] = pool, split, ti
        f["budget"] = len(info["s"])
        f["fold"] = info.get("fold", info["acquisition"])
        for name, p in (("body", pb), ("mix", pm), ("half", ph)):
            f[name+"_pred"] = p.argmax(1)
            for c in range(8):
                f[f"{name}_p{c}"] = p[:, c]
        f["disagree"] = f.body_pred.ne(f.mix_pred)
        f["gain"] = f.mix_pred.eq(f.truth).astype(int) - f.body_pred.eq(f.truth).astype(int)
        result.append(f)
    return pd.concat(result, ignore_index=True)


def risk_weights(frame):
    """Equal ports, acquisitions, draws/tasks, and present query classes."""
    nclass = frame.groupby(UNIT).truth.transform("nunique")
    nrow = frame.groupby(UNIT+["truth"]).truth.transform("size")
    ntask = frame[UNIT].drop_duplicates().groupby(["port", "acquisition"]).size()
    nacq = frame[["port", "acquisition"]].drop_duplicates().groupby("port").size()
    tasks = np.array([ntask.loc[(p, a)] for p, a in zip(frame.port, frame.acquisition)])
    acqs = frame.port.map(nacq).to_numpy()
    return 1.0 / (nclass.to_numpy()*nrow.to_numpy()*tasks*acqs*frame.port.nunique())


def fit_models(fit):
    train = fit.loc[fit.origin.eq("cross")].copy()
    weights = risk_weights(train)
    event = train.gain.ne(0).to_numpy()
    x, y = train.loc[event, FEATURES], train.loc[event, "gain"].eq(1).to_numpy(int)
    w = weights[event]
    w = w / w.mean() if len(w) else w
    models = {}
    if len(np.unique(y)) == 2:
        scaler = StandardScaler().fit(train[FEATURES], sample_weight=weights)
        logistic = LogisticRegression(C=.1, solver="lbfgs", max_iter=1000, random_state=20261015)
        logistic.fit(scaler.transform(x), y, sample_weight=w)
        models["logistic"] = make_pipeline(scaler, logistic)
        tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10,
            min_weight_fraction_leaf=.1, random_state=20261015)
        tree.fit(x, y, sample_weight=w)
        models["tree"] = tree
    return models


def score_models(frame, models):
    out = frame.copy()
    for name in ("logistic", "tree"):
        out["score_"+name] = models[name].predict_proba(frame[FEATURES])[:,1] if name in models else 0.
        out["choose_"+name] = out["score_"+name].gt(.5) & out.disagree
    return out


def metrics(frame, chosen, arm):
    pbody = frame[[f"body_p{c}" for c in range(8)]].to_numpy()
    pmix = frame[[f"mix_p{c}" for c in range(8)]].to_numpy()
    if arm == "half":
        prob = frame[[f"half_p{c}" for c in range(8)]].to_numpy()
    else:
        prob = np.where(np.asarray(chosen)[:,None], pmix, pbody)
    y = frame.truth.to_numpy(int)
    correct = prob.argmax(1) == y
    body_ok = pbody.argmax(1) == y
    values = frame[UNIT+["truth", "mmsi"]].copy()
    values["oa"] = correct.astype(float)*100
    values["macro_nll"] = -np.log(prob[np.arange(len(y)),y].clip(1e-12))
    values["macro_brier"] = ((prob-np.eye(8)[y])**2).sum(1)
    values["rescued"] = correct & ~body_ok
    values["harmed"] = ~correct & body_ok
    values["switched"] = np.asarray(chosen, dtype=float)
    unit = values.groupby(UNIT).agg(oa=("oa","mean"),rescued=("rescued","sum"),harmed=("harmed","sum"),
        switched=("switched","mean"),n=("oa","size"))
    macro = values.groupby(UNIT+["truth"])[["oa","macro_nll","macro_brier"]].mean().groupby(UNIT).mean()
    unit = unit.join(macro.rename(columns={"oa":"ba"}))
    expected_delta = 100*(unit.rescued-unit.harmed)/unit.n
    body_oa = pd.Series(body_ok*100., index=frame.index).groupby([frame[k] for k in UNIT]).mean()
    np.testing.assert_allclose(unit.oa-body_oa, expected_delta, atol=1e-10)
    unit["arm"] = arm
    return unit.reset_index()


def random_expected(frame, chosen, arm):
    """Exact risk expectation of uniform same-count switching among disagreements."""
    tmp = frame[UNIT+["truth"]].copy()
    n = frame.disagree.groupby([frame[k] for k in UNIT]).transform("sum").to_numpy()
    count = pd.Series(np.asarray(chosen), index=frame.index).groupby([frame[k] for k in UNIT]).transform("sum").to_numpy()
    rate = np.divide(count, n, out=np.zeros(len(frame)), where=n>0)*frame.disagree.to_numpy()
    y = frame.truth.to_numpy(int)
    p0 = frame[[f"body_p{c}" for c in range(8)]].to_numpy()
    p1 = frame[[f"mix_p{c}" for c in range(8)]].to_numpy()
    c0, c1 = p0.argmax(1)==y, p1.argmax(1)==y
    tmp["oa"] = 100*(c0+rate*(c1.astype(float)-c0))
    tmp["macro_nll"] = (1-rate)*(-np.log(p0[np.arange(len(y)),y].clip(1e-12))) + rate*(-np.log(p1[np.arange(len(y)),y].clip(1e-12)))
    tmp["macro_brier"] = (1-rate)*((p0-np.eye(8)[y])**2).sum(1) + rate*((p1-np.eye(8)[y])**2).sum(1)
    tmp["rescued"] = rate*(~c0 & c1)
    tmp["harmed"] = rate*(c0 & ~c1)
    tmp["switched"] = rate
    unit = tmp.groupby(UNIT).agg(oa=("oa","mean"),rescued=("rescued","sum"),harmed=("harmed","sum"),
        switched=("switched","mean"),n=("oa","size"))
    macro = tmp.groupby(UNIT+["truth"])[["oa","macro_nll","macro_brier"]].mean().groupby(UNIT).mean()
    unit = unit.join(macro.rename(columns={"oa":"ba"})).reset_index()
    unit["arm"] = arm+"_random"
    return unit


def evaluate(frame):
    tables = []
    for arm in ("body", "mix", "half", "logistic", "tree", "source_selected"):
        chosen = (frame.disagree if arm=="mix" else np.zeros(len(frame),bool)) if arm in ("body","mix","half") else frame["choose_"+arm].to_numpy()
        # Fixed mix includes its unchanged-class probability updates as well.
        if arm=="mix":
            chosen = np.ones(len(frame),bool)
        tables.append(metrics(frame, chosen, arm))
        if arm in ("logistic","tree","source_selected"):
            tables.append(random_expected(frame, chosen, arm))
    return pd.concat(tables,ignore_index=True)


def aggregate(units, extra=()):
    keys = ["pool", "split", "budget", "origin", "arm", *extra]
    cols = ["oa","ba","macro_nll","macro_brier","switched"]
    scene = units.groupby(keys+["port","acquisition"])[cols].mean()
    port = scene.groupby(keys+["port"])[cols].mean()
    return port.groupby(keys).mean().reset_index()


def event_diagnostics(frame):
    records = []
    for (pool, split, budget, origin), full in frame.groupby(["pool","split","budget","origin"]):
        g = full.loc[full.gain.ne(0)]
        for name in ("logistic","tree"):
            y = g.gain.eq(1)
            auc = roc_auc_score(y, g["score_"+name]) if y.nunique()==2 else np.nan
            records.append(dict(pool=pool,split=split,budget=budget,origin=origin,model=name,
                query_rows=len(full),query_ships=full.mmsi.nunique(),disagreements=int(full.disagree.sum()),
                rescue=int(y.sum()),harm=int((~y).sum()),event_ships=g.mmsi.nunique(),event_acquisitions=g.acquisition.nunique(),
                auc=auc,selected_rescue=int((g.gain.eq(1)&g["choose_"+name]).sum()),
                selected_harm=int((g.gain.eq(-1)&g["choose_"+name]).sum())))
    return records


def audit_partitions(source_folder, target_folder, source_tasks, target_tasks):
    sm = pd.read_csv(source_folder/"metadata.csv",dtype={"mmsi":str})
    tm = pd.read_csv(target_folder/"metadata.csv",dtype={"mmsi":str})
    def involved(meta,tasks):
        ids=sorted({i for t in tasks for i in t["q"]+t["s"]})
        return meta.iloc[ids]
    groups={k:involved(sm,source_tasks[k]) for k in ("fit","selection")}
    groups["target"]=involved(tm,target_tasks)
    for a,b in (("fit","selection"),("fit","target"),("selection","target")):
        assert not set(groups[a].mmsi)&set(groups[b].mmsi), (a,b,"identity overlap")
        assert not set(groups[a].acquisition)&set(groups[b].acquisition), (a,b,"acquisition overlap")
    return {name:dict(rows=len(g),ships=g.mmsi.nunique(),acquisitions=g.acquisition.nunique()) for name,g in groups.items()}


def main():
    start=time.perf_counter()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    frozen=json.loads((ROOT/"frozen.json").read_text(encoding="utf-8"))
    selections, partitions, coefficient_rows, all_source, models_by_pool = {}, {}, [], [], {}
    with threadpool_limits(limits=2):
        # Finish all source fits and persist decisions BEFORE computing target features/outcomes.
        for pool in frozen["pools"]:
            if not pool["selected"]:
                continue
            port=pool["port"]
            slug=port.replace(" ","_")
            folder=HERE/"body_context"/slug
            episodes=json.loads((folder/"episodes.json").read_text(encoding="utf-8"))
            tt=json.loads((ROOT/slug/"tasks.json").read_text(encoding="utf-8"))
            partitions[port]=audit_partitions(folder,ROOT/slug,episodes,tt)
            frames={}
            for split,key in (("fit","fit"),("selection","selection")):
                path=OUT/f"{slug}_{split}_features.csv"
                if path.exists():
                    frames[split]=pd.read_csv(path,dtype={"mmsi":str})
                else:
                    frames[split]=extract(folder,episodes[key],port,split)
                    frames[split].to_csv(path,index=False)
            models=fit_models(frames["fit"])
            models_by_pool[port]=models
            joblib.dump(models,OUT/f"{slug}_models.joblib")
            for name, model in models.items():
                if name=="tree":
                    (OUT/f"{slug}_tree.txt").write_text(export_text(model,feature_names=FEATURES,decimals=5),encoding="utf-8")
                else:
                    for feature, value in zip(FEATURES,model[-1].coef_[0]):
                        coefficient_rows.append(dict(pool=port,feature=feature,coefficient=value))
            frames={k:score_models(v,models) for k,v in frames.items()}
            sel=frames["selection"].loc[frames["selection"].origin.eq("cross")]
            weights=risk_weights(sel)
            choices={"body":0.}
            for name in ("logistic","tree"):
                choices[name]=float(np.sum(weights*sel.gain.to_numpy()*sel["choose_"+name].to_numpy())*100)
            selected=max(choices,key=choices.get)
            selections[port]=dict(chosen=selected,source_selection_delta_ba=choices,
                source_training_budget=16,threshold=.5,features=FEATURES)
            for split,frame in frames.items():
                frame["choose_source_selected"]=False if selected=="body" else frame["choose_"+selected]
                all_source.append(frame)
            events=frames["fit"].query("origin == 'cross' and gain != 0")
            print("SOURCE",port,"events",len(events),"ships",events.mmsi.nunique(),"selection",choices,"chosen",selected,flush=True)
        save_json(OUT/"SOURCE_FROZEN.json",selections)
        save_json(OUT/"partitions.json",partitions)
        pd.DataFrame(coefficient_rows).to_csv(OUT/"logistic_coefficients.csv",index=False)
        source=pd.concat(all_source,ignore_index=True)
        source.to_csv(OUT/"source_scored.csv",index=False)
        evaluate(source).to_csv(OUT/"source_metrics.csv",index=False)
        targets=[]
        replay=[]
        archived=pd.read_csv(ROOT/"ring_conditions/predictions.csv",dtype={"mmsi":str})
        keys=["port","acquisition","budget","origin","draw","sample_id"]
        for pool in frozen["pools"]:
            if not pool["selected"]:
                continue
            port=pool["port"]
            slug=port.replace(" ","_")
            folder=ROOT/slug
            tasks=json.loads((folder/"tasks.json").read_text(encoding="utf-8"))
            frame=extract(folder,tasks,port,"target")
            for name,arm in (("body","ship"),("mix","additive"),("half","body_half")):
                ref=archived.loc[archived.port.eq(port)&archived.arm.eq(arm),keys+[f"p{c}" for c in range(8)]]
                match=frame.merge(ref,on=keys,validate="one_to_one")
                assert len(match)==len(frame)
                actual=match[[f"{name}_p{c}" for c in range(8)]].to_numpy()
                expected=match[[f"p{c}" for c in range(8)]].to_numpy()
                np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
                replay.append(dict(pool=port,arm=arm,max_error=float(abs(actual-expected).max())))
            frame=score_models(frame,models_by_pool[port])
            selected=selections[port]["chosen"]
            frame["choose_source_selected"]=False if selected=="body" else frame["choose_"+selected]
            targets.append(frame)
            print("TARGET",port,"rows",len(frame),flush=True)
    target=pd.concat(targets,ignore_index=True)
    target.to_csv(OUT/"target_scored.csv",index=False)
    unit=evaluate(target)
    unit.to_csv(OUT/"target_metrics.csv",index=False)
    pools=aggregate(unit)
    pools.to_csv(OUT/"target_per_port.csv",index=False)
    cols=["oa","ba","macro_nll","macro_brier","switched"]
    summary=pools.groupby(["budget","origin","arm"])[cols].mean()
    summary.to_csv(OUT/"target_summary.csv")
    aggregate(unit,["draw"]).groupby(["budget","origin","arm","draw"])[cols].mean().to_csv(OUT/"target_by_draw.csv")
    source_metric=pd.read_csv(OUT/"source_metrics.csv")
    aggregate(source_metric).to_csv(OUT/"source_summary.csv",index=False)
    diagnostics=event_diagnostics(source)+event_diagnostics(target)
    pd.DataFrame(diagnostics).to_csv(OUT/"event_diagnostics.csv",index=False)
    target.loc[target.gain.ne(0)].to_csv(OUT/"target_rescue_harm.csv",index=False)
    save_json(OUT/"DONE.json",dict(seconds=time.perf_counter()-start,replay=replay,
        source_rows=len(source),target_rows=len(target),query_label_mutation_test="passed for every pool/split",
        source_only_decisions_frozen=True,training_seeds="deterministic probes; five support draws are not training seeds"))
    print(summary.to_string(),flush=True)


if __name__=="__main__":
    main()
