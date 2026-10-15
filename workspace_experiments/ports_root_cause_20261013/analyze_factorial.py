"""Paired factorial readout after all source/target predictions are frozen."""

from settings import *
from report import metrics

CELLS = {"OO":("original","original"),"OD":("original","display"),
    "DO":("display","original"),"DD":("display","display")}
PAIRS = {"target_only":("OD","OO"),"source_only":("DO","OO"),
    "source_coordination":("DD","OD"),"matched_net":("DD","OO"),"target_after_source":("DD","DO")}


def metadata(name, ids):
    base = "oss2" if name == "oss2_grdh" else name
    f = pilot.frame(base).set_index("sample_id").loc[ids].copy()
    identity = f["mmsi"] if name in PORTS else f["mmsi_or_ship_id"]
    f["identity"] = identity.astype(str).str.replace(r"\.0$","",regex=True)
    if name not in PORTS:
        details = pd.read_csv(OUT / "dataset_audit/public_metadata.csv",keep_default_na=False,
            usecols=["dataset","sample_id","noise_xml"])
        details = details[details.dataset == base].set_index("sample_id")
        f["noise_xml"] = details.loc[ids,"noise_xml"].astype(str).to_numpy()
    return f


def scopes(name, frame, saved, draw):
    ev = saved["eval_masks"][draw]
    values = dict(original=ev, common_query=np.all(saved["eval_masks"],axis=0))
    supportids = frame.identity.iloc[saved["support"][draw]]
    known = supportids[supportids.str.fullmatch(r"\d{9}")]
    values["query_unseen_support_vessel"] = ev & ~frame.identity.isin(known).to_numpy()
    if name not in PORTS:
        other = pilot.frame("oss2" if name == "oss1" else "oss1")
        values["no_other_release_vessel"] = ev & ~frame.identity.isin(other.mmsi_or_ship_id.astype(str)).to_numpy()
    if name == "oss2":
        for product in ("GRDH","SLC"):
            values["query_"+product] = ev & frame.scene_id.str.contains("_"+product+"_",regex=False).to_numpy()
    if name in ("oss2","oss2_grdh"):
        for value,label in (("0","clean"),("1","interference")):
            values["query_"+label] = ev & (frame.noise_xml == value).to_numpy()
    return {k:v for k,v in values.items() if v.any()}


def load(fold,name,source,condition):
    dst = OUT / "predictions" / source / condition / f"fold_{fold}" / name
    assert (dst / "COMPLETE.json").exists()
    return archive(dst / "predictions.npz"),np.load(dst / "probabilities.npy",mmap_mode="r")


def weighted_events(y,old,new):
    classes,counts = np.unique(y,return_counts=True)
    w = np.zeros(len(y),float)
    for c,n in zip(classes,counts):
        w[y == c] = 100./(len(classes)*n)
    correct_old,correct_new = old == y,new == y
    return w,~correct_old & correct_new,correct_old & ~correct_new


def main():
    assert (OUT / "PREDICTIONS_FROZEN.json").exists()
    assert (OUT / "REPLAY_COMPLETE.json").exists()
    rows,classes,flow,drift,adapt = [],[],[],[],[]
    max_identity_error = 0.
    for fold in FOLDS:
        for name in pilot.tasks(fold):
            caches = {cell:load(fold,name,*conditions) for cell,conditions in CELLS.items()}
            ref = caches["OO"][0]
            for saved,prob in caches.values():
                for k in ("ids","support","eval_masks"):
                    np.testing.assert_array_equal(saved[k],ref[k])
                np.testing.assert_allclose(prob.sum(-1,dtype=float),1.,atol=1e-7,rtol=0.)
                np.testing.assert_array_equal(prob[:,0,0],prob[:,1,0])
            yall = labels(name,ref["ids"])
            frame = metadata(name,ref["ids"])
            oldcache = {c:load(fold,name,"archive",c) for c in CONDITIONS}
            for draw in range(5):
                for scope,mask in scopes(name,frame,ref,draw).items():
                    y = yall[mask]
                    meta = dict(fold=fold,task=name,dataset="ports3" if name in PORTS else name,draw=draw,scope=scope)
                    for ai,arm in enumerate(ARMS):
                        for hi,head in enumerate(HEADS):
                            current = {}
                            predictions = {}
                            for cell,(saved,prob) in caches.items():
                                p = np.asarray(prob[draw,ai,hi,mask],float)
                                prediction = saved["predictions"][draw,ai,hi,mask]
                                predictions[cell] = prediction
                                value = metrics(y,prediction,p)
                                current[cell] = value
                                rows.append(dict(**meta,arm=arm,head=head,cell=cell,**value))
                                losses = np.sum((p-np.eye(8)[y])**2,axis=1)
                                for c in np.unique(y):
                                    ix = y == c
                                    classes.append(dict(**meta,arm=arm,head=head,cell=cell,class_id=int(c),
                                        n=int(ix.sum()),support_n=int(np.sum(yall[ref["support"][draw]] == c)),
                                        recall=100*np.mean(prediction[ix] == c),brier=losses[ix].mean()))
                            for contrast,(a,b) in PAIRS.items():
                                w,rescue,harm = weighted_events(y,predictions[b],predictions[a])
                                net = w[rescue].sum()-w[harm].sum()
                                error = abs(net-(current[a]["ba"]-current[b]["ba"]))
                                max_identity_error = max(max_identity_error,error)
                                assert error < 1e-10
                                baseline_correct = predictions["OO"] == y
                                target_correct = predictions["OD"] == y
                                lost = baseline_correct & ~target_correct
                                gained = ~baseline_correct & target_correct
                                flow.append(dict(**meta,arm=arm,head=head,contrast=contrast,
                                    rescue_ba=w[rescue].sum(),harm_ba=w[harm].sum(),delta_ba=net,
                                    delta_oa=current[a]["oa"]-current[b]["oa"],
                                    risk_gain_macro=current[b]["brier_macro"]-current[a]["brier_macro"],
                                    risk_gain_micro=current[b]["brier_micro"]-current[a]["brier_micro"],
                                    rescue_lost_ba=w[rescue & lost].sum(),rescue_existing_error_ba=w[rescue & ~baseline_correct].sum(),
                                    harmed_gained_ba=w[harm & gained].sum(),lost_mass_ba=w[lost].sum()))
                            if scope == "original":
                                for cell,condition in (("OO","original"),("OD","display")):
                                    old,op = oldcache[condition]
                                    oldmetric = metrics(y,old["predictions"][draw,ai,hi,mask],np.asarray(op[draw,ai,hi,mask],float))
                                    drift.append(dict(**meta,arm=arm,head=head,condition=condition,
                                        delta_ba=current[cell]["ba"]-oldmetric["ba"],
                                        delta_oa=current[cell]["oa"]-oldmetric["oa"],
                                        probability_max_error=float(np.max(np.abs(caches[cell][1][draw,ai,hi,mask]-op[draw,ai,hi,mask])))))
            print("READOUT",fold,name,flush=True)
    table = pd.DataFrame(rows)
    cls = pd.DataFrame(classes)
    effects = pd.DataFrame(flow)
    key = ["fold","task","dataset","draw","scope","arm","head"]
    check = table.merge(cls.groupby(key+["cell"])[["recall","brier"]].mean().reset_index(),on=key+["cell"],validate="one_to_one")
    np.testing.assert_allclose(check.ba,check.recall,atol=1e-10,rtol=0.)
    np.testing.assert_allclose(check.brier_macro,check.brier,atol=1e-10,rtol=0.)
    a = effects[effects.contrast == "source_coordination"]
    b = effects[effects.contrast == "source_only"]
    joined = a.merge(b,on=key,suffixes=("_ddod","_dooo"),validate="one_to_one")
    interaction = joined[key].copy()
    for metric in ("delta_ba","delta_oa","risk_gain_macro","risk_gain_micro"):
        interaction[metric] = joined[metric+"_ddod"]-joined[metric+"_dooo"]
    interaction["contrast"] = "interaction"
    for filename,value in (("metrics.csv",table),("classes.csv",cls),("effects.csv",effects),
        ("interaction.csv",interaction),("source_cache_drift.csv",pd.DataFrame(drift))):
        value.to_csv(OUT / filename,index=False)
    groups = ["dataset","scope","arm","head"]
    measures = ["oa","ba","brier_macro","brier_micro","nll_macro"]
    # Draws average within each task/fold before equal weighting of tasks.
    table.groupby(["fold","task"]+groups+["cell"])[measures].mean().reset_index().to_csv(OUT / "task_cells.csv",index=False)
    table.groupby(["fold","task"]+groups+["cell"])[measures].mean().groupby(groups+["cell"]).mean().reset_index().to_csv(OUT / "summary.csv",index=False)
    all_effects = pd.concat([effects,interaction],ignore_index=True)
    effectmeasures = ["delta_ba","delta_oa","risk_gain_macro","risk_gain_micro","rescue_ba","harm_ba","rescue_lost_ba","rescue_existing_error_ba","harmed_gained_ba","lost_mass_ba"]
    all_effects.groupby(["fold","task"]+groups+["contrast"])[effectmeasures].mean().reset_index().to_csv(OUT / "task_effects.csv",index=False)
    summary = all_effects.groupby(["fold","task"]+groups+["contrast"])[effectmeasures].mean().groupby(groups+["contrast"]).mean().reset_index()
    summary.to_csv(OUT / "effect_summary.csv",index=False)
    pd.DataFrame(drift).groupby(["dataset","arm","head","condition"])[["delta_ba","delta_oa","probability_max_error"]].mean().reset_index().to_csv(OUT / "source_cache_drift_summary.csv",index=False)
    # These are conditional stage effects, not additive independent causal shares.
    prior = table[table["head"] == "prior"]
    post = table[table["head"] != "prior"]
    stage = post.merge(prior,on=[k for k in key if k != "head"]+["cell"],suffixes=("_adapted","_prior"),validate="many_to_one")
    stage["delta_ba"] = stage.ba_adapted-stage.ba_prior
    stage["risk_gain_macro"] = stage.brier_macro_prior-stage.brier_macro_adapted
    stage.groupby(["dataset","scope","arm","head_adapted","cell"])[["delta_ba","risk_gain_macro"]].mean().reset_index().to_csv(OUT / "adaptation_stage.csv",index=False)
    write(OUT / "READOUT_COMPLETE.json",dict(metric_rows=len(table),class_rows=len(cls),effect_rows=len(effects),
        interaction_rows=len(interaction),max_repair_identity_error=max_identity_error,
        fixed_pairing=True,prior_r_equals_t=True,inference_frozen_before_metrics=True))
    print(summary[(summary.scope == "original") & (summary["head"] == "expert")][["dataset","arm","contrast","delta_ba"]].to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
