"""Retest unchanged source-only probes with matched support organization/budget."""

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.tree import export_text
from threadpoolctl import threadpool_limits

import probe_ring_predictability as old
from freeze_ring_matched import OUT


def status(frame):
    cross=frame.loc[frame.origin.eq("cross")]
    events=cross.loc[cross.gain.ne(0)]
    return dict(rows=len(cross),query_ships=int(cross.mmsi.nunique()),
        query_scenes=int(len(cross[["port","acquisition"]].drop_duplicates())),
        rescues=int(events.gain.eq(1).sum()),harms=int(events.gain.eq(-1).sum()),
        event_ships=int(events.mmsi.nunique()))


def main():
    started=time.perf_counter()
    torch.set_num_threads(2)
    frozen=json.loads((OUT/"frozen.json").read_text(encoding="utf-8"))
    template=pd.read_csv(old.OUT/"source_scored.csv",nrows=0)
    models,choices={},{}
    source_frames,coverage,coefs,source_units,diagnostics=[],[],[],[],[]
    with threadpool_limits(limits=2):
        for pool in frozen["pools"]:
            port=pool["pool"]
            slug=port.replace(" ","_")
            folder=old.HERE/"body_context"/slug
            choices[port]={}
            for org in ("single","mixed"):
                frames={}
                for split in ("fit","selection"):
                    tasks=pool["splits"][split][org]
                    path=OUT/f"{slug}_{org}_{split}_features.csv"
                    if path.exists():
                        frame=pd.read_csv(path,dtype={"mmsi":str})
                    elif tasks:
                        frame=old.extract(folder,tasks,port,split)
                        frame.to_csv(path,index=False)
                    else:
                        frame=template.copy()
                    frames[split]=frame
                choices[port][org]={}
                for budget in (16,32,64):
                    fit=frames["fit"].loc[frames["fit"].budget.eq(budget)].copy()
                    sel=frames["selection"].loc[frames["selection"].budget.eq(budget)].copy()
                    model=old.fit_models(fit) if len(fit) else {}
                    models[(port,org,budget)]=model
                    info={"pool":port,"organization":org,"budget":budget,"trained":bool(model),
                        "fit":status(fit),"selection":status(sel)}
                    for name,m in model.items():
                        if name=="tree":
                            (OUT/f"{slug}_{org}_B{budget}_tree.txt").write_text(
                                export_text(m,feature_names=old.FEATURES,decimals=5),encoding="utf-8")
                        else:
                            coefs.extend(dict(pool=port,organization=org,budget=budget,feature=f,coefficient=float(c))
                                for f,c in zip(old.FEATURES,m[-1].coef_[0]))
                    scored={k:old.score_models(v,model) if len(v) else v for k,v in (("fit",fit),("selection",sel))}
                    options={"body":0.}
                    if len(sel) and model:
                        validation=scored["selection"].loc[scored["selection"].origin.eq("cross")]
                        weights=old.risk_weights(validation)
                        for name in ("logistic","tree"):
                            options[name]=float(np.sum(weights*validation.gain.to_numpy()*validation["choose_"+name].to_numpy())*100)
                    selected=max(options,key=options.get)
                    info["selected"]=selected
                    info["selection_delta_ba"]=options
                    info["status"]=("trained_and_source_validated" if len(sel) and model else
                        "trained_without_matching_selection" if model else
                        "no_fit_tasks" if not len(fit) else "fit_events_missing_one_or_both_outcomes")
                    choices[port][org][str(budget)]=info
                    coverage.append(dict(pool=port,organization=org,budget=budget,status=info["status"],trained=bool(model),
                        **{"fit_"+k:v for k,v in info["fit"].items()},**{"selection_"+k:v for k,v in info["selection"].items()},selected=selected))
                    for split,frame in scored.items():
                        if not len(frame):
                            continue
                        frame["choose_source_selected"]=False if selected=="body" else frame["choose_"+selected]
                        frame["organization"]=org
                        frame["trained"]=bool(model)
                        source_frames.append(frame)
                        metric=old.evaluate(frame)
                        metric["organization"]=org
                        source_units.append(metric)
                        diagnostics.extend(dict(organization=org,trained=bool(model),**r) for r in old.event_diagnostics(frame))
                    print("SOURCE",port,org,budget,info["status"],info["fit"],"select",selected,flush=True)
        # Persist all learning and choices before the archived target scores are read.
        joblib.dump(models,OUT/"models.joblib")
        old.save_json(OUT/"SOURCE_FROZEN.json",choices)
        pd.DataFrame(coverage).to_csv(OUT/"learning_coverage.csv",index=False)
        pd.DataFrame(coefs).to_csv(OUT/"logistic_coefficients.csv",index=False)
        pd.concat(source_frames,ignore_index=True).to_csv(OUT/"source_scored.csv",index=False)
        pd.concat(source_units,ignore_index=True).to_csv(OUT/"source_metrics.csv",index=False)
        target=pd.read_csv(old.OUT/"target_scored.csv",dtype={"mmsi":str})
        metrics,decision_frames,availability=[],[],[]
        for recipe in ("single_budget","mixed_budget","single16_transfer","previous_mixed16"):
            for (port,budget),base in target.groupby(["pool","budget"],sort=True):
                if recipe=="previous_mixed16":
                    frame=base.copy()
                    available=True
                    trained_budget=16
                    selected="archived_source_selected"
                    status_value="previous_protocol"
                else:
                    org="mixed" if recipe=="mixed_budget" else "single"
                    trained_budget=16 if recipe=="single16_transfer" else int(budget)
                    model=models[(port,org,trained_budget)]
                    info=choices[port][org][str(trained_budget)]
                    frame=old.score_models(base,model)
                    selected=info["selected"]
                    frame["choose_source_selected"]=False if selected=="body" else frame["choose_"+selected]
                    available=bool(model)
                    status_value=info["status"]
                frame["recipe"]=recipe
                frame["trained"]=available
                frame["trained_budget"]=trained_budget
                metric=old.evaluate(frame)
                metric["recipe"],metric["trained"],metric["trained_budget"]=recipe,available,trained_budget
                metrics.append(metric)
                keep=old.UNIT+["sample_id","mmsi","truth","body_pred","mix_pred","gain","disagree",
                    "score_logistic","score_tree","choose_logistic","choose_tree","choose_source_selected",
                    "recipe","trained","trained_budget"]
                decision_frames.append(frame[keep])
                diagnostics.extend(dict(organization=recipe,trained=available,**r) for r in old.event_diagnostics(frame))
                availability.append(dict(recipe=recipe,pool=port,budget=int(budget),trained=available,
                    trained_budget=trained_budget,status=status_value,selected=selected))
        target_units=pd.concat(metrics,ignore_index=True)
        target_units.to_csv(OUT/"target_metrics.csv",index=False)
        pd.concat(decision_frames,ignore_index=True).to_csv(OUT/"target_decisions.csv",index=False)
        pd.DataFrame(availability).to_csv(OUT/"target_availability.csv",index=False)
        pd.DataFrame(diagnostics).to_csv(OUT/"event_diagnostics.csv",index=False)
        summaries=[]
        for recipe,group in target_units.groupby("recipe"):
            byport=old.aggregate(group)
            byport["recipe"]=recipe
            summaries.append(byport)
        byport=pd.concat(summaries,ignore_index=True)
        byport.to_csv(OUT/"target_per_port.csv",index=False)
        byport.groupby(["recipe","budget","origin","arm"])[["oa","ba","macro_nll","macro_brier","switched"]].mean().to_csv(OUT/"target_summary_with_fallback.csv")
        # Archived control endpoints must match, irrespective of the new source task recipe.
        original=pd.read_csv(old.OUT/"target_per_port.csv")
        baseline=byport.loc[byport.arm.isin(["body","mix","half"])]
        pair=baseline.merge(original,on=["pool","split","budget","origin","arm"],suffixes=("","_old"),validate="many_to_one")
        for col in ("oa","ba","macro_nll","macro_brier"):
            np.testing.assert_allclose(pair[col],pair[col+"_old"],atol=1e-10)
        # Reusing the old selector must also reproduce its full prior task metrics.
        replay=byport.loc[byport.recipe.eq("previous_mixed16")].merge(original,
            on=["pool","split","budget","origin","arm"],suffixes=("","_old"),validate="one_to_one")
        for col in ("oa","ba","macro_nll","macro_brier"):
            np.testing.assert_allclose(replay[col],replay[col+"_old"],atol=1e-10)
        old.save_json(OUT/"DONE.json",dict(seconds=time.perf_counter()-started,
            new_trained_models=sum(len(v) for v in models.values()),source_rows=sum(len(f) for f in source_frames),
            frozen_before_target=True,source_query_label_mutation="passed within extraction",
            old_target_and_selector_replay="passed",no_fit_or_single_outcome="body fallback explicitly marked, not learned",
            source_model_hyperparameters_unchanged=True))
        print(pd.DataFrame(coverage).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
