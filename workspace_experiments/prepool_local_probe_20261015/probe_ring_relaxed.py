"""Train unchanged probes on relaxed legal B32/B64 source tasks."""

import json
import time
import argparse

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.tree import export_text
from threadpoolctl import threadpool_limits

import probe_ring_predictability as old
from freeze_ring_relaxed import OUT as DEFAULT_OUT
from probe_ring_matched import status


def main(out=DEFAULT_OUT):
    OUT=out
    started=time.perf_counter()
    torch.set_num_threads(2)
    frozen=json.loads((OUT/"frozen.json").read_text(encoding="utf-8"))
    models,choices={},{}
    source_frames,source_metrics,diagnostics,coverage,coefs=[],[],[],[],[]
    with threadpool_limits(limits=2):
        for pool in frozen["pools"]:
            port=pool["pool"]
            slug=port.replace(" ","_")
            folder=old.HERE/"body_context"/slug
            frames={}
            for split in ("fit","selection"):
                path=OUT/f"{slug}_{split}_features.csv"
                if path.exists():
                    frames[split]=pd.read_csv(path,dtype={"mmsi":str})
                else:
                    frames[split]=old.extract(folder,pool["splits"][split],port,split)
                    frames[split].to_csv(path,index=False)
            choices[port]={}
            for b in pool["budgets"]:
                fit=frames["fit"].loc[frames["fit"].budget.eq(b)].copy()
                sel=frames["selection"].loc[frames["selection"].budget.eq(b)].copy()
                assert len(fit) and len(sel)
                w=old.risk_weights(fit)
                np.testing.assert_allclose(w.sum(),1.,atol=1e-12)
                model=old.fit_models(fit)
                models[(port,b)]=model
                scored={name:old.score_models(frame,model) for name,frame in (("fit",fit),("selection",sel))}
                options={"body":0.}
                if model:
                    validation=scored["selection"]
                    weights=old.risk_weights(validation)
                    np.testing.assert_allclose(weights.sum(),1.,atol=1e-12)
                    for name in ("logistic","tree"):
                        options[name]=float(100*np.sum(weights*validation.gain.to_numpy()*validation["choose_"+name].to_numpy()))
                selected=max(options,key=options.get)
                info=dict(trained=bool(model),selected=selected,selection_delta_ba=options,
                    fit=status(fit),selection=status(sel),threshold=.5,features=old.FEATURES,
                    status="fitted_with_matching_selection_tasks" if model else "insufficient_two_outcome_fit_events")
                choices[port][str(b)]=info
                coverage.append(dict(pool=port,budget=b,trained=bool(model),selected=selected,status=info["status"],
                    **{"fit_"+k:v for k,v in info["fit"].items()},**{"selection_"+k:v for k,v in info["selection"].items()}))
                for name,m in model.items():
                    if name=="tree":
                        (OUT/f"{slug}_B{b}_tree.txt").write_text(export_text(m,feature_names=old.FEATURES,decimals=5),encoding="utf-8")
                    else:
                        assert np.isfinite(m[0].scale_).all() and np.isfinite(m[-1].coef_).all()
                        coefs.extend(dict(pool=port,budget=b,feature=f,coefficient=float(c)) for f,c in zip(old.FEATURES,m[-1].coef_[0]))
                for split,frame in scored.items():
                    frame["choose_source_selected"]=False if selected=="body" else frame["choose_"+selected]
                    frame["trained"]=bool(model)
                    source_frames.append(frame)
                    source_metrics.append(old.evaluate(frame))
                    diagnostics.extend(dict(recipe="relaxed",trained=bool(model),**r) for r in old.event_diagnostics(frame))
                print("SOURCE",port,b,"trained",bool(model),info["fit"],"selection",options,"choose",selected,flush=True)
        # Complete every source-only decision before reading target caches.
        joblib.dump(models,OUT/"models.joblib")
        old.save_json(OUT/"SOURCE_FROZEN.json",choices)
        pd.DataFrame(coverage).to_csv(OUT/"learning_coverage.csv",index=False)
        pd.DataFrame(coefs).to_csv(OUT/"logistic_coefficients.csv",index=False)
        source=pd.concat(source_frames,ignore_index=True)
        source.to_csv(OUT/"source_scored.csv",index=False)
        pd.concat(source_metrics,ignore_index=True).to_csv(OUT/"source_metrics.csv",index=False)
        target=pd.read_csv(old.OUT/"target_scored.csv",dtype={"mmsi":str})
        strict=pd.read_csv(old.ROOT/"ring_matched/target_decisions.csv",dtype={"mmsi":str})
        strict=strict.loc[strict.recipe.eq("single_budget")]
        target_metrics,decisions,available=[],[],[]
        key=["pool","budget","port","acquisition","origin","draw","sample_id"]
        score_cols=["score_logistic","score_tree","choose_logistic","choose_tree","choose_source_selected"]
        for (port,b),model in models.items():
            base=target.loc[target.pool.eq(port)&target.budget.eq(b)].copy()
            for recipe in ("relaxed","previous_mixed16","strict_matched"):
                if recipe=="relaxed":
                    frame=old.score_models(base,model)
                    selected=choices[port][str(b)]["selected"]
                    frame["choose_source_selected"]=False if selected=="body" else frame["choose_"+selected]
                    trained=bool(model)
                elif recipe=="previous_mixed16":
                    frame=base.copy()
                    trained=True
                else:
                    prior=strict.loc[strict.pool.eq(port)&strict.budget.eq(b)]
                    frame=base.drop(columns=score_cols).merge(prior[key+score_cols+["trained"]],on=key,validate="one_to_one")
                    assert len(frame)==len(base) and frame.trained.nunique()==1
                    trained=bool(frame.trained.iloc[0])
                frame["recipe"],frame["trained"]=recipe,trained
                assert np.isfinite(frame[["score_logistic","score_tree"]].to_numpy()).all()
                metric=old.evaluate(frame)
                metric["recipe"],metric["trained"]=recipe,trained
                target_metrics.append(metric)
                keep=old.UNIT+["sample_id","mmsi","truth","body_pred","mix_pred","gain","disagree",*old.FEATURES,*score_cols,"recipe","trained"]
                decisions.append(frame[keep])
                available.append(dict(pool=port,budget=b,recipe=recipe,trained=trained))
                diagnostics.extend(dict(recipe=recipe,trained=trained,**r) for r in old.event_diagnostics(frame))
    metric=pd.concat(target_metrics,ignore_index=True)
    metric.to_csv(OUT/"target_metrics.csv",index=False)
    pd.concat(decisions,ignore_index=True).to_csv(OUT/"target_decisions.csv",index=False)
    pd.DataFrame(available).to_csv(OUT/"target_availability.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/"event_diagnostics.csv",index=False)
    byport=[]
    for recipe,g in metric.groupby("recipe"):
        a=old.aggregate(g)
        a["recipe"]=recipe
        byport.append(a)
    byport=pd.concat(byport,ignore_index=True)
    byport.to_csv(OUT/"target_per_port.csv",index=False)
    byport.groupby(["recipe","budget","origin","arm"])[["oa","ba","macro_nll","macro_brier","switched"]].mean().to_csv(OUT/"target_summary_with_fallback.csv")
    prior=pd.read_csv(old.OUT/"target_per_port.csv")
    replay=byport.loc[byport.recipe.eq("previous_mixed16")].merge(prior,on=["pool","split","budget","origin","arm"],suffixes=("","_old"),validate="one_to_one")
    for col in ("oa","ba","macro_nll","macro_brier"):
        np.testing.assert_allclose(replay[col],replay[col+"_old"],atol=1e-10)
    for recipe in ("relaxed","strict_matched"):
        ref=byport.loc[byport.recipe.eq(recipe)&byport.arm.isin(["body","mix","half"])].merge(
            prior,on=["pool","split","budget","origin","arm"],suffixes=("","_old"),validate="one_to_one")
        for col in ("oa","ba","macro_nll","macro_brier"):
            np.testing.assert_allclose(ref[col],ref[col+"_old"],atol=1e-10)
    old.save_json(OUT/"DONE.json",dict(seconds=time.perf_counter()-started,source_rows=len(source),
        actual_models=sum(len(m) for m in models.values()),target_pool_budgets=len(models),
        source_frozen_before_target=True,query_label_mutation="passed in extraction",
        weights_sum_one=True,old_target_replay=True,all_model_outputs_finite=True))
    print(pd.DataFrame(coverage).to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--query-coverage",action="store_true",help="Use the separately frozen coverage-only source tasks")
    args=parser.parse_args()
    main(DEFAULT_OUT/"query_coverage" if args.query_coverage else DEFAULT_OUT)
