"""Paired task/seed reporting, identity replay and class-wise gain accounting."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"ship_radar_geometry_design_20261016"
MET=["oa","ba","macro_nll","macro_brier"]
KEY=["port","fold","budget","origin","draw","sample_id","seed"]
P=[f"p{i}" for i in range(8)]


def markdown(frame, digits=5):
    text=["| "+" | ".join(map(str,frame.columns))+" |", "|"+"---|"*len(frame.columns)]
    for row in frame.itertuples(index=False,name=None):
        text.append("| "+" | ".join(f"{v:.{digits}f}" if isinstance(v,(float,np.floating)) else str(v) for v in row)+" |")
    return text


def main():
    metrics=pd.concat([pd.read_csv(p) for p in (HERE/"evaluation").glob("*/*_metrics.csv")],ignore_index=True)
    pred=pd.concat([pd.read_csv(p,dtype={"mmsi":str}) for p in (HERE/"evaluation").glob("*/*_predictions.csv")],ignore_index=True)
    metrics.to_csv(HERE/"metrics.csv",index=False)
    pred.to_csv(HERE/"predictions.csv",index=False)
    counts=metrics.loc[metrics.scope.eq("full")].groupby(["arm","seed"]).size()
    assert counts.nunique()==1 and len(counts)==18
    assert not pred.duplicated(KEY+["arm"]).any()
    baseline=pred.loc[pred.arm.eq("B")]
    history=pd.concat([pd.read_csv(p) for p in (PREV/"evaluation").glob("*/B_seed*_predictions.csv")],ignore_index=True)
    history=history.loc[history.budget.eq(32)]
    replay=baseline.merge(history[KEY+P],on=KEY,suffixes=("","_old"),validate="one_to_one")
    assert len(replay)==len(baseline)==len(history)
    error=float(np.max(np.abs(replay[P].to_numpy()-replay[[c+"_old" for c in P]].to_numpy())))
    assert error<1e-10,error
    perport=metrics.groupby(["scope","origin","arm","seed","port"])[MET].mean().reset_index()
    seedmeans=perport.groupby(["scope","origin","arm","seed"])[MET].mean().reset_index()
    perport.to_csv(HERE/"per_port.csv",index=False)
    seedmeans.to_csv(HERE/"seed_means.csv",index=False)
    fullsummary=seedmeans.groupby(["scope","origin","arm"])[MET].agg(["mean","std"])
    fullsummary.to_csv(HERE/"summary.csv")
    contrasts=[]
    for arm,ref in [("spatial","B"),("conditional","B"),("geometry","B"),("conditional","spatial"),("geometry","conditional"),("geometry","spatial"),("geometry_ship45","geometry"),("geometry_radar45","geometry")]:
        a=seedmeans.loc[seedmeans.arm.eq(arm)]
        b=seedmeans.loc[seedmeans.arm.eq(ref)]
        m=a.merge(b,on=["scope","origin","seed"],suffixes=("","_ref"),validate="one_to_one")
        for col in MET:
            m["delta_"+col]=m[col]-m[col+"_ref"]
        m["reference"]=ref
        contrasts.append(m[["scope","origin","seed","arm","reference"]+["delta_"+k for k in MET]])
    contrasts=pd.concat(contrasts,ignore_index=True)
    contrasts.to_csv(HERE/"paired_contrasts.csv",index=False)
    events=[]
    for arm in ["spatial","conditional","geometry","geometry_ship45","geometry_radar45"]:
        a=pred.loc[pred.arm.eq(arm)]
        m=a.merge(baseline[KEY+["correct","pred","kernel_margin","ap",*P]],on=KEY,suffixes=("","_B"),validate="one_to_one")
        m["rescue"]=m.correct & ~m.correct_B
        m["harm"]=~m.correct & m.correct_B
        m["delta_correct"]=m.correct.astype(int)-m.correct_B.astype(int)
        disabled=~m.enabled
        np.testing.assert_array_equal(m.loc[disabled,P],m.loc[disabled,[k+"_B" for k in P]])
        events.append(m)
    events=pd.concat(events,ignore_index=True)
    events.to_csv(HERE/"paired_events.csv",index=False)
    events.loc[events.rescue|events.harm].to_csv(HERE/"changed_cases.csv",index=False)
    # Contributions use full task class denominators, including unchanged classes.
    unit=["arm","seed","port","origin","fold","draw"]
    pieces=[]
    for key,g in events.groupby(unit,sort=False):
        nclass=g.truth.nunique()
        for cls,c in g.groupby("truth"):
            pieces.append(dict(zip(unit,key),truth=int(cls),delta_ba_contribution=100*c.delta_correct.mean()/nclass,
                               delta_oa_contribution=100*c.delta_correct.sum()/len(g),rescues=int(c.rescue.sum()),harms=int(c.harm.sum()),
                               query_ships=c.mmsi.nunique(),n=len(c)))
    contributions=pd.DataFrame(pieces)
    contributions.to_csv(HERE/"class_contributions_by_task.csv",index=False)
    clsport=contributions.groupby(["arm","seed","port","origin","truth"])[["delta_ba_contribution","delta_oa_contribution"]].sum().reset_index()
    denom=events.groupby(["arm","seed","port","origin"])["fold"].nunique()*5
    for i,row in clsport.iterrows():
        n=denom.loc[(row.arm,row.seed,row.port,row.origin)]
        clsport.loc[i,["delta_ba_contribution","delta_oa_contribution"]]/=n
    clsport.to_csv(HERE/"class_contributions_per_port.csv",index=False)
    expected=perport.loc[perport.scope.eq("full") & perport.arm.ne("B")].merge(perport.loc[perport.scope.eq("full") & perport.arm.eq("B")],on=["scope","origin","seed","port"],suffixes=("","_B"))
    sums=clsport.groupby(["arm","seed","port","origin"])["delta_ba_contribution"].sum()
    for row in expected.itertuples():
        assert abs(sums.loc[(row.arm,row.seed,row.port,row.origin)]-(row.ba-row.ba_B))<1e-9
    draws=metrics.groupby(["scope","origin","arm","seed","draw","port"])[MET].mean().groupby(["scope","origin","arm","seed","draw"]).mean().reset_index()
    draws.to_csv(HERE/"support_draw_means.csv",index=False)
    retrieval=[]
    for scope,f in (("full",pred),("geometry",pred.loc[pred.enabled])):
        x=f.groupby(["arm","seed","port","origin","fold","draw","truth"])[["ap","kernel_margin"]].mean()
        x=x.groupby(["arm","seed","port","origin","fold","draw"]).mean().groupby(["arm","seed","port","origin"]).mean().groupby(["arm","seed","origin"]).mean().reset_index()
        x["scope"]=scope
        retrieval.append(x)
    pd.concat(retrieval).to_csv(HERE/"retrieval.csv",index=False)
    training=[]
    for file in (HERE/"trained").glob("*/*_seed*.pt"):
        saved=torch.load(file,weights_only=False,map_location="cpu")
        row=dict(pool=file.parent.name,arm=file.stem.split("_seed")[0],seed=int(file.stem.split("_seed")[1]),unavailable=saved.get("unavailable",False))
        for k in ("steps","seconds","parameters","first_gradient","baseline_step"):
            row[k]=saved.get(k)
        row.update(saved["inventory"])
        training.append(row)
    training=pd.DataFrame(training)
    training.to_csv(HERE/"training_summary.csv",index=False)
    coverage=baseline.loc[baseline.seed.eq(0)].groupby(["port","origin"]).agg(query_events=("sample_id","size"),query_samples=("sample_id","nunique"),query_ships=("mmsi","nunique"),enabled_events=("enabled","sum"))
    coverage.to_csv(HERE/"target_coverage.csv")
    lines=["# 几何组织局部关系：固定200步结果", "", "2026-10-16。S32，3训练种子、5支持抽样；历史研发队列，不是独立确认。", "", "## 完整队列跨采集", ""]
    table=seedmeans.loc[seedmeans.scope.eq("full") & seedmeans.origin.eq("cross")].groupby("arm")[MET].mean().reset_index()
    lines+=markdown(table)
    lines += ["", "## 几何有效子集跨采集", ""]
    lines+=markdown(seedmeans.loc[seedmeans.scope.eq("geometry") & seedmeans.origin.eq("cross")].groupby("arm")[MET].mean().reset_index())
    lines += ["", "## 配对增量", "", "OA/BA为百分点。以下先按每训练种子配对，再取均值。", ""]
    lines+=markdown(contrasts.loc[contrasts.origin.eq("cross")].groupby(["scope","arm","reference"])[["delta_oa","delta_ba","delta_macro_nll","delta_macro_brier"]].mean().reset_index())
    lines += ["", "## 覆盖与解释限制", "", f"- {int((~training.unavailable).sum())}个实际训练模型；{int(training.unavailable.sum())}个无合法训练任务的回退记录，不称已训练。",
              "- 上海源fit没有同时满足有效支持和查询的任务；不借selection、不改变划分。",
              "- 完整队列保留缺失字段查询，其预测精确回退旧B。几何子集仅含实际可启用查询，不能冒充全港证据。",
              "- 固定200步末模型；源曲线用于诊断，不依据目标效果选步数或幅度。",
              "- spatial/conditional/geometry参数量相同；spatial条件输入为零，因此有效功能容量并非完全相同。",
              "- 角度扰动是固定权重依赖性诊断，不是物理干预；所有支撑核均重新求解。",
              "- 常量新增描述经支持中心化贡献为零；不代表所有残余信息都具有物理因果意义。",
              "", "## 验证", "", f"旧B全目标S32逐行概率回放最大误差{error:.3g}；无效查询精确回退；逐类贡献和完整BA增量核对通过。",
              "原始表见metrics.csv、paired_contrasts.csv、paired_events.csv、training_summary.csv；图像与支持分析另见CASE_ANALYSIS_ZH.md。"]
    (HERE/"RESULT_TABLES_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    result=dict(target_tasks_per_arm_seed=int(counts.iloc[0]),prediction_events=len(pred),trained_models=int((~training.unavailable).sum()),fallback_records=int(training.unavailable.sum()),baseline_replay_max=error,disabled_queries_exact=True,class_gain_accounting=True)
    (HERE/"RESULT_CHECK.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(result))


if __name__=="__main__":
    main()
