"""Paired scene/budget summaries with independent checks from saved predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent/"body_context_expansion"
OLD=ROOT.parent/"body_context"
METRICS=["oa","ba","macro_nll","macro_brier","macro_map_r"]
KEYS=["port","fold","budget","task","arm","seed","shuffle","origin","draw"]


def md(frame):
    frame=frame.copy()
    for c in frame.select_dtypes(include="number"):
        frame[c]=frame[c].map(lambda x:f"{x:.4f}")
    return "| "+" | ".join(frame.columns)+" |\n| "+" | ".join(["---"]*len(frame.columns))+" |\n"+"\n".join(
        "| "+" | ".join(map(str,row))+" |" for row in frame.itertuples(index=False,name=None))


def summarize(table):
    scene=table.groupby(["port","fold","budget","origin","arm"])[METRICS].mean()
    port=scene.groupby(["port","budget","origin","arm"]).mean()
    return scene,port,port.groupby(["budget","origin","arm"]).mean()


def main():
    frozen=json.loads((ROOT/"frozen.json").read_text(encoding="utf-8"))
    tables,predictions=[],[]
    for pool in frozen["pools"]:
        if not pool["selected"]:
            continue
        folder=ROOT/pool["port"].replace(" ","_")
        budgets=sorted({b for e in frozen["episodes"] if e["port"]==pool["port"] for b in e["budgets"]})
        for b in budgets:
            assert (folder/f"done_B{b}.json").exists(),(folder,b)
            tables.append(pd.read_csv(folder/f"metrics_B{b}.csv"))
            predictions.append(pd.read_csv(folder/f"predictions_B{b}.csv",dtype={"mmsi":str},
                usecols=KEYS+["sample_id","mmsi","truth","pred","margin","ap","nll","brier"]))
    table=pd.concat(tables,ignore_index=True)
    pred=pd.concat(predictions,ignore_index=True)
    pred["correct"]=pred.truth.eq(pred.pred)
    actual=pred.groupby(KEYS).correct.mean().mul(100).rename("oa_check").reset_index()
    ba=pred.groupby(KEYS+["truth"]).correct.mean().groupby(KEYS).mean().mul(100).rename("ba_check").reset_index()
    checked=table.merge(actual,on=KEYS,validate="one_to_one").merge(ba,on=KEYS,validate="one_to_one")
    np.testing.assert_allclose(checked.oa,checked.oa_check,atol=1e-10)
    np.testing.assert_allclose(checked.ba,checked.ba_check,atol=1e-10)
    real=table.loc[table.shuffle.eq(-1)]
    scene,port,summary=summarize(real)
    scene.to_csv(ROOT/"per_scene.csv")
    port.to_csv(ROOT/"per_port.csv")
    summary.to_csv(ROOT/"summary.csv")
    baseline=summary.xs("original",level="arm")
    delta=summary.reset_index().merge(baseline.reset_index(),on=["budget","origin"],suffixes=("","_original"))
    for metric in METRICS:
        delta["delta_"+metric]=delta[metric]-delta[metric+"_original"]
    delta.to_csv(ROOT/"effects_vs_original.csv",index=False)
    common={e["fold"] for e in frozen["episodes"] if 64 in e["budgets"]}
    _,_,common_summary=summarize(real.loc[real.fold.isin(common)])
    common_summary.to_csv(ROOT/"common_B64_cohort_summary.csv")
    by_seed=real.loc[real.seed.ge(0)].groupby(["seed","port","fold","budget","origin","arm"])[["oa","ba"]].mean()
    by_seed=by_seed.groupby(["seed","port","budget","origin","arm"]).mean().groupby(["seed","budget","origin","arm"]).mean()
    by_seed.to_csv(ROOT/"by_training_seed.csv")
    by_seed.groupby(["budget","origin","arm"]).agg(["mean","std"]).to_csv(ROOT/"training_seed_mean_std.csv")

    support_draw=real.groupby(["port","fold","budget","origin","arm","draw"])[["oa","ba"]].mean()
    support_draw=support_draw.groupby(["port","budget","origin","arm","draw"]).mean().groupby(["budget","origin","arm","draw"]).mean()
    support_draw.to_csv(ROOT/"by_support_draw.csv")
    common_draw=real.loc[real.fold.isin(common)].groupby(
        ["port","fold","budget","origin","arm","draw"])[["oa","ba"]].mean()
    common_draw=common_draw.groupby(["port","budget","origin","arm","draw"]).mean().groupby(
        ["budget","origin","arm","draw"]).mean().reset_index()
    common_delta=common_draw.merge(common_draw.loc[common_draw.arm.eq("original")].drop(columns="arm"),
        on=["budget","origin","draw"],suffixes=("","_original"))
    for metric in ("oa","ba"):
        common_delta["delta_"+metric]=common_delta[metric]-common_delta[metric+"_original"]
    common_delta.to_csv(ROOT/"common_cohort_support_draws.csv",index=False)
    common_variation=common_delta.groupby(["budget","origin","arm"])[["delta_oa","delta_ba"]].agg(["mean","std","min","max"])
    common_variation.to_csv(ROOT/"common_cohort_draw_variation.csv")

    base_keys=["port","fold","budget","origin","draw","sample_id"]
    realpred=pred.loc[pred.shuffle.eq(-1)].copy()
    pair_records=[]
    for reference in ("original","additive","static_600"):
        ref=realpred.loc[realpred.arm.eq(reference)].copy()
        join=base_keys.copy()
        if reference=="static_600":
            join+=["seed"]
        ref=ref[join+["pred","margin","ap"]]
        compare=realpred.merge(ref,on=join,validate="many_to_one",suffixes=("","_reference"))
        compare["rescued"]=compare.pred.eq(compare.truth)&compare.pred_reference.ne(compare.truth)
        compare["harmed"]=compare.pred.ne(compare.truth)&compare.pred_reference.eq(compare.truth)
        compare["reference"]=reference
        flips=compare.groupby(["reference","port","fold","budget","origin","arm","seed","truth"])[["rescued","harmed"]].sum()
        pair_records.append(flips.reset_index())
        changed=compare.loc[compare.rescued|compare.harmed]
        changed.to_csv(ROOT/f"changed_vs_{reference}.csv",index=False)
    pd.concat(pair_records).to_csv(ROOT/"class_flips.csv",index=False)

    # Previously used identities are not new-vessel evidence even in new acquisitions.
    initial_ids=set()
    for file in OLD.glob("*/metadata.csv"):
        meta=pd.read_csv(file,dtype={"mmsi":str})
        tasks=json.loads((file.parent/"episodes.json").read_text(encoding="utf-8"))["target"]
        positions=sorted({i for task in tasks for i in task["q"]+task["s"]})
        initial_ids.update(meta.iloc[positions].mmsi)
    fresh=realpred.loc[~realpred.mmsi.isin(initial_ids)]
    fresh_score=fresh.groupby(KEYS).correct.mean().mul(100).rename("oa").reset_index()
    fresh_ba=fresh.groupby(KEYS+["truth"]).correct.mean().groupby(KEYS).mean().mul(100).rename("ba").reset_index()
    fresh_score=fresh_score.merge(fresh_ba,on=KEYS)
    fresh_score.groupby(["port","fold","budget","origin","arm"])[["oa","ba"]].mean().groupby(
        ["port","budget","origin","arm"]).mean().groupby(["budget","origin","arm"]).mean().to_csv(ROOT/"initial_role_unseen_ships.csv")

    shuffled=table.loc[table.arm.eq("conditional_600")&table.budget.eq(16)]
    shuf=shuffled.groupby(["port","fold","origin","seed","shuffle"])[METRICS].mean()
    shuf.groupby(["origin","shuffle"]).mean().to_csv(ROOT/"gate_input_shuffle.csv")
    coverage=[]
    meta_all=pd.read_csv(ROOT/"required_samples.csv",dtype={"mmsi":str}).set_index("sample_id")
    composition=[]
    for e in frozen["episodes"]:
        for draw, support in enumerate(e["draws"]):
            for origin in ("same","cross"):
                for budget in e["budgets"]:
                    cnt=meta_all.loc[support[origin][:budget]].class_id.value_counts()
                    for cls,n in cnt.items():
                        composition.append(dict(port=e["port"],fold=e["fold"],draw=draw,origin=origin,
                            budget=budget,class_id=int(cls),count=int(n)))
    pd.DataFrame(composition).to_csv(ROOT/"support_class_counts.csv",index=False)
    for e in frozen["episodes"]:
        q=meta_all.loc[e["query"]]
        coverage.append(dict(port=e["port"],fold=e["fold"],query=len(q),ships=q.mmsi.nunique(),
            classes=q.class_id.nunique(),budgets=",".join(map(str,e["budgets"])),
            initial_role_seen_ships=e["query_ships_seen_in_initial"],
            initial_role_seen_acquisition=e["query_acquisition_seen_as_initial_support"]))
    coverage=pd.DataFrame(coverage)
    coverage.to_csv(ROOT/"coverage.csv",index=False)
    overall_q=meta_all.loc[[sid for e in frozen["episodes"] for sid in e["query"]]]
    expanded=summary.reset_index()
    b16=expanded.loc[expanded.budget.eq(16)].drop(columns="budget")
    rows=["# 船体与环带：新增采集扩围", "", "2026-10-15。第一阶段完成；未替换论文主模型。",
        "", "后续已补仅船体对照，见explainability/FINDINGS_ZH.md。共同10任务B64跨采集仅船体BA55.037高于本页组合53.805；本页旧结果保留，关于收益归属以补充消融为准，不能把组合相对原模型的增益全归为船体/环带互补。",
        "", "## 范围", "",
        f"冻结后评价20个港口-采集任务（5港各4个，对应17个不同采集）、{len(overall_q)}个查询观测、{overall_q.mmsi.nunique()}个不同MMSI。B16覆盖20任务，B32覆盖19任务，B64覆盖10任务。没有新增港口，也未启动OSS2/F_0921。不同任务可能共享采集，不能当作20个独立采集。",
        "",
        "这17个查询采集均未充当首轮六任务的查询或支持；117个查询观测（104个不同MMSI）的船只曾在首轮任一目标角色中出现。因此这是新增采集覆盖，不是全部新船或首次独立数据集。各模型源训练/模块学习/选择池与新增目标采集及MMSI均隔离；历史研发使用过的数据不称独立确认。",
        "", "## 结论先行", "",
        "1. 简单船体+环带组合值得作为有条件的候选保留。全覆盖跨采集BA增益在B16/B32/B64分别为+0.146/+0.214/+0.559个百分点，但覆盖集合不同。固定共同10任务、625个查询后，相应增益为-0.385/+0.586/+0.559个百分点。不是换一批容易样本后才变好，而是同一查询下对支持数量敏感；这还不能证明增加案例必然改善环境估计。",
        "2. 动态加权已经测试了实际训练600步的版本，不只是零初始化。全覆盖跨采集BA相比简单组合仅+0.006/+0.012/+0.000个百分点；B32静态600步取得同样OA/BA，B64二者均没有额外top-1收益。可观察的增益在简单组合中已经出现，不足以把动态调节作为已成立的核心贡献。",
        "3. 首轮环带单独补充的整体正增益未在新增B16覆盖上复现：跨采集OA/BA为-0.329/-0.103个百分点。不过宁波B16跨采集BA+1.200、青岛B64+2.400等局部收益仍保留，不因整体变弱而抹掉。",
        "4. 支持抽样并非次次获胜。B64简单组合跨采集BA在五次抽样中四次为正，范围-0.148至+1.139个百分点；五港平均中四港为正，上海为负。排除首轮出现过的船后，B64 BA增益为+0.904个百分点，但这是敏感性分析，不是另挑一个主结果。",
        "5. 所有结论仍是既有研发数据上的扩展证据。不同采集/港口/类别及共享船只的依赖不能靠重复抽样消除；本轮不宣称统计显著、因果校正或跨数据集成立。",
        "", "## 共同任务的预算对照", "",
        md(common_summary.reset_index().query("origin == 'cross' and arm in ['original','additive','static_600','conditional_600']")[["budget","arm","oa","ba"]]),
        "", "同一任务的B16支持是B32、B64的前缀；同采集与跨采集每类配额匹配。逐次抽样净增益见common_cohort_support_draws.csv，样本标准差见common_cohort_draw_variation.csv，不能把五次复用同一查询的抽样当五个独立测试集。支持逐类构成见support_class_counts.csv。",
        "",
        "共同10任务在三种预算下支持类别集合不变（按任务为3类或7类），因此收益变化不是补齐新的支持类别造成的。B32/B64各有4/5次抽样跨采集BA为正；净增益均值及抽样标准差分别为+0.586±0.790、+0.559±0.571个百分点。B64累计45次救回、38次误伤，涉及74个不同MMSI；这些是重复支持抽样下的改判事件，不能说救回45艘独立船。青岛B64收益来自类别0的3次救回，不能讲成该港各类均改善。Jebel Ali类别3有6次救回、16次误伤，说明BA净增仍伴随具体类别损失。",
        "",
        "最贴近当前证据的解释是：局部船体与环带表示参与支持案例匹配后，其价值依赖可用案例及类别；尚未证实环带识别了观测条件或动态权重纠正了物理干扰。B16真实门控输入比三次同采集置乱平均BA仅高约0.026个百分点，其中一次置乱还略高；不足以支持强条件对应机制。无需据此否认简单组合的预算范围内收益。",
        "", "## B16完整模型", "", md(b16),
        "", "先对支持抽样与训练种子取平均，再对港内场景平均，最后等港平均。每个模型始终输出八类，实际每任务查询类别数见coverage.csv。",
        "", "## 更多支持标签", "", md(expanded.loc[expanded.origin.eq("cross"),["budget","arm","oa","ba"]]),
        "",
        "上述预算覆盖集合不同，不能从跨行绝对准确率直接推断标签数量收益。共同10任务、同一查询、嵌套支持案例的预算对照见common_B64_cohort_summary.csv。所有门控仍按源端B16任务训练，未为B32/64另调参数。",
        "", "## 机制和变异", "",
        "源端选择版与固定600步版事先分别冻结；未按目标结果挑选检查点。种子0缺失末态重放40个源检查点、损失和原选择状态均零误差；种子1/2末态直接复用。固定600步的静态对照与动态版采用相同任务和优化设置。",
        "",
        "B16条件输入置乱仅改门控输入，独立环带分支保留；结果见gate_input_shuffle.csv。它检验体内/体外对应关系是否被有效使用，不直接证明散射物理校正。新增阶段没有搜索幅度、修改损失或重训骨干。",
        "",
        "训练种子和支持抽样变化分别保存，不能相加成独立船数。class_flips.csv及changed_vs_*.csv提供按真实类别、采集、船只的救回与误伤；多次抽样中的一次改判不是一艘新船。初始角色未见船子集另列，不用子集取代主结果。",
        "", "## 实现验证", "",
        "新浅层特征与16个旧样本重放相符。原始环带float32二阶矩存在约1e-7量级舍入差，两套结果均以独立float64像素均值/标准差校验；不是放宽物理或分类标准。PCA扩展时曾因矩阵批量形状出现2e-6量级漂移，已通过沿用旧变换批次纠正；旧坐标、先验、类别关系最大重放误差约1.6e-15。目标预测OA/BA从逐样本文件独立重算通过。",
        "",
        "每个done_B*.json保存缓存CPU读出耗时，不含新增原图编码与环带提取，不能冒充端到端成本。原始实验和主模型未覆盖。"]
    (ROOT/"FINDINGS_ZH.md").write_text("\n".join(rows)+"\n",encoding="utf-8")
    print("SUMMARY\n",summary.round(6).to_string())
    print("COMMON\n",common_summary[["oa","ba"]].round(6).to_string())
    print("checks",len(checked),"predictions",len(pred),"unique ships",overall_q.mmsi.nunique())


if __name__=="__main__":
    main()
