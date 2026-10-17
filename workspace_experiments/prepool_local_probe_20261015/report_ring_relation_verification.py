"""Report fixed feature-group controls and paired sample-level consequences."""

import json

import joblib
import numpy as np
import pandas as pd

import probe_ring_predictability as old
from report_ring_predictability import markdown
from verify_ring_relations import OUT, VARIANTS, features


def main():
    ports = pd.read_csv(OUT / "target_per_port.csv")
    decisions = pd.read_csv(OUT / "target_decisions.csv", dtype={"mmsi": str}, low_memory=False)
    draws = pd.read_csv(OUT / "target_per_draw.csv")
    diagnostics = pd.read_csv(OUT / "target_diagnostics.csv")
    coverage = pd.read_csv(OUT / "model_coverage.csv")
    tables = []
    for (protocol, pool, budget, origin), g in ports.groupby(["protocol", "pool", "budget", "origin"]):
        a = g.set_index("arm")
        row = dict(protocol=protocol, pool=pool, budget=budget, origin=origin)
        for name in (*VARIANTS, "mix", "half"):
            row[name + "_ba"] = a.loc[name, "ba"] - a.loc["body", "ba"]
            row[name + "_oa"] = a.loc[name, "oa"] - a.loc["body", "oa"]
        for name in VARIANTS:
            row[name + "_vs_random"] = a.loc[name, "ba"] - a.loc[name + "_random", "ba"]
            row[name + "_source_enabled_ba"] = (a.loc[name + "_enabled", "ba"] - a.loc["body", "ba"]
                                                  if name + "_enabled" in a.index else np.nan)
        row["full_vs_confidence5"] = a.loc["full18", "ba"] - a.loc["confidence5", "ba"]
        row["full_vs_poly18"] = a.loc["full18", "ba"] - a.loc["confidence_poly18", "ba"]
        tables.append(row)
    result = pd.DataFrame(tables)
    result.to_csv(OUT / "comparison.csv", index=False)
    cols = ["protocol", "pool", "budget", *[v + "_ba" for v in VARIANTS], "full_vs_confidence5", "full_vs_poly18"]
    cross = result.loc[result.origin.eq("cross")]

    focus = decisions.loc[decisions.protocol.eq("matched16") & decisions.pool.eq("Jebel Ali") &
                          decisions.budget.eq(16) & decisions.origin.eq("cross")].copy()
    keys = old.UNIT + ["sample_id", "mmsi", "truth", "body_pred", "mix_pred", "gain", "disagree"]
    rows = focus.pivot(index=keys, columns="variant", values=["score", "chosen", "ba_contribution"])
    rows.columns = [f"{a}_{b}" for a, b in rows.columns]
    rows = rows.reset_index()
    for v in VARIANTS:
        rows["chosen_" + v] = rows["chosen_" + v].astype(bool)
    rows["extra_ba_vs_conf"] = rows.ba_contribution_full18 - rows.ba_contribution_confidence5
    rows["extra_ba_vs_poly"] = rows.ba_contribution_full18 - rows.ba_contribution_confidence_poly18
    extra = rows.loc[rows.gain.ne(0) & (rows.extra_ba_vs_conf.ne(0) | rows.extra_ba_vs_poly.ne(0))].copy()
    extra.to_csv(OUT / "jebel16_changed_useful_events.csv", index=False)
    ships = rows.groupby(["mmsi", "truth"]).agg(extra_ba_vs_conf=("extra_ba_vs_conf", "sum"),
        extra_ba_vs_poly=("extra_ba_vs_poly", "sum"), query_samples=("sample_id", "nunique")).reset_index()
    ships = ships.loc[ships.extra_ba_vs_conf.ne(0) | ships.extra_ba_vs_poly.ne(0)]
    ships.to_csv(OUT / "jebel16_extra_ship_contributions.csv", index=False)
    best_ship = ships.loc[ships.extra_ba_vs_poly.astype(float).idxmax()]
    remaining_contribution = float(ships.extra_ba_vs_poly.sum() - best_ship.extra_ba_vs_poly)

    # Exact additive logit accounting is descriptive, not a causal attribution.
    raw = pd.read_csv(old.OUT / "target_scored.csv", dtype={"mmsi": str})
    raw = raw.loc[raw.pool.eq("Jebel Ali") & raw.budget.eq(16) & raw.origin.eq("cross")].copy()
    model = joblib.load(OUT / "models.joblib")[("matched16", "Jebel Ali", 16, "full18")]
    z = model[0].transform(features(raw, "full18"))
    contributions = z * model[-1].coef_[0]
    raw["confidence_logit"] = contributions[:, :5].sum(axis=1)
    raw["relation_logit"] = contributions[:, 5:12].sum(axis=1)
    raw["coverage_logit"] = contributions[:, 12:15].sum(axis=1)
    raw["stability_logit"] = contributions[:, 15:].sum(axis=1)
    raw["intercept"] = model[-1].intercept_[0]
    groups = ["confidence_logit", "relation_logit", "coverage_logit", "stability_logit", "intercept"]
    total = raw[groups].sum(axis=1).to_numpy()
    np.testing.assert_allclose(total, model.decision_function(features(raw, "full18")), atol=1e-12)
    score = model.predict_proba(features(raw, "full18"))[:, 1]
    np.testing.assert_allclose(1 / (1 + np.exp(-total)), score, atol=1e-12)
    join = ["pool", "budget", "port", "acquisition", "origin", "draw", "sample_id"]
    details = extra.merge(raw[join + old.FEATURES + groups], on=join, validate="one_to_one")
    details.to_csv(OUT / "jebel16_changed_event_feature_accounting.csv", index=False)

    d = draws.loc[draws.protocol.eq("matched16") & draws.pool.eq("Jebel Ali") &
                  draws.budget.eq(16) & draws.origin.eq("cross")].pivot(index="draw", columns="arm", values="ba")
    d = d.subtract(d.body, axis=0)
    d["full_vs_conf"] = d.full18 - d.confidence5
    d["full_vs_poly"] = d.full18 - d.confidence_poly18
    d = d.reset_index()
    d.to_csv(OUT / "jebel16_draw_comparison.csv", index=False)
    nscenes = rows.acquisition.nunique()
    scene = rows.groupby("acquisition")[["extra_ba_vs_conf", "extra_ba_vs_poly"]].sum().reset_index()
    scene["scene_delta_vs_conf"] = nscenes * scene.extra_ba_vs_conf
    scene["scene_delta_vs_poly"] = nscenes * scene.extra_ba_vs_poly
    scene["remaining_mean_vs_poly"] = (scene.scene_delta_vs_poly.sum() - scene.scene_delta_vs_poly) / (nscenes - 1)
    scene.to_csv(OUT / "jebel16_scene_comparison.csv", index=False)
    focus_result = cross.loc[cross.protocol.eq("matched16") & cross.pool.eq("Jebel Ali")].iloc[0]
    np.testing.assert_allclose(rows.extra_ba_vs_conf.sum(), focus_result.full_vs_confidence5, atol=1e-10)
    np.testing.assert_allclose(rows.extra_ba_vs_poly.sum(), focus_result.full_vs_poly18, atol=1e-10)
    diag = diagnostics.loc[diagnostics.protocol.eq("matched16") & diagnostics.pool.eq("Jebel Ali") &
                           diagnostics.budget.eq(16) & diagnostics.origin.eq("cross")]
    diag = diag[["variant", "selected_rescue", "selected_harm", "auc", "weighted_auc", "delta_ba"]]
    enabled_cols = ["protocol", "pool", "budget", *[v + "_source_enabled_ba" for v in VARIANTS]]
    done = json.loads((OUT / "DONE.json").read_text(encoding="utf-8"))
    old.save_json(OUT / "REPORT_CHECKS.json", dict(weighted_event_to_metric_replay=True,
        additive_logit_accounting=True, fixed_focus="matched16/Jebel Ali/B16/cross",
        independent_training_seeds=False))
    text = f"""# 查询支持关系是否超出置信度：冻结对照验证

## 结论

**有局部增量证据，可以保留，但准确贡献应写成“支持关系、覆盖与稳定性联合描述的选择”，不能写成单独增加相似度就有用。**

在原Jebel Ali、16支持候选上，full18相对body BA +{focus_result.full18_ba:.4f}个百分点；confidence5 +{focus_result.confidence5_ba:.4f}；同参数数量confidence_poly18 +{focus_result.confidence_poly18_ba:.4f}。完整描述额外超过两者{focus_result.full_vs_confidence5:+.4f}/{focus_result.full_vs_poly18:+.4f}个百分点。这个对照支持局部选择收益不完全由置信度或系数数量解释，不等于排除所有更强置信度映射。

但confidence_relation12只有+{focus_result.confidence_relation12_ba:.4f}，低于confidence5。完整18维比12维多覆盖与LOO稳定性6维，收益是新增这一组并重新拟合后的联合效果；尚不能单独认定覆盖、稳定性谁是原因。

扩大任务并未显示普遍优势。B32宁波/青岛固定置信度探针优于关系版；B64关系版减少置信度探针的误伤，但源端启用判定后均回退body。所有结果保留，不用目标最优变体替换部署策略。

## 设计

源数据、事件标签、样本权重、标准化拟合范围、正则C=.1、优化器、阈值.5及两个分类端点均固定。比较四种输入：5维置信度、18维置信度多项式、置信度加直接关系12维、原完整18维。18维多项式由预先固定的平方/交互项构成，不看目标标签选择，参数数与full18相同，但函数空间并不相同。

置信度本身来自支持适配后的预测，已经隐式受到支持集影响。本轮检验的是“在已有预测摘要之外，显式关系描述还有没有增量”，不是“模型此前完全没有利用支持信息”。

strict single B16所有可训练源池3个，以及最新expanded B32/B64七组，共10个源池/预算条件、40个确定性逻辑回归模型。保留旧协议与扩展协议分组，不把它们当成同一总体。先拟合并保存所有源端策略，再读目标缓存。B16宁波和新加坡无双结局拟合条件，未补造。

本轮为缓存层小模型对照，耗时{done['seconds']:.1f}秒，不含图像编码或端到端成本。full18源/目标预测与原记录最大误差{done['full18_replay_max_error']:.3g}，旧正信号成功重放。

## 固定探针的全部跨采集BA结果

各方法列相对body，最后两列是完整关系版相对置信度对照，单位百分点。全是同一目标任务上的配对比较，不是分别挑各自有利数据。相同模型的OA、same任务结果及随机对照见comparison.csv。

{markdown(cross[cols])}

## Jebel B16：具体改变了什么

{markdown(diag)}

完整关系版比confidence5少5次误伤，但少救回1次；比poly18少4次误伤，少救回1次。这是净增益的直接预测层来源，不代表已经知道具体海况或散射机制。事件含支持draw重复，不能等同独立船只。

相比只加直接关系的12维版，full18多保留2次救回、少2次误伤。说明“直接类别相似度”与“它是否受当前支持组成影响”应联合讨论，但还没有拆出两个子组的独立效应。

原始未加权事件AUC上升，而按任务BA权重计算的AUC未同步上升，见上表。不能将阈值附近净收益等同于所有事件上的排序都改善。

发生净效用差异的船只如下，贡献按原港/采集/任务/类权重计入总BA，而不是该船自己的准确率。代码同时保存全部查询，没有删掉无差异或不利样本。

{markdown(ships)}

最大正贡献船MMSI {best_ship.mmsi}占相对poly18增量{float(best_ship.extra_ba_vs_poly):+.4f}个百分点，超过净增量。按原权重将其额外贡献置零，其余贡献和为{remaining_contribution:+.4f}。这是贡献集中度诊断，不是删除样本重新归一化后的BA，也不否定原任务上的实际净收益；它说明还不能把增益说成分散覆盖大量独立船只。

每条变化事件的原始18特征、四组对logit的加性贡献及截距见jebel16_changed_event_feature_accounting.csv。加性分解与实际决策函数一致；这是模型内部记账，不是因果重要性。相关特征的系数不能直接解释为物理机制。

例：MMSI572345220在draw0中，confidence5选择分数0.664，full18降为0.446，因而不切换，避免一次误伤；完整模型中直接关系块对logit贡献-0.471，覆盖块-0.063，稳定性块-0.186，置信度块+0.160、截距+0.343。相反MMSI210530000在draw1中，本可被环带救回，却被full18以0.051分数拒绝，其直接关系块贡献-4.189；同一逻辑也可能错拦真正补证。两例都保留，不只展示正确拒绝。

## 支持抽样和采集范围

{markdown(d[['draw', *VARIANTS, 'full_vs_conf', 'full_vs_poly']])}

full18较body五次draw全正；但相对两个置信度对照均是2次赢、1次输、2次平，平均增益并非每次抽样都成立。五次draw共用查询，不是五个独立训练种子，不据此套独立样本t检验。

{markdown(scene)}

这是4个目标采集上的描述性分解，不能当大量独立目标港。remaining_mean_vs_poly为去掉该采集后其余采集的平均增量，可显示范围敏感性，不用于按目标标签筛选部署范围。

## 源端决定是否启用后的结果

每种特征版本独立按source selection的BA决定启用或body回退，没有在四种版本之间用目标效果挑赢家。NA表示没有匹配源selection，不代表零收益。

固定探针覆盖10个条件，启用策略只覆盖9个有selection的条件；不把覆盖不同的均值直接作差。

{markdown(cross[enabled_cols])}

特别是Jebel B16没有合格独立源selection，因此主正信号仍是冻结超参数探针，不是已确认的部署规则。Shanghai B16只有1个源selection救回/误伤事件，不能把源选择稳定性说得很强。

## 能写的贡献与不能跳过的边界

可写：在部分跨采集SAR少样本任务中，联合查询预测状态、类别支持关系、支持覆盖及局部关系稳定性，可以比仅凭置信度更有效地减少环带误伤，同时保留多数救回。

不能写：直接相似度本身普遍提升；每一组特征都已单独验证必要；通用可靠跨港选择；物理因果因素已识别；超过任何置信度方法；节省双路计算。

本轮已完成用户要求的最小关系价值验证，无需为了扩大正结果继续调阈值。保留Jebel B16局部候选与完整负结果，将本次对照补入组件说明。历史开发目标重复使用的限制仍在，后续论文确认应冻结该设计后另行安排，不把本轮称新独立测试。
"""
    (OUT / "FINDINGS_ZH.md").write_text(text, encoding="utf-8")
    print(cross[cols].to_string(index=False))
    print("FOCUS SHIPS\n", ships.to_string(index=False))
    print("FOCUS SCENES\n", scene.to_string(index=False))


if __name__ == "__main__":
    main()
