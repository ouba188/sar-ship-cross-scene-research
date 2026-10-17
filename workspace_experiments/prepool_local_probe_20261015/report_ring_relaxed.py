"""Compare relaxed source coverage without promoting fallback or repeated events."""

import json

import numpy as np
import pandas as pd

from freeze_ring_relaxed import OUT
from probe_ring_predictability import FEATURES, aggregate, risk_weights, save_json
from report_ring_predictability import markdown


def main():
    results, covers, draws, details, mechanisms = [], [], [], [], []
    for stage, folder in (("query8", OUT), ("expanded", OUT / "query_coverage")):
        coverage = pd.read_csv(folder / "learning_coverage.csv")
        coverage["stage"] = stage
        covers.append(coverage)
        ports = pd.read_csv(folder / "target_per_port.csv")
        ports = ports.loc[ports.recipe.eq("relaxed")]
        for (pool, budget, origin), g in ports.groupby(["pool", "budget", "origin"]):
            a = g.set_index("arm")
            c = coverage.loc[coverage.pool.eq(pool) & coverage.budget.eq(budget)].iloc[0]
            row = dict(stage=stage, pool=pool, budget=budget, origin=origin,
                       trained=bool(c.trained), selected=c.selected,
                       body_oa=a.loc["body", "oa"], body_ba=a.loc["body", "ba"])
            for arm in ("mix", "half", "logistic", "tree", "source_selected"):
                for metric in ("oa", "ba", "macro_brier", "macro_nll"):
                    value = a.loc[arm, metric] - a.loc["body", metric]
                    if not c.trained and arm in ("logistic", "tree"):
                        value = np.nan
                    row[f"{arm}_{metric}_delta"] = value
            for arm in ("logistic", "tree", "source_selected"):
                row[f"{arm}_vs_random_ba"] = a.loc[arm, "ba"] - a.loc[arm + "_random", "ba"]
                if not c.trained and arm != "source_selected":
                    row[f"{arm}_vs_random_ba"] = np.nan
            results.append(row)
        metrics = pd.read_csv(folder / "target_metrics.csv")
        metrics = metrics.loc[metrics.recipe.eq("relaxed")]
        d = aggregate(metrics, ["draw"]).pivot(
            index=["pool", "budget", "origin", "draw"], columns="arm", values="ba")
        d = d.subtract(d.body, axis=0).reset_index()
        d["stage"] = stage
        draws.append(d)
        decisions = pd.read_csv(folder / "target_decisions.csv", dtype={"mmsi": str})
        decisions = decisions.loc[decisions.recipe.eq("relaxed")]
        for (pool, budget, origin), g in decisions.groupby(["pool", "budget", "origin"]):
            w = risk_weights(g)
            contrib = 100 * w * g.gain * g.choose_source_selected
            c = coverage.loc[coverage.pool.eq(pool) & coverage.budget.eq(budget)].iloc[0]
            u = g.loc[g.disagree].groupby("task").choose_source_selected.agg(["sum", "count"])
            mechanisms.append(dict(stage=stage, pool=pool, budget=budget, origin=origin,
                selected=c.selected, trained=c.trained, disagreements=int(g.disagree.sum()),
                switched=int(g.choose_source_selected.sum()),
                available_rescues=int(g.gain.eq(1).sum()), available_harms=int(g.gain.eq(-1).sum()),
                selected_rescues=int((g.gain.eq(1) & g.choose_source_selected).sum()),
                selected_harms=int((g.gain.eq(-1) & g.choose_source_selected).sum()),
                partial_selection_tasks=int(((u["sum"] > 0) & (u["sum"] < u["count"])).sum()),
                ba_delta=float(contrib.sum())))
            for (sample, ship, truth), h in g.assign(ba_contribution=contrib).groupby(["sample_id", "mmsi", "truth"]):
                details.append(dict(stage=stage, pool=pool, budget=budget, origin=origin,
                    sample_id=sample, mmsi=ship, truth=truth,
                    contribution_pp=float(h.ba_contribution.sum()),
                    rescues=int((h.gain.eq(1) & h.choose_source_selected).sum()),
                    harms=int((h.gain.eq(-1) & h.choose_source_selected).sum())))

    result = pd.DataFrame(results)
    coverage = pd.concat(covers, ignore_index=True)
    draw = pd.concat(draws, ignore_index=True)
    mechanism = pd.DataFrame(mechanisms)
    result.to_csv(OUT / "stage_comparison.csv", index=False)
    coverage.to_csv(OUT / "stage_learning_coverage.csv", index=False)
    draw.to_csv(OUT / "stage_draw_deltas.csv", index=False)
    mechanism.to_csv(OUT / "stage_event_mechanisms.csv", index=False)
    pd.DataFrame(details).to_csv(OUT / "stage_query_contributions.csv", index=False)

    # Detect unintended endpoint changes when only query batching was enlarged.
    first = pd.read_csv(OUT / "source_scored.csv", dtype={"mmsi": str})
    expanded = pd.read_csv(OUT / "query_coverage/source_scored.csv", dtype={"mmsi": str})
    keys = ["pool", "split", "budget", "task", "sample_id"]
    joined = first.merge(expanded, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    assert len(joined) == len(first)
    probability_cols = [c for c in first if c.startswith(("body_p", "mix_p", "half_p"))]
    assert probability_cols
    max_diff = 0.
    for col in FEATURES + probability_cols:
        a, b = joined[col + "_a"].to_numpy(), joined[col + "_b"].to_numpy()
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)
        max_diff = max(max_diff, float(np.max(np.abs(a - b))))
    check = result.merge(mechanism, on=["stage", "pool", "budget", "origin"])
    np.testing.assert_allclose(check.source_selected_ba_delta, check.ba_delta, atol=1e-10)
    save_json(OUT / "REPORT_CHECKS.json", dict(original_source_rows_replayed=len(joined),
        original_features_probability_max_difference=max_diff, weighted_event_delta_replay=True))

    cross = result.loc[result.origin.eq("cross")]
    current = cross.loc[cross.stage.eq("expanded")]
    small = cross.loc[cross.stage.eq("query8")]
    compact = ["pool", "budget", "selected", "source_selected_oa_delta", "source_selected_ba_delta",
               "mix_ba_delta", "half_ba_delta", "source_selected_vs_random_ba"]
    learning = coverage[["stage", "pool", "budget", "fit_query_ships", "fit_rescues", "fit_harms",
                         "fit_event_ships", "selection_rescues", "selection_harms", "trained", "selected"]]
    events = mechanism.loc[mechanism.origin.eq("cross"), ["stage", "pool", "budget", "selected",
        "available_rescues", "available_harms", "selected_rescues", "selected_harms", "partial_selection_tasks"]]
    draw_view = draw.loc[draw.stage.eq("expanded") & draw.origin.eq("cross"),
                         ["pool", "budget", "draw", "source_selected", "source_selected_random"]]
    done = [json.loads((f / "DONE.json").read_text(encoding="utf-8")) for f in (OUT, OUT / "query_coverage")]
    text = f"""# 放宽源端支持组织：32/64支持学习器结果

## 结论

**材料能凑出来，也能完成训练；但本轮没有得到比既有方案更可靠的环带选择规则。**

- B32五池均完成fit和selection；B64宁波、青岛完成。其余三池没有同时满足两端容量，未补造或借用目标材料。
- 最初每组支持只配8条源查询，B64分别仅出现误伤或救回。保持每一个支持集合不变、补足同一源采集的合法不同船查询后，七组都出现两种事件并完成训练。说明前一轮不能训练确有查询覆盖不足这一限制，不等于全局没材料。
- 扩覆盖后的源端选择策略：B32宁波/青岛跨采集BA分别+0.1190/+0.1852个百分点；两者均与同任务同切换数量的随机期望相同，尚未体现样本级有利选择的额外收益。固定组合分别+0.4448/+0.8333，不能把选择器说成更强。
- B32杰贝阿里/上海/新加坡BA分别-0.2540/-0.2917/-0.3430；五池平均{current.loc[current.budget.eq(32), 'source_selected_ba_delta'].mean():+.4f}个百分点。OA与BA都列出，不用一个指标覆盖另一个。
- B64两池源端selection都选body，目标OA/BA保持不变，避开固定环带组合的退化。这里是“不启用”的正确退路，不是环带选择器涨点。
- 更关键：这两池B64目标端均无固定环带救回事件，宁波0救回/2误伤、青岛0救回/4误伤（含draw重复）。在只选择body或mix的限定动作下，即使知道全部标签的完美选择器也只能维持body的OA/BA。这是该目标范围的top-1收益空间为零，不是整个环带路线没有价值，也不限制其他预算、港口或连续概率指标。
- 既有Jebel Ali B16候选不被这轮否定：其BA+0.9260、相对同数量随机+0.2005的局部结果继续保留，见../ring_matched/FINDINGS_ZH.md。本轮不更换主模型，不扩大公开数据，也不继续堆新门控。

## 实际做了什么

只放宽源端任务组织：跨采集单一支持来源，至少2类，去掉same/cross共同容量及源查询场景40样本门槛；最初固定8条查询。保留原fit/selection边界、far_ok几何条件、有效MMSI、同港不同采集、查询与支持船只排除。32/64嵌套支持、五次抽样；未把五个源池合并训练。

第一轮有695个源任务、5560行重复查询暴露。覆盖补充沿用完全相同的任务和每一个支持ID，只增加同源采集合法查询，保留原8条；因各任务支持身份不同，扩展后的查询队列可有小差异。第一轮结果完整保留。扩展决策仅基于源端事件不足，在读取新目标结果之前冻结。

两轮都沿用18个预测时可得特征、逻辑回归C=0.1、深度2树、阈值0.5。模型只在救回/误伤事件上学二分类，源验证在完整查询上比较body/logistic/tree。先冻结所有源选择，再读取原目标缓存。船体/环带端点、目标查询和支持全部未改。

**覆盖补充不是单一纯因果干预。** 它同时改变真实船只覆盖、类别/任务组成和监督事件数量；原代码事件权重归一到均值1，新增事件也会改变固定C下的有效正则相对强度。本轮证明“可训练”，不把性能变化全归因于独立信息增加。

## 训练材料与事件

query_ships是该池/预算内去重船只；rescues/harms包含支持抽样重复暴露，不能当作独立样本。不同预算之间也可能重复。

{markdown(learning)}

尤其B64只有宁波5条、青岛14条fit事件船，仍不算充足的条件规律训练集。支持数量够，不等于救回/误伤监督够。部分5次draw重复同一支持集合，覆盖详见task_coverage.csv；这不会凭空提供5份独立证据。

## 同一目标任务上的两轮结果

单位：百分点；全部为相对船体body的变化，random列为源选择策略相对同任务同切换数量随机期望。half为半量船体补充、恢复另一半原D18的对照。NA表示模型未训练，不能将回退0当训练成功。

### 初始8查询

{markdown(small[compact])}

### 扩展合法查询

{markdown(current[compact])}

完整OA/BA、NLL/Brier变化及same/cross结果见stage_comparison.csv；宏平均按原任务口径等权汇总采集，不是把所有港样本简单合并的微平均。

## 为什么没有转化成稳定收益

1. 训练任务数不是有效监督量。初始B64只有一类事件；增加不同船覆盖解决了可拟合性，但没有自动解决规律的可迁移性。
2. 训练目标是“何时环带有利”，不是仅区分船型。环带与船体有不同预测时，真实救回/误伤才提供直接监督。宁波B64 fit只有8次事件、5条船，不能把356条查询暴露说成356条有效二分类监督。
3. 源端选出的正收益没有稳定延续到目标。比如扩覆盖后Jebel源验证逻辑回归+0.1515 BA、目标-0.2540；新加坡源验证+1.0458、目标-0.3430。观察支持“当前源规律迁移不足”，尚不能确定具体物理原因。
4. 部分小任务只有一个分歧样本，或策略对全部分歧全开/全关，此时同数量随机期望与策略一样。正的最终BA仍有使用价值，但不能证明学习到了逐样本辨别能力。partial_selection_tasks给出真正任务内部分选择的数量。
5. 固定融合本身存在有收益和有损失的范围。B32宁波、青岛更简单的固定组合本轮更好；B64回退body能保住结果。不会据此要求每港全胜，也不会通过目标结果反向挑部署策略。
6. 本轮两个B64目标池没有任何可救回事件，进一步扩大训练也不能让这两个既定端点的二选一超越body的OA/BA。下一次做选择器之前，应先区分“有救回但选不准”和“固定候选根本没有救回”这两个问题；目标标签可做事后收益上界诊断，不能用来训练或反选部署门控。

{markdown(events)}

五次支持抽样配对BA变化如下，它们共用查询，不是五个独立训练随机种子；不做把重复事件当独立样本的显著性宣称。

{markdown(draw_view)}

## 验证与后续取舍

- 两阶段均保存源冻结记录、训练模型、决策、端点指标；原目标端点和旧策略重放一致，原8源查询的特征/概率最大差{max_diff:.3g}，覆盖变化没有意外改动既有预测端点。
- 逐查询加权救回减误伤与报告BA增量一致，明细stage_query_contributions.csv，可定位真实受益与受损的船及支持draw。
- 两轮缓存计算分别{done[0]['seconds']:.1f}/{done[1]['seconds']:.1f}秒；不是端到端推理成本。选择器需要两端输出，不宣称省算力。
- 这里仍是历史研发目标集，范围发现属于探索，不是新独立确认。目标标签未进入本轮训练或选模型，但长期开发暴露仍需说明。

**收口决定：保持船体局部读出及既有有边界的环带候选；B16 Jebel选择器继续作为局部候选。本轮新增B32/B64选择器保留记录，不升级为主模块。** 后续若继续学习规律，优先需要更多独立源采集上的真实救回/误伤船只，而非重复原少量查询或扩大门控网络。无需现在再追加一轮。
"""
    (OUT / "FINDINGS_ZH.md").write_text(text, encoding="utf-8")
    print(current[compact].to_string(index=False))
    print("Replay max difference:", max_diff)


if __name__ == "__main__":
    main()
