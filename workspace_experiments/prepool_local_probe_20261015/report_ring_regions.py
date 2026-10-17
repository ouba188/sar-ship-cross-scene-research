"""Describe region/position evidence with paired endpoints and scoped utility."""

import json

import numpy as np
import pandas as pd

from report_ring_predictability import markdown
from probe_ring_regions import OUT, ROOT


def main():
    metrics = pd.read_csv(OUT / "metrics.csv")
    predictions = pd.read_csv(OUT / "predictions.csv", dtype={"mmsi": str})
    coverage = pd.read_csv(OUT / "region_coverage.csv")
    mean = pd.read_csv(OUT / "summary.csv")
    per_port = pd.read_csv(OUT / "per_port.csv")
    previous = pd.read_csv(ROOT / "ring_conditions/metrics.csv")
    keys = ["port", "fold", "budget", "origin", "draw"]
    for new, old in (("body", "ship"), ("near", "additive")):
        pair = metrics.loc[metrics.arm.eq(new)].merge(previous.loc[previous.arm.eq(old)],
            on=keys, suffixes=("_new", "_old"), validate="one_to_one")
        assert len(pair) == len(metrics.loc[metrics.arm.eq(new)])
        for col in ("oa", "ba", "macro_nll", "macro_brier"):
            np.testing.assert_allclose(pair[col + "_new"], pair[col + "_old"], atol=1e-10)
    half = previous.loc[previous.arm.eq("body_half")].copy()
    half_port = half.groupby(["port", "budget", "origin", "arm"])[["oa", "ba", "macro_nll", "macro_brier"]].mean().reset_index()
    half_mean = half_port.groupby(["budget", "origin", "arm"])[["oa", "ba", "macro_nll", "macro_brier"]].mean().reset_index()
    mean = pd.concat([mean, half_mean], ignore_index=True)
    per_port = pd.concat([per_port, half_port], ignore_index=True)
    rows = []
    for (budget, origin), g in mean.groupby(["budget", "origin"]):
        a = g.set_index("arm")
        for arm in a.index:
            row = dict(budget=budget, origin=origin, arm=arm, oa=a.loc[arm, "oa"], ba=a.loc[arm, "ba"])
            for ref in ("body", "near", "body_half"):
                for col in ("oa", "ba"):
                    row[f"{col}_vs_{ref}"] = a.loc[arm, col] - a.loc[ref, col]
            rows.append(row)
    delta = pd.DataFrame(rows)
    delta.to_csv(OUT / "effects.csv", index=False)
    port_delta = per_port.pivot(index=["port", "budget", "origin"], columns="arm", values="ba")
    port_delta = port_delta.subtract(port_delta.body, axis=0).reset_index()
    port_delta.to_csv(OUT / "per_port_ba_deltas.csv", index=False)
    draw = metrics.groupby(["port", "budget", "origin", "draw", "arm"])[["oa", "ba"]].mean().groupby(
        ["budget", "origin", "draw", "arm"])[["oa", "ba"]].mean().reset_index()
    draw = draw.pivot(index=["budget", "origin", "draw"], columns="arm", values="ba")
    draw = draw.subtract(draw.body, axis=0).reset_index()
    draw["position_vs_bag"] = draw.near_position - draw.near_bag
    draw["multiring_vs_near"] = draw.multiring - draw.near
    draw.to_csv(OUT / "draw_ba_deltas.csv", index=False)
    byclass, ship_rows, event_rows = [], [], []
    for (budget, origin), group in predictions.groupby(["budget", "origin"]):
        unit = ["port", "fold", "draw"]
        base = group.loc[group.arm.eq("body")].copy()
        nclass = base.groupby(unit).truth.transform("nunique")
        perclass = base.groupby(unit + ["truth"]).truth.transform("size")
        nscene = base[["port", "fold"]].drop_duplicates().groupby("port").size()
        ndraw = base[unit].drop_duplicates().groupby(["port", "fold"]).size()
        base["weight"] = 1 / (base.port.nunique() * base.port.map(nscene).to_numpy() *
            np.array([ndraw.loc[(p, f)] for p, f in zip(base.port, base.fold)]) * nclass.to_numpy() * perclass.to_numpy())
        np.testing.assert_allclose(base.weight.sum(), 1, atol=1e-12)
        join = unit + ["sample_id"]
        for arm in group.arm.unique():
            new = group.loc[group.arm.eq(arm)]
            m = base.merge(new[join + ["correct"]], on=join, suffixes=("_body", "_new"), validate="one_to_one")
            m["rescue"] = m.correct_new & ~m.correct_body
            m["harm"] = ~m.correct_new & m.correct_body
            m["contribution"] = 100 * m.weight * (m.correct_new.astype(int) - m.correct_body.astype(int))
            expected = delta.loc[delta.budget.eq(budget) & delta.origin.eq(origin) & delta.arm.eq(arm), "ba_vs_body"].iloc[0]
            np.testing.assert_allclose(m.contribution.sum(), expected, atol=1e-10)
            event_rows.append(dict(budget=budget, origin=origin, arm=arm,
                rescue=int(m.rescue.sum()), harm=int(m.harm.sum()),
                rescue_ships=int(m.loc[m.rescue, "mmsi"].nunique()), harm_ships=int(m.loc[m.harm, "mmsi"].nunique())))
            c = m.groupby("truth").agg(contribution_pp=("contribution", "sum"), rescue=("rescue", "sum"),
                harm=("harm", "sum"), query_ships=("mmsi", "nunique")).reset_index()
            c["budget"], c["origin"], c["arm"] = budget, origin, arm
            byclass.append(c)
            if budget == 32 and origin == "cross" and arm == "multiring":
                ships = m.groupby(["mmsi", "truth"]).agg(contribution_pp=("contribution", "sum"),
                    rescue=("rescue", "sum"), harm=("harm", "sum")).reset_index()
                ship_rows.append(ships)
    classes = pd.concat(byclass, ignore_index=True)
    classes.to_csv(OUT / "class_contributions.csv", index=False)
    pd.concat(ship_rows, ignore_index=True).to_csv(OUT / "B32_multiring_ship_contributions.csv", index=False)
    events = pd.DataFrame(event_rows)
    events.to_csv(OUT / "rescue_harm.csv", index=False)
    region = coverage[[c for c in coverage if c.endswith("_pixels")]].agg(["min", "median", "max"]).T.reset_index(names="region")
    region.to_csv(OUT / "region_sizes.csv", index=False)
    focus = delta.loc[delta.budget.eq(32) & delta.origin.eq("cross") & delta.arm.eq("multiring")].iloc[0]
    cross = delta.loc[delta.origin.eq("cross")]
    compact = cross.pivot(index="arm", columns="budget", values="ba_vs_body").reset_index()
    compact.columns = ["arm", "B16_vs_body", "B32_vs_body", "B64_vs_body"]
    focus_ports = port_delta.loc[port_delta.budget.eq(32) & port_delta.origin.eq("cross"),
        ["port", "near", "touch", "far", "wide", "multiring", "near_position", "near_bag"]]
    focus_draw = draw.loc[draw.budget.eq(32) & draw.origin.eq("cross"), ["draw", "near", "touch", "far", "multiring", "multiring_vs_near", "position_vs_bag"]]
    fclass = classes.loc[classes.budget.eq(32) & classes.origin.eq("cross") & classes.arm.eq("multiring")]
    fevent = events.loc[events.budget.eq(32) & events.origin.eq("cross") & events.arm.eq("multiring")].iloc[0]
    stats = dict(edge_fraction=float(coverage.outer_reaches_chip_edge.mean()), rows=len(coverage),
        endpoint_metric_replay=True, class_contribution_replay=True,
        scope="same490 frozen tasks, no new selector training or target tuning")
    (OUT / "REPORT_CHECKS.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    text = f"""# 环带范围与空间排列：首轮固定验证

## 结论

**有值得保留的区域信号：先前跳过的贴框信息并非一定应删；三层分开读取在B32有净收益。位置思想可以研究，但当前图像象限顺序未显示稳定优势，不能据此宣布Transformer位置编码已奏效。**

- B32三层分开读出较body OA {focus.oa_vs_body:+.4f}、BA {focus.ba_vs_body:+.4f}个百分点；较原near OA {focus.oa_vs_near:+.4f}、BA {focus.ba_vs_near:+.4f}；较已有half-body BA {focus.ba_vs_body_half:+.4f}。它已在相同完整分类流程中评价，不是孤立检索分数。
- B16贴框区BA相对body约+0.256，但OA仍略降；B64各上下文变体仍不及body。更大预算队列不同，不能把预算变化直接当因果解释。
- 位置保留16维相对同内容排序16维的跨采集BA，B16/B32/B64约-0.135/+0.039/-0.142个百分点。分块后的细粒度统计与显式方位是两回事；本轮不足以支持仅保留图像方位就更好，也没有否定相对船轴/观测几何位置或真正token交互。
- 数值输入仍为uint8缓存，没有运行拉伸前数据实验。可用性核查独立记录于SOURCE_DATA_ZH.md，未确认当前可直接读取的同样本高位深源。

## 实际范围与实现

五港20/19/10个B16/32/64任务，各五次支持draw，同/异采集配对，合计490任务单元、8方案、185840预测行。包括7660个去重源gallery与目标样本用于像素读出；源变换每个池独立，只用原合法source_train/gallery。

区域为box外0到2、2到8、8到16层膨胀区；三者互不重叠，合并等于0到16的wide。掩膜从56网格nearest映射128输入；仍在原chip内，不能视为扩大了原裁片视野。无空区域和空象限。outer16触及chip边界比例{stats['edge_fraction']:.2%}，这表示可用区域可能截断，不是完整远场观测。

每区同样4维VV/VH均值标准差，经源PCA中心化与单位化。multiring按等权拼接三个独立归一化区域，所以兼有信息范围与区域权重组织的变化；与wide的差不能被解释为纯信息增加。所有新增上下文仍与同一船体坐标等权组合，专家局部替换比例25%，类别关系、先验、支持求解和最终融合未动。

{markdown(region)}

## 相对船体的全部跨采集BA变化

单位：百分点。body_half为既有减半船体替换对照，不含新增环带；只是复用旧缓存，本轮未重新训练它。

{markdown(compact)}

OA、BA、相对原near与half-body的完整结果及同采集结果见effects.csv。没有因同采集也改善而否认收益。

## B32区域收益出现在哪里

{markdown(focus_ports)}

multiring相对body在4/5港为正，Jebel有退化；没有按港口挑不同方案组成一个事后最优系统。

{markdown(focus_draw)}

支持draw是固定查询的重复抽样，不是独立训练种子。逐类贡献如下，贡献按原港/采集/任务/类权重计入整体BA，不是某类自身召回率变化。

{markdown(fclass)}

相对body共{fevent.rescue}次救回、{fevent.harm}次误伤，涉及{fevent.rescue_ships}条救回船和{fevent.harm_ships}条误伤船（两集合可重叠）。具体船及贡献保存在B32_multiring_ship_contributions.csv。这里解释收益落在哪里，不把某个场景组成差异直接说成已确认物理原因。

## 如何借鉴Transformer位置编码

ViT原方法先形成patch embedding，再加入位置embedding，让后续编码器保留并使用局部内容与位置的对应。参考Dosovitskiy等，An Image is Worth 16x16 Words，ICLR2021/arXiv:2010.11929，方法3.1节。

适合我们的待验证设计是：先保留局部token，再加入相对船框坐标、距离带等位置描述，并通过非线性映射或token交互使用它。不是先将整圈压成4个均值标准差，再附加坐标。若只把固定位置向量相加后直接线性平均，也不能充分保留内容与位置对应。

本轮是低成本位置价值探针：原near分为相对框中心的四个图像象限，每象限4维；position按固定象限顺序拼接，bag对完全相同token按自身内容排序。二者均16维、同面积、同局部统计，区别是是否保留象限身份。固定排列可视为隐式位置槽位，但不是训练的Transformer/注意力。

bag排序控制保留局部异质性且抹去象限标签，存在排序不连续的局限，不是所有无位置编码方法的最优代表。图像上/下/左/右不等于船首/船尾；不带船轴方向时，跨船方向变化可能限制这种排列的转移，这是待检验解释，不是本轮证明的根因。

## 拉伸前数据的准确含义

这里指映射为显示8bit之前的数值栅格，可以是原GRD产品的DN，也可以是已定标的线性强度/后向散射量；二者不能混称。ESA SNAP Calibration文档说明Sentinel-1 Level-1通过定标LUT将DN转换为相应定标量。保存成float不代表已经定标，8bit除以255也不能找回之前的数值。

不要求回到原始回波，也不要求SLC复数。现有清单可追溯至GRDH派生8bit tile/PNG；部分历史ZIP/int16路径线索目前未找到，远端现存状态未知。详见SOURCE_DATA_ZH.md；本轮未下载、未全景解码、不把缺少本地路径等同于数据不存在。

## 取舍与验证

保留贴框和多距离带读出的限定正信号，暂不替换原主方案或重训QSRG门控。需要先确认新端点在源端亦能提供可用救回/误伤，再决定是否接入选择器。位置方案现阶段保持研究候选，不把弱混合结果包装为成熟Transformer模块。

body及原near全部490任务指标与旧缓存重放一致；原near重新像素提取后的五池预测误差低于5e-8。逐类加权救回减误伤与整体BA增量逐项重合。首次启动遇到旧脚本同名run导入冲突，已将掩膜提取放在独立进程，未改旧模块或任何原始数据。全部结果仍属历史研发队列的探索，非独立新数据确认。
"""
    (OUT / "FINDINGS_ZH.md").write_text(text, encoding="utf-8")
    print(compact.to_string(index=False))
    print("B32 multiring", focus.to_dict())
    print("B32 draws\n", focus_draw.to_string(index=False))
    print("B32 class contributions\n", fclass.to_string(index=False))
    print("Coverage", stats)


if __name__ == "__main__":
    main()
