"""Report observed ring-use regimes and actual query examples, not a selector."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from localization import spatial_masks, PIXELS

ROOT = Path(__file__).resolve().parent / "body_context_expansion"
OUT = ROOT / "ring_conditions"


def md(table):
    table = table.copy()
    for col in table.select_dtypes(include="number"):
        table[col] = table[col].map(lambda v: f"{v:.4f}")
    return "| " + " | ".join(table.columns) + " |\n| " + " | ".join(["---"]*len(table.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False, name=None))


def main():
    data = pd.read_csv(OUT / "paired_query_conditions.csv", dtype={"mmsi": str})
    overall = pd.read_csv(OUT / "overall_effects.csv")
    group = pd.read_csv(OUT / "subgroup_effects.csv")
    controls = pd.read_csv(OUT / "summary.csv")
    case_meta = pd.read_csv(OUT / "conditions.csv", dtype={"mmsi": str})
    v = pd.read_csv(OUT / "per_vessel_context.csv", dtype={"mmsi": str})
    pair = v.loc[v.origin.eq("cross") & v.budget.eq(64) & v.rescues.ge(2) & v.new_rescues.ge(2)
                 & v.avg_vs_half.gt(0) & v.avg_vs_shuffle.gt(0)].sort_values(["port", "sample_id"])
    pair.to_csv(OUT / "repeat_rescue_candidates.csv", index=False)
    assert set(pair.mmsi) == {"314867000", "636091308"}
    harm = v.loc[v.origin.eq("cross") & v.budget.eq(64) & v.mmsi.eq("311000493")]
    selected = pd.concat([pair, harm])
    plot_cases(selected, case_meta, data)
    summaries = []
    for _, row in selected.iterrows():
        g = data.loc[data.sample_id.eq(row.sample_id) & data.budget.eq(64) & data.origin.eq("cross")].sort_values("draw")
        g.to_csv(OUT / f"case_{row.mmsi}_all_draws.csv", index=False)
        summaries.append(dict(mmsi=row.mmsi, vessel=row.class_name, incidence=row.incidence_deg,
            ring_std=row.ring_std_vv, contrast=row.contrast_vv, rescues=row.rescues, harms=row.harms,
            body_correct=int(g.ship.sum()), real_ring_correct=int(g.additive.sum()), half_body_correct=int(g.body_half.sum()),
            shuffled_correct_expected=float(g.shuffle_mean.sum())))
    cases = pd.DataFrame(summaries)
    conditions = group.loc[group.condition.isin(["standardized_contrast_vv_bin", "raw_ring_vv_std_bin", "incidence_bin", "body_top2_gap_bin", "ring_support_match_bin"]) & group.origin.eq("cross")]
    cols = ["budget", "condition", "level", "ships", "rescues", "harms", "delta_ba_vs_ship", "delta_ba_vs_half", "delta_ba_vs_shuffle"]
    port = group.loc[group.condition.eq("port") & group.origin.eq("cross"), cols]
    types = group.loc[group.condition.eq("class_name") & group.origin.eq("cross"), cols]
    rows = ["# 哪些船、哪些条件下环带补充了框内证据", "", "2026-10-15。完成既有任务的配对对照与条件描述；未训练新模型、未扩公开数据集。",
        "", "## 结论", "",
        "发现了实际、可重复的个案补证，但尚未得到跨预算稳定的物理条件开关。最明确的实例是新加坡的原油轮314867000和集装箱船636091308：B64五次支持抽样中，真实组合分别比框内补充多判对两次，原模型及减半框内强度对照均未取得这两次救回。两艘船、两个采集，不是普遍船型规律，也不是两次独立数据集验证。",
        "",
        "在更广范围，真实环带未普遍胜过同采集内置乱环带；部分收益可以由减少框内替换幅度解释。环带有局部作用与当前固定组合的整体净收益不足可以同时成立。本轮只建立可解释边界，不据此训练或发布新门控。",
        "", "## 两个对照回答什么", "",
        "减半框内：只将原D18核的12.5%换成框内核，保留其他原核，不加入环带。真实组合中框内同样占12.5%，另12.5%给环带。故比较可区分环带内容与单纯框内强度变化。",
        "",
        "显式环带置乱：三个固定种子，把环带坐标在同采集、同查询/支持角色内重排；框内、源先验、标签和权重不变，并重新求解支持系数。本次置乱的是分类核中的环带，不是此前仅门控输入。它保留采集级分布，主要检验逐船对应关系；不能区分所有场景级语义与统计正则作用。",
        "", "## 全部配对净效果", "",
        md(overall[["budget", "origin", "delta_ba_vs_ship", "delta_ba_vs_half", "delta_ba_vs_shuffle", "rescues", "harms", "new_vs_original", "restore_original"]]),
        "",
        "所有差值均为真实组合减对照，BA单位为百分点。顺序为抽样内样本/类别、港内任务、等港汇总。B16/B32/B64分别20/19/10任务，预算覆盖不同。救回/误伤是重复支持抽样事件，不是独立船数。",
        "",
        "重要区别：B64跨采集的16次相对框内救回中，8次原模型也错，是相对原模型的新救回；另8次原模型本来正确，是恢复框内补充造成的损失。11次未被减半框内救回，2次三种环带置乱均未救回，其中1次原模型也错。最后一项是更强的个案线索，不是必须全满足才认可有效的标准。",
        "", "## 哪些船有具体证据", "", md(cases), "",
        "表中correct为同一查询五次支持抽样的判对次数；置乱为三个种子判对次数的期望平均。并非五个独立船只。正例按重复救回和对照表现事后选取，用于解释而非估计平均收益；末行为边界反例，按与原油轮相近的环带起伏、框内外对比选取，不用于拟合阈值。",
        "", "![实际SAR查询与支持抽样敏感性](ring_condition_cases.png)", "",
        "图中是模型实际uint8输入：青色为框内边界、黄色为环带边界，并非分割出的纯船体。热图为各次支持抽样的分类正确性；置乱行是三个种子的平均。",
        "", "## 从案例中能解释的机制", "",
        "集装箱船636091308：入射角36.26度，VV环带标准差0.0557、框内外标准化对比2.704。两次救回的支持抽样中，框内Top3类内相似度聚合都更偏向原油轮，环带聚合都更偏向真实的集装箱船；环带真实类对最强异类的相似度优势分别+0.145和+0.486。其他抽样中这种支持证据并不存在。这是同一张查询、相同观测条件，仅支持船改变的对照，说明可用案例是否提供相符类别证据比仅看角度更关键。该简单聚合只用于解释，不是实际分类器。",
        "",
        "原油轮314867000：入射角43.61度，VV环带标准差0.0754、对比2.429，两次救回。但环带最近类别聚合并没有直接把真实类排第一；原有逐支持分差分解显示核关系变化与系数重求解联合把概率差推过边界。因此不能将所有收益都讲成环带找到同类最近邻。",
        "",
        "反例311000493是成品/化学品船，入射角40.63度、环带标准差0.0677、对比2.570，与上述原油轮的统计条件接近，却两次被组合误伤。这排除了将这几个数值直接当作可靠启用规则的简单解释，但不否认特定船与支持组合中的净收益。",
        "", "## 观测条件结论", "",
        "1. 改判集中于框内补充模型的临界预测：所有跨采集救回和误伤均落入预测Top2间距最低三分组；实际间距最高约2.51个百分点。这解释了环带为何能改变这些决策，但困难并不等于适合加环带，误伤也集中在那里。",
        "2. VV环带起伏较小的B32子组有正信号：相对框内BA+2.611、相对减半框内+0.558、相对置乱均值+1.487个百分点；但只有7次救回/5次误伤，跨船型与场景组成会影响条件内BA。保持同场景同类、两端各至少3个不同查询的对照仅有2个可用单元，尚不足以把它写成低起伏条件定律。",
        "3. 40度以上入射角在B16有正信号，但B32/B64未延续。相对方向只有66/858查询可用，按方向分组的改判事件很少且预算间方向不一致；不能给出可靠角度窗口。速度没有建立可靠样本级关联，本轮不解释运动状态。",
        "4. 支持集本身是条件的一部分。B16/B32固定同一查询、仅比较框内均判错的不同抽样时，救回抽样的真实类环带优势较高的查询分别15/22、14/22；B64仅5/14。说明有案例匹配路径，但不是统一解释。此指标使用查询真实类作事后诊断，不可用于部署选择。",
        "",
        "像素统计是已拉伸输入的0到1量，不是sigma0、dB或海况。未将高标准差直接命名为强杂波，也未将框内对比定义为真实信噪比。连续分组使用无标签协变量三分位数，未按结果扫描阈值；本轮子组结果仍是探索，不是独立确认。",
        "", "## 逐港与逐船型", "", md(port), "", md(types),
        "", "## 预设条件全表", "", md(conditions[cols]),
        "", "## 文件与后续重点", "",
        "conditions.csv和CONDITION_SOURCES.md记录4289个查询/支持候选的字段来源；查询677/858有入射角、66/858有相对方向。condition_thresholds.json固定本轮描述分组。rescue_harm_conditions.csv、per_vessel_context.csv和case_*_all_draws.csv可追溯每次救回与误伤。within_scene_class_condition_contrasts.csv保留同场景同类对照，不能把不同类别构成当成物理因果。",
        "",
        "验证：490任务单元、七个方案共162610条预测逐行核对概率、argmax、正确性、NLL和Brier；真实组合在15个港口-预算池中回放相符，降强度核代数相符。环带置乱实际改变约96.51%的向量对应关系；不是零作用对照。见VERIFICATION.json。",
        "",
        "当前更有证据的方向是描述和验证‘某个临界查询与当前支持集之间，环带能否提供不同且有效的类别关系’，而不是按港口名、船型或入射角设置固定开关。先保留这些局部证据和失败边界；未证实可部署选择器的净收益前，不追加门控训练。"]
    (OUT / "FINDINGS_ZH.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
    print(cases.to_string(index=False))


def plot_cases(selected, metadata, data):
    torch.set_num_threads(2)
    frame = metadata.set_index("sample_id").loc[selected.sample_id].reset_index()
    pixel_rows = frame.row_index.to_numpy(int)
    ship, _, _, ring, _ = spatial_masks(pixel_rows)
    ship = F.interpolate(torch.from_numpy(ship[:,None]).float(), (128,128), mode="nearest")[:,0].numpy()
    ring = F.interpolate(torch.from_numpy(ring[:,None]).float(), (128,128), mode="nearest")[:,0].numpy()
    pixels = np.load(PIXELS, mmap_mode="r")
    fig, axes = plt.subplots(len(frame), 3, figsize=(11, 9), gridspec_kw={"width_ratios":[1,1,1.4]}, layout="constrained")
    plt.rcParams.update({"pdf.fonttype":42})
    for i, row in frame.iterrows():
        for channel, pol in enumerate(("VV", "VH")):
            ax = axes[i,channel]
            ax.imshow(pixels[int(row.row_index),channel], cmap="gray", vmin=0, vmax=255)
            ax.contour(ship[i], levels=[.5], colors=["#37d5dd"], linewidths=.7)
            ax.contour(ring[i], levels=[.5], colors=["#ffda55"], linewidths=.6)
            ax.set_title(f"{row.class_name.replace('_',' ')} | {pol}\nMMSI {row.mmsi}", fontsize=10)
            ax.set_axis_off()
        g = data.loc[data.sample_id.eq(row.sample_id)&data.budget.eq(64)&data.origin.eq("cross")].sort_values("draw")
        matrix = g[["original","ship","body_half","additive","shuffle_mean"]].to_numpy().T
        ax=axes[i,2]
        ax.imshow(matrix,cmap="RdYlGn",vmin=0,vmax=1,aspect="auto")
        ax.set(xticks=range(5),xticklabels=range(5),yticks=range(5),
            yticklabels=["Original","Body","Half body","Real ring","Shuffled ring"],xlabel="Support draw")
        ax.set_title(f"Correctness across support draws\nIncidence {row.incidence_deg:.2f} deg",fontsize=10)
        for y in range(5):
            for x in range(5):
                ax.text(x,y,f"{matrix[y,x]:.2g}",ha="center",va="center",fontsize=10)
    fig.savefig(OUT / "ring_condition_cases.png",dpi=170)
    fig.savefig(OUT / "ring_condition_cases.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
