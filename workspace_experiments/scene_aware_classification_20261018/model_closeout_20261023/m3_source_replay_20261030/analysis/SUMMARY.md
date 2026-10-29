# M3 source replay 三顺序分析

两数据集、两主要 arm、17/29/43 三顺序共 12 个训练端点的本轮预测与训练报告已齐，CPU 配对分析完成。**研究目标未达成，不支持替换原完整模型。** 私有集只有相对同预算续训对照的小幅净收益，公共集三顺序平均退化。`source_mix` 是 source_fit 混合训练，不是中心化。

## 范围与口径

- 私有 development：160 查询，7 个实际出现类别，fishing 为 0；OSS2 validation：228 查询，4 类。未读取最终测试集。
- 两 arm 都从对应顺序的原完整模型开始，仅更新 M3 原 value 参数；类别/船只抽样，前半批次匹配，后半批次来自 adapt 或 fit。同预算 54/96 步，其他细节见上级 PROTOCOL。
- 每条 NPZ 按 ID 与实际 manifest 配对，并核对标签、两臂保存基线、实际 reference IDs 和报告中指标。只使用本轮预测；缺文件或样本时明确 pending，不借用历史预测。
- 下表 OA/BA 为百分数，差值为百分点；NLL/Brier 未乘 100。均值为三顺序均值，SD 为描述性样本标准差。相同查询的跨顺序事件不是独立船只。

## 端点均值

| 数据 | 端点 | OA | BA | NLL | Brier |
|---|---|---:|---:|---:|---:|
| private | saved_baseline | 45.000000 | 35.733663 | 2.430989 | 0.840963 |
| private | adapt_only | 44.166667 | 35.675840 | 2.394498 | 0.844893 |
| private | source_mix | 44.375000 | 35.866316 | 2.402862 | 0.845614 |
| oss2 | saved_baseline | 58.479532 | 49.636699 | 1.381566 | 0.618582 |
| oss2 | adapt_only | 58.333333 | 49.413542 | 1.410364 | 0.622839 |
| oss2 | source_mix | 57.894737 | 49.242794 | 1.405884 | 0.621768 |

## 配对变化

| 数据 | 比较 | ΔOA 均值±SD | ΔBA 均值±SD | ΔNLL | ΔBrier | rescue/harm 事件 |
|---|---|---:|---:|---:|---:|---:|
| private | adapt_only - baseline | -0.833333 ± 1.572882 | -0.057823 ± 0.870647 | -0.036491 | +0.003930 | 4/8 |
| private | source_mix - baseline | -0.625000 ± 1.653595 | +0.132653 ± 0.870089 | -0.028127 | +0.004651 | 4/7 |
| private | source_mix - adapt_only | +0.208333 ± 0.360844 | +0.190476 ± 0.329914 | +0.008364 | +0.000721 | 1/0 |
| oss2 | adapt_only - baseline | -0.146199 ± 0.253224 | -0.223156 ± 1.385645 | +0.028799 | +0.004257 | 10/11 |
| oss2 | source_mix - baseline | -0.584795 ± 0.506448 | -0.393904 ± 0.410375 | +0.024318 | +0.003186 | 10/14 |
| oss2 | source_mix - adapt_only | -0.438596 ± 0.438596 | -0.170748 ± 1.053075 | -0.004480 | -0.001071 | 3/6 |

私有相对 adapt_only 的唯一 rescue 是 seed17 的 bulk query `5125`，恢复了原模型已经正确、被续训误伤的预测。seed29 两臂 OA/BA 相同，但存在 5 次错类之间的变化；seed43 两臂 top1 完全相同。不能把均值微升写成三顺序都获得识别收益。私有 source_mix 对 baseline 的 4 次 rescue 涉及 4 个查询，7 次 harm 涉及 6 个查询；相对 control 的 6 次 top1 变化中只有 1 次改变对错。

OSS2 相对 adapt_only：seed17 rescue/harm=1/1，seed29=0/2，seed43=2/3。三顺序 OA 一次持平、两次下降；BA 两次上升、一次下降，均值仍下降。公共 source_mix 对 baseline 的 rescue/harm=10/14，分别为 10/14 个不同查询。NLL/Brier 相对 control 略改善不等于识别收益，相对原完整模型二者仍变差。

## 类别组成与收益位置

| 数据 | 类别 | 查询数 | mix - baseline rescue/harm | mix - control rescue/harm |
|---|---|---:|---:|---:|
| private | bulk_carrier | 25 | 0/1 | 1/0 |
| private | chemical_tanker | 70 | 2/6 | 0/0 |
| private | crude_oil_tanker | 16 | 2/0 | 0/0 |
| private | general_cargo / container / tug / offshore | 12 / 22 / 7 / 8 | 各 0/0 | 各 0/0 |
| private | fishing | 0 | 无样本，recall=null | 无样本，recall=null |
| oss2 | bulk_carrier | 87 | 0/9 | 0/4 |
| oss2 | general_cargo | 23 | 1/1 | 0/1 |
| oss2 | product_chemical_tanker | 29 | 3/3 | 2/0 |
| oss2 | container_ship | 89 | 6/1 | 1/1 |

这些是三顺序事件累计，不是额外独立样本。私有 BA 微升和 OA 下降可追溯到类别净得失与不同分母；公共收益不足以抵消 bulk 等类别损失。原图中的相似局部形状跨类存在，不能用一个形状故事解释全部收益或退化。

## 新诊断的解释边界

主代理的 [共同偏移诊断](../diagnostic_summary/SHARED_SHIFT.md) 显示：去掉 softmax 无效的每查询全类平移后，12 个训练端点的 query-common offset 解释本轮 logit 变化能量的 92.3984%–97.1437%。OSS2 source_mix seed43 的 common-only 重建复现 228/228 top1，包括全部 11 次翻转。私有 source_mix seed17 为 158/160，一致覆盖实际 12 次翻转，但仍有额外不一致，common-only OA/BA 不等于真实端点。

因此目前证据更支持“本轮更新主要改变共同类别竞争偏置”，尚不能声称 source_mix 学会了更有效的查询特定参考关系。该比例仅描述**训练诱发的 centered-logit delta**，不描述整个 M3，也不是分类正确率或分类因果贡献的解释比例。共同偏移从开发预测均值估计，虽然未用标签，仍是事后诊断，不是部署候选。

主代理独立完成的 [norm_control 汇总](../diagnostic_summary/SUMMARY.md) 未加入主要 arms。私有 norm OA/BA=44.166667/35.339105，mix=44.375000/35.866316；公共 norm=58.333333/49.534456，mix=57.894737/49.242794。它说明等范数缩放不能在两数据集一致复制 mix 的全部结果；也不支持把 mix 的改动归纳为已验证的覆盖收益或一个已解决的机制。这里引用主代理单表，没有重跑或追加训练。

## 文件与查看记录

精确逐 seed/逐类结果在 `private/` 和 `oss2/` 下的 `metrics.csv`、`paired.csv`、`paired_means.csv`、`class_tradeoffs.csv`、`events.csv` 与 `summary.json`。最终运行回执为 `run_all_receipt.json`，两数据集三顺序 complete，pending 均为空，PyTorch 未导入。

原图事实、最终选图入口、原始路径索引及查看范围见 [CASE_NOTES.md](CASE_NOTES.md)。6 项针对性 CPU 检查通过，包括 ID 重排、缺样本、不借旧预测、四状态/错类变化、三顺序重复事件计数和真实 query/ref 原始像素保持。

本子任务只修改新目录中的 `analysis.py` 和 `analysis/`；无 GPU、无 Git、未修改主代理训练或诊断脚本。原完整模型保留，研究目标未达成。
