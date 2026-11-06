# 当前 M1/M2 代码与版本对应

2026-11-06。这里区分正式工程、研究载体和实验候选，不把“已同步源码”写成“已替换正式模型”。

## 用户确定的当前工作版

用户随后明确选择：**M1类别响应引导M2区域读出＋分类头小学习率联合更新**。
选择记录为 `current_version.json`，默认入口为 `current_model.py`。
默认读出使用既有种子17，不按评价集择优；17/29/43三个检查点均保留。
M1和原骨干在本轮固定，仅训练新增区域读出及分类头（头学习率1e-5）。
不采用 `frozen_head/`，不恢复频域，也不改变本轮M3。

```powershell
python -B -X utf8 E:/临时会话/m1_guided_m2_20261106/current_model.py
```

默认命令加载已选权重，在已有开发/评价缓存上重放并核对原结果，不启动新训练。
`load_readout()`提供同一个已训练读出算子的代码入口。`train --seed 17`
使用原联合优化配置；已有完成记录时不覆盖原训练。

这是已选定的**OSS2四分类工作版**，不是把四类权重直接装入八分类完整工程。
旧八分类工程与所有对照保留；选用该版本不改变已有收益的证据范围。

## 代码位置

| 部分 | 相对于 workspace_experiments 的位置 | 内容 |
|---|---|---|
| 正式工程快照 | `sar_m123_current_20261031/sar_m123/` | 本机与同步前仓库的已跟踪Python源码一致，本轮不覆盖 |
| 当前实验 M1 算子 | `m1_covariance_20261101/operators.py`，`RelationFeedback` | 均值支路、类别竞争响应与低秩二阶关系、残差特征调节；保留匹配均值/二阶对照 |
| M1 网络接入 | `m1_covariance_20261101/run.py`、`m1_coverage_20261101/experiment.py` | 当前源719载体在 layer2[3] 的 conv2 后、conv3 前使用 M1，VV/VH各自参数 |
| 原 M2 池化与内部处理 | `sar_m123_standalone/sar_m123/pool_replacement.py`、`current/spatial_m12.py`等 | 当前公共实验载体实际导入的研究运行库；不是自动更新正式工程 |
| 新 M1→M2 实验接口 | `m1_guided_m2_20261106/study.py`，`RegionalGuidance` | 用M1四个候选类别响应在每个几何区域内归一化，聚合layer2最终输出，保留均值和原区域槽位，增量补充分类输入 |
| 结果与实例 | `m1_guided_m2_20261106/analyze.py`、`diagnose.py` | 配对比较、原图/参考图、区域分差分解和实际M2输出接入核对 |
| 冻结头对照 | `study.py --freeze-head`，`frozen_head/` | 同一算子、同批次，只冻结原分类头；结果不如上轮，不作为新默认 |

## 当前实际路径

```text
VV / VH 原尺度 crop-pad chip
  -> 共享 R50 前缀
  -> layer2[3].conv2
  -> 每极化 M1(RelationFeedback)
  -> conv3 + 原残差 -> layer2 输出
  -> 原 M2 几何区域池化 + 新候选类别引导区域摘要
  -> 原深层区域摘要、全局摘要、原M3路径
  -> 原特征槽位上的增量更新
  -> 分类头
```

新摘要在代数上可接入实际浅层M2的pooled输出，已核对与缓存训练一致；本轮尚未把它做成每个bottleneck内的空间回写。M1响应来自conv2后的128维输入，而M2读取conv3及残差后的512维特征，两者同20×20网格。

公共载体不包含私有正式工程的 layer2[2] M2，不应把这里的结果归到含该块的完整私有工程。

## 本机运行入口

在原实验工作区，使用已有数据、权重与缓存：

```powershell
python -B -X utf8 E:/临时会话/m1_guided_m2_20261106/study.py --prepare
python -B -X utf8 E:/临时会话/m1_guided_m2_20261106/study.py
python -B -X utf8 E:/临时会话/m1_guided_m2_20261106/analyze.py
python -B -X utf8 E:/临时会话/m1_guided_m2_20261106/study.py --freeze-head
python -B -X utf8 E:/临时会话/m1_guided_m2_20261106/analyze.py --freeze-head
python -B -X utf8 E:/临时会话/m1_guided_m2_20261106/frozen_head/compare_head_policy.py
```

脚本采用已有检查点时自动跳过已完成训练。库中有历史绝对路径，换机器需按数据和权重位置调整。仓库是源码与证据归档，排除原图数据集、权重、大型缓存和私有清单，不宣称全新clone即可无数据复现实验。

实验的主要依赖链为：`study -> m1_coverage/experiment -> m1_covariance/run -> m1_tuning_position/tune -> m1_design_position/experiment -> m1_role_separation/run_pilot -> source_hard_task/oss2/model_task`；几何与匹配依赖以及研究运行库随本次源码同步。没有改动这些历史算法以改善本轮结果。

## 证据状态

- 上轮更新分类头时：类别引导比均值 OA/BA +0.882/+1.548 个百分点，但相对原模型 -0.088/+0.063，不能称完整系统明确升级。
- 本轮冻结分类头时：类别引导相对原模型 -2.205/-2.077 个百分点，因此不采用冻结头版本。
- 源719/913扩大覆盖的 M1 旧实验也保留，不能只用局部最优声称类别竞争或协方差机制已普遍有效。

同步仅保证相关源码和证据可查，不改变候选的科学结论。
