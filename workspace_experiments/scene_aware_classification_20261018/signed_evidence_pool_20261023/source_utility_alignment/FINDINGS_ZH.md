# 固定 signed-only：训练目标与选择目标 2x2

2026-10-23。36 条训练轨迹、72 个 selected heads 已完成。唯一架构是 113 参数 signed-only；原 native 特征、source OOF、划分、种子、200 步与 AdamW 均不变。Protocol 在运行前写盘；所有源选点保存后统一评价测试集。

## 结论与主次

**主候选固定为 jointCE+J，测试三种子净 +45/+59/+34，OA 与 BA 均高于原 A。** NLL 选择得到 +68/+69/+49，但只作次要分析，不按测试更高改选主候选。这是目标校正实验，不是新架构/创新性已成立，也不是独立外部确认。

原 balancedCE+balancedNLL 的 18 头 epoch 与权重精确重现（state 最大差 0），原测试 score 最大差 8.88e-16、预测完全一致。由此原负结果没有被改写。

| 训练 / 选择 | seed17/29/43 净变化 | 平均 OA | 平均 BA | NLL | balanced NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| A | 0/0/0 | 81.650571 | 80.265736 | .526008 | .548927 | .285327 |
| balancedCE / balancedNLL | -3/-5/-33 | 81.410594 | 80.600005 | .516207 | .527840 | .283148 |
| balancedCE / J | +5/-3/-6 | 81.627158 | 80.413815 | .524496 | .543797 | .285369 |
| jointCE / balancedNLL（次要） | +68/+69/+49 | 82.739245 | 80.589552 | .493849 | .533422 | .268173 |
| jointCE / J（主候选） | +45/+59/+34 | 82.458297 | 80.533011 | .509135 | .545004 | .275744 |

主候选三 seed 的 OA/BA 分别为 82.440737/80.503636、82.686567/80.665464、82.247586/80.429932。平均相对 A 为 OA +0.807726 pp、BA +0.267274 pp；不能把种子当独立数据集。

## 两个因子分别改变什么

保持 balancedNLL 选择时，改 jointCE 的逐 seed 净变化相对原单元为 +71/+74/+82；保持 J 选择时为 +40/+62/+40。仅在 balancedCE 轨迹上换 J，变化为 +8/+2/+27，但 pooled 仍近于 A；在 jointCE 上换 J，相对次要 NLL 单元为 -23/-10/-15。完整配对见 `factor_pairing.csv`。

因此本轮更强的证据支持“训练效用改变了固定特征的可用读出”，而非“只要按分类指标挑 checkpoint 就够”。选择目标仍影响具体预测与类别取舍，不能由这个 2x2 宣称普通校准/其他简单头也做不到；下一轮匹配对照另目录进行。

源验证汇总变化（selected 减 A，OA/BA 为 pp）：

| 单元 | ΔOA | ΔBA | Δbalanced NLL |
|---|---:|---:|---:|
| balancedCE / balancedNLL | -.592332 | -.156253 | -.024211 |
| balancedCE / J | +.188765 | +.350345 | -.010615 |
| jointCE / balancedNLL | +.520732 | -.506259 | -.016163 |
| jointCE / J | +.826661 | +.198184 | -.005423 |

J 为 0.5 OA+0.5 BA 的比例值，用正确计数的有理数严格比较；含 epoch0，平局保留最早，不按 NLL 打破平局。源验证选点并没有用到上表测试结果。validation pooling 不是重新选择依据。

## 数据与适用条件

测试仍为三类 5695 目标、4312 MMSI、163 pass、15 港，类数 bulk/fishing/general=3170/1147/1378。source OOF 只针对基础读出，修正头在源训练样本上拟合；整流水线不称 OOF。队列反复用于开发，不称外部独立确认。

每 seed 平均类别净变化：

| 单元 | bulk | fishing | general |
|---|---:|---:|---:|
| balancedCE / balancedNLL | -49.33 | -2.00 | +37.67 |
| balancedCE / J | -12.00 | +3.33 | +7.33 |
| jointCE / balancedNLL | +84.33 | -4.67 | -17.67 |
| jointCE / J | +59.33 | -7.00 | -6.33 |

主候选 recall 为 bulk 87.32913（A 85.45741）、fishing 85.78901（A 86.39930）、general 68.48089（A 68.94049）。整体收益主要来自散货，不是三类同时变好。六折主候选平均净变化 +3.67/+11/+4.67/+7.33/+7.67/+11.67；这是折内 seed 平均，不是每个 head 都赢。港口主要正项为 Shanghai +13.67、Singapore +7.67、Port Said/Qingdao 各 +6.33；Port Hedland -.67、Rotterdam -.33。完整 class/pass/port/fold 见对应 CSV，未把港口相关性当机制。

## 源验证的小样本与选点波动

每折 validation 585–967 例、26–28 pass；最少类别仅 108 例。单例从错变对使 J 改变约 0.081–0.240 pp，离散分辨率不可忽略。每轨迹检查 epoch0 及 40 个五步间隔点，没有增加搜索网格。

- balancedCE+J 选 epoch0 为 **8/18**，jointCE+J 为 **2/18**；两个 NLL 单元均为 0/18。零头不当作训练成功涨点，也不删除。
- J 最佳值与次高不同值的间隔中位数：balancedCE 0.0932 pp，jointCE 0.0485 pp；距最佳 J 不超过 0.1 pp 的候选点数中位数分别 1.5 与 4。jointCE+J 所选 epoch 范围 0–165，中位数 40；说明存在平台和小差选点。
- 对每个已选头的 validation 预测做 1000 次固定配对 pass/MMSI 块重采样，两种都报告。jointCE+J 的 ΔJ 区间分别均有 13/18 跨零，balancedCE+J 均为 18/18。每单元 pass 抽样累计 3 次缺类（已排除并记录），MMSI 为 0。
- 这些区间条件于已选模型，**不纠正同一 validation 最大化 J 的选择乐观性**，不代表 checkpoint 或超参数选择本身稳定；未用区间再选模型。test 没有再次调参。

## 原图：八张已实际查看

固定 seed17，各单元按 ID 取救回/误伤/CC/WW 首例，去重为 8 张，四单元均覆盖四种结果、无缺失。VV/VH 分开，原尺寸和 OBB 保留，图像区与四行结果区不重叠。以下只描述可见事实与固定真类-最强错类分差，不命名尾迹/旁瓣/物理因果。

- `0032439` Newcastle，64/64，斜横向较宽亮结构，VH 下方有框外亮延伸。A margin -.106423，四单元都救回；主候选 +.385597。
- `0033132` Ningbo-Zhoushan，64/64，细长斜亮结构，两极化周围纹理不同。原 NLL 单元 +.103352 -> -.034096 误伤；主候选本折 epoch0 原样保住 A。jointCE+NLL 则增至 +.342668。
- `0030906` Melbourne，72/80，近纵向长亮结构，VV 周围较亮。四单元 CC；主候选 epoch0 保持 +1.772555，不将其算作模块新贡献。
- `0030918` Melbourne，64/72，斜长亮结构及横向重复亮响应。两个 balancedCE 单元 WW；jointCE 两种选择都救回，主候选 margin -.152923 -> +.150350。固定原图下训练目标改变结果，但不证明网络识别某种物理现象。
- `0056041` Port Said，72/64，斜横亮结构，交叉方向存在亮延伸。前三单元误伤，主候选 +.035807 -> +.148776 保持正确；J 选择在此有正向取舍。
- `0053996` Port Klang，64/64，斜长亮结构与较暗周围。jointCE+NLL 将 general 误判 bulk，margin +.125685 -> -.014272；主候选 epoch0 保持 A。
- `0031200` Mombasa，80/64，纵向不均匀亮结构、背景纹理明显。四单元 WW；主候选 -.616256 -> -.592948，只略改善未救回，局部矩不能解决所有错误。
- `0052252` Port Hedland，64/64，斜长亮结构、暗周围。前三单元 CC；主候选 +.275495 -> -.006803 误伤。不能只展示 J 保住前例而隐去此例损失。

## 成本与交付

36x200=7200 优化步，CPU 训练合计 25.86 秒，最终评价 3.57 秒，总流程 31.24 秒；分析/绘图约 4.19 秒。没有重抽特征、新骨干/reader 拟合或额外轨迹。`selection_frozen.json` 记录测试评价前选点已固定；`replay_checks.csv` 保留原 18 头重现；最终 CSV 分数重构最大误差 1.78e-15。

文件：`run_alignment.py`、`analyze_alignment.py`、`PROTOCOL_ZH.md`；`selection_curves.csv`（含 epoch0）、`selections.csv`、`models/*.pt`、`predictions.csv`、`validation_predictions.csv`、`pooled.csv`、`role_pooled.csv`、`factor_pairing.csv`、`class_effects.csv`、`acquisition_effects.csv`、`source_counts.csv`、`validation_intervals.csv`、`review_cases.csv` 和 8 张 PNG。模型仅本地，未 Git。主候选不因次要单元测试更高而替换；本轮证据支持目标匹配的重要性，尚不支持局部矩或 signed 限制具有独立必要性。
