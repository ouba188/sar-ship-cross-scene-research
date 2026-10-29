# 冻结 M3 联合适配最终结果

2026-10-30。CPU只读分析已完成的预测/收据，无模型加载、训练或GPU调用。joint_low已冻结为原适配流程加M3 output LR3e-5；本轮目标尚未达成，不再调整该候选。原analysis/cases的16张图及CASE_NOTES_ZH.md不改动。

## 保存原版、新control与候选必须分开

saved是NPZ内保存原版端点；control是本轮重新执行原训练策略得到的模型。joint_dev的control s17相对saved最大logit差0.0666008，两条原正确样本31779(化学品→普通货)、34278(集装箱→散货)被误伤。joint_low s17相对新control的2条救回均只是恢复保存原版正确性，不能记为超越保存原版的M3收益。s29的新control又比saved多判对12478，s43正确性无变化。具体数值误差根因未证实，不把训练复现差异命名为新模块效果。

所有actual delta均为候选减对应参照；OA/BA为百分点，NLL/Brier越低越好。均值先按顺序配对，再对三个顺序平均，不是模型集成。SD是顺序间样本SD，不是独立样本置信区间。

| 范围 | 候选 | 参照 | 顺序 | ΔOA | ΔBA | ΔNLL | ΔBrier |
|---|---|---|---|---|---|---|---|
| joint_dev | control | saved | 17,29,43 | -0.208333 | -0.216450 | -0.000061 | -0.000130 |
| joint_dev | joint_low | saved | 17,29,43 | +0.000000 | +0.000000 | -0.001140 | -0.000289 |
| joint_dev | joint_high | saved | 17,29,43 | -0.416667 | -0.284477 | -0.002360 | -0.000256 |
| joint_dev | joint_low | control | 17,29,43 | +0.208333 | +0.216450 | -0.001079 | -0.000159 |
| joint_dev | joint_high | control | 17,29,43 | -0.208333 | -0.068027 | -0.002299 | -0.000126 |
| oss2_evaluation | control | saved | 17,29,43 | +0.088183 | +0.100402 | -0.001187 | -0.000460 |
| oss2_evaluation | joint_low | saved | 17,29,43 | -0.088183 | -0.060827 | +0.000301 | +0.000252 |
| oss2_evaluation | joint_low | control | 17,29,43 | -0.176367 | -0.161229 | +0.001487 | +0.000712 |
| expanded | control | saved | 17,29,43 | +0.010840 | -0.007476 | +0.000499 | +0.000054 |
| expanded | joint_low | saved | 17,29,43 | -0.086721 | -0.074937 | +0.000077 | +0.000017 |
| expanded | joint_low | control | 17,29,43 | -0.097561 | -0.067461 | -0.000423 | -0.000037 |

joint_low在development三个顺序与saved的OA/BA完全相同，只是风险指标微降；在expanded3075三个顺序OA均下降，未达到冻结扩围目标。以下同时保留新control参照，避免把重放漂移混入候选净收益。OSS2结论仅适用于历史四类检测后evaluation378，不能推广为全库或全新盲测。

## 各顺序 actual delta

| 范围 | 候选 | 参照 | seed | ΔOA | ΔBA | ΔNLL | ΔBrier | 救回/误伤 | 翻转 |
|---|---|---|---|---|---|---|---|---|---|
| joint_dev | control | saved | 17 | -1.250000 | -0.853432 | -0.001319 | -0.000281 | 0/2 | 2 |
| joint_dev | control | saved | 29 | +0.625000 | +0.204082 | +0.000563 | -0.000032 | 1/0 | 1 |
| joint_dev | control | saved | 43 | +0.000000 | +0.000000 | +0.000573 | -0.000077 | 0/0 | 0 |
| joint_dev | joint_low | saved | 17 | +0.000000 | +0.000000 | -0.002384 | -0.000963 | 0/0 | 1 |
| joint_dev | joint_low | saved | 29 | +0.000000 | +0.000000 | +0.001341 | +0.001553 | 0/0 | 0 |
| joint_dev | joint_low | saved | 43 | +0.000000 | +0.000000 | -0.002377 | -0.001457 | 0/0 | 0 |
| joint_dev | joint_high | saved | 17 | -0.625000 | -0.204082 | -0.007003 | -0.001724 | 0/1 | 3 |
| joint_dev | joint_high | saved | 29 | -0.625000 | -0.649351 | +0.005360 | +0.004749 | 0/1 | 3 |
| joint_dev | joint_high | saved | 43 | +0.000000 | +0.000000 | -0.005436 | -0.003795 | 0/0 | 0 |
| joint_dev | joint_low | control | 17 | +1.250000 | +0.853432 | -0.001065 | -0.000682 | 2/0 | 3 |
| joint_dev | joint_low | control | 29 | -0.625000 | -0.204082 | +0.000778 | +0.001585 | 0/1 | 1 |
| joint_dev | joint_low | control | 43 | +0.000000 | +0.000000 | -0.002950 | -0.001380 | 0/0 | 0 |
| joint_dev | joint_high | control | 17 | +0.625000 | +0.649351 | -0.005684 | -0.001442 | 1/0 | 3 |
| joint_dev | joint_high | control | 29 | -1.250000 | -0.853432 | +0.004797 | +0.004781 | 0/2 | 4 |
| joint_dev | joint_high | control | 43 | +0.000000 | +0.000000 | -0.006009 | -0.003718 | 0/0 | 0 |
| oss2_evaluation | control | saved | 17 | +0.000000 | +0.000000 | -0.002550 | -0.001009 | 0/0 | 0 |
| oss2_evaluation | control | saved | 29 | +0.264550 | +0.301205 | -0.000685 | -0.000176 | 1/0 | 1 |
| oss2_evaluation | control | saved | 43 | +0.000000 | +0.000000 | -0.000324 | -0.000196 | 0/0 | 0 |
| oss2_evaluation | joint_low | saved | 17 | -0.264550 | -0.182482 | +0.000036 | +0.000084 | 0/1 | 1 |
| oss2_evaluation | joint_low | saved | 29 | +0.000000 | +0.000000 | +0.000234 | +0.000246 | 0/0 | 0 |
| oss2_evaluation | joint_low | saved | 43 | +0.000000 | +0.000000 | +0.000631 | +0.000426 | 0/0 | 2 |
| oss2_evaluation | joint_low | control | 17 | -0.264550 | -0.182482 | +0.002586 | +0.001093 | 0/1 | 1 |
| oss2_evaluation | joint_low | control | 29 | -0.264550 | -0.301205 | +0.000920 | +0.000422 | 0/1 | 1 |
| oss2_evaluation | joint_low | control | 43 | +0.000000 | +0.000000 | +0.000956 | +0.000622 | 0/0 | 2 |
| expanded | control | saved | 17 | +0.000000 | -0.035629 | -0.000069 | +0.000120 | 1/1 | 8 |
| expanded | control | saved | 29 | +0.065041 | +0.028766 | +0.000817 | +0.000030 | 2/0 | 3 |
| expanded | control | saved | 43 | -0.032520 | -0.015567 | +0.000750 | +0.000013 | 0/1 | 4 |
| expanded | joint_low | saved | 17 | -0.065041 | -0.007340 | +0.001156 | +0.000330 | 1/3 | 9 |
| expanded | joint_low | saved | 29 | -0.065041 | -0.081286 | -0.001315 | -0.000229 | 3/5 | 15 |
| expanded | joint_low | saved | 43 | -0.130081 | -0.136186 | +0.000388 | -0.000051 | 2/6 | 12 |
| expanded | joint_low | control | 17 | -0.065041 | +0.028289 | +0.001225 | +0.000211 | 2/4 | 7 |
| expanded | joint_low | control | 29 | -0.130081 | -0.110052 | -0.002132 | -0.000259 | 2/6 | 14 |
| expanded | joint_low | control | 43 | -0.097561 | -0.120620 | -0.000362 | -0.000064 | 2/5 | 12 |

## 新control复现差异

| 范围 | seed | max|logit差| | 救回/误伤 | top1改变 |
|---|---|---|---|---|
| joint_dev | 17 | 0.06660080 | 0/2 | 2 |
| joint_dev | 29 | 0.05017853 | 1/0 | 1 |
| joint_dev | 43 | 0.03411579 | 0/0 | 0 |
| oss2_evaluation | 17 | 0.11101151 | 0/0 | 0 |
| oss2_evaluation | 29 | 0.04195869 | 1/0 | 1 |
| oss2_evaluation | 43 | 0.04753733 | 0/0 | 0 |
| expanded | 17 | 0.11527061 | 1/1 | 8 |
| expanded | 29 | 0.06806278 | 2/0 | 3 |
| expanded | 43 | 0.04162407 | 0/1 | 4 |

所有改变ID、前后logits、类别与正确性见baseline_reproduction.json。极小输入/保存重放误差不能保证训练轨迹与旧训练位级一致；本分析不新增根因推导。

## 已完成的小验证与源验证

semantic与semantic_live仅seed17，学习率分别1e-4与1e-3；live阶段的label是同LR的对照。OSS2源验证为validation228、source_adapt241训练66步，和最后OSS2联合适配96步不是同一实验。

| 范围 | 端点 | OA | BA | NLL | Brier |
|---|---|---|---|---|---|
| semantic | saved | 45.625000 | 34.928108 | 2.532032 | 0.835259 |
| semantic | label | 45.000000 | 34.724026 | 2.533670 | 0.836997 |
| semantic | label_shuffled | 45.000000 | 34.724026 | 2.532891 | 0.836079 |
| semantic_live | saved | 45.625000 | 34.928108 | 2.532032 | 0.835259 |
| semantic_live | label | 41.250000 | 33.673315 | 2.560673 | 0.853653 |
| semantic_live | label_live | 40.625000 | 33.469233 | 2.563617 | 0.853540 |
| semantic_live | label_live_shuffled | 43.125000 | 34.730829 | 2.546083 | 0.843598 |
| oss2_validation | saved | 57.894737 | 50.123534 | 1.397182 | 0.622810 |
| oss2_validation | total | 57.017544 | 48.399396 | 1.383560 | 0.614063 |
| oss2_validation | control | 57.017544 | 48.399396 | 1.386660 | 0.614197 |

| 范围 | 候选 | 参照 | ΔOA | ΔBA | ΔNLL | ΔBrier |
|---|---|---|---|---|---|---|
| semantic | label | saved | -0.625000 | -0.204082 | +0.001638 | +0.001739 |
| semantic | label_shuffled | saved | -0.625000 | -0.204082 | +0.000859 | +0.000820 |
| semantic | label | label_shuffled | +0.000000 | +0.000000 | +0.000779 | +0.000918 |
| semantic_live | label | saved | -4.375000 | -1.254793 | +0.028641 | +0.018395 |
| semantic_live | label_live | saved | -5.000000 | -1.458874 | +0.031586 | +0.018281 |
| semantic_live | label_live_shuffled | saved | -2.500000 | -0.197279 | +0.014051 | +0.008340 |
| semantic_live | label_live | label | -0.625000 | -0.204082 | +0.002945 | -0.000113 |
| semantic_live | label_live | label_live_shuffled | -2.500000 | -1.261596 | +0.017535 | +0.009942 |
| oss2_validation | total | saved | -0.877193 | -1.724138 | -0.013622 | -0.008747 |
| oss2_validation | control | saved | -0.877193 | -1.724138 | -0.010522 | -0.008613 |
| oss2_validation | total | control | +0.000000 | +0.000000 | -0.003100 | -0.000134 |

semantic正确/打乱标签均未优于原版；semantic_live三个端点均低于原版，当前数据不支持语义分支带来分类收益。源验证total/control均OA57.0175、BA48.3994，均低于该源验证baseline57.8947/50.1235；概率风险改善不等于分类目标达成。

## 最终类别变化与状态

计数是三个顺序的事件；不是三倍独立样本。公开四类名称单独定义，未沿用私有八类索引。

| 范围 | 参照 | 类 | 每序n | 原正确事件 | joint_low正确事件 | 救回 | 误伤 | WW改类 |
|---|---|---|---|---|---|---|---|---|
| oss2_evaluation | saved | bulk_carrier | 132 | 232 | 232 | 0 | 0 | 1 |
| oss2_evaluation | saved | general_cargo | 26 | 37 | 37 | 0 | 0 | 0 |
| oss2_evaluation | saved | product_chemical_tanker | 83 | 131 | 131 | 0 | 0 | 1 |
| oss2_evaluation | saved | container_ship | 137 | 296 | 295 | 0 | 1 | 0 |
| oss2_evaluation | control | bulk_carrier | 132 | 232 | 232 | 0 | 0 | 1 |
| oss2_evaluation | control | general_cargo | 26 | 37 | 37 | 0 | 0 | 0 |
| oss2_evaluation | control | product_chemical_tanker | 83 | 132 | 131 | 0 | 1 | 1 |
| oss2_evaluation | control | container_ship | 137 | 296 | 295 | 0 | 1 | 0 |
| expanded | saved | bulk_carrier | 947 | 1865 | 1862 | 2 | 5 | 4 |
| expanded | saved | fishing | 102 | 253 | 252 | 0 | 1 | 0 |
| expanded | saved | general_cargo | 301 | 261 | 263 | 2 | 0 | 1 |
| expanded | saved | chemical_tanker | 803 | 1175 | 1171 | 1 | 5 | 5 |
| expanded | saved | container | 361 | 428 | 427 | 1 | 2 | 3 |
| expanded | saved | crude_oil_tanker | 256 | 289 | 288 | 0 | 1 | 2 |
| expanded | saved | tug | 138 | 31 | 31 | 0 | 0 | 0 |
| expanded | saved | offshore | 167 | 63 | 63 | 0 | 0 | 1 |
| expanded | control | bulk_carrier | 947 | 1867 | 1862 | 1 | 6 | 3 |
| expanded | control | fishing | 102 | 253 | 252 | 0 | 1 | 0 |
| expanded | control | general_cargo | 301 | 261 | 263 | 2 | 0 | 0 |
| expanded | control | chemical_tanker | 803 | 1175 | 1171 | 1 | 5 | 5 |
| expanded | control | container | 361 | 428 | 427 | 1 | 2 | 1 |
| expanded | control | crude_oil_tanker | 256 | 288 | 288 | 1 | 1 | 0 |
| expanded | control | tug | 138 | 31 | 31 | 0 | 0 | 1 |
| expanded | control | offshore | 167 | 63 | 63 | 0 | 0 | 2 |

| 范围 | 参照 | 救回事件/唯一样本 | 误伤事件/唯一样本 |
|---|---|---|---|
| oss2_evaluation | saved | 0/0 | 1/1 |
| oss2_evaluation | control | 0/0 | 2/2 |
| expanded | saved | 6/6 | 14/14 |
| expanded | control | 6/6 | 15/15 |

## 新增原像素面板

仅为expanded3075与OSS2 evaluation378生成joint_low相对saved的救回、误伤、持续对、持续错各1；每例同时记录三个顺序的saved/control/joint_low分数与两套参照状态。不存在的状态明确缺失，不换参照凑数。优先多顺序同状态，其次固定baseline竞争类margin变化绝对值，最后ID/seed；无随机抽样。


| 范围 | 状态 | ID | 代表seed | 同状态顺序数 |
|---|---|---|---|---|
| expanded | rescued | 5045 | 29 | 1 |
| expanded | harmed | 35112 | 17 | 1 |
| expanded | CC | 31698 | 17 | 3 |
| expanded | WW | 22726 | 17 | 3 |
| oss2_evaluation | harmed | pub2_opensarship2_002633 | 17 | 1 |
| oss2_evaluation | CC | pub2_opensarship2_005423 | 17 | 3 |
| oss2_evaluation | WW | pub2_opensarship2_000138 | 17 | 3 |

缺失：oss2_evaluation / rescued，实际对应事件数 0。

入口cases/index.html；cases.json记录真实类别名称、模型分数、参考ID/顺序及缺极化状态，native_sources.json记录绝对原图路径、原始尺寸、crop/pad和OBB来源。复用既有panel/native helper；VV/VH分开，原始像元1:1、无resize或对比度拉伸。参考取实际源库并排除同ID/MMSI及不可用极化；展示同类与实际竞争类子集，不宣称这些是最高权重参考。

新面板均not_reviewed，主代理回看后写FINDINGS；本支线不依据外观编写物理解释。原16张图不重生成。

## 产物

metrics.csv：各端点实际指标；actual_deltas.csv：逐顺序对saved/new control的真实差值及翻转；class_changes.csv：逐类正确/救回/误伤；summary.json：均值、SD、范围及完成状态；baseline_reproduction.json：control重放改变的逐例证据；cases/：最终病例。复跑本目录analyze.py只重建这些结果，不触发训练或调参。
