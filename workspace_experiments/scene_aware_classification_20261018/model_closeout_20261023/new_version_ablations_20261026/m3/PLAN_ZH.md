# 新版 raw_summary 的 M3 算子替换

2026-10-26。本支线只修改本目录的 run.py、PLAN_ZH.md；运行产物也写本目录。主线负责摘要、区域输入实验以及最终原图分析和归档。这里不重训上游，不新增网络或频谱分支。

## 最小对照

固定 raw_summary base、两公开源双向四分类、原角色、种子17/29/43、每类8条共32条参考银行，以及各查询每类检索2条的实际参考。不是每个查询使用32条局部匹配。

| arm | 相似度 | 区域内对应 | metric适配 |
|---|---|---|---|
| both | 原PairMetric加权cosine | 原可微Sinkhorn | 从原both初始化，100步上限 |
| learned_fixed | 原PairMetric加权cosine | 固定同槽对应 | 同初始化、目标和100步上限 |
| cosine_soft | 普通cosine | 原可微Sinkhorn | 无可学习metric参数 |
| cosine_fixed | 普通cosine | 固定同槽对应 | 无可学习metric参数 |

普通cosine为现有源PCA空间token上的标准内积除范数，取消PairMetric通道权重；不是把固定对应改名为cosine。它只删除matcher的conditioned weights，空间频域gate仍可访问频域描述，因此不是整个模型去频域的消融。未使用源白化距离，因为真实代码本来就有可直接隔离的cosine基准。PairMetric仍是原96→4投影及8→32权重，合计640参数。

固定对应在body(8)、side(8)、end(4)各自内部使用同一OBB槽位。各区域质量归一化为a、b，T_ij=1[i=j] min(a_i,b_i)，余量不参与实槽匹配，视作进入dustbin。无输入相似度或可学习对应参数；保持原(sum T*sim, sum T)两统计量、两参考平均和VV/VH输出维度。此替换同时改变对应自由度与实槽匹配质量，不能宣称只隔离了一个抽象的“可微性”。没有完整槽间对齐时，固定对应允许未匹配，而不是重新分配到别槽。

Sinkhorn保持原eps=.1、60迭代及dustbin。both中的“学习匹配”指metric学习经Sinkhorn改变对应，solver本身没有另一套独立参数。cosine_soft是固定相似度驱动的软匹配，不声称训练了额外匹配网络。

## 配套训练与动态门

所有组使用utility_train优化和拟合读出scaler，selection选步及alpha；源evaluation标签训练时置-1。learned_fixed和both直接复用原aligned.fit_metric：原both_metric.pt初始化，经冻结的旧raw_summary polar+frequency决策反传，AdamW lr=.003、weight_decay=.01，最多100步，每步按24查询累积完整源损失。以原半样本/半类别平衡CE为训练目标；0/5/10/15/20/30/50/75/100步按exact J严格改善选最早步。首步梯度精确0时沿用原提前退出，记录零梯度，不伪称完成有效适配。cosine两组没有可优化metric，不做虚假的100次更新。

之后每组VV及centeredVH分别重新初始化并训练200步，重新拟合scaler；相同seed、初始化规则、AdamW、源样本次序及选择准则。alpha使用原0,.1,...,1网格及原tie规则；分别训练空间41维和空间频域185维gate各100步，均从同一B_fixed锚定。每组输出bare/B_fixed/spatial/frequency，24组配置、72个M3终点。JointHead不参与最终决策，故不增加其无关训练；polar每次单独设seed，不影响both重放。

三个门终点共享按frequency实际决策适配的metric，这是同一metric条件下的门信息/配套读出比较，不是每种门各自重新适配matcher。所有门前11维随当前base/delta重算，后174维沿用同参考的空间/频域描述；空间仅取前41维。

主线已确认保留版六组frequency选步均0，预测与B_fixed精确相同。本轮不能声称旧版已获得动态频域收益。gate_activity.csv逐源/目标、seed、arm记录所选步、相对B_fixed的权重变化数/最大值、分数变化数/最大值、预测翻转数及精确相等标记。非零checkpoint不自动代表门生效；权重变化也可能因polar为零而不影响分数。

## 必要重放与数据来源

以下路径以 E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023 为根：

- m3_aligned_training_20261026/run.py：复用FrozenDecision、fit_metric；既有seed目录的aligned_metric.pt、selection.json和source/cross_predictions.npz是保留新版主对照。both从原初始化重新适配，要求相同选点、local误差≤2e-5及B_fixed/frequency最终分数精确一致；不读取旧target指标决定训练。
- raw_region_bypass_20261026/downstream/raw_summary/{direction}/seed{seed}/：固定bare、原polar/frequency checkpoint、scaler、available及gate描述。新版保留base与此bare必须精确一致。
- public_m123_cross_dataset_20261026/{direction}/protocol.json：定位source_m3和target_cache；target_table.csv通过tr.read_table(labels=False)读取。
- {source}_m123_replication_20261026/stack_table.csv：源角色与身份；source_m3的local.npz、projection.npz、references.npz、seed{seed}/scalers.pt和both_metric.pt：源PCA、32参考、原metric标准化及初始化。
- public_m123_cross_dataset_20261026/normalization_repair_20261026/source_{source}/seed{seed}/predictions.npz与{direction}/seed{seed}/predictions.npz：旧local重放。源cache缺obs_id，通过对应direction/table.csv的source original_obs_id核对；目标obs_id和refs显式核对。
- oss1_m123_replication_20261026/train_m3.py及E:/临时会话/sar_m123_standalone/sar_m123/m3.py：Matcher、PairMetric、Sinkhorn、PolarHead、原fit_head/fit_gate；全部只读导入。

旧local与冻结决策器先在源selection重放，检测缓存/算子不一致；新版both重放检测适配、scaler及配套训练漂移。失败时停止并定位差异，不绕过检查生成已完成标记。固定参考检查限定于角色/身份/索引一致性，不新造数据划分。

## 运行

已在项目ClearSAR环境通过语法和真实导入检查，CUDA设备为NVIDIA GeForce RTX 5050 Laptop GPU。2026-10-26经主代理追加授权启动训练，session 80296；预计10–20分钟，结果以完成标记为准。在本目录运行：

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -B run.py train --device cuda
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -B run.py evaluate
```

训练命令仅训练、保存预测，不调用evaluate；必须完成两方向三种子四组共24配置和both重放后，才写predictions_saved.json。evaluate先验证该标记，再读取目标标签。运行中会持续写selections.csv、gate_activity.csv、checks.json，可由主代理查看进度。重新执行train会重跑本支线并移除旧完成标记，不修改历史资产。

输出预测含obs_id、refs、local、gate_raw、delta、门权重及所有分数。evaluate生成metrics/summary、paired、per_class、case_index供主线分层及原图回看；case_index保留CC/WW/救回/误伤、MMSI、pass_group、类分差与local变化，通过target_table回查原chip路径，通过refs查源stack_table和实际同类/混淆类支持chip。case_index不是已完成原图分析，不用代表例子估计发生比例。

该轮沿用已开发过的目标集及重复使用的源selection，不能称独立盲测。训练预算、零梯度情况、gate是否实际生效及both重放均应随结果报告；负结果保留。最终原图回看和GitHub同步由主线统一执行。

## 执行回执

24组训练及全部预测已完成，耗时769.22秒；训练session 80296与随后独立evaluate的session 55012均以exit code 0结束。evaluate在predictions_saved.json写出之后执行。六组both的选点、B_fixed及frequency分数均与保留版本精确一致；原local及冻结决策器重放记录在checks.json。

全部24个空间gate选0步且分数等于B_fixed。both六组frequency也均选0步且精确等于B_fixed。frequency实际激活仅出现在OSS1 seed29的learned_fixed/cosine_fixed（40步，目标各9个top-1翻转），以及OSS2 seed43的cosine_soft（15步，目标4个翻转）；翻转数不是净救回数，效果见paired.csv。

both的metric选点为OSS1 0/0/0、OSS2 0/75/100；learned_fixed为OSS1 0/0/0、OSS2 0/0/75。选0步不等于未执行优化，除OSS1 seed17因首步精确零梯度退出之外，学习组均执行100步预算，实际过程保存在各组metric_curve.csv。

以下仅为三种子frequency终点OA/BA百分数，不作区间或原图因果结论：

| arm | OSS1→OSS2 | OSS2→OSS1 |
|---|---|---|
| both | 54.1075 / 52.5581 | 61.4187 / 51.3820 |
| learned_fixed | 54.2516 / 52.8395 | 63.4093 / 52.0925 |
| cosine_soft | 54.1075 / 52.5581 | 60.8759 / 51.9651 |
| cosine_fixed | 54.2516 / 52.8395 | 63.6627 / 52.2475 |

完成的文件：metrics.csv、summary.csv、paired.csv、per_class.csv、gate_activity.csv、case_index.csv、selections.csv、checks.json及各组预测/checkpoint。原图、区间、主报告和Git仍由主线接续；evaluation_complete.json明确raw_review_pending=true。主run_readout.py只读检查未发现维度或训练接口实质问题，未修改。
