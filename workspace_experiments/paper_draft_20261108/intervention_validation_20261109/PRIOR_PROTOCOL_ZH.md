# 当前模型干预指导训练：历史协议只读核对

日期：2026-11-09。范围：核对历史脚本、结果及 OSS2 清单/缓存元数据，提出最少配套实验。未导入或执行训练代码，未运行 GPU，未改正式文件或实现当前干预前向。本文件为唯一新增交付物。以下“已确认”来自脚本与保存记录；新训练建议尚未执行。

## 结论

建议仅补当前 M1-guidedM2 + aggregateM3 的干预筛样适配与同类别配额普通筛样适配，使用现有 source_adapt **241 条**、固定32条参考，三组配对共 **6 次适配**；适配前结果复用。历史 source_fit 是 **478 条**，两者合计719，不能把旧筛中136条或旧八分类训练136条当作当前适配池大小。

旧实验的“普通选样”不是只对抽中样本训练分类头：全部适配样本参与分类损失，只更换证据约束的样本掩码；真正更新的是 ResNet layer3 最后一个瓶颈块的 **conv1/conv2/conv3 三个卷积权重**，共1,114,112参数。新版若仅训练头或新读出，应准确称为读出适配，不能作为局部骨干适配交付。

旧教师选择记录仅作历史证据。当前模型更换了 M1/M2/M3 和读出，必须用当前完整教师重算原图/干预分数、竞争类、筛选结果及新配额。不能混入旧 full_s17/full_s29/full_s43 教师分数，也不能认为同为 seed17 就是同一模型。

## 1. 历史普通选样到底做了什么

指定 expansion/FINDINGS_ZH.md 对应的是冻结推理扩围，而非新增训练：3075观测、1950船、18港、85过境、八分类、16参考。其 run.py:model_paths 实际读取以下历史权重：

- 筛样组：intervention_evidence_distill_20261027/one_sided/training/original_s{17,29,43}/adaptation.pt。
- 普通组：intervention_mask_selection_20261027/training/matched{101,202,303}_s{17,29,43}/adaptation.pt。
- 这里选用的是 **original** 单侧下界组，不是 intervention 单侧下界组。

intervention_mask_selection_20261027/run.py:20 的规则为：对全部136条源适配样本，按真实类别枚举候选；每类选取数量等于干预筛中该类数量，使用 numpy.default_rng(101/202/303) 在该类全部候选中**无放回均匀抽样**。不按难度、置信度、船只或过境再匹配，也不排除与干预组重叠。每个随机掩码在不同训练次序间保持固定。

干预筛中101/136条，八类配额为 `[31,2,22,25,5,10,3,3]`；三个普通掩码与干预组重叠80、79、80条。所有136条都参加CE；仅floor损失乘不同掩码。不能将普通组解释成“去除干预选中样本后的剩余样本”，也不能将101当作分类训练集总量。

历史干预筛选先要求结构干预相对原图的NLL改善、固定原图最强竞争类分差改善，以及相对配对均值控制的额外分差改善均达到0.01，再选择额外分差最大的候选。evidence_distill/run.py:37-49 又在实际教师分数上检查真类NLL下降及相对当时最强竞争类的分差上升。原始候选与实际用于该路径的教师输出有区别，不能只读旧 choice.useful 而忽略最终 useful。

### 实际损失与系数

旧 selected-original 与 matched 普通组均使用：

`L = weighted_CE + 0.1 * weighted_KL_correct_teacher + lambda * masked_margin_floor`

- 原图教师为冻结教师；KL方向是 `KL(teacher || student)`，仅原图教师判对的样本启用。
- 对每个非真类k，floor为 `relu((teacher_y-teacher_k)-(student_y-student_k))^2`，对C-1个竞争类平均。
- 原图教师分差就是floor；**不使用 max(原图分差, 干预后分差)**。干预用于挑选施加约束的样本，不直接成为此比较中的蒸馏分差目标。
- CE、KL、floor均除以整个batch的权重和；floor不按筛中子集重新归一化。类别权重为 `N/(C*n_class)`。
- `lambda=0.3683066666126251`，普通组直接复用，不按各随机掩码重标定。
- 该系数最初由源端前两个batch、共32条的梯度标定得到：`0.25 * norm(mean_grad_CE_keep) / norm(mean_grad_evidence)`；保存梯度范数11.9652977和8.12183094。run_floor调用时标定的是intervention/max-floor目标，随后original-floor和matched也共用该系数，故不能声称每个最终训练组的初始梯度比都精确为0.25。
- 这不是旧feature-MSE指导训练：teacher_features.npz在这里还承担目标加载/诊断，但最终训练目标没有feature-MSE项，也没有干预图像增强CE项。

### 更新范围与预算

intervention_local_feature_20261027/run.py:29-41 先冻结全模型，仅开启 `backbone.model.layer3[-1].conv{1,2,3}`。当前公共实现将其具体命名为 `backbone.model.layer3.5.conv{1,2,3}.weight`；分类头、M1/M2/M3参数、其他骨干和BN均冻结，模型保持eval模式。冻结参数不等于切断它们对可训练输入的反向传播。

evidence_distill/run.py:29、153-176及original_s17/complete.json确认：AdamW，LR=3e-5，weight_decay=0.01，6轮，batch16，136条每轮9步，**共54步**；每轮末尾8条也训练。没有学习率调度或梯度裁剪。三个seed17/29/43只改变适配顺序，初始化相同；不可称为三个独立预训练模型。

扩围主比较是每个训练次序内，selected指标减三个matched指标平均，再平均三个次序，不是概率集成。额外OA -0.0542、BA +0.0824个百分点，NLL -0.020376、Brier -0.002503；历史结果主要支持概率质量改善，不能预写新版必然提升top-1。

## 2. OSS2清单实数

直接解析 final_oss2_replication_20261029/manifests/*.json，按label计数；四列类别依次为散货、普通货、成品油/化学品、集装箱。

| 清单 | 观测 | 不同MMSI | 不同pass_group | 四类观测数 |
|---|---:|---:|---:|---|
| source_fit | 478 | 448 | 9 | 138 / 47 / 95 / 198 |
| source_adapt | 241 | 235 | 4 | 83 / 26 / 46 / 86 |
| validation | 228 | 223 | 4 | 87 / 23 / 29 / 89 |
| selection | 194 | 193 | 5 | 64 / 18 / 40 / 72 |
| evaluation | 378 | 369 | 5 | 132 / 26 / 83 / 137 |
| references | 32 | 32 | 9 | 8 / 8 / 8 / 8 |

source_fit与source_adapt的ID及MMSI交集均为0；32参考全部来自source_fit。719是478+241；本轮建议不扩入selection194。

旧OSS2 protocol.py:19规定fit为8轮/batch24/160步，adapt为6轮/batch16/**96步**。241条每轮16步，最后一批1条，不drop_last。旧适配仍是LR3e-5、AdamW wd0.01、同一floor系数，更新上述三个卷积权重。旧fit另有模块参数LR0.001，不能误移到adapt阶段。新719条读出训练的180步/batch24/headLR1e-5是另一阶段协议，不能与局部骨干96步混写。

关键范围说明：m1_coverage_20261101/experiment.py:25-34 明确把source_fit与source_adapt合成small719；当前m1_guided_m2_20261106/study.py也在small上训练。因此241条可继续作为既有适配子集，但**已被当前教师训练使用，不能再声称是教师未见过的源留出集**。这是源训练内的监督筛选与继续适配，不是独立泛化证据。

## 3. 选择记录与缓存复用边界

### 已存在的旧OSS2选择记录

目录 `final_oss2_replication_20261029/teachers` 已有full/plain各三份JSON及receipt。full记录每份241条，教师均为该seed的 `training/full_s{seed}/fit/model.pt`：

| 旧seed | 最终useful数 | 四类选中配额 |
|---|---:|---|
| 17 | 136 | 42 / 22 / 25 / 47 |
| 29 | 137 | 42 / 20 / 32 / 43 |
| 43 | 142 | 42 / 19 / 35 / 46 |

记录含id、label、original、altered、choice、selected_intervention。可借用其格式和追溯信息，**不可直接复用其分数、竞争类、useful或配额**。也不可只把旧chosen操作在新教师上复评就称为完整重选：新教师可能选中其他候选，需要评估现有完整候选集及配对控制。普通掩码也应按新配额重新生成，可复用随机种子规则。

2026-11-08的当前interventions.py已经改成image-only候选，不重算每个候选的显式频率字段。因此旧记录还对应不同的执行历史；不要用当前脚本版本倒推旧缓存已经具有新版语义。旧plain既换模型结构又关指导项，也不是当前“同模型、同floor、随机掩码”的普通选样对照。

### 可复用的内容

- 原始VV/VH、检测几何、native160裁剪/补零与有效区域、标签映射、清单身份、参考ID和合法配对规则：沿用当前既定语义。
- final_oss2/cache/complete.json记载1519条、float32、native_size160、stop2、init为sar_m123_current_20261027/weights/init.pt；cache/prefix.npy为冻结layer2输出，另存masks、valid、available等。只要当前使用的该段权重、输入及索引一致，这类**待训练层之前**的原图缓存可复用；本次未跑前向验证，不能宣称跨版本数值一致已经验证。
- 新版M1可能插入layer2内部，因此旧“layer2结束后的prefix”未必是当前实时干预/适配的合法入口。优先让主代理沿当前前向所需的边界使用已有early_x/early_identity等缓存，不为节省计算绕过当前M1。
- m1_guided_m2_20261106/cache下的x、response、features、score是固定上游输出。它们可用于未改输入/未改上游时的基线回放或读出训练；**骨干适配时不能把依赖被更新卷积的末端features当常量输入**。凡位于被更新参数之后的量，均需在对应前向中重新计算并保留梯度。
- 同理，原图缓存不能作为干预图的特征；只有在前缀权重、干预图及操作规则相同的情况下，干预前缀才可复用。本次未发现可直接认定为当前完整教师的既成干预选择缓存。

### 当前教师身份必须明确

current_version.json指定joint-head `class_s{17,29,43}/final.pt`，上游 `m1_coverage_20261101/small_class_s17/final.pt`，M3为 `source_hard_task_20261101/oss2/aggregate_17/last.pt`，固定32参考、无显式频率分支。该阶段的M3是既有固定残差路径，未与新M1/M2联合重训。

因此当前教师是这个组合的完整打分函数，不能仅拿RegionalGuidance头当教师，也不能换回旧standalone full模型。主代理的干预前向需说明当前固定anchor/参考路径哪些量随查询图像变化、哪些按设计固定；不要顺便改成另一种联合M3架构。零干预时应回到当前对应seed的已有打分，之后再冻结教师生成指导数据。

## 4. 最少配套实验建议

以下是建议，不是已经执行的训练。

1. **只设两个新增训练组。** guided使用当前教师重算的最终useful掩码；ordinary在同一个241条适配池中逐类无放回均匀抽取相同数量，只替换floor掩码。两组的CE、KL、原图教师floor、全部241条训练样本、类别权重、参考及完整前向保持一致。不要只对选中样本做CE，不额外增加clean-only、旧模型复训或新M3消融。
2. **三组配对、共6次适配。** 对当前class_s17/29/43各自复制同一初始完整模型给guided与ordinary；各用自己的冻结完整教师重算241条选择。随机掩码seed101/202/303预先分别对应17/29/43，组内使用相同训练次序。这样同时提供三个当前读出状态和三个普通抽样的配对结果，但不能单独估计“教师seed变化”和“抽样变化”的方差，也不能声称复现了历史3×3全交叉设计。不要在见到评价结果后挑选随机掩码。
3. **训练范围以实际局部骨干为准。** 最小延续方案只开放当前主干对应layer3.5的conv1/2/3，冻结头、当前M1/M2/M3参数及BN；在主代理完成的当前前向里让梯度真实穿过后续固定运算。冻结anchor支路仍按原设计冻结，不能误更新同名anchor权重。对两个组采用完全一致策略。若主代理最终选择仅训练头/读出，必须明确这只回答读出适配，未完成局部骨干协议。
4. **从旧OSS2适配预算直接起步。** 241条、6轮、batch16、96步、AdamW LR3e-5、wd0.01；使用 `CE+0.1*KL_correct+0.3683066666126251*original_floor`。这复用已有超参数，未声称其对新教师最优；不要按两组分别重标定lambda，也不要沿用719条阶段的180步却称“相同适配预算”。若出现明确优化失衡，先依据训练项、梯度和更新幅度定位，再共同调整且保留原结果，不预设扩展扫参。
5. **复用适配前基线，做配对收尾。** 开发228、评价378、32参考与各seed对应基线保持一致。报告guided-minus-ordinary及两者各自相对适配前的OA/BA/NLL/Brier、逐seed结果与实际样本/船/过境单位。当前上游共用seed17，这不是三套独立骨干。已有评价集有研发暴露，不称新盲测，也不按其结果选种子、系数或停止轮数。

若当前重选导致某类全选、全不选，普通组在该类必然相同；照实记录可区分的样本数量即可。若所有类掩码都被配额唯一确定，两组已无筛选差异，不应训练重复组制造“比较”。这只需从新选择记录判断，不增加独立实验。

### 足够且必要的针对性验证

- 零干预的当前完整前向与对应已保存基线分数/预测配对：检出教师或执行路径混用；不一致则先修主代理前向，不拿不一致的教师开始筛选。
- 记录新guided与ordinary的逐类配额、身份和源范围：检出抽样池/配额/顺序错配；错了只重建该掩码或索引。新选择结果使用最终useful，而不是仅内部choice.useful。
- 正常训练中记录三个实际卷积的梯度、训练前后权重差及冻结项状态：检出计算图被缓存切断、更新到错误参数或BN状态漂移。至少在一次真实反向传播与训练结束检查，不因初始original-floor为零而误判指导无效；CE先推动模型后floor才可能活跃。若梯度始终为零，不能只凭AdamW权重衰减造成的差值宣称有效学习。
- 结束后按已有规则回看持续正确/错误、救回/误伤的代表原始VV/VH及实际同类/混淆类支持chip。新案例从该次真实预测中选，复用原图和已有面板来源，记录ID、seed、分差与可见事实；无需全量审计或自动重训。

## 5. 证据定位

以下路径相对 `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/`，行号为本次所读脚本的位置：

- `intervention_selection_expansion_20261027/FINDINGS_ZH.md`及`run.py:model_paths`：扩围结果、实际读取的selected-original与matched权重。
- `intervention_mask_selection_20261027/run.py:20,32,82,88`及`selection_receipt.json`：普通抽样、配额、相同原图floor及固定系数。
- `intervention_evidence_distill_20261027/run.py:37,69,125`、`run_floor.py`、`floor_objectives.py`、`one_sided/loss_scale.json`及`one_sided/training/original_s17/complete.json`：教师有效性、标定、实际训练与下界含义。
- `intervention_local_feature_20261027/run.py:29,160,166`及`intervention_guided_m1_20261027/objectives.py:60`：三个可训练卷积、类别权重与CE/KL来源。
- `final_oss2_replication_20261029/protocol.py:19`、`run.py:34,137,223`、`manifests/*.json`、`teachers/full_s*.{json,receipt.json}`、`cache/complete.json`：实际OSS2数量、96步适配、各自旧教师与缓存边界。

当前版本证据位于工作区：`m1_guided_m2_20261106/current_version.json`、`current_model.py`、`study.py:30,92,196`；`m1_coverage_20261101/experiment.py:25`；`sar_m123_standalone/sar_m123/current/model.py:13,50`、`training.py:14,24`及`interventions.py:71,97`。本次没有重新测量模型前向、梯度或性能，相关验证由主代理在实际实现和训练时完成。
