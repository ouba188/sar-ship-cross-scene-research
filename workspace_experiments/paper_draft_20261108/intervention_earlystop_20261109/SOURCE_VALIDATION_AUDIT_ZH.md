# 当前 OSS2 class_s17 源端早停队列来源核查

日期：2026-11-09。范围：只读当前完整组合的入口、历史划分、训练订单及回执；未运行 GPU、未训练、未重新推理、未使用目标指标选参。唯一新增文件为本报告。下述路径均相对于 `E:/临时会话/`，行号对应本次读取版本。

## 结论与本轮决策

本轮采用 **source_adapt241 中的适配阶段留出105／训练130**，不使用 selection194。该留出按过境和船只与本轮适配训练隔离，但当前教师已在历史训练中见过这105条；因此只能检验、选择当前适配过程，不能称为全流程独立源验证或教师未见泛化验证。

194条没有进入当前所选 full_s17 → aggregate_17 → small_class_s17 → class_s17 的 OSS2 梯度训练链。其完整集合却不满足当前“平台＋绝对轨道”过境隔离：69条与719条源训练共享 S1A_010568；余下125条满足样本、MMSI和该过境定义下的隔离。125条也不等于一份从未用于研究的源保留集，已作为扩围评价并写入现有论文。

不能仅因旧字段叫 `selection` 就断言它天然是目标测试：其最初来源是2026-10-23 fold0的 validation包，原协议明确称其为源验证。反过来，也不能据此悄然改写当前论文中的评价用途。若未来明确重设协议，125条可作为有历史开发暴露的验证集复用，但须说明用途变更，并不能再把相关125/503条结果当作独立于该选择过程的评价证据。本轮已明确不走此路径。

在本次追踪到的现有四类、可运行源清单范围内，排除本轮禁用的 validation228，不存在可直接替代105条方案的、四类齐全且教师未见的源验证队列。此结论不扩大为“全项目所有原始数据都无候选”。

## 当前权重链与194条的训练暴露

当前配置证据：`m1_guided_m2_20261106/current_version.json:4` 指定 arm=class、默认seed17；第8行指定 `class_s{seed}/final.pt`，第9行指定 `m1_coverage_20261101/small_class_s17/final.pt`；第19行指定 M3 `m3_frequency_increment_20261028/source_hard_task_20261101/oss2/aggregate_17/last.pt`。`current_model.py:10` 的加载入口执行该配置。

| 阶段 | 实际训练输入与落盘证据 | selection194是否训练 |
|---|---|---|
| 初始 full_s17 / fit | source_fit478；训练回执160步、8轮；固定32参考来自source_fit | 否 |
| 初始 full_s17 / adapt | source_adapt241；训练回执96步、6轮；教师目标与这241条逐ID对应 | 否 |
| M3 aggregate_17 / last.pt | source_fit478＋source_adapt241=719；episodes.json有120批、719个不同查询；实际抽样参考701个不同记录，也均在719内 | 否 |
| M1 small_class_s17 / final.pt | groups.small=719；orders.json有180批、4314次暴露、719个不同记录，与194交集为0 | 否 |
| 新 M1-guidedM2 class_s17 / final.pt | groups.small=719；orders.json有180批、4320次暴露、719个不同记录，与194交集为0 | 否 |
| 旁支 expanded_class_s17 / final.pt | groups.expanded=913；orders.json有180批、4228次暴露、913个不同记录，包含全部194条 | 是，但不是当前选中的上游 |

### 初始 full_s17

- 目录：`scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029/`。
- `run.py:137` 的 train_one 在第138行按阶段选择 source_fit 或 source_adapt，第139行取得对应记录及references；第143行 fit从共享init初始化，adapt加载本臂fit模型。
- 第162行开始的适配目标读取及逐ID断言，将适配教师目标限定在source_adapt；第174行按该records生成训练批次，第191行反向、第192行更新。`select_one`第227行同样只读取source_adapt和references。
- `training/full_s17/fit/training.json:18` 记载478条、160步、8轮；`training/full_s17/adapt/training.json:18` 记载241条、96步、6轮，两者均声明validation_selection=false。
- 共享初始化为 `sar_m123_current_20261027/weights/init.pt`；该目录 `README_ZH.md:66` 将其标为尚未完成源训练的初始化。本核查结论针对上述当前OSS2训练链，不宣称穷尽所有预训练及历史研究暴露。

### M3 aggregate_17 与选中的 M1 上游

- `m3_frequency_increment_20261028/source_hard_task_20261101/oss2/model_task.py:32` 的 records只组装fit、adapt、validation、references；第38行将train限定为前478＋241条。selection根本不在此入口的rows内。
- 同文件第62行加载初始 `training/full_s17/adapt/model.pt`。`train_task.py:24` 获取上述train，第45行开始四轮训练；每批查询与episode参考均从train取得，实际episodes落盘记录覆盖索引0至718，未越入其他集合。
- `m1_role_separation_20261101/run_pilot.py:26` 的 anchor加载aggregate_17/last.pt。`m1_design_position_20261101/experiment.py:33` 从该anchor取carrier；`m1_covariance_20261101/run.py:72` 继承该网络；`m1_coverage_20261101/experiment.py:184` 起继续继承并构建当前M1。
- 注意 `aggregate_17/receipt.json:10` 中 selected_epoch=0描述的是另存的selected.pt；当前配置和anchor实际使用 **last.pt**，不能据该字段误称当前M3没有训练。

### records与实际训练索引的区别

- `m1_coverage_20261101/experiment.py:25` 的records读取六种manifest；第28行把fit478、adapt241、validation228、selection194、evaluation378拼成1519条。
- 第31行small只含fit＋adapt；第32行expanded才追加selection。第211行以groups[cohort]指定训练索引，第229行生成orders，第234行起仅迭代这些orders计算损失与更新。
- selection在统一rows中对应947至1140；仅被读取、缓存或无梯度计算初始难度，不等于进入small的梯度更新。初始难度权重也只在ids_train内使用（第224至227行）。
- `m1_guided_m2_20261106/study.py:21` 明确加载small_class_s17；第51行的缓存注释说明selection只缓存、不拟合；第206行train_ids=groups.small，第213行保存orders，第214行起按这些索引训练。
- 以上判断同时依据代码与实际落盘orders/episodes，未只凭目录名判断。expanded旁支的194训练暴露不应错算到当前small上游，也不能抹去其历史研究暴露。

## selection194的原始来源及过境重叠

1. `scene_aware_classification_20261018/oss2_distribution_transfer_20261023/prepare_feasibility.py` 中splits将27个“平台＋绝对轨道＋datatake”采集分为六包；fold0的bag0是test，bag1是validation，其余四包为候选train，并排除验证／测试MMSI。
2. 对旧 `membership.csv` 按sample_id关联：本次194条全部属于fold=0、pass_bag=1、role=validation。`MODEL_PROTOCOL_ZH.md:17` 明确alpha由源验证balancedNLL选择，第25行明确仅源验证选训练点。由此不能将“selection”字段本身作为目标测试身份的充分证据。
3. `scene_aware_classification_20261018/model_closeout_20261023/oss2_m123_replication_20261026/build_stack_split.py:15` 将bag1映射为selection；第121行source_members仅取旧role=train，再拆分fit_train、fit_validation、utility_train。`DATA_PROTOCOL_ZH.md:45` 也明确“原validation对应selection”。因此194并非新造的source_fit/source_adapt留出。
4. `final_oss2_replication_20261029/converter.py:60` 原样保留role，按protocol映射split；没有把194转换成source_adapt。`protocol.py:16` 的角色映射仍为selection→selection。

实际元数据交集如下，未使用分类结果筛选：

| 比较 | 结果 |
|---|---|
| selection194与small719的记录ID交集 | 0 |
| MMSI交集 | 0 |
| 平台＋绝对轨道交集 | S1A_010568，共享轨道上的selection为69条 |
| selection的共享轨道datatake | S1A_010568_00FB6C，69条 |
| small的同轨道datatake | S1A_010568_00FB64，38条 |
| 剔除共享轨道后 | 125条、125艘船、4个轨道，四类42/10/11/62 |

剩余四轨道为 S1A_007418、S1A_008818、S1A_011035、S1A_011311。当前M3的 `model_task.py:47` 以pass_group前两段定义orbit；这解释了旧按完整datatake字符串报告的隔离，与当前更粗过境定义下出现重叠之间的差别。

历史用途也有明确证据：`m3_frequency_increment_20261028/source_hard_task_20261101/oss2/internal_contribution/expanded/FINDINGS_ZH.md:15` 将194与378作为新增扩围队列，第16行说明69条共享轨道和125条严格子集；`paper_draft_20261108/MANUSCRIPT_ZH.md:420` 已将125、378、503列为相关的扩围评价集合。故本轮保留原用途，不将125重命名为未经开发的源验证。

## 是否还有可直接复用的干净源验证

仅对上述既有四类stack_membership做了元数据级候选核查，不扫描全项目数据、不重新检测图像。旧role=train共有1196条；剔除当前训练719的ID、MMSI及orbit后剩280条，全部落在旧fit_validation角色：

- 228条是本轮明确禁用的validation228。
- 47条双极化均无合格OBB；另2条同样不可运行且有旧身份预留排除原因。
- 另有3条可运行，但原规则将其MMSI预留给utility_train中最终无合格OBB的对应观测，所以它们没有进入正式manifest。这3条全是class2：000425（S1A_006718）、003354与003801（S1A_010218），完整ID前缀均为pub2_opensarship2_。

这3条既不构成四类验证，又位于旧fit_validation采集范围，并需改变已有身份预留排除规则；本轮不恢复它们。现有正式、可运行、四类完整的源队列，没有另一个满足本轮禁用228和不改194用途边界的教师未见候选。

## 本轮105／130划分的可复用事实

仅用source_adapt241元数据统计，四个过境全部覆盖四类：

| 过境 | 观测数 | class0/1/2/3 |
|---|---:|---|
| S1A_006346 | 20 | 6/4/3/7 |
| S1A_008060 | 32 | 3/3/4/22 |
| S1A_011268 | 105 | 38/10/24/33 |
| S1A_011618 | 84 | 36/9/15/24 |

按“最大且覆盖全类”预选S1A_011268得到105条、105艘船。其余136条中排除与留出同MMSI的6条，最终训练130条、130艘船、3个过境，四类43/16/22/49。排除的完整ID：

```text
pub2_opensarship2_006225
pub2_opensarship2_006346
pub2_opensarship2_006347
pub2_opensarship2_006359
pub2_opensarship2_006378
pub2_opensarship2_006409
```

留出与source_fit478无MMSI重叠；与固定32参考无MMSI或orbit重叠。上述105条全部属于旧source_adapt，已参与初始full_s17的adapt及后续M3、M1、新M2的719训练，所以不应把当前拆分描述成补建了全流程独立验证。

本轮主代理已采用的协议为：LR=3e-5/1e-5 × 普通/干预 × 3个适配seed；候选epoch0至6，仅用105条的四类平均NLL选择；目标228/378仅在选择固定后读取。balancedNLL在此为各类NLL均值再等权平均，不因训练／留出类别数量不均改变四类权重。适配训练样本、样本权重及干预筛选配额应以130条训练子集为准，留出仅用于早停选择。

用户在本次核查结束前告知：首seed四条轨迹的源hold balancedNLL均从第1轮变差，均选择epoch0，目标尚未读取。这是主代理提供的进度，本报告未重新读取或验证这些新实验曲线；不将其写成所有seed的完成结论。

## 核查边界

本报告证明的是当前选中OSS2训练链的数据入口、实际训练索引与可复用队列的身份／过境关系；没有进行新实验、模型性能复验或全项目预训练审计。没有修改正式模型、清单、协议、论文或Git仓库，也没有扩大本轮实验网格。既有目标评价暴露作为历史事实保留，本次不使用其结果选学习率、epoch或留出集合。
