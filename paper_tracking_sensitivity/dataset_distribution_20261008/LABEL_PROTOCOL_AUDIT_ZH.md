# 六数据集标签空间与实验范围核对

核对日期：2026-10-08。对象是 `handoff_budget_sweep_20261008` 已完成的固定联合模型预算回放，不是各数据集所有历史版本。本文仅核对现有代码、清单和缓存；没有训练或改变样本筛选。完整数值分布与图表由同目录主报告提供。

## 1. 先回答“几分类”

| 本轮数据组 | 模型输出类别数 | 目标全集实际存在类别 | 目标任务与样本范围 |
|---|---:|---|---|
| 自建24港 | 8 | 单港3、6、7或8类；合计8类 | 23个可评估目标港，共38,027条；完整24港语料38,091条 |
| F_0921 | 8 | 8类 | 一个全局目标任务，123,489条 |
| OpenSARShip 1.0 | 8 | 8类 | 本地原生GRDH中细粒度标签兼容子集，1,349条 |
| OpenSARShip 2.0 | 8 | 8类 | 本地原生清单中细粒度标签兼容子集，3,745条 |
| PACS | 7 | 每域7类 | 4个留一域任务，合计9,991条 |
| Office-Home | 65 | 每域65类 | 4个留一域任务，合计15,588条 |

依据：[本轮任务清单代码](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_handoff_budget_sweep.py:38)、[冻结任务记录](E:/Hermes/paper_tracking/sensitivity/handoff_budget_sweep_20261008/POLICIES_SAVED.json:1)及其中每个 `manifest_path` 对应的真实标签列。

**正确表述是“统一8类输出空间下的跨港/跨数据集船型识别”，不能写成“24个港口均为完整8分类”。** 目标港只有3类时，模型仍能输出其他5类，输出不存在的类别仍算分类错误；并没有用查询真值预先把预测空间缩减成3类。

## 2. SAR统一8类映射

| class_id | 原始规范类别名 | 中文名称 |
|---:|---|---|
| 0 | bulk_carrier | 散货船 |
| 1 | fishing_vessel | 渔船 |
| 2 | general_cargo | 杂货船 |
| 3 | product_chemical_tanker | 成品油/化学品船 |
| 4 | container_ship | 集装箱船 |
| 5 | crude_oil_tanker | 原油船 |
| 6 | tug_towing | 拖轮/拖带船 |
| 7 | offshore_supply | 海工补给/支持船 |

类别ID来自[映射定义](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/opensarship2_labels.py:26)，并已与[24港源清单](E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv:1)中 `class_id,class_name` 去重结果逐项一致。中英文名称不能按字母顺序重新编号。

### 2.1 24港目标类别覆盖

| 目标全集存在类别数 | 港口 | 缺失类别 |
|---:|---|---|
| 3 | Antwerp-Bruges、Busan、Fujairah、Hamburg、Ningbo-Zhoushan、Qingdao、Rotterdam、Shanghai | 3、4、5、6、7 |
| 6 | Port Hedland | 1、4 |
| 7 | Mombasa、Newcastle | 7 |
| 8 | Callao、Houston、Jebel Ali、Los Angeles、New York、Port Klang、Port Said、Richards Bay、Santos、Singapore、Sydney Botany、Tanger Med | 无 |

完整语料另含Melbourne的64条、3类样本；同样取64条支持样本后没有查询样本，因此未纳入当前23个目标港的预算评估，不是按效果剔除目标港。排除条件见[原始继承实验](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_residual_expert.py:33)。上表表示支持集划分前目标全集的覆盖；每轮查询集是否仍包含某个极少数类，应看该轮排除支持行后的计数。

外层按港口留出；源候选是其他港，再剔除与目标港共享非空MMSI或产品ID的源行。源数据因此不是机械的“38,091减目标港行数”。见[外层划分](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/build_oof_predictions.py:142)和[身份排除](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/build_oof_predictions.py:121)。已只读取本轮23个 `joint_policies.npz` 的 `source_rows` 并回查源清单：各目标任务保留的源集合均有8类。

### 2.2 F_0921范围

本地芯片索引共385,650行。历史配对clean档案共有165,880行，clean规则记录为 `land_px_224 == 0`；在该已有集合上，按 `class_raw` 做明确细粒度船型映射，最终保留123,489行。165,880是映射前clean候选数，**不是本轮目标查询集规模，也不是本轮使用165,880个目标标签**。

映射只读 `class_raw`，不把历史粗粒度 `class8` 字段当成本轮8类标签；不增加类别配额，也不依据分类效果选样。来源：[映射实现](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/f0921_cohort.py:99)、[样本纳入实现](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/f0921_cohort.py:119)、[原始映射统计](E:/Hermes/paper_tracking/sensitivity/f0921_transfer_20261007/label_mapping_summary.json:1)。

本轮F_0921是**一个汇总目标任务**，64个支持标签在123,489行中全局抽取，不是每个港区簇分别抽64个。其地理簇、最近港字段可做分布描述，不能据此把本轮结果改称“多个独立留一港实验”。

源集合从38,091行中剔除8,374行，保留29,717行、全部8类。源排除使用所有165,880个clean候选的非空MMSI和对应瓦片采集候选产品，范围刻意先于细标签过滤，因此 `source_split.json` 的 `target_clean_rows=165880` 与本轮目标123,489不矛盾。见[源排除代码](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/f0921_cohort.py:203)、[源划分记录](E:/Hermes/paper_tracking/sensitivity/f0921_transfer_20261007/source_split.json:1)。这是数据范围说明，不是对跨域困难原因的归因。

### 2.3 OpenSARShip 1.0范围

当前原生XML读取器识别出41个产品目录、5,673条船舶记录，包括22个GRDH产品和19个SLC产品；GRDH共3,030行。当前实验选GRDH作为目标队列，再保留可明确映射到上述8类的1,349行。SLC不进入本轮目标评估，但其身份参加源集合排除。

未纳入的1,681个GRDH记录为粗粒度标签1,384、非支持标签229、歧义标签68。不能把粗标签 `Cargo` 强行映射为 `general_cargo`，也不能把 `Tanker` 自动当作原油船。双极化文件按场景与XML中心坐标配对；本轮兼容GRDH样本均有文件，不按文件列表次序贴标签，不按质量或准确率再筛。

来源：[原生读取和筛选代码](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/opensarship1_data.py:16)、[GRDH选择](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/opensarship1_data.py:51)、[映射统计](E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/label_mapping_summary.json:1)。这里的41是实际解析到的产品目录数，不据此断言用户磁盘中所有文件、极化文件或该公开数据集全球发行版本都只有41景。

本轮源头部训练保留37,343行，排除与全部原生GRD+SLC目标身份重叠的748个源行；见[源划分实现](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_opensarship1_transfer.py:37)、[源划分记录](E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/source_split.json:1)。

### 2.4 OpenSARShip 2.0范围

本轮从本地原生manifest的9,120行出发，保留3,745个细粒度标签兼容样本，排除5,375行：粗标签3,997、非支持标签1,224、歧义标签154。不是用户之前列出的5,006片历史特征缓存，也不是常见的3类共享标签协议。

映射依次查 `label_elaborated`、`label_primary`、`label_rough`，取第一个非空字段；若精细字段非空但不可映射，不继续用更粗字段“救回”。3,745行中3,720来自elaborated、25来自primary；没有质量或干扰水平过滤。来源：[标签规则](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/opensarship2_labels.py:79)、[实际选择](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/extract_opensarship2_transfer.py:76)、[映射统计](E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/label_mapping_summary.json:1)。

源排除对照的是全部9,120行身份，不仅是最终3,745行；保留源36,996行，剔除1,095行。见[源划分实现](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_opensarship2_transfer.py:31)、[源划分记录](E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/source_split.json:1)。

## 3. PACS与Office-Home的类别来源

类别名直接读取本地Parquet schema中的 `huggingface.info.features.label.names`，整数ID沿用Parquet `label` 原值，未按本次目标频率重新筛类或编号。提取方式见[元数据读取](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/generic_transfer_data.py:49)。完整机器可读名称表在[标签协议JSON](E:/Hermes/paper_tracking/sensitivity/dataset_distribution_20261008/label_protocol.json)。

PACS依次为：0 dog、1 elephant、2 giraffe、3 guitar、4 horse、5 house、6 person。四域为art_painting、cartoon、photo、sketch。

Office-Home是原始65类，ID 0为Alarm_Clock，64为Webcam，全部名称见JSON；不是历史试验中的全局top-8子集。四域为Art、Clipart、Product、Real World。

两者均以一个域为目标、其他三个域为源；4次循环覆盖四个目标域。每个域的所有本地样本保留，不采用现成训练/测试折再截取目标子集。源/目标行选择见[域划分代码](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/generic_transfer_data.py:73)。两个表征来自同一RGB图像的两个编码器，不是两种传感器。

本轮联合模型PACS/Office-Home属于已标注的 `retrained_rgb_reference` 比较组；不能与历史旧编码器实验拼为同一个数值基线。任务来源见[本轮RGB库存](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_handoff_budget_sweep.py:63)。

## 4. 支持样本、缺类与BA的准确解释

每个目标任务每次使用64条支持标签，原始种子0至4无放回均匀抽样，无类别配额；五次是支持集抽样，不是五个独立数据集，也不是五次网络初始化。支持行不进入该轮查询指标。见[PACS/OH抽样](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_generic_transfer.py:110)、[SAR抽样](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_opensarship2_transfer.py:145)、[本轮继承支持集](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_handoff_budget_sweep.py:75)。

必须区分三种“缺类”：源训练没见过、整个目标队列不存在、目标队列存在但64标签中没抽到。这里的门控使用的是第三种情况：专家预测类别必须在该轮支持集中出现，才有交接资格；这不代表该类没有源知识或从模型输出中被删除。见[门控资格](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_handoff_budget_sweep.py:105)。

Office-Home总计65类而支持标签仅64条，所以一轮不可能覆盖全部65类；在不均衡分布与重复抽到同类时还会缺更多类。主统计给出的本轮实际覆盖为平均39.45类、范围34至46类。不能把这个设置称作“每类一个目标标签”，也不能把缺少目标支持标签等同于零样本分类。

OA按查询样本数加权；BA是该轮查询真值实际存在类别的召回率均值，目标查询中不存在的类不作为零召回强行计入分母，但错误预测到这些类仍会降低真实类召回率。定义见[指标函数](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_generic_transfer.py:64)。各轮先求指标，再按任务等权聚合，因此24港的BA不是把全部港样本拼接后算一个8类BA。

这组报告描述的是本地既有开发语料与固定实验协议，不能写成全公开数据集标准排行榜、完全无目标标签域泛化、或者尚未接触的独立确认集。以上限定不改变本次预算曲线的内部对比，只规定它回答的问题范围。
