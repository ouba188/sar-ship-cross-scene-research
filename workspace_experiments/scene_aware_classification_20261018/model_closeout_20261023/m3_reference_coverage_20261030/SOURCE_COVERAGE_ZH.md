# M3 源参考覆盖核查

2026-10-30，独立 CPU 支线。只读源清单、代码、缓存头和源行的 available/slot_masks；未加载模型、训练、使用 GPU、查看原图或读取 3075 / OSS2 evaluation 标签。本文件描述输入覆盖，不评价新实验效果。完整清单留在原本地位置，归档由主代理统一处理。

## 会改变设计的结论

1. **自建 general_cargo 的 VH 同类参考只有 1 个。** refs16 每类 2 船、2 过境，但 class2 的 22755 仅 VV，VH 只有 6913。同船排除后，source_fit 的 184 条有效 VH 查询中，6913 无同类合法 VH 参考，其余 183 条各只有 1 个；source_adapt 的 22 条该类有效 VH 查询也各只有 1 个。这是该极化同类覆盖不足，不是全部参考失效。source_fit 内已有 154 艘船、55 次过境的有效 VH，可直接扩大候选。
2. **OSS2 全库覆盖 9 次过境，掩盖了类内缺口。** refs32 的四类过境数为 3/7/3/7；bulk 和 product_chemical_tanker 分别遗漏 6 次有源样本的过境。每类 source_fit 都存在覆盖全部 9 次过境、MMSI 互异且 VV/VH 均有效的 9 条组合。这是候选可用性的事实，不意味着按过境选样必然提高分类。
3. **不要先取每艘船的第一条观测再考虑极化。** 自建 fishing 的 132 艘可用 VH 船中，首行去重只留下 121 艘有效 VH；chemical_tanker、tug 各损失 1 艘。OSS2 首行去重还会在 bulk、chemical、container 各丢失 1 艘可用 VV 船。保留全体源观测作为候选、选中后禁选相同 MMSI，可以保留这些选择机会。
4. **两套预算含义不同。** 自建 2→4 条/类可增加独立船和过境；OSS2 8→16 条/类只有 9 个源过境，16 条不可能代表 16 次独立过境。更多参考的收益应与同预算随机组比较；不能将参考条数等同独立场景证据。
5. **自建精确缓存只支持当前 source_fit 候选直接复用。** 1205 条 fit、16 条原参考均已写入 float32 prefix；136 条 source_adapt 全部不在这份精确缓存的有效行收据内。数组行号合法不代表该行已经计算。源适配若需要新缓存，应沿主代理现有原生输入流程计算。

## 来源与选择方式

路径缩写：`ROOT = E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023`，`RELEASE = E:/临时会话/sar_m123_current_20261027`。

- 自建：`backbone_m123_layout_20261026/layout_pilot.py:143` 的 train 从 fit_train 按类、按清单顺序 drop_duplicates(mmsi)，用单个 `default_rng(17)` 逐类无放回抽 2 船。没有显式极化、港口或过境约束。后续 joint/separate/selective 管线继承固定银行；`sar_m123_standalone/tools/export_current.py` 导出到 RELEASE/manifests/references.json。实际 16 条为 16 船、13 过境、9 港口；15 双极化，1 仅 VV。
- OSS2：`oss2_m123_replication_20261026/run.py` 复用 OSS1 的 `train_m3.py:251`。源 fit 的有质量槽位 layer1 特征拟合 32 维 PCA（random_state=17）；对存在船体槽位的候选按 MMSI 保留首行，再按类对 body descriptor 做 KMeans(8, random_state=17, n_init=10)，选各中心最近且未重复的真实样本。此旧代码还曾按查询挑 top2，但**当前完整模型直接使用全体32条参考**，不能把旧 top2 流程当作当前接口。
- `final_oss2_replication_20261029/converter.py` 按旧 m3_results/reference_bank.csv 的 obs_id 恢复记录并保留顺序。refs32 为 32 船、9 过境；27 双极化、2 仅 VV、3 仅 VH。public.py 中“每类前8条 chosen”用于旧修复尺度校准，**不是 references.json 的选样来源**。
- 两库均确认为 source_fit 子集，不增加独立监督样本。独立船按 MMSI 定义；过境按 pass_group 定义。本次源清单每条观测的 (MMSI, pass_group) 都不同，但同船跨过境重复存在。
- `current/data.py:pair_mask` 排除同 id 和同 MMSI，不排除同 pass_group。模型随后受极化和槽位有效性约束。不要为了“独立”悄悄新增同过境排除，改变既有任务；可将跨过境数量作为描述性诊断。

## 实际源端规模

自建 source_fit 为 1205 观测 / 1004 船 / 88 过境 / 19 港口；1036 双极化、169 仅 VV。source_adapt 为 136 / 128 船 / 20 过境，128 双极化、8 仅 VV。

| 自建类号/类别 | fit观测/船/过境 | fit有效VH船/过境 | 原参考VV/VH数 | 原参考港口数 |
|---|---:|---:|---:|---:|
| 0 bulk_carrier | 367/306/75 | 304/75 | 2/2 | 2 |
| 1 fishing | 285/232/16 | 132/15 | 2/2 | 1 |
| 2 general_cargo | 193/162/57 | 154/55 | 2/1 | 2 |
| 3 chemical_tanker | 170/141/33 | 135/33 | 2/2 | 1 |
| 4 container | 69/62/26 | 62/26 | 2/2 | 2 |
| 5 crude_oil_tanker | 66/55/23 | 54/23 | 2/2 | 2 |
| 6 tug | 34/29/22 | 28/21 | 2/2 | 2 |
| 7 offshore | 21/17/11 | 10/10 | 2/2 | 2 |

fishing 原参考均在 Ningbo-Zhoushan，chemical_tanker 均在 Jebel Ali，源候选分别覆盖 6、12 港口。最小类 offshore 仍有 10 个双极化、不同船且不同过境的可行组合，2→4 的预算不存在候选数量障碍。

OSS2 source_fit 为 478 / 448 船 / 9 过境；444 双极化、11 仅 VV、23 仅 VH。source_adapt 为 241 / 235 船 / 4 过境；227 双极化、5 仅 VV、9 仅 VH。清单没有独立港口字段，不能把缺失值统计成“一个港口”。

| OSS2类号/类别（原八类号） | fit观测/船 | 原参考过境数 | 原参考VV/VH数 | 源双极化观测数 |
|---|---:|---:|---:|---:|
| 0 bulk_carrier（0） | 138/135 | 3 | 7/8 | 128 |
| 1 general_cargo（2） | 47/45 | 7 | 7/7 | 41 |
| 2 product_chemical_tanker（3） | 95/91 | 3 | 7/7 | 83 |
| 3 container_ship（4） | 198/177 | 7 | 8/8 | 192 |

OSS2 fit 查询同船排除后，各极化仍至少有 6 个同类合法参考（container 至少7），不存在自建 class2 VH 的零同类情况。其问题是多样性和代表性待验证，不能复述成同类参考缺失。

## 可以直接定位的补充候选

`source_candidate_examples.csv` 只列12条源端可行例子，不是新银行或推荐排序。例子优先补原参考未涉及的港口/过境，并用已有 ship_axis 长度元数据展示变化；没有按开发/最终测试表现筛选。

- general_cargo：33211（Ningbo-Zhoushan）、10173（Richards Bay）、7175（Santos），均双极化，与原参考船不同。
- fishing：15610（Shanghai）、3242/3293（Antwerp-Bruges，不同过境），可避免只代表 Ningbo-Zhoushan。
- chemical_tanker：26982/31087（Singapore，不同过境）、35257（Santos），可补 Jebel Ali 之外的候选。
- offshore：2141（Port Hedland）、3187（Santos）、30434（Singapore），均双极化。已有 VV ship_axis 长度约38.05/23.93/16.07像素；原13000/13276约12.20/11.87像素。只是检测/轴元数据，未看原图，不推断船只实际尺度、物理原因或必然收益。

上述样本均在已写入的 source_fit 精确缓存内；CSV 给出缓存行号，原图与几何从 source_fit 对应 id 获取，不复制完整清单。

## 缓存索引和字段

### 自建

源记录只有 id/MMSI/pass/port/label/VV/VH，没有当前缓存行号。用 `backbone_m2_pilot_20261026/table.csv` 的 **obs_id列与CSV位置**建立 id→位置映射，其他列无需加载。本次仅物化这一列。不能把 source_index、row_index、id 数值当位置。

1205 fit 行散布于 0..1984；16 references 为 93..1913；整个数组 N=1986。以 `spatial_m12_refit_20261029/cache/complete.json.original_indices` 为精确 prefix/masks 已写入范围（共1365行）。新参考取 fit 的任意候选均已缓存；adapt不在该集合。

| forward_from字段 | ROOT下文件 | dtype | 每行形状 |
|---|---|---|---|
| prefix | spatial_m12_refit_20261029/cache/prefix.npy | float32 | 2,512,20,20 |
| masks | spatial_m12_refit_20261029/cache/masks.npy | float32 | 2,3,160,160 |
| valid | pool_replacement_20261026/cache/valid.npy | float16 | 2,1,160,160 |
| available | backbone_m2_pilot_20261026/cache/available.npy | float32 | 2 |
| slot_masks | joint_readout_20261026/cache/slots.npy | float32 | 2,20,10,10 |
| frequency | joint_readout_20261026/cache/energy.npy | float32 | 2,20,3,10,10 |
| local_slot_masks | separate_evidence_20261026/cache/slots.npy | float32 | 2,20,20,20 |
| local_frequency | separate_evidence_20261026/cache/energy.npy | float32 | 2,20,3,20,20 |
| field2 / field3 | selective_m123_20261027/cache/field2.npy、field3.npy | float32 | 2,3,20,20 / 2,3,10,10 |

### OSS2

`final_oss2_replication_20261029/cache/<forward_from字段>.npy`，N=1519，全部 float32，形状与上表相同。记录 **indices.table_index** 才是数组索引。fit为0..477、validation为478..705、adapt为706..946、selection为947..1140；按字段取行，勿用 obs_id/manifest_position/target_manifest_row 替代。

旧 `run.Cache()` 读取 all.json 和含标签 complete.json，protocol.cohorts() 读取 evaluation；本支线均未调用。只加载所需源清单、缓存头以及源索引切片。public.py 的 Cache 是 CUDA 实现，也未调用。

两套源清单的极化非空状态与 available 一致；source_fit 的各有效极化均有正船体槽位质量，未发现“清单有效但源船体槽位全空”。这是输入合法性检查，不代表全部区域匹配质量良好。prefix只是冻结 layer2 输出，不是可跨模型复用的学习后M3 reference value；改变layer2前参数或输入预处理会使其失配。

## 对当前实验的具体建议

保持原库、同预算随机与覆盖库配对，候选只来自 source_fit，选中后排除相同 MMSI；源标签用于组织类内候选，不允许按查询真值挑类。记录实际类×极化×过境覆盖即可，不新增一轮全面审计。

覆盖距离如果包含“查询与自身/同船参考”，其改善可能含推理时被 pair_mask 屏蔽的零距离。将这一距离称为源描述覆盖，另用**合法异船配对后的实际同类权重/有效同类数**回答关系是否改善；不能用覆盖均值直接证明分类受益。独立船数也不保证过境多样性，上表可用于解释结果，不要求强加新硬约束。

source_fit本身已用于原模型训练；source_adapt也已用于适配，不是独立验证。OSS2 validation228用于源开发、selection194可用于冻结后复核，历史暴露说明沿用 OSS2_HANDOFF.md。当前开发重复不读取最终测试标签，不能把开发涨点或关系权重提升称为最终目标达成。

## 文件与证据边界

- source_coverage_by_class.csv：12行类级规模、极化、过境、首行去重损失。
- oss2_source_class_pass.csv：36行类×源过境计数，可直接看出遗漏。
- source_candidate_examples.csv：12条候选元数据，不是完整清单。
- source_coverage_summary.json：汇总、合法同类计数和执行范围。最大“独立船且独立过境”数通过源元数据二分匹配获得，不使用特征或预测。

未重做上一轮M3分数、最终测试案例或根因报告。当前可确认的是可用候选和覆盖缺口；是否改善实际关系与完整分类，由主代理本轮冻结替换/匹配训练及后续独立CPU分析回答。
