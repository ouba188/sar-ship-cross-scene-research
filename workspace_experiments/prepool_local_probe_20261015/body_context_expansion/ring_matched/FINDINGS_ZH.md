# 匹配源支持组织与预算：复测结果

## 结论

**匹配后出现一个明确值得保留的局部正信号：Jebel Ali、16个支持。并未得到全港、全预算均有效的选择器；更大的限制是当前合法源任务覆盖不足。**

- 该任务逻辑回归较船体BA +0.9260个百分点、OA +1.0757个百分点；五次支持draw的BA增量全部为正。
- 较固定环带组合BA +0.1819，较同任务同切换数量随机期望 +0.2005；本轮不是仅仅“少用一些环带”。五次draw较随机期望都为正，但这些是同一查询队列的支持重复抽样，不是五次独立训练。
- 固定组合15次救回、10次误伤；选择器保留14次救回、剩5次误伤，即拦住5次误伤、错过1次救回。14次救回涉及12条船，5次误伤涉及5条船（两集合可能同船在不同draw出现）。
- 同样源query/支持类配额下，多采集训练对照只有+0.1001 BA；旧训练方案为+0.2248。支持组织匹配对该局部任务有实测支持，但没有据此宣称它是所有失败的唯一原因。
- 该池没有合格的独立source selection任务，因此这里是**预先固定C和0.5阈值的探针结果**，不是源验证成功选出的部署规则。目标结果未用于调参数，发现的港口适用范围仍属探索性。

## 本轮实际完成的匹配

源端单采集支持、固定查询、每类配额、same/cross共同容量、嵌套B16/B32/B64、五次draw、船只与几何排重均匹配目标组织。沿用原source split，不把selection借作fit，也没有合并五池。每个合法pool/budget分别训练原18特征逻辑回归和二层树；阈值、正则、树深度和叶节点约束不变。

新增混合采集对照与单采集任务共享query、预算、逐类数量；实际能否取到多个采集列于task_coverage.csv。Qingdao等部分控制任务只能来自一个合法采集，不能把这些任务算作真正多采集干预。

冻结840个任务（包括same/cross、两种组织、fit/selection、预算和draw重复），产生27820行源端记录，实际拟合28个探针。目标仍使用上轮完全相同的查询、支持和两端概率，不新增公开数据。

## 匹配覆盖必须如实看

下表是single源端的实际覆盖。fit_query_scenes是源query的港口-采集场景数；事件包含五次支持抽样的重复观察。trained只表示两类事件均存在并完成拟合，不表示有独立匹配验证，更不表示效果通过。

| pool | budget | fit_query_scenes | fit_rescues | fit_harms | fit_event_ships | selection_query_scenes | selection_rescues | selection_harms | trained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Jebel Ali | 16 | 6 | 11 | 6 | 16 | 0 | 0 | 0 | True |
| Jebel Ali | 32 | 4 | 13 | 5 | 16 | 0 | 0 | 0 | True |
| Jebel Ali | 64 | 1 | 1 | 2 | 3 | 0 | 0 | 0 | True |
| Ningbo-Zhoushan | 16 | 1 | 0 | 1 | 1 | 2 | 1 | 6 | False |
| Ningbo-Zhoushan | 32 | 0 | 0 | 0 | 0 | 2 | 2 | 7 | False |
| Ningbo-Zhoushan | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| Qingdao | 16 | 6 | 12 | 8 | 16 | 2 | 17 | 7 | True |
| Qingdao | 32 | 0 | 0 | 0 | 0 | 2 | 5 | 4 | False |
| Qingdao | 64 | 0 | 0 | 0 | 0 | 2 | 5 | 10 | False |
| Shanghai | 16 | 5 | 5 | 3 | 6 | 3 | 1 | 0 | True |
| Shanghai | 32 | 4 | 5 | 2 | 7 | 0 | 0 | 0 | True |
| Shanghai | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| Singapore | 16 | 1 | 0 | 0 | 0 | 1 | 0 | 4 | False |
| Singapore | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| Singapore | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False |

实际可训练single探针：B16三个池（Jebel、Qingdao、Shanghai），B32两个池（Jebel、Shanghai），B64一个池（Jebel，仅3个事件/3条船）。宁波B16仅1次误伤，新加坡B16零事件，无法训练二类收益选择器。不能把这些格子的body回退记成“已匹配且没有损失”。

有single训练并有matching selection任务的只有Qingdao B16与Shanghai B16，后者selection只有一个有效事件。Jebel三个预算和Shanghai B32均无匹配selection；源选择策略按预先约定回退body，不拿目标挑策略。

## 各任务结果，不把缺模型算胜利

以下均是相对船体的跨采集BA百分点。NA表示没学成匹配模型，非零收益或零退化。B64 Jebel的0是模型确实训练后在全部42个分歧查询上选择关闭，其他池B64的NA则是根本没有该预算模型，两者不能混淆。

| pool | budget | trained | status | logistic | logistic_random | mix | half |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jebel Ali | 16 | True | trained_without_matching_selection | 0.9260 | 0.7256 | 0.7442 | 0.3024 |
| Jebel Ali | 32 | True | trained_without_matching_selection | -0.2842 | -0.2842 | -0.2842 | -0.4180 |
| Jebel Ali | 64 | True | trained_without_matching_selection | 0.0000 | 0.0000 | -1.5136 | -0.7363 |
| Ningbo-Zhoushan | 16 | False | fit_events_missing_one_or_both_outcomes | NA | NA | 0.3178 | 0.4453 |
| Ningbo-Zhoushan | 32 | False | no_fit_tasks | NA | NA | 0.4448 | -0.1212 |
| Ningbo-Zhoushan | 64 | False | no_fit_tasks | NA | NA | -0.5317 | -1.4257 |
| Qingdao | 16 | True | trained_and_source_validated | -1.0000 | -0.9167 | -1.0000 | -0.5000 |
| Qingdao | 32 | False | no_fit_tasks | NA | NA | 0.8333 | 1.0185 |
| Qingdao | 64 | False | no_fit_tasks | NA | NA | -4.6667 | -4.0000 |
| Shanghai | 16 | True | trained_and_source_validated | 0.1111 | 0.1111 | 0.1111 | 0.1111 |
| Shanghai | 32 | True | trained_without_matching_selection | -0.1806 | -0.1806 | -0.1806 | -0.2639 |
| Shanghai | 64 | False | no_fit_tasks | NA | NA | 0.0000 | 0.8333 |
| Singapore | 16 | False | fit_events_missing_one_or_both_outcomes | NA | NA | -0.3404 | 0.0203 |
| Singapore | 32 | False | no_fit_tasks | NA | NA | -0.0181 | 0.6207 |
| Singapore | 64 | False | no_fit_tasks | NA | NA | 0.5522 | -0.0442 |

same结果、源端选择结果、NLL/Brier及所有负结果保留在target_metrics.csv。target_summary_with_fallback.csv只表示带body回退的操作结果，不可当作全覆盖学习效果主表。主比较应使用matched_results_with_missing_as_na.csv及common_trained_scope_summary.csv。

## Jebel Ali B16：究竟改善在哪里

### 同目标队列下的训练配方对照

| recipe | logistic | logistic_random | mix | half |
| --- | --- | --- | --- | --- |
| mixed_budget | 0.1001 | -0.0461 | 0.7442 | 0.3024 |
| previous_mixed16 | 0.2248 | 0.3429 | 0.7442 | 0.3024 |
| single16_transfer | 0.9260 | 0.7256 | 0.7442 | 0.3024 |
| single_budget | 0.9260 | 0.7256 | 0.7442 | 0.3024 |

single_budget与mixed_budget共用源查询、预算和类配额，因此比较比旧训练与新训练的直接差值更有辨别力。不同支持组织会改变支持船组成和救回/误伤监督，这是该组织干预的作用路径，并不等价于证明某种物理散射机制。

### 五次支持抽样

| draw | logistic | logistic_random | mix |
| --- | --- | --- | --- |
| 0 | 0.5215 | 0.0765 | -0.0340 |
| 1 | 0.8333 | 0.8003 | 1.6402 |
| 2 | 1.6922 | 1.6412 | 1.6922 |
| 3 | 0.3912 | 0.1956 | 0.3912 |
| 4 | 1.1920 | 0.9142 | 0.0313 |

### 四个采集

| acquisition | selector_oa_gain | selector_ba_gain | vs_random_ba | vs_mix_ba |
| --- | --- | --- | --- | --- |
| S1A_060963_07982C | 1.4925 | 1.6190 | 0.8957 | 0.3810 |
| S1A_061277_07A4D4 | 2.2222 | 1.6667 | -0.3704 | 0.0926 |
| S1A_062188_07C8BF | 0.5882 | 0.6679 | 0.0159 | 0.0952 |
| S1C_005551_00B113 | 0.0000 | -0.2494 | 0.2608 | 0.1587 |

3/4采集的BA较船体为正，另一个为负；这是适用边界，不否定整体净收益。相对同数量随机，3/4采集更好。

### 类别贡献

| truth | query_samples | query_ships | rescues | harms | selector_ba_contribution_pp | mix_ba_contribution_pp |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 24 | 23 | 0 | 0 | 0.0000 | 0.1852 |
| 1 | 1 | 1 | 0 | 0 | 0.0000 | 0.0000 |
| 2 | 27 | 26 | 4 | 2 | 0.4167 | 0.2083 |
| 3 | 78 | 75 | 4 | 2 | 0.0635 | -0.0238 |
| 4 | 18 | 17 | 0 | 0 | 0.0000 | 0.0000 |
| 5 | 15 | 15 | 3 | 0 | 0.3401 | 0.3401 |
| 6 | 9 | 6 | 0 | 0 | 0.0000 | 0.0000 |
| 7 | 28 | 25 | 3 | 1 | 0.1058 | 0.0343 |

贡献按相同任务内类别、采集和港口权重计算，可加总还原完整BA增量；不是各类别BA变化直接相加。类别编号沿用原八类映射，个别类别缺失不会虚构样本。

### 是否学到了可区分的关系

| organization | split | rescue | harm | event_ships | auc | selected_rescue | selected_harm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single | fit | 11 | 6 | 16 | 0.8182 | 9 | 2 |
| mixed | fit | 6 | 7 | 13 | 0.8571 | 5 | 0 |
| single_budget | target | 15 | 10 | 22 | 0.7533 | 14 | 5 |
| mixed_budget | target | 15 | 10 | 22 | 0.5400 | 5 | 3 |
| single16_transfer | target | 15 | 10 | 22 | 0.7533 | 14 | 5 |
| previous_mixed16 | target | 15 | 10 | 22 | 0.4667 | 10 | 7 |

该目标队列事件排序AUC从旧探针0.4667提高到0.7533；对应14次救回、5次误伤。这同时有排序与净收益证据，不只看训练拟合。不过事件只有22条不同船，且无本池匹配source selection，不能叫已充分确认的跨港通用规律。

回归主要利用船体/环带的类支持一致性、局部删支持稳定性、支持类别占比等组合。ring_loo_stability在该模型中系数为负，说明不能把结果预先讲成“环带越稳定越值得信任”。这些相关输入的联合规则不等于单项因果解释；系数文件已完整保存。

## 为什么大预算没有补齐

独立只读检查确认，问题并非把预算参数改成64就能解决：需要同港至少两次采集，扣除固定query、同船与重叠裁片后，仍有足够逐类共同支持；fit与selection还必须互不共享船和采集。

- **Jebel：** source selection的Hamburg两采集逐类共同容量最多15，连B16都组不出。fit中Port Klang一方向排重后same58不足B64，反方向可用，所以只剩一个B64场景。
- **Qingdao：** 三个大Singapore采集被旧split拆成一大采集在fit、两大采集在selection，存在明确碎片化。但总共三个大采集，仍组不出两组采集互斥的B64配对。
- **Ningbo：** Busan、Tanger Med配对被split拆散，重新按配对划分有机会增加B32。恢复历史被剔除的validation记录只给出B64乐观上限，当前固定query排重后same60，不能称为已可用64。
- **Shanghai、Singapore：** 目前合法源validation即使恢复遗漏记录并暂忽略split，逐类共同容量也不足以支持B64。不能靠合池或从validation借船补齐。

因此这一轮还不能回答“全五池同预算匹配后是否都能学到规律”。已完成的是当前合法覆盖上的匹配对照，以及覆盖不足的具体定位。

## 根因判断与保留方案

1. **可保留：** Jebel B16的单采集训练选择器，固定设计下五次支持draw净BA均为正，且平均超过固定组合、半强度船体和同数量随机。它是局部候选，不直接替换全局主模型。
2. **不能推广：** Qingdao B16仍退化；Shanghai B16只是继承一个环带救回，未超随机；两个匹配B32均无净提升；B64只有3事件学习后全关闭。
3. **已排除的实现问题：** 新任务索引、船只/采集隔离、预算嵌套与配额，查询标签改写不改变预测/特征，以及旧目标端点/旧选择器重放均通过。未发现需要修代码才能解释的退化。
4. **下一步若继续：** 在不改变模型的前提下，优先按合法采集配对组织源fit/selection，补齐B16/B32可学习事件；B64需要更多合法源配对采集覆盖，不能仅重排当前缓存就声称完成。先冻结本轮Jebel候选，不依据目标结果再调C或阈值。

## 技术记录

小探针及目标缓存评价运行约35.6秒，不含既有骨干/特征提取。14棵树中12棵为根节点常量预测，符合未改变的min_samples_leaf=10；不能把它们写成学到了条件分支。

Ningbo混合B16的标准化曾出现浮点方差-2.19e-47的警告；常量特征实际scale被设为1，所有模型参数、尺度和输出均为有限数，未发现传播NaN。这是已定位的数值舍入，不用于解释任务得失，也没有据结果更换实现。

文件：frozen.json/PLAN_ZH.md记录先验设计；learning_coverage.csv和SOURCE_FROZEN.json记录可学习性/选择；models.joblib、logistic_coefficients.csv与tree.txt保存模型；per_port_ba_deltas.csv、per_port_draw_ba_deltas.csv、jebel_B16_class_contributions.csv保存细分结果；DONE.json与REPORT_CHECKS.json记录验证。

数据仍是历史开发数据，不是新独立确认；固定模型五次支持抽样不等于五个训练种子。相似模型或负结果不否定局部收益，但也不把未训练格子的回退结果包装成成功。
