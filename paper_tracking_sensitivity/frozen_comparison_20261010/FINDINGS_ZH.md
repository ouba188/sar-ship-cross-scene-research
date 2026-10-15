# 冻结主模型的匹配对比实验

日期：2026-10-10。第一批23模式、25任务、125组支持抽样、2875条任务级结果全部完成。原主模型基础/最终预测与5%交接名单逐项复现。此处的模式数包括重复一致性参考行，不等于23个独立公开算法。

## 结论

当前5%交接在三个集合的平均OA/BA均优于同预算随机、低margin与高entropy交接，也优于更严格的相同正收益资格集合/相同实际交接数量对照。但不限预算的简单双分支softmax均值融合在三个集合的OA/BA都更高。因此可以说有限专家接管下排序有效，不能说5%系统的绝对精度最高。当前已经全量计算两路，尚无真实计算节省证据；预算的部署含义必须有实际应用依据，不能用虚构成本回避融合强基线。主版本不根据这些目标成绩重新选择预算或模块。

## 完整识别比较

| method | Self24 (23 tasks) OA / BA | F_0921 OA / BA | OpenSARShip 2.0 OA / BA |
| --- | --- | --- | --- |
| source_prior | 46.7331 / 45.3975 | 24.1993 / 20.0255 | 17.5985 / 17.8748 |
| prototype64 | 52.9628 / 42.1648 | 26.1325 / 21.3685 | 36.4466 / 28.8167 |
| ridge_mean64 | 50.7019 / 41.5326 | 20.6859 / 20.3828 | 32.0293 / 28.3375 |
| ridge_phi64 | 57.2441 / 43.8997 | 24.7797 / 21.0680 | 38.2559 / 29.7700 |
| ridge_full_kernel64 | 59.0767 / 46.5394 | 22.3486 / 20.7112 | 29.3236 / 27.4115 |
| logistic_mean64 | 53.9736 / 41.1209 | 23.1002 / 20.6548 | 36.9139 / 29.5253 |
| r2d2_frozen_head | 60.8344 / 38.3096 | 30.4556 / 19.5561 | 43.7164 / 28.7064 |
| r2d2_normalized_head | 61.7551 / 33.0651 | 30.7505 / 19.0928 | 43.8305 / 28.6717 |
| ours_base | 61.1956 / 52.4994 | 30.5264 / 26.0348 | 33.0562 / 31.2783 |
| previous_full | 62.4329 / 53.7806 | 30.8931 / 26.2879 | 34.3385 / 31.9526 |
| ours_5pct | 63.2084 / 54.2383 | 31.1072 / 26.3593 | 35.3545 / 32.3724 |
| expert_all | 64.9579 / 55.3750 | 31.8746 / 27.1475 | 39.7827 / 32.6082 |
| softmax_average_all | 65.5743 / 55.9873 | 33.1662 / 28.0031 | 40.7661 / 33.9147 |

source_prior为零目标标签参考。其余头共享同64标签；ridge/proto/logistic是标准强基线，不是完整公开深度模型复现。mean为冻结C4的源标准化PCA128单位向量；phi包含数据集特定几何；full_kernel读取与主方案相同学习核但不带源分数锚点。三者的信息与先验访问不同，分别标注而非混作单变量消融。岭回归正则、逻辑回归C与原型距离在既有源情景上选定，目标不选参数；源端表示本身未与这些情景完全隔离。原型/普通头仅输出支持中观察到的类，源先验方法能保留八类先验。OSS2有普通头OA高于我们而BA低于我们的情况，不能写成全部指标全面胜出。

R2D2行是明确的冻结特征协议适配：通过源查询交叉熵反传岭回归求解器，学习正则和分数尺度；不是普通网格ridge，也不是端到端R2D2完整复现。固定30轮、末轮输出，没有目标端选轮数；八类自然支持分布不同于原论文均衡N-way/K-shot。此行不用于宣称战胜原论文完整方法。

两条元ridge结果均保留：frozen_head是不带截距的初版启发式头；normalized_head补齐官方公式中的常数列和平均平方误差尺度。后者初始lambda=1/64，使64支持样本下有效求和正则仍为1；这是单位尺度转换，不是目标选参。alpha仍初始1，30轮/Adam .01不变。官方默认lambda50且不学习，原发布配置alpha可为1e-4或10；本实验不使用这些默认训练设置，也不重训骨干，仍只称冻结特征协议适配。

对齐公式后BA没有改善的结果同样保留。类别表显示，该适配在F_0921对渔船/杂货船/近海供应船的召回仅约1.69/1.86/0.85%，在OSS2对成品/化学品油轮、集装箱船则高于当前主模型，而对拖轮/近海供应船召回约1.10/0.75%。因此高OA与低BA确实对应类别取舍。已验证公式与独立原始空间解、梯度及官方平均损失形式一致；但未证明训练已达到最优，不能把差距全部归因于损失不平衡或据此否定完整R2D2。两版本class_metrics/class_summary已保存。

## 交接策略比较

| dataset | mode | oa | ba | handoff_rate |
| --- | --- | --- | --- | --- |
| f0921 | entropy_common05 | 30.7633 | 26.2922 | 4.9998 |
| f0921 | margin_common05 | 30.8999 | 26.3072 | 4.9998 |
| f0921 | margin_positive_matched | 30.9456 | 26.3346 | 4.9998 |
| f0921 | ours_5pct | 31.1072 | 26.3593 | 4.9998 |
| f0921 | random_common05 | 30.6304 | 26.0958 | 4.9998 |
| f0921 | random_positive_matched | 30.6585 | 26.1319 | 4.9998 |
| oss2 | entropy_common05 | 33.3822 | 31.2811 | 4.9986 |
| oss2 | margin_common05 | 33.7680 | 31.6687 | 4.9986 |
| oss2 | margin_positive_matched | 33.7517 | 31.6184 | 4.9986 |
| oss2 | ours_5pct | 35.3545 | 32.3724 | 4.9986 |
| oss2 | random_common05 | 33.5887 | 31.2248 | 4.9986 |
| oss2 | random_positive_matched | 33.5887 | 31.2191 | 4.9986 |
| self24ports | entropy_common05 | 61.9308 | 53.2788 | 4.9325 |
| self24ports | margin_common05 | 62.3985 | 53.4669 | 4.9325 |
| self24ports | margin_positive_matched | 62.4398 | 53.5205 | 4.9325 |
| self24ports | ours_5pct | 63.2084 | 54.2383 | 4.9325 |
| self24ports | random_common05 | 62.0300 | 53.2802 | 4.9325 |
| self24ports | random_positive_matched | 62.4002 | 53.6385 | 4.9325 |

common05使用共同的查询/预测分歧/专家类别已观察条件，不依赖我方收益分数符号。positive_matched进一步固定我方正收益资格集合和实际数量，只比较排序。随机每支持抽样一个预定随机数种子，共5个，不是独立的大规模随机策略重复。软最大分数只是排序/融合运算，不宣称已校准概率。

## 当前版本预算曲线

| dataset | mode | oa | ba | handoff_rate | expert_decision_use_rate |
| --- | --- | --- | --- | --- | --- |
| f0921 | budget_000 | 30.5264 | 26.0348 | 0.0000 | 0.0000 |
| f0921 | budget_005 | 31.1072 | 26.3593 | 4.9998 | 4.9998 |
| f0921 | budget_010 | 31.4636 | 26.5881 | 9.9996 | 9.9996 |
| f0921 | budget_020 | 31.8710 | 26.9091 | 20.0000 | 20.0000 |
| f0921 | budget_050 | 31.8396 | 26.9229 | 46.8506 | 46.8506 |
| f0921 | budget_100 | 31.8047 | 26.8786 | 48.2299 | 48.2299 |
| f0921 | expert_all | 31.8746 | 27.1475 | 100.0000 | 100.0000 |
| f0921 | softmax_average_all | 33.1662 | 28.0031 | nan | 100.0000 |
| oss2 | budget_000 | 33.0562 | 31.2783 | 0.0000 | 0.0000 |
| oss2 | budget_005 | 35.3545 | 32.3724 | 4.9986 | 4.9986 |
| oss2 | budget_010 | 36.6096 | 32.7630 | 9.9973 | 9.9973 |
| oss2 | budget_020 | 37.9897 | 32.8798 | 19.9946 | 19.9946 |
| oss2 | budget_050 | 39.3860 | 32.0250 | 49.1877 | 49.1877 |
| oss2 | budget_100 | 39.5979 | 32.1450 | 51.5132 | 51.5132 |
| oss2 | expert_all | 39.7827 | 32.6082 | 100.0000 | 100.0000 |
| oss2 | softmax_average_all | 40.7661 | 33.9147 | nan | 100.0000 |
| self24ports | budget_000 | 61.1956 | 52.4994 | 0.0000 | 0.0000 |
| self24ports | budget_005 | 63.2084 | 54.2383 | 4.9325 | 4.9325 |
| self24ports | budget_010 | 64.3299 | 55.1088 | 9.4325 | 9.4325 |
| self24ports | budget_020 | 65.0510 | 55.6545 | 16.2236 | 16.2236 |
| self24ports | budget_050 | 65.2261 | 55.6221 | 23.3157 | 23.3157 |
| self24ports | budget_100 | 65.2297 | 55.6228 | 23.3338 | 23.3338 |
| self24ports | expert_all | 64.9579 | 55.3750 | 100.0000 | 100.0000 |
| self24ports | softmax_average_all | 65.5743 | 55.9873 | nan | 100.0000 |

100%门控仍有资格与正收益过滤，不等于全专家。全量融合不属于硬交接，handoff_rate记为空，expert_decision_use_rate为100%；后者是专家输出参与最终决策比例，不是特征计算比例。5%不是普遍最优点，放宽预算也非单调改善BA：OSS2到50%时BA低于5%。不按目标结果事后修改主工作点。

![Budget curves](E:/Hermes/paper_tracking/sensitivity/frozen_comparison_20261010/budget_comparison.png)

阴影为五组支持抽样指标的标准差，不是独立训练置信区间。横轴是离散预算档位；虚线参考使用不限预算专家信息。

## 可解释性同步补齐

| dataset | task | sample_id | version | gate_score | rank | cap | admitted | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f0921 | f0921 | f0921_349180 | previous | 0.9359 | 22642 | 6171 | False | rank_exceeds_budget |
| f0921 | f0921 | f0921_349180 | current | 0.9500 | 22164 | 6171 | False | rank_exceeds_budget |
| f0921 | f0921 | f0921_207042 | previous | 1.8649 | 7760 | 6171 | False | rank_exceeds_budget |
| f0921 | f0921 | f0921_207042 | current | 1.8622 | 0 | 6171 | False | same_prediction |
| oss2 | oss2 | pub2_opensarship2_005681 | previous | 0.7817 | 1143 | 184 | False | rank_exceeds_budget |
| oss2 | oss2 | pub2_opensarship2_005681 | current | 0.7833 | 1121 | 184 | False | rank_exceeds_budget |
| oss2 | oss2 | pub2_opensarship2_006630 | previous | 2.3422 | 209 | 184 | False | rank_exceeds_budget |
| oss2 | oss2 | pub2_opensarship2_006630 | current | 2.3423 | 182 | 184 | True | inside_budget |
| self24ports | Callao | returned_local_knowledge_0000338 | previous | 0.7478 | 20 | 20 | True | inside_budget |
| self24ports | Callao | returned_local_knowledge_0000338 | current | 0.7391 | 21 | 20 | False | rank_exceeds_budget |
| self24ports | Hamburg | pair_0018485 | previous | 1.3125 | 106 | 77 | False | rank_exceeds_budget |
| self24ports | Hamburg | pair_0018485 | current | 1.4813 | 68 | 77 | True | inside_budget |

已复用原六张案例，补齐旧/新交接分数、资格、排名、截止数和实际原因。rank=0表示未进入资格集合，不表示排名第一。原预测名单重放一致；这些是计算决策轨迹，不是物理散射解释。25个任务、共125组的全量轨迹在targets/**/routing_trace.npz。

## 尚未完成的比较

完整端到端公开SAR分类强方法与完整公开learning-to-defer方法尚未复现。R2D2冻结头只填补对应适配基线，不能替代这两项。公开方法核读及可复现性见PUBLIC_METHODS_ZH.md（该支线归档后生效）。本报告不声称完整SOTA比较已经齐备。

SARATR-X可执行性复查已完成：本地有官方源码，现有CUDA环境可导入HiViT；约1.044GB官方权重匿名401、现有凭据403，未获下载授权。已向用户询问可用权重路径；不绕过门控，不拿随机初始化模型冒充该方法。官方预训练包含OpenSARShip，具体目标重叠仍待核实。

## 文件

summary.csv、metrics.csv、draw_metrics.csv、class_metrics.csv、MAIN_COMPARISON_TABLE.csv、case_routing_explanations.csv、PROTOCOL.json；r2d2_frozen_head/单独保存适配方法协议、源训练参数、目标预测与结果。全部结果是历史开发数据，同五组支持抽样，不是五训练种子；原实验工件未覆盖。
