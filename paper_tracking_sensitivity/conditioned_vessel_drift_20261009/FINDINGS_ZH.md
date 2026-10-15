# 同船观测条件与隐空间变化

日期：2026-10-09。仅源数据条件归因诊断，主模型未改动，未读取目标查询标签。

## 输入与问题

逐目标入射角来自按完整产品编号下载的公开annotation网格及已有位置插值；仅用sidecar status=ok的28628/38091条记录。缺失不填补，代表性SAFE模板不参与归因。
同港同MMSI同采集时间重复观测先平均；任一参与行缺少条件的聚合观测不进入本轮。标签冲突身份按各源划分剔除。MMSI仍为身份代理。
同船差分同时包含位置、姿态、船舶状态及观测条件变化，不预设它等于噪声。本轮以位置差分为对照，检验入射角是否提供额外的可迁移解释。

## 固定方法

沿用23个排除目标身份/产品的源划分。按MMSI固定随机留出约1/3船；另固定留出3/6/9/12月采集。拟合只用其余船和其余月份。先按子集筛选观测，再在同港同船内配对相邻时相；四组的间隔分布可以不同。
报告拟合、新船、新月份、新船且新月份四组。新月份组不含拟合产品；新船组不含拟合船身份。原有PCA/Qs/Qd利用过外层全部源数据，因此这里检验给定表示下条件响应的外推，不是整个表示学习流程的独立验证。
入射角输入为t=(angle-38)/5及t平方的两次观测差；位置输入为局部东西、南北位移/10km。零截距加权岭回归，固定ridge=0.01，不调参。三种模型为角度、位置、角度+位置。权重按类别、类内港、港内船、船内配对逐级等权。
输出分别为原128维标准化R50保留特征和39至43维既有[Qs,Qd]坐标。skill=1-预测残差能量/原同船差分能量，以预测零变化为基准；它不是分类准确率，也不是因果解释率。

## 条件响应结果

| 空间 | 留出方式 | 输入 | 平均skill (%) | skill为正的源划分数 |
|---|---|---|---:|---:|
| retained128 | fit | incidence | 1.4307 | 23/23 |
| retained128 | fit | incidence_position | 2.2821 | 23/23 |
| retained128 | fit | position | 0.8615 | 23/23 |
| retained128 | new_month | incidence | -0.4514 | 0/23 |
| retained128 | new_month | incidence_position | -1.3174 | 0/23 |
| retained128 | new_month | position | -0.8669 | 0/23 |
| retained128 | new_ship | incidence | -0.7069 | 0/23 |
| retained128 | new_ship | incidence_position | -1.3539 | 0/23 |
| retained128 | new_ship | position | -0.6800 | 0/23 |
| retained128 | new_ship_month | incidence | -0.4096 | 1/23 |
| retained128 | new_ship_month | incidence_position | -1.2027 | 0/23 |
| retained128 | new_ship_month | position | -0.7985 | 0/23 |
| stable_drift | fit | incidence | 3.5296 | 23/23 |
| stable_drift | fit | incidence_position | 4.3665 | 23/23 |
| stable_drift | fit | position | 0.8450 | 23/23 |
| stable_drift | new_month | incidence | 2.7344 | 23/23 |
| stable_drift | new_month | incidence_position | 2.5806 | 23/23 |
| stable_drift | new_month | position | -0.3641 | 5/23 |
| stable_drift | new_ship | incidence | 0.9187 | 18/23 |
| stable_drift | new_ship | incidence_position | 0.2886 | 13/23 |
| stable_drift | new_ship | position | -0.7259 | 0/23 |
| stable_drift | new_ship_month | incidence | 2.7805 | 23/23 |
| stable_drift | new_ship_month | incidence_position | 2.4844 | 23/23 |
| stable_drift | new_ship_month | position | -0.5249 | 5/23 |

23个源划分大量重叠，不能当作23次独立发现。正skill只表示预测比零差分好，不能自动证明条件的因果效应。

## 覆盖及定位

| 留出方式 | 每折配对数范围 | 每折源港数范围 | 每折类别数范围 |
|---|---:|---:|---:|
| fit | 2393-3349 | 19-21 | 8-8 |
| new_month | 987-1511 | 19-21 | 8-8 |
| new_ship | 1115-1731 | 18-21 | 8-8 |
| new_ship_month | 506-814 | 18-21 | 8-8 |

incremental_skill.csv保存角度+位置相对位置的额外skill；class_metrics.csv和source_port_metrics.csv保留负结果，防止总体平均掩盖特定船型/港口退化。best_scalar仅是留出结果的事后方向/幅度诊断，不用于重新调节模型或报告校正性能。

## 本轮边界

对位置的控制只包括一阶局部位移，不能排除泊位背景、船舶姿态、状态及其他未测因素。缺失覆盖不均；插值角度也不等于地形校正局部入射角。
尚未进行条件校正或新的完整模型训练，不提供虚构的BA/OA提升。本轮结论与下一步见DECISION_ZH.md。
