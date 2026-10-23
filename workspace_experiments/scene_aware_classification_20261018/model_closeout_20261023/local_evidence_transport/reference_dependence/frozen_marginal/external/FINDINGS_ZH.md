# 冻结参数外推：3075条历史外部观测

2026-10-24。已完成，主计算进程正常退出（exit code 0），推理及保存约8.32秒。没有训练、选择参数、拟合外部尺度或按结果调参。只写本 external 目录，未操作Git，未做原图分析。

## 结论

源端joint的OA/BA正信号没有原样转移到本外部队列。相对anchor，joint平均OA仅增加0.0108个百分点，BA下降0.9419个百分点；NLL和Brier改善。三个基础种子分别净救回 -21 / 0 / +22。

这不能说明所有补充关系无效，但本轮不支持“真实参考配对在外部优于置乱配对”的主张：shuffle307的平均OA、BA、NLL和Brier均优于joint。不得据此外部结果改选shuffle307作为新主方案。五臂均沿用各自源端已选系数，因而是冻结算法整臂对照，不是固定同一系数只交换配对的干预。

## 外推与重放

- 3075条、1950艘、85个过境组，八类齐全；源表3387行，参考64条。199条的VH局部mass为零。
- 沿用历史 external_confirmation 的源身份排除；并非全新blind测试。原fold0有1561条，另外两个fold共1514条。
- 源PCA32和单位化处理复用；零mass位置置零。参考索引不变，外部query索引偏移3387。
- anchor尺度固定0.3938252627849579，seed17/29/43倍率固定0.1/0.1/0.5。
- D、置乱D、nonlinear均使用源scales.csv；nonlinear先做有效类别中心化、single尺度，再做乘积及nonlinear尺度。
- 五臂分别使用源weights中的7项系数，没有重新学习6项marginal系数。
- 11条源样本端到端重放，其中5条缺VH。k/a/pi/valid、局部分数、D、nonlinear、标准化特征、anchor及五臂预测最大绝对误差8.8818e-16。详见source_replay.json。
- seed29的joint源系数全零，因此该种子与anchor完全一致；不是外推漏接模块。

## 全量结果

三个冻结基础种子的均值；OA/BA为百分数，其余为现有softmax评分，不新增温度校准。

| 臂 | OA | BA | NLL | Brier |
|---|---:|---:|---:|---:|
| full | 44.5420 | 38.3404 | 1.736650 | 0.749729 |
| anchor | 44.4878 | 38.9170 | 1.731123 | 0.750321 |
| joint | 44.4986 | 37.9752 | 1.706246 | 0.745144 |
| nonlinear | 44.5637 | 38.0465 | 1.719304 | 0.747125 |
| shuffle101 | 44.6938 | 39.0119 | 1.723595 | 0.748539 |
| shuffle211 | 44.4878 | 38.9170 | 1.731123 | 0.750321 |
| shuffle307 | 44.9648 | 39.2778 | 1.699122 | 0.744306 |

| joint对anchor | 救回 | 误伤 | 净救回 |
|---|---:|---:|---:|
| seed17 | 120 | 141 | -21 |
| seed29 | 0 | 0 | 0 |
| seed43 | 56 | 34 | +22 |

相同3075条观测在三个种子下重复使用，不能把9225次预测当成9225个独立样本。

## 逐类变化

净救回为三个种子相对anchor的计数之和，不是独立船只计数。每种子的OA、BA、NLL、Brier、救回和误伤完整保存在class_metrics.csv。

| 类别 | 单次观测数 | joint平均召回率% | 净救回合计 |
|---|---:|---:|---:|
| bulk_carrier | 947 | 58.1837 | -1 |
| fishing_vessel | 102 | 72.2222 | -31 |
| general_cargo | 301 | 23.1451 | +24 |
| product_chemical_tanker | 803 | 49.7302 | +16 |
| container_ship | 361 | 36.1958 | -4 |
| crude_oil_tanker | 256 | 44.0104 | -4 |
| tug_towing | 138 | 8.9372 | 0 |
| offshore_supply | 167 | 11.3772 | +1 |

普通货船和成品/化学品油轮有真实的净救回，不能因不是所有类别受益而抹去。但渔船只有102条却净损失31次预测，对等权BA影响明显；这是本轮BA下降的数值构成，不等于已经解释其图像根因。需要主代理结合保存的预测和实际参考chip进一步分析。

三个fold上joint对anchor平均OA变化分别约 -0.1922 / -0.3968 / +0.8355个百分点，BA均未提高。完整port、pass_group及original_test_fold分组见group_metrics.csv，不在外部选择有利分组作为验证结论。

## 交付

- run_external.py：独立外推入口，复用父算法函数但不调用prepare或fit。
- PROTOCOL_ZH.md：运行前冻结协议。
- source_replay.json：源小样本对齐和预测重放。
- localcache.npz：3075行对应的k/a/pi/valid、joint局部分数和区域分数，含instance_ids、bank及偏移索引。
- features.npz：u、L、有效性、raw D、nonlinear_raw及五臂7维特征。
- predictions_seed17/29/43.npz：full、anchor、五臂预测，附标签、instance_ids和bank。
- metrics.csv / summary.csv / class_metrics.csv / paired_results.csv / group_metrics.csv：完整数值。
- receipt.json：冻结参数路径、队列大小和无训练/无选择记录。

没有修改源结果，没有新数据下载，没有重审计全库。原图回看、总报告与Git同步按委派由主代理负责。
