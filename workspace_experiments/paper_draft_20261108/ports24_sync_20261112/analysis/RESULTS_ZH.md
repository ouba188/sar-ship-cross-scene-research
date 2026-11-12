# 24 港 Full-Tuned 分析

状态：真实完整预测已统计；原始 chip 面板仍需主代理目视审核。

主比较为 Full-Tuned 对原 Full。旧 Plain/SE 仅为历史不同训练配方的系统对照，不能作为匹配微调基线或单独模块因果对照。

| 方法 | OA mean +/- SD | BA mean +/- SD | Macro-F1 mean +/- SD | NLL mean | Brier mean |
|---|---:|---:|---:|---:|---:|
| Full-Tuned | 52.076 +/- 0.244 | 48.970 +/- 0.808 | 46.520 +/- 0.637 | 1.4139 | 0.6462 |
| Full | 48.412 +/- 1.203 | 43.902 +/- 0.654 | 42.459 +/- 0.937 | 2.0670 | 0.7622 |
| Plain | 42.927 +/- 0.620 | 40.172 +/- 1.041 | 38.385 +/- 0.784 | 1.8320 | 0.7745 |
| SE | 42.309 +/- 0.523 | 41.218 +/- 0.668 | 38.742 +/- 0.693 | 1.6081 | 0.7410 |
| Old-Query | 48.412 +/- 1.286 | 43.873 +/- 0.741 | 42.489 +/- 1.172 | 2.0967 | 0.7647 |
| Old-QueryRef | 48.347 +/- 1.230 | 43.748 +/- 0.590 | 42.316 +/- 0.938 | 2.0954 | 0.7645 |
| Tuned-Query | 52.249 +/- 0.244 | 49.481 +/- 0.746 | 46.963 +/- 0.627 | 1.4319 | 0.6485 |
| Tuned-QueryRef | 52.379 +/- 0.474 | 49.691 +/- 0.675 | 47.198 +/- 0.554 | 1.4261 | 0.6467 |

| Candidate - Baseline | Metric | Delta | MMSI paired 95% CI |
|---|---|---:|---|
| Full-Tuned - Full | oa | +3.6640 | [+2.1746, +5.2014] |
| Full-Tuned - Full | ba | +5.0678 | [+3.4819, +6.6712] |
| Full-Tuned - Full | macro_f1 | +4.0615 | [+2.6183, +5.5738] |
| Full-Tuned - Full | nll | -0.6532 | [-0.7334, -0.5765] |
| Full-Tuned - Full | brier | -0.1160 | [-0.1363, -0.0961] |
| Old-Query - Full | oa | +0.0000 | [-0.1732, +0.1618] |
| Old-Query - Full | ba | -0.0291 | [-0.3155, +0.2401] |
| Old-Query - Full | macro_f1 | +0.0306 | [-0.2966, +0.3393] |
| Old-Query - Full | nll | +0.0296 | [+0.0145, +0.0464] |
| Old-Query - Full | brier | +0.0025 | [+0.0001, +0.0052] |
| Old-QueryRef - Full | oa | -0.0650 | [-0.3117, +0.1543] |
| Old-QueryRef - Full | ba | -0.1536 | [-0.5135, +0.1863] |
| Old-QueryRef - Full | macro_f1 | -0.1423 | [-0.5326, +0.2129] |
| Old-QueryRef - Full | nll | +0.0284 | [+0.0132, +0.0454] |
| Old-QueryRef - Full | brier | +0.0023 | [-0.0002, +0.0049] |
| Old-QueryRef - Old-Query | oa | -0.0650 | [-0.2401, +0.1162] |
| Old-QueryRef - Old-Query | ba | -0.1245 | [-0.3374, +0.0994] |
| Old-QueryRef - Old-Query | macro_f1 | -0.1729 | [-0.3816, +0.0422] |
| Old-QueryRef - Old-Query | nll | -0.0012 | [-0.0026, +0.0003] |
| Old-QueryRef - Old-Query | brier | -0.0002 | [-0.0007, +0.0004] |
| Tuned-Query - Full-Tuned | oa | +0.1734 | [-0.0107, +0.3717] |
| Tuned-Query - Full-Tuned | ba | +0.5113 | [+0.1639, +0.9019] |
| Tuned-Query - Full-Tuned | macro_f1 | +0.4423 | [+0.0639, +0.8438] |
| Tuned-Query - Full-Tuned | nll | +0.0180 | [+0.0086, +0.0282] |
| Tuned-Query - Full-Tuned | brier | +0.0024 | [+0.0001, +0.0047] |
| Tuned-QueryRef - Full-Tuned | oa | +0.3035 | [-0.0107, +0.6154] |
| Tuned-QueryRef - Full-Tuned | ba | +0.7214 | [+0.3054, +1.1684] |
| Tuned-QueryRef - Full-Tuned | macro_f1 | +0.6777 | [+0.2346, +1.1432] |
| Tuned-QueryRef - Full-Tuned | nll | +0.0122 | [+0.0027, +0.0231] |
| Tuned-QueryRef - Full-Tuned | brier | +0.0005 | [-0.0021, +0.0031] |
| Tuned-QueryRef - Tuned-Query | oa | +0.1301 | [-0.1074, +0.3656] |
| Tuned-QueryRef - Tuned-Query | ba | +0.2101 | [-0.0216, +0.4525] |
| Tuned-QueryRef - Tuned-Query | macro_f1 | +0.2354 | [+0.0158, +0.4701] |
| Tuned-QueryRef - Tuned-Query | nll | -0.0058 | [-0.0084, -0.0032] |
| Tuned-QueryRef - Tuned-Query | brier | -0.0018 | [-0.0029, -0.0008] |
| Tuned-QueryRef - Old-QueryRef | oa | +4.0325 | [+2.5132, +5.5851] |
| Tuned-QueryRef - Old-QueryRef | ba | +5.9428 | [+4.3470, +7.5773] |
| Tuned-QueryRef - Old-QueryRef | macro_f1 | +4.8815 | [+3.4396, +6.3975] |
| Tuned-QueryRef - Old-QueryRef | nll | -0.6694 | [-0.7545, -0.5901] |
| Tuned-QueryRef - Old-QueryRef | brier | -0.1178 | [-0.1378, -0.0981] |

类别顺序：0=bulk_carrier, 1=fishing_vessel, 2=general_cargo, 3=product_chemical_tanker, 4=container_ship, 5=crude_oil_tanker, 6=tug_towing, 7=offshore_supply

aggregate/ 包含逐种子、三种子均值/样本SD、8类逐类、逐港、逐pass_group、原始实际可用性、配对变化和MMSI置信区间。
OA/BA/F1 为百分数，变化为百分点；NLL/Brier 变化为损失单位。BA 仅平均子集真值存在类，macro-F1 固定8类、零分母记0。
MMSI bootstrap 重采样船只并保留其全部观测，方法及三个固定种子成对；先算各种子指标再平均，不把三个种子复制成独立船只。
BA抽样若丢失原子集存在类别则排除并记录有效次数。CI针对 all/availability；逐港和逐过境提供描述性三种子统计，未为每个小组重复bootstrap。
CI条件于当前场景和已训练模型，不估计新港口泛化或选点不确定性；历史测试队列不是新盲测。
可用性分层固定使用原始 prepare_view 的有效输入，recovery 后不重分组，避免比较不同分母。

Recovery纳入要求：每臂3种子的validation/evaluation/complete.json全部齐备。未完成臂跳过且写入private/sources_local.json，不推测其结果。
query_recovery、reference_increment_given_query_recovery、query_reference_total 分别报告query单步、给定query恢复后的reference增量及总作用。
原训练权重上的恢复与微调权重上的恢复均为冻结权重输入对照，不是恢复后重新训练。

生成 20 张原生像素面板：gallery/index.html；private/cases_local.json保留逐种子预测、分差、合法参考及原始路径。
主比较目标为3rescues/3harms/3stablecorrect/3stablewrong；若某组不足，不伪造或跨种子重复凑数，缺口见aggregate/case_coverage.csv。
完整recovery臂另选4--8幅代表，优先覆盖救回/误伤；代表案例不是总体比例。
参考排除同MMSI及同orbit（pass_group前两段），再要求与query共享可用极化；图中参考是合法示例，不宣称最高贡献。
评分margin为各模型自身logit尺度，不能把跨模型绝对幅度变化直接当证据强度。

**reportpending：主代理仍需查看VV/VH及合法参考，解释代表案例与未解释的失败；脚本不自动编造图像机制或判定论文增益。**
private/和gallery/含私人路径、MMSI/原始图像，保留本地；aggregate/不含逐样本ID/MMSI或原始文件路径。未执行Git、训练或模型推理。
