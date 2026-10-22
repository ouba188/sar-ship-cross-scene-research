# 八类冻结外推：坐标纠错与完整结果

## 结论

**已确认并修正条件输入的坐标语义问题。修正后所有arm/seed的OA都高于原运行，但仍低于冻结A；两种门控的BA均有小正，区间仍跨零。不能以原负结果直接否定方向，也不能把全部损失归因于坐标问题。**

原 `gated_transfer/` 的27头、逐例分数、12个图像事件和分析表原样保留。这里独立训练27头；只将训练条件由各inner拼接的 `oof_whole64[train]` 换为统一outer-source坐标的 `whole64[train]`。训练仍使用合法 `oof/oq`，Ridge、PCA、q、baseline、权重、种子、优化器、200步预算和sourceval选择不变。实现逐行差异见 `implementation_change.diff`。

修正是**基线/贡献OOF + outer-source无标签共享条件预处理**，不是整流水线OOF。held source参与无标签outer PCA，但validation/test不参与fit；这一点按用户授权接受并明确记录。

## 可达的坐标不一致

同一source held观察在inner与outer条件坐标下的结果如下，诊断未重拟合任何模型：

| 诊断 | 实测 |
|---|---:|
| source折内位置 | 13229（不是13229个独立观察） |
| inner fits | 12 |
| 向量cosine均值 / 中位数 | 0.237341 / 0.239002 |
| 向量cosine为负比例 | 7.801% |
| 同索引64维Pearson相关均值 | 0.241433 |
| PCA同索引轴signed / absolute cosine均值 | 0.242563 / 0.637580 |
| 同索引轴负符号 | 277 / 768 |
| 交叉最佳轴absolute cosine均值 | 0.782520 |
| 交叉最佳仍是同索引 | 64.0625% |

逐inner向量cosine均值为0.121874~0.340431，不是仅某一个异常包。所有768个维度相关均有定义。`coordinate_axes.csv` 同时保存PCA标准化空间轴、除以各自StandardScaler尺度后的原stem有效轴比较；轴余弦不冒充全链路相等。轴旋转/次序、符号、前后尺度都可能改变坐标，因此不能只靠翻符号概括修复。

这些值和未对齐的调用路径确认了真实可达的语义不一致：共享MLP无法默认把四个inner以及outer的第k维视为同一方向。q已回固定原layer3通道，不受该PCA索引问题影响。诊断本身不能量化全部性能损失；后面的固定对照才给出本次改动的实际影响。

## 全结果

冻结A：4744条互斥test、2973 MMSI、85过境；OA **39.523609%**、BA **40.349061%**、NLL **1.714015**、balanced NLL **1.683299**、Brier **0.762398**。这是新八类layer3 baseline，不是旧A+D，也不是三类strong读出的成绩。

下表为三seed算术均值，未选择最好seed。OA/BA为百分数；NLL为自然对数平均；Brier为各类平方误差之和再平均，不除8。

| 版本 | arm | OA | BA | NLL | balanced NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| 原坐标未对齐 | original_centered | 39.080944 | 40.392711 | 1.660168 | 1.624067 | 0.747481 |
| 修正统一坐标 | original_centered | 39.172288 | 40.489220 | 1.660045 | 1.623325 | 0.746985 |
| 原坐标未对齐 | score_scale_decoupled | 39.347948 | 40.106497 | 1.650652 | 1.625247 | 0.744110 |
| 修正统一坐标 | score_scale_decoupled | 39.432265 | 40.426935 | 1.650852 | 1.624630 | 0.743755 |
| 原坐标未对齐 | additive | 37.282181 | 38.919012 | 1.698582 | 1.672843 | 0.762665 |
| 修正统一坐标 | additive | 37.928612 | 39.323985 | 1.621590 | 1.625030 | 0.750660 |

| arm | seed | 原净救回 | 修正净救回 | 修正减原净数 | 修正减原OA pp | 修正减原BA pp |
|---|---:|---:|---:|---:|---:|---:|
| original_centered | 17 | -25 | -17 | +8 | +0.168634 | +0.145909 |
| original_centered | 29 | -18 | -16 | +2 | +0.042159 | +0.142292 |
| original_centered | 43 | -20 | -17 | +3 | +0.063238 | +0.001326 |
| score_scale_decoupled | 17 | -6 | -5 | +1 | +0.021079 | +0.272526 |
| score_scale_decoupled | 29 | -8 | -5 | +3 | +0.063238 | +0.318516 |
| score_scale_decoupled | 43 | -11 | -3 | +8 | +0.168634 | +0.370271 |
| additive | 17 | -115 | -75 | +40 | +0.843170 | +0.799182 |
| additive | 29 | -85 | -82 | +3 | +0.063238 | -0.163887 |
| additive | 43 | -119 | -70 | +49 | +1.032884 | +0.579625 |

修正相对A：original OA -0.351321pp / BA +0.140158pp；decoupled OA -0.091343pp / BA +0.077873pp；additive OA -1.594997pp / BA -1.025076pp。两种门控均为小负OA、小正BA，不能说全面退化，也不能说获得分类胜利。additive的普通NLL最低但OA/BA较差，说明概率评分与top1目标不能混用；NLL/Brier下降也不是未经额外测量的“校准误差已解决”。

所有27头都由sourceval选择非零step：original 70~200，decoupled 60~200，additive 40~55。原运行additive为10~25，说明坐标修正影响了可用学习过程与选步结果；没有手工选checkpoint或根据测试调参。

## 配对区间

固定2000次cluster bootstrap；先对同一实例三seed的指标差平均，再分别按MMSI/过境重采样。下列为修正减原的95%区间，单位pp：

| arm | OA均差 | MMSI OA CI | 过境 OA CI | BA均差 | MMSI BA CI | 过境 BA CI |
|---|---:|---|---|---:|---|---|
| original_centered | +0.091343 | [-0.057530,+0.245795] | [-0.081236,+0.260886] | +0.096509 | [-0.076468,+0.287202] | [-0.099445,+0.311122] |
| score_scale_decoupled | +0.084317 | [-0.162354,+0.314562] | [-0.210037,+0.369787] | +0.320438 | [+0.001796,+0.638875] | [-0.011007,+0.733346] |
| additive | +0.646431 | [-0.328168,+1.689336] | [-0.335161,+1.621719] | +0.404973 | [-0.764508,+1.642296] | [-0.847518,+1.666491] |

不能只报decoupled BA的MMSI正区间而省略过境区间跨零。修正相对A，original OA两种区间均负、BA均跨零；decoupled OA/BA均跨零；additive OA均负、BA均跨零。原运行相对A和arm间的完整区间保留在原目录；修正对应表和原→修正表各自独立。

## 数据组成与剩余损失

修正后各类平均净正确数（相对A）：

| 类别 | test n | original | decoupled | additive |
|---|---:|---:|---:|---:|
| bulk_carrier | 1290 | -25.667 | -3.333 | -48.333 |
| fishing_vessel | 145 | -0.667 | -0.333 | -11.333 |
| general_cargo | 439 | -2.333 | -1.667 | +10.000 |
| product_chemical_tanker | 1480 | -4.000 | -5.667 | -36.000 |
| container_ship | 519 | +7.333 | +3.000 | +12.667 |
| crude_oil_tanker | 405 | +3.333 | +1.000 | -12.000 |
| tug_towing | 202 | +1.000 | +0.333 | +4.333 |
| offshore_supply | 264 | +4.333 | +2.333 | +5.000 |

original主要损失集中在bulk，部分container/crude/offshore收益不足以补偿其数量。decoupled削弱了bulk损失，但仍有product/chemical净损失。additive在general/container有收益，同时损害两个大类与fishing，不能只看总体NLL判定分类机制更好。逐类NLL/Brier也非全面改善，例如门控tug的NLL/Brier均变差。

测试有效VH4436条，缺失VH308条；全队列仍为13783/15251有效VH。修正original在缺失/有效VH组的平均净数分别-1/-15.667，decoupled +1.667/-6，additive +14/-89.667。类别和观测构成不同，这些分层不是删除VH的因果效应。16结构组全保留，缺失VH中心偏移未清零。

采集层保存全部OA/BA/NLL/Brier：original平均净损失较大的Port Klang/S1A_061742_07B768和Richards Bay/S1C_005252_00A6D0各-2；decoupled的Singapore/S1A_061843_07BB4D为-2.667；additive的Singapore/S1A_062368_07CFC8为-14.333。它们用于定位数据群，不据此挑港口或重新定超参。

## 原图与实际读出

原12个案例事件和修正12个事件已分别用view_image实际查看，均展示原160crop/pad、VV/VH和whole实际有效面积mask。C4特征复用，未重新提图；图上不把fg/bg当此次条件输入。原/修正所选案例并集的逐例分数见 `original_vs_fixed_cases.csv`，不是只挑修正后成功样本。

修正运行每臂救回/误伤/CC/WW四例均按预定seed17、obs_id最小选取，无后备换seed。以下分差是同一真类对明确竞争类的固定分差，不等于每次都选同一个top2竞争类：

| arm / 状态 | instance_id | A→修正分差 | 图像事实与数值作用 |
|---|---|---:|---|
| original / 救回 | returned_local_knowledge_0014474 | -0.004033→+0.030329，chemical对bulk | 双pol均为小画布横向亮主体；共同项+0.028932、对比项+0.005430，共同项占主要数值增量。不证明模型识别了确定的物理船型结构。 |
| original / 误伤 | returned_20260915_0000475 | +0.011765→-0.070949，bulk对offshore | 双pol可见斜向亮主体；共同项-0.082239，对比项-0.000475，误伤主要沿共同方向发生，坐标修正后仍存在。 |
| original / CC | returned_20260915_0004683 | +0.701831→+0.979169，general对container | VV底纹较强、目标小，VH也为小亮区；共同项+0.287769，对比项-0.010431，已有正确证据被增强。 |
| original / WW | pair_0032717 | -1.811164→-1.978010，bulk对general | 细长分段亮响应且VV有横向亮线；共同项-0.289883，对比项+0.123037，仍错。原未对齐运行同例为-2.6188，修正减轻但未救回。 |
| decoupled / 救回 | returned_local_knowledge_0023338 | -0.018212→+0.027660，chemical对tug | 双pol为紧凑小亮区；共同项-0.008259，对比项+0.054131，跨界来自对比项，非正标量缩放本身。 |
| decoupled / 误伤 | returned_local_knowledge_0010098 | +0.249706→-0.022618，container对bulk | VV横纵亮线较强、VH亮主体更紧凑；共同项+0.057825，对比项-0.330149翻转。不能把亮线外观独自判为误伤原因。 |
| decoupled / CC | returned_20260915_0004683 | +0.701831→+0.819317，general对container | 与original同原图，共同项+0.181365、对比项-0.063879，仍判对。 |
| decoupled / WW | pair_0032717 | -1.811164→-0.981634，bulk对general | 与上述同原图；共同项-0.213699、对比项+1.043229明显减轻错误，尚不足跨界。 |
| additive / 救回 | returned_local_knowledge_0014460 | -0.460308→+0.643194，container对general | VV有较强延伸亮线、VH较紧凑；直接类残差分差+1.103502。与decoupled误伤图有相似可见响应并不意味着同一机制。 |
| additive / 误伤 | returned_20260915_0001878 | +0.222945→-1.083005，bulk对crude | VV/VH斜向连续亮主体，与baseline正确回看同例；残差分差-1.305950覆盖了原有效类证据。 |
| additive / CC | returned_20260915_0004683 | +0.833499→+0.790868，general对chemical | 同原图；竞争类不同，故不能与门控对container的margin直接相减。 |
| additive / WW | pair_0032717 | -1.811164→-1.366515，bulk对general | 残差分差+0.444649，错误减轻但未救回。 |

对已训练original头，令a=m+c，则score=(1+m)A-m*b+sum(c*q)。所以共同门控不是对含b的A纯正缩放；固定b的相对变化可以翻转argmax。上述original误伤/救回的分解支持这个**局部数值路径**。decoupled的score=(1+m)*(A+sum(c*q))，正共同缩放本身不改内部argmax，所选误伤由contrast造成。两臂重新训练后a也不同，不把代数分解说成跨模型训练因果证明。

正确和错误都可能被共同正幅度增强；因此概率评分改善与错误置信度增强可同时发生。修正后门控平均m约0.40~0.44，对比RMS低于共同幅度但并非0。没有在本轮另训constant控制，不能据此声称八类必须样本自适应，亦不能据均值大就断言完全不自适应。

## 验证、成本和边界

- 原/修正均27头：门控2608参数，additive2344参数；不人为配平。9组arm/seed各4744 test；`original_vs_fixed_predictions.csv` 精确为 **42696** 行，即4744x9，不能视为42696独立样本。
- 原/修正的标签、身份、fold、test下标、有效VH和全部baseline分数逐项相等。原数据与原输出未改。参数和选步变化来自同一既定训练过程接收不同条件坐标。
- zero-init精确回到A；所有保存分数/控制值有限；18个门控头逐样本正共同缩放argmax不变；decoupled完整输出与其内部分数argmax相同。
- 修正27头checkpoint重载全队列重放最大误差0；float32训练公式与float64固定读出最大分数差6.64e-7，test top1差0。没有step0被选中；零初始化的精确回A路径实际验证，不把它冒充观察到了step0选择。
- 修正训练总墙钟22.32秒，其中头训练/验证约17.53秒；原运行22.34秒。坐标诊断约0.27秒。配对统计/原图渲染另计，所有模型/缓存仅本地。
- 原/修正代码、三折逐seed指标、逐类/采集/有效VH、NLL/Brier、配对区间、原图路径、组贡献均保留。不新增超参、不重训base/PCA、不改父目录、不Git。

**能保留的结论**：原运行确有条件坐标不一致；统一源条件坐标改善了本次OA均值，原负结果不可单独否定方法。修正后的positive evidence gate较普通additive更少伤害top1，decoupled更接近A；两种门控仍只是小正BA、负OA，且相对A的BA区间跨零。尚不能确认八类分类净增益，更不能声称完整模型胜利。剩余损失的局部数值路径、类别集中与原图现象已定位，但没有证据把全体错误归于唯一因素。

## 主要文件

- `PROTOCOL_ZH.md`：训练前固定纠错协议。
- `diagnose_coordinates.py`、`coordinate_summary.csv`、`coordinate_samples.csv`、`coordinate_dimensions.csv`、`coordinate_axes.csv`、`coordinate_diagnosis.json`：坐标证据。
- `run_transfer.py`、`models/`、`readouts/`、`predictions.csv`、`metrics.csv`、`pooled.csv`、`mean_metrics.csv`、`curves.csv`、`verification.csv`：修正训练与重放。
- `analyze_transfer.py`、`class_metrics.csv`、`acquisition_metrics.csv`、`vh_metrics.csv`、`month_metrics.csv`、`paired_intervals.csv`：固定结果分析。
- `compare_original.py`、`original_vs_fixed_metrics.csv`、`original_vs_fixed_predictions.csv`、`original_vs_fixed_intervals.csv`、`original_vs_fixed_cases.csv`、`implementation_change.diff`：原/修正精确配对。
- `review/`、`review_cases.csv`、`review_group_contributions.csv`：12个已实际查看的修正图像事件及可追溯分差。原12个事件仍在相邻原目录。
