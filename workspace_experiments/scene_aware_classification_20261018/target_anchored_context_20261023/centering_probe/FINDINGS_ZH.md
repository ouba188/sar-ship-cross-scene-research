# 中心锚定与原特征门控：配对结果

完成日期：2026-10-23。本轮全部新增内容位于 centering_probe；不改父目录代码/缓存，不 Git。两个固定新臂、六折、三种子，共36头已训练、保存、重放完成。

## 结论

**在当前固定网络、训练预算与源验证选择下，移除中心后两臂总体均低于各自 centered 对照。** 外部臂相对 baseline A 净收益从父 +2/+7/+11 变为 0/-5/-14；own-full 从 +14/+8/+16 变为 -25/-27/-11。结果支持保留本设置的中心锚定，但不证明其对所有特征门控/SE 网络均必要。

不能说 uncentered 全面退化：相对 A，uncentered own-full 三种子平均 BA 仍 +0.143pp；相对 centered own-full，它在有邻居的1165目标子集上平均多判对7.67个，损失主要在无邻居4530目标子集。外部臂也有 centered 未救回而 uncentered 救回的目标。

## 匹配条件与实现验证

沿用5695目标、4312 MMSI、163采集组及 expanded 六折。外部臂沿用父 has 激活1165目标，own-full 激活全体。输入 env 和 own env 原样取父缓存；无新提像素/特征、检测、标签处理或范围重选。

导入父 Anchored 网络：79->32->32、.5*tanh、末层零初始化、200步、种子17/29/43、AdamW lr=.003/weight_decay=.01、原类别/MMSI权重。每5步按源验证 balanced NLL 比较，改善阈值1e-6，保留0步精确回退。参数3616，与父一致。

唯一设计变化为 qraw_g=W_g h_g=qcenter_g+origin_g，其中 qcenter_g=W_g(h_g-h0_g)，origin_g=W_g h0_g。原 baseline base/oof **直接使用父缓存精确值**，没有用重拟合分数替换。

- 6个outer、24个源OOF fit分别调用 run_strong.View/combine/fit_head，原特征保持float32拟合；selected kind/alpha不重新选择，显式W及h0计算为float64。
- 每个OOF held只使用对应fit的origin，fit遵循原采集组及MMSI排除。outer与OOF分别保存fit/held、W、h0和origin。
- 分组为原通道：stem 16组，每极化8组，每组8通道的mean/max绑定；deep按原顺序16组。没有PCA轴分组。layer3-only折stem前16组为零，保持原32组结构。
- centered组和+A(h0)与直接线性读出最大误差5.33e-15；raw组和+[A(h0)-sum(origin)]最大误差8.52e-13；qraw=qcenter+origin误差2.84e-13。
- 30次refit的q/oq与父缓存最大差0，分数最大差0，top1变化0。保存的新输入中base/oof逐元素精确等于父缓存。
- 36个磁盘checkpoint重放最大误差0、top1变化0；无激活目标的外部增量严格为0。fold4外部三种子均选0步，输出严格回到A；其余头均有源验证选中的非零步。
- 保持父流程的OOF边界：baseline/group reader内层重拟合；env的外层源投影缓存没有改成全链条内层OOF。本实验只隔离中心化变化，不增加新的全链条OOF声明。

## 测试结果

所有数值为5695目标的pooled测试结果，非训练/验证表现。每目标每臂每seed恰有一次测试预测，三个seed不是三份独立数据。

| 臂 | seed | OA | BA | 相对A净正确数 |
|---|---:|---:|---:|---:|
| 父 anchored | 17 | 81.405 | 80.047 | +2 |
| 父 anchored | 29 | 81.493 | 80.113 | +7 |
| 父 anchored | 43 | 81.563 | 80.196 | +11 |
| uncentered_external | 17 | 81.370 | 79.901 | 0 |
| uncentered_external | 29 | 81.282 | 79.931 | -5 |
| uncentered_external | 43 | 81.124 | 79.693 | -14 |
| 父 own_regions_full | 17 | 81.615 | 80.457 | +14 |
| 父 own_regions_full | 29 | 81.510 | 80.421 | +8 |
| 父 own_regions_full | 43 | 81.651 | 80.436 | +16 |
| uncentered_own_full | 17 | 80.931 | 80.014 | -25 |
| uncentered_own_full | 29 | 80.896 | 80.118 | -27 |
| uncentered_own_full | 43 | 81.176 | 79.850 | -11 |

A为81.369622/79.851353。三种子均值：uncentered_external为81.258414/79.841761，uncentered_own_full为81.000878/79.994299。

### 与各自centered父臂配对

| 新臂 | seed | 救回父错误 | 误伤父正确 | 净变化 | OA差pp | BA差pp |
|---|---:|---:|---:|---:|---:|---:|
| external | 17 | 35 | 37 | -2 | -.0351 | -.1460 |
| external | 29 | 32 | 44 | -12 | -.2107 | -.1828 |
| external | 43 | 31 | 56 | -25 | -.4390 | -.5030 |
| own-full | 17 | 106 | 145 | -39 | -.6848 | -.4421 |
| own-full | 29 | 113 | 148 | -35 | -.6146 | -.3025 |
| own-full | 43 | 105 | 132 | -27 | -.4741 | -.5855 |

配对区间先对同目标三seed正确性差求平均，再沿用父bootstrap：2000次，seed20261022，按采集或MMSI分别分组。不是联合双向聚类区间。

| 新臂减父臂 | 均值OA/BA差pp | 采集OA 95%CI | MMSI OA 95%CI | 采集BA 95%CI | MMSI BA 95%CI |
|---|---|---|---|---|---|
| external - anchored | -.2283/-.2773 | [-.5168,+.0621] | [-.4911,+.0235] | [-.6616,+.0757] | [-.5840,+.0171] |
| own-full - own-full | -.5912/-.4434 | [-1.0896,-.0633] | [-1.0145,-.1392] | [-1.0058,+.0609] | [-.9121,+.0342] |

own-full的OA下降在两种分组区间下都不跨0；BA仍跨0。external虽三seed都低于centered，但区间跨0，不能包装为强统计定论。

## 数据分层与作用位置

相对各自父臂，三seed平均每类净正确数：external bulk -5.33、fishing -7.33、general -.33；own-full bulk -32.00、fishing -13.67、general +12.00。own-full召回从父84.942/86.922/69.448变成83.933/85.731/70.319，类别竞争偏移并非所有类同降。

分有/无外部邻居：external差异全部发生在1165有邻居目标，净-13.00；4530无邻居目标严格无变化。own-full在有邻居子集相对父净+7.67（OA+.6581pp、BA+.6760pp），无邻居子集净-41.33（OA-.9124pp、BA-.7228pp）。这是样本条件关联，不证明邻居存在本身导致效果。

163采集组按三seed平均净变化统计：external 16组改善、31组下降、116组持平；own-full 36组改善、52组下降、75组持平。仅作为差异分布，不将每组视为等样本量证据。

## 为什么不能仅称普通缩放完全等价

对同一组门控系数a，去中心增加的是sum_g a_g origin_g，它依赖输入环境和共享门控，不是父实验中的独立class-bias头。新旧重新训练后的系数不同，精确分解为：

`score_raw-score_centered = sum_g (a_raw-a_centered)*qcenter_g + sum_g a_raw*origin_g`。

两项分别称“门控变化项”和“origin项”，只做固定checkpoint代数读出，没有另训或删项选择。36头分解对实际分数差最大误差1.05e-6。

存在真实的尺度变化：fold4的stem group7，source centered贡献RMS为.07350，origin RMS为213.91347；group15为.08733对216.49376。group7在bulk-vs-general上的origin差为491.29971。原通道57在source的均值3.607516、标准差.000578，显式W类差135.156182，单通道origin类差487.578062。中心化时主要使用很小的偏离；raw门控会调制较大的源均值贡献，原线性截距只在不门控时抵消它。

这不是分组/重放失败，也不是引入新信息：W及中心化q与父完全相同。它说明去中心会改变有效偏置及优化尺度。例如下述own案例group7的raw gate约-.001，而父gate约±.49；很小raw系数也可产生约.5的类分差变化。仍不能据此把总体退化全部归因于这个组、宣称已隔离“偏置”与“优化条件”两种解释，或外推为所有SE门控无效。本轮未追加归一化、删组、重调学习率等实验。

## 已实际查看原始VV/VH的八例

种子29，按相对A的救回/误伤/CC/WW各取一例/臂，先限active，再优先已有父图、最后ID排序。分类与分差来自本轮表；复用父图上的文字是父模型结果，不能当作新结果。7张父图直接引用，仅外部救回缺少父图，新增 `case_pair_0052473.png`。八张均已用view_image实际查看；每张分开显示原尺度VV/VH。外部图显示最近三个或全部不足三个邻居，模型仍使用所有合法节点，未重读整景。

表中分差为真类减当前最大其他类，A/center/raw依次为原强基线、相应父臂、新臂。

| 臂/类型 | ID | 真类 | A/center/raw分差 |
|---|---|---|---|
| external救回 | pair_0052473 | bulk | -.19128/-.10234/+.00758 |
| external误伤 | pair_0031576 | general | +.15278/-.07063/-2.01156 |
| external CC | pair_0032599 | general | +1.50756/+2.11145/+1.95264 |
| external WW | pair_0044888 | fishing | -2.80106/-3.83804/-3.20825 |
| own救回 | pair_0031587 | general | -.25602/+.47145/+.41605 |
| own误伤 | pair_0030918 | bulk | +.06723/-.11442/-.00399 |
| own CC | pair_0030906 | bulk | +1.43839/+1.19718/+1.11671 |
| own WW | pair_0031200 | bulk | -.46155/-.87514/-1.06536 |

- **0052473**：VV/VH中心均可见横向紧凑亮主体、背景较暗；显示的三个真实邻居也是不同大小的亮主体，没有从图上识别其类别。raw以很窄分差救回父模型仍错的目标。对固定A竞争类，门控变化项+.20810、origin项-.09819，总+.10991。此例净改善来自两项合成，不是origin项单独为正。
- **0031576**：VV/VH都可见多亮结构及明显颗粒起伏，显示的邻居窗也有较亮区域。center误判fishing，raw误判bulk，说明最强竞争类切换。general-vs-fishing分差从-.07063到+.22184，origin项+.55301；但general-vs-bulk从+.41063到-2.01156，门控变化项-.55682、origin项-1.86537。只看旧竞争类会错误地说“证据改善”；实际top1严重变差。图像不能证明岸线、海况等特定物理原因。
- **0032599**：VV/VH可见短小斜向/横向亮结构，唯一邻居在VH中较弱，不能由邻居外观推出其类别。center/raw均保持general正确；门控变化项-.08048与origin项-.07833使分差稍减但未翻转。说明一定幅度负增量不等于误伤。
- **0044888**：两极化中心是紧凑亮结构，两个邻居有细长亮结构；与某些货船案例外观相似，但不据此改标签。center/raw都误判general；门控变化项+.15494、origin项+.47486让raw比center的分差好+.62979，仍未克服原来大的错误分差。这是保留的raw局部改善反例。
- **0031587**：无外部邻居，VV周围纹理较亮，VH较暗，目标是近横向亮主体且附近有小亮点。center/raw都救回general；门控变化项-.69890、origin项+.64350大幅抵消，总分差略低于center。group7的origin贡献+.56829，并不意味着该组物理上识别了船型。
- **0030918**：无邻居，VV可见穿过主体的横向条纹，VH主体更紧凑。A对bulk仅有+.06723优势；center/raw均误判general，但raw只差-.00399。门控变化项+.26482、origin项-.15438，整体比center更接近正确。不能把全部误伤都归于raw独有问题。
- **0030906**：无邻居，两极化为明显较长的近竖向亮主体，VV背景更亮。三者均对bulk；门控变化项+.23678、origin项-.31725，raw分差比center低但仍有1.11671优势。可保留的条件是原类别分差充足时本轮扰动未改变结论，不称机制已证。
- **0031200**：无邻居，VV中心亮主体上方有串联亮点且纹理较亮，VH主体更窄、背景更暗。三者均误判general；门控变化项+.45769却被origin项-.64791抵消，raw比center再差-.19022。这是origin项可抵消有利门控变化的具体模型证据，不能仅凭原图将串联亮点命名为某种物理散射。

案例是事后描述，不代表总体占比；已保留raw成功、center/raw共同失败与持续正确的范围。不把八例强行改写成中心化总是必要的故事。

## 文件与完成状态

- `run_uncentered.py`、`analyze_centering.py`、`PROTOCOL_ZH.md`：独立运行/分析脚本及固定协议。
- `models/`：36个新checkpoint；`origins/`：30个源拟合的W/h0/origin/fit/held；6份 `fold*_inputs.npz`：新贡献缓存及原baseline精确值。
- `predictions.csv`：34170行；`metrics.csv`：36头；`pooled.csv`：各seed与有/无邻居分层；`curves.csv`：1440个验证点。
- `paired_predictions.csv`、`paired_metrics.csv`、`paired_intervals.csv`、`class_effects.csv`、`acquisition_effects.csv`：与父臂和A配对。
- `origin_verification.csv`、`group_origins.csv`、`replay.csv`、`decomposition_verification.csv`：分组、源OOF、checkpoint及分解证据。
- `origin_scale.csv`、`stem_origin_channels.csv`、`modulation_summary.csv`：原点尺度与通道门控统计；`case_group_contributions.csv` 使用固定A竞争类，`case_rival_decomposition.csv` 同时列全部其他类以避免竞争类切换误判。
- `review_cases.csv`：八例当前分差、预测、原图绝对路径与复用父图路径；1张新增图，其余7张仍保留在父目录。
- `summary.json`、`analysis_summary.json`、`run.log`、`analysis.log`：运行记录。训练与30次源reader重拟合合计54.50秒；两次分析均无新拟合，第二次仅补全所有竞争类与origin尺度表。

本轮目标是中心化对照，不是新模块搜索。当前证据支持主流程继续保留centered版本及其小正结果，同时保留raw的局部收益和不确定区间。没有后续实验或Git操作。
