# Seed17 原图回看与初步证据

本文件保留首轮 seed17 的观察，不随后续 29/43 汇总改写。已在 CPU 分析子任务实际查看下列 13 个 query 的 26 张 VV/VH 面板，包括各面板中实际 reference bank 的同类和混淆类 refs。其他已生成面板未在本文件声称已审阅。原生图路径、尺寸、crop/pad、有效性与实际 ref ID 位于对应比较目录的 `native_sources.json`、`cases.json`；图像按原像素尺度显示，OBB 另列覆盖。下表文件位置以 `analysis/pilot17/` 为根，VV/VH 文件尾分别为 `_vv.png`、`_vh.png`。该目录从同一轮未改动的seed17预测和实际refs重建，保留原选择与来源；最终三seed面板在analysis/private和analysis/oss2，独立于本快照。

选择规则为每个比较、每个状态先按跨 seed 状态出现次数，再按固定 rival margin 变化绝对值、ID、seed 排序，取不重复 query；本轮仅 seed17，无随机抽样。四状态缺失如实记录，不补造案例。以下可见事实不是已证明物理机制，代表样本也不是总体比例。

|数据/ID|已查看面板前缀|可见事实与实际 refs|seed17 模型证据及边界|
|---|---|---|---|
|private 31926|private/ce_vs_saved_baseline/rescued_31926|VV/VH 为较长近竖直亮体；同类原油船 refs29922/2077较短且近横向，bulk refs34478/31512也有长亮体。单凭外观不能可靠分型。|真类对bulk margin baseline -0.3823，CE +0.2879，hybrid +0.1955，rank -0.4162。CE/hybrid为新增救回，rank未保留；CE margin新增+0.6701中共同项+0.4284、残差+0.2417。|
|private 24863|private/ce_vs_saved_baseline/harmed_24863|VV/VH为小型斜亮体，chemical refs24527/7790和tug refs12041/7429也含小型点状或短条状目标。|chemical对tug从+0.3469变为CE -0.3299、hybrid -0.1860，rank +0.4992。CE共同项-0.7612，残差+0.0844；此例误伤可由共同偏移跨越小边界解释，不是支持图缺失。|
|private 22789|private/ce_vs_saved_baseline/CC_22789|紧凑亮点；同类offshore refs13000/13276也紧凑，bulk refs34478/31512较长。VV有明显延伸线状响应，VH较弱；不命名其物理来源。|四模型都正确。offshore对bulk baseline+1.4990，CE+3.4127，rank+1.7228，hybrid+3.1842。CE的+1.9137新增margin含共同项+1.6136，不能把持续正确当新救回。|
|private 28198|private/ce_vs_saved_baseline/WW_28198|VV/VH亮体横向延伸，VV旁边还可见线状响应/另一亮结构；与紧凑同类refs明显不同，与bulk34478有局部外观相似。|真实offshore相对bulk baseline -19.0410，CE改善到-17.1726仍很负。baseline/rank判bulk，CE/hybrid改判crude仍错。涨margin不等于可用分类证据，视觉相似不是标签错误证明。|
|private 36526|private/rank_vs_ce/rescued_36526|VV/VH斜短亮体，chemical与tug refs均有近似形态和不同方向，不足以从外观确立唯一机制。|baseline原本正确，CE chemical-tug=-0.0049，rank+0.9634，hybrid+0.1721。相对CE救回是恢复原有正确；hybrid-CE新增+0.1769中共同项+0.1617。|
|private 33658|private/rank_vs_saved_baseline/harmed_33658|VV为较长斜亮体；VH因manifest标记对齐无效而不可用，不是观察到黑图。bulk refs为长亮体，fishing refs为更弱小的亮点。|baseline bulk-fishing仅+0.1168，三arms均误判fishing；rank margin变为-0.1264。共同项-0.1186与残差-0.1246均不利。此例不能说rank损失避免了bias或证明缺VH就是根因。|
|OSS2 001098|oss2/ce_vs_saved_baseline/rescued_pub2_opensarship2_001098|VV/VH细长斜亮体，同类general refs004017/005079和bulk refs006462/006477均有小尺度条状/点列，方向与尺寸不同。|general-bulk从-0.5230到CE+0.2161、hybrid+0.1321，rank仍-0.4347。CE新增+0.739068几乎全由共同项+0.739248复现，不能据此称关系选择改进。|
|OSS2 003355|oss2/ce_vs_saved_baseline/harmed_pub2_opensarship2_003355|VV/VH短横亮体，同类chemical refs006746/005117较斜，general refs004017/005079也含紧凑点列。|chemical-general从+0.3079到CE -0.6097、hybrid -0.5315，rank仍+0.2401。rank-CE的+0.8498中共同项+0.7816；rank相对CE救回但相对baseline并非新增正确。|
|OSS2 003502|oss2/ce_vs_saved_baseline/CC_pub2_opensarship2_003502|VV/VH短亮点列，VV垂向延伸较显著；同类和general refs同样小而有多亮点。|四模型都正确，chemical-general margin从baseline+2.3245降至CE+1.3568、hybrid+1.4375，rank+2.2418。较大初始margin使同方向不利共同项尚未翻转。|
|OSS2 001130|oss2/ce_vs_saved_baseline/WW_pub2_opensarship2_001130|VV/VH小而横向的亮点列，实际chemical和general refs均存在相似点列，肉眼不足以判别。|始终误判general；真类margin baseline -2.4932，CE -3.5137，rank -2.5275，hybrid -3.4026。rank减轻CE损害但未纠错；不把减轻损害写成新识别能力。|
|OSS2 003555|oss2/rank_vs_saved_baseline/rescued_pub2_opensarship2_003555|VV/VH近竖直细长多亮点，与container005219有形态相似；另一同类001541为横向，bulk refs也含点列。|baseline container-bulk=-0.1188，rank+0.1880，CE/hybrid也正确。rank新增margin+0.3068中共同项+0.2402，即共同偏移单独足以跨阈；图像相似不能证明这次更新学会方向匹配。|
|OSS2 003461|oss2/rank_vs_saved_baseline/harmed_pub2_opensarship2_003461|VV/VH斜向点列，bulk与container refs形态有重叠且尺度不同。|baseline bulk-container=+0.1451，rank -0.0339，CE/hybrid同样误伤。rank共同项-0.2402、残差+0.0612，残差减害仍不足。和003555构成同一类偏移的相反收益。|
|OSS2 003623|oss2/hybrid_vs_ce/rescued_pub2_opensarship2_003623|VV/VH近横亮点列，bulk006462/006477也有横向点列；container bank也存在横向001541，不能单凭方向断言分类机制。|baseline原本正确，CE bulk-container=-0.0266，hybrid+0.0404。hybrid-CE总改变量+0.06699，共同项-0.01993、残差+0.08691，残差决定本例救回；这是相对CE恢复而非相对baseline新收益。|

## 首轮结论边界

开发集合法跨MMSI、跨pass、异类query pair：private共6262对、21个有数据类对，fishing缺失导致7个类对缺失；OSS2共10198对、6类对齐全。它们不是独立样本数。每个有数据类对内部均匀平均，然后类对等权；缺失项为null，未补零。评价只读本轮开发预测和原manifest，未用于训练或选样。

|数据|baseline pair logistic/rank accuracy|CE|rank|hybrid|
|---|---|---|---|---|
|private|1.029460 / 0.781547|1.011240 / 0.783610|1.026278 / 0.780711|1.013539 / 0.782682|
|OSS2|0.547336 / 0.840461|0.550999 / 0.839633|0.542209 / 0.839754|0.548821 / 0.840178|

rank的pair logistic两域均改善，但rank accuracy两域都小幅下降，尚不能把源排序改善等同开发排序正确率改善。private CE/hybrid在开发pair logistic上的改善大于rank；公开rank优于CE/hybrid的pair logistic。还需29/43检验重复性，不能将seed17描述为最终三seed结果。

rank相对baseline新增logit在去掉逐query全类平移后，共同偏移能量占private73.17%、OSS2 81.28%。OSS2共同偏移单独复现rank全部228个top1以及3个翻转（1救回、1误伤、1错类切换）。pair gap对共同类偏移不敏感是损失的代数性质，参数更新仍可能产生这种副作用。

支持的结论是“本轮目标与步数下，源排序、开发pair logistic、rank accuracy和最终分类有不同收益边界，部分相对CE优势是避免其类偏移损害”。不支持“M3根本不可用”“query-specific更新完全无效”或特定物理机制已证明。003623的残差救回本身就是后一种过强结论的反例；目前没有足以据此推广的新增分类净收益。
