# Relational Kernel：四例固定读出复核

## 结论

四张原VV/VH及实际源参考面板已全部`view_image`查看；新局部算子、保存的源scaler/checkpoint及最终分数均直接重放通过。**没有新训练、骨干提取、换参考或新选点。**

seed17的role_contrast相对ordered救回组116条、误伤组108条、CC1263条、WW1588条；按各组最小obs_id固定选85、353、35、189，不按分差挑案例。四例不是总体比例估计，更不是空间关系必要性的确认。

最重要的病例边界：85是保住full原已正确的判断，避免ordered误伤，不是从full新救回；353是role_contrast对full及ordered均造成的新误伤；35持续正确但余量下降；189在已采用结构有效参考均值后仍错且分差恶化，不能将其全部失败归给旧缺失分母。

## 实现与直接重放

`analyze_cases.py`直接调用当前`run.py`的`project`、`compute`、`raw_inputs`和Head，加载`models/{arm}_17.pt`，比较`external_local.npz`及保存的`results/external_{arm}_17.npz`。source与external的打乱随机序列按原始全表长度/顺序生成，再取本次四行；没有重新随机打乱来挑结果。

|臂|已选步|新算子对缓存最大误差|最终head分数最大误差|
|---|---:|---:|---:|
|anchored|55|1.1921e-7|4.7684e-7|
|ordered|50|1.1921e-7|2.3842e-7|
|role_contrast|100|1.1921e-7|5.3644e-7|
|mispaired|90|1.1921e-7|7.1526e-7|

四臂top1完全重放，结构有效参考数与缓存逐项完全相同。此seed四臂均非零步；不能忽略它们选步和已学习head不同，把最终差异解释成同一head上单独减一个交叉项。

局部数值核查使用当前固定body运输T，side/end项分别为：

`O_ij = <b_i,b'_j> <n_i,n'_j>`

`X_ij = <b_i,n'_j> <n_i,b'_j>`

`K_ij = O_ij - X_ij`

先按相应区域支持mask汇总`T*O`、`T*X`、`T*K`，再按当前结构可用参考数求均值。body仍为原`T*<b,b'>`，不是将body也改成平方核。四例均满足`wedge=ordered-cross`，最大闭合误差4.47e-8。真实OT拒配质量没有被重新归一；结构缺失与语义拒配保持区分。

**这些是局部输入描述项，不是非线性MLP最终logit的可加归因。** 图中列出的true-minus-fixed-rival输入差，可解释进入head的量如何改变，不能将其直接逐项相加解释预测分差。

## 四例分差与竞争类

rescue85的固定竞争类取ordered的实际错误类2；harm353取role_contrast实际错误类4。CC/WW取role_contrast最高非真类。另保存每个模型自己的final rival及own-top1 margin，避免固定竞争类margin为正却误称模型一定正确。本次四例这两种竞争类恰一致。

|病例/真类|固定竞争类|full margin|ordered margin|role_contrast margin|mispaired margin|
|---|---:|---:|---:|---:|---:|
|85 rescue / bulk0|general2|+.425735|-.188889|+.029715|-.046041|
|353 harm / bulk0|container4|+.061878|+.089411|-.015198|+.004238|
|35 CC / general2|chemical3|+.418270|+.528102|+.092228|+.174674|
|189 WW / general2|fishing1|-1.333109|-2.414468|-2.765092|-2.535424|

### 85：相对ordered救回，实为保护full

实际真类参考21013/26615，竞争参考3536/7884；VV/VH均两条有效。query为长窄、近纵向的多个亮响应，真类参考更集中且朝向不同，竞争参考包含暗短体和亮短体。可见差异不支持简单按外观给出类别因果。

side输入margin在VV由ordered -.14709减cross -.09031得到wedge -.05678，VH由-.19048减-.11317得到-.07732；end的VV/VH wedge为+.04196/+.02012。交叉项改变了输入对比，但并非所有区域都变为真类优势。最终role_contrast保留+.029715，而其对full的残差分差实际为-.396019；不能说该头创造了原本不存在的正确证据。

### 353：真实误伤，不能由区域命名保护

真类参考33678/23781，竞争参考21323/8286，全部极化可用。query、bulk参考和container参考均可见长亮体；VV的宽亮区域与VH的窄亮条存在差异，没有某个单一可见布局能直接解释标签。

VV side输入margin由ordered -.01035减cross +.04709变成wedge -.05744；VH side反而由+.00359变成+.00561。最终role_contrast由full的+.061878减去.077076而误判container，ordered与mispaired仍正确。不能把一个极化的输入改善直接等同于最终分类改善，也不能说角色交叉扣除普遍抑制混淆。

### 35：持续正确，role_contrast余量更小

真类参考21927/12625，竞争参考9086/24840，全部极化可用。query亮体较短，VV背景更亮、纹理更强，四个参考目标整体更长；展示的上下文与主体差异不等于物理机制。

VV/VH side wedge输入margin约-.00800/-.01557；end约-.02037/-.01784。full、ordered及role_contrast都判general，但role_contrast最终余量从full+.418270降为+.092228，不能用“仍然正确”掩盖该例不利分差变化。

### 189：有效均值处理后仍持续错误

真类参考3536/5626，竞争参考15663/16248。真类5626的VH局部输入无效；当前新算子的VH真类参考计数为1，竞争类为2，VV均为2。panel显示原PNG并明确标记无效，没有将无效局部输入画成可用匹配。

query的目标及背景整体偏暗，真类3536也偏暗、5626较亮；fishing参考有弱而细的亮响应。不能从相似亮度直接断言分类物理原因。VH side由ordered -.01991减cross +.02599变为wedge -.04590，end wedge为+.00401；VV side wedge仍约-.14022。role_contrast最终margin -2.765092，比full和ordered都更不利。

因此本例说明：**修正结构缺失平均并不自动解决可迁移类别证据或残差利用问题**。本轮没有用其外部结果调尺度、改参考或启动新训练。

## 全局边界：不能声称空间身份必要

主输出中，mispaired的external BA三个seed均高于role_contrast：

|seed|role_contrast BA|mispaired BA|
|---|---:|---:|
|17|38.8974|39.3659|
|29|40.5009|40.7367|
|43|38.9331|39.0976|

85中mispaired变错、353中mispaired保住正确，只说明病例作用并不一致，不能反过来以85一例宣称真实空间依附必要。当前mispaired只打乱side对body的依附，end等仍保留；它也不是彻底删除一切空间信息的对照。应保留这组阴性机制证据，同时不将其扩写为所有局部关系都无用。

该3075队列为已开发历史目标的身份/过境隔离子集，不是新盲测；本次仅完成固定病例解释，主流程负责总体统计、数学审阅与归档。

## 文件与完成状态

- `../analyze_cases.py`：ASCII、仅冻结推理及四病例生成。
- `case_selection.csv`：四组计数、最小obs_id选择规则、实际错误类与final rival。
- `case_85.png`、`case_353.png`、`case_35.png`、`case_189.png`：四张已实际查看的2×5原VV/VH及真实参考面板，完整160裁补视野，无运输连线。
- `case_margins.csv`、`case_scores.csv`：完整类别分数、固定margin、各自top1 margin和相对full残差分差。
- `local_terms.csv`、`per_reference_terms.csv`：全部八候选类的ordered/cross/wedge、真实匹配量、结构可用计数及源参考ID；不是只存解释所需两类。
- `image_records.json`：实际PNG、source/external索引、原尺寸、裁补位置与局部有效标记。
- `replay_checks.json`、`run_cases.log`：直接算子/冻结head/结构计数/项闭合回执。

本次四例完成后停止，无新训练、无运行进程、无父实验改动、无Git。
