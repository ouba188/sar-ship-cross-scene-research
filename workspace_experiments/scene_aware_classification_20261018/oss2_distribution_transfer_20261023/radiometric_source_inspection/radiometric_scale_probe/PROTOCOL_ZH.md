# 显示锚尺度source-only固定九头

2026-10-23，首次实现/训练前冻结。只写本子目录，沿用fold0 source train1542/validation279；不读test用于匹配/推理/评价，不Git、不扩特征或参数网格。此协议取代上级草案中的log-max候选，不修改上级文件。

## 唯一新增信息

主流程无标签像素拟合p≈clip(a*sqrt(f),0,255)，a在0<p<255且sqrt(f)>0像元上through-origin最小二乘得到。现有全部band fitted MAE最大约.3797，rounded match中位>.9986，解释了单纯maxnorm的部分偏差；不宣称找到官方pipeline。仅使用source_correspondence已保存的amplitude_anchor_255=255/a，新增s_p=log(255/a_p)两维，按VV/VH排序。不是max，不增加拟合残差、饱和率、float分位数或其他辐射统计作为模型输入。max仅描述。单位未核实，不称Sigma0/RCS/绝对物理散射尺度；饱和像元的原值也不能由该锚精确恢复。

共同12维：[centered B八类分数8，native PNG log RMS两极化2，log(H*W)两极化2]。RMS从PNG/255原有效像素计算，无padding，无float均值反推。追加两个尺度槽成为14维；共享14→16 GELU→8，376参数，末层0，输出按八类减均值后加冻结B。

三臂png_summary槽为0/0、own_scale槽为本图s、matched_scale槽为匹配对象整组s。相同seed初始化一致，共同12槽完全相同；scaler仅源train拟合。未输入产品/pass/MMSI/真类/AIS/heading；匹配用身份和几何控制，但这些不是预测输入。名义容量相同，summary的有效输入维数较少。没有新R50/reader/T拟合。

## 单一固定匹配

train/validation分别匹配；train还在原inner reader内。同pass_group、同两极化精确H/W、不同MMSI才合法。使用固定seed17的centered B八维和两PNG log RMS作为距离坐标，逐维尺度只由source train估计。每块全部合法候选边按标准化欧氏距离、两端排序ID排序，贪心最短未占用边互配；无随机重试或距离阈值，三seed共用donor映射，不看标签/float尺度/拟合输出选donor。

整组VV/VH尺度互换，块内尺度经验分布不变。不匹配者保持本图尺度，不删除对象、不放松规则。保存覆盖、距离、未匹配原因、按类别/采集/尺寸实际构成和事后同类率；报告全体和预定matched子集，不以子集结果重选epoch。该对照限制采集和尺寸混杂，但不是精确条件抽样，不能声称排除全部类别先验或处理因素。matched方案只是诊断控制，不部署。

## 预算与合法数据流

3臂×seed17/29/43=9头，各200步jointCE=.5普通CE+.5八类均衡CE，AdamW .003/.01。每5步sourceval精确J=(OA+BA)/2含0，平局最早。各seed使用自己的原B：train_base对应源OOF reader分数经原T，validation用outer对应B。T/本头不是二级OOF，不能称整流水线OOF。

没有目标域选择/额外epoch/第二匹配规则；全部1542/279始终保留。模型初值=B，保存selected、完整曲线、train/val预测与replay。核对ID/极化/源reader及MMSI/pass隔离、anchor有限正数、同seed初值、匹配合法性/分布保留、scaler与checkpoint重放。无额外bootstrap（单开发fold）。

## 报告和停止

三臂对B及own对两个控制，报告源validation OA/BA/NLL/balancedNLL/Brier、类别、pass、matched子集；完整训练曲线包含0步和terminal，避免把回退说成未学习。固定seed17 own相对B四结局首ID，实际查看query和其donor的VV/VH PNG/同极化共享范围Patch_Cal log显示；未匹配明确自己、缺失结局明确无，不按GT选donor。图显示三头预测以区分借用尺度与重新训练的控制。

未来own推理仅本chip PNG+来源兼容且极化/配准明确的Patch_Cal，通过同一无标签像素公式取得anchor，使用冻结source scaler/B/头；不借其他对象统计。float/任一pol缺失、非有限/负值、顺序不明、配准不符、无可用未饱和正像元或拟合斜率非有限/非正，则整个增量回退原B，不填epsilon/伪尺度、不新增缺失头。数据单位/处理来源不兼容也不能声称可直接迁移。

正信号也只支持这批source里显示锚尺度的条件性补充用途，不是官方显示算法或物理散射机制证明，不是普通校准创新。完成九头、原图和FINDINGS后停止。

## 九头完成后授权的固定头诊断

保持各seed已选own_scale头/epoch/scaler/B不变，仅推理时替换为现有donor两anchor；不重训、不重选点、不改变三臂。先重放原own完整分数，再报告279全validation与214已匹配子集的OA/BA/NLL/Brier、改判及救回/误伤。29/43若为零头必须保持零效应，不换checkpoint。此为posthoc固定head输入干预，可能带来条件输入分布偏移，不取代三臂公平训练比较，不等于真实物理干预。另查看主流程指定005340/005344及首次swap救回/误伤的实际query/donor。
