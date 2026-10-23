# 冻结完整链外部推理交付

2026-10-23 23:01:31（Asia/Shanghai）完成。零训练、零新选点、未运行局部匹配或local分类头；未修改父目录、未Git。运行进程已结束。

## 实际队列与隔离

**3075条严格隔离目标已完成完整链推理，不是把4744条默认为均已隔离。**

实际`baseline/fold0_inputs.npz['test']`只有1561条；fold1/2分别1632/1551，三折不重复并集才是历史4744条。与当前源四角色核对后，1669条存在MMSI重叠，过境重叠均为0：

|重叠源角色|目标重叠行数|重叠MMSI数|
|---|---:|---:|
|strategy_fit|685|452|
|utility_train|333|208|
|selection|365|226|
|evaluation|286|137|

依用户“发现真实重叠则明确隔离子集”的授权，保守排除与全部源角色重叠的目标；包括虽不训练但已参与源开发观察的evaluation身份。保留3075条、1950艘船、85个过境，原fold0/1/2分别保留1561/756/758条。此队列与源四角色均无obs_id、MMSI或过境交集。

`eligibility_4744.csv`保存全部原候选与逐角色/字段排除原因，`excluded.csv`保存1669条排除记录。没有修改标签、补MMSI或根据预测剔除。该历史队列并非盲测，本次也是见源结果后追加的外推覆盖确认；不能把3075条上的结果直接混称为旧4744条结果。

## 权重、输入与输出语义

- 唯一权重来源：`regional_utility_pilot/upstream/models`的axis_ridge、three-seed five/neighborhood/calibration/attention/body/full及reader_layer1/reader_layer3；tau来自同upstream的`scales.json`。未使用历史全量模型替换任何source-pilot层。
- A/D复用全局OLD缓存，由obs_id映射，处理已移除31237后的行偏移。T使用保存的源scaler与checkpoint。每个目标统一用本轮source-pilot权重，不随其历史fold改用旧模型。
- 新提取为原生160裁补、不缩放、真实footprint、VV/VH独立、4C4实际骨干前向。一次前向同时产生旧链摘要与stem64/layer1 256细槽；无空间特征图落盘。
- `summary1`保留冻结粗区域`min_neff>=4`规则；20槽只记录neff，不采用该门槛。两者不能混用。缺失极化为原有0与false，不伪造VH。
- `strategies_seed*.npz`的顺序为`s0/body/full`，其中s0为A/D/T后的attention输出；body和full均分别接在s0上，不是串行叠加body再full。
- 标签仅用于最后覆盖表，没有计算目标分类指标或据目标标签选checkpoint/缩放/算子。

## 重放与验证

固定12个源obs：15、31、83、88、104、113、119、135、150、220、1128、11556。重放通过后才处理目标。

|检查|实际最大误差/结果|
|---|---|
|源mean3/hb3/summary1/quality/geometry/valid对原缓存|0|
|源stem/layer1 tokens、mass/neff/geometry/valid对原缓存|0|
|源axis/A/D/U/B对原缓存|8.8818e-16|
|源三seed s0/body/full对原缓存|8.8818e-16，所有top1相同|
|目标layer3 GAP对原全局冻结features|9.1405e-6|
|目标layer1 W-GAP与逐极化mean线性闭合|2.5757e-14|
|目标layer3 W-GAP与逐极化mean线性闭合|4.4409e-15|
|摘要、tokens、策略分数|均有限；无效支持按规则为0|
|三seed策略序列化、global row_index、instance_id|精确一致|

详细值与运行前约定容差见`replay_checks.json`及`PROTOCOL_ZH.md`；没有因失败放宽容差。

## 覆盖

VV有效3075，VH有效2876，199条缺失/禁用VH，共5951个有效极化窗口。以下独立船与过境是各类内部计数，不应跨类直接相加作为总体独立数。

|class_id / 类|观测|MMSI|过境|有效VH|
|---|---:|---:|---:|---:|
|0 bulk|947|688|77|923|
|1 fishing|102|95|12|49|
|2 general|301|233|61|287|
|3 chemical|803|410|57|761|
|4 container|361|217|45|350|
|5 crude|256|133|48|252|
|6 tug|138|84|43|122|
|7 offshore|167|90|32|132|

特别是fishing仅49/102有VH，后续总体和分层结果不能忽略这一覆盖差异。这里只报告事实，不据此改变模型或匹配规则。

## 交付文件与成本

所有文件均在当前`external_confirmation/`目录：

- `run_external.py`、`PROTOCOL_ZH.md`：独立实现和冻结协议。
- `table.csv`、`valid.npz`：3075条顺序与原ROOT全局row_index、极化有效标记。
- `tokens.npz`：stem[3075,2,20,64]、layer1[3075,2,20,256]、mass、neff、geometry、valid、instance_ids、regions；与源tokens同schema。
- `summaries.npz`：indices、mean3[3075,2,8]、hb3[3075,2,8,2]、summary1[3075,2,3,8,3]、quality、geometry、region_valid、valid。
- `base_seed17/29/43.npz`：axis、A、D、U、B及indices。
- `strategies_seed17/29/43.npz`：logits[3075,3,8]、regional_raw、indices、instance_ids、role、strategy_names、class_ids、seed。
- `eligibility_4744.csv`、`excluded.csv`、`isolation.json`：完整候选、排除原因与隔离回执。
- `coverage_by_class.csv`、`coverage_by_fold_port.csv`：覆盖统计。
- `replay_checks.json`、`complete.json`、`run.log`：源重放、目标完成校验与运行日志。

离线生成总耗时500.71秒，其中目标联合提取484.62秒；GPU为RTX5050 Laptop，记录峰值分配280711680 bytes。总耗时包含源重放、图像读取、联合提取、既有缓存读取、小头推理、压缩保存及检查，**不是新模块单样本推理延迟或整套旧系统端到端部署测量**。原A/D特征是缓存复用，未计其重新从像素生成成本。

主流程可直接使用本目录tokens、table与三seed策略，沿用已经冻结的源PCA/参考库和local heads执行外部确认；本任务未替主流程做该阶段，也未选择最终方案。
