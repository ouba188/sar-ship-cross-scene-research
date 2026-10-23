# 两种简单极化读出：预先冻结协议

2026-10-23。只写本目录，不修改父目录27个候选头、代码或预测，无Git。只新增3fold×3seed(17/29/43)×2arm=18轨迹，无网格，无新增消融。

## 固定信息与读出

复用run_polarization.inputs/load_pol及其parent的load_fold/calibrated、损失与精确J。原A/W/标签/划分/native160/C4/有效footprint不变，读取原moments_pol，不提新图像特征、不重拟合Reader。使用真实pol组织，不调用aggregate替换。每fold/seed的B=C(A)固定，source train用C(A_OOF)及对应inner-held矩；validation/test用outer矩。C非二级全栈OOF，限制不变。

- free_shared：原11维descriptor及同样source train标准化，11→16 GELU→1，209参数，末层全零。直接输出每类残差再减类均值；无tanh、无κ乘法载体、无额外幅度约束。
- static_pol：8×2共16个参数，初始0；tanh系数乘真实每类每pol κ，再pol求和并减类均值。系数跨图固定，不依赖descriptor/B/σ，B只作固定加法基线；不是逐图gate。调用相同inputs以取得真实carrier，但标准化descriptor不参与该臂输出。invalid VH载体零，保留VV。

不宣称两臂与226参数候选容量严格匹配。free共享类映射、static含类索引常量，函数约束和对称性不同。static是刻意更小的校准控制，不假称它消耗所有11维信息。

## 训练与冻结

每轨迹200步，AdamW lr=.003 / weight_decay=.01，jointCE=.5普通CE+.5八类CE均值。源validation J=.5OA+.5BA每5步含0，精确比较，严格改进才替换，平局保留最早。全部18个选点和模型保存后统一test。零头回B、验证保存重放、与父预测B对齐。CPU执行，无参数调整。

## 分析与停止

输出所有seed的source validation与test OA/BA/NLL/Brier、逐fold/逐类/VH/采集，源验证915次折暴露/816唯一观测单独说明，test4744互斥观测。每seed配对父pol_preserved与pol_preserved_capped（以及描述性aggregate参照），不把种子视为独立样本；pass/MMSI配对区间只覆盖固定预测。

固定seed17，每臂rescued/harmed/CC/WW按首ID取原VV/VH实例并实际查看；缺少翻转则明确缺失并补对错例。标题明确free为直接残差、static为固定系数κ读出，记录旧/新竞争类，不将free残差分解为κ载体或物理散射作用。

回答现有条件载体约束是否必要，而非为了结构复杂胜出。若简单读出相近或更好，保留路别信息的原matched证据，但不把free对B的收益单独当作极化信息的因果证明，也不抹去原候选正负结果。全结果保留，本轮到18轨迹停止。
