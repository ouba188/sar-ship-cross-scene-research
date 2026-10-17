# 几何组织局部关系：固定200步结果

2026-10-16。S32，3训练种子、5支持抽样；历史研发队列，不是独立确认。

## 完整队列跨采集

| arm | oa | ba | macro_nll | macro_brier |
|---|---|---|---|---|
| B | 62.86291 | 57.15543 | 1.50329 | 0.66673 |
| conditional | 62.60535 | 56.98274 | 1.50502 | 0.66726 |
| geometry | 62.75521 | 56.91308 | 1.50521 | 0.66746 |
| geometry_radar45 | 62.75695 | 56.94742 | 1.50503 | 0.66738 |
| geometry_ship45 | 62.89878 | 56.97280 | 1.50524 | 0.66760 |
| spatial | 62.61580 | 56.96818 | 1.50458 | 0.66711 |

## 几何有效子集跨采集

| arm | oa | ba | macro_nll | macro_brier |
|---|---|---|---|---|
| B | 34.95591 | 38.94709 | 1.77681 | 0.78109 |
| conditional | 30.67019 | 35.12169 | 1.79112 | 0.78634 |
| geometry | 33.36861 | 35.76720 | 1.81362 | 0.79783 |
| geometry_radar45 | 33.33333 | 35.50911 | 1.81308 | 0.79735 |
| geometry_ship45 | 35.90829 | 36.71193 | 1.81451 | 0.79971 |
| spatial | 30.88183 | 35.23986 | 1.78651 | 0.78472 |

## 配对增量

OA/BA为百分点。以下先按每训练种子配对，再取均值。

| scope | arm | reference | delta_oa | delta_ba | delta_macro_nll | delta_macro_brier |
|---|---|---|---|---|---|---|
| full | conditional | B | -0.25756 | -0.17269 | 0.00173 | 0.00053 |
| full | conditional | spatial | -0.01044 | 0.01456 | 0.00044 | 0.00015 |
| full | geometry | B | -0.10770 | -0.24235 | 0.00192 | 0.00073 |
| full | geometry | conditional | 0.14986 | -0.06966 | 0.00019 | 0.00020 |
| full | geometry | spatial | 0.13941 | -0.05510 | 0.00063 | 0.00035 |
| full | geometry_radar45 | geometry | 0.00174 | 0.03434 | -0.00018 | -0.00008 |
| full | geometry_ship45 | geometry | 0.14357 | 0.05971 | 0.00003 | 0.00014 |
| full | spatial | B | -0.24711 | -0.18725 | 0.00129 | 0.00039 |
| geometry | conditional | B | -4.28571 | -3.82540 | 0.01431 | 0.00525 |
| geometry | conditional | spatial | -0.21164 | -0.11817 | 0.00461 | 0.00162 |
| geometry | geometry | B | -1.58730 | -3.17989 | 0.03681 | 0.01674 |
| geometry | geometry | conditional | 2.69841 | 0.64550 | 0.02250 | 0.01149 |
| geometry | geometry | spatial | 2.48677 | 0.52734 | 0.02711 | 0.01311 |
| geometry | geometry_radar45 | geometry | -0.03527 | -0.25808 | -0.00053 | -0.00048 |
| geometry | geometry_ship45 | geometry | 2.53968 | 0.94474 | 0.00089 | 0.00188 |
| geometry | spatial | B | -4.07407 | -3.70723 | 0.00970 | 0.00363 |

## 覆盖与解释限制

- 36个实际训练模型；9个无合法训练任务的回退记录，不称已训练。
- 上海源fit没有同时满足有效支持和查询的任务；不借selection、不改变划分。
- 完整队列保留缺失字段查询，其预测精确回退旧B。几何子集仅含实际可启用查询，不能冒充全港证据。
- 固定200步末模型；源曲线用于诊断，不依据目标效果选步数或幅度。
- spatial/conditional/geometry参数量相同；spatial条件输入为零，因此有效功能容量并非完全相同。
- 角度扰动是固定权重依赖性诊断，不是物理干预；所有支撑核均重新求解。
- 常量新增描述经支持中心化贡献为零；不代表所有残余信息都具有物理因果意义。

## 验证

旧B全目标S32逐行概率回放最大误差0；无效查询精确回退；逐类贡献和完整BA增量核对通过。
原始表见metrics.csv、paired_contrasts.csv、paired_events.csv、training_summary.csv；图像与支持分析另见CASE_ANALYSIS_ZH.md。
