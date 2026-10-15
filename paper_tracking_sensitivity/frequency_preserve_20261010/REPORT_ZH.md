# 频带保留实验冻结读出 / Frozen frequency preservation readouts

家族为 old、independent、standalone、preserved；learned 为保留的主分支，共 30 模式。训练、评估、训练诊断与最终中文 FINDINGS 由主流程负责。本文件只生成描述性报告，不作因果推断或目标域赢家选择。

Main-run protocol: preserved objective = standalone objective + 0.25 * retain_hinge. Training anchors use SOURCE TRAIN queries only: the retained learned main is wrong and exactly one of four frozen independent teacher bands is correct. The target margin is the teacher true-class-minus-best-other-class margin clamped to [0, 0.1]. Other bands gaining correctness are not penalized. These are protocol definitions, not training verification.

训练锚点仅来自源域训练 query；目标 margin 为 teacher margin 截断到 [0, 0.1]。源域验证 probe 与训练锚点分开，验证或目标标签不会用于建立训练锚点。retention 使用冻结部署频带读出，不是 standalone-only ridge 读出或训练 hinge 的重算。

All 25 matching target COMPLETE markers and policies are required; every source SELECTION rule is loaded before any target-label read. The reused report checks source_selected against its frozen rule, rows, IDs, supports and support labels. Globals are restored afterward. Labels use the existing manifest paths and row alignment; no predictions or selections are changed.

OA/BA/macro_f1_8 为百分数，delta 为百分点。BA 使用当前评估集合出现的真实类别，macro F1 固定八类。目标指标先平均任务内五个 draw，再等权平均任务。base 无专家交接；full 使用各模式自己的路由；frozen_route 统一使用 learned admission mask。原 metrics、task_means、summary、draw_metrics、class_metrics、events CSV 的 delta 相对同阶段 learned；source_selection.csv 保留冻结源域选择。

Complementarity, band_events and rescue_overlaps retain base/frozen_route diagnostics. Label-assisted oracles preserve learned-correct queries and rescue main-wrong queries. Best-single-band oracles optimize OA and BA separately within task/draw only as upper bounds; union_minus_best_single_oracle measures the extra union gain. They are not deployable policies.

## 固定对照 BA / Fixed controls

| dataset | mode | base | full | frozen_route |
| --- | --- | --- | --- | --- |
| f0921 | independent_equal | 26.1399 | 26.4456 | 26.4355 |
| f0921 | learned | 26.0348 | 26.3593 | 26.3593 |
| f0921 | old_equal | 26.1005 | 26.4069 | 26.3941 |
| f0921 | preserved_equal | 26.1459 | 26.4526 | 26.4424 |
| f0921 | source_selected | 26.0277 | 26.3534 | 26.3442 |
| f0921 | standalone_equal | 26.1398 | 26.4459 | 26.4349 |
| oss2 | independent_equal | 31.1496 | 32.1821 | 32.1881 |
| oss2 | learned | 31.2783 | 32.3724 | 32.3724 |
| oss2 | old_equal | 30.9430 | 32.0829 | 31.9921 |
| oss2 | preserved_equal | 31.0968 | 32.1683 | 32.1295 |
| oss2 | source_selected | 31.2420 | 32.3104 | 32.2930 |
| oss2 | standalone_equal | 31.1171 | 32.2214 | 32.1527 |
| self24ports | independent_equal | 52.3612 | 54.1313 | 54.0076 |
| self24ports | learned | 52.4994 | 54.2383 | 54.2383 |
| self24ports | old_equal | 52.3278 | 54.1356 | 53.9788 |
| self24ports | preserved_equal | 52.3748 | 54.1388 | 54.0211 |
| self24ports | source_selected | 52.5625 | 54.3365 | 54.2139 |
| self24ports | standalone_equal | 52.3557 | 54.1642 | 54.0044 |

## 全部配对差值 / All matched differences

| dataset | stage | suffix | comparison | delta_oa | delta_ba | delta_macro_f1_8 | positive_draws_ba | draws |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f0921 | base | HH | preserved-independent | 0.0023 | 0.0010 | 0.0023 | 3 | 5 |
| f0921 | base | HH | preserved-standalone | 0.0010 | 0.0016 | 0.0009 | 3 | 5 |
| f0921 | base | HH | standalone-independent | 0.0013 | -0.0006 | 0.0014 | 2 | 5 |
| f0921 | base | HL | preserved-independent | 0.0157 | 0.0067 | 0.0080 | 4 | 5 |
| f0921 | base | HL | preserved-standalone | 0.0096 | 0.0056 | 0.0076 | 4 | 5 |
| f0921 | base | HL | standalone-independent | 0.0062 | 0.0011 | 0.0004 | 3 | 5 |
| f0921 | base | LH | preserved-independent | 0.0055 | 0.0037 | -0.0010 | 2 | 5 |
| f0921 | base | LH | preserved-standalone | 0.0041 | 0.0015 | 0.0018 | 2 | 5 |
| f0921 | base | LH | standalone-independent | 0.0015 | 0.0021 | -0.0028 | 2 | 5 |
| f0921 | base | LL | preserved-independent | 0.0099 | 0.0072 | 0.0285 | 3 | 5 |
| f0921 | base | LL | preserved-standalone | 0.0120 | 0.0126 | 0.0150 | 4 | 5 |
| f0921 | base | LL | standalone-independent | -0.0021 | -0.0053 | 0.0135 | 3 | 5 |
| f0921 | base | equal | preserved-independent | 0.0065 | 0.0060 | 0.0121 | 2 | 5 |
| f0921 | base | equal | preserved-standalone | 0.0058 | 0.0060 | 0.0090 | 4 | 5 |
| f0921 | base | equal | standalone-independent | 0.0006 | -0.0001 | 0.0031 | 1 | 5 |
| f0921 | base | support_local | preserved-independent | 0.0196 | 0.0112 | 0.0218 | 2 | 5 |
| f0921 | base | support_local | preserved-standalone | 0.0016 | -0.0016 | 0.0010 | 1 | 5 |
| f0921 | base | support_local | standalone-independent | 0.0180 | 0.0128 | 0.0208 | 2 | 5 |
| f0921 | base | support_static | preserved-independent | -0.0181 | 0.0326 | 0.0195 | 3 | 5 |
| f0921 | base | support_static | preserved-standalone | 0.0066 | 0.0134 | 0.0116 | 3 | 5 |
| f0921 | base | support_static | standalone-independent | -0.0248 | 0.0191 | 0.0079 | 3 | 5 |
| f0921 | frozen_route | HH | preserved-independent | 0.0026 | 0.0023 | 0.0031 | 3 | 5 |
| f0921 | frozen_route | HH | preserved-standalone | 0.0011 | 0.0019 | 0.0012 | 4 | 5 |
| f0921 | frozen_route | HH | standalone-independent | 0.0015 | 0.0004 | 0.0019 | 3 | 5 |
| f0921 | frozen_route | HL | preserved-independent | 0.0149 | 0.0036 | 0.0069 | 3 | 5 |
| f0921 | frozen_route | HL | preserved-standalone | 0.0078 | 0.0024 | 0.0054 | 3 | 5 |
| f0921 | frozen_route | HL | standalone-independent | 0.0071 | 0.0011 | 0.0015 | 3 | 5 |
| f0921 | frozen_route | LH | preserved-independent | 0.0050 | 0.0051 | 0.0006 | 2 | 5 |
| f0921 | frozen_route | LH | preserved-standalone | 0.0031 | 0.0002 | 0.0006 | 2 | 5 |
| f0921 | frozen_route | LH | standalone-independent | 0.0019 | 0.0049 | -0.0000 | 3 | 5 |
| f0921 | frozen_route | LL | preserved-independent | 0.0053 | 0.0005 | 0.0235 | 3 | 5 |
| f0921 | frozen_route | LL | preserved-standalone | 0.0107 | 0.0137 | 0.0146 | 4 | 5 |
| f0921 | frozen_route | LL | standalone-independent | -0.0053 | -0.0132 | 0.0089 | 3 | 5 |
| f0921 | frozen_route | equal | preserved-independent | 0.0057 | 0.0069 | 0.0125 | 2 | 5 |
| f0921 | frozen_route | equal | preserved-standalone | 0.0049 | 0.0075 | 0.0091 | 4 | 5 |
| f0921 | frozen_route | equal | standalone-independent | 0.0008 | -0.0006 | 0.0033 | 1 | 5 |
| f0921 | frozen_route | support_local | preserved-independent | 0.0190 | 0.0091 | 0.0210 | 2 | 5 |
| f0921 | frozen_route | support_local | preserved-standalone | 0.0023 | -0.0005 | 0.0019 | 3 | 5 |
| f0921 | frozen_route | support_local | standalone-independent | 0.0167 | 0.0096 | 0.0191 | 2 | 5 |
| f0921 | frozen_route | support_static | preserved-independent | -0.0207 | 0.0257 | 0.0164 | 3 | 5 |
| f0921 | frozen_route | support_static | preserved-standalone | 0.0060 | 0.0125 | 0.0108 | 3 | 5 |
| f0921 | frozen_route | support_static | standalone-independent | -0.0267 | 0.0132 | 0.0056 | 3 | 5 |
| f0921 | full | HH | preserved-independent | 0.0034 | 0.0034 | 0.0043 | 4 | 5 |
| f0921 | full | HH | preserved-standalone | 0.0010 | 0.0011 | 0.0006 | 3 | 5 |
| f0921 | full | HH | standalone-independent | 0.0024 | 0.0023 | 0.0037 | 3 | 5 |
| f0921 | full | HL | preserved-independent | 0.0159 | 0.0051 | 0.0081 | 4 | 5 |
| f0921 | full | HL | preserved-standalone | 0.0089 | 0.0036 | 0.0066 | 4 | 5 |
| f0921 | full | HL | standalone-independent | 0.0070 | 0.0015 | 0.0015 | 2 | 5 |
| f0921 | full | LH | preserved-independent | 0.0034 | 0.0024 | -0.0034 | 2 | 5 |
| f0921 | full | LH | preserved-standalone | 0.0045 | 0.0013 | 0.0002 | 2 | 5 |
| f0921 | full | LH | standalone-independent | -0.0011 | 0.0011 | -0.0035 | 3 | 5 |
| f0921 | full | LL | preserved-independent | 0.0084 | 0.0003 | 0.0248 | 3 | 5 |
| f0921 | full | LL | preserved-standalone | 0.0118 | 0.0143 | 0.0152 | 4 | 5 |
| f0921 | full | LL | standalone-independent | -0.0034 | -0.0140 | 0.0096 | 2 | 5 |
| f0921 | full | equal | preserved-independent | 0.0060 | 0.0070 | 0.0137 | 3 | 5 |
| f0921 | full | equal | preserved-standalone | 0.0029 | 0.0067 | 0.0080 | 4 | 5 |
| f0921 | full | equal | standalone-independent | 0.0031 | 0.0003 | 0.0057 | 1 | 5 |
| f0921 | full | support_local | preserved-independent | 0.0169 | 0.0096 | 0.0191 | 2 | 5 |
| f0921 | full | support_local | preserved-standalone | 0.0018 | -0.0019 | 0.0009 | 2 | 5 |
| f0921 | full | support_local | standalone-independent | 0.0151 | 0.0115 | 0.0182 | 2 | 5 |
| f0921 | full | support_static | preserved-independent | -0.0240 | 0.0237 | 0.0143 | 3 | 5 |
| f0921 | full | support_static | preserved-standalone | 0.0066 | 0.0131 | 0.0111 | 3 | 5 |
| f0921 | full | support_static | standalone-independent | -0.0306 | 0.0106 | 0.0032 | 3 | 5 |
| oss2 | base | HH | preserved-independent | -0.0054 | -0.0353 | -0.0298 | 0 | 5 |
| oss2 | base | HH | preserved-standalone | 0.0054 | -0.0223 | -0.0189 | 2 | 5 |
| oss2 | base | HH | standalone-independent | -0.0109 | -0.0130 | -0.0108 | 1 | 5 |
| oss2 | base | HL | preserved-independent | -0.0380 | 0.0223 | -0.0092 | 2 | 5 |
| oss2 | base | HL | preserved-standalone | -0.0109 | -0.0088 | -0.0100 | 0 | 5 |
| oss2 | base | HL | standalone-independent | -0.0272 | 0.0311 | 0.0008 | 3 | 5 |
| oss2 | base | LH | preserved-independent | -0.0380 | -0.0469 | -0.0480 | 2 | 5 |
| oss2 | base | LH | preserved-standalone | -0.0217 | -0.0292 | -0.0371 | 1 | 5 |
| oss2 | base | LH | standalone-independent | -0.0163 | -0.0177 | -0.0109 | 2 | 5 |
| oss2 | base | LL | preserved-independent | -0.0815 | 0.0169 | -0.0573 | 2 | 5 |
| oss2 | base | LL | preserved-standalone | -0.0380 | -0.0475 | -0.0173 | 2 | 5 |
| oss2 | base | LL | standalone-independent | -0.0435 | 0.0645 | -0.0399 | 3 | 5 |
| oss2 | base | equal | preserved-independent | -0.0272 | -0.0527 | -0.0457 | 2 | 5 |
| oss2 | base | equal | preserved-standalone | 0.0163 | -0.0203 | -0.0061 | 1 | 5 |
| oss2 | base | equal | standalone-independent | -0.0435 | -0.0324 | -0.0395 | 1 | 5 |
| oss2 | base | support_local | preserved-independent | -0.0054 | -0.0206 | -0.0144 | 1 | 5 |
| oss2 | base | support_local | preserved-standalone | 0.0000 | -0.0444 | -0.0329 | 1 | 5 |
| oss2 | base | support_local | standalone-independent | -0.0054 | 0.0237 | 0.0185 | 3 | 5 |
| oss2 | base | support_static | preserved-independent | -0.0435 | -0.0093 | -0.0221 | 0 | 5 |
| oss2 | base | support_static | preserved-standalone | -0.0272 | -0.0347 | -0.0295 | 1 | 5 |
| oss2 | base | support_static | standalone-independent | -0.0163 | 0.0255 | 0.0074 | 1 | 5 |
| oss2 | frozen_route | HH | preserved-independent | 0.0000 | -0.0327 | -0.0279 | 0 | 5 |
| oss2 | frozen_route | HH | preserved-standalone | 0.0054 | -0.0223 | -0.0205 | 2 | 5 |
| oss2 | frozen_route | HH | standalone-independent | -0.0054 | -0.0104 | -0.0074 | 1 | 5 |
| oss2 | frozen_route | HL | preserved-independent | -0.0380 | 0.0223 | -0.0084 | 2 | 5 |
| oss2 | frozen_route | HL | preserved-standalone | -0.0163 | -0.0115 | -0.0137 | 0 | 5 |
| oss2 | frozen_route | HL | standalone-independent | -0.0217 | 0.0337 | 0.0053 | 3 | 5 |
| oss2 | frozen_route | LH | preserved-independent | -0.0380 | -0.0619 | -0.0511 | 1 | 5 |
| oss2 | frozen_route | LH | preserved-standalone | -0.0217 | -0.0292 | -0.0367 | 1 | 5 |
| oss2 | frozen_route | LH | standalone-independent | -0.0163 | -0.0328 | -0.0144 | 1 | 5 |
| oss2 | frozen_route | LL | preserved-independent | -0.0652 | 0.0382 | -0.0377 | 3 | 5 |
| oss2 | frozen_route | LL | preserved-standalone | -0.0435 | -0.0479 | -0.0228 | 2 | 5 |
| oss2 | frozen_route | LL | standalone-independent | -0.0217 | 0.0862 | -0.0149 | 3 | 5 |
| oss2 | frozen_route | equal | preserved-independent | -0.0380 | -0.0586 | -0.0525 | 2 | 5 |
| oss2 | frozen_route | equal | preserved-standalone | 0.0109 | -0.0232 | -0.0110 | 1 | 5 |
| oss2 | frozen_route | equal | standalone-independent | -0.0489 | -0.0354 | -0.0415 | 1 | 5 |
| oss2 | frozen_route | support_local | preserved-independent | -0.0163 | -0.0406 | -0.0292 | 0 | 5 |
| oss2 | frozen_route | support_local | preserved-standalone | -0.0054 | -0.0470 | -0.0404 | 0 | 5 |
| oss2 | frozen_route | support_local | standalone-independent | -0.0109 | 0.0064 | 0.0112 | 2 | 5 |
| oss2 | frozen_route | support_static | preserved-independent | -0.0272 | -0.0007 | -0.0081 | 1 | 5 |
| oss2 | frozen_route | support_static | preserved-standalone | -0.0272 | -0.0350 | -0.0329 | 1 | 5 |
| oss2 | frozen_route | support_static | standalone-independent | 0.0000 | 0.0343 | 0.0249 | 1 | 5 |
| oss2 | full | HH | preserved-independent | 0.0109 | -0.0254 | -0.0213 | 2 | 5 |
| oss2 | full | HH | preserved-standalone | 0.0163 | -0.0168 | -0.0146 | 2 | 5 |
| oss2 | full | HH | standalone-independent | -0.0054 | -0.0086 | -0.0067 | 2 | 5 |
| oss2 | full | HL | preserved-independent | -0.0435 | 0.0197 | -0.0127 | 2 | 5 |
| oss2 | full | HL | preserved-standalone | -0.0109 | -0.0085 | -0.0109 | 0 | 5 |
| oss2 | full | HL | standalone-independent | -0.0326 | 0.0282 | -0.0018 | 3 | 5 |
| oss2 | full | LH | preserved-independent | -0.0543 | -0.0498 | -0.0543 | 1 | 5 |
| oss2 | full | LH | preserved-standalone | -0.0217 | -0.0317 | -0.0377 | 1 | 5 |
| oss2 | full | LH | standalone-independent | -0.0326 | -0.0181 | -0.0166 | 1 | 5 |
| oss2 | full | LL | preserved-independent | -0.0652 | 0.0328 | -0.0425 | 4 | 5 |
| oss2 | full | LL | preserved-standalone | -0.0272 | -0.0464 | -0.0158 | 2 | 5 |
| oss2 | full | LL | standalone-independent | -0.0380 | 0.0792 | -0.0268 | 3 | 5 |
| oss2 | full | equal | preserved-independent | -0.0380 | -0.0137 | -0.0326 | 3 | 5 |
| oss2 | full | equal | preserved-standalone | -0.0163 | -0.0531 | -0.0478 | 1 | 5 |
| oss2 | full | equal | standalone-independent | -0.0217 | 0.0394 | 0.0152 | 3 | 5 |
| oss2 | full | support_local | preserved-independent | -0.0272 | -0.0583 | -0.0383 | 1 | 5 |
| oss2 | full | support_local | preserved-standalone | -0.0163 | -0.0647 | -0.0498 | 1 | 5 |
| oss2 | full | support_local | standalone-independent | -0.0109 | 0.0064 | 0.0115 | 2 | 5 |
| oss2 | full | support_static | preserved-independent | -0.0109 | 0.0160 | 0.0062 | 2 | 5 |
| oss2 | full | support_static | preserved-standalone | -0.0109 | -0.0218 | -0.0202 | 1 | 5 |
| oss2 | full | support_static | standalone-independent | -0.0000 | 0.0378 | 0.0264 | 1 | 5 |
| self24ports | base | HH | preserved-independent | 0.0092 | 0.0019 | 0.0018 | 2 | 5 |
| self24ports | base | HH | preserved-standalone | 0.0010 | -0.0008 | -0.0009 | 3 | 5 |
| self24ports | base | HH | standalone-independent | 0.0082 | 0.0028 | 0.0027 | 3 | 5 |
| self24ports | base | HL | preserved-independent | 0.0407 | 0.0336 | 0.0263 | 5 | 5 |
| self24ports | base | HL | preserved-standalone | -0.0081 | -0.0117 | -0.0065 | 2 | 5 |
| self24ports | base | HL | standalone-independent | 0.0488 | 0.0453 | 0.0328 | 5 | 5 |
| self24ports | base | LH | preserved-independent | 0.0173 | -0.0157 | -0.0043 | 1 | 5 |
| self24ports | base | LH | preserved-standalone | -0.0207 | -0.0451 | -0.0372 | 2 | 5 |
| self24ports | base | LH | standalone-independent | 0.0380 | 0.0294 | 0.0329 | 4 | 5 |
| self24ports | base | LL | preserved-independent | 0.0981 | 0.1356 | 0.0929 | 4 | 5 |
| self24ports | base | LL | preserved-standalone | -0.0033 | -0.0036 | -0.0012 | 3 | 5 |
| self24ports | base | LL | standalone-independent | 0.1015 | 0.1392 | 0.0942 | 4 | 5 |
| self24ports | base | equal | preserved-independent | 0.0011 | 0.0135 | 0.0068 | 3 | 5 |
| self24ports | base | equal | preserved-standalone | 0.0114 | 0.0191 | 0.0100 | 4 | 5 |
| self24ports | base | equal | standalone-independent | -0.0103 | -0.0056 | -0.0032 | 3 | 5 |
| self24ports | base | support_local | preserved-independent | 0.0089 | 0.0046 | 0.0037 | 2 | 5 |
| self24ports | base | support_local | preserved-standalone | -0.0009 | -0.0024 | 0.0004 | 2 | 5 |
| self24ports | base | support_local | standalone-independent | 0.0097 | 0.0070 | 0.0033 | 2 | 5 |
| self24ports | base | support_static | preserved-independent | 0.0639 | 0.1344 | 0.0923 | 5 | 5 |
| self24ports | base | support_static | preserved-standalone | -0.0206 | -0.0214 | -0.0144 | 2 | 5 |
| self24ports | base | support_static | standalone-independent | 0.0845 | 0.1558 | 0.1067 | 4 | 5 |
| self24ports | frozen_route | HH | preserved-independent | 0.0003 | 0.0012 | -0.0006 | 4 | 5 |
| self24ports | frozen_route | HH | preserved-standalone | -0.0006 | -0.0018 | -0.0013 | 1 | 5 |
| self24ports | frozen_route | HH | standalone-independent | 0.0009 | 0.0030 | 0.0007 | 4 | 5 |
| self24ports | frozen_route | HL | preserved-independent | 0.0203 | 0.0221 | 0.0175 | 5 | 5 |
| self24ports | frozen_route | HL | preserved-standalone | -0.0106 | -0.0127 | -0.0068 | 2 | 5 |
| self24ports | frozen_route | HL | standalone-independent | 0.0309 | 0.0349 | 0.0243 | 5 | 5 |
| self24ports | frozen_route | LH | preserved-independent | 0.0139 | -0.0153 | -0.0062 | 1 | 5 |
| self24ports | frozen_route | LH | preserved-standalone | -0.0409 | -0.0533 | -0.0430 | 1 | 5 |
| self24ports | frozen_route | LH | standalone-independent | 0.0547 | 0.0379 | 0.0367 | 4 | 5 |
| self24ports | frozen_route | LL | preserved-independent | 0.0709 | 0.1188 | 0.0819 | 4 | 5 |
| self24ports | frozen_route | LL | preserved-standalone | -0.0054 | -0.0047 | 0.0000 | 3 | 5 |
| self24ports | frozen_route | LL | standalone-independent | 0.0763 | 0.1235 | 0.0819 | 4 | 5 |
| self24ports | frozen_route | equal | preserved-independent | -0.0006 | 0.0136 | 0.0060 | 3 | 5 |
| self24ports | frozen_route | equal | preserved-standalone | 0.0098 | 0.0167 | 0.0075 | 3 | 5 |
| self24ports | frozen_route | equal | standalone-independent | -0.0103 | -0.0032 | -0.0015 | 3 | 5 |
| self24ports | frozen_route | support_local | preserved-independent | -0.0036 | -0.0026 | 0.0013 | 2 | 5 |
| self24ports | frozen_route | support_local | preserved-standalone | -0.0058 | -0.0064 | -0.0020 | 2 | 5 |
| self24ports | frozen_route | support_local | standalone-independent | 0.0021 | 0.0038 | 0.0032 | 2 | 5 |
| self24ports | frozen_route | support_static | preserved-independent | 0.0388 | 0.1150 | 0.0767 | 4 | 5 |
| self24ports | frozen_route | support_static | preserved-standalone | -0.0204 | -0.0196 | -0.0133 | 0 | 5 |
| self24ports | frozen_route | support_static | standalone-independent | 0.0592 | 0.1346 | 0.0900 | 4 | 5 |
| self24ports | full | HH | preserved-independent | -0.0071 | 0.0035 | -0.0006 | 1 | 5 |
| self24ports | full | HH | preserved-standalone | -0.0008 | -0.0039 | -0.0046 | 2 | 5 |
| self24ports | full | HH | standalone-independent | -0.0063 | 0.0075 | 0.0040 | 3 | 5 |
| self24ports | full | HL | preserved-independent | 0.0271 | 0.0247 | 0.0198 | 3 | 5 |
| self24ports | full | HL | preserved-standalone | -0.0102 | -0.0180 | -0.0145 | 1 | 5 |
| self24ports | full | HL | standalone-independent | 0.0373 | 0.0427 | 0.0343 | 3 | 5 |
| self24ports | full | LH | preserved-independent | 0.0210 | 0.0241 | 0.0139 | 4 | 5 |
| self24ports | full | LH | preserved-standalone | 0.0053 | 0.0212 | 0.0017 | 4 | 5 |
| self24ports | full | LH | standalone-independent | 0.0157 | 0.0029 | 0.0122 | 3 | 5 |
| self24ports | full | LL | preserved-independent | 0.1004 | 0.0988 | 0.0812 | 5 | 5 |
| self24ports | full | LL | preserved-standalone | -0.0004 | -0.0165 | -0.0012 | 2 | 5 |
| self24ports | full | LL | standalone-independent | 0.1008 | 0.1153 | 0.0825 | 5 | 5 |
| self24ports | full | equal | preserved-independent | -0.0012 | 0.0075 | 0.0019 | 4 | 5 |
| self24ports | full | equal | preserved-standalone | -0.0039 | -0.0255 | -0.0150 | 1 | 5 |
| self24ports | full | equal | standalone-independent | 0.0027 | 0.0329 | 0.0169 | 4 | 5 |
| self24ports | full | support_local | preserved-independent | -0.0008 | -0.0058 | -0.0012 | 1 | 5 |
| self24ports | full | support_local | preserved-standalone | -0.0086 | -0.0115 | -0.0045 | 2 | 5 |
| self24ports | full | support_local | standalone-independent | 0.0077 | 0.0056 | 0.0033 | 2 | 5 |
| self24ports | full | support_static | preserved-independent | 0.0644 | 0.1253 | 0.0894 | 5 | 5 |
| self24ports | full | support_static | preserved-standalone | -0.0129 | -0.0130 | -0.0109 | 2 | 5 |
| self24ports | full | support_static | standalone-independent | 0.0773 | 0.1383 | 0.1003 | 4 | 5 |

matched_* CSV 比较 standalone-independent、preserved-independent、preserved-standalone，固定保留 equal、四单频带、support_static、support_local 七种读出与三个阶段。先按同 dataset/task/draw/stage/suffix 相减，再汇总；不按目标结果筛选模式。positive_draws 严格计数五个 support draw 中 delta > 0 的次数；summary 使用每 draw 的任务等权均值，task_means 使用任务内 draw；positive_task_draws 计数各 task/draw 正差值。OA、BA、F1 均保存。

## 保留诊断 / Retention diagnostics

| split | dataset | baseline_anchor_count | standalone_anchor_retained | preserved_anchor_retained | standalone_lost_baseline_union_rescued_by_preserved | standalone_union_lost_by_preserved | standalone_oracle_delta_ba_vs_independent | preserved_oracle_delta_ba_vs_independent | preserved_oracle_delta_ba_vs_standalone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source_validation | f0921 | 42 | 34 | 36 | 4 | 2 | 0.0284 | 0.1821 | 0.1537 |
| source_validation | oss2 | 62 | 48 | 50 | 2 | 0 | -0.1609 | -0.0097 | 0.1512 |
| source_validation | self24ports | 788 | 548 | 659 | 119 | 37 | -0.1469 | -0.0361 | 0.1109 |
| target_postfreeze | f0921 | 5649 | 4523 | 4623 | 239 | 257 | -0.0084 | 0.0308 | 0.0392 |
| target_postfreeze | oss2 | 222 | 178 | 183 | 9 | 4 | -0.0751 | 0.0773 | 0.1524 |
| target_postfreeze | self24ports | 914 | 670 | 777 | 108 | 43 | -0.0063 | 0.0463 | 0.0526 |

retention.csv 的 source_validation unit 是 query_scores_XX.npz 的 episode；target_postfreeze unit 是五次 support draw，仅使用 base_predictions 的 eval_mask query。源域验证不是训练锚点统计；同源域端口可能参与表征训练，不能称为表征独立验证。训练诊断以主流程记录为准。

Baseline anchors fix learned-main-wrong AND exactly-one-independent-band-correct. Retention requires that same anchor band to remain correct, even if other bands become correct too. CSV includes LL/LH/HL/HH anchor denominators and retained counts. Union rescues include all main-wrong queries with any correct band, including non-unique rescues. New/lost union counts compare each family with independent. Recovery counts baseline union rescues lost by standalone but recovered by preserved; standalone_union_lost_by_preserved includes every standalone union rescue lost by preserved, whether or not it belonged to the independent baseline.

Oracle BA delta weights each query by 100 / (K * n_true_class), using all true classes in that task/draw or source episode, never only rescue/anchor classes. Main-correct contributions cancel in the difference. retention_task_summary averages episode/draw BA deltas within task; retention_summary averages tasks equally. Counts sum query events, including repeated queries across draws/episodes/tasks, and are not independent replications or significance tests.
