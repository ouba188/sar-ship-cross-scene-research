# 追加冻结推理协议

2026-10-23，18头完成且source_pooled已读后，主流程基于用户持续推进研究的总体授权，追加固定test推理并交由本支线执行；不是用户当场逐项明确追加授权。原PROTOCOL_ZH.md及源结果保持不变；本文件覆盖其中“本轮不评test”的后续执行限制，不改任何训练定义。

## 追加时已知结果

源验证每seed 915条fold-instance、816独立对象；sigma_only相对B净+8/+3/+13，kappa_only +17/+8/+19，原full +13/+7/+14。full对sigma的OA/BA三seed均高；kappa的OA高于full，而full的BA更高。由此追加组成分工能否在原留出集保留的检验。

## 固定推理

- 只用已保存的18个checkpoint及各自源scaler、选步，原3fold和seed17/29/43不变。参数、T强度、输入屏蔽方式均不更改。
- 冻结B=T(oldAD)，test使用同一outer mean/moments/valid；完整full与mean_only直接复用既有对应test逐行分数，不重训、不重新选点。
- 两个新臂只做一次固定test推理，保存独立test_predictions/pooled/paired/类、fold、port、pass、valid表；与full、mean及B配对。不以本轮test挑融合/力度/额外臂。
- test seed17每新臂rescued/harmed/CC/WW各按instance_id再fold取首例，实际看原VV/VH与160原尺度crop/pad；缺结局就报告，不凑例。
- 不追加bootstrap，不新增训练。推理完成后报告summary，再收口文稿。

这是见source结果后的追加组成验证；该历史test此前反复参与研发，不是盲测或新独立确认。保留正负及逐类代价，不因某一指标优劣抹去已有有效full。源验证与test分别报告，不将多seed当独立样本。

## 完成记录

同一批18个源选checkpoint的固定test推理已完成，新增训练0、选点/scaler修改0。相对full，sigma_only净−20/−26/−31、kappa_only净+18/−7/−5；完整数值与类取舍见FINDINGS_ZH.md及test独立表。此处记录完成事实，不根据结果回改上述协议。

本支线agent已实际查看两臂四结局共8幅test面板；主流程已实际查看其中6个唯一对象4919/7310/5025/7819/3494/3504。面板数与唯一对象数分别记载，均不构成新的独立数据。
