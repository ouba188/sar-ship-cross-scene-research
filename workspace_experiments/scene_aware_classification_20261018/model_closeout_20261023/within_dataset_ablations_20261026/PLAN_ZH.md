# 新版数据集内跨采集对照

2026-10-26。执行用户批准的新版OSS1、OSS2数据集内跨采集消融。

## 固定协议

- 使用new_version_ablations_20261026已冻结的source_predictions，读取各stack_table中role=evaluation。此前已完成42组读出和24组匹配拟合并保存内部留出预测，本轮不重复训练，不将新评价冒称新训练。
- OSS1四类63条、一个目标过境；OSS2四类378条、五个目标过境。种子17/29/43，32个源参考，目标标签不参与拟合或选模。重新核对实际obs_id、角色、船只和过境隔离，防止误把selection或cross预测当内部测试。
- M1/M2：七臂full/mean_meta/attention_only/mean_only/raw_only/merged_raw/body_raw，比较bare和配套M3的最终frequency；同时列B_fixed。attention_only仍包含均值；full包含72维类别摘要和32维质量几何。
- M3：both/learned_fixed/cosine_soft/cosine_fixed，以及固定、空间、空间—频域融合。匹配器在读出七臂中冻结，在M3四臂中按原规则适配；不把这两种条件混写。
- 固定位置对应同时改变位置限制与匹配质量，不声称只检验可微性。仅船体是读出输入对照，骨干感受野及后端并非全系统无背景。
- 每臂报告OA/BA/NLL/Brier、三个种子均值标准差、配对救回误伤、逐类逐过境收益。OSS1读出分析按类别分层船只bootstrap，M3另列船只簇bootstrap，均只描述固定单过境内部；OSS2按五过境重抽样，条件于已有队列，不声称新盲测或全数据覆盖。
- 原图回看采用固定seed17、每个正确性状态最低obs_id；无翻转也看持续判对/错。VV/VH原生160裁补、实际参考与分差；不凭亮点命名物理机制。
- 保留旧版同折结果和简单对照；本轮不修改模型，不据内部测试反向选结构/种子。

## 分工与交付

主代理完成读出分解、总报告与Git同步；子代理独立完成M3评价及其案例，写集仅m3/。代码、统计、派生案例同步研究仓库；权重、原始chip与NPZ不上传。
