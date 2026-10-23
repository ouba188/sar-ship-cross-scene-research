# Mean-only 同辅助 W 控制：预先固定协议

## 问题与范围

已有 local209 在强 B=T(oldAD) 上三个 seed 的 OA/BA 均有正增量。本轮检验这些局部分布读出是否优于同辅助 W 的普通 GAP 均值；不是重新调参，也不改变已有效 local。只新增 3 fold × seed17/29/43 共 9 头。旧 local、extra_score、T 与 reader 全冻结，只读取已有预测作配对，不重训、不写父级、不 Git。

## 输入与方法

使用 calibrated_signed_transfer/signed_features.npz 的 mean_pol[15,N,2,8]，15 reader 为 3 outer 和每 outer 的 4 inner。每条每极化均值 m_pc 是原 native160 crop/pad、有效 footprint、四次 C4 对原辅助 W 类方向响应的加权空间均值再平均；保持原 reader/图像/标签，不提图。

构造 slots=repeat(mean_pol[...,None],2,axis=-1)，即 [N,2pol,8class,2slot]，两槽都是 mean，不称 sigma、kappa 或物理 moments。复用原 descriptor：每类 [center(B)_c, m_VV,c, center(m_VV)_c, m_VV,c, center(m_VV)_c, m_VH,c, center(m_VH)_c, m_VH,c, center(m_VH)_c, validVV, validVH]。复用 11→16 GELU→1 类共享 209 参数头，输出去类公共项再加 B；不乘 κ、不 tanh。两重复槽保持名义架构与相同初始化，但有效自由度不完全等价，不能叫信息量/容量严格相同。

训练用冻结 T(oldAD_OOF) 与各合法 inner held 对应的 mean_pol；源 val/test 用 outer mean_pol 和冻结外层 B。旧 AD 与新 W 的 OOF 分包不要求相同，但各自排除 query MMSI，且 inner fit/held 均属相应 outer train。T 用过源训练标签，故仍是两阶段共享源监督，不是二级或全栈 OOF；两臂历史对照共享这一边界。valid0 的原均值严格为零，不删除有效 VV，也不在源标准化后额外抹零以改变既有 descriptor 规则。

## 固定训练与选点

- seed17/29/43 对应旧 seed0/1/2；逐 seed 重建既有 local 的初始化并逐张量 exact 比较，保存本轮初始 state。
- 本折 source train 的 descriptor 全部样本/类轴拟合 mu/std，std 下限 1e-8；验证测试不入 fit。相同标准化作用于验证测试。
- 200 步 full-batch AdamW，lr=.003、weight_decay=.01；jointCE=.5普通 CE+.5八类均衡 CE。其余参数完全不调。
- 每 5 步以 sourceval J=.5OA+.5BA 选择，含第 0 步，精确平局保留最早；源验证用于选择，不是独立确认。9 个选点全部保存并落盘 selection_frozen.json 后才进行 test 标签评价。
- 最后一层零初始化，零头精确回 B；冻结 B 与 strong_calibration_increment 既有逐分数重放，top1 必须不变。

## 针对性检查

训练前检查 mean_pol 形状/有限性、invalid pol 原值为零、cache/cohort/fold instance_id 一致。对 15 reader 重新用缓存 means.sum(pol)+intercept 重放辅助 A/OOF，容差沿提取阶段 1e-4 且 top1 不变；检查每 inner fit/held 身份不相交、仅来自本折源 train，held 一次覆盖 train，outer fit 精确等于 train。这些检查针对接错均值维度、用错 reader 或误拼 ID 的实际风险；失败先修接入，不悄悄改变标准。

## 交付

保存 9 checkpoint、选择曲线/源验证/测试逐样本分数、OA/BA/NLL/Brier、逐类/港/VH/采集表。与既有 local、extra_score、B 按 fold/instance/seed 配对；沿用固定预测 pass/MMSI 聚类 2000 bootstrap、seed20261023，不把重复 seed 当独立样本，不用测试结果选择新配置。seed17 每个救回/误伤/CC/WW 首 instance_id 原 VV/VH 面板，短名 mean_{outcome}.png，实际 view_image。FINDINGS 保留正负结果及有效自由度/T 非 OOF/缓存成本边界，完成即收口。

训练完成后的用户追加描述统计：仅用保存预测，分 sourceval/test 逐 seed/类计 local 独对、mean 独对、两者救回 B 错的交叠和损伤 B 对的交叠；不构造测试 oracle，不训练融合或搜索力度。这不改变已固定模型或选点。
