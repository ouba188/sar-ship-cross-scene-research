# 定向压缩修正的独立分析

仅分析，所有写入限制在本 analysis 目录。不修改 Plato 脚本，不重新训练，不改变原90头。待36新头的 predictions、selections、selection_curves、selection_frozen、summary 全部齐备再运行。新实现已核对：预测沿用既有 train_heads.py 表契约，实际完成标记为 summary.json，已据此修正分析入口，不猜列名或补造结果。

## 先学习与优化

逐 fold/seed/arm 对齐原 local/additive：比较200步源训练损失轨迹、选中 epoch、源验证 J/OA/BA/NLL/Brier，以及0步与选中点差值。报告 step5、step200 记录损失，而不虚构缺失的step0训练损失；既有日志的训练损失来自更新前、验证指标来自更新后，注明这一时间点区别。损失下降不自动等于有效学习，源验证参与选择也不作为独立泛化证据。

单列新旧 epoch0、新回退B、从B激活和二者均为0的头。新头若选择0，严格等于B；相对原头即使净救回，也只能称回退撤销原头损伤，不能称训后图像证据收益。

源 OOF sigma 中位数 s 的 asinh 同尺度处理属于针对长尾输入/优化的修正，不作为独立创新。mu/B不变、descriptor scaler源训练重拟合的实现由主流程负责；分析不会用测试重定s、挑尺度或换选点。

已读取主 PROTOCOL_ZH.md，确认 predictions 沿用原契约、arm仍为local/additive。主流程另将输出初始化/选点状态的真实单样本jointCE加权梯度能量及原头配对；结果齐后优先读取这些既有机制表，不重复计算梯度。平方范数份额不是AdamW更新份额，选点epoch不同也会影响比较；不能只凭损失下降或矩方差缩小声称尾部问题已修复。变换后明确称“压缩矩”，不再称原空间矩的精确积分。

## 再系统效用

新 local/additive 各比较：冻结B、原同名臂、原mean_only、原extra_score；严格按 role/seed/fold/instance 配对，核对类别、MMSI、pass、valid及B分数相同。保存源验证/测试 pooled、逐类、逐折、class×fold、pass/product/VH分层 OA/BA/NLL/Brier，以及相对对应参照的净救回/误伤。

测试每 seed 单独做 pass/MMSI 两种配对 cluster CI（沿用2000次、seed20261023），不把三seed当独立样本，不包含拟合/选点不确定性，不是双向聚类。源验证只描述，明确已用于选点。先保留原90头正负结果，不以压缩后是否变正改变比较范围。

## 原图

seed17，新头相对原同名头各选 rescued/harmed/CC/WW 的首 instance_id、再 fold，最多8张；缺组明示，不追挑。原PNG VV/VH和实际224原尺度pad分别显示，旁列B/原头/新头分数和新减原的增量，记录旧及最终竞争类。标题明确参照不是B；epoch0醒目标明。生成后人工逐张实际 view_image，才更新 images_viewed 并写 FINDINGS。只把可见事实、分差变化和总体分层联系起来，不从外观命名物理原因。

## 运行

使用 ClearSAR Python 的 `-X utf8 -B` 运行 `analyze_robust.py --check-inputs` 可检查输入是否齐备，不写结果；全部齐备且接口核对后运行同脚本不带参数。所有统计和渲染仅使用已冻结预测/原PNG，无GPU提特征或训练。
