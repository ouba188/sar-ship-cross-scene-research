# 独立整套实现与验证记录

日期：2026-11-21。此阶段交付代码，未进行新的真实 SAR 训练或 LOPO 精度实验。

## 实现范围

`sar_port` 是可单独运行的代码包，不依赖历史实验目录的 Python 导入。包含原生 chip/OBB 处理、R50/M1/M2、CCREL/PCEA、源端真实代表原型选择、类别先验与有符号效应学生、动态父子粒度、完整局部匹配与拒配、第三特征层、门控重组、训练/评估/导出/无标签推理。

`configs/full.json` 显式开启第三层和门控，并给出各监督项的示例权重；这些数值未经精度寻优。`bootstrap.json` 保持相同架构，仅关闭新增辅助损失，便于严格加载同折源模型。另提供两层、基础模型及 20 个单项消融配置。所有配置可直接修改；不同形状的架构应使用同一通用预训练起点分别训练。

## 验证及其能检出的失败

| 检查 | 结果 | 检查目的 |
|---|---|---|
| `python -m unittest discover -s tests -v` | 40 项通过 | 原型身份隔离、源折暴露、候选类别效应符号与竞争类切换、缺极化、教师空间支持、门控梯度、完整对应与拒配强度等 |
| `python scripts/check_model.py --device cuda` | 两层/三层门控各 2 次合成更新通过 | 完整计算图、主要模块梯度、动态 source bank、无参考时回退、无查询标签依赖 |
| 全关基础/内容型/hard同层aggregate/cross组合 | 前向有限 | 受支持开关组合的维度和分支接线 |
| 冻结 bank 保存/加载重放 | 最大 logit 差 0 | 导出 bank 不遗漏 source 统计或隐式可变状态 |
| `python scripts/check_cli.py --device cuda` | 全流程通过 | split→随机 bank→bootstrap→覆盖 bank→teacher/full→evaluate→export→无标签 predict |
| CLI 导出前后概率比较 | 最大差 0 | 部署时无参考原图仍能重放 |
| 真实 chip 只读检查 | 4 艘独立船/4 个采集组，2 query＋2 reference | 真实原生尺寸、独立 VV/VH 几何、有效区域与第三层输入连接 |

GPU 模型小检查在 2 query＋2 reference、float32、RTX 5050 Laptop 上峰值约 1.04 GiB；这不是完整 8 类原型预算的显存或吞吐基准。CLI 合成检查的 teacher 每个学生 batch 做 25 次前向（原图、12 次干预、12 次亮度对照）。合成更新只用于检测反传/接线错误，其 loss 或准确率不构成科研收益。

聚合结果见 `verification/model_check.json`、`verification/cli_check.json` 和 `verification/native_check.json`。原图、私有样本身份、路径、临时权重及完整日志留在 `local_validation`，不进入代码分发或 GitHub。

## 本次修正的具体接线问题

1. 禁配边从 Sinkhorn 核中排除，避免在归一化时吸收质量；同层 aggregate 对照逐层保边际重组，不重新引入跨层边。两层/三层重复相同证据的对照不再多票或丢票。
2. 匹配后的条件 value 与绝对接受比例分开。人为将 transport 缩小为 epsilon 时，条件距离保持不变，残差强度同步缩小；关闭可靠性门控的对照保留旧不衰减行为。
3. effect 监督使用所有候选类别的原图减干预 margin，竞争类别随 logits 实际切换。attention 和 effect 使用不同有效性 mask，不把 teacher 判错一律丢弃。
4. teacher 空间支持与 backbone 使用相同的原点 0 格点面积积分；attention 从概率质量转为 token 密度时避免重复计算边界有效面积。
5. PCEA source 统计显式包含在 reference bank 中，去掉实例内可变 source 状态；门控 donor 由训练随机种子控制。
6. selector 预算损失实际加入训练；仅 ResNet 使用骨干学习率，新增模块使用模块学习率；`--freeze PREFIX` 只冻结参数，不切断所需计算图。

## 原始 chip 回看与结论边界

已分开查看此次 2 个查询和 2 个参考的原始 VV/VH，保持原生显示尺度。参考中包含近点状紧凑亮响应和较短分散响应，查询中包含更长、更强的亮结构及不同极化显著性；图像的有效宽高也不完全一致。这支持保留原尺度、独立极化几何、有效覆盖与可变局部读出，但不能证明原型/粒度模块能涨点。另一个查询类别在该极小检查 bank 中没有同类参考，检查只验证缺类时数值和路径可运行，不作为闭集评估。

本次真实 chip 前向使用随机初始化，没有优化和精度比较，因此没有“被救回/被误伤”实验结论。私有案例 ID、路径和可见事实另存 `local_validation/native_review_local.md`。正式实验仍需按既定源折协议，在配对预算下比较完整模型，并回看实际同类/混淆类支持和预测翻转。

已完成的是可运行、可配置的工程实现。弱响应图不是物理标注，有符号干预效应不是因果识别证明，门控剩余分支不等同纯环境；没有宣称 ZoomViT 原作者完整训练复现，也未证明新 LOPO 收益。软选择/期望 token 预算不等于稀疏执行或实际加速。
