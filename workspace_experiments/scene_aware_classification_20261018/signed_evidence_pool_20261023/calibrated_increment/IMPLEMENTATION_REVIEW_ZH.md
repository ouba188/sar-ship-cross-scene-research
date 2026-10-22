# 窄实现审阅

2026-10-23。结论：**在本次指定的七项范围内，未发现实际泄漏、接错基线/矩或臂间条件不一致等需要修复的实现错误。无需因本审阅中止或重复训练。** 这不是108条训练结果均已完成/重放通过的声明。

采用 karpathy-guidelines 的范围约束，只读 run_increment.py、PROTOCOL_ZH.md 和直接调用的 inputs/raw_input/Correction/ScoreOnly/exact_j。另执行一次轻量缓存与checkpoint元数据核对，无拟合、优化器运行或重复训练；没有扩展到完整结果审计。

## 七项核对

| 项目 | 实现证据与结论 |
|---|---|
| C冻结 | run_increment.py:32–40 加载对应存盘 C，eval、requires_grad_(False)、inference_mode 后转 NumPy；82行优化器只持有新增 model.parameters()。C不参与新增梯度。 |
| B与零头 | apply 返回原score加冻结校准残差，未漏原A路径；40行分别生成 C(A_outer) 和 C(A_OOF)。60–64行对照既有同fold/seed的score-only J输出检查B。75–79行保存零末层初始状态和0步候选；最终121行用原double B加新增delta，零delta回到B。训练/选点前向保留既有float32口径。 |
| 源内标准化 | 69–73行 descriptor 已用B替代A，mu/sd只从raw[train]计算。score臂逐分数轴统计，signed/std按样本及类别共享统计，沿用原结构；验证/测试仅应用这些统计。C本身使用checkpoint保存的mu/scale，没有重新拟合。 |
| seed/checkpoint | 19–21、32行按partition/fold/seed取明确的score_only_J文件，不重新选择C。新增头每臂重置相同seed，97–100行文件名含partition/fold/arm/seed。实际36个C的fold、seed、arm、selector和预处理维度元数据均匹配。 |
| 矩与partition | 28–31行同时切换inputs、cache和origins；run_alignment.inputs:49–60按对应reader_names取得outer矩，并按该partition的inner-held索引拼OOF矩。实际两cache和12个fold输入的instance_ids均与targets顺序一致，源train每条inner覆盖恰好一次；12折及inner的MMSI/pass隔离核对通过。没有把新split矩接到旧split。 |
| 测试未选模 | 85–96行训练仅索引tr，J选择仅索引va；J由exact_j的有理计数计算，只严格改善才更新，因此0步有效、平局最早。103–106行冻结全部选择后，107行起统一评价新增头test。60–64行此前访问test仅用于既有B的分数/top1重放，不比较真值、不调整选点；不是按测试表现选模型。 |
| 三臂同条件 | 每个partition/fold/seed的三臂共用同一d、B、train/val/test、outer/inner矩来源、jointCE、200步、AdamW .003/.01及每5步J选择。实测模型实例参数数为signed113/std81/score115；普通score臂forward不使用传入moments，std只取其三项descriptor并乘sigma，signed取五项并乘kappa。 |

直接依赖依据：source_utility_alignment/matched_controls/run_controls.py 的 ScoreOnly/make_model/raw_input；source_utility_alignment/run_alignment.py:39–60 的 exact_j/inputs；train_signed.py:22–52 的 descriptors/Correction。类中心化、原矩载体与GELU均沿用这些实际定义。

## 保留的解释边界，不是新增错误

- 二级C在这批源标签上训练过。当前 `C(A_OOF)` 不是校准全栈OOF，也不是新增残差头的样本外错误监督。协议及summary里的 calibrator_OOF=False 已如实说明；普通源训练监督本身合法。
- validation先选C再选残差，存在阶段性选模复用；两split又来自已开发队列。不能把验证改进或跨split一致当独立外部确认，代码没有将其变成test选点。
- 三臂不是严格同容量/同对称性：115参数score臂有联合类别映射与偏置，局部臂共享类别函数且有tanh乘矩载体。它们符合声明的实用增量对照，不能将所有差异唯一归因于kappa。
- `selection_frozen.json` 的 test_evaluated=False 表示新增头尚未最终测试评价，不应解释成此前从未访问测试输入；脚本已对冻结B做无标签重放。这不构成测试效用泄漏。

只新增本审阅文件。训练脚本/协议/缓存均未修改，未操作Git；本审阅启动的只读核对进程已退出，主线训练由主代理管理。
