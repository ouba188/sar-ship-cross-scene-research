# M2 bottleneck位置：已核实文献与本轮边界

2026-10-28。已读取官方全文的方法和消融段落，不以摘要代替核实。文献原PDF、pdftotext全文和部分表格页渲染在 `literature/`。本笔记不要求新增训练。

## 1. SE：默认位置与真正的内部3x3消融

Hu等，Squeeze-and-Excitation Networks，arXiv:1709.01507。

- **CVPR 2018原版**：`literature/SE_CVPR2018.pdf`，Sec.3.3、Fig.3、Table1。默认SE作用在残差变换输出，位于与identity相加之前；不是conv2之前。ResNet50四stage的3/4/6/3个残差块各放一个，共16个，这是按Table1结构计数。Sec.4报告去掉最后stage的SE，top1错误率损失小于0.1个百分点，额外参数占比约10%降至4%；删除3个后为13个，由结构推得，原文没有把它包装为独立的1/5/10数量扫描。
- **arXiv v4，2019-05-16扩展版**：`literature/SE_arxiv_v4.pdf`，Sec.6.4/6.5、Tables13–15、Fig.5。这些额外位置消融必须标明版本，不能声称全在CVPR原版中。
- Table14的SE-PRE是**整个残差单元之前**，SE-POST是**残差相加及ReLU之后**，SE-Identity在identity支路。它们与本轮pre3x3/post3x3命名不等价。

| SE v4 Table14设计 | ImageNet top1错误率 | top5错误率 |
|---|---:|---:|
| 默认SE | 22.28 | 6.03 |
| SE-PRE | 22.23 | 6.00 |
| SE-POST | 22.78 | 6.35 |
| SE-Identity | 22.20 | 6.15 |

**最直接相关的是Table15**：将SE放入残差单元内部、3x3卷积之后。默认SE为22.28/6.03错误率、28.1M参数、3.87GFLOPs；SE_3x3为22.48/6.02、25.8M、3.86GFLOPs。原文明确说明3x3通道更少，因此容量降低。它支持“内部插入可作为效率选项”，不证明内部必胜，更没有本轮conv2前后配对的直接结论。该段落只明确3x3之后，未据此推断本实现BN/ReLU的精确操作顺序。

Table13按stage加入SE的top1错误率：baseline23.30，stage2为23.03，stage3为23.04，stage4为22.68，all为22.28；这是不同stage覆盖范围的比较，不是相同位置固定容量的纯数量消融。文中的stage3对应通常ResNet的layer2，不能直接把文献stage编号当作PyTorch layer编号。

## 2. GCNet：最后1x1之后，不是conv2之后

Cao等，GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond，arXiv:1904.11492。已读 `literature/GCNet_ICCVW2019.pdf`，Sec.5.1.1、Table2，尤其positions/stages正文；表格页已渲染并查看。

- Table2(a)单模块NL/SNL/GC均置于c4最后一个残差块**之前**；COCO bbox AP分别38.0/38.1/38.1，baseline37.2。all GC为39.4。
- Table2(b)比较afterAdd与after1x1：bbox AP均39.4，mask AP分别35.8/35.7。after1x1明确是残差支路**最后一个1x1卷积之后**，位于相加之前；作者因相近表现选择after1x1。不能引用为“GCNet证实3x3前后最佳”。
- Table2(c)在c3/c4/c5各stage的残差块加入GC，bbox AP为37.9/38.9/38.7，三stage合用39.4。按ResNet50结构推得分别4/6/3个，合用13个；这里的all不是四stage共16个。阶段、通道容量与数量同时改变，不是纯数量控制。
- Table4(a)ImageNet单GC为77.34% top1，all GC为77.70%，baseline76.88%。这是该任务/训练配置的证据，不能直接移植为SAR收益。

GC采用全局上下文加法，本项目M2是三片区域摘要及各自区域回写；二者不是同一模块的复现。

## 3. Non-local：单模块阶段比较与1/5/10数量比较

Wang等，Non-local Neural Networks，arXiv:1711.07971。已读 `literature/Nonlocal_CVPR2018.pdf`，Sec.5.1、Table2(b,c)，以及Sec.6。

- Table2(b)：一个NL放在所选stage最后一个残差块**之前**。R50 C2D基线Kinetics top1为71.8%；res2/res3/res4/res5分别72.7/72.9/72.7/72.3%。不是最后块内部conv2插入。
- Table2(c)：1个放res4；5个为res4三个、res3两个，间隔残差块；10个覆盖R50的res3/res4全部4+6个残差块。R101使用对应的残差块，不能说“10个覆盖R101这两stage全部块”。

| 数量 | R50 top1/top5 | R101 top1/top5 |
|---|---|---|
| baseline | 71.8 / 89.7 | 73.1 / 91.0 |
| 1 | 72.7 / 90.5 | 74.3 / 91.3 |
| 5 | 73.8 / 91.0 | 75.1 / 91.7 |
| 10 | 74.3 / 91.2 | 75.1 / 91.6 |

数量收益并非无限单调：R101的5到10个top1持平、top5略降；Sec.6的COCO骨干添加更多NL也报告边际回报递减。原论文支持单独研究数量，不支持本轮自动扩成全排列。

## 4. 本轮准确解释

以 `PROTOCOL_ZH.md` 为本轮冻结协议。新增浅层M2仅在layer2最后块layer2.3作用一次，VV/VH分别一套参数；原深层模块保留。

| arm | 新增M2作用点 | 特征 | 新增参数 |
|---|---|---|---:|
| baseline | 无 | — | 0 |
| stage | layer2最终输出 | 512x20x20 | 197328 |
| pre3x3 | layer2.3的BN1/ReLU后、conv2前 | 128x20x20 | 49872 |
| post3x3 | layer2.3的BN2/ReLU后、conv3前 | 128x20x20 | 49872 |

参数值与本轮runs.json一致：baseline总可训练1114112，stage1311440，内部1163984。两极化、每个三区域、hidden32，容量差来自输入/输出通道；内部新增参数仅stage的约25.3%。**pre/post等容量，内部对stage同时改变位置、容量及残差支路作用路径，不是纯位置因果隔离。**运行时间也受缓存边界不同影响，不能将runs.json的训练秒数直接当作端到端推理效率结论。

已有原图/扩围结果支持检验区域摘要如何被后续卷积利用，但不证明内部一定改善。复用baseline/stage六个旧模型并逐个重放，只新训练pre/post各三seed；本轮不是12次新训练。数据为136适配、160反复使用的开发查询、16固定参考；开发集只有3过境、7个出现类，三seed不等于三个独立数据集。

现已产生数值：pre平均OA43.9583、BA33.3570，相对baseline均降低，三seed均无救回；post平均OA44.7917、BA34.5395，相对baseline是OA略降/BA略升且NLL/Brier改善。内部两臂准确率都低于stage。post相对等容量pre的OA/BA/NLL/Brier三seed均改善，仅支持本轮范围内post更合适，不证明内部胜过stage。详细seed变化及负结果见 `NUMERIC_REPORT_ZH.md`、`metrics.csv`。同权重bypass及主线path_diagnosis只解释训练后依赖，不替代重新训练对照或像素物理因果。

## 5. 核实来源与交付

官方下载来源（仅作来源记录；本地全文是本次核实依据）：

```text
https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf
https://arxiv.org/pdf/1709.01507v4
https://openaccess.thecvf.com/content_ICCVW_2019/papers/NeurArch/Cao_GCNet_Non-Local_Networks_Meet_Squeeze-Excitation_Networks_and_Beyond_ICCVW_2019_paper.pdf
https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf
```

`analyze.py`仅导入上一轮analyze.py的CPU指标/原图函数，动态读取arms/seeds，读取本次小型logits及当前release manifests，不导入runner/model、不读取大型cache。默认运行输出metrics/classes/grouped/bypass CSV、summary、NUMERIC_REPORT、CASES图库；`CASES/cases.json`沿用旧字段并增加reference_arm/per_seed/panels。比较stage时旧baseline_*字段指reference_arm，已在baseline_field_semantics中明示。

图库含指定诊断ID：34278、28657、28198、35666、20791、35906、20328；同时包含内部对stage的案例及实际同类/竞争类参考。VV/VH分开、原生1:1显示，完整16参考库在data_preview。生成图库不代表已人工回看；本支线目前未声称完成这些新案例的人工图像解释，由主线回看并结合path_diagnosis收尾。
