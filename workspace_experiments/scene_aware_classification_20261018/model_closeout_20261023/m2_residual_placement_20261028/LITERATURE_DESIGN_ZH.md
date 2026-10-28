# M2残差合并位置：文献依据与冻结设计

2026-10-28。本笔记复用并重新核对上一轮本地原文与方法笔记，不把文献中的收益当成本任务已获得的结果。原图解释由主线实际查看所选面板后完成。

## 本轮要回答的问题

上一轮128通道的conv2前后插入与512通道stage输出同时改变位置和容量，不能用于单独判断残差相加前后哪个位置更合适。本轮保持layer2最后块、512通道、每极化独立M2、hidden32、每臂新增197328参数，只比较残差合并与最终ReLU周围的作用点。

| arm | 名称 | 准确作用点 | 新增参数 | 训练来源 |
|---|---|---|---:|---|
| baseline | 无新增浅层M2 | 原模型 | 0 | 3个旧模型复用并由主线重放 |
| preadd | C | conv3及BN之后、identity相加之前 | 197328 | 3次新训练 |
| postadd | D | identity相加之后、最终ReLU之前 | 197328 | 3次新训练 |
| stage | E | 最终ReLU之后 | 197328 | 3个旧模型复用并由主线重放 |

seed固定17/29/43，136源适配、160开发查询、16真实固定参考不变。共6次新训练与6个复用模型，不是12次新训练。开发集反复使用且仅3过境、7个实际出现类；所有八类仍进入概率归一化与Brier。主线负责训练、非零回写顺序检查、重放及路径干预，本支线只分析保存的logits与原图。

## SE原文支持什么

本地来源：`../m2_block_placement_20261028/literature/SE_CVPR2018.pdf`及其全文，扩展版`SE_arxiv_v4.txt`的Sec.6.5、Table14/15与Fig.5；上一轮已核实笔记为`../m2_block_placement_20261028/LITERATURE_DESIGN_ZH.md`。

SE默认模块作用在残差变换输出、identity相加之前，因此为C提供位置层面的参考。arXiv v4（2019-05-16）的Table14中，默认SE的ImageNet top1/top5错误率为22.28/6.03，SE-PRE为22.23/6.00，SE-POST为22.78/6.35，SE-Identity为22.20/6.15。Sec.6.5明确SE-PRE在整个残差单元之前，SE-POST在与identity相加后的ReLU之后；因此SE-POST更接近本轮E，不是D，也不是上一轮post3x3。上述额外位置表来自扩展版，不能全归于CVPR 2018原版。

Table15的SE_3x3为22.48/6.02、25.8M参数，对照默认SE为22.28/6.03、28.1M。原文明确内部3x3处通道少、容量也降低。这支持上一轮容量混杂的识别，但不证明本轮C、D、E的顺序。SE是全局摘要驱动的乘法通道缩放，本项目为三区域摘要及区域回写；本轮不复现SE。

## GCNet原文与精确顺序边界

本地原文`../m2_block_placement_20261028/literature/GCNet_ICCVW2019.txt`，Sec.5.1.1、Table2(b)报告afterAdd与after1x1：COCO bbox AP均39.4，mask AP分别35.8/35.7。正文把after1x1定义为残差支路最后一个1x1卷积之后，作者据相近结果选择它作为默认。它不是conv2/3x3之后。

2026-10-28补核官方仓库`xvjiarui/GCNet`的`mmdet/models/backbones/resnet.py`，Bottleneck.forward的247–265行：conv3、norm3、context_block、identity相加、最终ReLU。这个默认顺序在位置层面与C相符，但模块内容、训练任务、覆盖块数量均不同。论文正文的afterAdd未在该段精确拆出相加和ReLU的边界；当前官方默认实现也不是其afterAdd消融的独立实现证据。因此**不把D称为GCNet afterAdd的精确复现，也不声称GCNet已比较本轮完整C/D/E顺序**。

官方代码核实地址：

```text
https://raw.githubusercontent.com/xvjiarui/GCNet/master/mmdet/models/backbones/resnet.py
```

本地GC原文用于核对表格，官方代码用于补核默认操作顺序。GC使用全局上下文加法，本项目仍是区域回写，不能用相同位置名称掩盖模块差异。

## 可检验的解释与本轮证据

C让M2读取残差变换、更新在identity合并后接受ReLU；D让M2读取已经混合identity的特征、更新仍接受ReLU；E读取经过ReLU的特征，新增更新在该块内不再经过这个ReLU。由此可以检验“区域摘要输入与其更新经过非线性的时机会影响类别竞争”，但不能仅凭三臂成绩断言某个物理机制成立。

CPU输出覆盖全部六个非自比较：C/D/E对baseline、C/D对E、D对C。报告逐seed及均值/标准差的OA、BA、NLL、Brier，逐类召回、港口/过境/极化分层，救回/误伤事件与独立查询/船只数。同权重bypass仅反映训练后路径依赖，不代替重新训练对照。若概率质量与准确率相反，分别报告，不强行排序成全面获胜。

案例固定seed17、每个比较/结局取初始固定竞争类分差变化绝对值最大的一个，ID打破平局；空结局不补造。展示原生VV/VH和模型crop/pad，保持1像素对1像素，附实际同类、固定竞争类及预测混淆类参考。案例记录保存三seed预测、分差及真实路径；生成图库不等于人工读图。主线结合路径干预解释后，才能形成最终原图结论。代表例不作为总体比例。

## 当前状态

分析脚本已完成正式CPU分析，默认读取本目录完整`predictions.npz`。C/D相对baseline四项指标均为三seed改善；相对E则NLL/Brier三seed改善、BA三seed降低。C/D/E平均OA分别46.6667/46.2500/46.4583，BA分别36.0543/35.9182/36.9726；不支持C/D全面替代E。详细数值与负结果见`NUMERIC_REPORT_ZH.md`和`summary.json`。

图库另补主线全部10条路径诊断案例及seed43的D对C误伤代表例，共25组比较、15个不同查询、50张VV/VH面板。实际读图仍由主线完成，本支线不声称已经查看。未修改主线operator/run/diagnose/FINDINGS、记忆或Git。
