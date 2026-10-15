# 来源、检索与实际阅读范围

日期：2026-10-12。独立文献分支；只在本目录写入文件，没有运行实验、修改实验脚本或重算结果。已读项目 `codex-brain/INDEX.md`、`02-user-preferences.md`、`04-workflows.md`、`09-research-handoff-20261012.md`，以及本轮 `PROTOCOL_ZH.md`、`FROZEN_CHOICE.json`。使用 literature-review skill 组织来源和阅读边界，不将工作流指南当成学术证据。

本轮是围绕指定问题的六篇近邻原始研究定向调研，不是穷尽性系统综述；不能据此宣称全网首次。采用四篇本轮选定基准论文，另选 DKT 对照核条件推断、Ovadia 等对照概率风险及校准。PDF 经现有 pdftotext 转为本地文本，无依赖安装。官方代码仅下载阅读，不执行。

## 检索记录

通过 web.run 执行标题检索及扩展检索。早期若干调用返回空结果；随后显式指定 response_length 的检索正常返回原始来源。部分 web.open PDF 返回超时、403 或 OpenReview 浏览器验证；对应全文通过 PowerShell 直接下载作者 arXiv、CVF 或 PMLR PDF，并用 pdftotext 读取，故不把失败页面当作阅读证据。

成功返回结果的检索式：

- `Meta-Dataset ICLR 2020 Triantafillou`
- `A Broader Study of Cross Domain Few Shot Learning 600 same episodes IBM`
- `Bennequin Few Shot Image Classification Benchmarks Too Far Reality Semantic Task Sampling 2022`
- `WILDS Koh 2021 benchmark PMLR`
- `Patacchiola Bayesian Meta Learning Deep Kernels NeurIPS 2020`
- `Ovadia Can You Trust Your Model's Uncertainty 2019 Brier score`
- `"paired" "support set" "few-shot" evaluation`
- `"support set" "sampling" "few-shot" benchmark sensitivity`

检索返回的聚合站、论坛和百科未用来支持论文技术判断。扩展检索还发现 Setlur、Li、Smith 的 *Is Support Set Diversity Necessary for Meta-Learning?*，只读了作者 [arXiv 摘要与元数据](https://arxiv.org/abs/2011.14048)，其摘要讨论训练任务固定支持集；未纳入六篇深读卡，也没有凭摘要断言它与本方案不存在技术重叠。ICCV 2025 的 *Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting* 仅检索命中，未阅读全文，亦不作为新颖性排除证据。后续若把论文定位扩大为通用少样本基准，应进一步核查这类近邻。

## 六篇核心来源

### R1. Meta-Dataset

Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku Evci, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol, Hugo Larochelle. *Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples*. ICLR, 2020.

- [作者全文，arXiv:1903.03096](https://arxiv.org/abs/1903.03096)，本地 PDF 标注 v4，2020-04-08。
- [ICLR 页面](https://iclr.cc/virtual_2020/poster_rkgAGAVKPr.html)；[官方代码](https://github.com/google-research/meta-dataset)。代码仅定位来源，本轮未逐文件审阅。
- 本地 `sources/meta_dataset.pdf`、`sources/meta_dataset.txt`。
- 深读：第 3.1、3.2 节，第 5 节 ways/shots 分析与 Figure 2 相关正文，附录 .1 reporting recommendations、.2 sampling algorithm。文本约 219-335、431-438、522-537、783-872 行。
- 未声称：复现全部基线、读完全部数据集附录、审计现代版本 sampler。

### R2. BSCD-FSL / IBM CDFSL

Yunhui Guo, Noel C. Codella, Leonid Karlinsky, James V. Codella, John R. Smith, Kate Saenko, Tajana Rosing, Rogerio Feris. *A Broader Study of Cross-Domain Few-Shot Learning*. ECCV, 2020.

- [作者全文，arXiv:1912.07200](https://arxiv.org/abs/1912.07200)，本地 PDF 标注 v3，2020-07-17。
- [官方仓库](https://github.com/IBM/cdfsl-benchmark)；[官方 finetune.py](https://github.com/IBM/cdfsl-benchmark/blob/master/finetune.py)。
- 本地 `sources/cdfsl.pdf`、`sources/cdfsl.txt`、`sources/cdfsl_README.md`、`sources/cdfsl_finetune.py`。
- 深读：第 4 节问题定义，第 5 节方法结构/源模型选择，第 6 节完整 evaluation setup，第 7 节相关结果说明；重点文本 252-275、436-464 行。
- 官方实现实际检查：finetune.py 的 support/query 切分约 80-91 行、accuracy 和区间约 147-161 行、600 次及每类 15 query 参数约 164-209 行；README General information 与 Specific Tasks。
- 未声称：只凭 numpy seed 就证明所有方法实际 episode ID 完全一致；共享 600 episodes 的设计承诺来自论文第 6 节。未运行官方复现。

### R3. Semantic Task Sampling

Etienne Bennequin, Myriam Tami, Antoine Toubhans, Celine Hudelot. *Few-Shot Image Classification Benchmarks Are Too Far From Reality: Build Back Better With Semantic Task Sampling*. CVPR Workshops, 2022, pp. 4767-4776.

- [CVF 原始页面](https://openaccess.thecvf.com/content/CVPR2022W/VDU/html/Bennequin_Few-Shot_Image_Classification_Benchmarks_Are_Too_Far_From_Reality_Build_CVPRW_2022_paper.html)。
- [本轮选定正式 PDF](https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Bennequin_Few-Shot_Image_Classification_Benchmarks_Are_Too_Far_From_Reality_Build_CVPRW_2022_paper.pdf)；[作者 arXiv:2205.05155](https://arxiv.org/abs/2205.05155)。
- [论文列出的官方实现](https://github.com/sicara/semantic-task-sampling)，本轮未逐文件审阅。
- 本地正式版本 `sources/realistic_fsl_cvprw.pdf`、`.txt`；作者版本 `sources/realistic_fsl.pdf`、`.txt`。正式版本用于核对定义；作者版的文本行号用于快捷定位。
- 深读：第 3 节任务与评价分布、第 4.1/4.2 节 coarsity 和 Algorithm 1、第 5/6 节具体任务数与 shots/ways、第 6.2 节结果解释、第 7 节局限及未来工作。正式 PDF pp. 4769-4774；作者版文本约 130-447 行。
- 未声称：论文已经研究个体支持身份，或证明所有语义难度分层都会逆转方法排名。

### R4. WILDS

Pang Wei Koh, Shiori Sagawa, Henrik Marklund, et al. *WILDS: A Benchmark of in-the-Wild Distribution Shifts*. ICML, PMLR 139:5637-5664, 2021. 完整作者表以 PMLR 为准。

- [PMLR 原始页面](https://proceedings.mlr.press/v139/koh21a.html)；[正式 PDF](https://proceedings.mlr.press/v139/koh21a/koh21a.pdf)。
- [官方代码](https://github.com/p-lambda/wilds)，本轮只核实论文列出的项目位置，未逐文件审阅。
- 本地 `sources/wilds.pdf`、`sources/wilds.txt`。
- 深读：第 3 节 domain mixture 形式化；第 4 节 iWildCam、Camelyon17、FMoW 和 Amazon 评价定义；第 5 节 fixed-train/fixed-test/randomized/average 比较与 Table 1 说明；第 6 节结果及 discussion 的模型选择边界。文本约 196-275、365-447、514-526、561-587 行。
- 未声称：读完扩展版全部 G/H/I 附录、WILDS 本身是一套目标标注预算少样本协议，或 fixed-test 就等价于本项目核替换。

### R5. Deep Kernel Transfer

Massimiliano Patacchiola, Jack Turner, Elliot J. Crowley, Michael O'Boyle, Amos Storkey. *Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels*. NeurIPS, 2020.

- [NeurIPS 页面](https://papers.nips.cc/paper/2020/hash/b9cfe8b6042cf759dc4c0cccb27a6737-Abstract.html)；[正式主文 PDF](https://proceedings.neurips.cc/paper/2020/file/b9cfe8b6042cf759dc4c0cccb27a6737-Paper.pdf)。
- [作者全文含附录，arXiv:1910.05199](https://arxiv.org/abs/1910.05199)，本地 v4，2020-10-13；[官方代码](https://github.com/BayesWatch/deep-kernel-transfer)。
- 本地 `sources/dkt.pdf`、`sources/dkt.txt`、`sources/dkt_README.md`、`sources/dkt_DKT.py`。作者名单以论文为准，README 的简略 BibTeX 漏列一名作者，不照抄。
- 深读：第 3、3.1、3.2 节式 (6)-(8)、第 5.2/5.3 节、附录 D/E 的 kernel comparison、Table 5/7/9 相关描述。文本约 198-312、448-529、840-899、925-939 行。
- 实现检查：`methods/DKT.py` 中 `correct(..., N=0, laplace=False)`、199-270 行；support 条件化、默认零次测试优化、sigmoid(mean) 后 argmax。训练目标只抽查 `train_loop` 的相关代码，不作全仓库审计。

### R6. Predictive Uncertainty Under Dataset Shift

Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, Jasper Snoek. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*. NeurIPS, 2019.

- [NeurIPS 页面](https://papers.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)；[正式主文 PDF](https://proceedings.neurips.cc/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf)。
- [作者全文含附录，arXiv:1906.02530](https://arxiv.org/abs/1906.02530)，本地 v2，2019-12-17。
- 本地 `sources/uncertainty_shift.pdf`、`sources/uncertainty_shift.txt`。
- 深读：第 3 节 NLL/Brier/ECE 定义及式 (1)，第 4.1 的分布偏移实验设计和 Figure 1 解释；定位并读取附录中 Brier reliability/resolution 说明。文本约 149-255、1887-1891 行；其他任务的详细超参数和全部图表未逐项审阅。

## 本轮本地结果引用

NOVELTY_ZH.md 中的实验数字来自主代理在当前消息中提供的完成结果，本分支未重新计算。实验协议来自本轮父目录 `PROTOCOL_ZH.md`；a=0.75 的源端选择来自 `FROZEN_CHOICE.json`。文献判断与这些内部证据分开标注，不能把本项目结果归于外部论文。

本目录源码文件是下载的阅读材料，不是实验脚本修改；正式 PDF、作者稿、代码快照不重复计为额外研究。所有短引文加在一起远少于逐来源 25 个英文词，其余为解释性转述。
