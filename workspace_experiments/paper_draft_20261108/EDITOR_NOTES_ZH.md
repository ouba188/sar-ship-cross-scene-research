# 新版论文工作说明

更新日期：2026-11-09。正文为 `MANUSCRIPT_ZH.md`，本说明不是论文正文。

同日后续用户授权“多数沿用，只补最要紧”：已新增当前组合M3配套对照，见末尾缺口更新及正文6.6/表9。以下“本轮未新增实验”等句保留为前一次图文整理的历史记录，不代表本次补实验状态。含新表的IEEE版为 `manuscript_ieee_20261109_experiments.pdf`（13页）；原 `manuscript_ieee.pdf` 被外部程序占用，未强制关闭用户程序。Markdown、IEEE源文件、HTML及单栏PDF已同步更新。

11月9日增补：用户五页流程图已作为图3–7接入；原案例顺延图8–9。参考文献增至27篇并按首次出现顺序编号，以下历史核对记录中的文献编号保留旧稿口径，映射见literature_20261109/CITATION_NUMBER_MAP_ZH.md。新增13篇的正文阅读、方法差异及来源均已单独归档。本轮未新增实验。

## 本轮完成范围

新建完整结构的中文论文讨论稿，包括中英文题目和摘要、引言、相关工作、问题与干预证据、M1/M2/M3公式、分阶段优化、已有实验表、正负案例及讨论。没有训练、调参、修改模型或把旧权重迁移为新模型。10月27日旧稿和10月31日证据汇编均保留原样。

用户提供的流程图原件保留，论文副本校正M2输入名与映射、M3行列边际及比较步骤等。自然观测、输入干预和模块案例沿用已有真实面板，未改变SAR像素。总图的源端干预训练与当前模块阶段实验仍分开报告。

## 模型版本锁定

以 `../m1_guided_m2_20261106/current_version.json` 选择的OSS2四分类工作版为准。

- M1：`RelationFeedback`，每极化一份；layer2[3]的conv2之后、conv3之前。
- M2：layer2输出区域读出及类别响应引导补充；layer3[5]的conv1之后、conv2之前区域回写；M3后layer3输出的深层区域读出。
- M3：`Model('aggregate', seed=17)`，`MaskedCostLocalMatch`，类别证据聚合与回写。最近M1/M2训练使用其固定锚点残差，不实时随新版特征重算。
- 无额外layer2[2] M2，无M1–M3显式频域预测分支；历史八分类工程独立保留。
- 末次训练仅更新新区域读出和分类头。新增读出17/29/43三个种子共享上游seed17，不能写成三套从头联合训练。

新版算法写入方法，但没有把历史八分类干预成绩算作新版性能。M3的独立阶段收益也没有与M1/M2收益相加。

## 结果来源

| 正文位置 | 主要来源 | 用途及边界 |
|---|---|---|
| 第3节、表1/2 | `../research_problem_evidence_document_20261031/研究问题与证据汇编.md`，第1、3、5–8节 | 已归档的问题发现和受控输入依赖；不是新版主表 |
| 3.1框内/整图训练 | `../same_vessel_gallery_20261019/crude_input_controls_20261021/FINDINGS_ZH.md` | 历史Ridge配套输入对照，不是当前神经模块消融 |
| 第4节M1 | `../m1_covariance_20261101/operators.py`、`run.py` | 实际算子和位置 |
| 第4节M2 | `../m1_guided_m2_20261106/study.py`、`../sar_m123_standalone/sar_m123/bottleneck_evidence.py` | 响应引导读出与内部池化回写分开 |
| 第4节M3 | `../m3_frequency_increment_20261028/class_competitive_matching_20261031/operator_block.py`、`local_scale_repair_20261101/scaled_operator.py` | 聚合仍计算传输和参考权重，不是廉价均值 |
| 4.6干预流程 | `../sar_m123_standalone/sar_m123/current/interventions.py`、`training.py` | 最新去掉显式描述复核；floor当前使用原图教师，而非干预后教师 |
| 表4 | `../m1_coverage_20261101/coverage.csv` | 源、开发、评价的类/船/过境组成 |
| 表5/6、图6 | `../m1_guided_m2_20261106/FINDINGS_ZH.md`、`averages.csv`、`summary.csv` | 当前新版读出，固定上游和32源参考 |
| 表7、图7 | `../m3_frequency_increment_20261028/source_hard_task_20261101/oss2/paired_repeats/FINDINGS_ZH.md`、`internal_contribution/writer_review/FINDINGS_ZH.md` | M3独立开发对照，不能标成新组合的消融 |
| 表8 | 同上 `internal_contribution/expanded/FINDINGS_ZH.md`、`mean_std.csv` | 125/378/503三队列嵌套关系，保留负结果 |
| 6.5训练约束 | `../scene_aware_classification_20261018/model_closeout_20261023/intervention_selection_expansion_20261027/FINDINGS_ZH.md` | 历史3075队列，额外优势主要在概率评分 |

以上路径供本机追溯。GitHub归档不含原始数据、参考清单、检查点或大型缓存，不能承诺无数据即可重现实验。

## 文献使用

正文[1]–[11]复用 `../paper_draft_20261027/REFERENCES.md` 中已经核实的元数据和全文阅读范围，并保持对应编号。旧稿[3]只核实书目信息，本稿不引用其详细算法或数值。[4]–[6]使用已读期刊方法正文；[7]以作者稿核对方法，不引用未经比对的最终期刊数值。

[12] DeepEMD方法依据已有本地精读记录 `../m3_frequency_increment_20261028/resolution_matching_20261031/M3_NEXT_NOVELTY_ZH.md`，本轮另从CVF官方页面核对作者、题目和页码（12203–12213）。官方入口：

https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.html

本轮没有声称穷尽截至2026年11月全部相近工作。[13]的作者、题目和页码已由CVF官方页面核对。[14] SCRC-Net已读取用户本地PDF首页和第5页3.3节，核对作者、期刊、DOI及2026年6月22日上线日期；正文加入其结构约束采样与本文区域证据读出的比较，不将可旋转采样作为本文创新。原文位置：`E:/Docms/无监督域适应/translated/0917/related_papers/SCRC-Net.pdf`。尚未完成该方法在本文同协议下的性能对照，不能据此声称优于它。

## 投稿前只补最关键内容

2026-11-09最小补实验更新：`essential_experiments_20261109/FINDINGS_ZH.md`已完成当前组合M3末阶段配套移除，新增3次180步训练，复用旧完整组合及mean/content结果。评价378的M3平均增量OA+0.353、BA+0.760个百分点，NLL下降0.00963，Brier近乎持平；开发228为+0.877/+1.149。正文新增6.6及表9，原表1–8保持原值。该结果补上当前组合的一项关键对照，但不是从头全网络消融，不替代旧E/R负结果，也没有完成新版干预适配。以下缺口按此更新理解，避免重复运行已补部分。已有结果复用范围见同目录REUSE_ZH.md。

1. **当前整套工作版的统一结果主表。** 在一份明确冻结的协议下重放/评估新组合，并设置相同信息、训练机会与参考预算的强基线。现在稿中有阶段性有效结果，但没有把它伪装成最终全系统消融。
2. **对应主张的最小配套消融。** 至少区分M1反馈、M1引导M2读出、M2区域身份、M3类别证据回写；不把不同历史载体的差值相加。能复用缓存的先复用，不自动重启大量架构搜索。
3. **干预约束增量验证已补，结果不支持替换权重。** `intervention_validation_20261109/FINDINGS_ZH.md`完成新版教师重筛241条、128条有效和两组各三次96步适配。评价378中干预对普通OA−0.088、BA+0.157个百分点；两组适配都弱于原模型，源内拟合却改善。正文6.7及表10保留这一结果。当前正式模型不变，不再把此项列为未做，也不沿用历史训练收益为新版背书。
4. **锁定稿件范围及直接相关文献。** 现有证据适合“干预启发的局部证据组织”主线；尚不支持因果因素分离、普适不变表征、精细匹配优越性或SOTA标题。扩大数据集是扩大主张时的任务，不是把所有既往实验推翻重做。

本稿已可供导师讨论结构、机制和现有数据，不是已经完成投稿验证的终稿。没有承诺当前三个模块均具独立强算法原创性；创新论证聚焦信息如何组织、接口如何利用及具体可验证的改进。

本轮最新IEEE交付为`manuscript_ieee_20261109_intervention.pdf`，13页，已查看更新后的第11–12页；原`manuscript_ieee.pdf`仍被阅读器占用，未强行关闭或覆盖。HTML和单栏PDF已同步更新。之前“没有完成新版干预适配”的表述只描述上一步状态，本轮已由表10和上述第3项取代。

## 校验口径

- 检查数字和模型版本，目的是防止旧版数据误归新版或把重复事件写成独立船。
- 检查公式与代码，目的是防止描述不存在的实时M3、layer2[2]或干预后目标分差。
- 检查PDF页面及HTML资源，目的是发现公式越界、缺图和不可读表格。
- 不为本轮写作重跑模型训练或扩大实验。
