# 新版论文工作说明

更新日期：2026-11-08。正文为 `MANUSCRIPT_ZH.md`，本说明不是论文正文。

## 本轮完成范围

新建完整结构的中文论文讨论稿，包括中英文题目和摘要、引言、相关工作、问题与干预证据、M1/M2/M3公式、分阶段优化、已有实验表、正负案例及讨论。没有训练、调参、修改模型或把旧权重迁移为新模型。10月27日旧稿和10月31日证据汇编均保留原样。

用户要求流程图自己绘制，因此图3、图4、图5只留图位及图题，不生成流程图。自然观测、输入干预和模块案例沿用已有真实面板，未改变SAR像素。

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

1. **当前整套工作版的统一结果主表。** 在一份明确冻结的协议下重放/评估新组合，并设置相同信息、训练机会与参考预算的强基线。现在稿中有阶段性有效结果，但没有把它伪装成最终全系统消融。
2. **对应主张的最小配套消融。** 至少区分M1反馈、M1引导M2读出、M2区域身份、M3类别证据回写；不把不同历史载体的差值相加。能复用缓存的先复用，不自动重启大量架构搜索。
3. **干预约束在当前组合的增量验证。** 最新图像-only选择流程尚未在选定组合上重新训练；与同配额普通选样配对即可，优先检查概率评分与分类结果，不预设其必定涨OA。
4. **锁定稿件范围及直接相关文献。** 现有证据适合“干预启发的局部证据组织”主线；尚不支持因果因素分离、普适不变表征、精细匹配优越性或SOTA标题。扩大数据集是扩大主张时的任务，不是把所有既往实验推翻重做。

本稿已可供导师讨论结构、机制和现有数据，不是已经完成投稿验证的终稿。没有承诺当前三个模块均具独立强算法原创性；创新论证聚焦信息如何组织、接口如何利用及具体可验证的改进。

## 校验口径

- 检查数字和模型版本，目的是防止旧版数据误归新版或把重复事件写成独立船。
- 检查公式与代码，目的是防止描述不存在的实时M3、layer2[2]或干预后目标分差。
- 检查PDF页面及HTML资源，目的是发现公式越界、缺图和不可读表格。
- 不为本轮写作重跑模型训练或扩大实验。
