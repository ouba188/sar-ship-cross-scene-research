# 外部稳定坐标、漂移与入射角固定接入

日期：2026-10-10。此报告冻结本地低覆盖角度版本；后续在线恢复必须单独出高覆盖版本，不覆盖本报告、原 incidence CSV 或原预测。

## 结论

已完成 F0921/OSS2 的独立实现与运行，以及24港23个任务的匹配共享核心对照。稳定空间不是只增加 Qs 核：本次实际包含 source-only Qs、正交保护 Qd、0.5 漂移衰减、源规范化、对应源先验与同64标签适配。神经专家、已有条件均值关系和分频表示冻结；不能称为历史24港整套训练过程的完全迁移。

F0921 稳定分支退化主要定位到新增稳定源先验，核单独接入影响远小于先验，且统一头核单独接入有正增益。OSS2 入射角有明确的匹配增量，稳定选择没有显示普适优势。结果不经目标选优，所有预定臂和负结果均保留。

## 可信覆盖与映射

- OSS2：3745/3745，100%。原生 Ship.xml 的 SARShipInformation/Incidence；按 scene + 文件名 x/y 中心 + MMSI 唯一匹配，并核对经纬度误差不超过10m。Metadata.xml 的 incidence_near/far 明确 unit=deg；同一数据格式按度解释。范围30.7338至45.925679度。Chip_Direction、heading、AIS航向未用于入射角。
- OSS2 原始 XML 有错写闭合标签、船名编码和未转义字符；采用结构恢复解析，随后仍逐样本要求数值中心、MMSI和坐标一致。恢复日志记录在 oss2_units.json，全部3745个目标身份匹配成功。
- F0921：44765/123489，36.25019%。原始场景 acquisition 来自 f0921_tile_acq.json 的 tile/sensor/t/code/lon/lat；按 tile + 五位小数位置关联，再以 sensor/start/code 精确对应产品。实际使用产品 geolocation 网格凸包内插值，不外推、不用船向、不用场景中间常数代替。
- F0921 缺失：missing_product=78471，ambiguous_acquisition=139，outside_grid=106，missing_grid=8。已知产品中至少1073行与旧 tile首景映射不同，故旧 f0921_chip_incidence.csv 不能直接复用。
- 定点检查三个既有 SAFE 目录，共361个目录，无 annotation 网格 XML；有 manifest/AIS轻量元数据不能视为具备逐船入射角。未全盘扫描、未下载。
- 后续 recover_f0921_ambiguous.py 对139行在对应shp全体记录中按MMSI和五位小数坐标核对：113行无唯一身份位置匹配，26行存在两个采集候选，0行唯一。139行全部保留ambiguous并使用缺失回退。此前按row_in_shp得到的7条仅为未确认候选，虽单条MMSI/位置吻合，仍不能排除同船同坐标不同采集时刻；f0921_native_recovered.json保留作原始核对记录，不并入任何实验。此结论仅涉及入射角元数据归属，不推断历史裁片问题。
- 在线恢复交接：recovery_inventory/missing_rows.csv 有78724行；acquisition_requests.csv 有12874个去重请求项；歧义候选另存。请求键是 sensor + acquisition_start + product_suffix，不能以tile首景替代。
- 清理后的源端可靠角度覆盖：F源75.40129%，OSS源74.87837%。源角度也不是全量；无角度行的二次条件基为0，表示缺失输入处理，未制造角度值。

F缺角查询完整保留：angle_only 精确回退到 matched_refit，joint 精确回退到 stable_drift。五次支持均仍为原64个，支持元数据缺失时角度基为0，不额外申请目标标签。subset指标只是同一全量策略在相同子集上的读出，不重新抽样或筛掉困难查询。

## 协议与固定权重

F源29717行、OSS源36996行，严格复用各自 source_split.npz，不扩大源身份队列。F目标使用最新 detector_recrop 队列123489行；OSS目标3745行。两者保留原5组64标注和eval_masks，八类输出。

Qs：源标准化原图R50，经同港类别对比、类内跨港均值惩罚1、SVD候选子空间；F秩28、OSS秩26。Qd：去掉Qs重叠后的类内港间均值方向，秩16。Qs正交保护误差约4e-16。普通PCA控制同样在清理后源的标准化原图特征拟合，秩/规范化/新增核权重/源头拟合一致。pca_drift沿用相同Qs保护的Qd，以单独替换新增坐标选择，未声称Qd保护PCA。

漂移幅度0.5；稳定适配核基础权重1、专家0.25，沿用24港已有读出。源稳定分支缩放sqrt(128/r)、sqrt(192/r)。入射角为 t=(degree-38)/5 与 t²，基础核系数1/2，专家1/8。频率0.25、目标ridge8、源balanced ridge1、融合/统一核等权0.5。无权重搜索、无目标query调参。

matched_refit以当前部署输入重拟合源分类先验：基础用源C4表示，专家用已冻结的部署尾部表示。其他臂只在这个匹配控制上改变对应知识输入。原模型original单独保留，不能把matched_refit与original之差都记为知识收益。外部两数据的original基础/专家/融合/统一预测均逐draw完全复现。

## 外部全量结果

OA/BA均为百分数，五次支持抽样均值。

| dataset | arm | readout | oa | ba |
| --- | --- | --- | --- | --- |
| f0921 | angle_only | fusion | 34.66218 | 27.61564 |
| f0921 | angle_only | unified | 34.78647 | 27.21015 |
| f0921 | drift_only | fusion | 34.40438 | 27.64788 |
| f0921 | drift_only | unified | 34.66299 | 27.22495 |
| f0921 | joint | fusion | 32.49277 | 26.20191 |
| f0921 | joint | unified | 32.63715 | 25.99072 |
| f0921 | matched_refit | fusion | 34.53612 | 27.64994 |
| f0921 | matched_refit | unified | 34.72878 | 27.24262 |
| f0921 | original | fusion | 33.16621 | 28.00308 |
| f0921 | original | unified | 33.34965 | 27.60632 |
| f0921 | pca_drift | fusion | 33.65623 | 27.16413 |
| f0921 | pca_drift | unified | 34.02568 | 26.80065 |
| f0921 | pca_rankmatched | fusion | 33.65104 | 27.10387 |
| f0921 | pca_rankmatched | unified | 33.99554 | 26.73925 |
| f0921 | stable_drift | fusion | 32.43492 | 26.18059 |
| f0921 | stable_drift | unified | 32.62921 | 25.99585 |
| f0921 | stable_only | fusion | 32.34142 | 26.23101 |
| f0921 | stable_only | unified | 32.48888 | 25.95486 |
| oss2 | angle_only | fusion | 43.55882 | 34.66859 |
| oss2 | angle_only | unified | 43.11328 | 34.17861 |
| oss2 | drift_only | fusion | 42.16246 | 34.40749 |
| oss2 | drift_only | unified | 41.84189 | 33.71221 |
| oss2 | joint | fusion | 42.90682 | 34.50803 |
| oss2 | joint | unified | 42.77642 | 33.88752 |
| oss2 | matched_refit | fusion | 42.08639 | 34.08732 |
| oss2 | matched_refit | unified | 41.85276 | 33.69002 |
| oss2 | original | fusion | 40.76610 | 33.91470 |
| oss2 | original | unified | 41.07036 | 33.66770 |
| oss2 | pca_drift | fusion | 41.66259 | 33.79486 |
| oss2 | pca_drift | unified | 41.58653 | 33.41217 |
| oss2 | pca_rankmatched | fusion | 41.79842 | 33.77058 |
| oss2 | pca_rankmatched | unified | 41.83102 | 33.58501 |
| oss2 | stable_drift | fusion | 41.30399 | 33.83455 |
| oss2 | stable_drift | unified | 41.64086 | 33.66536 |
| oss2 | stable_only | fusion | 41.41809 | 33.84619 |
| oss2 | stable_only | unified | 41.83102 | 33.75673 |

相对匹配控制：F融合stable_only为-1.41893 BA，stable_drift为-1.46935，angle_only为-0.03430；OSS融合angle_only为+0.58127，stable_only为-0.24114，stable_drift为-0.25277。OSS稳定单独33.84619略高于同秩PCA33.77058，但仍低于matched34.08732；F稳定26.23101也低于PCA27.10387。不能声称稳定选择独特且普适优于普通维数增加。

## F角度子集

| subset | arm | oa | ba |
| --- | --- | --- | --- |
| all | angle_only | 34.66218 | 27.61564 |
| all | joint | 32.49277 | 26.20191 |
| all | matched_refit | 34.53612 | 27.64994 |
| all | stable_drift | 32.43492 | 26.18059 |
| angle_missing | angle_only | 33.66453 | 27.08074 |
| angle_missing | joint | 31.51056 | 25.66428 |
| angle_missing | matched_refit | 33.66453 | 27.08074 |
| angle_missing | stable_drift | 31.51056 | 25.66428 |
| angle_supported | angle_only | 36.41662 | 28.41403 |
| angle_supported | joint | 34.22007 | 27.03651 |
| angle_supported | matched_refit | 36.06888 | 28.62708 |
| angle_supported | stable_drift | 34.06051 | 27.05600 |

受支持子集只加角度相对matched，OA增加约0.34774，但BA减少0.21305；全量BA减少0.03430。类别比例使全量BA变化不等于覆盖率乘子集BA变化。缺角子集与相应无角度臂逐预测相同，不能据本地36.25%覆盖结果断言角度全量迁移效果。

## 24港共享核心

已完成23个有效留港任务、每任务5次原64标注、四种读出。matched_refit是无Qs、无Qd、无入射角的共享核心；每臂重拟合对应源先验和目标支持残差，不是硬删旧分数权重。按任务均值再等权汇总。

| arm | readout | oa | ba |
| --- | --- | --- | --- |
| angle_only | fusion | 64.31734 | 55.81883 |
| angle_only | unified | 64.29107 | 55.49789 |
| drift_only | fusion | 64.09631 | 55.50041 |
| drift_only | unified | 63.94627 | 55.11350 |
| joint | fusion | 64.41104 | 55.44775 |
| joint | unified | 64.47869 | 55.25062 |
| matched_refit | fusion | 64.10710 | 55.45470 |
| matched_refit | unified | 63.97812 | 55.07983 |
| original | fusion | 65.57430 | 55.98728 |
| original | unified | 65.66696 | 55.71262 |
| stable_drift | fusion | 64.22101 | 55.29875 |
| stable_drift | unified | 64.25163 | 55.01169 |
| stable_only | fusion | 64.26433 | 55.29694 |
| stable_only | unified | 64.34187 | 55.06708 |

共享核心融合BA55.45469，加入角度55.81883（+0.36413）；仅稳定55.29694（-0.15776），稳定/漂移55.29875（-0.15595），联合55.44775（-0.00695）。原历史完整模型55.98728保留作参考；联合重拟合臂与original不同，涉及C4源先验、部署专家输入匹配和缺角回退，不能将两者差简单解释为删模块收益。

该对照统一了当前表示读出/源先验/目标适配的知识开关。神经分频仍是历史训练后的固定表示，24港分频曾在带知识的源情景上训练，故尚不是“从源训练起完全同构”的新系统；本次没有重训分频或专家。几何、规范化和先验随任务源队列各自拟合，这是共享算法而非共享同一组数值参数。

## 稳定退化的2×2诊断

以下是诊断，不替代主结果。prior-only保持matched核仅换stable源先验，kernel-only保持matched先验仅加stable核。核权重、正则、支持标签完全相同；00与11两端逐样本精确复现原matched/stable结果。

| dataset | arm | readout | oa | ba |
| --- | --- | --- | --- | --- |
| f0921 | matched_00 | fusion | 34.53612 | 27.64994 |
| f0921 | matched_00 | unified | 34.72878 | 27.24262 |
| f0921 | stable_both_11 | fusion | 32.34142 | 26.23101 |
| f0921 | stable_both_11 | unified | 32.48888 | 25.95486 |
| f0921 | stable_kernel_only_01 | fusion | 34.57695 | 27.62105 |
| f0921 | stable_kernel_only_01 | unified | 34.77108 | 27.37043 |
| f0921 | stable_prior_only_10 | fusion | 31.05108 | 25.63989 |
| f0921 | stable_prior_only_10 | unified | 31.78513 | 25.58507 |
| oss2 | matched_00 | fusion | 42.08639 | 34.08732 |
| oss2 | matched_00 | unified | 41.85276 | 33.69002 |
| oss2 | stable_both_11 | fusion | 41.41809 | 33.84619 |
| oss2 | stable_both_11 | unified | 41.83102 | 33.75673 |
| oss2 | stable_kernel_only_01 | fusion | 42.23852 | 34.16430 |
| oss2 | stable_kernel_only_01 | unified | 41.96686 | 33.43817 |
| oss2 | stable_prior_only_10 | fusion | 41.15186 | 33.98479 |
| oss2 | stable_prior_only_10 | unified | 41.43983 | 33.69476 |

BA相对00的受控效应：

| dataset | readout | prior_effect | kernel_effect | both_effect | interaction |
| --- | --- | --- | --- | --- | --- |
| f0921 | fusion | -2.01004 | -0.02888 | -1.41893 | 0.62000 |
| f0921 | unified | -1.65755 | 0.12781 | -1.28777 | 0.24197 |
| oss2 | fusion | -0.10253 | 0.07698 | -0.24113 | -0.21558 |
| oss2 | unified | 0.00475 | -0.25185 | 0.06671 | 0.31381 |

F融合：仅稳定先验-2.01004，仅稳定核-0.02888，两者-1.41893，交互+0.62000个百分点。稳定先验的有害变化是当前退化的主要直接来源，稳定核部分抵消这一损失。统一头仅稳定核+0.12781而仅稳定先验-1.65755，进一步排除了“候选坐标完全没有可用支持信息”的解释。

F融合逐类，稳定先验使集装箱召回39.58985降至28.45272、渔船23.49179降至18.37244，同时部分油轮/拖轮/海工类增加。它是类别取舍不均的迁移，不是所有类别都下降。不能只凭此认定源稳定发现方向无效；也尚未区分源系数外推、分支缩放下的正则几何、真实跨域类别条件改变各占多少。

OSS融合仅稳定先验-0.10253、仅稳定核+0.07698、联合-0.24114，交互为负；统一头的核单独也非正。因此不能把F的归因比例推广到OSS，或将核单独诊断挑为新主模型。没有再调参数。

## 验证、边界与文件

验证已检测并排除：外部原模型接入漂移（四读出完全复现）、支持/查询重叠、身份/行序错位、Qd破坏Qs、漂移核符号或缩放错误（直接特征核与低秩公式误差<2e-15）、缺角分支静默丢查询（回退逐预测一致）。无新增测试框架。CPU两线程、无GPU、输出约81MB级。

五次支持抽样重叠，不是五次独立训练；数据已开发使用，F专家继承历史F语料SSL暴露，非独立确认。source-only候选约束源类内港间一阶均值，不等于因果不变性。后续高覆盖版本由在线恢复支线提供，须单独冻结并对照原36.25%版本。

代码（均在 sensitivity/oof_reliability_check 下，旧文件未改）：

- external_knowledge_metadata.py：只读原生元数据、低覆盖侧表；已有CSV现设为保留。
- external_knowledge_metadata_audit.py：SAFE定点审计、原生row初步核对；7条记录不是确认恢复值，后续全记录歧义核对优先。
- external_knowledge_recovery_inventory.py：在线恢复请求清单与annotation_grid解析接口。
- external_knowledge_run.py：源几何/对应先验、外部9臂、四读出、分覆盖读出。
- external_knowledge_ports.py：24港23任务7臂共享核心对照。
- external_knowledge_diagnose.py：稳定先验/核2×2诊断，含精确端点复现。
- external_knowledge_verify.py：真实源特征的保护/漂移/核公式验证。
- external_knowledge_report.py：报告、逐draw配对差值、缺角回退核对。

结果根目录：external_knowledge_20261010。

- PROTOCOL.json、metadata_coverage.json、f0921_incidence.csv、oss2_incidence.csv、coverage_by_class.csv。
- f0921/ 与 oss2/：source_fit.pkl、SOURCE_READY.json、policies.npz、FROZEN.json、draw_coverage.csv。
- metrics.csv、class_metrics.csv、summary.csv、paired_contrasts.csv：外部主结果与固定配对差值。
- self24ports/：每港source_heads.pkl和policies.npz、逐港/逐类/等权汇总。
- diagnostic_prior_kernel/：两外部预测、逐类指标、summary.csv、factorial_effects.csv。
- verification.json、closeout_checks.json、local_safe_audit.json、f0921_native_row_audit.csv、f0921_native_recovered.json。
- recovery_inventory/：远端恢复交接清单；不覆盖原低覆盖侧表。

运行入口：ClearSAR python -B external_knowledge_run.py --stage all；ports/diagnose/verify/report各独立入口。已完成结果默认复用，在线高覆盖版本须另目录，不能覆盖本版。
