# 显示处理与跨数据集失效：独立设计审阅

日期：2026-10-13。状态：设计审阅完成；新实验结果尚未收到，未运行拟合、推断或数值审计。仅创建本文件，未修改共享代码或历史产物。以用户最新方案为准：fresh original/display 源特征，fold2/3，各自重拟合 geometry/heads，主臂 R/T/DD，不训练重建器。

## 判断与最小设计

更新后的设计合理，能检验“旧源端拟合与 display 输入不匹配是否造成可恢复的损失”。不能仅凭它认定唯一物理根因，或把失败直接归为对比度信息丢失。省去 MLP/P 重建对照不会妨碍本轮首要问题；T-P 和残差独占性不属于本轮新增结论。

每一折、每个任务、每个 draw、每个 readout 保留以下四格。O/D 分别表示 original/display；M 表示 BA/OA 等越大越好的指标，风险另行保持清楚的符号。

| 源拟合输入 | 目标支持与查询 O | 目标支持与查询 D |
|---|---|---|
| fresh O | M_OO：本轮正式基线 | M_OD：目标单侧处理 |
| fresh D | M_DO：反向失配对照 | M_DD：两端处理 |

- M_DD - M_OD：固定 display 目标，源端配套重拟合的恢复量。
- M_DD - M_OO：匹配协议的实际净变化，决定修复后是否仍有任务损失。
- I = (M_DD - M_OD) - (M_DO - M_OO)：源端处理效果是否依赖目标处理。恢复量为正但 I 很小，不能特别归因于两端匹配。
- 同时报告 M_OD - M_OO、M_DO - M_OO；不能只挑 DD 优于 OD。

固定原五次 B64 支持、标签、顺序和每次查询 ID，保持 ridge=8、temperature=1、equal-softmax fusion。只做冻结 display，不增加强度或裁切搜索。R/T/DD 的绝对表现与 T-R、DD-T 分别报告，避免把先验路径收益混作第二路核收益。

## 必需的 Held-Source 比较

最低成本方案就是已有自建留港任务：fold2 的 Los Angeles，fold3 的 Port Klang/Santos。它们属于源语料域，但未进入对应折源拟合，适合作为 held-source-domain 诊断；不是新的独立确认集，也不称严格 IID 源测试。

这些港口和 OSS1/OSS2/OSS2_GRDH 都必须给出四格 R/T/DD 的 prior-only 与 B64 结果。无需为了本轮重新切源训练集；如果改切训练行，旧 source-fit 基线便不再匹配。源训练集上的准确率或重建 MSE 不能替代 held-source 表现。

聚合维持三港等权，公开任务分别按两折等权；再显示同折港口对公开目标、同类别的变化，防止 fold2 的一个港与 fold3 的两个港被隐式重加权。先给每任务和每类别结果，尤其保留渔船、集装箱、原油轮；不把港口缺类直接算成八类 BA 的零召回。

解释规则：

- DD 相对 OD 恢复、且 I 支持匹配依赖：支持源端拟合不匹配是当前冻结管线损失的一部分；用 DD-OO 量出仍剩多少损失。
- held-source 改善而公开目标仍退化：更接近跨域迁移限制，不能只解释为源拟合没学会。
- held-source 与公开目标在匹配 DD 下仍退化：支持该显示处理在当前冻结表征/读出下代价较大；尚不能证明物理信息不可恢复或编码器重训也无效。
- T-R 变小但 R/T 绝对表现提高：保留第一路追赶解释，不宣布第二路被破坏。

## Source-Only Prior 的定义

保存 r50、dual、base 三个源头的 query logits/probabilities。R 与 T 的 prior-only expert 应完全相同，DD 使用 dual；这是本轮主臂不需要 mapper 的原因之一。

prior-only compact C4 必须是 0.5 * [softmax(base prior) + softmax(expert prior)]，两路都不加支持修正。若 base 仍经过支持适配，应另标为混合条件，不能叫 source-only。prior-only 指标也用每次同一 query mask 评估，才能与 B64 配对。

可定义 A_st = M_adapted,st - M_prior,st，报告 A 的四格差异；这是“加入支持修正的条件效果”。源先验本身会改变支持残差目标，geometry 又会改变核，softmax/argmax 非线性，因此不是独立可相加的先验原因与适配原因。当前无需再加交叉拼接 heads/geometry 的拟合实验。

## 拟合边界与实际接入风险

1. 用户计划的 36095 行只是两折特征缓存并集。每次拟合严格使用对应历史 source_rows：fold2 31335 行，fold3 27417 行。并集含另一折的留出港，绝不可全并集拟合 PCA、尺度、discarded statistics 或任一 head。
2. 保留 crossfit.permitted_rows 的完整排除：两公开原始 source_split 的交集，再与本折每个留港任务允许行求交。不可仅从当前 1349/3745 八类评估清单重建排除范围；历史公开清除覆盖了更广原始身份集合。
3. 缓存下标与全清单行号区分。旧 source_arrays 契约是 raw/c4 可按全局 rows 索引，而 extra 已按折筛好。新并集紧凑缓存必须先把全局行号映射为缓存位置；标签依然来自全局 manifest 对应行。混用可产生错行而不一定越界。
4. 同一源条件下重拟合所有输入依赖状态：raw PCA128、SSL PCA64、中心/尺度、retained/discarded geometry、r50/dual/base heads。仅换 head 或仅换 PCA 不构成本轮配套重拟合。
5. 原 prepare() 在 geometry.pkl 存在时直接返回，predict() 在 COMPLETE.json 存在时跳过。新输出按 source_condition/fold/target_condition 隔离，不能落到旧 fits 路径。原 predict() 无条件加载 mapper 并循环 KINDS；无 mapper 主臂实现须去掉该依赖，保留原 R/T/DD 公式。
6. 显示 quantile 是单张 VV/VH 的 pooled q01/q99；固定函数不依赖类标签、其他查询或整个目标集统计。源/目标必须复用同一函数和编码精度路径。

## 数值版本与最小检查

这里的“旧目标特征”应固定为 ports_input_controls_20261012/features 下 original/display 缓存，回放参考其 predictions；更早 ports_public_data_study_20261012 的目标特征有已记录数值漂移。fresh O 和 fresh D 都使用这份完全相同的目标缓存对照，不另提目标。

仅建议以下能检测具体失败、且失败后会改变行动的检查，不扩展测试框架：

| 检查 | 要发现的具体失败 | 失败后的处理 |
|---|---|---|
| 逐折 fit_rows 与历史 source_rows 完全一致；缓存 ID/标签对齐；查询/支持保持历史值 | 并集误拟合、全局/局部下标混淆、条件配对破坏 | 先修行映射和拟合输入，废弃受影响新结果 |
| 旧 geometry + 固定目标缓存，用新 R/T/DD 路径回放旧预测与概率，覆盖两折、O/D、两 readout | 缩减实现丢失尺度、偏置、类型转换、支持权重或融合规则 | 修推断实现，未对齐前不解释新拟合收益 |
| archived-source original fit 与 fresh-source original fit 在同一目标缓存上的逐格差异；旧 geometry 作为基准 | 源缓存重提取/运行环境差异伪装为 display 效应 | 单列漂移，正式效应改用 fresh OO/OD，漂移足以改变结论时先追对应类和翻转样本 |
| prior-only R=T；compact prior 两路修正均为零；实际 R/T/DD 支持修正复用 coefficients | prior 命名错误、遗漏 base 适配或双加修正 | 修读出定义与对应导出 |
| 各类召回均值还原 BA；逐样本 rescue-harm 还原 paired BA；逐类风险还原宏 Brier | 分母、缺类、draw/fold/port 权重错误 | 修统计与汇总，不重跑特征/拟合 |

“exact oldgeometry replay”需同时区分预测完全相同和概率误差，不将可接受浮点容差写成位级相同。原图 fresh fit 的微小硬预测漂移尤其可能被少数类 BA 放大；不要求重新提取后位级一致，也不拿历史最大漂移作为通用免责阈值。

## 高效复用与结论边界

源并集 O/D 各提取一次，按折 ID 映射访问。四套 geometry/head 拟合一次后共享给同折全部目标，不按任务重复拟合。原 R/T/DD 在旧脚本里两个 mapper kind 完全一致，新路径无须保留重复 kind 计算。缓存 support-union 坐标/核、分块 query 前向；无需构造全 query-query 核。

保持两 readout 可区分 expert 变化与 compact base 路径影响。旧 T-P/PE-P 证据只作历史背景，不冒充 display 源重拟合后的新结果。无需新增 MLP、bootstrap 或训练种子来完成这一轮有限干预诊断。

尚未被这个实验隔离的因素：display 同时改变灰度偏置、尺度、截断、取整及极化关系；编码器历史预训练暴露；公开产品/采集与年份差异；支持和查询船舶/产品共享；公开释放间身份重叠。OSS2_GRDH 是 OSS2 子集，两个公开释放也不是独立重复。主实验可以完成当前可检验根因的核查，但应允许结论为“确认部分原因，剩余未定位”。数据集事实核查由 Hypatia 负责，本审阅不重复开展。

## 精确代码入口

- `E:/临时会话/ports_public_data_study_20261012/crossfit.py:35`：permitted_rows；`:56`：source_arrays 的全局/子集数组契约；`:107`：configure_evaluation 会改全局 evaluate 路径。
- `E:/临时会话/public_oss_comparison_20261012/common.py:43`：shared_frozen 两公开 source_rows 交集。
- `E:/临时会话/public_oss_comparison_20261012/evaluate.py:27`：prepare 全套 geometry/heads；`:101`：predict；`:134`：旧 mapper 循环；`:181`：重建诊断，本轮无需沿用。
- `E:/临时会话/ports_input_controls_20261012/predict_inputs.py:8`：data_for 按 ID 复用 original/display 目标缓存，含 GRDH 子集映射。
- `E:/临时会话/ports_input_controls_20261012/transforms_fixed.py:14`：冻结输入变换；`extract_inputs.py:17`：编码器；`:36`：C4、bfloat16/float16 池化与 R18 提取规则。
- `E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/backbone_ablation_core.py:54`：r50_views 保留 sqrt192/sqrt2560 比例；`E:/临时会话/dual_cue_decomposition_20261012/run.py:41`：ssl_map。
- `E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/unified_head_core.py:16`：coefficients 的类别均衡权重、2*onehot-1 和 ridge=8。
- `E:/临时会话/ports_mechanism_evidence_20261012/analyze_flows.py:27`：完整 query 的类别权重；`:36`：rescue/harm 配对分解；`:89`：先任务内均值再汇总。

后续独立数值审计只需要读取新实验 protocol、fit_rows/IDs、各格概率/预测、prior-only 输出与两类 original 回放报告。最低核对全部四格主指标、配对交互量和聚合权重，再对关键类别重算救回/误伤；本文件不预先宣告数值通过。
