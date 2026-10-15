# 24 港跨场景完整模型：冻结折与运行接口

**状态更新（2026-10-15）：**以下是冻结时的历史记录；两港四景完整模型已运行，见同目录 `FULL_MODEL_FINDINGS_ZH.md`，六港扩围汇总见 `../ports_six_results_20261015/FINDINGS_ZH.md`。下文“尚未训练/评价”仅描述冻结时状态。

2026-10-14。这里只完成**划分和数据资产准备**，没有启动 GPU 训练或完整 C4／类别关系／预算交接评价。四折均在查看本轮模型结果前按元数据和类别容量冻结，目标**整港**留出；查询来自目标港的一次采集，同采集和异采集支持都来自目标港，后者固定为另一采集。目标港的全部采集和全部 MMSI（包括未进入查询/支持的船）均从源 SSL、源验证和源分类头拟合中排除。源验证还与 SSL 训练按采集、MMSI 隔离。

## 可运行折

| 目标港/查询采集 | 异采集支持采集 | SSL 训练 / 源验证 / 源头拟合 | 查询船 | 查询八类计数，类 0–7 |
| --- | --- | ---: | ---: | --- |
| Singapore / `S1A_060968_079865` | `S1A_061668_07B48D` | 24948 / 1830 / 26778 | 167 | 20, 6, 6, 75, 13, 15, 14, 18 |
| Singapore / `S1A_061668_07B48D` | `S1A_060968_079865` | 24948 / 1830 / 26778 | 166 | 20, 4, 7, 76, 18, 14, 13, 14 |
| Port Klang / `S1A_060888_079528` | `S1C_005316_00A908` | 26264 / 2370 / 28634 | 66 | 9, 3, 5, 22, 12, 7, 6, 2 |
| Port Klang / `S1A_061063_079C2A` | `S1C_005316_00A908` | 26264 / 2370 / 28634 | 58 | 14, 3, 4, 18, 10, 5, 2, 2 |

每折源训练/验证均有八类：Singapore 各有 614/62 个采集，Port Klang 各有 595/69 个采集。源验证按预定十组采集划分，从满足八类、至少 1000 张验证样本的候选中选身份去重后源训练量最大的一组，仅用源元数据决策；这比旧五组划分保留了更多合法源数据。Singapore 验证集的第 7 类只有 19 张，模型选择时须注意该类波动。逐类源数、支持候选数、采集数见 `fold_summary.csv`；逐角色、采集、产品类型、类别的裁片和独立船数见各折 `composition.csv`。目标查询每船仅一张；同/异采集 B16 均为**八类各 2 艘、五次配对抽样**，两种支持与查询的 MMSI 不重叠。Port Klang 的跨采集对照也跨 S1A/S1C 卫星；不能归因于单一采集变化。

同港两个查询折已去掉重复查询 MMSI，但并非独立港口，也不能把五次支持抽样当五个训练种子；跨折支持候选和来源可能复用。建议主表逐折列数，港口等权汇总，并同时给船级 OA、八类 BA 及稀有类敏感性结果。Port Klang 后两折类 6/7 的查询各只有 2–3 艘，BA 易受个别船影响。当前支持采用目标类别信息预先配额，是匹配诊断协议，不应表述为预测时盲选 16 个标注案例。

## 运行接口

- `python freeze.py` 重建冻结 CSV/索引；`python verify.py` 验证源/目标隔离、索引一致、八类覆盖及五次 B16 类配对。脚本仅读既有清单、裁片缓存 ID 和形状，**不训练模型**。
- 每折目录名为 `<Port>__<QueryAcquisition>`；`split.npz` 的数组为 `ssl_train`、`source_validation`、`source_fit`、`query`、`support_same`、`support_cross`、`support_classes`。前四项都是与 `manifest.csv`/`source_inputs128.npy` 对齐的全局行索引；两个支持数组形状为 `(5,16)`，`support_classes` 长 16 且顺序与每次支持行一致。
- 同名 CSV：`source_train.csv`、`source_validation.csv`、`source_fit.csv`、`query.csv`、`query_labels.csv`、`support_candidates.csv`、`support_draws.csv`。`query.csv` 不带标签，`query_labels.csv` 单独保存评价真值；支持 CSV 有 `support_origin=same|cross`、`draw=0..4`、`budget=16`。下游源分类头和源 PCA 仅在 `source_fit` 上拟合，不能把目标查询/支持加入源头；目标支持仅用于该折适配。
- 原始清单：`E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv`；完整 128 像素对缓存：`D:/SAR_completion_cache_20261010/source_inputs128.npy`；行 ID：相邻 `source_ids.npy`。冻结脚本已核验三者行序/长度一致。

## 仍缺的计算资产

新折的合法源 SSL R18 checkpoint 和对应**完整** G50/G18/D18 特征尚不存在，主代理须按每港 `ssl_train` 训练/选源验证 checkpoint，然后至少编码 `source_fit ∪ query ∪ 所有支持抽样`。**同港两折的源划分完全相同，只需两个 SSL checkpoint，不必四次重复训练。** 可复用 `scene_cross_dataset_redesign_20261014/ports_split/train_source_r18.py` 和 `f0921_split/evaluate_pilot.py` 的训练/编码实现，但其现有命令行只接受旧 Rotterdam/Singapore 两折，不能直接指向本目录。`cross_dataset/source24_raw.npz` 仅覆盖本轮约 61%–63% 源拟合行及部分目标查询，且其 D18 来自另一套源语料，不能作为本轮严格逐折 D18；本目录没有复制该缓存。旧 `pooled_features.npy` 是历史空间特征管线，不等于新像素编码 G50（抽查同索引的值已不同），不应静默混用。完整 C4、类别关系、预算交接仍须在这套冻结行索引上重新接入、记录成本与逐折结果。

本轮验证：`verify.py` 四折全通过；两港各两个折的重复查询船数均为 0，单折源/目标采集与身份交集均为 0。范围边界是**两个有足够单采集八类容量的目标港、四个相关折**；不能据此声称整个 24 港都能做同协议完整八分类。其他港口若扩展，应另定义缺类查询/支持协议，不要硬并进这张主表。
