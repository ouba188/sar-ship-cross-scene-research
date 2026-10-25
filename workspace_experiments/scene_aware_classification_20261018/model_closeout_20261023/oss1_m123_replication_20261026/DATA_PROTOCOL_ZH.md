# OSS1 四类冻结 M1-M3 复现：数据协议与固定划分

后续状态：本文件保留训练前的数据冻结记录；当前训练和原图回看已完成，最终结果见同目录FINDINGS_ZH.md及m3_results_grid01。下文“尚未训练/提取”指制定数据协议时，不是最终运行状态。

日期：2026-10-26。状态：数据准备完成；未提取新特征、未训练、未使用 GPU、未操作 Git。用户已选择先做四类充分覆盖复现。当前只固定一个外层 split，训练 seeds 为 17/29/43；不基于 3075 结果调整超参数，不按 OSS1 测试效果选折。

## 1. 当前唯一训练入口

本目录绝对路径：

`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/oss1_m123_replication_20261026/`

**使用 `stack_table.csv` 和 `stack_indices.npz`，以 `STACK_SPLIT.json` 为划分记录。** `four_class_roles.csv` 及其 source/val/test 文件保留为父划分与追溯依据，不是本次最终训练角色。

| 角色 | 观测 | 独立 MMSI | 过境数 | 用途 |
|---|---:|---:|---:|---|
| fit_train | 405 | 384 | 9 | A/D/T、M1/M2 拟合；类别 reader、PCA、参考库来源 |
| fit_validation | 89 | 87 | 3 | 前端超参数和 checkpoint 选择；不反传 |
| utility_train | 122 | 118 | 4 | 冻结前端后的 M3 训练与其训练统计 |
| selection | 187 | 177 | 4 | M3 超参数、融合和 checkpoint 选择；不反传 |
| evaluation | 63 | 63 | 1 | 完整模型冻结后评价 |

五角色共866条、829艘船。十个角色对的 sample_id、MMSI、pass_group 交集均为零，结果保存在 `STACK_SPLIT.json/isolation`。utility_train、selection、evaluation 均不进入前端拟合或前端选择，因此 M3 的训练输入不是前端在其自身拟合样本上的输出。

这不是全栈 nested OOF：前端逐级 A/D/T/M1/M2 在 fit_train 内仍可能使用前级的 in-sample 输出；M3 内部多个阶段如果共用 utility_train，也不是独立交叉拟合。明确这些边界，不把新协议描述成全流水线 OOF。

## 2. 原始来源与选折依据

数据总表：`E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/target_manifest.csv`。

已有元数据清单：本目录 `data_inventory.csv`、`view_inventory.csv`。本轮仅从这些清单和已存在的合法 split 生成角色，不再全面审计原始数据或读取预测结果。

固定采用最早的 strict OSS1 pilot，保持其源/目标边界：

- source 父表：`E:/临时会话/scene_cross_dataset_redesign_20261014/strict_training/oss1/source_train.csv`。
- val 父表：`E:/临时会话/scene_cross_dataset_redesign_20261014/strict_training/oss1/source_validation.csv`。
- test 父表：`E:/临时会话/scene_cross_dataset_redesign_20261014/public_split/oss1_query.csv`。
- 旧目标支持候选 `E:/临时会话/scene_cross_dataset_redesign_20261014/public_split/oss1_support_candidates.csv` 不加入 source、reference、selection 或 evaluation；旧支持抽样文件也不使用。

先取用户指定四类，并对三个父角色一致要求至少一路合格 OBB，得到660/187/63条；对应 `four_class_source.csv`、`four_class_val.csv`、`four_class_test.csv`。双路无框及非四类条目逐条在 `four_class_exclusions.csv`，没有根据预测排除样本。

然后仅拆660条 source：将其16个排序后的 pass 用 `numpy.random.default_rng(17).permutation` 打乱，前9个分配 fit_train，随后3个 fit_validation，最后4个 utility_train。按 utility_train > fit_validation > fit_train 的优先级保留跨角色 MMSI，低优先角色中同船观测剔除。首次固定方案即四类齐全，无候选搜索或结果导向调整。共剔除44条（source 的6.67%），清单为 `stack_mmsi_exclusions.csv`。

旧 val 原样成为 selection，旧 test 原样成为 evaluation。不把旧 val 合并回 fit_train 重训。各角色的确切 pass 列表及分配保存在 `STACK_SPLIT.json`，逐 pass/类别计数在 `stack_pass_coverage.csv`。

evaluation 固定为 `S1A_010568_00FB64` 的63条四类有框查询，是历史使用过的单过境队列；不称盲测、全1349条评价、三折复现或跨多目标过境确认。先完成本折三个 seed，再决定是否按预先冻结的规则扩围；本协议没有批准按 test 最优折挑结果。

## 3. 类别与覆盖

原始标注不修改。新四类头的编码映射为：

| 新 class_id | original_class_id | class_name |
|---:|---:|---|
| 0 | 0 | bulk_carrier |
| 1 | 2 | general_cargo |
| 2 | 3 | product_chemical_tanker |
| 3 | 4 | container_ship |

下表按新 class_id 0/1/2/3 顺序，格式为观测数 / 独立船数：

| 角色 | 散货 | 普通货 | 化学品 | 集装箱 |
|---|---:|---:|---:|---:|
| fit_train | 270/254 | 33/33 | 33/30 | 69/67 |
| fit_validation | 66/64 | 8/8 | 9/9 | 6/6 |
| utility_train | 89/87 | 8/7 | 11/10 | 14/14 |
| selection | 142/134 | 14/14 | 13/12 | 18/17 |
| evaluation | 15/15 | 7/7 | 10/10 | 31/31 |

所有角色均覆盖四类，前端 fit_train 足以构建每类8艘参考。utility 的普通货仅7艘，fit_validation 的集装箱仅6艘；可做当前有限范围复现，不能把三个训练 seed 或重复观测当作增加了独立船或采集证据。逐极化可用性见 `stack_coverage.csv`。

## 4. 字段与调用契约

```python
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/oss1_m123_replication_20261026')
table = pd.read_csv(root / 'stack_table.csv', dtype={'mmsi': str})
with np.load(root / 'stack_indices.npz') as z:
    ids = {name: z[name] for name in z.files}
fit = table.iloc[ids['fit_train']]
utility = table.iloc[ids['utility_train']]
```

- `table_index`、`row_index`、`up_index`：均为当前866行总表的0起紧凑索引。各单角色 CSV 保留这些全表索引；NPZ 数组必须索引总表或按总表顺序提取的数组，不可直接索引重置后的单角色表。
- `obs_id`、`target_manifest_row`：原1349条清单索引；`sample_id` 为持久连接键。不要把紧凑索引作为 detection_views 的 global_row。
- `role`：五角色名；`legacy_role`、`legacy_table_index` 保留父划分追踪。`upstream_role` 仅前两个角色取原角色名，其余为 external。
- `class_id`：0..3；`original_class_id`：0/2/3/4；`class_name` 原语义不变。所有输出、权重、BA 类空间均须使用本次四类，不可把旧8类 checkpoint 截列冒充重训。
- `mmsi`：按字符串读取；`pass_group` 等于 `overpass`，由平台、绝对轨道与 datatake 构成；`scene_id` 为原产品名。
- `chip_path_vv/vh`：原生单极化 Patch_Uint8 路径；`raw_patch_path`、`calibrated_patch_path`、`ship_xml_path`、`metadata_xml_path` 提供原始来源追踪。
- `valid_vv/vh`：当前检测可用标记；`obb_vv/vh` 为同义来源字段；`axis_vv/vh` 另记长轴有效性，不能误当与本次 OBB 入选条件相同。
- `reference_candidate`：只有 fit_train 为 True；`input_scale=1.0`、`canvas_size=160`。
- 几何通过 `four_class_views.csv` 的 `(obs_id, pol)` 连接，含原生高宽、cx/cy、length/width、ship_axis、confidence、detected、axis_valid、raw_xywhr。该视图文件还包含父划分的44条后续排除记录，必须按 stack_table 的 obs_id 取行。

原脚本 `upstream/pipeline.py::load()` 的 role==strategy_fit 断言及其他旧表地址不能原封不动使用；Darwin 的入口应按上述五角色与紧凑索引取数据。这里只提供数据契约，不改训练代码。

## 5. OBB 与原生图像

几何来源：`E:/临时会话/oss1_joint_geometry_20261017/detection_views.json`；其生成器为同目录 `detect_axis.py`，冻结检测器记录为 `E:/Docms/SARlabelimg/550k.pt`。检测器预训练样本来源尚未审计，不能声称检测器从未见过 OSS1。

本次每极化合格条件为已有 detected=True 且 confidence>=0.25，使用真实检测框；不使用未检出时的占位中心/长宽。当前模型的区域几何要求合格 OBB，并不要求长宽比>=1.2；故保留 axis_valid 为诊断字段，不额外添加这个几何筛选。

至少一路合格的观测保留；另一极化图像仍存在时，不把它伪造为有框目标。适配现有 prepare_view 接口时，无合格框的一路使用空有效检测、target_index=None、无猜测 ship_axis；基础背景路径可沿现有实现读取图像，M1/M2/M3 的目标证据须使用缺失 mask。不得复制另一路框或另一极化特征来补齐。两路均无合格框者已在父划分阶段排除。

沿用原生160画布 crop/pad、VV/VH分开、C4及 ImageNet R50 预处理；不 resize 成同一船尺寸，不将显示灰度解释成物理后向散射。图像周边补零是既有画布操作，与禁止重复参考补齐不同。特征提取尚未运行，若实际 crop_pad 报目标超出画布，须记录具体 sample_id 并回报，不静默 resize 或删样本改变固定分母。

## 6. 参考库契约

`stack_reference_candidates.csv` 仅由 fit_train 产生，按 obs_id 排序后每 MMSI 保留第一条，四类候选独立船数为254/33/30/67。实际 bank 还未选，不能称已有32张最终参考。

主线程在新特征提取完成后，沿用原 `local_evidence_transport/match_readout.py::make_bank()` 的规则：先确认有效 body mass，再按 MMSI 去重，对每类做 `KMeans(n_clusters=8, random_state=17, n_init=10)`，选8艘不同船作为32张参考；每个查询按原规则每类检索2张不同参考。若需与原实现完全一致，mass过滤应先于按MMSI去重，使用 `stack_fit_train.csv` 中全部405条作为输入；候选文件提供的是几何层面的容量清单，不替代实际mass过滤。

不得从 fit_validation、utility_train、selection、evaluation 或旧目标支持池借参考；不得重复船、重复行、复制特征 padding 凑8张。实际 bank 必须另存 sample_id/MMSI/obs_id/类别及来源角色，和全局紧凑索引的对应关系。PCA、类别 reader 及参考检索统计限于 fit_train。M3训练/选择/评价与参考库已由五角色划分隔离 MMSI 和 pass。

## 7. 缓存边界及最小接入

本机已确认1349条、1110艘、22过境；2698张VV/VH图像以及每条 raw/calibrated patch、XML路径存在。八类精确配置不能直接跑：拖船/海工船没有合格目标框，海工船合法源折只有3/4/3艘，不足每类8艘。四类协议是用户批准的范围调整，不是八类结果。

**没有发现可直接复用的 OSS1 源合法完整 A/D/T+M1/M2 权重或对应 native 多层缓存。** 已确认缓存如下：

- `E:/临时会话/same_vessel_gallery_20261019/causal_frequency_side_20261024/round23_public_oss1/native_features.npy`：1349x260，R50 stem 全局 mean/max 加几何；不含 native layer1/layer3 图，不能恢复 M1/M2 摘要。
- 同侧线 `round26_polarization_coverage/private_feature_changes.npy`：28104x2x128，诊断干预的特征变化，不是完整模型输入；source_readout/round25 的 ridge 归因亦不是当前完整上游模型。
- `E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/features.npz`：paired_r50 1349x2048、vh_ssl 1349x512；既有 resize/不同预处理，不可代替 native160 多层路径。
- `E:/临时会话/oss1_joint_geometry_20261017/*/local_features.npz`：源合法 R18 的局部摘要，骨干、输入和 token 定义不同，不可替代 R50 layer1/layer3。

最小接入范围如下，均为待主线程实现，不是已完成结果：

1. 用866行固定表适配原图/OBB，提取冻结 ImageNet R50 的 native stem、layer1、layer3 路径所需特征。复用 `E:/临时会话/sar_m123_standalone/sar_m123/{input,backbone,baseline,m12,tokens}.py` 的计算定义，不能加载24港监督 assets 后称 OSS1 复现。无需重训骨干。
2. 按 fit_train 拟合 A/D/T、layer1/layer3 类别 reader、尺度统计和 M1/M2；仅 fit_validation 选 checkpoint。具体参考入口为 `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/regional_utility_pilot/upstream/pipeline.py::{fit_base,fit_aux,fit_strategies}` 与同目录 `extract_source.py`。后者原实现不存多层图，按 reader/温度分阶段提取摘要；源reader变化后的摘要必须重算。
3. 输出按 stack_table 对齐的各 seed 基线与 M1/M2 分数、源合法摘要、有效性和局部特征。M1须保留均值支路，M2沿用冻结旧区域池化；四类 class 轴和基线分类头尺寸适配为4，不改变已冻结算子与超参。
4. M3 runner 接手 utility_train/selection、fit_train参考库，重训 OSS1 度量、极化头和 gate；评价前冻结全部选择。utility_train 用于训练统计，selection只作选择。上游不得用 utility_train/selection 标签或统计拟合 reader、温度或标准化。
5. 冻结后评价同一63条，报告三个 seed、逐类结果与适用范围，按既有要求回看原始VV/VH及实际参考。没有新模型结果前，不填写准确率。

## 8. 交付文件与重建

- `stack_table.csv`、`stack_indices.npz`、`STACK_SPLIT.json`：主入口、角色索引、隔离与协议记录。
- `stack_fit_train.csv`、`stack_fit_validation.csv`、`stack_utility_train.csv`、`stack_selection.csv`、`stack_evaluation.csv`：各角色明细。
- `stack_reference_candidates.csv`、`stack_coverage.csv`、`stack_pass_coverage.csv`、`stack_mmsi_exclusions.csv`：参考容量、覆盖和身份排除。
- `four_class_*`：父划分、原始筛选排除及原生视图追溯；不覆盖五角色协议。
- `build_four_class_split.py`、`build_stack_split.py`：仅元数据的可重建脚本。写入严格限本目录，无模型导入、GPU、训练或 Git 操作。

重建命令（顺序执行）：

```powershell
python -B E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/oss1_m123_replication_20261026/build_four_class_split.py
python -B E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/oss1_m123_replication_20261026/build_stack_split.py
```

已执行的检查只覆盖实际接入风险：连接键及原身份一致、各角色四类覆盖、sample/MMSI/pass隔离、每类源参考容量与非目标来源。若这些失败则不得发布角色表；本轮均通过。尚未进行原图到新特征的数值重放或模型验证。
