# OSS2 四类 M3 数据协议

2026-10-26。数据支线完成；未训练分类器、未操作 Git。主代理拥有 `run.py`、训练与分析脚本。本文件只说明数据准备，不宣称四组 M3 的结果。

## 可直接使用的接口

本目录绝对路径：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/oss2_m123_replication_20261026/`。

| 文件 | 用途 |
|---|---|
| `stack_table.csv` | 固定 1,519 行的完整训练/验证/选择/评价表；主代理已开始提取，禁止重排行序 |
| `view_inventory.csv` | 全部 2,202 个四类 GRDH 样本、4,404 路输入；以 `(obs_id, pol)` 唯一连接 |
| `geometry.csv` | 以 `sample_id` 连接雷达轴；全部 `radar_axis_valid=False`，`radar_axis_rad` 为空 |
| `data_inventory.csv` | 检测前后完整四类清单，包括无框行、可运行性和排除原因 |
| `detection_views.json` | 本轮冻结检测器输出、原生坐标框与关联结果；不是历史已存在缓存 |
| `STACK_SPLIT.json` | 固定划分、隔离检查、每角色计数和参考船预算 |
| `DATA_INTERFACE.json` | 绝对路径、标准 schema 和跨数据集输入路径 |

标准列：

- `class_id` 与 `y` 均为新的 0..3：0 bulk_carrier、1 general_cargo、2 product_chemical_tanker、3 container_ship；`original_class_id` 分别为 0、2、3、4。
- `sample_id` 是持久样本键。`obs_id=manifest_position` 是原 3,745 行 OSS2 target_manifest 的位置，不能用 GRDH 紧凑索引代替。`target_manifest_row` 是更早清单的行号，含义也不同。
- `legacy_row_index` 保留旧 GRDH 2,746 行 cohort 的索引。`table_index=row_index=up_index` 是当前 stack 的连续 0..1518；只用于本表顺序。
- `pass_group=pass_id=overpass`，格式为平台、绝对轨道、datatake，例如 `S1A_010568_00FB64`。不把产品片段当独立过境。
- `mmsi`、`mmsi_or_ship_id` 应按字符串读取；本次四类清单没有同 MMSI 跨类别冲突。
- `role` 为 `fit_train / fit_validation / utility_train / selection / evaluation`。`reference_candidate` 仅 fit_train 为真。
- `valid_vv/vh` 表示原 prepare 后的有效 target；`obb_vv/vh` 表示通过 conf>=0.25 的检测。双路无框不进入 stack，单路有效保留。
- `input_scale=1.0`、`canvas_size=160`，保持 native 像元，无缩放。

视图表兼容 OSS1 prepare_inventory 所需字段：`obs_id,sample_id,pol,path,exists,detected,confidence,raw_xywhr,axis_valid,cx,cy,length,width,ship_axis,native_w,native_h`。`raw_xywhr` 是 JSON 数组字符串，保留检测器原 width/height/angle 的耦合；长轴角另存于 ship_axis，不混用。检测不到时 raw_xywhr 为 `null`，不构造替代框。

## 来源与固定划分

源 cohort、membership 和 pass_assignment 均来自 `E:/临时会话/scene_aware_classification_20261018/oss2_distribution_transfer_20261023/`。沿用旧 27 个 GRDH 过境、seed=20261023 的六包，固定 fold0，不读取准确率、不重抽。

| 角色 | 原 pass_bag | 观测数 | 独立 MMSI | 过境数 |
|---|---|---:|---:|---:|
| fit_train | 2、3 | 478 | 448 | 9 |
| fit_validation | 4 | 228 | 223 | 4 |
| utility_train | 5 | 241 | 235 | 4 |
| selection | 1 | 194 | 193 | 5 |
| evaluation | 0 | 378 | 369 | 5 |
| 合计 | 全部 | 1519 | 1468 | 27 |

先沿用旧 fold0 membership 的 train/validation/test 和原 MMSI 排除；原 validation 对应 selection。合法 train 内再按 bag2/3、bag4、bag5 分配角色，MMSI 优先级为 utility_train、fit_validation、fit_train。保留整个候选 held bag 的身份预留，检测是否成功不改变身份归属。角色间 sample_id、MMSI、pass_group 两两交集全部为 0，见 STACK_SPLIT.json 的 10 组检查。

每角色都覆盖四类；evaluation 的 5 个过境并非各自都有四类，不伪造缺类样本。`stack_pass_coverage.csv` 提供实际过境/类别分母。

| 新类别 | fit观测/船 | validation观测 | utility观测 | selection观测 | evaluation观测 |
|---|---:|---:|---:|---:|---:|
| 0 bulk | 138 / 135 | 87 | 83 | 64 | 132 |
| 1 general | 47 / 45 | 23 | 26 | 18 | 26 |
| 2 chemical | 95 / 91 | 29 | 46 | 40 | 83 |
| 3 container | 198 / 177 | 89 | 86 | 72 | 137 |

参考候选只取 fit_train，同 MMSI 保留第一条，得到 135/45/91/177 艘，足够每类 8 艘、合计 32 艘。该支线没有选择最终参考库，不填充重复参考；主程序沿用 OSS1 源内 KMeans(8, random_state=17, n_init=10)。

## 检测与原图可运行性

未找到可复用的 OSS2 全队列 OBB 缓存，因此本轮用 `E:/Docms/SARlabelimg/550k.pt` 新执行冻结检测，共 4,404 路，检测约 92.7 秒。规则与 OSS1 `E:/临时会话/oss1_joint_geometry_20261017/detect_axis.py` 相同：原生 uint8 居中补零至 1024、conf=0.08/iou=0.3；中心距离不超过 max(8, 0.3*min(h,w))，中心位于图内，长边>=4、短边>=1.5、长边<=1.5*max(h,w)，取合法候选最高置信度框；下游阈值 conf>=0.25。长短轴交换同时调整角度；长宽比>=1.2 时标记 axis_valid。

这只是冻结公共定位器，不是新训练。其预训练样本来源未审计，不能声称从未见过 OSS2。雷达轴没有可靠来源，全部设为无效，不使用 AIS heading 或 XML head/tail 冒充雷达轴。

原生输入为既有公共 PNG；manifest 的 `source_path_vv/vh` 保留原 `OpenSARShip_2/.../Patch_Uint8/Visual_*.tif` 来源。PNG notes 记录 native_patch_uint8=1、resize=0。全量检查 PNG 实际存在、文件头大小与 cohort 一致；本轮不重新校准或改写像素。

数据准备前直接查看了四类按 sample_id 第一条的 VV/VH 原图：`000016` bulk 77x77、`000011` general 49x49、`000003` chemical 73x73、`000004` container 113x113（完整 ID 均以 `pub2_opensarship2_` 开头）。可见事实是两路均有近中心细长/断续亮响应，各例大小和散射分布不同；不据此推断物理成因或全体比例。它们后来均获得双路有效框；主代理另做覆盖性输入回看。

| 新类别 | 检测前观测 | 至少一路框 | 双路框 | 无框 |
|---|---:|---:|---:|---:|
| 0 | 590 | 565 | 539 | 25 |
| 1 | 231 | 154 | 124 | 77 |
| 2 | 611 | 368 | 313 | 243 |
| 3 | 770 | 752 | 723 | 18 |
| 合计 | 2202 | 1839 | 1699 | 363 |

四类检测前为 1,738 艘船，检测后 1,504 艘船。无框排除类别相关，当前下游结论只能覆盖“检测后四类分类”，不能报作原始 OSS2 全队列分类性能。`inventory_exclusions.csv` 完整保留 363 行；stack 排除中无框主原因仅 280 行，因为其他 83 行首先被 MMSI 隔离排除。不要把 280 误写成全队列无框数。

`stack_exclusions.csv` 共 683 行，互斥主原因：旧 excluded_train_mmsi 236、旧 excluded_val_test_mmsi 49、预留 utility 身份 94、预留 validation 身份 24、剩余双路无框 280。

全量 4,404 路执行原 `sar_m123.input.prepare_view`，等价于 prepare_inventory 构造 spec 后的实际图像准备，未加载分类骨干。1,839 条至少一路 valid_target；没有接受框却无目标像素的情形；没有 prepare 异常或目标跨度>160，`prepare_protocol_blockers.csv` 为空。未来若真实目标 mask 跨度>160，脚本停止并报告协议阻塞，不自动删除、换框或缩放。

## 大图裁剪的实际影响

19 条 native 窗口大于 160，全部保留在 data_inventory，38 路均有有效框。实际 prepare 是以目标 mask 范围居中且受原图边界约束的 crop/pad，不是固定的原图中心裁剪；最大 native 205x205。

19 条检测目标 mask 像素全部保留；原窗口面积保留率为 0.609161..0.987616，框外上下文有损失。已有 XML 估计 head/tail 坐标在 38 路实际裁窗内，但这些坐标沿用中心减半窗的估计原点，不能当作船体完整性的独立证明，也不能据此保证所有分类证据无损。

stack 内含其中 16 条，其余 3 条因 MMSI 隔离排除，非因图大。实际角色：fit 3、validation 6、utility 5、selection 1、evaluation 1。逐路 crop_x/y、crop_width/height、pad_left/top、目标与上下文像素保留量见 `crop160_impact.csv`。旧 224 读出不在本轮协议内。

## 跨数据集全合法目标队列

两个方向都用新队列全部可用目标，既包括其原 evaluation，也包括原 fit/validation/utility/selection 或被该内部划分隔离的行。跨数据集时必须冻结来源数据集已经训练好的全部模型、校准与参考库，不在目标端重新 fit 或选参数。

排除规则：目标 MMSI 或 pass_group 与来源模型任一 fit_train、fit_validation、utility_train、selection 行相交即排除。来源内部 evaluation 不属于拟合/选择集。保留历史公开数据开发暴露说明，这不是全新盲测。

| 方向 | 合法候选 | 源交集排除 | 最终观测 | 独立船 | 过境 |
|---|---:|---:|---:|---:|---:|
| OSS1→OSS2 | 1839 | 220 | 1619 | 1347 | 27 |
| OSS2→OSS1 | 1222 | 301 | 921 | 742 | 21 |

OSS1→OSS2 的源四角色为 803 条、766 艘；220 条均因源 MMSI，无源 pass 交集。OSS2→OSS1 的源四角色为 1,141 条、1,099 艘；204 条有 MMSI 交集，133 条有 pass 交集，两者并集 301（交集 36）。具体每行原因见 `cross_dataset_source_exclusions.csv`。

- OSS1→OSS2 表：本目录 `cross_oss1_to_oss2_table.csv`；视图与雷达表使用本目录 `view_inventory.csv`、`geometry.csv`。1,619 条全部属于已经全量 prepare 检查通过的 OSS2 清单。
- OSS2→OSS1 表：本目录 `cross_oss2_to_oss1_table.csv`；视图使用邻目录 `oss1_m123_replication_20261026/view_inventory.csv`；雷达使用 `E:/临时会话/oss1_joint_geometry_20261017/geometry.csv`。921 条额外执行 1,842 路原 prepare_view 检查，全部可运行、没有目标 mask 像素裁失。
- 两表均有标准 `class_id/y/pass_group/pass_id/obs_id/sample_id/mmsi/role`，且所有行 `role=evaluation`。它们只用于冻结模型迁移推理，不能当作有源训练角色的 stack 调用训练。
- `obs_id` 只在各自数据集内唯一；跨数据集禁止单用 obs_id 合并缓存，需按数据集与 sample_id 区分。
- `CROSS_DATASET_PROTOCOL.json` 保存方向计数；`cross_*_coverage.csv` 保存类别分母。

清单层面的重叠：OSS1 原 1,349 行与 OSS2 原 3,745 行共享 419 个 MMSI；只看 OSS2 GRDH 八类时共享 274 个；本轮四类 GRDH 时共享 265 个。三者均共享一个 pass `S1A_010568_00FB64`。`cross_dataset_overlap.csv` 的明确范围是 OSS1 八类全清单对 OSS2 本轮四类 GRDH，不能把其中 265 误称为完整两库的全部共享身份。

## 复现与冻结状态

准备代码为 `build_inventory.py` 和 `build_stack_split.py`，运行环境为 `D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe -X utf8 -B`。首次 inventory 加 `--detect --batch-size 8`；已有 detection_views.json 时可复用检测，不重复推理。随后运行 build_stack_split.py。脚本不读取历史预测或准确率。

主代理已开始 raw 提取，当前 stack/table/detection 行序冻结；后续复现须使用独立输出目录或明确清理相应缓存，不能在已有提取过程中重建输入。数据支线没有修改训练器、run.py、Git 或原图。冻结检测器初始化时 Ultralytics 自动报告了用户 settings.json 的 schema 更新；本支线没有主动改动系统配置或安装依赖。
