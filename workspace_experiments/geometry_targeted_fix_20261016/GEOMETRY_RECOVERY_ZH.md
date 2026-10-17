# 固定队列几何恢复结果

2026-10-16。已实际完成CPU恢复，不训练、不使用GPU、不下载、不解码整景、不修改旧文件。覆盖是元数据可用性，不是分类收益。

## 结论

| 固定缓存联合队列 | 恢复前 | 恢复后 | 新增 |
| --- | ---: | ---: | ---: |
| 样本 | 8622 | 8622 | 0 |
| 船轴有效 | 1145 | 1145 | 0 |
| 雷达方向有效 | 961 | 3477 | 2516 |
| 入射角有效 | 6448 | 6448 | 0 |

新增2516个雷达方向全部来自船轴无效样本，因此三项齐全覆盖没有增加。这修复了真实的字段依赖，但不会自动扩大仍坚持三项齐全的新C训练范围。完整38091行输出的有效数为船轴4640、雷达6744、入射角28628；固定队列外的原有列逐值保持不变。

## 实际恢复方式

- 限定为旧五港`geometry_relation_slots_20261016/slots_cache/*.npz`的`rows`并集，包含当前源fit/selection和目标查询、支持。角色覆盖只统计原S32任务，不调整样本或支持组合。
- 船轴复用`ship_axis_proxy.measure_axis/world_axis/combine_axes`，原128像素输入、Gaussian 0.8、细长比至少2、中心偏移至多0.1、边界拒绝、双极化差至多15度全部保留。旧缓存已拒绝的样本不反复重测。
- 固定队列4264个样本具备源VV/VH精确仿射，其中4263个已有轴提取缓存。唯一未计算者缺少必要target center，未用图像中心冒充目标中心，最终没有新接受船轴。
- 雷达独立读取205份本地同产品注释网格，复用`relative_radar_geometry.range_azimuth`及其网格范围/最近节点20km限制，再用原`geometry_metadata.image_axis`完成WGS84方位到源仿射、真实PNG宽高和等边resize坐标的映射。
- 雷达恢复不再要求船轴通过估计器。坐标来自精确源仿射和目标中心，或原入射角表允许的非AIS回退坐标来源；没有通过类别、预测结果或目标标签选择。
- 入射角使用同产品本地网格和原`join_chip_incidence.interpolate`的凸包内插值，不跨产品借角度。剩余2174个缺少允许的坐标，未新增。
- 船轴/雷达轴均为图像向右为0、向下为pi/2的无向轴，单位弧度、范围[0,pi)。雷达是地距方向代理，船轴是亮响应轴，均不冒充真实船首或精确波束方向。

## 剩余缺口

| 字段 | 状态 | 样本数 |
| --- | --- | ---: |
| 船轴 | 缺少精确源仿射，尚未计算 | 4358 |
| 船轴 | 旧缓存missing_or_weak_view，未通过成对接受 | 3075 |
| 船轴 | 旧双极化方向不一致拒绝 | 43 |
| 船轴 | 有源仿射但缺target center，尚未计算 | 1 |
| 雷达 | 缺精确源仿射，不能映射图像方向 | 4358 |
| 雷达 | 缺本地同产品完整方向网格 | 787 |
| 入射角 | 缺允许的坐标来源 | 2174 |

主要本地路径可访问，但上述4358行在当前manifest/精确georef映射中无法关联原始仿射，不能把地理角直接当像素角。787行缺的是带line/pixel的本地同产品方向网格；普通仅lat/lon/incidence网格不补造line/pixel。此次按授权范围停止，不无限搜索历史源景。

## 对训练和目标启用的影响

任务可启用条件沿用至少2个有效支持，并至少1个有效查询。以下是恢复表上的两种门控，不是已经重训的结果。

| 队列 | 三项齐全: 任务/不同查询 | 只要求船轴: 任务/不同查询 |
| --- | ---: | ---: |
| Jebel Ali target cross | 15/53 | 15/53 |
| Singapore target cross | 0/0 | 15/43 |
| Ningbo-Zhoushan target cross | 0/0 | 0/0 |
| Qingdao target cross | 0/0 | 0/0 |
| Shanghai target cross | 0/0 | 0/0 |
| Shanghai source fit cross | 0/0 | 8/16 |
| Qingdao source selection cross | 0/0 | 10/176 |

Singapore新增可启用的43个查询和81个有效支持来自允许雷达/入射角缺失，不是恢复了43个新船轴。这种门控变化在旧表上也成立，必须与2516个新雷达字段分开归因。Jebel Ali与Singapore源selection仍无可启用几何查询，不能拿这些集合单独选几何模块检查点。完整按池、阶段、角色和same/cross的统计见CSV。

## 接入注意

1. `recovered_geometry.csv`保留旧全部列，按原manifest排列，以`global_row`和`instance_id`核对；缺角度仍为NaN，valid为0。加载建议`low_memory=False`，新增provenance仅在限定队列填充。
2. 新增`ship/radar/incidence_recovery_status`与`*_provenance`。原有效字段及对应旧诊断列不修改。
3. 本次没有任何船轴变化，因此旧ship-slot缓存无需因这次元数据恢复重新GPU提取。雷达/入射角作为条件时读取新表即可。
4. 旧B基线必须继续使用其冻结的旧元数据。给旧B直接换新雷达输入会改变基线，不能与原结果混算；新增分支才用恢复表。
5. 三项齐全训练对照不会因本次恢复扩大；可选字段门控应作为另一个明确配置，不能把覆盖收益算成估计器精度收益。

## 文件与验证

- `recover_geometry.py`：实际执行完成，CPU-only，未发网络请求。
- `recovered_geometry.csv`：38091行，完整模型输入侧表。
- `recovered_geometry_coverage.json`：恢复计数和状态。
- `recovered_geometry_role_coverage.csv`：固定S32支持/查询角色覆盖。
- `recovered_geometry_task_coverage.csv`：旧/新表、三项齐全/船轴必需下的任务和独立样本覆盖。
- `recovered_geometry_port_coverage.csv`：限定8622个样本的逐港计数。
- `test_recover_geometry.py`：4项CPU测试通过，覆盖船轴无效仍可恢复雷达、方向单位/图像y符号、20km限制不放宽、旧有效值保护及可选字段门控。
- 完整恢复运行另验证全部旧有效字段保持不变、队列外原列保持不变、manifest行序/ID一致、有效角度合法且无效角度仍为NaN。

运行命令：

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' 'E:/临时会话/geometry_targeted_fix_20261016/recover_geometry.py'
```
