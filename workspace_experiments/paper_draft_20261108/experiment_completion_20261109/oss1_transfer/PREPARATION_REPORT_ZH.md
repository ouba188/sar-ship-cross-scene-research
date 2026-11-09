# OSS2 当前模型到 OSS1 冻结外推：数据准备

状态：`ready_for_frozen_raw_inference`。仅 CPU 原图预处理，无 GPU、训练、模型前向、权重读取、下载或 Git 操作。

## 可执行入口

- 目标清单：`E:\临时会话\paper_draft_20261108\experiment_completion_20261109\oss1_transfer\target_manifest.json`。保留 G 组全部 921 条及原查询顺序。
- 固定参考：`E:\临时会话\paper_draft_20261108\experiment_completion_20261109\oss1_transfer\references.json`；32 条、每类 8 条，顺序与当前模型及旧 reference_bank 一致。
- 目标记录含 id、mmsi、pass_group、label、scene_id、vv/vh、image、OBB corners、target_index、ship_axis 和 radar_axis_rad。
- JSON 为列表，可直接交给 task.data.read_manifest；数据准备不调用 raw_current.py。
- obs_id 在 OSS1/OSS2 内各自编号；跨库按 MMSI 和平台绝对轨道比对，不能直接比较局部 obs_id。
- indices.query_row 为 0..920；legacy_combined_row 仅保留历史溯源，不沿用旧缓存索引。

## 源暴露与隔离

- source719 = source_fit478 + source_adapt241。当前 class_s17/29/43 的实际 orders 与上游 small_class_s17 orders 均覆盖同一719条。
- M3 aggregate_17/last.pt 的 episodes 已逐项核对查询与动态参考；动态参考属于719源池，不能把训练暴露缩成固定32参考。
- current groups、m1_coverage manifest、公开管线六组 manifests 逐条一致；固定参考全部来自 source_fit。
- 平台绝对轨道从 scene_id 解析，并与 pass_group 前两段交叉核对；数据段后缀不参与轨道隔离。
- 当前源拟合基础为 final_oss2 full_s17 fit478/adapt241；aggregate M3、后续M1和末端M2沿此源池继续训练。初始化重放由主代理处理，此处不加载权重。

| 比对组 | 源观测 | 目标同MMSI观测 | 目标同平台绝对轨道观测 |
|---|---:|---:|---:|
| small | 719 | 0 | 0 |
| references | 32 | 0 | 0 |
| validation | 228 | 0 | 0 |
| selection | 194 | 0 | 0 |
| evaluation | 378 | 71 | 0 |
| train_and_development_947 | 947 | 0 | 0 |
| historical_four_roles_1141 | 1141 | 0 | 0 |

OSS2历史evaluation与本目标有 71 条/48 船同MMSI关联。它不是上述719训练暴露，但两套评价不能称完全独立船只证据。
辅助清单 target_manifest_no_oss2_eval_overlap.json 保留 850 条，供需要时做独立船只敏感性分析；不替换正式921队列。
921本身已有历史OSS1迁移结果，不称新盲测。validation为开发暴露；selection为历史候选/诊断暴露，不能等同当前选定分支的梯度训练。

## 类别与原图

共 921 观测、742 船、21 平台绝对轨道（旧pass_group为21）。
类别映射保持旧八类 [0,2,3,4] 到新四类 [0,1,2,3]。

| 新类ID | 类别 | 数量 | 双可用极化 | 仅VV | 仅VH | 输入未就绪 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | bulk_carrier | 675 | 650 | 7 | 18 | 0 |
| 1 | general_cargo | 71 | 59 | 8 | 4 | 0 |
| 2 | product_chemical_tanker | 72 | 68 | 2 | 2 | 0 |
| 3 | container_ship | 103 | 98 | 3 | 2 | 0 |

- 逐张使用已有 sar_m123.input.prepare_view：原始显示TIFF、OBB角点及native160裁剪/补零，不缩放、不重做检测、不转换原始SAR幅度。
- target_views.csv 保存全部VV/VH库存字段、OBB原始xywhr、置信度、轴、真实图像尺寸、prepare后几何与placement及错误。
- 未配置/无有效OBB的极化不等于原始文件缺失；文件存在性与模型可用性分别报告。
- 原始Patch、Patch_Cal、Ship.xml和Metedata.xml路径及当前存在性保留在target_manifest.csv；不假设多波段原始SAR的极化通道顺序。

缺失/不一致统计：

```json
{
  "target_not_ready": 0,
  "strict_excluded": 0,
  "references_not_ready": 0,
  "target_vv_missing": 0,
  "target_vh_missing": 0,
  "target_views_prepare_error": 0,
  "availability_changed": 0,
  "raw_patch_missing": 0,
  "calibrated_patch_missing": 0,
  "ship_xml_missing": 0,
  "metadata_xml_missing": 0
}
```

## 交付与复跑

- prepare_transfer.py：重建全部清单、实际暴露统计和此报告；逐张读取小chip后释放，不打开任何NPZ。
- target_manifest.json / references.json / source719_manifest.json：兼容当前公共数据reader。
- target_manifest.csv / target_views.csv / reference_views.csv：路径、OBB、极化及CPU预处理溯源。
- source_exposure.csv / source719.csv：1519角色全表及源719，记录实际每阶段暴露次数。
- overlap_details.csv / class_counts.csv / preparation_summary.json：重叠明细、类别统计和机器可读结果。

```powershell
& 'D:\Program_documents\Anaconda_envs\envs\ClearSAR\python.exe' -B 'E:\临时会话\paper_draft_20261108\experiment_completion_20261109\oss1_transfer\prepare_transfer.py'
```

本轮环境记录：默认Anaconda Python在首张TIFF像素解码时无traceback退出，原因未确认；改用项目已有ClearSAR环境后全部完成。无需安装或下载，复跑采用上面的ClearSAR解释器。

## 最小剩余工作

921条输入与32参考的数据入口已就绪，没有阻断原图准备的缺失。后续由主代理将target_manifest.json交给已验证的raw_current.py执行冻结外推。此处不声称模型输出已生成，也不替代主代理的原图前向与共同初始化核对。
