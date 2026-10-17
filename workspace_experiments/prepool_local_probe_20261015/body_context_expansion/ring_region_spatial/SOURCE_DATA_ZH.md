# “拉伸前 SAR 数据”与当前同样本的源数据可用性

核查日期：2026-10-15。只读本地清单、历史索引、指定文件路径及少量文件头；唯一新增文件为本文。未运行实验、下载数据、访问远端、扫描全盘或解码整景栅格。

## 结论

当前可确认的是：4,289 个需求样本均对应 **GRDH 产品派生的 PNG / 8bit 切片链路**，当前输入缓存也确为 uint8。历史索引提供了一部分同产品 ZIP 和 `int16` 目录 TIFF 的上游线索，但检查的登记路径当前均不存在；**本次没有确认任何可直接读取的、与这批样本配准对应的拉伸前高位深栅格、定标 sigma0 或 SLC 复数数据**。

这不等于“原数据彻底没有了”：远端 server2 的现存文件、未检查的归档副本和未登记迁移位置仍未知。本次结果不能外推成全盘或全部服务器的不存在证明。

## “拉伸前”具体指什么

这里指生成显示用 8 位图像之前的数值影像。必须进一步说明其数值语义，不能只说“原始 SAR”：

| 层级 | 含义 | 本次证据 |
| --- | --- | --- |
| PNG / 8bit UTM tile | 显示拉伸、量化后的图像；通常只保留 0–255 显示值 | 当前样本、源路径命名及缓存支持此层级；8 张实测 PNG 文件头为 8bit 灰度 |
| 较高位深 measurement DN / 未拉伸栅格 | 产品测量数值或其重投影结果，可能仍未辐射定标 | 历史 ZIP / `int16` TIFF 路径是候选线索；没有可访问文件头确认其实际 dtype、振幅/功率语义或处理历史 |
| 定标强度 / sigma0 | 有定标步骤、系数和单位依据的后向散射量；还需区分线性值与 dB | 当前可用文件未建立这条处理证据链；不能把 DN、DN 平方或 float 数组自动称为 sigma0 |
| SLC complex | 已聚焦的复数影像，保留 I/Q 或等价幅相信息 | 需求表的 30 个产品均为 GRDH，未发现已链接的同次观测 SLC 复数源；不能从现有 GRDH / PNG 恢复相位 |
| 原始回波 raw echoes | 成像聚焦前的传感器回波 | 不是本任务所说的“拉伸前数据”，本次也未查此类数据 |

“拉伸前”不要求回到原始回波，也不必是 SLC；经过地理编码但尚未显示拉伸的数值栅格也可能符合需求。**float dtype 不证明已定标，`int16` 目录名不证明实际为 16 位 DN，`UTM` 只说明几何产品命名，`source_tiff` 只表示相对下游 PNG 的来源。** 8bit 转 float 或除以 255 不能恢复量化/截断前的测量值。具体拉伸公式、阈值及其是否按景/极化变化，本次未取得。

## 样本范围与已确认输入

- 需求表：`E:/临时会话/prepool_local_probe_20261015/body_context_expansion/required_samples.csv`。
- 主清单：`E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv`。
- 4,289 个唯一需求 ID 全部匹配；逐需求行按 `row_index` 取主清单，`instance_id == sample_id` 且 product 相等，失配为 0。涉及 30 个唯一产品，全部 `_IW_GRDH_1SDV_`。
- legacy 共 2,478 条：Jebel Ali 92、Ningbo-Zhoushan 958、Qingdao 552、Shanghai 813、Singapore 63。其 `source_tiff/source_json` 为空，但全部 `tile_base` 含 `UTM_8bit`。
- returned_20260915 共 1,811 条：Jebel Ali 776、Singapore 1,035。VV/VH 的 `source_tiff` 和 `source_json` 均有值；全部源 TIFF 路径含 `UTM_8bit`，位于服务器的 `int8_grid_4x4_o256` 链路。路径存在于清单不代表服务器文件当前仍存在；本次未连接服务器。
- 4,289 条都有 VV/VH PNG 路径；只对下表 4 个样本的 8 张 PNG 检查了现存文件头，没有把抽查写成全量文件读取结果。
- `D:/SAR_completion_cache_20261010/source_inputs128.npy`：只读 NPY 头，版本 1.0，shape `(38091, 2, 128, 128)`，dtype `uint8`，C order。该 shape 是整份缓存规模，不是需求样本数。
- 当前读取逻辑 `E:/临时会话/prepool_local_probe_20261015/extract_ring_conditions.py` 将缓存像素转 float64 再除以 255；这里的 `raw_box/ring_*` 指显示像素统计，不是定标前测量 DN 或物理海况。既有口径见 `E:/临时会话/prepool_local_probe_20261015/body_context_expansion/ring_conditions/CONDITION_SOURCES.md`。

## 实际样本追溯

下表各 PNG 均存在，文件头为 PNG、mode L、bit depth 8、color type 0。尺寸为宽×高；没有解码像素或整景。

| 需求样本 ID | 产品末段 / 切片 | VV / VH 尺寸 | 血缘结论 |
| --- | --- | --- | --- |
| returned_local_knowledge_0002713 | 060963_07982C_CF20 / r03_c01 | 64×64 / 64×64 | 本地回传 PNG → server2 8bit tile；原 tile / JSON 未远端验证 |
| returned_20260915_0000415 | 060963_07982C_CF20 / r02_c02 | 64×64 / 64×64 | 同产品、不同切片的本地 merged 回传 PNG；不是额外高位深源 |
| pair_0032730 | 059918_07715F_4982 / r02_c03 | 72×72 / 72×72 | legacy PNG；历史索引只给出未核验远端 ZIP 线索 |
| pair_0032742 | 062543_07D685_F009 / r02_c04 | 72×72 / 80×80 | legacy PNG；历史本地 ZIP / int16 TIFF 登记路径当前不存在 |

### returned_local_knowledge_0002713

产品全名：`S1A_IW_GRDH_1SDV_20250913T021543_20250913T021606_060963_07982C_CF20`；MMSI `572345220`，offshore_supply，server2。

实际检查的本地 PNG：

```text
E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/images/Jebel Ali/offshore_supply/S1A_IW_GRDH_1SDV_20250913T021543_20250913T021606_060963_07982C_CF20_VV_UTM_8bit__r03_c01__e0a4dfe1e0da609e.png
E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/images/Jebel Ali/offshore_supply/S1A_IW_GRDH_1SDV_20250913T021543_20250913T021606_060963_07982C_CF20_VH_UTM_8bit__r03_c01__0146f6bc50a6eb5a.png
```

本地血缘表：`E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata/server2/manifest.csv`。通过 obj_uid `e0a4dfe1e0da609e` / `0146f6bc50a6eb5a` 匹配到 VV/VH，product、MMSI 一致，source_object_index 均为 12。记录的 tile 内裁剪框为 VV `[3233,1795,3297,1859]`、VH `[3232,1796,3296,1860]`，均 64×64；这些是 tile 坐标，不可直接用作原始 measurement 的行列坐标。

该表记录的服务器源 TIFF / JSON 精确路径：

```text
/root/autodl-tmp/asf_download_v2_tmp/v5/geocode/Jebel Ali/utm_out/VV/int8_grid_4x4_o256/S1A_IW_GRDH_1SDV_20250913T021543_20250913T021606_060963_07982C_CF20_VV_UTM_8bit__r03_c01.tif
/root/autodl-tmp/asf_download_v2_tmp/v5/geocode/Jebel Ali/utm_out/VH/int8_grid_4x4_o256/S1A_IW_GRDH_1SDV_20250913T021543_20250913T021606_060963_07982C_CF20_VH_UTM_8bit__r03_c01.tif
/root/autodl-tmp/asf_download_v2_tmp/v5/geocode/Jebel Ali/utm_out/VV/int8_grid_4x4_o256/labels_json/S1A_IW_GRDH_1SDV_20250913T021543_20250913T021606_060963_07982C_CF20_VV_UTM_8bit__r03_c01.json
/root/autodl-tmp/asf_download_v2_tmp/v5/geocode/Jebel Ali/utm_out/VH/int8_grid_4x4_o256/labels_json/S1A_IW_GRDH_1SDV_20250913T021543_20250913T021606_060963_07982C_CF20_VH_UTM_8bit__r03_c01.json
```

这些路径来自本地 manifest，未读取服务器 TIFF/JSON 本体。该产品未命中本次读取的 7 月 product_status 索引；这不能证明没有原产品。

### 两个 legacy 样本

实际检查的 VV PNG 精确路径如下；VH 使用主清单独立路径检查，尺寸已列上表：

```text
E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_merged_20260915/images/Ningbo-Zhoushan/bulk_carrier/S1A_IW_GRDH_1SDV_20250703T095436_20250703T095501_059918_07715F_4982_VV_UTM_8bit__r02_c03__df04a517b1b92b68.png
E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_merged_20260915/images/Ningbo-Zhoushan/bulk_carrier/S1A_IW_GRDH_1SDV_20251230T095431_20251230T095456_062543_07D685_F009_VV_UTM_8bit__r02_c04__a61c56521a9e2f79.png
```

`pair_0032730` 的 VV/VH tile 内裁剪起点分别为 `(5811,1920)` / `(5812,1922)`，边长均 72。历史索引记录的 ZIP 为 `http://10.128.2.137:18082/Ningbo-Zhoushan/171_91/S1A_IW_GRDH_1SDV_20250703T095436_20250703T095501_059918_07715F_4982.zip`，状态 `remote_zip_only_unverified`；未请求此地址。

`pair_0032742` 的 VV/VH 裁剪起点分别为 `(196,778)` / `(191,770)`，边长 72 / 80。历史登记的以下路径本次均未找到：

```text
F:/SAR_AIS_T/Ningbo-Zhoushan/171_91/S1A_IW_GRDH_1SDV_20251230T095431_20251230T095456_062543_07D685_F009.zip
E:/SAR_AIS_T_zip_without_8bit_20260703_103258/F_SAR_AIS_T/Ningbo-Zhoushan/171_91/S1A_IW_GRDH_1SDV_20251230T095431_20251230T095456_062543_07D685_F009.zip
E:/SAR_AIS_T_zip_without_8bit_20260703_103258/F_SAR_AIS_T/Ningbo-Zhoushan/171_91/utm_out/VH/int16/S1A_IW_GRDH_1SDV_20251230T095431_20251230T095456_062543_07D685_F009_VH_UTM.tif
E:/SAR_AIS_T_zip_without_8bit_20260703_103258/F_SAR_AIS_T/Ningbo-Zhoushan/171_91/utm_out/VH/int16/S1A_IW_GRDH_1SDV_20251230T095431_20251230T095456_062543_07D685_F009_VH_UTM.tif.aux.xml
```

## 归档索引检查边界

读取了以下既有索引，按需求表的完整 product 名匹配；没有重新运行它们的扫描脚本：

- `E:/临时会话/sar_asset_audit_20260704_182055/product_status.csv`
- `E:/临时会话/sar_asset_audit_20260704_182055/zip_checks.csv`
- `E:/临时会话/five_drive_sar_audit_20260804/physical_local_v2/physical_tiff_inventory.csv`
- `E:/临时会话/five_drive_sar_audit_20260804/local_full_def/all_eligible_tiff_copies.csv`

product_status 命中 20/30 个产品；其历史状态为 `ok_int8_present` 10、`remote_zip_only_unverified` 8、`bad_or_incomplete_only` 1、`missing_zip_and_int8` 1。**这些是历史标签，不是当前可用性。** 其中 `sample_assets` 将 `int16/*_UTM.tif` 也标成 `int8`，故不能按汇总标签判 dtype。

zip_checks 中与需求产品匹配的 11 条本地 ZIP 路径，当前 Test-Path 均为 false。product_status 中提及的 5 个 `int16` VH TIFF 及 5 个 `.aux.xml` 精确路径也均为 false，无法读取 TIFF 头或定标元数据。5 个候选产品末段为 `061143_079F62_97AE`、`061318_07A66F_1889`、`062018_07C214_78ED`、`062543_07D685_F009`、`062543_07D685_8E60`，关联本次 645 个需求样本；645 只是同产品线索覆盖，**不是可用拉伸前样本数**。

两个 8 月 TIFF 索引各命中同一组 30 条路径、14 个产品，均为 `UTM_8bit` 命名，登记路径当前也均未找到。旧路径不可用的原因可能是迁移、删除或存储未挂载，本次未判定。

## 对“同样本有没有”的准确回答

已经有同样本的 8bit 图像及部分明确的裁剪血缘；目前没有证实可用的拉伸前同样本数值数据。若后续取得同产品 measurement 或未拉伸 UTM 栅格，需要结合原有重投影、切片偏移和裁剪信息建立同观测、同位置对应；仅有相同 MMSI 或 product 名还不能宣称像素级对齐。若要把值称为 sigma0，还需实际的定标记录或可执行定标所需的产品系数，不能以 float 类型替代证据。本次未扩展这项工作。
