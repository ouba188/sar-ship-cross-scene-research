# 产品注释恢复结果

后续扩展已完成：完整 `manifest.csv` 现为 87 个成功产品；本报告保留最初 4 产品/787 行的记录。
新增 83 产品/4358 行的执行、零下载量和完整覆盖见 `EXTENDED_RECOVERY_ZH.md`。

2026-10-16。原始目标已完成：固定 8622 行队列中，实际状态为
`not_computed_missing_local_product_grid` 的 787 行全部取得同产品完整 VV annotation
及 line/pixel/latitude/longitude/incidenceAngle 网格。不是像素角度恢复完成；最终坐标转换由主线处理。

## 实际范围与成果

输入 `geometry_targeted_fix_20261016/recovered_geometry.csv` 实际有 38091 行。
程序读取 `geometry_relation_slots_20261016/slots_cache/*.npz` 的 rows 并集确认 8622 行，
按真实状态选择 787 行，并以 global_row 与 instance_id 核对原 manifest 后取得产品名。

| 精确产品 | 目标行 | 网格节点 | 原始 VV XML 字节 |
| --- | ---: | ---: | ---: |
| S1A_IW_GRDH_1SDV_20250315T182624_20250315T182649_058319_07359F_C0C6 | 29 | 210 | 1836529 |
| S1A_IW_GRDH_1SDV_20250809T224807_20250809T224832_060465_078475_18D1 | 354 | 210 | 1624760 |
| S1A_IW_GRDH_1SDV_20250901T112532_20250901T112557_060793_079162_B38D | 305 | 210 | 1701871 |
| S1C_IW_GRDH_1SDV_20250708T154757_20250708T154822_003131_006597_9D79 | 99 | 210 | 1824542 |

4 份完整 XML 总计 6987702 字节，均复用历史本地文件。全部 manifest 状态为
`recovered_local_exact_product`，无未恢复产品或注释缺字段。没有重新下载完整 XML、整景或 measurement。

## 找到的原因

历史脚本 `visual_reliable_baseline/download_safe841_metadata_20260917.py` 已从
`sentinel-s1-l1c` 取得原始 XML，保存于
`visual_reliable_baseline/artifacts/safe841_metadata_20260917/products/<product>.SAFE/annotation/iw-vv.xml`。

另一路 `relative_radar_geometry.fetch_grid` 在线取尾部并生成 radar_grid 缓存。
其历史 `E:/Hermes/paper_tracking/sensitivity/relative_observation_geometry_20261009/radar_fetch_status.csv`
对这四产品记录的是：C0C6 TLS EOF、18D1/B38D 读取超时、9D79 连接提前关闭。
因此本轮已证实的是：这些精确产品的原始注释实际存在，缺的是后来方向缓存的成功生成及本地复用。
不能将缺缓存解释为卫星产品没有方向网格。

## 来源核实与有限联网

内建 web.run 及搜索连接器调用没有返回可用内容。随后直接通过 HTTPS 读取官方文档，
保存原始 HTML，并向实际 annotation URL 发出匿名 Range 请求；所有请求有 20 秒 socket 超时，
读取阶段另有 35 秒截止检查与严格字节上限，无无限重试。未读取或使用任何账号凭证。

- AWS 官方数据登记页：`https://registry.opendata.aws/sentinel-1/`，HTTP 200，9575 字节。
  列出 `sentinel-s1-l1c` GRD 桶及运营方文档入口。存档 `source_checks/aws_registry.html`。
- Sentinel Hub 运营方目录文档：`https://roda.sentinel-hub.com/sentinel-s1-l1c/GRD/readme.html`，
  HTTP 200，4206 字节。说明年月日不补零、mode/polarization/原产品名目录组织，
  并注明 Requester Pays 及部分 metadata 的 HTTP 访问方式。存档 `source_checks/sentinel_hub_readme.html`。
- Copernicus 官方 OData 文档：`https://documentation.dataspace.copernicus.eu/APIs/OData.html`，
  HTTP 200，403052 字节。产品目录可通过 Products(UUID)/Nodes 逐级列出，
  Nodes 列表可匿名访问；产品下载部分要求授权 token。没有把匿名列目录等同于匿名下载所有文件。
  本次已有完整本地 XML，未启动 CDSE 账号登录或整包下载。存档 `source_checks/cdse_odata.html`。
- ASF 官方 API 文档：`https://docs.asf.alaska.edu/api/keywords/`，HTTP 200，80095 字节。
  Sentinel-1 产品类型中明确列出 METADATA_GRD_HD 等元数据类型，可作为后续其他产品的检索途径。
  本轮无需继续 ASF 元数据包/ZIP range 路径，未声称其下载认证已验证。

4 个精确 annotation URL 均来自原 `source_objects.json` 的对象键，写于 `manifest.csv.source_url`。
2026-10-16 11:12 UTC 对每个 URL 请求 `Range: bytes=0-4095`，全部 HTTP 206，
返回 Content-Range 总长度与本地完整 XML 长度一致，4096 字节前缀逐字节相同。
总 annotation 网络载荷仅 16384 字节；这证明这些对象当前可匿名读取及对应头部相同，
不是对整个远端文件重新下载比对的声明，也不推广为整库匿名访问承诺。
结构化收据见 `source_checks.json`；运营方 readme 是随后单独成功读取，存档另在 source_checks 目录。

## 验证与主线接入

对每份完整 XML 核对 missionId、GRD、VV、IW、起止时刻、绝对轨道号、十六进制产品 datatake
对应的十进制 missionDataTakeId；原产品目录及历史对象键保留产品完整尾缀。
同时确认 Ground Range、210 节点与 count 相等、line/pixel 不重复且非负、经纬度/入射角合法，
不同 line 中存在多个 pixel，可支持原有同 line 地距方向估计。

首次实现将 projection 放在 imageInformation 下读取，实际 XML 位于
generalAnnotation/productInformation；按真实结构修正后四产品全部通过。
随后实际调用现有 `relative_radar_geometry._validate`，四份输出全部通过，无训练或角度计算。

主线读取本目录 `manifest.csv`，按完整 product 精确关联，使用 `local_grid_path`。
JSON 文件名为 `<product>.json`，键保持现有消费者要求的 `status: ok`、`product`、`points`；
每个 point 含完整五字段。`local_annotation_path` 指向本目录原始 XML 副本。
`target_rows.csv` 提供原 787 行的 global_row / instance_id / product 映射。
不要将本目录 status 直接写作最终 radar_valid：源图像精确仿射、目标坐标、网格适用域和角度转换仍由主线判定。

所有新文件仅写入本 `product_annotations` 目录。旧表、旧缓存及主线 georef 文件未改动。
主线新增 georef 后产生的更多缺网格产品尚不在本次四产品 manifest；等待其明确清单后再扩展。

## 改动文件

- `recover_product_annotations.py`：固定原目标的可重跑恢复脚本。
- `check_public_sources.py`：有限匿名来源核实脚本。
- `manifest.csv`、`target_rows.csv`、`summary.json`：主线导入入口与覆盖记录。
- `annotations/*.xml`：4 份完整原始 VV 注释副本。
- `radar_grid/*.json`：4 份消费者兼容的真实五字段网格。
- `source_checks.json`、`source_checks/*`：7 次首批请求收据、4 份小前缀及4份官方网页存档。
- `MAIN_HANDOFF.md`、`RECOVERY_REPORT_ZH.md`：主线交接与本报告。

运行环境沿用 `D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe`，未安装依赖。
