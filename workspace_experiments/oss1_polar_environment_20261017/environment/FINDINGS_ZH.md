# OSS1 环境实际下载与匹配结果

2026-10-17。已完成 22 次采集、1,349 个样本的实际匹配，未训练模型，推理保持 SAR-only。所有本任务脚本、数据缓存、表格和报告均在本目录。

## 结果

| 变量 | 实际有效样本 | 缺测 |
|---|---:|---:|
| ERA5 10 m 风 u/v，各项 | 1,349 / 1,349 | 0 |
| ERA5 有效波高、平均波周期、平均波向，各项 | 1,293 / 1,349 | 56 |
| GLORYS12 表层 u/v、位温、盐度，各项 | 1,347 / 1,349 | 2 |
| 9 项同时有效 | 1,292 / 1,349（95.77%） | 57 |

22 景均成功访问。累计下载并缓存 **302,845,282 字节（288.82 MiB）**：ERA5 259,501,527，CMEMS 43,295,552，官方 STAC 48,203。其中环境变量数据块 302,565,243 字节，其余为坐标和元数据。扩展全 22 景阶段新增 284,370,966 字节，小于 320 MiB 运行上限和本轮设置的 512 MiB 工作上限。统计不包含安装包、网页检索和初期未保存的元数据探测流量；初期没有下载环境数值。最终离线重建和验证新增网络流量均为 0。

## 来源与匹配

- 未发现标准 CDS/CMEMS 配置、环境变量或 CMEMS 标准 netrc 主机凭据；`cdsapi.Client` 实际配置检查失败，ECMWF 官方 ARCO 波浪接口返回 HTTP 401。没有索取或输出秘密。公共路径已解决本轮实际读取，不再需要用户提供凭据。
- ERA5 使用 **Google Research ARCO ERA5 公共副本**，`full_37-1h-0p25deg-chunk-1.zarr-v3`，对应 CDS `reanalysis-era5-single-levels`，并非 ERA5-Land 或其他模式替代品。归档名称 v3 与内部 Zarr 格式版本 2 是不同概念。
- 公共副本通过 CDS 请求把所有变量放到 0.25 度网格；**波浪也已重采样到 0.25 度**，不能称其具有独立的 0.25 度物理分辨率，也不能冒充标准 CDS 的 0.5 度波浪交付网格。本轮保留该真实来源和限制。
- CMEMS 使用官方 STAC 明确给出的 `geoChunked` 地址，产品 `GLOBAL_MULTIYEAR_PHY_001_030`，数据集 `cmems_mod_glo_phy_my_0.083deg_P1D-m_202311`，GLORYS12V1，1/12 度规则交付网格。四变量同一最浅层，实际深度 **0.49402499198913574 m**；温度字段是 `thetao` 位温（摄氏度），盐度是 `so`（源单位 `1e-3` / Practical Salinity Unit）。
- 使用每船真实经纬度和 UTC 时间，空间采用交付网格最近点，不跨岸找最近有效海点，不做空间或时间插值。ERA5 最近整点小时保留真实时间差；GLORYS12 按 SAR 所属日期取日平均，不把 22 UTC 舍入成次日。
- GLORYS12 ARCO 坐标标签实际为当天 00:00；官方 PUM 1.7 第 9 页明确日平均支持范围为午夜至午夜、中心在中午。表格同时保存原始标签、当日 `[00:00, 次日00:00)` 和中心 12:00。PUM 第 8 页说明**不包含潮汐**，不能作为港口瞬时流或潮流的实测替代。
- 风方向为来向，流方向为去向，均为地理正北起顺时针；波向保留源数据来向。没有对角度做算术平均，也未直接与无符号图像轴相减。
- 逻辑请求仅为样本小时/日和表层格点。但对象存储有不可避免的物理块额外传输：ERA5 每块为单小时全球 721×1440；CMEMS 每块为 2081 日×1 层×16×16。实际只读取所需块，重复请求命中缓存；未下载全球多年全场。下载日志完整列出键和字节数。

## 缺测、尺度与独立性

波浪缺测 56 例中，50 例的 ERA5 **大气**陆地比例不超过 0.5，只有 6 例超过 0.5，因此大气陆海掩膜不能替代波浪有效掩膜。全体 11 例大气陆地比例大于 0.5，风值仍按来源保留，并附比例供分析筛选。`source_missing_at_nearest_cell` 只陈述源格点无值，不把每个缺失都强行归因为陆地。

CMEMS 缺测两例：2016-03-28 的 `x16498:y4160`（30.38543 N, 122.3332 E），以及 2016-06-30 的 `x2950:y8068`（39.18353 N, 119.0284 E）。前者波浪有效，后者也缺波浪。缺测没有填零或邻域外推，完整标识见 `missing_environment_samples.csv`。

ERA5 最近点距离中位数/最大值为 **10.9184 / 17.4647 km**，CMEMS 为 **3.3745 / 6.2955 km**。这些是区域背景，不是几十米 SAR 裁片的局地观测。表层温盐和波向也不证明图像中存在某种波纹、尾迹或内波机制。

1,349 行仅对应 **288 个 ERA5 格点时刻（五变量齐全 256）**、**634 个 CMEMS 格点时刻（四变量齐全 632）**；最多分别有 25 / 12 张船图共享一条记录。这些格点之间仍有时空相关性，不能当作同数独立观测。采集级统计只涉及 22 景；地域、季节、船型组成和支持关系混杂需由主任务结合真实图像分析。

## 具体核验

样本 `S1A_IW_GRDH_1SDV_20150823T221233_20150823T221258_007396_00A2CA_54D0:x1673:y6973`，SAR 时间 **2015-08-23 22:12:45 UTC**，船位 **39.10183 N, 119.1583 E**。

| 来源 | 实际匹配 | 数值 |
|---|---|---|
| ERA5 | 22:00 UTC；39.0 N, 119.25 E；13.8171 km | u10=-4.080252、v10=-5.337749 m/s；Hs=0.510832 m；周期=2.954313 s；波来向=43.931320 度 |
| CMEMS | 当日日平均；39.0833320618 N, 119.1666641235 E；2.1799 km；深度0.494025 m | u=-0.048219、v=-0.050661 m/s；位温=20.134251°C；盐度=27.591784 |

验证使用缓存原始 Zarr 标量，独立于 xarray 数值解码，检查 packed 值、scale_factor/add_offset、fill、坐标、时间与 CSV。5 个样本覆盖三个地理区域及两类缺测，共 10 组来源案例全部通过；同时检查全体 1,349 行唯一性、22 景、有效标记与值一致、时间窗、表层深度及共享格点值一致。复核发现并修正了 float32 经度归一化与 CSV 精度损失；重新从缓存导出后通过，见 `verification.json`。

## 文件与复现

主要入口：`sample_environment.csv`（1,349 行宽表）、`sample_environment_long.csv`（12,141 行变量长表）、`acquisition_requests.csv`（22 景）、`authenticated_request_recipes.json`（未来标准接口请求）、`coverage_by_acquisition.csv`、`acquisition_environment_summary.csv`（不重复计船的格点均值，不汇总波向）、`unique_environment_records.csv`、`environment_summary.json`、`status.json`。来源与访问证据在 `provenance.json`、`*_source_inventory.json`、`source_access_probes.json` 和 `download_log.csv`。

在本目录执行，现有 Python 环境主依赖未改变；新增库位于本目录 `vendor/`，版本见 `requirements-local.txt`：

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' ./fetch_match_environment.py --offline --max-download-mib 0
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' ./verify_matching.py
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' ./summarize_environment.py
```

首次联网运行去掉 `--offline`，指定 `--max-download-mib 320`。脚本也接受 `--manifest`，必需列为 `dataset,sample_id,acquisition,time,lat,lon`，时间为可解析 UTC；本轮只处理 OSS1，未扩展其他数据集天气下载。验收脚本按当前 OSS1 的 1,349/22 设计，换数据集时应另定实际验收规模。

## 官方/生产者来源

2026-10-17 核查：[CDS ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)，[ECMWF ARCO 产品说明](https://confluence.ecmwf.int/spaces/CKB/pages/505390919)，[Google 公共副本](https://github.com/google-research/arco-era5)及[实际 CDS 请求配置](https://github.com/google-research/arco-era5/blob/main/raw/daily/era5_sl_hourly.cfg)，[CMEMS 官方产品](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)、[PUM 1.7](https://documentation.marine.copernicus.eu/PUM/CMEMS-GLO-PUM-001-030.pdf)及缓存中的官方 STAC。ERA5 DOI `10.24381/cds.adbb2d47`；CMEMS DOI `10.48670/moi-00021`。
