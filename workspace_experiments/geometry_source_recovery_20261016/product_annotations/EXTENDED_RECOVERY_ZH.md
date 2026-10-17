# 固定队列产品注释扩展

2026-10-16。新增 83/83 个精确产品恢复成功。`manifest.csv` 已保留此前四产品并追加新记录，
共 87/87 个成功产品，覆盖原固定 8622 行队列全部 5145 个 radar_valid=0 样本的产品注释需求。
这表示产品网格已备齐，不表示 5145 个像素方向已计算或一定全部适用。

## 选择范围

重新读取原 `geometry_targeted_fix_20261016/recovered_geometry.csv`，
并使用 `geometry_relation_slots_20261016/slots_cache/*.npz` 的 rows 并集固定 8622 行。
在该集合中按实际 `radar_valid=0` 取 5145 行，逐行核对 global_row 和 instance_id，
再以原 manifest 中完整 product 关联，得到 87 个产品。

按 `old.GEOMETRY/radar_grid/<product>.json` 的 status=ok 且内部 product 精确相等过滤，
本轮这 87 个产品没有命中旧成功网格；扣除此前已恢复的 4 个，剩 83 产品、4358 行。
所有新增样本原 radar_recovery_status 均属于缺精确 georef 的那批；产品网格可提前备齐，
无需等主线 sourcebridge 完成。没有改动样本队列、源图像关联或任何角度。

## 执行与下载量

执行前已报告 83 个产品的完整 VV 注释在历史目录全部存在，总计 144007584 字节，
预计下载量为 0。实际 83 个全部通过完整 XML 解析与产品身份检查，复用原历史下载来源，
无需触发任何网络请求。因此本轮网络请求数 0、下载字节 0、整景/measurement 下载 0。
没有新的网络长操作；之前独立来源检查脚本仍保留 20 秒 socket 超时及读取字节上限。

注释原始来源路径、source_url、完整副本路径、网格路径、网格点数及状态逐产品记录在
`extended_products.csv`，同一批完整记录同时合入 `manifest.csv`。
每份 XML 均核对卫星、模式、极化、起止时刻、轨道号、datatake、Ground Range 和网格字段，
并核对同精确产品历史 source_objects.json 的对象键与文件长度。
新 JSON 复用现有 status/product/points 格式，保留原始 line/pixel/latitude/longitude/incidenceAngle，
没有从三列入射角缓存补造 line/pixel。

## 发布与验证

所有新 XML 和网格 JSON 写入同目录临时文件后替换，待它们完成后，才将完整 manifest
写到 `manifest.csv.tmp`，关闭文件并原子替换正式 manifest。
此前 4 条成功记录逐字段保留，最终 manifest 无重复 product，87 个 JSON 全部通过
现有 `relative_radar_geometry._validate` 校验，并检查实际节点数与 manifest 相符。
未恢复产品集合为空。没有训练、像素角度转换或旧表写入。

## 主线导入

完整入口为 `manifest.csv`，按 product 精确关联 `local_grid_path`。
`extended_products.csv` 为新增 83 产品列表；`extended_target_rows.csv` 为新增 4358 行映射。
`extended_summary.json` 记录本轮与合并后的覆盖；原 `summary.json`、`target_rows.csv`
继续保留最初 4 产品/787 行的范围，不将旧收据改写成新结果。
主线按有坐标的样本实际使用，并继续执行原方向计算的网格适用域约束。

新增 `extend_product_annotations.py`，复用 `recover_product_annotations.py` 中的单产品恢复函数。
原脚本的 CSV 发布已改为原子替换，重跑原 4 产品时也按 product 合并，避免抹掉扩展记录。
本轮所有写入限定在 product_annotations/；原全局网格、旧 recovered_geometry.csv 和主线文件均未修改。
