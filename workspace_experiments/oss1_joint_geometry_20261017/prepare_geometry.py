"""Prepare exact-row OSS1 GRDH radar metadata; no ship-axis estimation or fits."""

from pathlib import Path
import json
import math
import re
import xml.etree.ElementTree as ET

import fitz
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import tifffile


HERE = Path(__file__).resolve().parent
WORK = HERE.parent
NATIVE = Path("D:/Documents/project_root/OpenSARShip_1")
MANIFEST = Path("E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/target_manifest.csv")
SCENES = WORK / "scene_cross_dataset_redesign_20261014"
SLC = WORK / "slc_functional_coordinates_four_regions/opensarship1_slc_functional_coordinates_no_ais_v1.csv"
CENTER = re.compile(r"_x(\d+)_y(\d+)(?:_(vv|vh))?\.tif$", re.I)


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def patch_index(folder):
    records = {}
    for path in folder.glob("*.tif"):
        match = CENTER.search(path.name)
        if match:
            key = tuple(map(int, match.group(1, 2)))
            if key in records:
                raise ValueError(f"Nonunique native patch center: {folder}/{key}")
            records[key] = path
    return records


def tiff_info(path):
    if path is None or not path.is_file():
        return {"exists": False, "width": math.nan, "height": math.nan,
                "channels": math.nan, "orientation": math.nan, "dtype": ""}
    with tifffile.TiffFile(path) as handle:
        page = handle.pages[0]
        orientation = page.tags.get("Orientation")
        return {"exists": True, "width": page.imagewidth, "height": page.imagelength,
                "channels": page.samplesperpixel,
                "orientation": int(orientation.value) if orientation else 1,
                "dtype": str(page.dtype)}


def display_alignment(raw_path, row):
    """One fixed first-row probe per product: detect export transpose/rotation."""
    raw = tifffile.imread(raw_path)
    results = []
    for channel, pol in ((1, "vv"), (0, "vh")):
        displayed = tifffile.imread(row[f"chip_path_{pol}"])
        original = raw[:, :, channel]
        transforms = {"identity": original}
        for k in range(4):
            if k:
                transforms[f"rot{k * 90}"] = np.rot90(original, k)
            transforms[f"flip_rot{k * 90}"] = np.rot90(original[:, ::-1], k)
        scores = {name: float(spearmanr(value.ravel(), displayed.ravel()).statistic)
                  for name, value in transforms.items() if value.shape == displayed.shape}
        best = max(scores, key=scores.get)
        results.append({"polarization": pol, "best_transform": best,
                        "identity_rho": scores.get("identity", math.nan),
                        "best_nonidentity_rho": max(v for k, v in scores.items() if k != "identity")})
    return results


def load_product(scene, representative):
    folder = NATIVE / scene
    root = ET.parse(folder / "Metedata.xml").getroot()
    attrs = {node.attrib["name"]: node.attrib.get("value", "")
             for node in root.find("Abstracted_Metadata").findall("attrib")}
    if attrs.get("PRODUCT") != scene:
        raise ValueError(f"Metadata PRODUCT does not match folder: {scene}")
    ships = {}
    for ordinal, node in enumerate(ET.parse(folder / "Ship.xml").getroot().findall("ship")):
        sar = {child.tag: child.text or "" for child in node.find("SARShipInformation")}
        key = (int(sar["Center_x"]), int(sar["Center_y"]))
        if key in ships:
            raise ValueError(f"Nonunique Ship.xml center: {scene}/{key}")
        ships[key] = (ordinal, sar, node.findtext("AISShipInformation/MMSI", "").strip(),
                      node.findtext("MarineTrafficInformation/Elaborated_type", "").strip())
    raw = patch_index(folder / "Patch")
    calibrated = patch_index(folder / "Patch_Cal")
    with fitz.open(folder / "ReadMe.pdf") as document:
        text = "\n".join(page.get_text() for page in document)
    documented_axes = "x(range)" in text and "y(azimuth)" in text
    key = (int(representative.center_x), int(representative.center_y))
    probe = display_alignment(raw[key], representative)
    probe_ok = all(item["best_transform"] == "identity"
                   and item["identity_rho"] > item["best_nonidentity_rho"]
                   for item in probe)
    native_geometry = (attrs.get("PRODUCT_TYPE") == "GRD"
                       and attrs.get("SAMPLE_TYPE") == "DETECTED"
                       and attrs.get("map_projection") == "-"
                       and attrs.get("is_terrain_corrected") == "0"
                       and attrs.get("srgr_flag") == "1")
    return dict(attrs=attrs, ships=ships, raw=raw, calibrated=calibrated,
                axis_evidence_valid=documented_axes and native_geometry and probe_ok,
                documented_axes=documented_axes, probe=probe,
                representative_sample_id=representative.sample_id)


def fold_coverage(frame):
    folds = [("pilot", SCENES / "strict_training/oss1", SCENES / "public_split", "oss1_")]
    for folder in sorted((SCENES / "public_expansion/oss1").iterdir()):
        if folder.is_dir() and (folder / "fit/pilot_features.npz").is_file():
            folds.append((folder.name, folder / "fit", folder, ""))
    index = frame.set_index("sample_id")
    counts = []
    for name, cache, episode, prefix in folds:
        source = cache if (cache / "source_train.csv").is_file() else episode
        roles = [("source_train", source / "source_train.csv"),
                 ("source_validation", source / "source_validation.csv"),
                 ("query", episode / f"{prefix}query.csv"),
                 ("support_draws_B16", episode / f"{prefix}support_draws.csv")]
        for role, path in roles:
            records = pd.read_csv(path, dtype={"sample_id": str})
            if "budget" in records:
                records = records[records.budget == 16]
            ids = records.sample_id.drop_duplicates()
            matched = index.loc[ids]
            counts.append(dict(fold=name, role=role, rows=len(records), unique_samples=len(ids),
                               radar_axis_valid=int(matched.radar_axis_valid.sum()),
                               incidence_valid=int(matched.incidence_valid.sum())))
    return pd.DataFrame(counts)


def main():
    manifest = pd.read_csv(MANIFEST, dtype={"sample_id": str, "mmsi_or_ship_id": str})
    if manifest.sample_id.duplicated().any() or not manifest.product_type.eq("GRDH").all():
        raise ValueError("Expected unique current GRDH manifest, not an SLC table")
    cached_ids = np.load(SCENES / "public_expansion/oss1_ids.npy").astype(str)
    np.testing.assert_array_equal(cached_ids, manifest.sample_id.to_numpy(str))
    slc = pd.read_csv(SLC, usecols=["scene_id", "sample_key", "incidence_deg"])
    products = {scene: load_product(scene, group.iloc[0])
                for scene, group in manifest.groupby("scene_id", sort=False)}
    rows = []
    for global_row, row in manifest.iterrows():
        product = products[row.scene_id]
        attrs = product["attrs"]
        key = (int(row.center_x), int(row.center_y))
        ordinal, sar, mmsi, elaborated = product["ships"][key]
        expected_id = f"{row.scene_id}:x{key[0]}:y{key[1]}"
        if (row.sample_id != expected_id or int(row.xml_ship_index) != ordinal
                or row.mmsi_or_ship_id != mmsi or row.label_elaborated != elaborated):
            raise ValueError(f"Manifest/XML identity mismatch: {row.sample_id}")
        record = dict(global_row=int(global_row), target_manifest_row=int(row.target_manifest_row),
                      sample_id=row.sample_id, scene_id=row.scene_id, product=row.scene_id,
                      product_type=row.product_type,
                      acquisition="_".join((row.scene_id.split("_")[0], *row.scene_id.split("_")[-3:-1])),
                      xml_ship_index=ordinal, center_x=key[0], center_y=key[1], mmsi=mmsi,
                      class_id=int(row.class_id), class_name=row.class_name,
                      label_elaborated=elaborated, identity_match_valid=True,
                      ship_xml_path=str(NATIVE / row.scene_id / "Ship.xml"),
                      metadata_xml_path=str(NATIVE / row.scene_id / "Metedata.xml"))
        for pol in ("vv", "vh"):
            path = Path(row[f"chip_path_{pol}"])
            match = CENTER.search(path.name)
            if (path.parent.parent.name != row.scene_id or match is None
                    or tuple(map(int, match.group(1, 2))) != key or match.group(3).lower() != pol):
                raise ValueError(f"Manifest/image identity mismatch: {path}")
            record[f"chip_path_{pol}"] = str(path)
            record.update({f"native_{name}_{pol}": value for name, value in tiff_info(path).items()})
        for folder, field in (("raw", "raw_patch"), ("calibrated", "calibrated_patch")):
            path = product[folder].get(key)
            record[f"{field}_path"] = str(path) if path else ""
            record.update({f"{field}_{name}": value for name, value in tiff_info(path).items()})
        width, height = record["native_width_vv"], record["native_height_vv"]
        extent_x = abs(number(sar.get("UpperLeft_x")) - number(sar.get("LowerRight_x"))) + 1
        extent_y = abs(number(sar.get("UpperLeft_y")) - number(sar.get("LowerRight_y"))) + 1
        size_ok = (width == record["native_width_vh"] == record["raw_patch_width"] == extent_x
                   and height == record["native_height_vh"] == record["raw_patch_height"] == extent_y)
        record["xml_crop_extent_matches"] = bool(size_ok)
        axis_valid = (product["axis_evidence_valid"] and size_ok
                      and record["native_orientation_vv"] == record["native_orientation_vh"] == 1)
        incidence = number(sar.get("Incidence"))
        record.update(radar_axis_valid=bool(axis_valid), radar_valid=bool(axis_valid),
                      radar_axis_rad=0.0 if axis_valid else math.nan,
                      radar_range_axis_deg=0.0 if axis_valid else math.nan,
                      radar_azimuth_axis_deg=90.0 if axis_valid else math.nan,
                      radar_axis_semantics="undirected_native_image_axis_mod180_xright_ydown",
                      radar_axis_source="ReadMe_Fig2+native_GRD_metadata+TIFF_orientation+export_probe",
                      radar_direction_sign_valid=False,
                      radar_look_direction_image_deg=math.nan,
                      radar_look_bearing_geographic_deg=math.nan,
                      incidence_deg=incidence if 0 < incidence < 90 else math.nan,
                      incidence_rad=math.radians(incidence) if 0 < incidence < 90 else math.nan,
                      incidence_valid=bool(0 < incidence < 90),
                      incidence_source="Ship.xml/SARShipInformation/Incidence",
                      ground_range_spacing_m=number(attrs.get("range_spacing")),
                      azimuth_spacing_m=number(attrs.get("azimuth_spacing")),
                      orbit_pass=attrs.get("PASS", ""), antenna_pointing=attrs.get("antenna_pointing", ""),
                      image_map_projection=attrs.get("map_projection", ""),
                      is_terrain_corrected=number(attrs.get("is_terrain_corrected")),
                      srgr_flag=number(attrs.get("srgr_flag")),
                      sar_acquisition_time=sar.get("Time_UTC", ""),
                      sar_center_longitude=number(sar.get("Center_longitude")),
                      sar_center_latitude=number(sar.get("Center_latitude")),
                      ship_axis_estimated=False, ais_direction_used=False)
        rows.append(record)
    geometry = pd.DataFrame(rows)
    counts = fold_coverage(geometry)
    geometry.to_csv(HERE / "geometry.csv", index=False)
    valid_inc = geometry.loc[geometry.incidence_valid, "incidence_deg"]
    probes = [{"scene_id": name, "sample_id": item["representative_sample_id"], **probe}
              for name, item in products.items() for probe in item["probe"]]
    probe_frame = pd.DataFrame(probes)
    summary = dict(rows=len(geometry), products=geometry.scene_id.nunique(), ships=geometry.mmsi.nunique(),
                   classes=geometry.class_id.nunique(), radar_axis_valid=int(geometry.radar_axis_valid.sum()),
                   incidence_valid=int(geometry.incidence_valid.sum()),
                   paired_native_available=int((geometry.native_exists_vv & geometry.native_exists_vh).sum()),
                   raw_available=int(geometry.raw_patch_exists.sum()),
                   calibrated_available=int(geometry.calibrated_patch_exists.sum()),
                   xml_crop_extent_matches=int(geometry.xml_crop_extent_matches.sum()),
                   slc_rows=len(slc), slc_products=slc.scene_id.nunique(),
                   exact_slc_product_overlap=len(set(slc.scene_id) & set(geometry.scene_id)),
                   incidence_min_deg=float(valid_inc.min()), incidence_max_deg=float(valid_inc.max()),
                   export_probe_count=len(probe_frame),
                   export_probe_identity_best=int(probe_frame.best_transform.eq("identity").sum()),
                   export_probe_min_identity_rho=float(probe_frame.identity_rho.min()),
                   cache_ids_exact=True)
    notes = ["# OpenSARShip 1.0 联合几何元数据准备", "",
             "## 本轮输出与范围", "",
             "只准备元数据，不训练、不估计船轴、不改既有输入或划分。当前八分类队列是 GRDH，"
             "不是所提供的 SLC 功能坐标表。geometry.csv 严格保留现有 target_manifest.csv 的行序。",
             "", "```json", json.dumps(summary, indent=2, ensure_ascii=False), "```", "",
             "## 主代理接入约定", "",
             "- `global_row` 是当前 1349 行 manifest / oss1_inputs128.npy 的零基行号；"
             "`target_manifest_row` 是历史全量标签映射的行号，不可当作本缓存行号。",
             "- `sample_id` 按产品 + 原生中心 x/y 对回 Ship.xml，同时逐行检查 XML 序号、MMSI、"
             "细分类标签及 VV/VH 文件中心；没有按相似船型或文件遍历顺序拼接。",
             "- `chip_path_vv/vh` 是模型使用的未 resize Uint8 TIFF；`raw_patch_path` 是原始幅度双通道，"
             "`calibrated_patch_path` 是标定双通道。ReadMe 定义两者通道 0=VH、1=VV。",
             "- `radar_range_axis_deg`、`radar_azimuth_axis_deg` 从图像向右为 0 度，向下为 90 度，"
             "是模 180 度的无向轴。有效时分别为 0、90。联合 YOLO 船轴时必须使用同一图像坐标。",
             "- 主读出接口：`radar_axis_rad` 为上述距离轴的弧度值，`radar_valid` 为有效性；"
             "`incidence_rad` 为入射角弧度值，`incidence_valid` 独立标记。无效角留空，不填伪零。",
             "- `incidence_deg` 是对应 GRDH 样本中心的入射角，来自 Ship.xml 的 SAR 字段，"
             "不是平面旋转角；有效性独立于雷达轴与后续船轴质量。",
             "- 不输出 AIS 航向或人工船轴；MMSI 只用于身份和分组，class_id/name 只用于标签。",
             "- `radar_direction_sign_valid=False`，有向视线和地理方位字段留空。右视、升降轨"
             "不能直接变成裁片里的有向视线。当前只确认横向距离轴、纵向方位轴。",
             "- 本队列雷达无向轴与图像坐标轴重合，`radar_axis_rad` 全为 0 是真实栅格语义，"
             "不是各船都有不同雷达平面角。相对于检测船轴的夹角仍随船变化，入射角也随样本变化。"
             "因此不能把单独拼接常数 0 的效果解释成新增观测信息；联合局部组织应与图像坐标读出"
             "公平比较，区分坐标重新表达与入射角条件信息。",
             "", "## 直接依据与有限验证", "",
             "1. 每个产品随附 ReadMe.pdf，正文第 2 页图 2 标注 x(range)、y(azimuth)；"
             "第 7 页把 Chip_Direction 定义为船相对于方位向的角度，North_Direction 是船相对于北的方向。"
             "因此两字段均未用作雷达观测角。",
             "2. 22 个实际产品的 Metedata.xml/Abstracted_Metadata 均为 GRD/DETECTED，"
             "map_projection='-'、is_terrain_corrected=0、srgr_flag=1；range_spacing=10m，"
             "azimuth_spacing 逐产品保留真实值，不把采样间隔说成分辨率。",
             "3. 逐张读取 TIFF 尺寸与 Orientation，并将显示裁片、原始幅度裁片和 XML 包围尺寸对照。"
             "产品中第一条 manifest 样本的 VV/VH，另用幅度与 Uint8 的像素秩相关检查 8 种转置/翻转/旋转，"
             "这是每产品代表样本的导出方向核对，不是全部 2698 张图的逐像素重建。",
             "方向核查比较 identity 是否优于其他方向，不用绝对亮度秩相关阈值作为几何质量门槛；"
             "部分 Uint8 显示增强会降低原始幅度与显示值的相关程度，这不等于方向不明。",
             "4. public_expansion/cache_oss1.py 仅灰度读取和双线性 resize，未旋转或翻转；"
             "缓存 oss1_ids.npy 与 manifest sample_id 全序列精确一致。下游新增旋转、翻转或转置增强时，"
             "必须同步变换雷达轴。",
             "", "## 不跨产品移植的内容", "",
             f"提供的 SLC 表有 {len(slc)} 行、{slc.scene_id.nunique()} 个产品，与本轮 GRDH 产品精确交集为 0。"
             "相同采集的 SLC/GRD 也不是同一像素网格，不能移植其图像方向、中心坐标或按行拼接入射角。",
             "该表和生成脚本中的 orientation_relative_azimuth_deg、func_slc_axis_image_deg、"
             "func_sar_ship_axis_geo_deg 均未进入本输出。",
             "", "## 还未恢复的量及可用路径", "",
             "当前无向轴足够表达船轴与距离向/方位向的相对轴角，但不能区分朝雷达一侧与背雷达一侧。"
             "若后续需要有向地面视线，原始 Metedata.xml 已含 Orbit_State_Vectors、first/last_line_time、"
             "line_time_interval、latitude/longitude/slant_range_time 网格以及右视标志；应确认裁片像素到"
             "原始栅格的有向映射，再用轨道与目标定位求解，并作图核对。未在本轮推算或伪造。",
             "注意：XML UpperLeft/LowerRight 的数值命名可能与数组左右大小相反，本轮只用绝对跨度"
             "验证裁片尺寸，不据此推断 near/far 的正负。",
             "", "## 四折覆盖", "",
             "保持 public_full_20261014/run.py 的 pilot + 3 expanded 划分，不合并各折训练角色。"
             "support_draws_B16 的 rows 包含五次抽样重复，unique_samples 才是该折支持样本数。", "",
             counts.to_markdown(index=False), "", "## 类别数量", "",
             geometry.groupby(["class_id", "class_name"]).size().rename("samples").reset_index().to_markdown(index=False),
             "", "## 导出方向探针", "", probe_frame.to_markdown(index=False), "",
             "## 可复现来源", "", f"- Manifest: `{MANIFEST.as_posix()}`",
             f"- Script: `{(HERE / 'prepare_geometry.py').as_posix()}`",
             f"- SLC reference (not joined): `{SLC.as_posix()}`",
             f"- Dataset documentation: `{(NATIVE / geometry.scene_id.iloc[0] / 'ReadMe.pdf').as_posix()}`",
             f"- Existing cache: `{(SCENES / 'public_expansion/cache_oss1.py').as_posix()}`", ""]
    (HERE / "GEOMETRY_NOTES_ZH.md").write_text("\n".join(notes), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
