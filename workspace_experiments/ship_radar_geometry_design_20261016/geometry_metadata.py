"""Assemble label-free image-frame geometry; no model imports or training."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Geod, Transformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRACK = Path("E:/Hermes/paper_tracking")
GEOMETRY = TRACK / "sensitivity/relative_observation_geometry_20261009"
MANIFEST = ROOT / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv"
SYNC = Path("E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata/local_knowledge_sync_v1/private")
INCIDENCE = TRACK / "pixel_correction_review/chip_incidence_v1/incidence_by_instance.csv"
SOURCE = ROOT / "prepool_local_probe_20261015/body_context_expansion/ring_relaxed/query_coverage/frozen.json"
GEOD = Geod(ellps="WGS84")
GEOMETRY_COLUMNS = ("ship_axis_rad", "radar_axis_rad", "incidence_rad", "ship_valid",
                    "radar_valid", "incidence_valid", "ship_confidence")


@lru_cache(None)
def projection(epsg):
    return Transformer.from_crs(4326, epsg, always_xy=True)


def image_axis(azimuth, longitude, latitude, geo, width, height):
    """WGS84 azimuth -> native TIFF pixel vector -> square resized PNG axis.

    Crop offsets and pixel-center translations cancel in a displacement.
    A symmetric 100 m chord approximates the local projection Jacobian.
    """
    transformer = projection(int(geo["epsg"]))
    endpoints = [GEOD.fwd(longitude, latitude, azimuth, distance)[:2]
                 for distance in (-50., 50.)]
    xy = np.asarray([transformer.transform(*p) for p in endpoints])
    gt = geo["transform"]
    matrix = np.array([[gt[1], gt[2]], [gt[4], gt[5]]], dtype=float)
    native = np.linalg.solve(matrix, xy[1] - xy[0])
    resized = native / np.array([width, height], dtype=float)
    return float(np.arctan2(resized[1], resized[0]) % np.pi)


def axis_error(a, b):
    return np.abs((np.asarray(a) - b + np.pi / 2) % np.pi - np.pi / 2)


def load_georefs():
    result = {}
    for path in sorted(SYNC.glob("server*_georef.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("status") == "ok":
                result[row["remote_source_tiff"]] = row
    return result


def assemble():
    frame = pd.read_csv(MANIFEST, low_memory=False)
    assert frame.instance_id.is_unique
    pairs = pd.read_csv(GEOMETRY / "paired_axes.csv").set_index("instance_id")
    relative = pd.read_csv(GEOMETRY / "relative_conditions.csv").set_index("instance_id")
    views = pd.read_csv(GEOMETRY / "view_axes.csv", low_memory=False).set_index(["instance_id", "polarization"])
    inc = pd.read_csv(INCIDENCE).set_index("instance_id")
    assert pairs.index.is_unique and relative.index.is_unique and views.index.is_unique and inc.index.is_unique
    georefs = load_georefs()
    records = []
    for index, row in frame.iterrows():
        sid = row.instance_id
        record = dict(global_row=index, manifest_row=index, sample_id=sid, instance_id=sid,
                      ship_axis_rad=np.nan, radar_axis_rad=np.nan, incidence_rad=np.nan,
                      ship_valid=0, radar_valid=0, incidence_valid=0, ship_confidence=0.,
                      ship_status="no_existing_axis", radar_status="no_existing_range_axis",
                      incidence_status="no_incidence_join", coordinate_frame="resized_image_x_right_y_down",
                      reference_view="vv", ship_disagreement_deg=np.nan,
                      ship_min_elongation=np.nan, nearest_grid_distance_m=np.nan)
        if sid in inc.index:
            ir = inc.loc[sid]
            record["incidence_status"] = ir.status
            if ir.status == "ok" and np.isfinite(ir.incidence_deg):
                assert 0 < ir.incidence_deg < 90
                record.update(incidence_rad=np.deg2rad(ir.incidence_deg), incidence_valid=1)
        pair = pairs.loc[sid] if sid in pairs.index else None
        rel = relative.loc[sid] if sid in relative.index else None
        if pair is not None:
            record["ship_status"] = pair.status
        if rel is not None:
            record["radar_status"] = rel.relative_status
            record["nearest_grid_distance_m"] = rel.nearest_grid_distance_m
        if pair is not None and bool(pair.accepted):
            assert np.isfinite(pair.body_axis_deg)
            record["ship_disagreement_deg"] = pair.disagreement_deg
            for pol in ("vv", "vh"):
                view = views.loc[(sid, pol)]
                geo = georefs.get(row["source_tiff_" + pol])
                if geo is None:
                    record["ship_status"] = "missing_georef_for_image_mapping"
                    continue
                # Read actual PNG headers, not detector box width/height fields.
                with Image.open(row["image_relpath_" + pol]) as image:
                    width, height = image.size
                assert width > 0 and height > 0
                record[f"native_width_{pol}"] = width
                record[f"native_height_{pol}"] = height
                record[f"epsg_{pol}"] = int(geo["epsg"])
                record[f"ship_measured_axis_rad_{pol}"] = np.deg2rad(view.angle_pixel_deg) % np.pi
                recovered = image_axis(view.axis_true_north_deg, view.longitude, view.latitude,
                                       geo, width, height)
                record[f"axis_roundtrip_error_deg_{pol}"] = np.rad2deg(axis_error(
                    recovered, record[f"ship_measured_axis_rad_{pol}"]))
                record[f"ship_axis_rad_{pol}"] = image_axis(
                    pair.body_axis_deg, view.longitude, view.latitude, geo, width, height)
                record[f"ship_elongation_{pol}"] = view.elongation
                if rel is not None and rel.relative_status == "ok":
                    record[f"radar_axis_rad_{pol}"] = image_axis(
                        rel.range_axis_deg, view.longitude, view.latitude, geo, width, height)
            if "ship_axis_rad_vv" in record and "ship_axis_rad_vh" in record:
                minimum = min(record["ship_elongation_vv"], record["ship_elongation_vh"])
                record.update(ship_axis_rad=record["ship_axis_rad_vv"], ship_valid=1,
                              ship_status="accepted_paired_axis_in_vv_frame", ship_min_elongation=minimum,
                              ship_confidence=(minimum**2 - 1.) / (minimum**2 + 1.))
            if "radar_axis_rad_vv" in record and "radar_axis_rad_vh" in record:
                record.update(radar_axis_rad=record["radar_axis_rad_vv"], radar_valid=1,
                              radar_status="cached_ground_range_proxy_in_vv_frame")
                record["radar_vv_vh_frame_difference_deg"] = np.rad2deg(axis_error(
                    record["radar_axis_rad_vv"], record["radar_axis_rad_vh"]))
        records.append(record)
    result = pd.DataFrame(records)
    for name in ("ship", "radar", "incidence"):
        value = name + ("_rad" if name == "incidence" else "_axis_rad")
        valid = result[name + "_valid"].astype(bool)
        assert result.loc[valid, value].notna().all()
        assert result.loc[~valid, value].isna().all()
        assert result.loc[valid, value].between(0, np.pi, inclusive="left").all()
    errors = result[[f"axis_roundtrip_error_deg_{p}" for p in ("vv", "vh")]].to_numpy()
    # Detect wrong y sign, missing resize correction or mismatched source affine.
    assert np.nanmax(errors) < .01, np.nanmax(errors)
    assert result.global_row.tolist() == list(range(len(frame)))
    assert result.sample_id.tolist() == frame.instance_id.tolist()
    return frame, result


def coverage_line(name, group, ships=None, acquisitions=None):
    valid = group[["ship_valid", "radar_valid", "incidence_valid"]]
    joint = valid.all(axis=1).sum()
    return (f"| {name} | {len(group)} | {ships if ships is not None else '-'} | "
            f"{acquisitions if acquisitions is not None else '-'} | "
            f"{int(group.ship_valid.sum())} | {int(group.radar_valid.sum())} | "
            f"{int(group.incidence_valid.sum())} | {joint} |")


def report(frame, result):
    lines = ["# 几何元数据覆盖与坐标说明", "", "2026-10-16；仅组装既有无标签几何，不训练、不按目标表现挑选。", "",
             "## 读取接口", "", "CSV按原manifest顺序覆盖全部样本。`global_row`和`manifest_row`为0起始行号；"
             "`sample_id`与`instance_id`相同。", "",
             "`geometry[N,7]`列顺序：`" + ", ".join(GEOMETRY_COLUMNS) + "`。",
             "缺失角度为NaN，valid为0，缺失ship_confidence为0；加载后须先保留mask再将NaN置0。"
             "0不是缺失角度的物理含义。三个valid分别标记可用性，不表示统计独立。", "",
             "## 坐标和可解释边界", "",
             "- 两个axis均是无向轴，范围[0,pi)，以图像向右为0、向下为pi/2；不是船首方向。"
             "incidence是[0,pi/2]弧度，不是图像旋转角。",
             "- 船轴复用既有VV/VH接受门限：各自亮散射二阶矩、elongation>=2、中心偏移和边界筛选，"
             "双视图地理轴差<=15度；不新增按标签选择的阈值。成对地理轴映射到VV图像作为主坐标，"
             "每极化的映射和直接测量轴均保留作诊断。",
             "- ship_confidence=(r_min^2-1)/(r_min^2+1)，r_min是双视图最小细长比。"
             "它是协方差各向异性强度，不是正确方向概率；双极化角差单列。",
             "- 雷达轴复用relative_conditions.range_axis_deg的绝对地理方位，不用relative_signed_deg推算。"
             "来源为同产品注释网格固定line、增加pixel的地距轴代理，既有覆盖内/最近网格<=20km限制保留。"
             "不是精确雷达波束方位；nearest_grid_distance_m只是质量诊断，不伪造置信概率。",
             "- 将WGS84方位在样本处构造对称100米局部弦，投影到源EPSG；"
             "对GeoTIFF完整2x2仿射矩阵求逆，再按真实PNG宽高转换到等边resize坐标。"
             "该变换包含投影子午线收敛、y符号及非等比缩放；不假定北向图像、不把y向下当北向。",
             "- 原生PNG来自轴对齐窗口裁片，未旋转。裁切/像素中心平移不影响位移轴；"
             "128x128双线性resize及当前浅层读取后续112x112等比resize不改变本表方向。"
             "若训练另加旋转/翻转或非等比缩放，必须同步变换这两个轴，不能原样使用。",
             "- VV为共同token网格参考；保留VV/VH映射差。这只是方向，不声称两个裁片逐像素精配准。"
             "物理船轴仍是亮散射主体代理，不是人工船体标注。",
             "- 未扩大旧几何提取范围：雷达轴缓存原来只针对通过船轴筛选者计算。"
             "因此雷达缺失部分来自历史筛选，不等于这些图像从物理上无法得到雷达方向。", "",
             "## 溯源文件", "",
             f"- manifest: `{MANIFEST}`",
             f"- 三个方向缓存: `{GEOMETRY}` 内view_axes.csv / paired_axes.csv / relative_conditions.csv。",
             f"- 源仿射: `{SYNC}` 内server*_georef.jsonl，仅status=ok且源TIFF路径精确匹配。",
             f"- 入射角: `{INCIDENCE}`，status=ok且有限值；不依赖observations的已知MMSI子集。",
             "- 裁片: `v5_server_toolkit_20260912/scripts/build_v5_context_chips.py` 的Image.crop；"
             "合并: `visual_reliable_baseline/merge_returned_v5_dataset.py` 保留crop_left/top与源路径。",
             "- 轴提取/方向: `E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/"
             "extract_ship_axis_proxy.py`、`ship_axis_proxy.py`、`relative_radar_geometry.py`。", "",
             "## 覆盖", "", "| 范围 | 不同样本 | 船只 | 采集 | 船轴 | 雷达轴 | 入射角 | 三项齐全 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
             coverage_line("全manifest", result, frame.mmsi.nunique(), frame["product"].nunique())]
    for port, indexes in frame.groupby("port").groups.items():
        lines.append(coverage_line(port, result.loc[indexes]))
    lines += ["", "## 五池实际B32源episode覆盖", "",
              "按冻结query_coverage的B32任务汇总唯一支持/查询样本，不把重复episode暴露算成独立证据。"
              "每池独立列出，不能合池训练；以下不使用目标标签。", "",
              "| 池/阶段/角色 | 不同样本 | 船只 | 采集 | 船轴 | 雷达轴 | 入射角 | 三项齐全 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    frozen = json.loads(SOURCE.read_text(encoding="utf-8"))
    for pool in frozen["pools"]:
        meta = pd.read_csv(Path(pool["source_folder"]) / "metadata.csv", dtype={"mmsi": str})
        assert np.array_equal(meta.position, np.arange(len(meta)))
        for split, tasks in pool["splits"].items():
            tasks = [t for t in tasks if t["budget"] == 32]
            for role in ("s", "q"):
                positions = sorted({i for t in tasks for i in t[role]})
                selected = meta.iloc[positions]
                chosen = result.iloc[selected.global_row.to_numpy(int)]
                assert chosen.sample_id.tolist() == selected.sample_id.tolist()
                lines.append(coverage_line(f"{pool['pool']}/{split}/{role}", chosen,
                                           selected.mmsi.nunique(), selected.acquisition.nunique()))
    lines += ["", "### 对本轮设计的直接限制", "",
              "- Shanghai池fit查询只有5/706例双轴齐全；这限制了几何交互的实际监督，"
              "不能用训练episode重复次数代替不同有效样本量。",
              "- Jebel Ali池selection支持/查询均无双轴；Singapore池selection查询无双轴。"
              "这些选择集仍可评价完整分类与缺失回退，但不能单独检验查询几何交互。",
              "- Ningbo-Zhoushan、Qingdao、Shanghai港的现有船轴均为0覆盖；"
              "零覆盖是旧可精确关联的返回数据/筛选范围限制，不是这三个港没有物理方向。"
              "在这些目标港，C与B/A的差异不能直接归因于使用了可靠双轴。",
              "- 因此完整队列结果与双轴可靠子集结果必须分开解释。"
              "本文件不自行扩展轴提取、不修改原训练划分，也不根据该限制挑目标港。"]
    errors = result[[f"axis_roundtrip_error_deg_{p}" for p in ("vv", "vh")]].to_numpy()
    delta = result.radar_vv_vh_frame_difference_deg.dropna()
    lines += ["", "## 已运行的针对性验证", "",
              f"- 原像素轴 -> 既有地理轴 -> 本次像素轴回放，最大误差{np.nanmax(errors):.8f}度。"
              "阈值0.01度用于发现符号/仿射/缩放错误，不用于筛选模型收益。",
              f"- VV/VH雷达方向映射差：中位数{delta.median():.8f}度，最大{delta.max():.8f}度。",
              "- 全manifest行序、sample_id和源episode global_row匹配，缺失值与valid一致；"
              "入射角与轴均检查合法范围。",
              "- 不读取预测结果、不调整目标抽样、不启动训练、不下载大景。"
              "覆盖是数据可用性，不是分类效果证据。"]
    (HERE / "GEOMETRY_COVERAGE_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    frame, result = assemble()
    result.to_csv(HERE / "geometry_metadata.csv", index=False, float_format="%.12g")
    report(frame, result)
    print(result[["ship_valid", "radar_valid", "incidence_valid"]].sum().to_string())
    print(f"Saved {len(result)} rows to {HERE}")
