"""Recompute manuscript dataset counts from existing local manifests."""
from collections import Counter
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FULL = WORK / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv"
SPLIT = WORK / "same_vessel_gallery_20261019/crude_input_controls_20261021/source_coverage/split_membership.csv"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (ROOT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    full = read_csv(FULL)
    diagnostic = [r for r in read_csv(SPLIT) if r["partition"] == "test"]
    ports, dports = Counter(r["port"] for r in full), Counter(r["port"] for r in diagnostic)
    classes, dclasses = Counter(r["class_id"] for r in full), Counter(r["class_id"] for r in diagnostic)
    names = {r["class_id"]: r["class_name"] for r in full}
    port_rows = [dict(port=p, total=ports[p], classes=len({r["class_id"] for r in full if r["port"] == p}),
                      diagnostic=dports[p]) for p in sorted(ports)]
    class_rows = [dict(class_id=c, class_name=names[c], total=classes[c], diagnostic=dclasses[c]) for c in sorted(classes)]
    assert len(full) == 38091 and len(diagnostic) == 4744
    assert len({r["obs_id"] for r in diagnostic}) == 4744
    manuscript = (ROOT / "MANUSCRIPT_ZH.md").read_text(encoding="utf-8")
    for r in port_rows:
        line = f"| {r['port']} | {r['total']:,} | {r['classes']} | {r['diagnostic']:,} |"
        assert line in manuscript, line
    write_csv("dataset_24ports_by_port.csv", port_rows)
    write_csv("dataset_24ports_by_class.csv", class_rows)
    products = {r["product"] for r in full}
    dates = sorted(p.split("_")[4][:8] for p in products)
    passes = {(p.split("_")[0], p.split("_")[6]) for p in products}
    platforms = dict(Counter(r["product"].split("_")[0] for r in full))
    summary = dict(observations=len(full), ports=len(ports), classes=len(classes),
                   products=len(products), platform_absolute_orbit_groups=len(passes),
                   first_date=dates[0], last_date=dates[-1], platforms=platforms,
                   diagnostic_observations=len(diagnostic), diagnostic_ports=len(dports),
                   diagnostic_mmsi=len({r['mmsi'] for r in diagnostic}),
                   full_manifest=str(FULL), diagnostic_split=str(SPLIT))
    (ROOT / "dataset_profile_counts.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    start = manuscript.index("## 5.1 数据集介绍")
    end = manuscript.index("## 5.3 实现细节与训练配置")
    excerpt = manuscript[start:end]
    text = "# 数据集说明与统计来源\n\n更新日期：2026-11-11；当前实验范围为OSS2与自建24港。不训练、不修改划分、不改标签。\n\n"
    text += "## 与正文一致的数据说明\n\n" + excerpt
    text += "\n## 本地来源\n\n"
    sources = [FULL, SPLIT, WORK / "chip_quality_20261017/ais_metadata/summary.json",
               WORK / "same_vessel_gallery_20261019/condition_outcomes_20261021/full_cohort_groups.csv",
               Path("E:/Hermes/paper_tracking/sensitivity/dataset_distribution_20261008/all24ports_class_counts.csv"),
               Path("E:/Hermes/paper_tracking/sensitivity/dataset_distribution_20261008/LABEL_PROTOCOL_AUDIT_ZH.md"),
               WORK / "m1_coverage_20261101/coverage.csv", WORK / "m1_coverage_20261101/manifest.json"]
    for path in sources:
        text += f"- [{path.name}](<{path.as_posix()}>)\n"
    text += "\n## 口径说明\n\n"
    text += "总库、筛选输入和实验子集分别统计。24港总库含38,091条观测；历史38,027条／23港是扣除仅64条的Melbourne之后的另一任务口径，不取代总库。原报告中的64目标支持及5%交接不是当前32源参考协议。\n\n"
    text += "OSS2使用IW-GRDH四分类队列，训练719、验证228、测试378条；本地解析产品规模不等于实际分类队列规模，也不宣称公开发行总规模。此前不同产品或类别口径的历史实验不纳入当前主表。\n\n"
    text += "AIS汇总记录18,766个非空MMSI及198条缺失MMSI；总库身份数不等于观测数。日期根据产品号恢复，并与本地产品注释端点核对；像素间距与空间分辨率不混写。预处理PNG不是原始复数SAR。\n"
    (ROOT / "DATASET_PROFILE_ZH.md").write_text(text, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
