"""Index the actual query/reference chips shown or discussed in this manuscript."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    rows = []

    def add(location, role, case_id, pol, path, provenance, original=""):
        path = str(path or "")
        rows.append(dict(location=location, role=role, case_id=str(case_id),
                         polarization=pol.upper(), chip_path=path,
                         status=("present" if Path(path).is_file() else "missing_on_disk")
                         if path else "unavailable_in_experiment",
                         original_tiff=str(original or ""),
                         original_tiff_exists=bool(original and Path(original).is_file()),
                         provenance=str(provenance)))

    source = WORK / "paper_draft_20261027/figure_sources.csv"
    for row in read_csv(source):
        if row["main_figure"] == "True":
            add("图1", "同船重复观测", row["obs_id"], row["pol"], row["source_path"], source)

    directory = WORK / "paper_evidence_20261027/full_model_interventions_20261027"
    source = directory / "case_records.json"
    cases = read_json(source)
    refs = {int(row["obs_id"]): row for row in read_csv(directory / "reference_bank.csv")}
    for oid in (10488, 26628):
        case = next(row for row in cases if row["obs_id"] == oid)
        location = "图2及第3.4节" if oid == 10488 else "第3.4节误伤例"
        for pol in ("vv", "vh"):
            add(location, "查询", oid, pol, case["source_image_" + pol], source)
        if oid == 10488:
            for ref_id in case["reference_ids"]:
                for pol in ("vv", "vh"):
                    add("图2", "实际参考", ref_id, pol, refs[ref_id]["image_" + pol], directory / "reference_bank.csv")

    galleries = [
        ("图8", WORK / "m1_guided_m2_20261106/cases.json", 1142),
        ("图9", WORK / "m3_frequency_increment_20261028/source_hard_task_20261101/oss2/internal_contribution/writer_review/cases.json", 735),
    ]
    for location, source, index in galleries:
        case = next(row for row in read_json(source)["cases"] if row["index"] == index)
        for entry in case["records"]:
            record = entry["record"]
            for pol in ("vv", "vh"):
                view = record.get(pol) or {}
                add(location, entry["role"], record["id"], pol, view.get("image"), source,
                    record.get("source_paths", {}).get(pol))

    source = WORK / "m1_coverage_20261101/manifest.json"
    records = {r["id"]: r for r in read_json(source)}
    mentions = [("第6.2节误伤例", "002394"), ("第6.4节误伤例", "000295"),
                ("第6.6节纠错例", "002950"), ("第6.6节误伤例", "000694"),
                ("第6.7节保护例", "002251"), ("第6.7节误伤例", "002146")]
    for location, suffix in mentions:
        record = records["pub2_opensarship2_" + suffix]
        for pol in ("vv", "vh"):
            add(location, "查询", record["id"], pol, (record.get(pol) or {}).get("image"), source,
                record.get("source_paths", {}).get(pol))

    with (ROOT / "chip_locations.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    present = [r for r in rows if r["status"] == "present"]
    missing = [r for r in rows if r["status"] == "missing_on_disk"]
    lines = ["# 正文船舶chip本地位置", "", "更新：2026-11-09。只整理来源，不改图像、不重训。",
             "", f"共 {len(rows)} 条案例—极化记录：{len(present)} 条输入文件在本地，"
             f"{len(missing)} 条有路径但未找到文件，{len(rows)-len(present)-len(missing)} 条在实际实验输入中不可用。",
             "", "`chip_locations.csv` 给出完整路径、极化、角色、来源索引及可获得的原始TIFF路径。",
             "图3—7是作者提供的算法示意图，不属于独立实验案例；其中嵌入的示意chip未据此反推原始样本身份。",
             "缺失极化保留实验中的不可用状态，不用其他图像补成一对。以下链接指向实际输入PNG；TIFF追溯路径见CSV。", ""]
    last = None
    for row in rows:
        if row["location"] != last:
            lines += ["## " + row["location"], ""]
            last = row["location"]
        path = row["chip_path"].replace("\\", "/")
        label = f"{row['case_id']} · {row['role']} · {row['polarization']}"
        if path:
            lines.append(f"- [{label}](<{path}>)")
        else:
            raw = row["original_tiff"].replace("\\", "/")
            suffix = f" 可追溯[原始TIFF](<{raw}>)，不等同于实验已使用。" if row["original_tiff_exists"] else ""
            lines.append(f"- {label}：该极化未进入实验输入。" + suffix)
    (ROOT / "CHIP_LOCATIONS_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(records=len(rows), present=len(present), missing=missing,
                          unavailable=len(rows)-len(present)-len(missing)), ensure_ascii=False))


if __name__ == "__main__":
    main()
