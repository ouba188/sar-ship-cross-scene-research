"""Select traceable same-vessel SAR chips for qualitative comparison."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path("E:/临时会话")
OUT = Path(__file__).resolve().parent
PORTS = ROOT / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv"
F_MANIFEST = Path("E:/Hermes/paper_tracking/sensitivity/f0921_transfer_20261007/target_manifest.csv")
F_CHIPS = Path("E:/Hermes/paper_tracking/f0921_chips/F0921_VV.npy")
F_INDEX = Path("E:/Hermes/paper_tracking/f0921_chips/index.csv")
OSS1 = Path("E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/target_manifest.csv")
OSS2 = Path("E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/target_manifest.csv")
PRODUCT_PAIRS = ROOT / "imaging_coupling_20261013/product_pairs_frozen.csv"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def identity(row, field):
    value = row[field].strip()
    return value if len(value) == 9 and value[0] in "234567" and value.isdecimal() else ""


def groups(rows, field):
    result = defaultdict(list)
    for row in rows:
        key = identity(row, field)
        if key:
            result[key].append(row)
    return result


def date(product):
    parts = product.split("_")
    return next((p[:8] for p in parts if len(p) >= 15 and p[:8].isdigit() and p[8] == "T"), "")


def row_record(dataset, row):
    if dataset == "24port":
        return dict(dataset=dataset, mmsi=row["mmsi"], class_name=row["class_name"],
                    sample_id=row["instance_id"], port=row["port"], product=row["product"],
                    date=date(row["product"]), image=row["image_relpath_vv"], array_row="",
                    label_raw=row["class_name"], legacy_index_class8="")
    if dataset == "F_0921":
        return dict(dataset=dataset, mmsi=row["mmsi"], class_name=row["class_name"],
                    sample_id=row["sample_id"], port="unverified", product=row["scene"],
                    date=row["scene"].split("_")[3][:8], image=str(F_CHIPS), array_row=row["row"],
                    label_raw=row["class_raw"], legacy_index_class8="")
    return dict(dataset=dataset, mmsi=row["mmsi_or_ship_id"], class_name=row["class_name"],
                sample_id=row["sample_id"], port="unavailable", product=row["scene_id"],
                date=date(row["scene_id"]), image=row["chip_path_vv"], array_row="",
                label_raw=row["label_value"], legacy_index_class8="")


def consistent(rows, label="class_name"):
    return len({r[label] for r in rows}) == 1


def choose_port_examples(p):
    found = []
    for class_name in ("bulk_carrier", "general_cargo", "container_ship", "product_chemical_tanker"):
        options = []
        for mmsi, rows in p.items():
            valid = [r for r in rows if r["class_name"] == class_name and Path(r["image_relpath_vv"]).is_file()]
            ports = {r["port"] for r in valid}
            if len(ports) >= 3 and consistent(rows):
                options.append((len(ports), len({r["product"] for r in valid}), mmsi, valid))
        if not options:
            continue
        _, _, mmsi, rows = max(options, key=lambda x: (x[0], x[1], x[2]))
        by_port = {}
        for row in sorted(rows, key=lambda r: (r["port"], r["product"])):
            by_port.setdefault(row["port"], row)
        samples = list(by_port.values())
        picks = [samples[0], samples[len(samples) // 2], samples[-1]]
        found.append((f"{class_name} | MMSI {mmsi}", [row_record("24port", x) for x in picks]))
    return found


def choose_cross_examples(a, b, a_name, b_name, class_order, max_rows=4):
    found = []
    for class_name in class_order:
        options = []
        for mmsi in a.keys() & b.keys():
            aa = [r for r in a[mmsi] if r["class_name"] == class_name]
            bb = [r for r in b[mmsi] if r["class_name"] == class_name]
            if not aa or not bb or not consistent(a[mmsi] + b[mmsi]):
                continue
            if a_name != "F_0921":
                aa = [r for r in aa if Path(r["image_relpath_vv"] if a_name == "24port" else r["chip_path_vv"]).is_file()]
            if b_name != "F_0921":
                bb = [r for r in bb if Path(r["image_relpath_vv"] if b_name == "24port" else r["chip_path_vv"]).is_file()]
            if not aa or not bb:
                continue
            ra = [row_record(a_name, r) for r in aa]
            rb = [row_record(b_name, r) for r in bb]
            distinct_dates = len({r["date"] for r in ra + rb})
            options.append((distinct_dates, min(len(ra), 4) + min(len(rb), 4), mmsi, ra, rb))
        if not options:
            continue
        _, _, mmsi, ra, rb = max(options, key=lambda x: (x[0], x[1], x[2]))
        ra.sort(key=lambda r: (r["date"], r["sample_id"]))
        rb.sort(key=lambda r: (r["date"], r["sample_id"]))
        picks = [ra[0], ra[-1], rb[0], rb[-1]]
        # The same source record is shown once if a vessel has only one observation.
        picks = list({(r["dataset"], r["sample_id"]): r for r in picks}.values())
        found.append((f"{class_name} | MMSI {mmsi}", picks))
        if len(found) == max_rows:
            break
    return found


def choose_product_examples(oss2):
    pairs = read(PRODUCT_PAIRS)
    found = []
    by_id = {r["sample_id"]: r for rows in oss2.values() for r in rows}
    for class_name in ("fishing_vessel", "bulk_carrier", "container_ship", "product_chemical_tanker"):
        options = []
        for pair in pairs:
            if pair["class_name_grdh"] != class_name or pair["same_label"] != "True" or pair["same_sar_datetime"] != "True":
                continue
            ga = by_id.get(pair["sample_id_grdh"])
            sa = by_id.get(pair["sample_id_slc"])
            if not ga or not sa or not Path(ga["chip_path_vv"]).is_file() or not Path(sa["chip_path_vv"]).is_file():
                continue
            length = min(float(ga["visible_length_px"] or 0), float(sa["visible_length_px"] or 0))
            options.append((length, pair["identity"], ga, sa))
        if not options:
            continue
        _, mmsi, ga, sa = max(options, key=lambda x: (x[0], x[1]))
        found.append((f"{class_name} | MMSI {mmsi}", [row_record("OSS2", ga), row_record("OSS2", sa)]))
    return found


def choose_label_conflicts(ports, f0921):
    requested = (("tug_towing", "bulk_carrier"),
                 ("bulk_carrier", "general_cargo"),
                 ("product_chemical_tanker", "crude_oil_tanker"))
    found = []
    for port_label, f_label in requested:
        options = []
        for mmsi in ports.keys() & f0921.keys():
            pa, fa = ports[mmsi], f0921[mmsi]
            if {r["class_name"] for r in pa} != {port_label} or {r["class_name"] for r in fa} != {f_label}:
                continue
            valid = [r for r in pa if Path(r["image_relpath_vv"]).is_file()]
            if valid:
                options.append((len(valid), len(fa), mmsi, valid, fa))
        if not options:
            continue
        _, _, mmsi, pa, fa = max(options, key=lambda x: (x[0], x[1], x[2]))
        pa.sort(key=lambda r: (date(r["product"]), r["instance_id"]))
        fa.sort(key=lambda r: (r["scene"], r["sample_id"]))
        found.append((f"MMSI {mmsi}", [row_record("24port", pa[0]), row_record("F_0921", fa[0])]))
    return found


def choose_all_axes(ports, f0921, oss1, oss2):
    pairs = read(PRODUCT_PAIRS)
    by_id = {r["sample_id"]: r for rows in oss2.values() for r in rows}
    found = []
    for mmsi in sorted(ports.keys() & f0921.keys() & oss1.keys() & oss2.keys()):
        all_rows = ports[mmsi] + f0921[mmsi] + oss1[mmsi] + oss2[mmsi]
        if not consistent(all_rows):
            continue
        port_rows = [r for r in ports[mmsi] if Path(r["image_relpath_vv"]).is_file()]
        by_port = {}
        for row in sorted(port_rows, key=lambda r: (r["port"], r["product"])):
            by_port.setdefault(row["port"], row)
        if len(by_port) < 2:
            continue
        oss1_rows = [r for r in oss1[mmsi] if Path(r["chip_path_vv"]).is_file()]
        if not oss1_rows:
            continue
        pair = next((r for r in pairs if r["identity"] == mmsi and r["same_label"] == "True"
                     and r["same_sar_datetime"] == "True"
                     and r["sample_id_grdh"] in by_id and r["sample_id_slc"] in by_id
                     and Path(by_id[r["sample_id_grdh"]]["chip_path_vv"]).is_file()
                     and Path(by_id[r["sample_id_slc"]]["chip_path_vv"]).is_file()), None)
        if not pair:
            continue
        ps = list(by_port.values())
        f_row = max(f0921[mmsi], key=lambda r: (r["scene"], r["sample_id"]))
        o1_row = min(oss1_rows, key=lambda r: (r["scene_id"], r["sample_id"]))
        chips = [row_record("24port", ps[0]), row_record("24port", ps[-1]),
                 row_record("F_0921", f_row), row_record("OSS1", o1_row),
                 row_record("OSS2", by_id[pair["sample_id_grdh"]]),
                 row_record("OSS2", by_id[pair["sample_id_slc"]])]
        found.append((f"{all_rows[0]['class_name']} | MMSI {mmsi}", chips))
    return found


def image_data(row, f_array):
    if row["array_row"]:
        return np.asarray(f_array[int(row["array_row"])])
    with Image.open(row["image"]) as im:
        return np.asarray(im.convert("L"))


def render(name, title, cases, f_array, max_cols, show_chip_class=False):
    fig, axes = plt.subplots(len(cases), max_cols, figsize=(3.25 * max_cols, 3.0 * len(cases)), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    records = []
    for i, (label, chips) in enumerate(cases):
        for j in range(max_cols):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j >= len(chips):
                ax.axis("off")
                continue
            chip = chips[j]
            arr = image_data(chip, f_array)
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            caption = f"{chip['dataset']}  {chip['port']}\n{chip['date']}  {('SLC' if '_SLC_' in chip['product'] else 'GRDH')}"
            if show_chip_class:
                caption += f"\nlabel: {chip['class_name']}"
            ax.set_title(caption, fontsize=8)
            if j == 0:
                ax.set_ylabel(label.replace(" | ", "\n"), fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
            records.append(dict(panel=name, row=i + 1, column=j + 1, **chip,
                                width=arr.shape[1], height=arr.shape[0],
                                pixel_min=int(arr.min()), pixel_max=int(arr.max())))
    fig.tight_layout(rect=(0.06, 0.02, 1, 0.975))
    fig.savefig(OUT / f"{name}.png", dpi=180, facecolor="white")
    plt.close(fig)
    return records


def main():
    p = groups(read(PORTS), "mmsi")
    f = groups(read(F_MANIFEST), "mmsi")
    o1 = groups(read(OSS1), "mmsi_or_ship_id")
    o2 = groups(read(OSS2), "mmsi_or_ship_id")
    f_array = np.load(F_CHIPS, mmap_mode="r")
    assert f_array.shape[0] == 385650 and f_array.dtype == np.uint8
    sets = [
        ("all_axes", "Same recorded MMSI across ports, datasets, acquisitions and products", choose_all_axes(p, f, o1, o2), 6),
        ("ports24", "Same recorded MMSI across verified ports and acquisitions", choose_port_examples(p), 3),
        ("port_f0921", "Same recorded MMSI in 24-port and F_0921 chip collections", choose_cross_examples(p, f, "24port", "F_0921", ("bulk_carrier", "container_ship", "general_cargo", "product_chemical_tanker")), 4),
        ("oss1_oss2", "Same recorded MMSI in OpenSARShip 1.0 and 2.0", choose_cross_examples(o1, o2, "OSS1", "OSS2", ("bulk_carrier", "container_ship", "general_cargo", "product_chemical_tanker")), 4),
        ("grdh_slc", "Same recorded MMSI and SAR time: GRDH versus SLC (OSS2)", choose_product_examples(o2), 2),
        ("label_conflicts", "Same recorded MMSI, conflicting dataset labels", choose_label_conflicts(p, f), 2),
    ]
    selected_f = {int(chip["array_row"]): chip for _, _, cases, _ in sets for _, chips in cases
                  for chip in chips if chip["dataset"] == "F_0921"}
    checked = set()
    with F_INDEX.open(encoding="utf-8-sig", newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream)):
            if row_number not in selected_f:
                continue
            chip = selected_f[row_number]
            assert row["row"] == str(row_number)
            assert row["mmsi"] == chip["mmsi"]
            assert row["scene"] == chip["product"]
            assert row["class_raw"] == chip["label_raw"]
            chip["legacy_index_class8"] = row["class8"]
            checked.add(row_number)
    assert checked == set(selected_f)
    print("F_0921 array/index/manifest rows verified:", len(checked),
          "legacy labels disagree:", sum(chip["legacy_index_class8"] != chip["class_name"]
                                          for chip in selected_f.values()), flush=True)
    records = []
    for name, title, cases, cols in sets:
        if cases:
            records.extend(render(name, title, cases, f_array, cols, show_chip_class=name == "label_conflicts"))
            print(name, len(cases), "vessels", sum(len(x[1]) for x in cases), "chips", flush=True)
        else:
            print(name, "NO MATCH", flush=True)
    with (OUT / "chip_provenance.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
