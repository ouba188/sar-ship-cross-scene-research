"""Assemble read-only review evidence from existing quality outputs."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode

QUALITY_DIR = Path(__file__).resolve().parent.parent


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def integer(value):
    result = number(value)
    return int(result) if result is not None else None


def truth(value):
    return str(value).strip().lower() in {"true", "1", "1.0"}


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


class ReviewData:
    def __init__(self, quality_dir=QUALITY_DIR):
        self.quality_dir = Path(quality_dir)
        root = self.quality_dir
        attributes = {r["instance_id"]: r for r in read_csv(root / "ais_attribute_review.csv")}
        inventory = list(read_csv(root.parent / "obb_consistent_repair_20261016/inventory.csv"))
        experiment_ids = {r["instance_id"] for r in inventory}
        # Role row numbers belong to the experiment inventory, not the AIS sidecar.
        experiment_rows = {integer(r["global_row"]): r["instance_id"] for r in inventory}
        roles = defaultdict(set)
        for row in read_csv(root.parent / "obb_consistent_repair_20261016/roles.csv"):
            identity = experiment_rows.get(integer(row["global_row"]))
            if identity:
                roles[identity].add(row["split"] + ":" + row["role"])

        chips = {}
        total_manifest = 0
        for row in read_csv(root / "per_chip_quality.csv"):
            total_manifest += 1
            identity = row["instance_id"]
            if truth(row["multi_any_25"]) or truth(row["length_screen_any"]) or identity in attributes:
                if identity in chips:
                    raise ValueError("Duplicate candidate instance_id: " + identity)
                chips[identity] = row

        metadata = {r["instance_id"]: r for r in read_csv(root / "ais_metadata/ais_metadata_sidecar.csv")
                    if r["instance_id"] in chips}
        views = defaultdict(dict)
        for row in read_csv(root / "per_view_quality.csv"):
            if row["instance_id"] in chips:
                views[row["instance_id"]][row["pol"].lower()] = row
        detections = defaultdict(dict)
        with (root / "full/canvas512/detections.jsonl").open(encoding="utf-8-sig") as stream:
            for line in stream:
                row = json.loads(line)
                if row["instance_id"] in chips:
                    detections[row["instance_id"]][row["pol"].lower()] = finite_json(row["detections"])
        rechecks = {}
        for row in read_csv(root / "canvas_recheck_per_chip.csv"):
            rechecks[integer(row["global_row"])] = {
                key: truth(value) if value.lower() in {"true", "false"} else number(value)
                for key, value in row.items() if key != "global_row"
            }

        self.items = []
        self.details = {}
        self.by_mmsi = defaultdict(list)
        for identity, chip in chips.items():
            meta = metadata.get(identity, {})
            attr = attributes.get(identity, {})
            flags = []
            for flag, enabled in (
                ("multi", truth(chip["multi_any_25"])),
                ("length", truth(chip["length_screen_any"])),
                ("ais_conflict", truth(meta.get("vessel_static_conflict")) or truth(attr.get("vessel_static_conflict"))),
                ("ais_extreme", truth(attr.get("extreme_total_length_review"))),
                ("ais_limit", truth(attr.get("encoding_limit_pattern_review"))),
            ):
                if enabled:
                    flags.append(flag)
            item = dict(id=identity, global_row=integer(chip["global_row"]),
                        mmsi=chip["mmsi"], port=chip["port"], class_name=chip["class_name"],
                        flags=flags, in_experiment=identity in experiment_ids,
                        roles=sorted(roles[identity]), ais_length_m=number(meta.get("ais_length_m")),
                        ais_width_m=number(meta.get("ais_width_m")))
            self.items.append(item)
            detail = dict(item, product=chip["product"], ais={
                "length_m": item["ais_length_m"], "width_m": item["ais_width_m"],
                "scope": meta.get("ais_size_scope", ""),
                "variants": finite_json(json.loads(meta.get("vessel_static_variants_json") or "[]")),
                "source": meta.get("ais_size_source", ""), "line": integer(meta.get("ais_source_line")),
                "record_match": meta.get("ais_record_match_quality", ""),
            }, views={}, recheck=rechecks.get(item["global_row"]))
            for pol in ("vv", "vh"):
                view = views[identity][pol]
                detail["views"][pol] = {
                    "image_url": "/api/image?" + urlencode({"id": identity, "pol": pol}),
                    "width": integer(view["native_w"]), "height": integer(view["native_h"]),
                    "path": view["image"],
                    "anchor": [number(view["anchor_x"]), number(view["anchor_y"])],
                    "target_index": integer(view["target_index"]), "count": integer(view["n_25"]),
                    "length_m": number(view["target_length_m"]), "width_m": number(view["target_width_m"]),
                    "length_px": number(view["target_length_px"]), "width_px": number(view["target_width_px"]),
                    "length_ratio": number(view["length_ratio"]), "width_ratio": number(view["width_ratio"]),
                    "scale_status": view["scale_status"], "match_status": view["target_match_status"],
                    "detections": detections[identity][pol],
                }
            self.details[identity] = detail
            if item["mmsi"] and item["mmsi"].lower() != "nan":
                self.by_mmsi[item["mmsi"]].append(identity)
        self.counts = dict(total_manifest=total_manifest, candidates=len(self.items),
                           in_experiment=sum(item["in_experiment"] for item in self.items))

    def candidate(self, identity):
        detail = self.details[identity]
        keys = ("id", "global_row", "product", "port", "ais_length_m", "ais_width_m")
        related = [{key: self.details[other][key] for key in keys}
                   for other in self.by_mmsi.get(detail["mmsi"], []) if other != identity]
        return dict(detail, related=related)

    def source_text(self, identity):
        ais = self.details[identity]["ais"]
        path, line = ais["source"], ais["line"]
        heading = "Path: {}\nAIS source line: {}\n".format(path or "(not recorded)", line)
        if not path or line is None or line < 1:
            return 404, heading + "Unavailable: no recorded AIS source path/line.\n"
        try:
            selected = []
            found = False
            with Path(path).open(encoding="utf-8-sig", newline="") as stream:
                for current, text in enumerate(stream, 1):
                    if current > line + 2:
                        break
                    if current >= max(1, line - 2):
                        selected.append("{}{}: {}".format("> " if current == line else "  ", current, text))
                    if current == line:
                        found = True
            if not found:
                return 404, heading + "Unavailable: recorded line is beyond end of local file.\n"
            return 200, heading + "\n" + "".join(selected)
        except (OSError, UnicodeError) as exc:
            return 404, heading + "Unavailable: {}: {}\n".format(type(exc).__name__, exc)
