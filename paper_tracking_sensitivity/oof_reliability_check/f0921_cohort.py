"""Bounded F_0921 labels and source identity exclusion, without feature loading.

prepare_target(out) returns the retained manifest in clean-archive order.
prepare_source_split(out, manifest) returns positional source indices. It uses
ALL clean target identities, including rows excluded by the fine-label bridge.
Product exclusion conservatively includes every acquisition candidate in each
clean tile; the historical tile lookup contains only its first acquisition.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from opensarship2_labels import (
    AMBIGUOUS_LABELS, CLASSES, CLASS_TO_ID, map_label, normalize_label,
    summarize_mapping,
)


ROOT = Path("E:/Hermes/paper_tracking")
INDEX = ROOT / "f0921_chips/index.csv"
CLEAN_ARCHIVES = (
    ROOT / "f0921_feats/f0921_feats_VV_clean.npz",
    ROOT / "f0921_feats/f0921_feats_VH_clean.npz",
    ROOT / "f0921_ssl_align/f0921_ssl_clean.npz",
)
ACQUISITIONS = ROOT / "f0921_tile_acq.json"
TILE_PRODUCTS = ROOT / "f0921_tile_products.json"
OLD_OVERLAP = ROOT / "f0921_pool_overlap.json"
POLICY = (
    "Purge source rows matching any nonempty MMSI from ALL clean target rows "
    "before label filtering, or any Sentinel product acquisition candidate "
    "in their tiles. Candidates match satellite, acquisition start and product "
    "suffix code. No port/day approximation or class quotas. Existing frozen "
    "encoders are not retrained; exclusion applies to source heads/gate."
)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _iter_acquisitions(path):
    """Decode the large JSON array one record at a time with a 64 KiB buffer."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer = stream.read(65536).lstrip()
        if not buffer.startswith("["):
            raise ValueError("Acquisition metadata must be a JSON array")
        buffer = buffer[1:]
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            try:
                record, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = stream.read(65536)
                if not chunk:
                    raise
                buffer += chunk
                continue
            yield record
            buffer = buffer[end:]


def load_clean_rows():
    """Read only the three NPZ row members, never their feature arrays."""
    reference = None
    for path in CLEAN_ARCHIVES:
        with np.load(path, allow_pickle=False) as archive:
            rows = archive["row"]
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError(f"Expected a one-dimensional integer row key: {path}")
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"Duplicate clean row keys: {path}")
        if reference is None:
            reference = rows
        elif not np.array_equal(reference, rows):
            raise ValueError(f"VV/VH/SSL clean row alignment differs: {path}")
    return reference


def join_clean_rows(index, rows):
    """Join chip IDs, not CSV positions; retain the archive's feature order."""
    keys = pd.DataFrame({"row": rows, "clean_feature_row": np.arange(len(rows))})
    clean = keys.merge(index, on="row", how="left", sort=False,
                       validate="one_to_one", indicator=True)
    if not clean["_merge"].eq("both").all():
        missing = clean.loc[clean["_merge"].ne("both"), "row"].tolist()
        raise ValueError(f"Clean row keys missing from chip index: {missing[:10]}")
    return clean.drop(columns="_merge").sort_values("clean_feature_row").reset_index(drop=True)


def map_f0921_labels(frame):
    """Map class_raw alone; preserve historical class8 and every original field."""
    result = frame.copy()
    result["label_field"] = "class_raw"
    result["label_value"] = result.class_raw.fillna("")
    result["label_normalized"] = result.label_value.map(normalize_label)
    result["class_name"] = result.label_value.map(map_label)
    result["class_id"] = pd.array(result.class_name.map(CLASS_TO_ID), dtype="Int64")
    result["label_included"] = result.class_name.notna()
    result["label_mapping_reason"] = "unsupported_label"
    normalized = result.label_normalized
    result.loc[normalized.isin(AMBIGUOUS_LABELS), "label_mapping_reason"] = "ambiguous_label"
    coarse = normalized.isin({"cargo", "cargo_ship", "cargoship", "tanker", "tank_ship"})
    coarse |= normalized.str.startswith(("cargo_haz", "tanker_haz"))
    result.loc[coarse, "label_mapping_reason"] = "coarse_label"
    result.loc[normalized.eq(""), "label_mapping_reason"] = "missing_label"
    result.loc[result.label_included, "label_mapping_reason"] = "mapped"
    return result


def prepare_target(out):
    """Write all-clean mapping/summary and the compatible target manifest.

    row and target_manifest_row are original chip-array indices;
    clean_feature_row is the shared VV/VH/SSL clean-archive position.
    """
    out = Path(out)
    rows = load_clean_rows()
    index = pd.read_csv(INDEX, dtype={"row": np.int64, "mmsi": str}, keep_default_na=False)
    mapped = map_f0921_labels(join_clean_rows(index, rows))
    mapped["sample_id"] = "f0921_" + mapped.row.astype(str)
    mapped["target_manifest_row"] = mapped.row
    target = mapped.loc[mapped.label_included].reset_index(drop=True)
    summary = summarize_mapping(mapped)
    summary.update(
        index_rows=len(index), clean_rows=len(mapped), retained_rows=len(target),
        source_output_classes=list(CLASSES), class_quotas_applied=False,
        clean_policy="Existing chip-level clean archives: land_px_224 == 0",
        provenance={"index": str(INDEX), "clean_archives": [str(p) for p in CLEAN_ARCHIVES],
                    "archive_key": "row", "label_source": "class_raw",
                    "taxonomy": "opensarship2_labels.py:CLASSES/map_label",
                    "historical_label_only": "class8",
                    "row_alignment": "Unique chip IDs; identical VV/VH/SSL row arrays"},
    )
    out.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(out / "label_mapping.csv", index=False)
    _write_json(out / "label_mapping_summary.json", summary)
    target.to_csv(out / "target_manifest.csv", index=False)
    return target


def resolve_product_overlap(scenes, source_products, acquisitions, tile_products):
    """Return source product IDs supported by acquisition candidates in clean tiles.

    A candidate need not be the acquisition of a retained chip. Including all
    tile candidates is intentional: the historical one-product map cannot
    establish chip-level product identity. No new product catalog is needed.
    """
    scenes = set(scenes)
    candidates = {
        (r["tile"], r["sensor"], r["t"], r["code"])
        for r in acquisitions if r["tile"] in scenes
    }
    covered = {r[0] for r in candidates}
    missing = scenes - covered
    if missing:
        raise ValueError(f"Missing acquisition provenance for {len(missing)} clean tiles: "
                         f"{sorted(missing)[:3]}")
    products = []
    for product in sorted(set(source_products) - {""}):
        parts = product.removesuffix(".SAFE").split("_")
        if len(parts) != 9 or not parts[0].startswith("S1"):
            raise ValueError(f"Source product is not a Sentinel product ID: {product}")
        products.append((product, parts[0], parts[4], parts[-1]))
    catalog = pd.DataFrame(products, columns=["product", "sensor", "t", "code"])
    candidates = pd.DataFrame(sorted(candidates), columns=["scene", "sensor", "t", "code"])
    shared = candidates.merge(catalog, on=["sensor", "t", "code"], how="inner")
    overlap = sorted(shared["product"].unique())
    historical = {v["product"].removesuffix(".SAFE") for k, v in tile_products.items() if k in scenes}
    first_only = [p for p in overlap if p.removesuffix(".SAFE") in historical]
    provenance = {
        "method": "Conservative clean-tile acquisition candidates joined to source product IDs",
        "join_fields": ["sensor", "t", "code"],
        "chip_level_product_assignment": False,
        "clean_tiles": len(scenes), "candidate_acquisitions": len(candidates),
        "multi_product_clean_tiles": int((candidates.groupby("scene").size() > 1).sum()),
        "source_products_considered": len(catalog), "shared_products": len(overlap),
        "historical_first_product_shared": len(first_only),
        "additional_products_beyond_first": sorted(set(overlap) - set(first_only)),
        "shared_product_scenes": {
            p: sorted(group.scene.unique()) for p, group in shared.groupby("product")
        },
        "acquisitions": str(ACQUISITIONS), "historical_first_product_map": str(TILE_PRODUCTS),
        "historical_map_generator": str(ROOT.parent / "scripts/f0921_acq_map.py"),
        "historical_map_limitation": "Generator uses tiles.setdefault: first acquisition only",
        "old_overlap_report": str(OLD_OVERLAP),
        "old_overlap_not_used": "Aggregate four-port/day counts, not product identities",
    }
    return np.asarray(overlap, dtype=str), provenance


def prepare_source_split(out, manifest):
    """Write source_split.npz/json and return source indices in original order.

    manifest is the caller's dict of aligned labels, ports, mmsi and product
    arrays. prepare_target(out) must run first. No features or models are read.
    """
    out = Path(out)
    clean = pd.read_csv(out / "label_mapping.csv", usecols=["row", "mmsi", "scene"],
                        dtype={"mmsi": str, "scene": str}, keep_default_na=False)
    labels = np.asarray(manifest["labels"], dtype=np.int64)
    mmsi = np.asarray(manifest["mmsi"], dtype=str)
    products = np.asarray(manifest["product"], dtype=str)
    if any(len(manifest[key]) != len(labels) for key in ("ports", "mmsi", "product")):
        raise ValueError("Source manifest columns have different lengths")
    if not np.isin(labels, np.arange(len(CLASSES))).all():
        raise ValueError("Source labels must use canonical IDs 0..7")
    tile_products = json.loads(TILE_PRODUCTS.read_text(encoding="utf-8"))
    shared_products, provenance = resolve_product_overlap(
        clean.scene, products, _iter_acquisitions(ACQUISITIONS), tile_products,
    )
    target_mmsi = np.asarray(sorted(set(clean.mmsi) - {""}), dtype=str)
    mmsi_hit = np.isin(mmsi, target_mmsi)
    product_hit = np.isin(products, shared_products)
    excluded = mmsi_hit | product_hit
    retained = np.flatnonzero(~excluded)
    counts = np.bincount(labels[retained], minlength=len(CLASSES))
    summary = {
        "source_rows_before": len(labels), "source_rows": len(retained),
        "excluded_source_rows": int(excluded.sum()), "target_clean_rows": len(clean),
        "overlaps": {
            "mmsi": {"source_rows": int(mmsi_hit.sum()),
                     "shared_ids": len(np.unique(mmsi[mmsi_hit]))},
            "product": {"source_rows": int(product_hit.sum()), "shared_ids": len(shared_products)},
            "both_source_rows": int((mmsi_hit & product_hit).sum()),
        },
        "source_class_counts": {name: int(counts[i]) for i, name in enumerate(CLASSES)},
        "missing_source_classes": [name for i, name in enumerate(CLASSES) if counts[i] == 0],
        "source_port_counts": pd.Series(np.asarray(manifest["ports"])[retained]).value_counts().to_dict(),
        "policy": POLICY, "provenance": provenance,
    }
    np.savez_compressed(
        out / "source_split.npz", source_rows=retained,
        excluded_source_rows=np.flatnonzero(excluded),
        mmsi_excluded_source_rows=np.flatnonzero(mmsi_hit),
        product_excluded_source_rows=np.flatnonzero(product_hit),
        target_clean_rows=clean.row.to_numpy(dtype=np.int64),
        shared_target_mmsi=np.unique(mmsi[mmsi_hit]), shared_products=shared_products,
        source_class_counts=counts, class_names=np.asarray(CLASSES),
        policy=np.asarray(POLICY), metadata_json=np.asarray(json.dumps(summary)),
    )
    _write_json(out / "source_split.json", summary)
    return retained
