#!/usr/bin/env python3
"""Build sea-ship redetection previews and optionally publish them immediately."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window
from ultralytics import YOLO


CLASSES = [
    "bulk_carrier", "cargo_coarse", "container_ship", "crude_oil_tanker", "dredger", "fishing_vessel",
    "general_cargo", "heavy_load_carrier", "lpg_lng_tanker", "offshore_supply", "passenger_ship",
    "pilot_port_tender", "pleasure_craft", "product_chemical_tanker", "reefer_cargo", "ro_ro_vehicle_carrier",
    "sailing_vessel", "tanker_coarse", "tug_towing",
]
ALGORITHM_VERSION = "high_recall_v5_original_json_only_obb_geometry_v1"


def configure_rasterio_data() -> None:
    package_root = Path(rasterio.__file__).resolve().parent
    candidates = {
        "PROJ_DATA": package_root / "proj_data",
        "PROJ_LIB": package_root / "proj_data",
        "GDAL_DATA": package_root / "gdal_data",
    }
    for name, fallback in candidates.items():
        current = Path(os.environ[name]) if os.environ.get(name) else None
        if (current is None or not current.is_dir()) and fallback.is_dir():
            os.environ[name] = str(fallback)


configure_rasterio_data()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-root", type=Path)
    source.add_argument("--tile-manifest", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-overviews", action="store_true")
    parser.add_argument("--publish-active", action="store_true")
    parser.add_argument("--overwrite-manual", action="store_true")
    parser.add_argument("--skip-backup", action="store_true")
    parser.add_argument("--worldcover-cache", type=Path)
    parser.add_argument("--worldcover-timeout", type=float, default=20.0)
    options = parser.parse_args()
    if options.publish_active and not (options.overwrite_manual and options.skip_backup):
        parser.error("--publish-active requires --overwrite-manual and --skip-backup")
    if (options.overwrite_manual or options.skip_backup) and not options.publish_active:
        parser.error("--overwrite-manual and --skip-backup require --publish-active")
    if options.worldcover_timeout <= 0:
        parser.error("--worldcover-timeout must be positive")
    return options


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest_model(path: Path) -> str:
    if path.is_file():
        return digest(path)
    if not path.is_dir():
        raise FileNotFoundError(f"model path does not exist: {path}")
    value = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"model directory is empty: {path}")
    for item in files:
        value.update(item.relative_to(path).as_posix().encode("utf-8"))
        value.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def validate_label_payload(payload: dict, tif: Path, image_width: int, image_height: int) -> None:
    if payload.get("image") != tif.name or payload.get("classes") != CLASSES:
        raise ValueError(f"invalid published label header: {tif}")
    for index, obj in enumerate(payload.get("objects") or []):
        cls_id = obj.get("cls_id")
        points = np.asarray(obj.get("pts"), dtype=np.float64)
        if not isinstance(cls_id, int) or not 0 <= cls_id < len(CLASSES):
            raise ValueError(f"invalid class ID at object {index}: {tif}")
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError(f"invalid geometry at object {index}: {tif}")
        if ((points[:, 0] < 0) | (points[:, 0] > image_width)).any():
            raise ValueError(f"x coordinate outside image at object {index}: {tif}")
        if ((points[:, 1] < 0) | (points[:, 1] > image_height)).any():
            raise ValueError(f"y coordinate outside image at object {index}: {tif}")


def publish_active_label(
    json_path: Path,
    tif: Path,
    payload: dict,
    source_json_sha256: str,
    image_width: int,
    image_height: int,
    overwrite_manual: bool,
) -> dict:
    validate_label_payload(payload, tif, image_width, image_height)
    before_sha256 = digest(json_path)
    if before_sha256 != source_json_sha256 and not overwrite_manual:
        raise RuntimeError(f"active JSON changed during processing: {json_path}")

    classes_path = tif.parent / "classes.txt"
    classes_before_sha256 = digest(classes_path) if classes_path.is_file() else ""
    expected_classes = "\n".join(CLASSES) + "\n"
    if not classes_path.is_file() or classes_path.read_text(encoding="utf-8-sig").splitlines() != CLASSES:
        atomic_text(classes_path, expected_classes)
    if classes_path.read_text(encoding="utf-8-sig").splitlines() != CLASSES:
        raise RuntimeError(f"failed to synchronize locked classes: {classes_path}")

    atomic_json(json_path, payload)
    after_sha256 = digest(json_path)
    validate_label_payload(
        json.loads(json_path.read_text(encoding="utf-8-sig")), tif, image_width, image_height
    )
    return {
        "published_active": True,
        "active_before_sha256": before_sha256,
        "active_after_sha256": after_sha256,
        "classes_txt": str(classes_path),
        "classes_before_sha256": classes_before_sha256,
        "classes_after_sha256": digest(classes_path),
        "publish_status": "published_manual_overwrite" if before_sha256 != source_json_sha256 else "published",
        "backup_created": False,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def starts(length: int, size: int, overlap: float) -> list[int]:
    if length <= size:
        return [0]
    stride = max(1, int(round(size * (1.0 - overlap))))
    values = list(range(0, length - size + 1, stride))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def rotated_points(cx: float, cy: float, width: float, height: float, angle: float) -> np.ndarray:
    theta = math.radians(angle)
    cosine, sine = math.cos(theta), math.sin(theta)
    corners = np.array([
        [-width / 2, -height / 2], [width / 2, -height / 2],
        [width / 2, height / 2], [-width / 2, height / 2],
    ], dtype=np.float32)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return corners @ rotation.T + np.array([cx, cy], dtype=np.float32)


def object_box(obj: dict) -> tuple[float, float, float, float, float]:
    raw = obj.get("cxcywh_angle")
    if isinstance(raw, list) and len(raw) >= 5:
        return tuple(float(value) for value in raw[:5])
    points = np.asarray(obj["pts"], dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    side_a = float(np.linalg.norm(points[1] - points[0]))
    side_b = float(np.linalg.norm(points[2] - points[1]))
    return float(center[0]), float(center[1]), side_a, side_b, 0.0


def axis_aligned_box(detection: dict, padding: float = 1.25) -> tuple[float, float, float, float, float]:
    theta = math.radians(float(detection["angle"]))
    cosine, sine = abs(math.cos(theta)), abs(math.sin(theta))
    width = max(8.0, (cosine * float(detection["width"]) + sine * float(detection["height"])) * padding)
    height = max(8.0, (sine * float(detection["width"]) + cosine * float(detection["height"])) * padding)
    return detection["cx"], detection["cy"], width, height, 0.0


def clamp_axis_box(box: tuple[float, float, float, float, float], image_width: int,
                   image_height: int) -> tuple[float, float, float, float, float]:
    cx, cy, width, height, angle = box
    width = max(1.0, min(width, 2.0 * cx, 2.0 * (image_width - cx)))
    height = max(1.0, min(height, 2.0 * cy, 2.0 * (image_height - cy)))
    return cx, cy, width, height, angle


def box_iou(first: dict, second: dict) -> float:
    def bounds(item: dict) -> tuple[float, float, float, float]:
        theta = math.radians(float(item["angle"]))
        cosine, sine = abs(math.cos(theta)), abs(math.sin(theta))
        half_width = float(item["width"]) / 2.0
        half_height = float(item["height"]) / 2.0
        extent_x = cosine * half_width + sine * half_height
        extent_y = sine * half_width + cosine * half_height
        cx, cy = float(item["cx"]), float(item["cy"])
        return cx - extent_x, cy - extent_y, cx + extent_x, cy + extent_y

    ax0, ay0, ax1, ay1 = bounds(first)
    bx0, by0, bx1, by1 = bounds(second)
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_detections(detections: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for candidate in sorted(detections, key=lambda item: item["conf"], reverse=True):
        duplicate = False
        for current in kept:
            distance = math.hypot(candidate["cx"] - current["cx"], candidate["cy"] - current["cy"])
            short_side = min(candidate["width"], candidate["height"], current["width"], current["height"])
            if distance <= max(3.0, 0.5 * short_side) or box_iou(candidate, current) >= 0.25:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def detect_ships(model: YOLO, gray: np.ndarray, crop_size: int, overlap: float,
                 confidence: float, device: str) -> list[dict]:
    height, width = gray.shape
    positions = [(x, y) for y in starts(height, crop_size, overlap) for x in starts(width, crop_size, overlap)]
    detections: list[dict] = []
    for offset in range(0, len(positions), 8):
        batch_positions = positions[offset:offset + 8]
        crops = []
        for x, y in batch_positions:
            patch = gray[y:min(y + crop_size, height), x:min(x + crop_size, width)]
            canvas = np.zeros((crop_size, crop_size), dtype=np.uint8)
            canvas[:patch.shape[0], :patch.shape[1]] = patch
            crops.append(cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB))
        results = model.predict(
            source=crops, imgsz=crop_size, conf=confidence, iou=0.30, device=device,
            verbose=False, save=False,
        )
        for (x, y), result in zip(batch_positions, results):
            obb = getattr(result, "obb", None)
            if obb is None or len(obb) == 0:
                continue
            xywhr = obb.xywhr.detach().cpu().numpy()
            scores = obb.conf.detach().cpu().numpy()
            for values, score in zip(xywhr, scores):
                cx, cy, box_width, box_height, angle_rad = (float(value) for value in values[:5])
                cx += x
                cy += y
                if not (0 <= cx < width and 0 <= cy < height):
                    continue
                long_side, short_side = max(box_width, box_height), min(box_width, box_height)
                if not (4.0 <= long_side <= 140.0 and 1.5 <= short_side <= 70.0 and long_side / max(short_side, 1.0) <= 25.0):
                    continue
                detections.append({
                    # xywhr's angle refers to the original width axis, not the longest side.
                    "cx": cx, "cy": cy, "width": box_width, "height": box_height,
                    "angle": math.degrees(angle_rad), "conf": float(score),
                })
    return deduplicate_detections(detections)


def warmup_detector(model: YOLO, options: argparse.Namespace) -> None:
    """Load the inference runtime once before any per-image failure can be recorded."""
    if options.device.lower().startswith("intel:"):
        from openvino import Core

        requested = options.device.split(":", 1)[1].upper()
        available = Core().available_devices
        if requested not in available:
            raise RuntimeError(f"OpenVINO device {requested!r} is unavailable: {available}")
    blank = np.zeros((options.crop_size, options.crop_size, 3), dtype=np.uint8)
    results = model.predict(
        source=[blank], imgsz=options.crop_size, conf=options.conf, iou=0.30,
        device=options.device, verbose=False, save=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"detector warmup returned {len(results)} results")


def build_open_water_mask(gray: np.ndarray, scale: int = 8) -> np.ndarray:
    small_width = max(1, math.ceil(gray.shape[1] / scale))
    small_height = max(1, math.ceil(gray.shape[0] / scale))
    small = cv2.resize(gray, (small_width, small_height), interpolation=cv2.INTER_AREA)
    valid = small > 0
    if np.count_nonzero(valid) < 64:
        return np.zeros_like(small, dtype=np.uint8)
    source = small.astype(np.float32)
    mean = cv2.GaussianBlur(source, (0, 0), 2.5)
    square_mean = cv2.GaussianBlur(source * source, (0, 0), 2.5)
    deviation = np.sqrt(np.maximum(square_mean - mean * mean, 0.0))
    mean_threshold = float(np.percentile(mean[valid], 72.0))
    deviation_threshold = float(np.percentile(deviation[valid], 72.0))
    extreme_deviation = float(np.percentile(deviation[valid], 94.0))
    land_seed = valid & (((mean >= mean_threshold) & (deviation >= deviation_threshold)) | (deviation >= extreme_deviation))
    land_seed = cv2.morphologyEx(land_seed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(land_seed, 8)
    land = np.zeros_like(land_seed)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= 80:
            land[labels == label] = 1
    land = cv2.dilate(land, np.ones((25, 25), np.uint8))
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8))
    return ((valid > 0) & (land == 0)).astype(np.uint8)


def worldcover_tile_name(latitude: int, longitude: int) -> str:
    latitude_code = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}"
    longitude_code = f"{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}"
    return f"ESA_WorldCover_10m_2021_v200_{latitude_code}{longitude_code}_Map.tif"


class WorldCoverMissingError(FileNotFoundError):
    pass


class WorldCoverDownloadError(RuntimeError):
    pass


def validate_worldcover_file(path: Path) -> None:
    with rasterio.open(path) as source:
        if source.count < 1 or source.width < 1 or source.height < 1 or source.crs is None:
            raise ValueError(f"invalid WorldCover raster: {path}")
        source.read(1, window=Window(0, 0, min(64, source.width), min(64, source.height)))


def cached_worldcover_source(url: str, cache_root: Path, timeout: float) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / Path(urllib.parse.urlparse(url).path).name
    missing_marker = destination.with_suffix(destination.suffix + ".missing_404")
    if missing_marker.is_file():
        raise WorldCoverMissingError(url)
    if destination.is_file():
        try:
            validate_worldcover_file(destination)
            return destination
        except (OSError, ValueError, rasterio.errors.RasterioIOError):
            invalid = destination.with_name(f"{destination.name}.invalid.{int(time.time())}")
            destination.replace(invalid)

    temporary = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    try:
        try:
            maximum_time = max(timeout, timeout * 3)
            command = [
                "curl.exe", "--fail", "--location", "--silent", "--show-error",
                "--connect-timeout", str(timeout), "--max-time", str(maximum_time),
                "--retry", "1", "--retry-delay", "1", "--user-agent",
                "sar-v5-worldcover-cache/1.0", "--output", str(temporary),
                "--write-out", "%{http_code}", url,
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=maximum_time + 10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            status_code = result.stdout.strip()
            if status_code == "404":
                atomic_text(missing_marker, f"{url}\nchecked_at={utc_now()}\n")
                raise WorldCoverMissingError(url)
            if result.returncode != 0:
                detail = result.stderr.strip() or f"curl exit {result.returncode}"
                raise WorldCoverDownloadError(f"WorldCover download failed: {url}: {detail}")
            if not status_code.startswith("2"):
                raise WorldCoverDownloadError(f"WorldCover HTTP {status_code}: {url}")
        except (OSError, subprocess.TimeoutExpired) as error:
            raise WorldCoverDownloadError(f"WorldCover download failed: {url}: {error!r}") from error
        validate_worldcover_file(temporary)
        temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def worldcover_sources(dataset) -> list[tuple[str, str]]:
    left, bottom, right, top = transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds, densify_pts=21)
    longitude_cells = range(math.floor(left / 3) * 3, math.floor((right - 1e-9) / 3) * 3 + 1, 3)
    latitude_cells = range(math.floor(bottom / 3) * 3, math.floor((top - 1e-9) / 3) * 3 + 1, 3)
    sources = []
    for latitude in latitude_cells:
        for longitude in longitude_cells:
            name = worldcover_tile_name(latitude, longitude)
            url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}"
            sources.append((name, url))
    return sources


def worldcover_urls(dataset, scale: int = 8, cache_root: Path | None = None,
                    timeout: float = 20.0) -> tuple[np.ndarray, list[str]]:
    height = math.ceil(dataset.height / scale)
    width = math.ceil(dataset.width / scale)
    transform = dataset.transform * Affine.scale(scale, scale)
    cover = np.zeros((height, width), dtype=np.uint8)
    sources = []
    environment = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MAX_RETRY": "4",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "GDAL_HTTP_CONNECTTIMEOUT": str(timeout),
        "GDAL_HTTP_TIMEOUT": str(timeout),
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    }
    with rasterio.Env(**environment):
        for _, url in worldcover_sources(dataset):
            temporary = np.zeros_like(cover)
            last_error = None
            for attempt in range(4):
                try:
                    source_path = cached_worldcover_source(url, cache_root, timeout) if cache_root else url
                    with rasterio.open(source_path) as source:
                        reproject(
                            source=rasterio.band(source, 1), destination=temporary,
                            src_transform=source.transform, src_crs=source.crs,
                            dst_transform=transform, dst_crs=dataset.crs,
                            src_nodata=0, dst_nodata=0, resampling=Resampling.nearest,
                        )
                    last_error = None
                    break
                except WorldCoverMissingError:
                    sources.append(f"{url}#missing_404_assumed_ocean")
                    last_error = None
                    break
                except (WorldCoverDownloadError, rasterio.errors.RasterioIOError,
                        rasterio.errors.WarpOperationError, OSError, ValueError) as error:
                    last_error = error
                    time.sleep(2 ** attempt)
            if last_error is not None:
                if "HTTP response code: 404" in str(last_error):
                    sources.append(f"{url}#missing_404_assumed_ocean")
                    continue
                raise last_error
            if any(item == f"{url}#missing_404_assumed_ocean" for item in sources):
                continue
            cover[temporary > 0] = temporary[temporary > 0]
            sources.append(f"{url}#cache={source_path}" if cache_root else url)
    if not sources:
        raise RuntimeError("WorldCover returned no source tiles for the image")
    land = ((cover > 0) & (cover != 80)).astype(np.uint8)
    shore_buffer = cv2.dilate(land, np.ones((15, 15), dtype=np.uint8))
    # WorldCover uses nodata outside mapped land, including the open ocean.
    return (((cover == 0) | (cover == 80)) & (shore_buffer == 0)).astype(np.uint8), sources


def open_water_metrics(gray: np.ndarray, water_mask: np.ndarray, detection: dict, scale: int = 8) -> dict:
    cx, cy = detection["cx"], detection["cy"]
    mask_x = min(water_mask.shape[1] - 1, max(0, int(cx / scale)))
    mask_y = min(water_mask.shape[0] - 1, max(0, int(cy / scale)))
    coarse_water = bool(water_mask[mask_y, mask_x])
    radius = int(max(64.0, min(160.0, 4.0 * max(detection["width"], detection["height"]))))
    x0, y0 = max(0, int(cx - radius)), max(0, int(cy - radius))
    x1, y1 = min(gray.shape[1], int(cx + radius + 1)), min(gray.shape[0], int(cy + radius + 1))
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return {"open_water": False, "coarse_water": coarse_water, "clutter_density": 1.0, "valid_fraction": 0.0}
    valid_fraction = float(np.count_nonzero(patch)) / float(patch.size)
    source = patch.astype(np.float32)
    residual = cv2.GaussianBlur(source, (0, 0), 0.9) - cv2.GaussianBlur(source, (0, 0), 6.0)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    bright = residual > median + max(4.0, 3.5 * 1.4826 * mad)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    center_radius = max(12.0, 1.5 * max(detection["width"], detection["height"]))
    annulus = ((xx - cx) ** 2 + (yy - cy) ** 2 >= center_radius ** 2)
    clutter_density = float(np.count_nonzero(bright & annulus)) / max(1, int(np.count_nonzero(annulus)))
    return {
        "open_water": bool(coarse_water and valid_fraction >= 0.55 and clutter_density <= 0.035),
        "coarse_water": coarse_water,
        "clutter_density": clutter_density,
        "valid_fraction": valid_fraction,
    }


def match_old_to_detections(old_objects: list[dict], detections: list[dict]) -> tuple[dict[int, int], list[dict]]:
    old_centers = [object_box(obj)[:2] for obj in old_objects]
    nearest_detection = []
    for index, detection in enumerate(detections):
        distances = [
            math.hypot(detection["cx"] - other["cx"], detection["cy"] - other["cy"])
            for other_index, other in enumerate(detections) if other_index != index
        ]
        nearest_detection.append(min(distances) if distances else float("inf"))
    edges = []
    limits = (48.0, 96.0, 160.0)
    minimum_confidence = (0.08, 0.15, 0.30, 0.50)
    for old_index, (cx, cy) in enumerate(old_centers):
        for detection_index, detection in enumerate(detections):
            distance = math.hypot(cx - detection["cx"], cy - detection["cy"])
            tier = next((idx for idx, limit in enumerate(limits) if distance <= limit), None)
            if tier is None and distance <= 192.0 and nearest_detection[detection_index] >= 96.0:
                tier = len(limits)
            if tier is not None and detection["conf"] >= minimum_confidence[tier]:
                edges.append((tier, distance, -detection["conf"], old_index, detection_index))
    matches: dict[int, int] = {}
    used_detections = set()
    audit = []
    for tier, distance, _, old_index, detection_index in sorted(edges):
        if old_index in matches or detection_index in used_detections:
            continue
        matches[old_index] = detection_index
        used_detections.add(detection_index)
        audit.append({"old_index": old_index, "detection_index": detection_index, "tier": tier + 1, "distance_px": distance})
    return matches, audit


def preserve_review_object(source: dict, status: str) -> dict:
    result = dict(source)
    meta = dict(result.get("meta") or {})
    meta["redetect_status"] = status
    result["meta"] = meta
    return result


def preview_object(source: dict, detection: dict, status: str, image_width: int, image_height: int) -> dict:
    cx, cy, width, height, angle = clamp_axis_box(axis_aligned_box(detection), image_width, image_height)
    half_width, half_height = width / 2.0, height / 2.0
    result = dict(source)
    result["pts"] = [
        [cx - half_width, cy - half_height],
        [cx + half_width, cy - half_height],
        [cx + half_width, cy + half_height],
        [cx - half_width, cy + half_height],
    ]
    result["cxcywh_angle"] = [cx, cy, width, height, angle]
    result["conf"] = float(detection["conf"])
    meta = dict(result.get("meta") or {})
    meta.update({
        "redetect_status": status,
        "redetect_model_conf": float(detection["conf"]),
        "redetect_angle_policy": "axis_aligned_for_chip_crop",
        "redetect_obb_cxcywh_angle_deg": [float(detection[key]) for key in
                                         ("cx", "cy", "width", "height", "angle")],
    })
    result["meta"] = meta
    return result


def draw_overview(gray: np.ndarray, old_objects: list[dict], preview_objects: list[dict], out_path: Path) -> None:
    height, width = gray.shape
    scale = min(1.0, 1400.0 / width, 900.0 / height)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for obj in old_objects:
        cx, cy, _, _, _ = object_box(obj)
        cv2.circle(image, (int(round(cx)), int(round(cy))), 7, (0, 220, 255), 2)
    for obj in preview_objects:
        points = np.rint(np.asarray(obj["pts"], dtype=np.float32)).astype(np.int32)
        color = (255, 0, 255) if int(obj.get("cls_id", -1)) == len(CLASSES) else (255, 255, 0)
        cv2.polylines(image, [points], True, color, 2)
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def process_image(model: YOLO, tif: Path, json_path: Path, output_root: Path, options: argparse.Namespace) -> dict:
    source_json_bytes = json_path.read_bytes()
    source_json_sha256 = hashlib.sha256(source_json_bytes).hexdigest()
    source_tif_stat = tif.stat()
    with rasterio.open(tif) as dataset:
        gray = dataset.read(1)
        water_mask, water_sources = worldcover_urls(
            dataset, cache_root=options.worldcover_cache, timeout=options.worldcover_timeout
        )
    if gray.dtype != np.uint8:
        raise ValueError(f"expected uint8 TIFF: {tif}")
    label = json.loads(source_json_bytes.decode("utf-8-sig"))
    if label.get("image") != tif.name or label.get("classes") != CLASSES:
        raise ValueError(f"incompatible JSON: {json_path}")
    old_objects = list(label.get("objects") or [])
    detections = detect_ships(model, gray, options.crop_size, options.overlap, options.conf, options.device)
    for detection in detections:
        detection.update(open_water_metrics(gray, water_mask, detection))
    eligible = [item for item in detections if item["open_water"]]
    strict_matches, strict_audit = match_old_to_detections(old_objects, eligible)
    old_metrics = {}
    for old_index, old_object in enumerate(old_objects):
        if old_index in strict_matches:
            continue
        cx, cy, width, height, angle = object_box(old_object)
        old_metrics[old_index] = open_water_metrics(gray, water_mask, {
            "cx": cx, "cy": cy, "width": width, "height": height, "angle": angle,
        })
    extended_matches = {}
    matches = strict_matches
    match_audit = strict_audit
    used = set(matches.values())
    preview = []
    audit_rows = []
    for old_index, old_object in enumerate(old_objects):
        if old_index not in matches:
            cx, cy, width, height, angle = object_box(old_object)
            metrics = old_metrics[old_index]
            if metrics["open_water"]:
                preview.append(preserve_review_object(old_object, "old_unmatched_preserved_open_water"))
                status = "old_unmatched_preserved_open_water"
            else:
                status = "old_unmatched_excluded_not_open_water"
            audit_rows.append({
                "status": status, "old_index": old_index, "old_cx": cx, "old_cy": cy,
                "coarse_water": metrics["coarse_water"], "clutter_density": metrics["clutter_density"],
                "valid_fraction": metrics["valid_fraction"],
            })
            continue
        detection_index = matches[old_index]
        detection = eligible[detection_index]
        status = "matched_recentered"
        output = preview_object(old_object, detection, status, gray.shape[1], gray.shape[0])
        preview.append(output)
        cx, cy, _, _, _ = object_box(old_object)
        audit_rows.append({
            "status": status, "old_index": old_index,
            "old_cx": cx, "old_cy": cy, "new_cx": detection["cx"], "new_cy": detection["cy"],
            "distance_px": math.hypot(cx - detection["cx"], cy - detection["cy"]),
            "model_conf": detection["conf"], "clutter_density": detection["clutter_density"],
            "detection_width": detection["width"], "detection_height": detection["height"],
        })
    for detection_index, detection in enumerate(eligible):
        if detection_index in used:
            continue
        audit_rows.append({
            "status": "detector_only_excluded_by_policy", "new_cx": detection["cx"], "new_cy": detection["cy"],
            "model_conf": detection["conf"], "clutter_density": detection["clutter_density"],
            "detection_width": detection["width"], "detection_height": detection["height"],
        })
    for detection in detections:
        if not detection["open_water"]:
            audit_rows.append({
                "status": "excluded_nearshore_or_clutter", "new_cx": detection["cx"], "new_cy": detection["cy"],
                "model_conf": detection["conf"], "coarse_water": detection["coarse_water"],
                "clutter_density": detection["clutter_density"], "valid_fraction": detection["valid_fraction"],
            })

    key = hashlib.sha256(str(tif).encode("utf-8")).hexdigest()[:12]
    image_root = output_root / "images" / key
    preview_path = image_root / "labels_json" / json_path.name
    if digest(json_path) != source_json_sha256 and not options.overwrite_manual:
        raise RuntimeError(f"source JSON changed during processing: {json_path}")
    current_tif_stat = tif.stat()
    if (current_tif_stat.st_size, current_tif_stat.st_mtime_ns) != (source_tif_stat.st_size, source_tif_stat.st_mtime_ns):
        raise RuntimeError(f"source TIFF changed during processing: {tif}")
    payload = {"image": tif.name, "classes": CLASSES, "objects": preview}
    atomic_json(preview_path, payload)
    audit_path = image_root / "audit.csv"
    write_csv(audit_path, audit_rows)
    overview_path = image_root / "overview.png"
    if options.write_overviews:
        draw_overview(gray, old_objects, preview, overview_path)
    publication = {
        "published_active": False,
        "active_before_sha256": source_json_sha256,
        "active_after_sha256": source_json_sha256,
        "classes_txt": "",
        "classes_before_sha256": "",
        "classes_after_sha256": "",
        "publish_status": "preview_only",
        "backup_created": False,
    }
    if options.publish_active:
        publication = publish_active_label(
            json_path, tif, payload, source_json_sha256, gray.shape[1], gray.shape[0], options.overwrite_manual
        )
        if publication["active_after_sha256"] != digest(preview_path):
            raise RuntimeError(f"active JSON does not match preview after publish: {json_path}")
    return {
        "key": key, "tif": str(tif), "json": str(json_path), "json_sha256": source_json_sha256,
        "tif_size": source_tif_stat.st_size, "tif_mtime_ns": source_tif_stat.st_mtime_ns,
        "old_objects": len(old_objects), "model_detections": len(detections), "eligible_detections": len(eligible),
        "matched": len(matches), "matched_strict": len(strict_matches),
        "matched_extended_unique": len(extended_matches),
        "old_preserved_open_water": sum(
            row["status"] == "old_unmatched_preserved_open_water" for row in audit_rows
        ),
        "old_excluded_not_open_water": sum(
            row["status"] == "old_unmatched_excluded_not_open_water" for row in audit_rows
        ),
        "new_candidates": 0,
        "detector_only_excluded": len(eligible) - len(used),
        "old_unmatched": len(old_objects) - len(matches),
        "excluded_nearshore_or_clutter": len(detections) - len(eligible),
        "water_mask_sources": ";".join(water_sources),
        "preview_json": str(preview_path), "audit_csv": str(audit_path),
        "overview": str(overview_path) if options.write_overviews else "",
        "mean_shift_px": float(np.mean([row["distance_px"] for row in match_audit])) if match_audit else 0.0,
        **publication,
    }


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for tif in sorted(root.rglob("*.tif")):
        candidates = [tif.with_suffix(".json"), tif.parent / "labels_json" / f"{tif.stem}.json"]
        json_path = next((path for path in candidates if path.is_file()), None)
        if json_path:
            pairs.append((tif, json_path))
    return pairs


def manifest_pairs(path: Path) -> list[tuple[Path, Path]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"tile_tif", "tile_json"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"tile manifest must contain {sorted(required)}: {path}")
    pairs = []
    seen = set()
    for row_number, row in enumerate(rows, start=2):
        tif = Path(row["tile_tif"])
        json_path = Path(row["tile_json"])
        normalized = str(tif).casefold()
        if normalized in seen:
            raise ValueError(f"duplicate tile_tif at row {row_number}: {tif}")
        seen.add(normalized)
        pairs.append((tif, json_path))
    return pairs


def config_payload(options: argparse.Namespace) -> dict:
    source_path = options.tile_manifest or options.input_root
    payload = {
        "source_kind": "tile_manifest" if options.tile_manifest else "input_root",
        "source_path": str(source_path.resolve()),
        "source_sha256": digest(options.tile_manifest) if options.tile_manifest else None,
        "model": str(options.model.resolve()),
        "model_sha256": digest_model(options.model),
        "device": options.device,
        "crop_size": options.crop_size,
        "overlap": options.overlap,
        "confidence": options.conf,
        "max_images": options.max_images,
        "write_overviews": options.write_overviews,
        "publish_active": options.publish_active,
        "overwrite_manual": options.overwrite_manual,
        "skip_backup": options.skip_backup,
        "algorithm_version": ALGORITHM_VERSION,
        "search_radii_px": [48, 96, 160, 192],
        "old_unmatched_policy": "preserve_open_water_without_forced_snap",
        "detector_only_policy": "audit_only_excluded_from_json",
        "angle_policy": "axis_aligned_for_chip_crop",
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["config_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def pair_key(tif: Path) -> str:
    return hashlib.sha256(str(tif).encode("utf-8")).hexdigest()[:12]


def completed_summary(marker_path: Path, config_sha256: str) -> tuple[dict | None, str | None]:
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("config_sha256") != config_sha256:
            return None, "config_changed"
        summary = marker["summary"]
        source_json = Path(summary["json"])
        current_json_sha256 = digest(source_json)
        if summary.get("published_active"):
            if current_json_sha256 != summary.get("active_after_sha256"):
                return None, "source_json_changed_after_completion"
            if not summary.get("active_after_sha256") == marker.get("preview_sha256"):
                return None, "published_json_does_not_match_preview"
        elif current_json_sha256 != summary["json_sha256"]:
            return None, "source_json_changed_after_completion"
        tif = Path(summary["tif"])
        stat = tif.stat()
        if (stat.st_size, stat.st_mtime_ns) != (int(summary["tif_size"]), int(summary["tif_mtime_ns"])):
            return None, "source_tif_changed_after_completion"
        for field, hash_field in (("preview_json", "preview_sha256"), ("audit_csv", "audit_sha256")):
            if digest(Path(summary[field])) != marker[hash_field]:
                return None, f"{field}_changed_or_missing"
        return summary, None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return None, f"invalid_completion_marker:{error!r}"


def all_completion_summaries(run_root: Path, config_sha256: str) -> list[dict]:
    summaries = []
    for marker_path in sorted((run_root / "completion").glob("*.json")):
        summary, error = completed_summary(marker_path, config_sha256)
        if summary is not None and error is None:
            summaries.append(summary)
    return summaries


def write_progress(run_root: Path, total: int, completed: int, failed: int,
                   current_tif: Path | None, state: str) -> None:
    atomic_json(run_root / "progress.json", {
        "state": state,
        "total": total,
        "completed": completed,
        "failed": failed,
        "remaining": max(0, total - completed),
        "current_tif": str(current_tif) if current_tif else None,
        "updated_at": utc_now(),
        "pid": os.getpid(),
    })


def aggregate_summary(run_root: Path, summaries: list[dict], errors: list[dict], options: argparse.Namespace) -> None:
    write_csv(run_root / "per_image_summary.csv", summaries)
    write_csv(run_root / "errors.csv", errors)
    status_counts = Counter()
    for row in summaries:
        for key in ("old_objects", "model_detections", "eligible_detections", "matched", "matched_strict",
                    "matched_extended_unique", "old_preserved_open_water", "old_excluded_not_open_water",
                    "new_candidates", "detector_only_excluded", "old_unmatched",
                    "excluded_nearshore_or_clutter"):
            status_counts[key] += int(row.get(key, 0))
    atomic_json(run_root / "summary.json", {
        "images": len(summaries), "errors": len(errors), **dict(status_counts),
        "model": str(options.model), "confidence": options.conf,
        "algorithm_version": ALGORITHM_VERSION,
        "search_radii_px": [48, 96, 160, 192],
        "old_unmatched_policy": "preserve_open_water_without_forced_snap",
        "detector_only_policy": "audit_only_excluded_from_json",
        "angle_policy": "axis_aligned_for_chip_crop",
        "active_json_modified": bool(options.publish_active),
        "publish_active": bool(options.publish_active),
    })


def main() -> None:
    options = parse_args()
    config = config_payload(options)
    config_path = options.run_root / "run_config.json"
    existing_names = {path.name for path in options.run_root.iterdir()} if options.run_root.exists() else set()
    unexpected_names = {
        name for name in existing_names
        if name not in {"stdout.log", "stderr.log", "launcher.json", "launcher.pid"}
        and not name.startswith("breakaway_launcher")
    }
    if unexpected_names and not options.resume:
        raise FileExistsError(f"refusing to overwrite nonempty run root without --resume: {options.run_root}")
    options.run_root.mkdir(parents=True, exist_ok=True)
    if options.resume:
        if not config_path.is_file():
            raise FileNotFoundError(f"resume requires run_config.json: {config_path}")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config.get("config_sha256") != config["config_sha256"]:
            raise ValueError("resume configuration does not match the existing run")
    else:
        atomic_json(config_path, {**config, "created_at": utc_now()})

    pairs = manifest_pairs(options.tile_manifest) if options.tile_manifest else find_pairs(options.input_root)
    if options.max_images is not None:
        pairs = pairs[:options.max_images]
    if not pairs:
        raise FileNotFoundError("no TIFF/JSON pairs found")

    atomic_json(options.run_root / "worker.json", {
        "pid": os.getpid(), "python": sys.executable, "started_at": utc_now(),
        "config_sha256": config["config_sha256"],
        "worldcover_cache": str(options.worldcover_cache) if options.worldcover_cache else None,
        "worldcover_timeout": options.worldcover_timeout,
    })
    model = YOLO(str(options.model), task="obb")
    warmup_detector(model, options)
    summaries = all_completion_summaries(options.run_root, config["config_sha256"])
    completed_keys = {row["key"] for row in summaries}
    errors = []
    write_progress(options.run_root, len(pairs), len(completed_keys), 0, None, "running")
    for tif, json_path in pairs:
        key = pair_key(tif)
        marker_path = options.run_root / "completion" / f"{key}.json"
        if key in completed_keys:
            continue
        if marker_path.is_file():
            _, marker_error = completed_summary(marker_path, config["config_sha256"])
            if marker_error in {"source_json_changed_after_completion", "source_tif_changed_after_completion"}:
                error_row = {"tif": str(tif), "json": str(json_path), "error": marker_error}
                errors.append(error_row)
                atomic_json(options.run_root / "failures" / f"{key}.json", {**error_row, "failed_at": utc_now()})
                write_progress(options.run_root, len(pairs), len(completed_keys), len(errors), tif, "running")
                continue
        write_progress(options.run_root, len(pairs), len(completed_keys), len(errors), tif, "running")
        try:
            summary = process_image(model, tif, json_path, options.run_root, options)
            summaries.append(summary)
            completed_keys.add(key)
            atomic_json(marker_path, {
                "config_sha256": config["config_sha256"], "completed_at": utc_now(), "summary": summary,
                "preview_sha256": digest(Path(summary["preview_json"])),
                "audit_sha256": digest(Path(summary["audit_csv"])),
            })
            failure_path = options.run_root / "failures" / f"{key}.json"
            if failure_path.is_file():
                failure_path.unlink()
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        except Exception as error:
            error_row = {
                "tif": str(tif),
                "json": str(json_path),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            errors.append(error_row)
            atomic_json(options.run_root / "failures" / f"{key}.json", {**error_row, "failed_at": utc_now()})
            print(json.dumps(error_row, ensure_ascii=False), flush=True)
        write_progress(options.run_root, len(pairs), len(completed_keys), len(errors), tif, "running")
    summaries = all_completion_summaries(options.run_root, config["config_sha256"])
    aggregate_summary(options.run_root, summaries, errors, options)
    state = "completed" if not errors else "completed_with_errors"
    write_progress(options.run_root, len(pairs), len(summaries), len(errors), None, state)
    atomic_json(options.run_root / "exit_status.json", {
        "state": state, "exit_code": 1 if errors else 0, "finished_at": utc_now(),
        "completed": len(summaries), "errors": len(errors), "pid": os.getpid(),
    })
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
