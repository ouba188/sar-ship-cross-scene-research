import unittest
import csv
import hashlib
import json
import tempfile
import os
from unittest import mock
from pathlib import Path

import cv2
import numpy as np

from sea_ship_redetect_preview import (
    axis_aligned_box,
    build_open_water_mask,
    box_iou,
    clamp_axis_box,
    deduplicate_detections,
    digest_model,
    match_old_to_detections,
    open_water_metrics,
    preserve_review_object,
    preview_object,
    completed_summary,
    configure_rasterio_data,
    manifest_pairs,
    rotated_points,
    worldcover_tile_name,
    cached_worldcover_source,
    validate_worldcover_file,
    WorldCoverMissingError,
)


class SeaShipRedetectTests(unittest.TestCase):
    def test_rotated_detection_becomes_axis_aligned_crop(self):
        detection = {"cx": 50.0, "cy": 40.0, "width": 20.0, "height": 6.0, "angle": 45.0}
        cx, cy, width, height, angle = axis_aligned_box(detection)
        self.assertEqual((cx, cy, angle), (50.0, 40.0, 0.0))
        expected_extent = (20.0 + 6.0) / np.sqrt(2.0) * 1.25
        self.assertAlmostEqual(width, expected_extent)
        self.assertAlmostEqual(height, expected_extent)

    def test_preview_object_does_not_call_rotated_points(self):
        source = {"cls_id": 2, "conf": 0.4, "pts": [], "cxcywh_angle": [1, 1, 1, 1, 0]}
        detection = {"cx": 50.0, "cy": 40.0, "width": 20.0, "height": 6.0, "angle": 0.0, "conf": 0.9}
        with mock.patch("sea_ship_redetect_preview.rotated_points", side_effect=AssertionError("unexpected call")):
            result = preview_object(source, detection, "matched_recentered", 100, 100)
        self.assertEqual(result["pts"], [[37.5, 36.0], [62.5, 36.0], [62.5, 44.0], [37.5, 44.0]])

    def test_axis_box_is_shrunk_at_image_edge_without_moving_center(self):
        clamped = clamp_axis_box((2.0, 98.0, 20.0, 20.0, 0.0), 100, 100)
        self.assertEqual(clamped, (2.0, 98.0, 4.0, 4.0, 0.0))

    def test_adaptive_matching_and_unique_assignment(self):
        old = [
            {"cxcywh_angle": [10, 10, 8, 4, 0], "pts": rotated_points(10, 10, 8, 4, 0).tolist()},
            {"cxcywh_angle": [100, 100, 8, 4, 0], "pts": rotated_points(100, 100, 8, 4, 0).tolist()},
        ]
        detections = [
            {"cx": 14.0, "cy": 10.0, "width": 8.0, "height": 4.0, "angle": 0.0, "conf": 0.8},
            {"cx": 170.0, "cy": 100.0, "width": 8.0, "height": 4.0, "angle": 0.0, "conf": 0.9},
        ]
        matches, audit = match_old_to_detections(old, detections)
        self.assertEqual(matches, {0: 0, 1: 1})
        self.assertEqual([row["tier"] for row in audit], [1, 2])

    def test_low_confidence_wide_match_is_rejected(self):
        old = [{"cxcywh_angle": [10, 10, 8, 4, 0], "pts": rotated_points(10, 10, 8, 4, 0).tolist()}]
        detections = [{"cx": 180.0, "cy": 10.0, "width": 8.0, "height": 4.0, "angle": 0.0, "conf": 0.2}]
        matches, _ = match_old_to_detections(old, detections)
        self.assertEqual(matches, {})

    def test_preserved_review_object_keeps_geometry_and_class(self):
        source = {
            "cls_id": 3, "conf": 0.5, "pts": rotated_points(10, 10, 8, 4, 0).tolist(),
            "cxcywh_angle": [10, 10, 8, 4, 0],
        }
        preserved = preserve_review_object(source, "old_unmatched_preserved_open_water")
        self.assertEqual(preserved["cls_id"], 3)
        self.assertEqual(preserved["pts"], source["pts"])
        self.assertEqual(preserved["meta"]["redetect_status"], "old_unmatched_preserved_open_water")

    def test_detection_deduplication(self):
        detections = [
            {"cx": 10.0, "cy": 10.0, "width": 12.0, "height": 4.0, "angle": 0.0, "conf": 0.9},
            {"cx": 11.0, "cy": 10.0, "width": 12.0, "height": 4.0, "angle": 0.0, "conf": 0.5},
            {"cx": 50.0, "cy": 50.0, "width": 12.0, "height": 4.0, "angle": 0.0, "conf": 0.7},
        ]
        kept = deduplicate_detections(detections)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["conf"], 0.9)

    def test_box_iou_matches_rotated_corner_aabb(self):
        first = {"cx": 20.0, "cy": 30.0, "width": 18.0, "height": 5.0, "angle": 37.0}
        second = {"cx": 24.0, "cy": 31.0, "width": 14.0, "height": 7.0, "angle": -22.0}

        def explicit_bounds(item):
            points = rotated_points(
                item["cx"], item["cy"], item["width"], item["height"], item["angle"]
            )
            return points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()

        ax0, ay0, ax1, ay1 = explicit_bounds(first)
        bx0, by0, bx1, by1 = explicit_bounds(second)
        intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
            0.0, min(ay1, by1) - max(ay0, by0)
        )
        union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
        self.assertAlmostEqual(box_iou(first, second), intersection / union, places=6)

    def test_open_water_gate_rejects_large_clutter_region(self):
        rng = np.random.default_rng(7)
        image = rng.integers(4, 10, size=(512, 512), dtype=np.uint8)
        image[40:240, 280:500] = rng.integers(50, 230, size=(200, 220), dtype=np.uint8)
        cv2.ellipse(image, (120, 350), (10, 3), 0, 0, 360, 230, -1)
        mask = build_open_water_mask(image)
        sea = {"cx": 120.0, "cy": 350.0, "width": 20.0, "height": 6.0, "angle": 0.0}
        land = {"cx": 390.0, "cy": 140.0, "width": 20.0, "height": 6.0, "angle": 0.0}
        self.assertTrue(open_water_metrics(image, mask, sea)["open_water"])
        self.assertFalse(open_water_metrics(image, mask, land)["open_water"])

    def test_worldcover_tile_name(self):
        self.assertEqual(worldcover_tile_name(-21, 117), "ESA_WorldCover_10m_2021_v200_S21E117_Map.tif")
        self.assertEqual(worldcover_tile_name(51, -3), "ESA_WorldCover_10m_2021_v200_N51W003_Map.tif")

    def test_worldcover_cache_reuses_valid_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cached = root / "tile.tif"
            import rasterio
            from rasterio.transform import from_origin
            with rasterio.open(cached, "w", driver="GTiff", width=8, height=8, count=1,
                               dtype="uint8", crs="EPSG:4326", transform=from_origin(0, 8, 1, 1)) as dst:
                dst.write(np.ones((8, 8), dtype=np.uint8), 1)
            validate_worldcover_file(cached)
            with mock.patch("urllib.request.urlopen") as urlopen:
                result = cached_worldcover_source("https://example.test/tile.tif", root, 1.0)
            self.assertEqual(result, cached)
            urlopen.assert_not_called()

    def test_worldcover_cache_reuses_missing_404_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tile.tif.missing_404").write_text("missing", encoding="ascii")
            with mock.patch("urllib.request.urlopen") as urlopen:
                with self.assertRaises(WorldCoverMissingError):
                    cached_worldcover_source("https://example.test/tile.tif", root, 1.0)
            urlopen.assert_not_called()

    def test_manifest_pairs_rejects_duplicate_tile(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "tiles.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["tile_tif", "tile_json"])
                writer.writeheader()
                writer.writerows([
                    {"tile_tif": "A.tif", "tile_json": "A.json"},
                    {"tile_tif": "A.tif", "tile_json": "A.json"},
                ])
            with self.assertRaisesRegex(ValueError, "duplicate tile_tif"):
                manifest_pairs(manifest)

    def test_completion_marker_detects_source_json_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tif = root / "tile.tif"
            source_json = root / "tile.json"
            preview = root / "preview.json"
            audit = root / "audit.csv"
            tif.write_bytes(b"tif")
            source_json.write_text("{}", encoding="utf-8")
            preview.write_text("{}", encoding="utf-8")
            audit.write_text("status\n", encoding="utf-8")
            stat = tif.stat()
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            marker = root / "marker.json"
            marker.write_text(json.dumps({
                "config_sha256": "config",
                "summary": {
                    "tif": str(tif), "json": str(source_json), "json_sha256": digest(source_json),
                    "tif_size": stat.st_size, "tif_mtime_ns": stat.st_mtime_ns,
                    "preview_json": str(preview), "audit_csv": str(audit),
                },
                "preview_sha256": digest(preview), "audit_sha256": digest(audit),
            }), encoding="utf-8")
            summary, error = completed_summary(marker, "config")
            self.assertIsNotNone(summary)
            self.assertIsNone(error)
            source_json.write_text('{"changed": true}', encoding="utf-8")
            summary, error = completed_summary(marker, "config")
            self.assertIsNone(summary)
            self.assertEqual(error, "source_json_changed_after_completion")

    def test_model_directory_digest_is_content_and_path_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.xml").write_text("xml", encoding="ascii")
            (root / "model.bin").write_bytes(b"bin")
            first = digest_model(root)
            self.assertEqual(first, digest_model(root))
            (root / "model.bin").write_bytes(b"changed")
            self.assertNotEqual(first, digest_model(root))

    def test_rasterio_data_fallbacks_are_valid_directories(self):
        configure_rasterio_data()
        self.assertTrue(Path(os.environ["PROJ_DATA"]).is_dir())
        self.assertTrue(Path(os.environ["GDAL_DATA"]).is_dir())


if __name__ == "__main__":
    unittest.main()
