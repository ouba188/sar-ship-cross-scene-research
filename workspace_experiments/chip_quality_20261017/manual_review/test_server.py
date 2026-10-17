"""Integration checks; review decisions only go into TemporaryDirectory."""
import copy
import csv
import io
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from build_review_data import QUALITY_DIR, ReviewData, integer, read_csv, truth
from server import ReviewStore, create_server, json_bytes


@contextmanager
def running(data, state_dir):
    server = create_server(0, state_dir, data)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:{}".format(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(base, path, payload=None):
    req = Request(base + path, data=json_bytes(payload) if payload is not None else None,
                  headers={"Content-Type": "application/json"})
    try:
        response = urlopen(req, timeout=15)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, response.headers, response.read()


def get_json(base, path, payload=None):
    status, _, body = request(base, path, payload)
    if status != 200:
        raise AssertionError((status, body))
    return json.loads(body)


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ReviewData()

    def test_candidate_union_counts_and_evidence(self):
        chips = list(read_csv(QUALITY_DIR / "per_chip_quality.csv"))
        attributes = {row["instance_id"] for row in read_csv(QUALITY_DIR / "ais_attribute_review.csv")}
        expected = attributes | {row["instance_id"] for row in chips if truth(row["multi_any_25"])}
        expected |= {row["instance_id"] for row in chips if truth(row["length_screen_any"])}
        self.assertEqual(len(chips), 38091)
        self.assertEqual(len(self.data.items), len(expected))
        self.assertEqual(len(self.data.items), 6571)
        self.assertEqual({item["id"] for item in self.data.items}, expected)
        width_only = {row["instance_id"] for row in chips if truth(row["width_screen_any"])} - expected
        self.assertTrue(width_only)
        self.assertTrue(width_only.isdisjoint(self.data.details))
        inventory = list(read_csv(QUALITY_DIR.parent / "obb_consistent_repair_20261016/inventory.csv"))
        experiment_ids = {r["instance_id"] for r in inventory}
        self.assertEqual(self.data.counts["in_experiment"], len(expected & experiment_ids))
        self.assertEqual(self.data.counts["in_experiment"], 491)
        rows_to_ids = {integer(r["global_row"]): r["instance_id"] for r in inventory}
        expected_roles = {}
        for row in read_csv(QUALITY_DIR.parent / "obb_consistent_repair_20261016/roles.csv"):
            identity = rows_to_ids[integer(row["global_row"])]
            expected_roles.setdefault(identity, set()).add(row["split"] + ":" + row["role"])
        for item in self.data.items:
            self.assertEqual(item["in_experiment"], item["id"] in experiment_ids)
            self.assertEqual(set(item["roles"]), expected_roles.get(item["id"], set()))
        for row in read_csv(QUALITY_DIR / "per_view_quality.csv"):
            if row["instance_id"] in expected:
                view = self.data.details[row["instance_id"]]["views"][row["pol"]]
                self.assertEqual(view["anchor"], [float(row["anchor_x"]), float(row["anchor_y"])])
                self.assertEqual(view["target_index"], integer(row["target_index"]))
        # Strict serialization catches non-finite values in every detail/detection.
        json_bytes(self.data.details)
        identity = self.data.items[0]["id"]
        detail = self.data.candidate(identity)
        expected_related = {r["id"] for r in self.data.items
                            if r["mmsi"] == detail["mmsi"] and r["id"] != identity}
        self.assertEqual({r["id"] for r in detail["related"]}, expected_related)

    def test_http_contract_images_and_source(self):
        identity = self.data.items[0]["id"]
        with tempfile.TemporaryDirectory(prefix="chip-review-test-") as state, running(self.data, state) as base:
            listing = get_json(base, "/api/candidates")
            self.assertEqual(listing["counts"], self.data.counts)
            self.assertEqual(len(listing["items"]), 6571)
            detail = get_json(base, "/api/candidate?id=" + identity)
            self.assertEqual(detail, self.data.candidate(identity))
            for pol in ("vv", "vh"):
                status, headers, body = request(base, "/api/image?id={}&pol={}".format(identity, pol))
                self.assertEqual(status, 200)
                self.assertEqual(headers.get_content_type(), "image/png")
                self.assertTrue(body.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertEqual(body, Path(detail["views"][pol]["path"]).read_bytes())
            status, headers, body = request(base, "/api/source?id=" + identity)
            self.assertEqual(status, 200)
            self.assertEqual(headers.get_content_type(), "text/plain")
            text = body.decode("utf-8")
            ais = detail["ais"]
            self.assertIn(ais["source"], text)
            with Path(ais["source"]).open(encoding="utf-8-sig", newline="") as stream:
                lines = stream.readlines()
            for line in range(ais["line"] - 2, ais["line"] + 3):
                if 1 <= line <= len(lines):
                    self.assertIn("{}: {}".format(line, lines[line - 1]), text)
            self.assertEqual(request(base, "/api/image?id={}&pol=other".format(identity))[0], 400)
            for path in ("/api/candidate?id=missing", "/api/image?id=../../server.py", "/server.py",
                         "/../per_chip_quality.csv", "/state/reviews.json"):
                self.assertEqual(request(base, path)[0], 404)
            self.assertFalse((Path(state) / "reviews.json").exists())

    def test_missing_source_does_not_break_other_routes(self):
        identity = self.data.items[0]["id"]
        data = copy.copy(self.data)
        data.details = dict(self.data.details)
        with tempfile.TemporaryDirectory(prefix="chip-review-missing-") as state:
            data.details[identity] = dict(data.details[identity], ais=dict(
                data.details[identity]["ais"], source=str(Path(state) / "missing.txt")))
            with running(data, state) as base:
                status, headers, body = request(base, "/api/source?id=" + identity)
                self.assertEqual(status, 404)
                self.assertEqual(headers.get_content_type(), "text/plain")
                self.assertIn(b"FileNotFoundError", body)
                self.assertIn(b"missing.txt", body)
                self.assertEqual(get_json(base, "/api/candidate?id=" + identity)["id"], identity)
                self.assertEqual(request(base, "/api/image?id=" + identity)[0], 200)
                self.assertEqual(get_json(base, "/api/candidates")["counts"], data.counts)

    def test_isolation_persistence_and_exports(self):
        identities = [row["id"] for row in self.data.items[:4]]
        note = "\u4eba\u5de5\u786e\u8ba4,\u4fdd\u7559\"\u539f\u59cb\u6570\u636e\"\n\u7b2c\u4e8c\u884c"
        with tempfile.TemporaryDirectory(prefix="chip-review-state-") as root:
            state_a, state_b = Path(root) / "a", Path(root) / "b"
            with running(self.data, state_a) as base_a, running(self.data, state_b) as base_b:
                self.assertEqual(get_json(base_a, "/api/reviews"), {"reviews": {}})
                for identity, status in zip(identities, ("keep", "exclude", "defer", "pending")):
                    saved = get_json(base_a, "/api/review", dict(id=identity, status=status, reason="manual", note=note))["review"]
                    self.assertEqual(saved["status"], status)
                    self.assertEqual(saved["note"], note)
                    self.assertTrue(saved["updated_at"])
                self.assertEqual(get_json(base_b, "/api/reviews"), {"reviews": {}})
                self.assertFalse((state_b / "reviews.json").exists())
                before = get_json(base_a, "/api/reviews")
                for status in ("all", "keep", "exclude", "defer"):
                    result = get_json(base_a, "/api/export?format=json&status=" + status)
                    self.assertEqual(result["count"], len(self.data.items) if status == "all" else 1)
                    code, headers, body = request(base_a, "/api/export?format=csv&status=" + status)
                    self.assertEqual(code, 200)
                    self.assertIn("attachment", headers["Content-Disposition"])
                    rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
                    self.assertEqual(len(rows), result["count"])
                    reviewed = [r for r in rows if r["id"] in identities]
                    self.assertTrue(reviewed)
                    self.assertTrue(all(r["note"] == note for r in reviewed))
                    self.assertTrue(all(r["status"] == status for r in rows) if status != "all" else True)
                for payload in ({"id": "missing", "status": "keep"}, {"id": identities[0]},
                                {"id": identities[0], "status": "delete"},
                                {"id": identities[0], "status": "keep", "note": None}, []):
                    self.assertEqual(request(base_a, "/api/review", payload)[0], 400)
                self.assertEqual(get_json(base_a, "/api/reviews"), before)
                for query in ("format=xml", "status=delete"):
                    self.assertEqual(request(base_a, "/api/export?" + query)[0], 400)
            with running(self.data, state_a) as restarted:
                self.assertEqual(get_json(restarted, "/api/reviews"), before)
                reset = get_json(restarted, "/api/review", {"id": identities[0], "status": "pending"})
                self.assertEqual(reset["review"]["status"], "pending")

    def test_atomic_failure_preserves_saved_state(self):
        identity = self.data.items[0]["id"]
        with tempfile.TemporaryDirectory(prefix="chip-review-atomic-") as state:
            store = ReviewStore(state, self.data.details)
            store.save({"id": identity, "status": "keep"})
            before = store.path.read_bytes()
            with patch("server.os.replace", side_effect=OSError("simulated interrupted replacement")):
                with self.assertRaises(OSError):
                    store.save({"id": identity, "status": "exclude"})
            self.assertEqual(store.path.read_bytes(), before)
            self.assertEqual(store.snapshot()[identity]["status"], "keep")
            self.assertEqual([p.name for p in Path(state).iterdir()], ["reviews.json"])

    def test_concurrent_requests_keep_all_decisions(self):
        identities = [row["id"] for row in self.data.items[:12]]
        with tempfile.TemporaryDirectory(prefix="chip-review-concurrent-") as state, running(self.data, state) as base:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda identity: get_json(base, "/api/review", {
                    "id": identity, "status": "defer", "note": identity}), identities))
            self.assertEqual(len(results), len(identities))
            saved = get_json(base, "/api/reviews")["reviews"]
            self.assertEqual(set(saved), set(identities))
            self.assertEqual(saved, json.loads((Path(state) / "reviews.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
