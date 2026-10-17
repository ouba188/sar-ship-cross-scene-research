"""Local manual review server. Original data is always read-only."""
import argparse
import csv
import io
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from build_review_data import ReviewData

HERE = Path(__file__).resolve().parent
STATUSES = {"keep", "exclude", "defer", "pending"}
STATIC = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
          "/style.css": ("style.css", "text/css"), "/app.js": ("app.js", "text/javascript"),
          "/lucide.js": ("lucide.js", "text/javascript")}


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


class ReviewStore:
    def __init__(self, state_dir, valid_ids):
        self.path = Path(state_dir).resolve() / "reviews.json"
        self.valid_ids = frozenset(valid_ids)
        self.lock = threading.Lock()
        self.reviews = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def snapshot(self):
        with self.lock:
            return {key: dict(value) for key, value in self.reviews.items()}

    def save(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Review must be a JSON object")
        identity, status = payload.get("id"), payload.get("status")
        if not isinstance(identity, str) or identity not in self.valid_ids:
            raise ValueError("Unknown candidate id")
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError("Status must be keep, exclude, defer or pending")
        reason, note = payload.get("reason", ""), payload.get("note", "")
        if not isinstance(reason, str) or not isinstance(note, str):
            raise ValueError("Reason and note must be strings")
        with self.lock:
            review = dict(id=identity, status=status, reason=reason, note=note,
                          updated_at=datetime.now(timezone.utc).isoformat())
            updated = dict(self.reviews)
            updated[identity] = review
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="wb", dir=self.path.parent,
                                                 prefix=".reviews-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(json_bytes(updated))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            self.reviews = updated
            return dict(review)


class ReviewHandler(BaseHTTPRequestHandler):
    def respond(self, status, body, content_type="application/json", filename=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type if content_type == "image/png" else content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header("Content-Disposition", 'attachment; filename="{}"'.format(filename))
        self.end_headers()
        self.wfile.write(body)

    def error(self, status, message):
        self.respond(status, json_bytes({"error": message}))

    def do_GET(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        data = self.server.data
        if url.path == "/api/candidates":
            self.respond(200, json_bytes({"items": data.items, "counts": data.counts}))
        elif url.path == "/api/reviews":
            self.respond(200, json_bytes({"reviews": self.server.store.snapshot()}))
        elif url.path == "/api/export":
            self.export(query)
        elif url.path in {"/api/candidate", "/api/image", "/api/source"}:
            identity = query.get("id", [""])[0]
            if identity not in data.details:
                self.error(404, "Unknown candidate id")
                return
            if url.path == "/api/candidate":
                self.respond(200, json_bytes(data.candidate(identity)))
            elif url.path == "/api/source":
                status, text = data.source_text(identity)
                self.respond(status, text, "text/plain")
            else:
                pol = query.get("pol", ["vv"])[0]
                if pol not in {"vv", "vh"}:
                    self.error(400, "Polarization must be vv or vh")
                    return
                path = Path(data.details[identity]["views"][pol]["path"])
                try:
                    self.respond(200, path.read_bytes(), "image/png")
                except OSError as exc:
                    self.error(404, "Image unavailable: {}: {}".format(path, exc))
        elif url.path in STATIC:
            filename, mime = STATIC[url.path]
            path = HERE / filename
            if path.is_file():
                self.respond(200, path.read_bytes(), mime)
            else:
                self.error(404, "Frontend file is not available yet: " + filename)
        else:
            self.error(404, "Not found")

    def do_POST(self):
        if urlsplit(self.path).path != "/api/review":
            self.error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("A JSON request body is required")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            review = self.server.store.save(payload)
        except (ValueError, UnicodeError) as exc:
            self.error(400, str(exc))
            return
        except OSError as exc:
            self.error(500, "Review was not saved: " + str(exc))
            return
        self.respond(200, json_bytes({"review": review}))

    def export(self, query):
        fmt = query.get("format", ["csv"])[0]
        status = query.get("status", ["all"])[0]
        if fmt not in {"csv", "json"} or status not in {"all", "keep", "exclude", "defer"}:
            self.error(400, "Invalid export format or status")
            return
        reviews = self.server.store.snapshot()
        rows = []
        for item in self.server.data.items:
            review = reviews.get(item["id"], {})
            row = dict(item, status=review.get("status", "pending"), reason=review.get("reason", ""),
                       note=review.get("note", ""), updated_at=review.get("updated_at", ""))
            if status == "all" or row["status"] == status:
                rows.append(row)
        filename = "manual-review-{}.{}".format(status, fmt)
        if fmt == "json":
            self.respond(200, json_bytes({"items": rows, "count": len(rows)}), filename=filename)
        else:
            output = io.StringIO(newline="")
            fields = ["id", "global_row", "mmsi", "port", "class_name", "flags", "in_experiment", "roles",
                      "ais_length_m", "ais_width_m", "status", "reason", "note", "updated_at"]
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, flags=";".join(row["flags"]), roles=";".join(row["roles"])))
            self.respond(200, output.getvalue().encode("utf-8-sig"), "text/csv", filename)


def create_server(port=8765, state_dir=None, data=None):
    data = data if data is not None else ReviewData()
    store = ReviewStore(state_dir if state_dir is not None else HERE / "state", data.details)
    server = ThreadingHTTPServer(("127.0.0.1", port), ReviewHandler)
    server.data = data
    server.store = store
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", type=Path, default=HERE / "state")
    args = parser.parse_args()
    server = create_server(args.port, args.state_dir)
    print("Review server http://127.0.0.1:{} | {} | state: {}".format(
        server.server_port, server.data.counts, server.store.path), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
