"""Dependency-free loopback review UI with range streaming and immutable decisions."""
from __future__ import annotations
import json
import logging
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import secrets
import threading
import uuid
from urllib.parse import urlparse

from clipiq.cache import atomic_json, fingerprint
from clipiq.media.production import render, validate_interval

LOG = logging.getLogger(__name__)


class ReviewState:
    def __init__(self, output):
        self.output = Path(output).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.busy = False
        self.error = None
        self.manifest = None
        latest = self.output / "latest.json"
        if latest.exists():
            path = Path(json.loads(latest.read_text())["manifest"]).resolve()
            if not path.is_relative_to(self.output / "runs"):
                raise ValueError("Latest manifest is outside the output runs directory")
            self.manifest = json.loads(path.read_text())

    def snapshot(self):
        with self.lock:
            decisions = {}
            if self.manifest:
                folder = Path(self.manifest["output_dir"]) / "reviews"
                for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime_ns):
                    try:
                        item = json.loads(path.read_text())
                        decisions[item["clip_id"]] = item
                    except (OSError, ValueError, KeyError):
                        continue
            return {"busy": self.busy, "error": self.error, "manifest": self.manifest, "decisions": decisions}

    def start(self, data):
        from clipiq.application import run
        with self.lock:
            if self.busy:
                raise ValueError("A job is already running")
            video = Path(data["video"]).expanduser().resolve()
            if not video.is_file():
                raise ValueError(f"Video not found: {video}")
            self.busy, self.error = True, None
        def worker():
            try:
                result = run(video, self.output, top_k=int(data.get("top_k", 5)),
                    max_duration=float(data.get("max_duration", 60)),
                    no_speech=bool(data.get("no_speech", False)),
                    transcript=Path(data["transcript"]).expanduser() if data.get("transcript") else None,
                    legacy_dir=Path(data["legacy_dir"]).expanduser() if data.get("legacy_dir") else None)
                with self.lock:
                    self.manifest = result
            except Exception as exc:
                LOG.exception("Processing failed")
                with self.lock:
                    self.error = str(exc)
            finally:
                with self.lock:
                    self.busy = False
        threading.Thread(target=worker, daemon=True).start()

    def decide(self, data):
        with self.lock:
            if self.busy:
                raise ValueError("Wait for the current job to finish")
            if not self.manifest or data.get("run_id") != self.manifest["run_id"]:
                raise ValueError("This review is stale; refresh the page")
            manifest = self.manifest
            row = next((r for r in manifest["clips"] if r["clip_id"] == data["clip_id"]), None)
            if row is None or data["action"] not in {"approve", "reject", "preview"}:
                raise ValueError("Unknown clip or review action")
            start, end = float(data["start"]), float(data["end"])
            validate_interval(start, end, manifest["video_duration"], manifest["parameters"]["max_duration"])
            self.busy = True
        try:
            decision = {"decision_id": uuid.uuid4().hex, "run_id": manifest["run_id"],
                        "clip_id": row["clip_id"], "action": data["action"], "start": start, "end": end,
                        "original_proposal": {"start": row["start"], "end": row["end"]}}
            folder = Path(manifest["output_dir"]) / "reviews"
            if data["action"] != "reject":
                if fingerprint(Path(manifest["video_path"])) != manifest["source_fingerprint"]:
                    raise ValueError("Source video changed since analysis; start a new run")
                decision["export"] = render(Path(manifest["video_path"]), start, end,
                    folder / f"{decision['decision_id']}.mp4", duration=manifest["video_duration"],
                    maximum=manifest["parameters"]["max_duration"])
            atomic_json(folder / f"{decision['decision_id']}.json", decision)
            return decision
        finally:
            with self.lock:
                self.busy = False


def make_server(output, port=8765):
    state = ReviewState(output)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            LOG.debug(fmt, *args)

        def allowed_host(self):
            port = self.server.server_port
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def send_json(self, status, data):
            content = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(content)

        def do_POST(self):
            if not self.allowed_host() or self.headers.get("X-ClipIQ-Token") != state.token:
                return self.send_json(403, {"error": "Local session token required"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1000000:
                    raise ValueError("Invalid request length")
                data = json.loads(self.rfile.read(length))
                route = urlparse(self.path).path
                if route == "/api/run":
                    state.start(data)
                    return self.send_json(202, {"started": True})
                if route == "/api/review":
                    return self.send_json(200, state.decide(data))
                return self.send_json(404, {"error": "Unknown endpoint"})
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                return self.send_json(400, {"error": str(exc)})

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if not self.allowed_host():
                return self.send_json(403, {"error": "Use localhost or 127.0.0.1"})
            route = urlparse(self.path).path
            if route == "/":
                content = Path(__file__).with_name("review.html").read_text().replace("__TOKEN__", state.token).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Options", "DENY")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(content)
                return
            if route == "/api/videos":
                from clipiq.config import RAW_DIR
                videos = sorted(str(p.resolve()) for p in RAW_DIR.rglob("*")
                                if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"})
                return self.send_json(200, videos[:200])
            if route == "/api/state":
                return self.send_json(200, state.snapshot())
            try:
                manifest = state.snapshot()["manifest"]
                if not manifest:
                    raise ValueError("No run loaded")
                if route == "/media/source":
                    path = Path(manifest["video_path"])
                elif route.startswith("/media/clip/"):
                    clip_id = route.rsplit("/", 1)[-1]
                    row = next(r for r in manifest["clips"] if r["clip_id"] == clip_id)
                    path = Path(row["export"]["path"])
                elif re.fullmatch(r"/media/review/[a-f0-9]{32}", route):
                    path = Path(manifest["output_dir"]) / "reviews" / (route.rsplit("/",1)[-1] + ".mp4")
                else:
                    raise ValueError("Unknown media")
                if route != "/media/source" and not path.resolve().is_relative_to(state.output):
                    raise ValueError("Media is outside the output folder")
                return self.stream(path)
            except (ValueError, KeyError, OSError, StopIteration):
                return self.send_json(404, {"error": "Media not found"})

        def stream(self, path):
            size = path.stat().st_size
            start, end, partial = 0, size-1, False
            header = self.headers.get("Range")
            if header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
                if not match or not any(match.groups()):
                    return self.send_json(416, {"error": "Unsupported range"})
                a,b = match.groups()
                if a:
                    start, end = int(a), min(size-1, int(b)) if b else size-1
                else:
                    start = max(0, size-int(b))
                if start > end or start >= size:
                    return self.send_json(416, {"error": "Range outside file"})
                partial = True
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end-start+1))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with path.open("rb") as handle:
                    handle.seek(start)
                    remaining = end-start+1
                    while remaining:
                        block = handle.read(min(65536, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.state = state
    return server


def serve(output, port=8765):
    server = make_server(output, port)
    print(f"ClipIQ review: http://127.0.0.1:{server.server_port}", flush=True)
    print("Press Ctrl+C to stop. All processing and review files remain local.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
