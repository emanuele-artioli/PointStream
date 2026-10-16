"""Serve a static K8 review on loopback, including byte ranges for source-video seeking.

    python tools/datasets/k8_review_server.py --root /.../tennis/k8-DATE --port 17688

Read-only HTTP: marks are downloaded by the browser, never posted to the server.
"""
from __future__ import annotations

import argparse
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def byte_range(header: str, size: int) -> tuple[int, int]:
    """One HTTP byte range, as a half-open interval; reject unsatisfiable ranges."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if match is None or not any(match.groups()) or size <= 0:
        raise ValueError("invalid byte range")
    first, last = match.groups()
    if first:
        start, stop = int(first), min(size, int(last) + 1) if last else size
    else:
        suffix = int(last)
        start, stop = max(0, size - suffix), size
    if start >= stop or start < 0:
        raise ValueError("unsatisfiable byte range")
    return start, stop


class RangeHandler(SimpleHTTPRequestHandler):
    bounds: tuple[int, int] | None = None

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self) -> Any:
        self.bounds = None
        path = Path(self.translate_path(self.path))
        header = self.headers.get("Range")
        if header is None or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        try:
            self.bounds = byte_range(header, size)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        start, stop = self.bounds
        handle = path.open("rb")
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{stop - 1}/{size}")
        self.send_header("Content-Length", str(stop - start))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return handle

    def copyfile(self, source: Any, outputfile: Any) -> None:
        if self.bounds is None:
            return super().copyfile(source, outputfile)
        start, stop = self.bounds
        source.seek(start)
        remaining = stop - start
        while remaining:
            data = source.read(min(remaining, 1 << 20))
            if not data:
                break
            outputfile.write(data)
            remaining -= len(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=17688)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("review root does not exist")
    handler = partial(RangeHandler, directory=str(args.root.resolve()))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Review: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
