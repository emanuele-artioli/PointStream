"""Standard-library-only, killable hash/copy child for bounded NFS reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


def _stat_identity(path: Path) -> dict[str, int]:
    info = os.stat(path)
    return {
        "bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
        "inode": info.st_ino, "device": info.st_dev,
    }


def _stream_hash(path: Path, copy_to: Path | None = None, deadline: float | None = None) -> dict[str, Any]:
    """Hash in 1 MiB blocks; optionally write the same bytes to a copy."""
    before = _stat_identity(path)
    digest = hashlib.sha256()
    count = 0
    target = copy_to.open("xb") if copy_to is not None else None
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
                count += len(block)
                if target is not None:
                    target.write(block)
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"bounded read exceeded its deadline: {path}")
    finally:
        if target is not None:
            target.close()
    after = _stat_identity(path)
    if before != after or count != after["bytes"]:
        raise RuntimeError(f"file changed while it was read: {path}")
    return {"sha256": digest.hexdigest(), **after}


def _child(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="bounded child reads for background_smoke")
    commands = parser.add_subparsers(dest="action", required=True)
    one = commands.add_parser("hash-one")
    one.add_argument("path", type=Path)
    copy = commands.add_parser("copy-one")
    copy.add_argument("source", type=Path)
    copy.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    if args.action == "hash-one":
        print(json.dumps(_stream_hash(args.path)))
        return 0
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(_stream_hash(args.source, copy_to=args.destination)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_child(sys.argv[1:]))
