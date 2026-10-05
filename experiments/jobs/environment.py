"""Pack a Python environment into one immutable archive for host-local staging.

A cold ``import torch`` from the NFS-hosted environment took 70 s on gpu6 and
669 s on gpu5 (2026-10-05): it opens ~1,900 files at NFS latency. A job that
declares a packed ``environment`` runs from a verified local extraction instead.

Conda binaries resolve their libraries through ``$ORIGIN`` RPATHs and Python
derives ``sys.prefix`` from its executable, so an extracted copy runs in place
without rewriting prefixes. Console-script shebangs still name the original
prefix; fleet stages call the interpreter directly and never depend on them.
Editable installs keep resolving to their recorded source trees.

Standard library only: this runs on hosts with the system interpreter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

CHUNK_BYTES = 8 * 2**20


class PackError(RuntimeError):
    pass


def package_state(prefix: Path) -> dict[str, Any]:
    """Installed conda and pip distributions, with metadata identities.

    Any install or removal during packing changes this, which makes the archive
    an unknown mixture; packing then refuses to publish it.
    """
    meta = prefix / "conda-meta"
    conda = sorted(p.name for p in meta.glob("*.json")) if meta.is_dir() else []
    pip = sorted(p.name for p in prefix.glob("lib/python*/site-packages/*.dist-info"))
    digest = hashlib.sha256()
    for path in sorted([*(meta / n for n in conda), *prefix.glob("lib/python*/site-packages/*.dist-info/RECORD")]):
        info = path.stat()
        digest.update(f"{path.relative_to(prefix)}\0{info.st_size}\0{info.st_mtime_ns}\n".encode())
    return {"conda": conda, "pip": pip, "metadata_sha256": digest.hexdigest()}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def pack(prefix: Path, output_dir: Path, work_dir: Path) -> dict[str, Any]:
    """Archive ``prefix`` in ``work_dir`` (fast local disk), then publish to ``output_dir``."""
    prefix = prefix.resolve()
    if not (prefix / "bin" / "python").exists():
        raise PackError(f"not a Python environment: {prefix}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{prefix.name}-{stamp}"
    before = package_state(prefix)
    work_dir.mkdir(parents=True, exist_ok=True)
    partial = work_dir / f"{name}.tar.gz.partial"
    started = datetime.now(timezone.utc)
    try:
        with partial.open("wb") as sink:
            tar = subprocess.Popen(["tar", "-C", str(prefix), "--numeric-owner", "-cf", "-", "."], stdout=subprocess.PIPE)
            assert tar.stdout is not None
            gzip = subprocess.run(["gzip", "-1"], stdin=tar.stdout, stdout=sink, check=False)
            tar.stdout.close()
            if tar.wait() or gzip.returncode:
                raise PackError(f"archiving failed: tar {tar.returncode}, gzip {gzip.returncode}")
        if package_state(prefix) != before:
            raise PackError("environment packages changed while packing; archive discarded")
        sha256 = _hash(partial)
        manifest: dict[str, Any] = {
            "archive": f"{name}.tar.gz", "sha256": sha256, "bytes": partial.stat().st_size,
            "source_prefix": str(prefix), "host": socket.getfqdn(), "started": started.isoformat(),
            "finished": datetime.now(timezone.utc).isoformat(), "packages": before,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / manifest["archive"]
        published = output_dir / f".{manifest['archive']}.partial"
        with partial.open("rb") as reader, published.open("wb") as writer:
            while block := reader.read(CHUNK_BYTES):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if _hash(published) != sha256:
            published.unlink()
            raise PackError("published archive does not match the packed bytes")
        published.chmod(0o444)
        os.replace(published, target)
        (output_dir / f"{name}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {**manifest, "path": str(target)}
    finally:
        partial.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack")
    packing.add_argument("--prefix", type=Path, required=True)
    packing.add_argument("--output-dir", type=Path, required=True, help="shared directory under the data root")
    packing.add_argument("--work-dir", type=Path, required=True, help="host-local directory for the temporary archive")
    args = parser.parse_args(argv)
    result = pack(args.prefix, args.output_dir, args.work_dir)
    print(json.dumps({k: result[k] for k in ("path", "sha256", "bytes")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
