"""Host-local staging for fleet stages.

The shared NFS home stays the only source of truth. A host's local disk is a
disposable cache of immutable, SHA256-identified inputs plus per-stage scratch
space. Nothing on it is edited in place or synchronized between hosts, so any
cache entry can be deleted at any time and is rebuilt on the next use.

Measured 2026-10-05: on the NFS home a small-file create costs ~200 ms and an
open 11-17 ms, against ~0.02 ms on gpu6's /local; one large sequential read ran
at 93 MB/s. Staging therefore moves bytes in a few large files and lets the
workload do its per-file I/O locally.
"""
from __future__ import annotations

import getpass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import time
from typing import Any
import uuid

LOCAL_ROOT_ENV = "PS_LOCAL_ROOT"
# Never fill a shared local disk: other users' jobs live on the same volume.
RESERVE_BYTES = 50 * 2**30
CHUNK_BYTES = 8 * 2**20
GIB = 2**30


class StagingError(RuntimeError):
    pass


def local_root() -> Path | None:
    """The host's writable PointStream cache root, or None on hosts without one.

    ``PS_LOCAL_ROOT`` overrides; otherwise ``/local/users/<user>/pointstream``,
    whose parent only an administrator can create.
    """
    configured = os.environ.get(LOCAL_ROOT_ENV)
    root = Path(configured) if configured else Path("/local/users") / getpass.getuser() / "pointstream"
    parent = root if root.is_dir() else root.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        return None
    root.mkdir(exist_ok=True)
    return root


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def admission_error(spec: dict[str, Any]) -> str | None:
    """Why this host cannot satisfy the job's declared local storage, if it cannot."""
    required = spec.get("local_storage_gib", 0)
    if not required:
        return None
    root = local_root()
    if root is None:
        return "host has no writable local storage root"
    if free_bytes(root) < required * GIB + RESERVE_BYTES:
        return f"host local storage has less than {required} GiB plus reserve free"
    return None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _make_read_only(path: Path) -> None:
    for directory, names, files in os.walk(path):
        for name in files:
            target = Path(directory) / name
            if not target.is_symlink():
                target.chmod(target.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        Path(directory).chmod(Path(directory).stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _make_writable(path: Path) -> None:
    for directory, _, _ in os.walk(path):
        Path(directory).chmod(Path(directory).stat().st_mode | stat.S_IWUSR)


def _remove(path: Path) -> None:
    if path.exists():
        _make_writable(path)
        shutil.rmtree(path)


def _copy_verified(source: Path, entry: Path, expected: str) -> Path:
    """Copy one immutable file into the cache, hashing the bytes as they stream."""
    target = entry / source.name
    partial = entry / f".partial-{uuid.uuid4().hex}"
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, partial.open("wb") as writer:
            while block := reader.read(CHUNK_BYTES):
                digest.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if digest.hexdigest() != expected:
            raise StagingError(f"input identity changed while staging: {source}")
        partial.chmod(0o444)
        # Concurrent stagers write identical verified bytes; either rename wins.
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def _extract(archive: Path, entry: Path) -> Path:
    tree = entry / "tree"
    if tree.is_dir():
        return tree
    partial = entry / f".partial-tree-{uuid.uuid4().hex}"
    try:
        with tarfile.open(archive) as bundle:
            bundle.extractall(partial, filter="data")
        _make_read_only(partial)
        try:
            partial.rename(tree)
        except OSError:
            if not tree.is_dir():
                raise
    finally:
        _remove(partial)
    return tree


def stage_input(item: dict[str, Any], root: Path | None) -> dict[str, Any]:
    """Return a verified path for one declared input, local when possible.

    Every byte is hashed against the declared SHA256 on every use: either as it
    streams from NFS into the cache, or from the local cache entry on a hit.
    """
    source = Path(item["path"])
    started = time.time()
    record: dict[str, Any] = {"name": item["name"], "source": str(source), "sha256": item["sha256"], "extract": item.get("extract", False)}
    if root is not None:
        entry = root / "cache" / item["sha256"]
        cached = entry / source.name
        size = source.stat().st_size
        if cached.is_file():
            if _hash_file(cached) != item["sha256"]:
                # A corrupted cache entry is disposable; NFS stays canonical.
                _remove(entry)
            else:
                record["mode"] = "local-hit"
        if "mode" not in record:
            reserve = size * (2 if record["extract"] else 1) + RESERVE_BYTES
            if free_bytes(root) >= reserve:
                entry.mkdir(parents=True, exist_ok=True)
                cached = _copy_verified(source, entry, item["sha256"])
                record["mode"] = "local-copy"
        if "mode" in record:
            record["path"] = str(_extract(cached, entry) if record["extract"] else cached)
            record["seconds"] = time.time() - started
            return record
    if record["extract"]:
        raise StagingError(f"extracting {item['name']} requires local storage; declare local_storage_gib")
    if _hash_file(source) != item["sha256"]:
        raise StagingError(f"input identity changed: {source}")
    record.update(mode="shared", path=str(source), seconds=time.time() - started)
    return record


def stage_environment(item: dict[str, Any], root: Path | None) -> dict[str, Any]:
    """Extract a packed environment locally and prove its interpreter runs from there."""
    record = stage_input({"name": "environment", **item, "extract": True}, root)
    prefix = Path(record["path"])
    python = prefix / "bin" / "python"
    probe = subprocess.run([str(python), "-c", "import sys; print(sys.prefix)"], capture_output=True, text=True, timeout=120, check=False)
    reported = probe.stdout.strip()
    if probe.returncode or not reported or Path(reported).resolve() != prefix.resolve():
        raise StagingError(f"staged interpreter does not run from its local prefix: {reported or probe.stderr.strip()[-300:]}")
    return {**record, "prefix": str(prefix), "python": str(python), "bin": str(prefix / "bin")}


def stage_inputs(items: list[dict[str, Any]], root: Path | None) -> list[dict[str, Any]]:
    return [stage_input(item, root) for item in items]


def scratch_directory(root: Path | None, job_id: str, stage: str, output: Path) -> Path:
    """Per-stage workspace: local when the host has a root, else beside the outputs."""
    directory = (root / "scratch" / f"{job_id}-{stage}") if root is not None else (output / "scratch")
    directory.mkdir(parents=True)
    return directory


def publish_scratch(scratch: Path, output: Path) -> dict[str, Any] | None:
    """Send ``scratch/publish`` back to the shared stage directory as one archive."""
    selected = scratch / "publish"
    if not selected.is_dir():
        return None
    archive = output / "published.tar"
    with tarfile.open(archive, "w") as bundle:
        bundle.add(selected, arcname="publish")
    with tarfile.open(archive) as bundle:
        members = len(bundle.getmembers())
    return {"path": str(archive), "sha256": _hash_file(archive), "bytes": archive.stat().st_size, "members": members}


def release_scratch(scratch: Path, output: Path) -> None:
    # Scratch beside the outputs is already shared and preserved with them.
    if not scratch.is_relative_to(output):
        _remove(scratch)
