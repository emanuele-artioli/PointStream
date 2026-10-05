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
GIB = 2**30
# Never fill a shared local disk: other users' jobs live on the same volume.
RESERVE_BYTES = 50 * GIB
# A RAM-backed root must always leave this much memory available to everyone.
RAM_MARGIN_MIN_BYTES = 64 * GIB
RAM_MARGIN_FRACTION = 0.25
CHUNK_BYTES = 8 * 2**20


class StagingError(RuntimeError):
    pass


def _candidates() -> list[Path]:
    configured = os.environ.get(LOCAL_ROOT_ENV)
    if configured:
        return [Path(configured)]
    user = getpass.getuser()
    # /local/users/<user> exists only where an administrator created it (gpu6 on
    # 2026-10-05); every host has a large RAM-backed /dev/shm.
    return [Path("/local/users") / user / "pointstream", Path("/dev/shm") / f"{user}-pointstream"]


def local_root() -> Path | None:
    """The host's writable PointStream cache root, or None on hosts without one.

    ``PS_LOCAL_ROOT`` overrides. Otherwise prefer ``/local/users/<user>/pointstream``
    and fall back to ``/dev/shm/<user>-pointstream``. systemd removes a user's
    /dev/shm files once none of their processes remain on the host, so that cache
    lives exactly as long as our workers and jobs do.
    """
    for root in _candidates():
        parent = root if root.is_dir() else root.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            continue
        root.mkdir(mode=0o700, exist_ok=True)
        # /dev/shm is world-writable: never use a directory someone else created.
        if root.stat().st_uid == os.getuid():
            return root
    return None


def ram_backed(path: Path) -> bool:
    """Whether ``path`` lives on tmpfs/ramfs, by the longest matching mount point."""
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return False
    target, best, kind = str(path.resolve()), -1, ""
    for line in mounts:
        fields = line.split()
        if len(fields) < 3:
            continue
        point = fields[1].replace("\\040", " ")
        if (target == point or target.startswith(point.rstrip("/") + "/")) and len(point) > best:
            best, kind = len(point), fields[2]
    return kind in {"tmpfs", "ramfs"}


def _memory() -> tuple[int, int]:
    """(MemTotal, MemAvailable) in bytes."""
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, rest = line.partition(":")
        values[name] = int(rest.split()[0]) * 1024
    return values["MemTotal"], values["MemAvailable"]


def available_bytes(root: Path) -> int:
    """Bytes we may still add under ``root`` while keeping the safety margin free."""
    free = shutil.disk_usage(root).free
    if not ram_backed(root):
        return free - RESERVE_BYTES
    total, available = _memory()
    margin = max(RAM_MARGIN_MIN_BYTES, int(total * RAM_MARGIN_FRACTION))
    return min(free, available - margin)


def admission_error(spec: dict[str, Any]) -> str | None:
    """Why this host cannot satisfy the job's declared local storage, if it cannot."""
    required = spec.get("local_storage_gib", 0)
    if not required:
        return None
    root = local_root()
    if root is None:
        return "host has no writable local storage root"
    if available_bytes(root) < required * GIB:
        return f"host local storage cannot add {required} GiB and keep its safety margin"
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


def _extract(archive: Path, entry: Path, root: Path) -> Path:
    tree = entry / "tree"
    if tree.is_dir():
        return tree
    partial = entry / f".partial-tree-{uuid.uuid4().hex}"
    try:
        with tarfile.open(archive) as bundle:
            # Compressed archives expand several-fold; check the real size first.
            needed = sum(member.size for member in bundle.getmembers())
            if available_bytes(root) < needed:
                raise StagingError(f"extracting {archive.name} needs {needed / GIB:.1f} GiB beyond the local safety margin")
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
            if available_bytes(root) >= size:
                entry.mkdir(parents=True, exist_ok=True)
                cached = _copy_verified(source, entry, item["sha256"])
                record["mode"] = "local-copy"
        if "mode" in record:
            record["path"] = str(_extract(cached, entry, root) if record["extract"] else cached)
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


def scratch_directory(root: Path | None, job_id: str, stage: str, output: Path, *, declared: bool) -> Path:
    """Per-stage workspace: local when the host has a root, else beside the outputs.

    Scratch writes are unbounded, so RAM-backed scratch is used only by jobs whose
    declared ``local_storage_gib`` admission confirmed fits within the margin.
    """
    local = root is not None and (declared or not ram_backed(root))
    directory = (root / "scratch" / f"{job_id}-{stage}") if local and root is not None else (output / "scratch")
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
