"""Read-only proof for an interrupted, unpublished source transfer."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tarfile


def verify_prepared_source(directory: Path, expected_sha256: str) -> dict:
    """Require an exact completed archive tree; never extract or replay work."""
    for name in ("ready.json", "spec.json", "owner", "run"):
        if (directory / name).exists():
            raise ValueError("published or owned request requires ordinary status reconciliation")
    proc_root = Path("/proc")
    if proc_root.exists():
        for proc in proc_root.iterdir():
            if not proc.name.isdigit() or int(proc.name) in (os.getpid(), os.getppid()):
                continue
            try:
                if proc.stat().st_uid != os.getuid():
                    continue
                command = (proc / "cmdline").read_bytes()
                if str(directory).encode() in command:
                    raise ValueError("owned process still references the prepared request")
            except OSError:
                continue
    archive = directory / "source.tar"
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("received archive identity mismatch")
    source = directory / "source"
    expected = set()
    with tarfile.open(archive) as bundle:
        for member in bundle:
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts or member.issym() or member.islnk():
                raise ValueError("unsafe archived path")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("unsupported archive entry")
            target = source / rel
            if target.is_symlink() or not target.is_file() or target.stat().st_size != member.size:
                raise ValueError("source extraction is incomplete")
            extracted = bundle.extractfile(member)
            assert extracted is not None
            with target.open("rb") as actual:
                for block in iter(lambda: extracted.read(1024 * 1024), b""):
                    if actual.read(len(block)) != block:
                        raise ValueError("extracted source identity mismatch")
            expected.add(rel.as_posix())
    actual_files = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    if actual_files != expected:
        raise ValueError("unexpected files in prepared source")
    return {
        "archive_sha256": expected_sha256,
        "verified_files": len(expected),
        "unpublished": True,
        "no_replay": True,
    }
