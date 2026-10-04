"""Bounded read-only export of artifacts from a recorded fleet job."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from experiments.jobs import fleet

MAX_BYTES = 512 * 1024
MAX_FILES = 12
READ_TIMEOUT = 20

# Fixed manager code, not a user-supplied workload or a new worker release.
READER = r'''
import base64, hashlib, json, pathlib, sys, time
started = time.monotonic()
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
relative = pathlib.PurePosixPath(sys.argv[2])
limit = int(sys.argv[3])
if relative.is_absolute() or '..' in relative.parts:
    raise ValueError('unsafe artifact path')
path = (root / str(relative)).resolve()
if root not in path.parents:
    raise ValueError('artifact symlink escapes job directory')
if not path.exists():
    print(json.dumps({'path': str(relative), 'status': 'missing', 'seconds': time.monotonic()-started}))
else:
    before = path.stat()
    if not path.is_file():
        raise ValueError('artifact is not a regular file')
    with path.open('rb') as stream:
        data = stream.read(limit+1)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('artifact changed during read')
    truncated = len(data) > limit
    data = data[:limit]
    print(json.dumps({'path': str(relative), 'status': 'read', 'source_bytes': before.st_size,
        'truncated': truncated, 'sha256': None if truncated else hashlib.sha256(data).hexdigest(),
        'base64': base64.b64encode(data).decode('ascii'), 'seconds': time.monotonic()-started}))
'''


def validate_paths(paths: list[str]) -> None:
    if not 1 <= len(paths) <= MAX_FILES or len(set(paths)) != len(paths):
        raise fleet.FleetError("select 1-12 distinct job artifacts")
    metadata = {"ready.json", "spec.json", "environment.json", "state.json", "gate.json", "validation.json", "campaign-error.json", "status.json", "manifest.json"}
    for value in paths:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not path.parts or "\x00" in value:
            raise fleet.FleetError("unsafe artifact path")
        if value not in metadata and path.parts[0] not in {"run", "smoke", "full", "canvas"}:
            raise fleet.FleetError("artifact must be job metadata or inside run/smoke/full/canvas")
        if path.suffix not in {".json", ".log", ".txt", ".jpg", ".png"}:
            raise fleet.FleetError("unsupported artifact type")


def export(record: dict[str, Any], paths: list[str], destination: Path) -> dict[str, Any]:
    validate_paths(paths)
    if destination.exists():
        raise fleet.FleetError("artifact export destination already exists")
    job = record["job_id"]
    if not fleet.JOB_ID_RE.fullmatch(job):
        raise fleet.FleetError("invalid recorded job ID")
    root = record.get("run_dir") or str(Path(record["config"]["base"]) / "inbox" / job)
    location = Path(root)
    if not location.is_absolute() or ".." in location.parts or location.parts[-4:] not in {("jobs", "fleet", "runs", job), ("jobs", "fleet", "inbox", job)}:
        raise fleet.FleetError("artifact root is not the recorded fleet job directory")
    alias = record.get("host_alias") or record["config"]["hosts"][0]
    if alias not in fleet.DEFAULT_HOSTS:
        raise fleet.FleetError("artifact host is not in the configured fleet")
    rows = []
    started = time.monotonic()
    for relative in paths:
        result = fleet._ssh(alias, ["/usr/bin/python3", "-c", READER, root, relative, str(MAX_BYTES)], timeout=READ_TIMEOUT)
        if result.returncode:
            raise fleet.FleetError(f"artifact read failed: {relative}: {result.stderr[-500:]}")
        row = json.loads(result.stdout.strip().splitlines()[-1])
        if row.get("path") != relative:
            raise fleet.FleetError("artifact reply path differs from selected path")
        if row.get("status") == "read":
            data = base64.b64decode(row["base64"], validate=True)
            if len(data) > MAX_BYTES or (not row["truncated"] and hashlib.sha256(data).hexdigest() != row["sha256"]):
                raise fleet.FleetError("artifact transport identity mismatch")
        rows.append(row)
    destination.mkdir(parents=True, exist_ok=False)
    for row in rows:
        if row.get("status") == "read":
            target = destination / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(row.pop("base64")))
            target.chmod(0o444)
    receipt = {"job_id": job, "host": alias, "root": root, "read_only": True,
               "seconds": time.monotonic()-started, "files": rows}
    target = destination / "export-receipt.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    target.chmod(0o444)
    return {"job_id": job, "destination": str(destination), "files": len(rows), "seconds": receipt["seconds"], "read_only": True}
