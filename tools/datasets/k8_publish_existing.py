"""Finish source publication for audited, never-run K8 requests; never creates new IDs.

Use only after canonical fleet status and shared-inbox inspection. Partial
transfers are preserved. The original spec/snapshot identities must match a
verified, already-published source archive before the frozen fleet publisher runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_unpublished(directory: Path) -> None:
    if (directory / "ready.json").exists() or any((directory / p).exists() for p in ("owner", "run", "state.json")):
        raise ValueError(f"{directory.name}: published/claimed/executed state; inspect, do not repair")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--trusted-job", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text())
    if digest(args.archive) != payload["archive_sha256"]:
        raise ValueError("pinned source archive checksum mismatch")
    with tarfile.open(args.archive) as archive:
        for member in archive.getmembers():
            if Path(member.name).is_absolute() or ".." in Path(member.name).parts or not (member.isfile() or member.isdir()):
                raise ValueError("unsafe source archive member")
    sys.path.insert(0, str(args.release))
    from experiments.jobs import inbox
    trusted = json.loads((args.base / "inbox" / args.trusted_job / "ready.json").read_text())
    if trusted["snapshot_sha256"] != payload["archive_sha256"]:
        raise ValueError("trusted published snapshot differs")
    receipt: list[dict[str, Any]] = []
    # The failed clients have exited. A lingering transfer process requires inspection.
    processes = subprocess.check_output(["ps", "-u", str(os.getuid()), "-o", "pid=,args="], text=True)
    for request in payload["requests"]:
        job = request["job_id"]
        if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", job):
            raise ValueError("unsafe job ID")
        if request["snapshot"]["snapshot_sha256"] != payload["archive_sha256"]:
            raise ValueError("request source differs")
        if any(job in line for line in processes.splitlines()):
            raise ValueError(f"{job}: transfer process may remain; inspect")
        directory = args.base / "inbox" / job
        assert_unpublished(directory)
        if not directory.exists():
            inbox.rpc(args.base, "prepare", {"job_id": job})
        source = directory / "source"
        existing_identity = inbox.source_identity(source) if source.is_dir() else None
        changed = existing_identity != trusted["source_sha256"]
        if changed:
            backup = directory / "source-before-publication-repair"
            if backup.exists():
                raise ValueError(f"{job}: prior repair exists; inspect before continuing")
            if source.exists():
                source.rename(backup)
            source.mkdir()
            subprocess.run(["tar", "-xf", str(args.archive), "-C", str(source)], check=True, timeout=90)
        if inbox.source_identity(source) != trusted["source_sha256"]:
            raise ValueError("restored source identity mismatch")
        assert_unpublished(directory)
        result = inbox.rpc(args.base, "publish", request)
        receipt.append({"job_id": job, "group": request["group"], "source_restored": changed,
                        "original_id_preserved": True, "result": result})
        args.receipt.write_text(json.dumps(receipt, indent=1) + "\n")
        print(json.dumps(receipt[-1]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
