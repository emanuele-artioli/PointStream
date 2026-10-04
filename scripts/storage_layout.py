"""Plan and reconcile an attended three-root storage migration.

Run check-hosts on the Mac; plan/apply/rollback run on the shared Linux home.
Only same-filesystem, no-clobber renames are permitted. Old paths outside the
code checkout become compatibility aliases. Never removes datasets or weights.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time
from typing import Any
import uuid

HOSTS = tuple(f"gpu{i}" for i in range(1, 7))
SCHEMA = "pointstream.storage-plan.v1"


class MigrationError(RuntimeError):
    pass


def identity(path: Path) -> dict[str, int]:
    value = path.lstat()
    return {
        key: int(getattr(value, f"st_{key}")) for key in ("dev", "ino", "mode", "size", "mtime_ns")
    }


def same_inode(path: Path, expected: dict[str, int]) -> bool:
    try:
        value = path.stat()
        return value.st_dev == expected["dev"] and value.st_ino == expected["ino"]
    except OSError:
        return False


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_plan(home: Path) -> dict[str, Any]:
    home = home.resolve()
    data = home / "Datasets"
    old = data / "pointstream-data"
    moves: list[dict[str, Any]] = []
    conflicts: list[dict[str, str]] = []
    aliases: list[str] = []
    planned: dict[Path, Path] = {}

    def backing(path: Path) -> Path:
        if path.exists() or path.is_symlink():
            return path
        if path in planned:
            return planned[path]
        for parent in path.parents:
            if parent in planned:
                return planned[parent] / path.relative_to(parent)
        return path

    def queue(source: Path, target: Path) -> None:
        if source.is_symlink():
            if source.resolve() == target.resolve():
                aliases.append(str(source))
            else:
                conflicts.append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "reason": "source is an unrecognized alias",
                    }
                )
            return
        if not source.exists():
            return
        effective = backing(target)
        if effective.exists() or effective.is_symlink():
            if source.is_dir() and effective.is_dir() and not (source / ".git").exists():
                for child in sorted(source.iterdir()):
                    queue(child, target / child.name)
                return
            conflicts.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "reason": "destination exists; no overwrite or automatic deduplication",
                }
            )
            return
        item = {"source": str(source), "target": str(target), "identity": identity(source)}
        moves.append(item)
        planned[target] = source

    # Promote existing categories, without another project-data umbrella.
    for name in ("assets", "audits", "jobs", "manifests", "outputs"):
        queue(old / name, data / name)
    for name in ("HOPformer", "dinov3", "deltadorsal"):
        queue(old / "third_party" / name, home / "Models" / name)
    queue(old / "third_party/detectron2", home / "detectron2")
    queue(old / "third_party/gvcrt-qualified-20261001", data / "gvcrt-qualified-20261001")
    for name in ("DCVC", "DiffuEraser", "HNeRV"):
        queue(old / "jobs/neural-bg/src" / name, home / "Models" / name)
    family_names = {"densepose": "DensePose", "hopformer": "HOPformer"}
    weight_root = old / "weights"
    if weight_root.exists():
        for source in sorted(weight_root.iterdir()):
            if source.name == "logs":
                queue(source, data / "audits/model-download-logs")
            else:
                queue(source, home / "Models" / family_names.get(source.name, source.name))
    asset_weights = old / "assets/weights"
    if asset_weights.exists():
        for source in sorted(asset_weights.iterdir()):
            if source.name == ".gitkeep":
                continue
            if source.suffix in {".txt", ".log"}:
                queue(source, data / "audits" / source.name)
                continue
            family = next(
                (
                    family
                    for prefix, family in (
                        ("yolo", "YOLO"),
                        ("mobileclip", "YOLO"),
                        ("sam", "SAM"),
                        ("fastsam", "SAM"),
                        ("i3d", "i3d"),
                        ("vgg", "vgg19"),
                        ("pix2pix", "pix2pix"),
                        ("spade4tennis", "spade4tennis"),
                    )
                    if source.name.lower().startswith(prefix)
                ),
                None,
            )
            target = home / "Models"
            if family and source.is_file():
                target /= family
            queue(source, target / source.name)
    queue(old / "assets/animate-anyone/profiles", home / "Models/AnimateAnyone/profiles")
    queue(old / "jobs/factory-bg-rd/checkpoints", home / "Models/HNeRV/checkpoints/factory-bg-rd")
    return {
        "schema": SCHEMA,
        "home": str(home),
        "created_at": time.time(),
        "moves": moves,
        "conflicts": conflicts,
        "existing_aliases": aliases,
        "retained": [
            "Datasets/pointstream-demo (named corpus)",
            "tracked documentation, historic manifests, and shipped demo assets",
            "active/paused checkouts and the manuscript repository",
        ],
    }


# Output contains no command lines or environment values.
REMOTE_PROBE = r"""
import json,os,socket
from pathlib import Path
home=Path(HOME_PATH)
probe=Path(PROBE_PATH)
visible=probe.is_file() and probe.read_text()==TOKEN
busy=[]
skip={os.getpid(),os.getppid()}
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name) in skip: continue
 try:
  if p.stat().st_uid != os.getuid(): continue
  command=(p/'cmdline').read_bytes()
  cwd=str((p/'cwd').resolve())
  relevant=any(str(home/x).encode() in command or str(home/x) in cwd for x in ('pointstream','Datasets','Models'))
  if relevant:
   busy.append({'pid':int(p.name),'name':(p/'comm').read_text().strip()})
 except (OSError,ValueError): pass
print(json.dumps({'host':socket.gethostname(),'shared_visible':visible,'busy':busy}))
"""


def ssh(host: str, code: str) -> str:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        host,
        shlex.join(["python3", "-c", code]),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=25)
    if result.returncode:
        raise MigrationError(result.stderr.strip() or f"SSH {host} exited {result.returncode}")
    return result.stdout


def check_hosts(home: Path, via: str) -> dict[str, Any]:
    token = uuid.uuid4().hex
    probe = home / "Datasets" / f".ps-storage-{token}"
    create = f"from pathlib import Path; p=Path({str(probe)!r}); f=p.open('x'); f.write({token!r}); f.flush(); __import__('os').fsync(f.fileno()); f.close()"
    ssh(via, create)
    try:
        code = (
            f"HOME_PATH={str(home)!r}\nPROBE_PATH={str(probe)!r}\nTOKEN={token!r}\n" + REMOTE_PROBE
        )

        def query(host: str) -> dict[str, Any]:
            try:
                return json.loads(ssh(host, code)) | {"alias": host}
            except (OSError, ValueError, subprocess.SubprocessError, MigrationError) as exc:
                return {"alias": host, "error": str(exc), "shared_visible": False, "busy": []}

        with ThreadPoolExecutor(max_workers=6) as pool:
            hosts = list(pool.map(query, HOSTS))
        return {
            "home": str(home),
            "checked_at": time.time(),
            "hosts": hosts,
            "ready": all(
                h["shared_visible"] and not h["busy"] and not h.get("error") for h in hosts
            ),
        }
    finally:
        ssh(
            via,
            f"from pathlib import Path; p=Path({str(probe)!r}); assert p.read_text()=={token!r}; p.unlink()",
        )


def validate_ready(report: dict[str, Any], home: Path) -> None:
    if report.get("home") != str(home) or not 0 <= time.time() - report.get("checked_at", 0) <= 60:
        raise MigrationError("need a matching host report less than 60 seconds old")
    hosts = report.get("hosts", [])
    if {h.get("alias") for h in hosts} != set(HOSTS) or len(hosts) != 6:
        raise MigrationError("all six host checks are required")
    if any(not h.get("shared_visible") or h.get("busy") or h.get("error") for h in hosts):
        raise MigrationError(
            "a host is unreachable, cannot see shared storage, or has an active reader/writer"
        )


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_noreplace(source: Path, target: Path) -> None:
    """Use the OS no-replace primitive; never fall back to overwriting rename."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux" and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = function(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = function(os.fsencode(source), os.fsencode(target), 4)
    else:
        raise MigrationError("this filesystem/platform needs an atomic no-replace rename")
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))
    sync_directory(source.parent)
    sync_directory(target.parent)


def validate_plan(plan: dict[str, Any], journal: Path) -> Path:
    if plan.get("schema") != SCHEMA:
        raise MigrationError("unknown plan schema")
    home = Path(plan["home"]).resolve()
    if plan.get("conflicts"):
        raise MigrationError("resolve every destination collision before applying this plan")
    if not home.is_dir() or not (home / "Datasets").is_dir() or not (home / "Models").is_dir():
        raise MigrationError("the three-root home does not exist on this machine")
    for move in plan["moves"]:
        source, target = Path(move["source"]), Path(move["target"])
        if not source.is_absolute() or not target.is_absolute():
            raise MigrationError("migration paths must be absolute")
        for path in (source, target):
            if not path.resolve().is_relative_to(home):
                raise MigrationError("migration path escapes the selected home")
        if not source.is_relative_to(home / "Datasets/pointstream-data"):
            raise MigrationError("only the legacy external data tree may be relocated")
        if not (
            target.is_relative_to(home / "Datasets")
            or target.is_relative_to(home / "Models")
            or target == home / "detectron2"
        ):
            raise MigrationError("destination is outside the approved storage roots")
        if target.is_relative_to(source):
            raise MigrationError("destination is inside its source")
        if journal.resolve().is_relative_to(source.resolve()):
            raise MigrationError("journal must be outside every moved subtree")
        parent = target.parent
        while not parent.exists():
            parent = parent.parent
        if parent.stat().st_dev != move["identity"]["dev"]:
            raise MigrationError("cross-filesystem relocation is forbidden")
    return home


def transact(
    plan: dict[str, Any], journal: Path, report: dict[str, Any], *, rollback: bool = False
) -> None:
    home = validate_plan(plan, journal)
    validate_ready(report, home)
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    events = (
        [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    )
    if events and any(e.get("plan_sha256") != digest for e in events):
        raise MigrationError("journal belongs to a different immutable plan")
    intents = {e["index"] for e in events if e["event"] == "intent"}
    lock = journal.with_suffix(journal.suffix + ".lock")
    journal.parent.mkdir(parents=True, exist_ok=True)
    lock.mkdir()  # no stealing a stale lock; reconciliation requires attention
    owner = lock / "owner.json"
    write_new(
        owner,
        {"pid": os.getpid(), "host": __import__("socket").gethostname(), "created_at": time.time()},
    )
    try:
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as handle:

            def record(event: str, index: int) -> None:
                handle.write(
                    json.dumps(
                        {
                            "event": event,
                            "index": index,
                            "plan_sha256": digest,
                            "utc": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())

            indexed = list(enumerate(plan["moves"]))
            if rollback:
                indexed.reverse()
            for index, move in indexed:
                source, target = Path(move["source"]), Path(move["target"])
                expected = move["identity"]
                if rollback:
                    if index not in intents:
                        continue
                    if (
                        same_inode(source, expected)
                        and not source.is_symlink()
                        and not target.exists()
                    ):
                        continue
                    if not same_inode(target, expected):
                        raise MigrationError(f"rollback target identity changed: {target}")
                    if source.is_symlink() and source.resolve() == target.resolve():
                        record("rollback_intent", index)
                        source.unlink()  # only this transaction's verified compatibility alias
                    elif source.exists() or source.is_symlink():
                        raise MigrationError(f"rollback source was recreated: {source}")
                    rename_noreplace(target, source)
                    record("rolled_back", index)
                    continue
                if (
                    index in intents
                    and source.is_symlink()
                    and source.resolve() == target.resolve()
                    and same_inode(target, expected)
                ):
                    continue
                if (
                    index in intents
                    and not source.exists()
                    and not source.is_symlink()
                    and same_inode(target, expected)
                ):
                    source.symlink_to(target, target_is_directory=stat.S_ISDIR(expected["mode"]))
                    sync_directory(source.parent)
                    record("linked", index)
                    continue
                if source.is_symlink() or not source.exists() or identity(source) != expected:
                    raise MigrationError(f"source changed since planning: {source}")
                if target.exists() or target.is_symlink():
                    raise MigrationError(f"destination appeared after planning: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                record("intent", index)
                rename_noreplace(source, target)
                record("moved", index)
                source.symlink_to(target, target_is_directory=stat.S_ISDIR(expected["mode"]))
                sync_directory(source.parent)
                if not same_inode(source, expected):
                    raise MigrationError(f"relocated identity check failed: {source}")
                record("linked", index)
    finally:
        owner.unlink()
        lock.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check-hosts", "apply", "rollback"))
    parser.add_argument("--home", type=Path, default=Path("/home/itec/emanuele"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--via", choices=HOSTS, default="gpu3")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--host-check", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ("plan", "check-hosts"):
            if args.out is None:
                parser.error("--out is required")
            value = (
                make_plan(args.home) if args.command == "plan" else check_hosts(args.home, args.via)
            )
            write_new(args.out, value)
            print(
                json.dumps(
                    {
                        "record": str(args.out),
                        "moves": len(value.get("moves", [])),
                        "conflicts": len(value.get("conflicts", [])),
                        "ready": value.get("ready"),
                    }
                )
            )
        else:
            if args.plan is None or args.journal is None or args.host_check is None:
                parser.error("--plan, --journal and --host-check are required")
            transact(
                json.loads(args.plan.read_text()),
                args.journal,
                json.loads(args.host_check.read_text()),
                rollback=args.command == "rollback",
            )
            print(json.dumps({"journal": str(args.journal), "status": args.command + " complete"}))
    except (MigrationError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
