"""Copy a weight into ``Models/<family>/`` and record where it came from.

    python3 -m tools.models.place FAMILY SOURCE [--name NAME] [--source-url URL]
                                  [--expect-sha256 HEX] [--note TEXT]

One sequential read hashes and copies the file; the copy is re-read and must
match. ``Models/<family>/MANIFEST.json`` gains an entry with the sha256, size,
source path and URL. An existing target is never overwritten: if it already has
the same sha256 the entry is only recorded, otherwise the command fails.
Standard library only, so it runs with any host interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

CANONICAL_MODELS = Path("/home/itec/emanuele/Models")
CHUNK = 16 * 2**20


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def place(family: str, source: Path, name: str, url: str | None, expect: str | None, note: str | None) -> dict[str, object]:
    root = Path(os.environ.get("PS_MODELS_ROOT", "") or CANONICAL_MODELS)
    target = root / family / name
    target.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    if target.exists():
        digest = _hash(target)
        if digest != _hash(source):
            raise SystemExit(f"{target} exists with different content; not overwriting")
    else:
        partial = target.with_name(f".{name}.partial")
        hasher = hashlib.sha256()
        with source.open("rb") as reader, partial.open("wb") as writer:
            while block := reader.read(CHUNK):
                hasher.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        digest = hasher.hexdigest()
        if _hash(partial) != digest:
            partial.unlink()
            raise SystemExit(f"copy of {source} does not match its source")
        partial.chmod(0o444)
        os.replace(partial, target)
    if expect and digest != expect:
        raise SystemExit(f"{target}: sha256 {digest} differs from the expected {expect}")
    entry = {
        "file": name, "sha256": digest, "bytes": target.stat().st_size,
        "copied_from": str(source), "source_url": url, "note": note,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = target.parent / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    manifest["files"] = [f for f in manifest["files"] if f["file"] != name] + [entry]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("family")
    parser.add_argument("source", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--source-url")
    parser.add_argument("--expect-sha256")
    parser.add_argument("--note")
    args = parser.parse_args(argv)
    entry = place(args.family, args.source, args.name or args.source.name, args.source_url, args.expect_sha256, args.note)
    print(json.dumps(entry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
