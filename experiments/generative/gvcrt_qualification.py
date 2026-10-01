"""Read-only GVC-RT prerequisite identity gate. Does not load models or run inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PIN = "d0e32bfa3e8e282f9a77437c223c605858eea637"


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def rate_ledger(payload_bytes, envelope_bytes, frames, fps, width, height):
    if min(frames, fps, width, height) <= 0 or min(payload_bytes, envelope_bytes) < 0:
        raise ValueError("Invalid ledger denominator or byte count")
    total = payload_bytes + envelope_bytes
    return {"payload_bytes": payload_bytes, "envelope_bytes": envelope_bytes,
            "total_bytes": total, "observed_seconds": frames / fps,
            "kbps": total * 8 * fps / frames / 1000,
            "original_crop_bpp": total * 8 / (frames * width * height)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--i-checkpoint", type=Path, required=True)
    parser.add_argument("--i-sha256", required=True)
    parser.add_argument("--p-checkpoint", type=Path, required=True)
    parser.add_argument("--p-sha256", required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.official_repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.official_repo), "status", "--porcelain"], text=True)
    if revision != PIN or dirty:
        raise SystemExit("Official source must be clean at the registered pin")
    checkpoints = {}
    for role, path, expected in (("I", args.i_checkpoint, args.i_sha256), ("P", args.p_checkpoint, args.p_sha256)):
        record = identity(path)
        if len(expected) != 64 or record["sha256"] != expected.lower():
            raise SystemExit(f"{role} checkpoint identity mismatch")
        checkpoints[role] = record
    print(json.dumps({"source_revision": revision, "checkpoints": checkpoints,
                      "status": "identity_only_not_model_qualification",
                      "remaining_gate": "strict tensor coverage, native entropy ABI, fresh decoder smoke"}, indent=2))


if __name__ == "__main__":
    main()
