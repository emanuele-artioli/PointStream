"""Reconstruct the frozen 235 tile RGB on its original CPU host."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

import numpy as np

from experiments.tennis.ball_starved_scaled import render_scaled
from src.codecs import svtav1

PACKET = "338c8d17c3bbaaedb70a9c82c75bf2567ac0acfb9e0b0289c73ad9151bedd4e5"


def run(args):
    started = time.monotonic()
    with tarfile.open(args.archive) as tar:
        report = json.load(tar.extractfile("publish/result.json"))
        (row,) = [r for r in report["results"] if r["mode"] == "tracked" and r["sha256"] == PACKET]
        packet = tar.extractfile("publish/" + row["file"]).read()
    if (
        hashlib.sha256(packet).hexdigest() != PACKET
        or report["frames"] != 154
        or report["fps"] != 60
    ):
        raise ValueError("packet identity")
    root = Path(os.environ["PS_SCRATCH_DIR"])
    work = root / "origin-rgb"
    work.mkdir()
    decoder = svtav1.tool("dav1d")
    rgb = render_scaled(packet, work, decoder["path"])
    actual = hashlib.sha256(rgb.tobytes()).hexdigest()
    if actual != row["rendered_sha256"]:
        raise ValueError(f"original pixel hash mismatch: {actual} != {row['rendered_sha256']}")
    publish = root / "publish"
    publish.mkdir()
    output = publish / "frames.npz"
    np.savez_compressed(output, frames=rgb)
    with output.open("rb") as f:
        npz_hash = hashlib.file_digest(f, "sha256").hexdigest()
    out = {
        "stage": os.environ["PS_STAGE"],
        "frames": 154,
        "fps": 60,
        "packet_sha256": PACKET,
        "rgb_sha256": actual,
        "npz_sha256": npz_hash,
        "npz_bytes": output.stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
    }
    data = json.dumps(out, indent=2) + "\n"
    (publish / "result.json").write_text(data)
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(data)


def validate():
    root = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((root / "result.json").read_text())
    checks = {
        "frames": r["frames"] == 154 and r["fps"] == 60,
        "packet": r["packet_sha256"] == PACKET,
        "digest": r["rgb_sha256"]
        == "45138443a2547750eab06cc0c94912aa7596a34f5f60e3c20bdd711f607f7f25",
        "size": r["npz_bytes"] > 0,
    }
    with tarfile.open(root / "published.tar") as tar:
        file = tar.extractfile("publish/frames.npz")
        checks["npz"] = hashlib.file_digest(file, "sha256").hexdigest() == r["npz_sha256"]
    report = {"passed": all(checks.values()), "checks": checks}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(report) + "\n")
    if not report["passed"]:
        raise SystemExit(str(checks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)


if __name__ == "__main__":
    main()
