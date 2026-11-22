"""Opt-in preservation of reviewed YUV frames after W3's four-pass check."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

from experiments.tennis import w3_neural_baseline as w3

TICKS = list(range(10, 240, 20))
FRAME_BYTES = 960 * 544 * 3 // 2
original = w3.code_point


def record_frames(a, source, work, fps, n, point):
    result = original(a, source, work, fps, n, point)
    if not result["accepted"] or not result["identical_frame_hashes"]:
        return result
    if point["scale"] != 2 or point["qp"] not in (5, 6, 7, 15):
        raise ValueError("unexpected selected point")
    raw = Path(result["decoded"])
    reports = json.loads((raw.parent / "reports.json").read_text())
    arrays = [
        reports["decode"][tag]["passes"][i]["frame_sha256"]
        for tag in ("decode-a", "decode-b")
        for i in (0, 1)
    ]
    ticks = [t for t in TICKS if t < n]
    output = Path(os.environ["PS_SCRATCH_DIR"]) / "publish" / "nad000"
    rows = []
    with raw.open("rb") as source_file, (output / "review12.yuv").open("wb") as target:
        for tick in ticks:
            source_file.seek(tick * FRAME_BYTES)
            data = source_file.read(FRAME_BYTES)
            digest = hashlib.sha256(data).hexdigest()
            if len(data) != FRAME_BYTES or any(v[tick] != digest for v in arrays):
                raise ValueError("saved YUV frame differs from a verified native decode")
            target.write(data)
            rows.append({"tick": tick, "sha256": digest})
    (output / "review12.json").write_text(
        json.dumps(
            {
                "frames": n,
                "qp": point["qp"],
                "ticks": rows,
                "container_sha256": result["container_sha256"],
            },
            indent=2,
        )
        + "\n"
    )
    return result


def check_frames():
    stage = Path(os.environ["PS_STAGE_DIR"])
    with tarfile.open(stage / "published.tar") as tar:
        meta = json.load(tar.extractfile("publish/nad000/review12.json"))
        data = tar.extractfile("publish/nad000/review12.yuv").read()
    expected = [t for t in TICKS if t < meta["frames"]]
    if [r["tick"] for r in meta["ticks"]] != expected or len(data) != len(expected) * FRAME_BYTES:
        raise ValueError("review frame coverage")
    if any(
        hashlib.sha256(data[i * FRAME_BYTES : (i + 1) * FRAME_BYTES]).hexdigest() != row["sha256"]
        for i, row in enumerate(meta["ticks"])
    ):
        raise ValueError("review frame digest")


if __name__ == "__main__":
    if "--validate" in sys.argv:
        w3.main()
        check_frames()
    else:
        w3.code_point = record_frames
        w3.main()
