"""Verify all source-denied receiver frame identities and export twelve lossless review PNGs."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from experiments.tennis.player_requalification import array_sha, sha256

TICKS = (9, 31, 49, 72, 95, 112, 135, 152, 175, 198, 215, 238)


def export(a):
    meta = json.loads(a.receipt.read_text())
    assert meta["ticks"] == list(range(240)) and len(meta["rows"]) == 240
    assert sha256(a.frames) == meta["frames_file_sha256"]
    assert meta["wire"]["active_ticks"] == [0, 240]
    assert meta["wire"]["ball_positions"] == 217 and meta["order"] == [
        "match102_far",
        "match102_near",
    ]
    assert meta["isolation"]["interfaces"] == ["lo"] and meta["kernels"]
    frames = np.load(a.frames, mmap_mode="r", allow_pickle=False)
    assert frames.shape == (240, 1080, 1920, 3) and frames.dtype == np.uint8
    a.out.mkdir(exist_ok=False, parents=True)
    rows = []
    for t, row in enumerate(meta["rows"]):
        rgb = frames[t]
        frame_hash = array_sha(rgb)
        assert row["tick"] == t and frame_hash == row["frame_sha256"]
        if t not in TICKS:
            continue
        file = a.out / f"tick-{t:03d}.png"
        assert cv2.imwrite(str(file), rgb[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 5])
        roundtrip = cv2.imread(str(file))[:, :, ::-1]
        assert np.array_equal(rgb, roundtrip) and array_sha(roundtrip) == frame_hash
        rows.append(
            {
                "tick": t,
                "file": file.name,
                "png_sha256": sha256(file),
                "receipt_typed_rgb_sha256": frame_hash,
                "raw_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
        )
    out = dict(
        source_sha256="fc5d3ee431f84417d2de3412402f4e8efb0546418ce96f1f0a1e2f1887a88b1a",
        native_fps="30000/1001",
        finish_job=a.job,
        wire=meta["wire"],
        receipt_sha256=sha256(a.receipt),
        all_240_frames_npy_sha256=meta["frames_file_sha256"],
        selected=rows,
    )
    (a.out / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("frames", "receipt", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--job", required=True)
    export(p.parse_args())
