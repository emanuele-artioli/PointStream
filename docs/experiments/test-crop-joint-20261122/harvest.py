"""Verify all source-denied receiver frame identities and export twelve lossless review PNGs."""

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
from experiments.tennis.player_requalification import array_sha, sha256

TICKS = (9, 31, 49, 72, 95, 112, 135, 152, 175, 198, 215, 238)


def decoded_frames(meta, path):
    if "frames_file_sha256" in meta:
        assert sha256(path) == meta["frames_file_sha256"] and meta["kernels"]
        assert meta["order"] == ["match102_far", "match102_near"]
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        assert arr.shape == (240, 1080, 1920, 3) and arr.dtype == np.uint8
        yield from arr
    else:
        assert sha256(path) == meta["frames_sha256"] and len(meta["witness_sha256"]) == 64
        with path.open("rb") as f:
            for row in meta["rows"]:
                size = struct.unpack(">I", f.read(4))[0]
                assert size + 4 == row["wire_bytes"]
                yield np.frombuffer(zlib.decompress(f.read(size)), np.uint8).reshape(1080, 1920, 3)
            assert f.read(1) == b""


def export(a):
    meta = json.loads(a.receipt.read_text())
    assert meta["ticks"] == list(range(240)) and len(meta["rows"]) == 240
    assert meta["wire"]["active_ticks"] == [0, 240] and meta["wire"]["ball_positions"] == 217
    assert meta["isolation"]["interfaces"] == ["lo"] and meta["isolation"]["mutation_denied"]
    source = json.loads(a.source_manifest.read_text())
    assert source["native_fps"] == "30000/1001" and set(map(int, source["ticks"])) == set(TICKS)
    a.out.mkdir(exist_ok=False, parents=True)
    rows = []
    for t, (row, rgb) in enumerate(zip(meta["rows"], decoded_frames(meta, a.frames), strict=True)):
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
                "source_raw_rgb_sha256": source["ticks"][str(t)]["raw_rgb_sha256"],
                "source_typed_rgb_sha256": source["ticks"][str(t)]["typed_rgb_sha256"],
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
        all_240_output_sha256=meta.get("frames_file_sha256", meta.get("frames_sha256")),
        foreground_cache_bytes=meta["wire"]["cache_bytes"],
        joint_rally_bytes=meta["wire"]["joint_bytes"],
        source_manifest_sha256=sha256(a.source_manifest),
        ticks={str(r["tick"]): r for r in rows},
        selected=rows,
    )
    (a.out / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("frames", "receipt", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(__file__).with_name("match102-source-review-frames.json"),
    )
    p.add_argument("--job", required=True)
    export(p.parse_args())
