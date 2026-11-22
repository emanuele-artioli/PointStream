"""Publish lossless source tick identifiers for the fixed independent TEST review."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
from experiments.tennis.player_requalification import array_sha, sha256
from harvest import TICKS


def main(a):
    key = "fc5d3ee431f84417d2de3412402f4e8efb0546418ce96f1f0a1e2f1887a88b1a"
    assert sha256(a.source) == key
    cap = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 30000 / 1001) < 0.001
    a.output.mkdir(exist_ok=False, parents=True)
    rows = {}
    for t in range(240):
        ok, bgr = cap.read()
        assert ok and bgr.shape == (1080, 1920, 3)
        if t not in TICKS:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        p = a.output / f"source-{t:03d}.png"
        assert cv2.imwrite(str(p), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 5])
        assert array_sha(cv2.imread(str(p))[:, :, ::-1]) == array_sha(rgb)
        rows[str(t)] = dict(
            tick=t,
            file=p.name,
            png_sha256=sha256(p),
            typed_rgb_sha256=array_sha(rgb),
            raw_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        )
    cap.release()
    assert set(map(int, rows)) == set(TICKS)
    (a.output / "manifest.json").write_text(
        json.dumps(
            dict(
                source_sha256=key,
                frames=240,
                native_fps="30000/1001",
                display=[1920, 1080],
                opencv=cv2.__version__,
                source_decoder="VideoCapture CAP_FFMPEG BGR to RGB8",
                ticks=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
