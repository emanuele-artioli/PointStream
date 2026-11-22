"""Stream the registered RGB source into a lossless, campus-decodable file with no raw-video scratch."""

import argparse
import json
from pathlib import Path
import subprocess
import cv2
from experiments.tennis.player_requalification import array_sha, sha256

HERE = Path(__file__).resolve().parent


def main(a):
    grid = json.loads((HERE / "match102-source-grid.json").read_text())
    assert sha256(a.source) == grid["source_sha256"] and not a.output.exists()
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "1920x1080",
            "-r",
            "30000/1001",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-g",
            "1",
            "-threads",
            "2",
            "-pix_fmt",
            "bgr0",
            "-frames:v",
            "240",
            "-f",
            "matroska",
            "-n",
            str(a.output),
        ],
        stdin=subprocess.PIPE,
    )
    cap = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
    for expected in grid["native_rgb_typed_sha256"]:
        ok, bgr = cap.read()
        assert ok and array_sha(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) == expected
        proc.stdin.write(bgr.tobytes())
    proc.stdin.close()
    assert proc.wait(timeout=90) == 0
    cap.release()
    check = cv2.VideoCapture(str(a.output), cv2.CAP_FFMPEG)
    for t, h in enumerate(grid["native_rgb_typed_sha256"]):
        ok, bgr = check.read()
        assert ok and array_sha(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) == h, t
    assert not check.read()[0]
    check.release()
    p = a.output.with_suffix(".json")
    p.write_text(
        json.dumps(
            dict(
                source_sha256=grid["source_sha256"],
                source_grid_sha256=sha256(HERE / "match102-source-grid.json"),
                lossless_sha256=sha256(a.output),
                bytes=a.output.stat().st_size,
                native_fps=grid["fps"],
                frames=240,
                pixel_format="FFV1 bgr0",
                verified_exact_native_rgb_ticks=240,
            ),
            indent=2,
        )
        + "\n"
    )
    print(p.read_text())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
