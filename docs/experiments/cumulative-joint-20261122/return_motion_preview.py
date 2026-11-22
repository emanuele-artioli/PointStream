"""Create a private 25-fps derivative of the fixed return comparison."""

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import common_player_ball as ball
from src.codecs import svtav1
from src.pointstream import tracked_patch

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data")
AUDIT = Path("/local/users/emanuele/pointstream/audits/return-regions-final-20261122")
BIND = ROOT / "tennis/cumulative-joint-20261122/return-cached-player-v5"
SOURCE = ROOT / "tennis/w7-tier2-codex-20261031/eval/sources/nadal_kyrgios__002.mp4"
OUT = Path("/local/users/emanuele/pointstream/audits/return-motion-preview-20261122")


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    carrier = (BIND / "receiver.pscp").read_bytes()
    sidecar = (BIND / "ball.sidecar").read_bytes()
    dec = svtav1.tool("dav1d")
    positions, _, _ = ball._open(carrier, sidecar, dec["sha256"])
    frames = np.load(AUDIT / "frames.npy", mmap_mode="r")
    control = np.memmap(AUDIT / "control.yuv", np.uint8, "r", shape=(240, 816, 960))
    tiles = np.memmap(AUDIT / "tile.yuv", np.uint8, "r", shape=(240, 96, 64))
    cap = cv2.VideoCapture(str(SOURCE), cv2.CAP_FFMPEG)
    assert cap.isOpened()
    target = OUT / "return-comparison.mp4"
    ff = svtav1.tool("ffmpeg")
    cmd = [
        ff["path"],
        "-v",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "1440x270",
        "-r",
        "25",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-threads",
        "1",
        "-n",
        str(target),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    written = 0
    try:
        for t in range(240):
            ok, src = cap.read()
            assert ok
            if t % 2:
                continue
            source = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            ordinary = cv2.resize(
                tracked_patch.yuv_to_rgb(control[t], 960, 544, False),
                (1920, 1080),
                interpolation=cv2.INTER_CUBIC,
            )
            joint = np.array(frames[t], copy=True)
            if positions[t] is not None:
                x, y = positions[t]
                joint[y : y + 32, x : x + 32] = tracked_patch.yuv_to_rgb(tiles[t], 64, 64, False)[
                    16:48, 16:48
                ]
            panels = []
            for label, image in zip(
                ("Source", "AV1", "PointStream"), (source, ordinary, joint), strict=True
            ):
                panel = cv2.cvtColor(
                    cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR
                )
                cv2.rectangle(panel, (0, 0), (145, 25), (0, 0, 0), -1)
                cv2.putText(
                    panel,
                    label,
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                panels.append(panel)
            process.stdin.write(np.hstack(panels).tobytes())
            written += 1
        process.stdin.close()
        if process.wait(timeout=120) != 0:
            raise RuntimeError(process.stderr.read().decode())
    finally:
        cap.release()
        if process.poll() is None:
            process.kill()
            process.wait()
    assert written == 120
    result = {
        "frames": written,
        "fps": 25,
        "duration_seconds": 4.8,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "bytes": target.stat().st_size,
        "scope": "private inspection derivative, every second native frame; not a formal study",
    }
    (OUT / "receipt.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
