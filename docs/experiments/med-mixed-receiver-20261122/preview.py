"""Make a private author preview from the same source and physical decoder outputs as scoring."""

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from src.pointstream.tracked_patch import yuv_to_rgb
from build import sha, SOURCES


def main(a):
    assert sha(a.source) == SOURCES[a.clip]
    direct, ordinary, neural = (json.loads(p.read_text()) for p in (a.joint, a.ordinary, a.neural))
    joint = direct[0]
    av1 = next(x for x in ordinary if x["width"] == a.av1_width)
    dcvc = neural[0]
    arms = [av1, dcvc, joint]
    dec = []
    for x in arms:
        p = Path(x["path"])
        assert sha(p) == x["sha256"]
        if x["kind"] == "rgb_npy":
            dec.append(np.load(p, allow_pickle=False, mmap_mode="r"))
        else:
            dec.append(np.memmap(p, np.uint8, "r", shape=(240, x["height"] * 3 // 2, x["width"])))
    a.out.mkdir(exist_ok=False)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "1536x940",
        "-r",
        "50",
        "-i",
        "pipe:0",
        "-frames:v",
        "240",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-n",
        str(a.out / "matched-rate.mp4"),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    cap = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
    start = {"000": 0, "019": 92, "024": 69}[a.clip]
    cv2.setNumThreads(1)
    for t in range(240):
        ok, src = cap.read()
        assert ok
        imgs = [src]
        for x, v in zip(arms, dec, strict=True):
            rgb = (
                v[t]
                if x["kind"] == "rgb_npy"
                else cv2.resize(
                    yuv_to_rgb(v[t], x["width"], x["height"], False),
                    (1920, 1080),
                    interpolation=cv2.INTER_CUBIC,
                )
            )
            imgs.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        labels = [
            "Source",
            f"SVT-AV1  {av1['kbps']:.2f} kbps",
            f"DCVC-UF  {dcvc['kbps']:.2f} kbps",
            f"PointStream  {joint['kbps']:.2f} kbps, cached   |   "
            + ("Object mode" if start <= t < start + 120 else "Conventional fallback"),
        ]
        board = np.full((940, 1536, 3), 30, np.uint8)
        for idx, (img, label) in enumerate(zip(imgs, labels, strict=True)):
            x = (idx % 2) * 768
            y = (idx // 2) * 470
            board[y + 38 : y + 470, x : x + 768] = cv2.resize(
                img, (768, 432), interpolation=cv2.INTER_AREA
            )
            cv2.putText(
                board,
                label,
                (x + 14, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.61,
                (250, 250, 250),
                1,
                cv2.LINE_AA,
            )
        process.stdin.write(board.tobytes())
    cap.release()
    process.stdin.close()
    assert process.wait(timeout=45) == 0
    note = dict(
        purpose="Private author preview. All metrics use original native decoded arrays. This viewing copy is resized and H.264-compressed.",
        native_frames=240,
        fps=50,
        arms=[{k: v for k, v in r.items() if k != "path"} for r in arms],
        source_sha256=SOURCES[a.clip],
        preview_sha256=sha(a.out / "matched-rate.mp4"),
    )
    (a.out / "manifest.json").write_text(json.dumps(note, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("source", "joint", "ordinary", "neural", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--av1-width", type=int, required=True)
    main(p.parse_args())
