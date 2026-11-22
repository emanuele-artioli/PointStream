"""Locate already-visible return court artifacts relative to player overlays."""

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import common_player_ball as ball, common_player_packet as player
from src.codecs import svtav1
from src.pointstream import tracked_patch

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data")
AUDIT = Path("/local/users/emanuele/pointstream/audits")
OUT = AUDIT / "return-background-diagnostic-20261122"
SOURCE = ROOT / "tennis/w7-tier2-codex-20261031/eval/sources/nadal_kyrgios__002.mp4"
BIND = ROOT / "tennis/cumulative-joint-20261122/return-cached-player-v5"
FINISH = ROOT / "jobs/fleet/inbox/20261122T015022Z-294757d5/full/published.tar"
TICKS = (30, 110, 190)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def support_for(box, mask):
    x, y, width, height = (round(float(v)) for v in box)
    width, height = max(1, width), max(1, height)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(1920, x + width), min(1080, y + height)
    full = np.zeros((1080, 1920), bool)
    if x0 < x1 and y0 < y1 and mask.any():
        inside = np.s_[y0 - y : y1 - y, x0 - x : x1 - x]
        full[y0:y1, x0:x1] = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )[inside].astype(bool)
    return full


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and len(os.sched_getaffinity(0)) == 1
    OUT.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    assert sha(SOURCE) == "f0551443eaff4e4b3539f21cd0c6ac66f55601d480ad099a571e8626db735e4a"
    assert sha(FINISH) == "c2154724af02655196976545f2b50160f322340b69d5c64734b9a4eccd6ba20f"
    carrier = (BIND / "receiver.pscp").read_bytes()
    _, _, players = player.unpack(carrier)
    names = player._embedded_profile(carrier)["order"]
    positions, _, _ = ball._open(
        carrier, (BIND / "ball.sidecar").read_bytes(), svtav1.tool("dav1d")["sha256"]
    )
    with tarfile.open(FINISH) as tar:
        masks = {
            name: np.load(
                io.BytesIO(tar.extractfile("publish/" + name + "-masks.npz").read()),
                allow_pickle=False,
            )["masks"]
            for name in names
        }
    base_file = AUDIT / "return-ball-independent-20261122/base.yuv"
    assert sha(base_file) == "ab8d5890ae3d0365e5dab07e9ee8f2eba934feb165b4cc1a20e35b20be2a5b5c"
    base = np.memmap(base_file, np.uint8, "r", shape=(240, 204, 240))
    result = np.load(AUDIT / "return-regions-final-20261122/frames.npy", mmap_mode="r")
    tiles = np.memmap(
        AUDIT / "return-regions-final-20261122/tile.yuv", np.uint8, "r", shape=(240, 96, 64)
    )
    control = np.memmap(
        AUDIT / "return-regions-final-20261122/control.yuv", np.uint8, "r", shape=(240, 816, 960)
    )
    cap = cv2.VideoCapture(str(SOURCE), cv2.CAP_FFMPEG)
    assert cap.isOpened()
    rows, strips = [], []
    for t in range(max(TICKS) + 1):
        ok, src = cap.read()
        assert ok
        if t not in TICKS:
            continue
        clean = cv2.resize(
            tracked_patch.yuv_to_rgb(base[t], 240, 136, False),
            (1920, 1080),
            interpolation=cv2.INTER_CUBIC,
        )
        combined = np.array(result[t], copy=True)
        union = np.zeros((1080, 1920), bool)
        for name in names:
            union |= support_for(players[name][0]["boxes"][t], masks[name][t])
        changed_without_players = int(np.count_nonzero(np.any(clean != combined, axis=2) & ~union))
        if positions[t] is not None:
            x, y = positions[t]
            combined[y : y + 32, x : x + 32] = tracked_patch.yuv_to_rgb(tiles[t], 64, 64, False)[
                16:48, 16:48
            ]
            union[y : y + 32, x : x + 32] = True
        changed_without_players_or_ball = int(
            np.count_nonzero(np.any(clean != combined, axis=2) & ~union)
        )
        assert changed_without_players == 0 and changed_without_players_or_ball == 0
        ordinary = cv2.resize(
            tracked_patch.yuv_to_rgb(control[t], 960, 544, False),
            (1920, 1080),
            interpolation=cv2.INTER_CUBIC,
        )
        panels = []
        for label, image in zip(
            ("Source", "Clean base", "Final", "AV1"),
            (cv2.cvtColor(src, cv2.COLOR_BGR2RGB), clean, combined, ordinary),
            strict=True,
        ):
            panel = cv2.cvtColor(
                cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR
            )
            cv2.rectangle(panel, (0, 0), (160, 25), (0, 0, 0), -1)
            cv2.putText(
                panel,
                f"{label} t={t}",
                (7, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)
        strips.append(np.hstack(panels))
        rows.append(
            {
                "tick": t,
                "changed_pixels_outside_player_masks_before_ball": changed_without_players,
                "changed_pixels_outside_player_and_ball_support_after_ball": changed_without_players_or_ball,
                "excluded_support_pixels": int(union.sum()),
            }
        )
    cap.release()
    assert len(rows) == 3
    image = OUT / "background-diagnostic.png"
    assert cv2.imwrite(str(image), np.vstack(strips))
    receipt = {
        "ticks": list(TICKS),
        "rows": rows,
        "preview_sha256": sha(image),
        "source_sha256": sha(SOURCE),
        "finisher_sha256": sha(FINISH),
        "base_yuv_sha256": sha(base_file),
        "runner_sha256": sha(Path(__file__)),
        "scope": "post-hoc diagnostic of three fixed frames; excludes exact player and ball support; visual line alignment is not a segmentation accuracy metric",
    }
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
