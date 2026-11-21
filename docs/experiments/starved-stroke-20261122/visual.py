"""Render the three fixed stroke phases from source, AV1, and real player-layer compositions."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.tennis import common_player_ball as ball
from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "starved-base-local-20261122"))
import crops as replay  # noqa: E402

TICKS = [78, 86, 94]
BOX = (854, 390, 1324, 1032)


def render(root):
    root.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    native = json.loads((HERE / "result.json").read_text())
    scores = json.loads((HERE / "scores.json").read_text())
    assert scores["crops_sha256"] == native["crops_sha256"]
    best = min(scores["means"], key=lambda k: scores["means"][k]["p1"] + scores["means"][k]["p2"])
    anchor = native["selected"][best]
    base = json.loads((HERE.parent / "starved-base-local-20261122/base-receipt.json").read_text())
    joint = next(r for r in base["outputs"] if r["arm"] == "clean" and r["crf"] == 60)
    source = replay.SOURCE / "ps-native-ctl-20261118-full/source.yuv"
    assert replay.file_sha(source) == base["inputs"]["original"]

    def read(row, w, h):
        assert replay.file_sha(Path(row["decoded"])) == row["decoded_sha256"]
        return np.memmap(row["decoded"], "uint8", "r", shape=(240, h * 3 // 2, w))

    def canvas(frame, w, h):
        return cv2.resize(
            yuv_to_rgb(frame, w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
        )

    videos = [
        np.memmap(source, "uint8", "r", shape=(240, 816, 960)),
        read(anchor, 480, 272),
        read(joint, 240, 136),
    ]
    folder = Path(joint["stream"]).parent
    positions, tile, size = ball._open(
        (folder / "players.pscp").read_bytes(),
        (folder / "ball.sidecar").read_bytes(),
        base["tools"][2]["sha256"],
    )
    assert size == (1920, 1080)
    raw = subprocess.check_output(
        [
            "/opt/homebrew/bin/ffmpeg",
            "-v",
            "error",
            "-c:v",
            "libdav1d",
            "-f",
            "ivf",
            "-i",
            "pipe:0",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        input=tile,
    )
    tiles = np.frombuffer(raw, np.uint8).reshape(240, 96, 64)
    finish = json.loads((replay.STAGE / "finish/finish.json").read_text())
    audit = json.loads((replay.STAGE / "finish-full-audit.json").read_text())
    assert audit["all_frames_exact"] and finish["carrier_sha256"] == audit["carrier_sha256"]
    images = {}
    hashes = {}
    with ExitStack() as stack:
        gens = dict(
            replay.bind_generation(
                stack,
                replay.AUDIT / f"player{i}-new-full.tar",
                replay.STAGE / f"player{i}-full.json",
                finish,
            )
            for i in (1, 2)
        )
        masks = {}
        for unit in replay.UNITS:
            path = replay.STAGE / "finish" / (unit + "-masks.npz")
            assert replay.file_sha(path) == finish["mask_files"][unit]
            z = stack.enter_context(zipfile.ZipFile(path))
            assert np.array_equal(replay.read_small(z, "t"), np.arange(240))
            masks[unit] = stack.enter_context(
                replay.array_frames(z, "masks", (240, 784, 512), np.bool_)
            )
        for t in range(max(TICKS) + 1):
            painted = {u: next(gens[u][1]) for u in replay.UNITS}
            current = {u: next(masks[u]) for u in replay.UNITS}
            if t not in TICKS:
                continue
            arms = [
                canvas(videos[0][t], 960, 544),
                canvas(videos[1][t], 480, 272),
                canvas(videos[2][t], 240, 136),
            ]
            support = np.zeros((1080, 1920), bool)
            for u in replay.UNITS:
                n, status = replay.overlay(arms[2], support, painted[u], gens[u][0][t], current[u])
                assert (
                    n == finish["rows"][t]["projected_pixels"][u]
                    and status == finish["rows"][t]["status"][u]
                )
            if positions[t] is not None:
                x, y = positions[t]
                arms[2][y : y + 32, x : x + 32] = yuv_to_rgb(tiles[t], 64, 64, False)[16:48, 16:48]
            for a, frame in enumerate(arms):
                pic = Image.fromarray(frame).crop(BOX)
                images[a, t] = pic
                hashes[f"{a}/{t}"] = hashlib.sha256(pic.tobytes()).hexdigest()
    fig = Image.new("RGB", (1120, 1370), "white")
    draw = ImageDraw.Draw(fig)
    fontpath = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font = ImageFont.truetype(fontpath, 22)
    small = ImageFont.truetype(fontpath, 18)
    draw.text(
        (20, 12),
        "A serve at a constrained running bitrate",
        font=ImageFont.truetype(fontpath, 31),
        fill="black",
    )
    titles = ["Preparation · 1.56 s", "Near contact · 1.72 s", "Follow-through · 1.88 s"]
    labels = [
        ["Source", "960 × 544"],
        ["Plain AV1" if best == "plain" else "ROI AV1", f"{anchor['rate']:.2f} kbps"],
        ["PointStream", "65.47 kbps*"],
    ]
    for col, t in enumerate(TICKS):
        x = 180 + col * 305
        draw.text((x + 10, 66), titles[col], font=font, fill="black")
        for arm in range(3):
            y = 104 + arm * 394
            fig.paste(images[arm, t].resize((281, 384), Image.Resampling.LANCZOS), (x, y))
            if col == 0:
                draw.multiline_text(
                    (18, y + 148), "\n".join(labels[arm]), font=font, fill="black", spacing=9
                )
    draw.text(
        (20, 1301),
        "*Ideal running price with both references already cached. Measured cold packet: 469.33 kbps.",
        font=small,
        fill="black",
    )
    draw.text(
        (20, 1331),
        "Source-selected instants. No independent racket representation. Private broadcast imagery for author review.",
        font=small,
        fill="black",
    )
    fig.save(root / "starved-serve.png")
    fig.save(root / "starved-serve.pdf", resolution=144)
    (root / "receipt.json").write_text(
        json.dumps(
            dict(
                ticks=TICKS,
                box=BOX,
                anchor=best,
                anchor_kbps=anchor["rate"],
                cold_kbps=joint["cold_kbps"],
                modeled_running_kbps=joint["optimistic_no_reference_kbps"],
                rgb_crop_hashes=hashes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fresh_output", type=Path)
    render(parser.parse_args().fresh_output)
