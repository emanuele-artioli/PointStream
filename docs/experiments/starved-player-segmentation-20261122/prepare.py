"""Assemble identical native 1080p frames for an RGB-only judge; keep labels local."""

from contextlib import ExitStack
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import cv2
import numpy as np
from experiments.tennis import common_player_ball as ball
from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "starved-base-local-20261122"))
import crops as replay  # noqa: E402


def main(folder):
    folder.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    base = json.loads((HERE.parent / "starved-base-local-20261122/base-receipt.json").read_text())
    control = json.loads((HERE.parent / "starved-stroke-20261122/result.json").read_text())[
        "selected"
    ]
    joint = next(r for r in base["outputs"] if r["arm"] == "clean" and r["crf"] == 60)
    rows = dict(**control, pointstream=joint)
    decodes = {}
    for name, row in rows.items():
        p = Path(row["decoded"])
        assert replay.file_sha(p) == row["decoded_sha256"]
        w, h = (240, 136) if name == "pointstream" else (480, 272)
        decodes[name] = np.memmap(p, "uint8", "r", shape=(240, h * 3 // 2, w))
    f = Path(joint["stream"]).parent
    pos, payload, _ = ball._open(
        (f / "players.pscp").read_bytes(),
        (f / "ball.sidecar").read_bytes(),
        base["tools"][2]["sha256"],
    )
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
        input=payload,
    )
    tiles = np.frombuffer(raw, np.uint8).reshape(240, 96, 64)
    finish = json.loads((replay.STAGE / "finish/finish.json").read_text())
    audit = json.loads((replay.STAGE / "finish-full-audit.json").read_text())
    assert audit["all_frames_exact"] and audit["carrier_sha256"] == finish["carrier_sha256"]
    assert (
        replay.file_sha(replay.BUNDLE)
        == "5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb"
    )
    labels = {f["tick"]: f for f in json.loads(replay.LABELS.read_text())["frames"]}
    keys = {}
    with ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(folder / "frames.zip", "x"))
        source = stack.enter_context(zipfile.ZipFile(replay.BUNDLE))
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
            p = replay.STAGE / "finish" / (unit + "-masks.npz")
            assert replay.file_sha(p) == finish["mask_files"][unit]
            z = stack.enter_context(zipfile.ZipFile(p))
            assert np.array_equal(replay.read_small(z, "t"), np.arange(240))
            masks[unit] = stack.enter_context(
                replay.array_frames(z, "masks", (240, 784, 512), np.bool_)
            )
        for t in range(max(replay.TICKS) + 1):
            crops = {u: next(gens[u][1]) for u in replay.UNITS}
            silhouettes = {u: next(masks[u]) for u in replay.UNITS}
            if t not in labels:
                continue
            original = source.read(f"frame-{t:03d}/source.png")
            assert hashlib.sha256(original).hexdigest() == labels[t]["source_png_sha256"]
            images = {"source": original}
            for name, frames in decodes.items():
                w = frames.shape[-1]
                rgb = cv2.resize(
                    yuv_to_rgb(frames[t], w, w * 136 // 240, False),
                    (1920, 1080),
                    interpolation=cv2.INTER_CUBIC,
                )
                if name == "pointstream":
                    support = np.zeros((1080, 1920), bool)
                    for unit in replay.UNITS:
                        n, status = replay.overlay(
                            rgb, support, crops[unit], gens[unit][0][t], silhouettes[unit]
                        )
                        assert (
                            n == finish["rows"][t]["projected_pixels"][unit]
                            and status == finish["rows"][t]["status"][unit]
                        )
                    if pos[t] is not None:
                        x, y = pos[t]
                        rgb[y : y + 32, x : x + 32] = yuv_to_rgb(tiles[t], 64, 64, False)[
                            16:48, 16:48
                        ]
                ok, arr = cv2.imencode(".png", rgb[:, :, ::-1])
                assert ok
                images[name] = arr.tobytes()
            for name, png in images.items():
                digest = hashlib.sha256(png).hexdigest()
                archive.writestr(digest + ".png", png)
                keys[digest] = [name, t]
    assert len(keys) == 48
    (folder / "key.json").write_text(json.dumps(keys, indent=2))
    print(replay.file_sha(folder / "frames.zip"))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("folder", type=Path)
    main(cli.parse_args().folder)
