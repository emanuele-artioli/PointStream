"""Source-fitted trajectory versus paid tile on a fixed opened return shot."""

import argparse
import hashlib
import io
import json
import math
import os
import struct
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.tennis import common_player_ball as player_ball, common_player_packet as player
from src.codecs import quality, svtav1
from src.pointstream import ball, tracked_patch

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data")
AUDIT = Path("/local/users/emanuele/pointstream/audits/return-regions-final-20261122")
BIND = ROOT / "tennis/cumulative-joint-20261122/return-cached-player-v5"
SOURCE = ROOT / "tennis/w7-tier2-codex-20261031/eval/sources/nadal_kyrgios__002.mp4"
VGG = ROOT / "visor/h3b-2026-10-11/inputs/dists/vgg16-397923af.pth"
TICKS = list(range(10, 231, 20))
GEN = {
    "nadal_kyrgios__002_at120__person_000000000000": (
        "20261122T002538Z-607ae615",
        "9b56dbaf14d261a9c0dfff4dbabb246a913fbf988d4fac8e9b45a7e993270e72",
    ),
    "nadal_kyrgios__002_at120__person_000000000002": (
        "20261122T010921Z-1a910eac",
        "7f44ead9afa12e4a75ecbe1fba33d28cdfb347070fe77bc54a0565cba5e08f1d",
    ),
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def member(path, name):
    with tarfile.open(path) as archive:
        return archive.extractfile("publish/" + name).read()


def paste(canvas, crop, box, mask):
    x, y, width, height = (round(float(v)) for v in box)
    width, height = max(1, width), max(1, height)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(1920, x + width), min(1080, y + height)
    if x0 < x1 and y0 < y1 and mask.any():
        inside = np.s_[y0 - y : y1 - y, x0 - x : x1 - x]
        region = np.s_[y0:y1, x0:x1]
        keep = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)[
            inside
        ].astype(bool)
        pixels = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LANCZOS4)[inside]
        np.copyto(canvas[region], pixels, where=keep[:, :, None])


def luma_sse(image, truth):
    small = cv2.resize(image, (480, 272), interpolation=cv2.INTER_AREA).astype(np.float64)
    y = np.clip(
        np.rint(
            16
            + 219
            * (0.2126 * small[:, :, 0] + 0.7152 * small[:, :, 1] + 0.0722 * small[:, :, 2])
            / 255
        ),
        0,
        255,
    ).astype(np.int32)
    delta = truth.astype(np.int32) - y
    return int(np.square(delta).sum(dtype=np.int64))


def tensor(image):
    return torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    start = time.monotonic()
    assert sha(SOURCE) == "f0551443eaff4e4b3539f21cd0c6ac66f55601d480ad099a571e8626db735e4a"
    assert sha(VGG) == "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
    variant = ROOT / "tennis/cumulative-joint-20261122/three-segment-trajectory-v1/delivery.psb"
    assert sha(variant) == "c53685733be3ccfce2673bc49e4ad603c998b9eccd24a5e12d6acb70782ada00"
    packet = variant.read_bytes()
    n = struct.unpack_from(">I", packet, 4)[0]
    header = json.loads(packet[8 : 8 + n])
    cursor = 8 + n
    sections = {}
    for entry in header["entries"]:
        raw = packet[cursor : cursor + entry["bytes"]]
        cursor += len(raw)
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        sections[entry["name"]] = raw
    assert cursor == len(packet)
    track, appearances = ball.decode_frames(sections["return_trajectory"], 240, 960, 544)
    velocity = ball.velocities(track)
    carrier = (BIND / "receiver.pscp").read_bytes()
    assert (
        hashlib.sha256(carrier).hexdigest()
        == "a0ec63326642130adb5d765a3e7c30aaf564ad5bbf7cd17d56594ffd82dc5b06"
    )
    parts, _, players = player.unpack(carrier)
    names = player._embedded_profile(carrier)["order"]
    dec = svtav1.tool("dav1d")
    tile_positions, tile, _ = player_ball._open(
        carrier, (BIND / "ball.sidecar").read_bytes(), dec["sha256"]
    )
    assert (
        hashlib.sha256(tile).hexdigest()
        == "f30860d5924a4d2b9980168a587a343eab8edb17479d4db4aa612997ee6f7b25"
    )
    native_file = args.output / "base.yuv"
    tracked_patch.decode_native(
        parts["base"],
        native_file,
        {"width": 240, "height": 136, "fps": [50, 1], "frames": 240},
        dec["real_path"],
    )
    assert sha(native_file) == "ab8d5890ae3d0365e5dab07e9ee8f2eba934feb165b4cc1a20e35b20be2a5b5c"
    native = np.memmap(native_file, np.uint8, "r", shape=(240, 204, 240))
    tiled_frames = np.load(AUDIT / "frames.npy", mmap_mode="r")
    tile_yuv = np.memmap(AUDIT / "tile.yuv", np.uint8, "r", shape=(240, 96, 64))
    control_yuv = np.memmap(AUDIT / "control.yuv", np.uint8, "r", shape=(240, 816, 960))
    truth_yuv = np.memmap(AUDIT / "source-yuv.yuv", np.uint8, "r", shape=(240, 408, 480))
    fitjob = ROOT / "jobs/fleet/inbox/20261122T015022Z-294757d5/full/published.tar"
    assert sha(fitjob) == "c2154724af02655196976545f2b50160f322340b69d5c64734b9a4eccd6ba20f"
    masks, crops = {}, {}
    for name in names:
        job, digest = GEN[name]
        archive = ROOT / f"jobs/fleet/inbox/{job}/full/published.tar"
        assert sha(archive) == digest
        generated = np.load(io.BytesIO(member(archive, name + ".npz")), allow_pickle=False)
        masks_file = np.load(io.BytesIO(member(fitjob, name + "-masks.npz")), allow_pickle=False)
        crops[name], masks[name] = generated["crops"], masks_file["masks"]
        assert np.array_equal(generated["box"], players[name][0]["boxes"]) and np.array_equal(
            masks_file["t"], np.arange(240)
        )
    centers = json.loads(
        (ROOT / "jobs/fleet/inbox/20261121T231717Z-f98e569a/full/track.json").read_text()
    )["centers"]
    net = quality.load_dists(VGG, "cpu")
    cap = cv2.VideoCapture(str(SOURCE), cv2.CAP_FFMPEG)
    assert cap.isOpened()
    rows, ball_rows = [], []
    luma = {"tile": 0, "trajectory": 0}
    frame_hash = hashlib.sha256()
    with torch.inference_mode():
        for t in range(240):
            ok, src = cap.read()
            assert ok
            base = cv2.resize(
                tracked_patch.yuv_to_rgb(native[t], 240, 136, False),
                (960, 544),
                interpolation=cv2.INTER_CUBIC,
            )
            streak = ball.render(base, t, track, velocity, appearances)
            generated = cv2.resize(streak, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            for name in names:
                paste(generated, crops[name][t], players[name][0]["boxes"][t], masks[name][t])
            frame_hash.update(generated.tobytes())
            tiled = np.array(tiled_frames[t], copy=True)
            if tile_positions[t] is not None:
                x, y = tile_positions[t]
                tiled[y : y + 32, x : x + 32] = tracked_patch.yuv_to_rgb(
                    tile_yuv[t], 64, 64, False
                )[16:48, 16:48]
            luma["tile"] += luma_sse(tiled, truth_yuv[t, :272])
            luma["trajectory"] += luma_sse(generated, truth_yuv[t, :272])
            if t not in TICKS:
                continue
            source = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            ordinary = cv2.resize(
                tracked_patch.yuv_to_rgb(control_yuv[t], 960, 544, False),
                (1920, 1080),
                interpolation=cv2.INTER_CUBIC,
            )
            images = {"ordinary": ordinary, "tile": tiled, "trajectory": generated}
            for name in names:
                x, y, w, h = [round(float(v)) for v in players[name][0]["boxes"][t]]
                x0, y0, x1, y1 = (
                    max(0, x),
                    max(0, y),
                    min(1920, x + max(1, w)),
                    min(1080, y + max(1, h)),
                )
                interp = cv2.INTER_AREA if x1 - x0 > 256 or y1 - y0 > 256 else cv2.INTER_LINEAR
                ref = cv2.resize(source[y0:y1, x0:x1], (256, 256), interpolation=interp)
                xref = tensor(ref)
                mask = torch.ones((1, 1, 256, 256))
                assert abs(float(quality.masked_dists(net, xref, xref, mask)[0])) < 1e-5
                result = {}
                for arm, img in images.items():
                    got = cv2.resize(img[y0:y1, x0:x1], (256, 256), interpolation=interp)
                    mse = float(np.mean((ref.astype(np.float64) - got) ** 2))
                    result[arm] = {
                        "dists": float(quality.masked_dists(net, xref, tensor(got), mask)[0]),
                        "rgb_psnr": 10 * math.log10(255**2 / mse),
                    }
                rows.append({"tick": t, "unit": name, **result})
            if centers[t] is not None:
                cx, cy = centers[t]
                x0 = max(0, min(1920 - 64, int(cx) - 32))
                y0 = max(0, min(1080 - 64, int(cy) - 32))
                ref = source[y0 : y0 + 64, x0 : x0 + 64]
                xr = tensor(ref)
                mask = torch.ones((1, 1, 64, 64))
                r = {}
                for arm, img in images.items():
                    crop = img[y0 : y0 + 64, x0 : x0 + 64]
                    mse = float(np.mean((ref.astype(np.float64) - crop) ** 2))
                    r[arm] = {
                        "dists": float(quality.masked_dists(net, xr, tensor(crop), mask)[0]),
                        "rgb_psnr": 10 * math.log10(255**2 / mse),
                    }
                ball_rows.append({"tick": t, "sender_center": centers[t], **r})
    cap.release()
    assert len(rows) == 24 and luma["tile"] == 3187126762

    def means(records):
        return {
            arm: {m: float(np.mean([r[arm][m] for r in records])) for m in ("dists", "rgb_psnr")}
            for arm in ("ordinary", "tile", "trajectory")
        }

    result = {
        "ticks": TICKS,
        "body_support": "same expanded paid rectangles, 256x256 RGB",
        "body_rows": rows,
        "body_means": means(rows),
        "ball_support": "64x64 centered on sender prediction, not ground truth",
        "ball_rows": ball_rows,
        "ball_means": means(ball_rows) if ball_rows else None,
        "return_luma_sse": luma,
        "return_luma_psnr": {
            k: 10 * math.log10(255**2 * 480 * 272 * 240 / v) for k, v in luma.items()
        },
        "trajectory_frames": 240,
        "trajectory_rgb_stream_sha256": frame_hash.hexdigest(),
        "packet_sha256": sha(variant),
        "runner_sha256": sha(Path(__file__)),
        "dists_record": quality.dists_record(VGG),
        "elapsed_seconds": time.monotonic() - start,
        "scope": "opened development /002, source-fitted streak; no independent ball or body labels",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("body_means", "ball_means", "return_luma_psnr", "elapsed_seconds")
            }
        )
    )


if __name__ == "__main__":
    main()
