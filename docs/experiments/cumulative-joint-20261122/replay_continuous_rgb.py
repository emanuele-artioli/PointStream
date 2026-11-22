"""Stream the paid three-shot receiver as one ordered 480x272 RGB sequence."""

import hashlib
import io
import json
import os
import struct
import subprocess
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import common_player_ball as ball, common_player_packet as player
from src.codecs import svtav1
from src.pointstream import tracked_patch

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data")
AUDIT = Path("/local/users/emanuele/pointstream/audits")
OUT = AUDIT / "continuous-rgb-replay-20261122"
GEN = {
    "nadal_kyrgios__000_at120__person_000000000000": (
        "20261118T162820Z-841078f9",
        "fc0e4a380c47116332bdcd6dfedc268c2a34f4d4b3eca7608d764acfb912d664",
    ),
    "nadal_kyrgios__000_at120__person_000000000001": (
        "20261118T160156Z-9fdf17fb",
        "4894f619d171db28d63356ac6b1ae45acf4e49e3550f80ee26b7f329ee04d411",
    ),
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def member(path, name):
    with tarfile.open(path) as tar:
        return tar.extractfile("publish/" + name).read()


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


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and len(os.sched_getaffinity(0)) == 1
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cv2.setNumThreads(1)
    packet = (
        ROOT / "tennis/cumulative-joint-20261122/three-segment-transport-v2/delivery.psb"
    ).read_bytes()
    assert (
        hashlib.sha256(packet).hexdigest()
        == "7ad5e13d4007d5510e70c2d9892c082817d2f4f59e07cdc38641db172b471f64"
    )
    n = struct.unpack_from(">I", packet, 4)[0]
    header = json.loads(packet[8 : 8 + n])
    offset = 8 + n
    sections = {}
    for row in header["entries"]:
        data = packet[offset : offset + row["bytes"]]
        offset += len(data)
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        sections[row["name"]] = data
    assert offset == len(packet)
    dec, ff = svtav1.tool("dav1d"), svtav1.tool("ffmpeg")
    init = (AUDIT / "initial-cached-player-v3-20261122/receiver.pscp").read_bytes()
    ret = (AUDIT / "return-cached-player-v5-20261122/receiver.pscp").read_bytes()
    assert (
        hashlib.sha256(init).hexdigest()
        == "45a6554b7fc05d07c308c22f148d70c035c6dde838abc08fe38f8aedf0002d1b"
    )
    assert (
        hashlib.sha256(ret).hexdigest()
        == "a0ec63326642130adb5d765a3e7c30aaf564ad5bbf7cd17d56594ffd82dc5b06"
    )
    initial_parts, _, initial_players = player.unpack(init)
    order = player._embedded_profile(init)["order"]
    init_pos, init_tile, _ = ball._open(init, sections["initial_ball"], dec["sha256"])
    return_pos, return_tile, _ = ball._open(ret, sections["return_ball"], dec["sha256"])
    tracked_patch.decode_native(
        initial_parts["base"],
        OUT / "initial-base.yuv",
        {"width": 240, "height": 136, "fps": [50, 1], "frames": 240},
        dec["real_path"],
    )
    tracked_patch.decode_native(
        init_tile,
        OUT / "initial-ball.yuv",
        {"width": 64, "height": 64, "fps": [50, 1], "frames": 240},
        dec["real_path"],
    )
    tracked_patch.decode_native(
        return_tile,
        OUT / "return-ball.yuv",
        {"width": 64, "height": 64, "fps": [50, 1], "frames": 240},
        dec["real_path"],
    )
    base = np.memmap(OUT / "initial-base.yuv", np.uint8, "r", shape=(240, 204, 240))
    first_ball = np.memmap(OUT / "initial-ball.yuv", np.uint8, "r", shape=(240, 96, 64))
    last_ball = np.memmap(OUT / "return-ball.yuv", np.uint8, "r", shape=(240, 96, 64))
    last = np.load(AUDIT / "return-regions-final-20261122/frames.npy", mmap_mode="r")
    finisher = ROOT / "jobs/fleet/inbox/20261118T171044Z-cb2e385d/full/published.tar"
    assert sha(finisher) == "48542980357c28e5e2857e2b0ecb52759e6b78ea5da2f343f148cfc771d6be13"
    crops, masks = {}, {}
    for unit in order:
        job, digest = GEN[unit]
        archive = ROOT / f"jobs/fleet/inbox/{job}/full/published.tar"
        assert sha(archive) == digest
        crops[unit] = np.load(io.BytesIO(member(archive, unit + ".npz")), allow_pickle=False)[
            "crops"
        ]
        masks[unit] = np.load(
            io.BytesIO(member(finisher, unit + "-masks.npz")), allow_pickle=False
        )["masks"]
    whole = hashlib.sha256()
    segments = []
    count = 0

    def add(frame, segment_hash):
        nonlocal count
        assert frame.shape == (272, 480, 3) and frame.dtype == np.uint8
        raw = frame.tobytes()
        whole.update(raw)
        segment_hash.update(raw)
        count += 1

    def rgb_canvas(yuv):
        return cv2.resize(
            tracked_patch.yuv_to_rgb(yuv, 240, 136, False),
            (1920, 1080),
            interpolation=cv2.INTER_CUBIC,
        )

    sh = hashlib.sha256()
    for t in range(240):
        canvas = rgb_canvas(base[t])
        for unit in order:
            paste(canvas, crops[unit][t], initial_players[unit][0]["boxes"][t], masks[unit][t])
        if init_pos[t] is not None:
            x, y = init_pos[t]
            canvas[y : y + 32, x : x + 32] = tracked_patch.yuv_to_rgb(first_ball[t], 64, 64, False)[
                16:48, 16:48
            ]
        add(cv2.resize(canvas, (480, 272), interpolation=cv2.INTER_AREA), sh)
    segments.append({"mode": "cached-player-ball", "frames": 240, "rgb_sha256": sh.hexdigest()})
    gap = OUT / "fallback.ivf"
    gap.write_bytes(sections["fallback_ivf"])
    sh = hashlib.sha256()
    command = [
        ff["path"],
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-c:v",
        "libdav1d",
        "-i",
        str(gap),
        "-vf",
        "scale=480:272:flags=bicubic:in_color_matrix=bt709:out_color_matrix=bt709,format=rgb24",
        "-filter_threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = 480 * 272 * 3
    for _ in range(10825):
        raw = process.stdout.read(frame_bytes)
        assert len(raw) == frame_bytes
        add(np.frombuffer(raw, np.uint8).reshape(272, 480, 3), sh)
    assert not process.stdout.read(1) and process.wait(timeout=30) == 0
    segments.append({"mode": "native-fallback", "frames": 10825, "rgb_sha256": sh.hexdigest()})
    sh = hashlib.sha256()
    for t in range(240):
        canvas = np.array(last[t], copy=True)
        if return_pos[t] is not None:
            x, y = return_pos[t]
            canvas[y : y + 32, x : x + 32] = tracked_patch.yuv_to_rgb(last_ball[t], 64, 64, False)[
                16:48, 16:48
            ]
        add(cv2.resize(canvas, (480, 272), interpolation=cv2.INTER_AREA), sh)
    segments.append({"mode": "cached-player-ball", "frames": 240, "rgb_sha256": sh.hexdigest()})
    assert count == 11305 and sum(x["frames"] for x in segments) == count
    result = {
        "frames": count,
        "fps": [50, 1],
        "rgb_size": [480, 272],
        "rgb_stream_sha256": whole.hexdigest(),
        "segments": segments,
        "packet_sha256": hashlib.sha256(packet).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "scope": "source-free sequential display frames from paid inputs and already sealed player/SAM outputs; not live inference or a viewer study",
    }
    (OUT / "receipt.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
