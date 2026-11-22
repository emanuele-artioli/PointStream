"""CPU compatibility and paid-order smoke; synthetic compositing, no quality result."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import common_player_al028_finish as finish
from experiments.tennis import common_player_packet as packet
from experiments.tennis import common_receiver_manifest as manifest
from src.pointstream import ball

ROOT = Path("/private/tmp/ps-al028-donor25-corrected-20261122")
SHA = "3aa0bbd4bfc7cd2c479dc2ce8128144a24c96770c01ed9f19f55c77f0f0f525f"
YUV = Path("/private/tmp/ps-al028-native25-pinned.yuv")
BALL = Path("/private/tmp/ps-al028-ball-sender-full.payload")


def main():
    carrier = (ROOT / "players.pscp").read_bytes()
    profile = packet._embedded_profile(carrier)
    section, _, geometry = packet.unpack(carrier)
    base = section["base"]
    packet._ivf(base, 160, 96, (25, 1))
    section, roles = packet.wire.unpack(carrier, profile)
    verified = manifest.verify(ROOT / "players.manifest", SHA)
    if hashlib.sha256(YUV.read_bytes()).hexdigest() != finish.YUV_SHA:
        raise ValueError("pinned native YUV differs")
    if len(base) != 8858 or sum(roles.values()) != len(carrier) or roles["reference"] != 155683:
        raise ValueError("paid cold/warm section accounting")
    pos, look = ball.decode_frames(BALL.read_bytes(), 240, 960, 544)
    if ball.encode_frames(pos, look, 960, 544) != BALL.read_bytes():
        raise ValueError("paid ball payload round-trip")
    ticks = [0, 11, 23, 43]
    native = np.memmap(YUV, np.uint8, mode="r", shape=(240, 144, 160))
    velocity = ball.velocities(pos)
    for tick in ticks:
        small = finish.yuv_to_rgb(native[tick], 160, 96, False)
        canvas = cv2.resize(small, (960, 544), interpolation=cv2.INTER_CUBIC)
        painted = ball.render(canvas, tick, pos, velocity, look)
        display = cv2.resize(painted, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        name = profile["order"][0]
        synthetic = np.full((784, 512, 3), 17, dtype=np.uint8)
        mask = np.ones((784, 512), dtype=bool)
        composed, row = finish.compose_tick(
            display, [name], {name: (synthetic, [0, 0, 100, 100])}, {name: mask}
        )
        if (
            not np.all(composed[:100, :100] == 17)
            or not np.array_equal(composed[100:, 100:], display[100:, 100:])
            or row["status"][name] != "composed"
        ):
            raise ValueError("player-after-ball raster order")
    result = {
        "scope": "CPU compatibility smoke only: real paid carrier/native base/ball, synthetic player and mask. No model output or quality result.",
        "carrier_sha256": SHA,
        "native_sha256": finish.YUV_SHA,
        "frames": 240,
        "fps": 25,
        "tested_ticks": ticks,
        "order": "decoded base -> fitted ball -> synthetic player",
        "roles": roles,
        "manifest_bytes": verified["manifest_bytes"],
        "ball_bytes": BALL.stat().st_size,
        "units": list(geometry),
        "smoke": "pass",
    }
    Path(
        "docs/experiments/joint-replication-feasibility-20261122/al028-finish-cpu-smoke.json"
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
