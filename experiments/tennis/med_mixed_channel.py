"""Read the paid mixed-mode stream and reconstruct the native background and ball."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import common_receiver_manifest as manifest
from src.pointstream import ball
from src.pointstream.tracked_patch import yuv_to_rgb


def decode(joint: Path, cache: Path, carrier: Path, expected: str) -> dict:
    """Regenerate the exact authenticated generator input from the delivered bytes."""
    blob, initial = joint.read_bytes(), cache.read_bytes()
    jp, cp = packet._embedded_profile(blob), packet._embedded_profile(initial)
    delivered, joint_roles = packet.wire.unpack(blob, jp)
    assets, init_roles = packet.wire.unpack(initial, cp)
    if (
        set(jp) != {"kind", "player", "cache_sha256", "ball_canvas", "display", "order"}
        or jp["kind"] != "joint-trajectory-v1"
        or jp["cache_sha256"] != packet.wire.sha(initial)
        or jp["ball_canvas"] != [960, 544]
        or jp["display"] != [1920, 1080]
        or jp["order"] != ["clean_base", "ball", "far_player", "near_player"]
        or set(cp) != {"type", "resources", "runtime"}
        or cp["type"] != "joint-initialization-v1"
        or set(delivered) != {"base", "poses", "trajectory"}
        or set(assets) != {"ref0", "ref1"}
        or sum(joint_roles.values()) != len(blob)
        or sum(init_roles.values()) != len(initial)
    ):
        raise ValueError("unexpected mixed-mode packet roles or cache identity")
    profile = {**jp["player"], "resources": cp["resources"]}
    packet._profile(profile)
    copied = packet.wire.pack(
        profile,
        [
            ("base", "frame", delivered["base"]),
            ("keypoints", "frame", delivered["poses"]),
            *((name, "reference", assets[name]) for name in sorted(assets)),
        ],
    )
    runtime = json.loads(manifest.encode(expected))
    del runtime["carrier_sha256"], runtime["composition"]
    if (
        packet.wire.sha(copied) != expected
        or carrier.read_bytes() != copied
        or runtime != cp["runtime"]
        or profile["coded_size"] != [160, 96]
        or profile["recipe"] not in (packet.RECIPE, packet.RECIPE_25, packet.RECIPE_2997, packet.RECIPE_60)
        or len(profile["order"]) != 2
        or len(profile["runs"]) != 2
    ):
        raise ValueError("mixed packet cannot reproduce exact paid conditioning")
    packet._ivf(delivered["base"], 160, 96, tuple(profile["recipe"]["fps"]))
    # Only the PNG-capable generation worker decodes appearance. Geometry uses KPS2 alone.
    _, units = packet._bounded_keypoints(delivered["poses"], profile["order"])
    runs = {(r["start"], r["end"]) for r in profile["runs"]}
    if len(runs) != 1:
        raise ValueError("expected two players on the same active interval")
    start, stop = runs.pop()
    length = 120 if profile["recipe"] == packet.RECIPE else 240
    if stop - start != length or not 0 <= start < stop <= 240:
        raise ValueError("unsupported active interval")
    geometry = {}
    for u in units:
        if not np.array_equal(u["t"], np.arange(start, stop)):
            raise ValueError("missing paid pose")
        geometry[u["name"]] = u["t"], u["kp"], packet.k6b.run_boxes(list(u["kp"]))
    pos, looks = ball.decode_frames(delivered["trajectory"], 240, 960, 544)
    if ball.encode_frames(pos, looks, 960, 544) != delivered["trajectory"] or any(
        p is not None for t, p in enumerate(pos) if not start <= t < stop
    ):
        raise ValueError("noncanonical trajectory or symbol outside the active run")
    return dict(
        profile=profile,
        base=delivered["base"],
        geometry=geometry,
        start=start,
        stop=stop,
        pos=pos,
        velocity=ball.velocities(pos),
        looks=looks,
        wire=dict(
            joint_sha256=packet.wire.sha(blob),
            cache_sha256=packet.wire.sha(initial),
            joint_bytes=len(blob),
            cache_bytes=len(initial),
            joint_roles=joint_roles,
            cache_roles=init_roles,
            active_ticks=[start, stop],
            ball_positions=sum(p is not None for p in pos),
        ),
    )


def smoke_ticks(start: int, stop: int | None = None) -> list[int]:
    if stop is not None and stop - start == 240:
        return [0, 39, 119, 239]
    return sorted([start, start + 39, start + 91, 0 if start else 239])


def canvas(raw: np.ndarray, tick: int, channel: dict | None = None) -> np.ndarray:
    rgb = yuv_to_rgb(raw, 160, 96, False)
    if channel is not None and channel["start"] <= tick < channel["stop"]:
        scene = cv2.resize(rgb, (960, 544), interpolation=cv2.INTER_CUBIC)
        rgb = ball.render(scene, tick, channel["pos"], channel["velocity"], channel["looks"])
    return cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC)
