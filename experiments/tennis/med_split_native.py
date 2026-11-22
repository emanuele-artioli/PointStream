"""Verify and natively decode the ordered independently priced tennis shot sections."""

import struct
import subprocess

import cv2
import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import med_mixed_channel as mixed
from experiments.tennis import w8_receiver_native_bootstrap as native
from experiments.tennis.player_requalification import sha256
from src.pointstream.tracked_patch import yuv_to_rgb


def audit(binary, out):
    # All allowable command lines are literal, single-use native IVF decode calls.
    commands = [native.decode_argv(binary, out / f"v{i}.ivf", out / f"v{i}.yuv") for i in range(3)]
    hooks = {tuple(argv): native._audit_policy(argv) for argv in commands}
    default = next(iter(hooks.values()))

    def check(event, args):
        if event == "subprocess.Popen":
            key = tuple(args[1])
            if key not in hooks:
                raise PermissionError("unexpected split native call")
            hooks[key](event, args)
        else:
            default(event, args)

    return check


def decode(path, original, channel, out, ff):
    data = path.read_bytes()
    profile = packet._embedded_profile(data)
    prior = packet._embedded_profile(original)
    op, _ = packet.wire.unpack(original, prior)
    fields, roles = packet.wire.unpack(data, profile)
    first, last = channel["start"], channel["stop"]
    intervals = [
        (0, first, "conventional", 320, 192),
        (first, last, "objects", 160, 96),
        (last, 240, "conventional", 320, 192),
    ]
    parts = [
        dict(name=f"v{i}", begin=b, end=e, mode=m, width=w, height=h)
        for i, (b, e, m, w, h) in enumerate(v for v in intervals if v[1] > v[0])
    ]
    if (
        profile != {**prior, "kind": "joint-split-trajectory-v1", "parts": parts}
        or set(fields) != {"poses", "trajectory", *(r["name"] for r in parts)}
        or any(fields[k] != op[k] for k in ("poses", "trajectory"))
        or sum(roles.values()) != len(data)
        or not 1 <= len(data) <= len(original)
    ):
        raise ValueError("invalid fixed split timeline, rate, or changed paid foreground")
    decoded = []
    for p in parts:
        name, w, h = p["name"], p["width"], p["height"]
        n = p["end"] - p["begin"]
        coded = fields[name]
        header = struct.unpack_from("<4sHH4sHHIIII", coded)
        if header != (b"DKIF", 0, 32, b"AV01", w, h, 50, 1, n, 0):
            raise ValueError("invalid native segment format")
        i = 32
        for tick in range(n):
            size, pts = struct.unpack_from("<IQ", coded, i)
            if not size or pts != tick or (i := i + 12 + size) > len(coded):
                raise ValueError("invalid split segment tick")
        if i != len(coded):
            raise ValueError("trailing split data")
        ivf, yuv = out / (name + ".ivf"), out / (name + ".yuv")
        ivf.write_bytes(coded)
        subprocess.run(
            native.decode_argv(ff, ivf, yuv),
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
        if yuv.stat().st_size != n * w * h * 3 // 2:
            raise ValueError("incomplete native decode")
        decoded.append((p, np.memmap(yuv, np.uint8, "r", shape=(n, h * 3 // 2, w)), sha256(yuv)))
    wire = {
        **channel["wire"],
        "joint_sha256": packet.wire.sha(data),
        "joint_bytes": len(data),
        "joint_roles": roles,
        "sections": parts,
    }
    return decoded, wire


def canvas(parts, tick, channel):
    for p, raw, _ in parts:
        if p["begin"] <= tick < p["end"]:
            frame = raw[tick - p["begin"]]
            if p["mode"] == "objects":
                return mixed.canvas(frame, tick, channel)
            rgb = yuv_to_rgb(frame, p["width"], p["height"], False)
            return cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    raise ValueError("undelivered tick")
