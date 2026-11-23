"""Re-encode the authenticated Sydney background without regenerating the foreground."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.tennis import common_player_packet as packet
from experiments.tennis import common_receiver_manifest as manifest
from experiments.tennis import med_mixed_channel as mixed
from experiments.tennis import w8_receiver_native_bootstrap as native
from experiments.tennis.player_requalification import sha256
from experiments.tennis.shanghai_ball_rate_crf import TOOLS
from src.codecs import svtav1

FRAMES = 240
CLEAN_SHA = "810783b90f05fd6f7413a6ff2d9a1db4a8903c2d5bdce05a89ddbe81e07238e8"
ORIGINAL_SHA = "f9ab35e8d3cd6708e726fe57d9d7ba7cc8943285ac4535524491f5cfe0598384"
CACHE_SHA = "1f191ca3d078c258c6252003d746b6077c268bb98cff7260c4c960cab5a8d569"
CAPS = (42500, 68000, 96000)


def rebind(base, size, original, initial):
    """Construct the paid stream and its expanded generation carrier."""
    old = packet._embedded_profile(original)
    sections, _ = packet.wire.unpack(original, old)
    cp = packet._embedded_profile(initial)
    images, _ = packet.wire.unpack(initial, cp)
    if old["cache_sha256"] != packet.wire.sha(initial):
        raise ValueError("original initialization identity differs")
    player = {**old["player"], "coded_size": list(size)}
    if player["recipe"] != packet.RECIPE_60:
        raise ValueError("expected the native 60 fps Sydney stream")
    carrier = packet.build(
        {**player, "resources": cp["resources"]}, base, sections["poses"], images
    )
    joint = packet.wire.pack(
        {**old, "player": player},
        [
            (k, "frame", base if k == "base" else sections[k])
            for k in ("base", "poses", "trajectory")
        ],
    )
    return joint, carrier, manifest.encode(packet.wire.sha(carrier))


def encode_inputs(clean, output, ffmpeg, size, frames):
    w, h = size
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            "960x544",
            "-r",
            "60",
            "-i",
            str(clean),
            "-vf",
            f"scale={w}:{h}:flags=area",
            "-frames:v",
            str(frames),
            "-f",
            "rawvideo",
            "-n",
            str(output),
        ],
        check=True,
        timeout=180,
    )
    assert output.stat().st_size == frames * w * h * 3 // 2


def encode(raw, work, ffmpeg, encoder, size, frames, crf):
    w, h = size
    work.mkdir()
    stream, decoded = work / "stream.ivf", work / "decoded.yuv"
    svtav1.run(
        svtav1.encode_command(
            encoder,
            raw,
            stream,
            width=w,
            height=h,
            fps=Fraction(60),
            frames=frames,
            crf=crf,
            preset=4,
            full_range=False,
        ),
        290,
    )
    subprocess.run(
        native.decode_argv(Path(ffmpeg), stream, decoded),
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    assert decoded.stat().st_size == frames * w * h * 3 // 2
    data = stream.read_bytes()
    assert len(svtav1.ivf_frames(data)) == frames
    return data, sha256(decoded)


def next_crf(sizes):
    for cap in CAPS:
        below = [q for q, n in sizes.items() if n <= cap]
        above = [q for q, n in sizes.items() if n > cap]
        if below and max(sizes[q] for q in below) >= 0.90 * cap:
            continue
        if below and above:
            lo, hi = max(above), min(below)
            q = (lo + hi) // 2
        elif below:
            q = min(sizes) - 12
        else:
            q = max(sizes) + 8
        if 1 <= q <= 70 and q not in sizes:
            return q
    return None


def run(a):
    stage = Path(os.environ["PS_STAGE_DIR"])
    if a.validate:
        result = validate(stage)
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result) + "\n")
        return
    start = time.monotonic()
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "wider"
    work.mkdir()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and len(os.sched_getaffinity(0)) == 1
    assert a.frames == {"smoke": 24, "full": FRAMES}[os.environ["PS_STAGE"]]
    assert sha256(a.clean) == CLEAN_SHA and a.clean.stat().st_size == FRAMES * 960 * 544 * 3 // 2
    assert sha256(a.original) == ORIGINAL_SHA and sha256(a.cache) == CACHE_SHA
    tools = {name: svtav1.tool(name) for name in ("SvtAv1EncApp",)}
    ffmpeg = str(Path(sys.prefix) / "bin/ffmpeg")
    encoder = tools["SvtAv1EncApp"]
    assert sha256(Path(ffmpeg)) == TOOLS["ffmpeg"] and encoder["sha256"] == TOOLS["SvtAv1EncApp"]
    original, initial = a.original.read_bytes(), a.cache.read_bytes()
    size = (a.width, a.width * 3 // 5)
    raw = work / "wider.yuv"
    encode_inputs(a.clean, raw, ffmpeg, size, a.frames)
    rates = {}
    native_parity = None
    if a.frames == FRAMES:
        old_raw = work / "original.yuv"
        encode_inputs(a.clean, old_raw, ffmpeg, (160, 96), FRAMES)
        old_base, old_decoded = encode(
            old_raw, work / "original", ffmpeg, encoder["path"], (160, 96), FRAMES, 70
        )
        rebuilt, _, _ = rebind(old_base, (160, 96), original, initial)
        if rebuilt != original or old_decoded != a.original_native_sha:
            raise ValueError("failed original Sydney encoded and decoded byte parity")
        native_parity = old_decoded
    planned = [50] if a.frames != FRAMES else [35, 50, 65]
    sizes = {}
    while planned and len(rates) < 6:
        q = planned.pop(0)
        base, yuv_sha = encode(raw, work / f"q{q}", ffmpeg, encoder["path"], size, a.frames, q)
        result = {"crf": q, "native_sha256": yuv_sha, "encoded_bytes": len(base)}
        if a.frames == FRAMES:
            joint, carrier, carrier_manifest = rebind(base, size, original, initial)
            folder = work / f"q{q}"
            paths = {
                "rally.psj": joint,
                "players.pscp": carrier,
                "initial.psr": initial,
                "players.manifest": carrier_manifest,
            }
            for name, data in paths.items():
                (folder / name).write_bytes(data)
            checked = mixed.decode(
                folder / "rally.psj",
                folder / "initial.psr",
                folder / "players.pscp",
                packet.wire.sha(carrier),
                allow_wider_base=True,
            )
            assert checked["base"] == base and checked["coded_size"] == size
            sizes[q] = len(joint)
            result.update(
                joint_bytes=len(joint),
                initial_bytes=len(initial),
                files={name: packet.wire.sha(data) for name, data in paths.items()},
            )
            if not planned and (following := next_crf(sizes)) is not None:
                planned.append(following)
        rates[q] = result
    chosen = {}
    for cap in CAPS:
        affordable = [q for q, size_bytes in sizes.items() if size_bytes <= cap]
        if affordable:
            q = max(affordable, key=lambda candidate: sizes[candidate])
            chosen[cap] = q
            for name in rates[q]["files"]:
                source = work / f"q{q}" / name
                target = stage / f"crf{q}-{name}"
                if not target.exists():
                    target.write_bytes(source.read_bytes())
    result = {
        "frames": a.frames,
        "size": size,
        "foreground_bytes_reused": True,
        "clean_sha256": CLEAN_SHA,
        "original_sha256": ORIGINAL_SHA,
        "original_native_parity": native_parity,
        "warm_budgets": chosen,
        "rates": list(rates.values()),
        "elapsed_s": time.monotonic() - start,
    }
    (stage / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    validate(stage)
    if a.frames == FRAMES:
        (stage / "full-validation.json").write_text(json.dumps(validate(stage)) + "\n")


def validate(stage):
    r = json.loads((stage / "calibration.json").read_text())
    assert r["clean_sha256"] == CLEAN_SHA and r["foreground_bytes_reused"]
    if r["frames"] == FRAMES:
        assert r["original_native_parity"] and r["warm_budgets"]
        for rate in r["rates"]:
            q = rate["crf"]
            if q in r["warm_budgets"].values():
                assert all(
                    sha256(stage / f"crf{q}-{name}") == v for name, v in rate["files"].items()
                )
    else:
        assert r["frames"] == 24 and len(r["rates"]) == 1
        assert r["elapsed_s"] * 105 < 1250
    return {"passed": True, "frames": r["frames"], "native": True, "sha_bound_clean": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validate", action="store_true")
    for k in ("clean", "original", "cache"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--original-native-sha")
    p.add_argument("--width", type=int, choices=(320, 640), default=320)
    p.add_argument("--frames", type=int, choices=(24, FRAMES))
    run(p.parse_args())
