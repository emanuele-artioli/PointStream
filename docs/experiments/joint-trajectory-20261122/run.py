"""Encode, reparse and recompose a cleaned court, cached players and fitted ball."""

from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "starved-base-local-20261122"))
import crops  # noqa: E402
from experiments.tennis import common_player_packet as players  # noqa: E402
from experiments.tennis import w8_payload as wire  # noqa: E402
from src.codecs import svtav1  # noqa: E402
from src.pointstream import ball, ball_stream  # noqa: E402
from src.pointstream.tracked_patch import yuv_to_rgb  # noqa: E402

CLEAN = Path("/private/tmp/ps-clean-court-grabcut-20261119/clean.yuv")
SYMBOLS = Path("/private/tmp/ps-machine-twenty-exact-20261122/pointstream.psbl")
CAPS = (20, 33)
WIDTH, HEIGHT = 160, 96


def cpu():
    return sum(
        resource.getrusage(k).ru_utime + resource.getrusage(k).ru_stime
        for k in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    )


def sha(data):
    return hashlib.sha256(data).hexdigest()


def open_packet(blob):
    profile = players._embedded_profile(blob)
    return profile, wire.unpack(blob, profile)[0]


def run(out):
    start = cpu()
    cv2.setNumThreads(1)
    out.mkdir(exist_ok=False)
    tools = {k: svtav1.tool(k) for k in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    assert (
        crops.file_sha(CLEAN) == "4f27937bff116d51aab914a1c2e7262b51ef14483a2e2455551c2b786a684dfc"
    )
    assert (
        crops.file_sha(SYMBOLS)
        == "35fac120eb3e3e1ac9472ab81664e1bd682092abbed1b7adc7ec1c148afc07b3"
    )
    original = (crops.RUN / "clean-60/players.pscp").read_bytes()
    assert sha(original) == "c5d62404913130a759cbe0ced630dc8fa9743140e3dacbc74078d7afd0d5e14f"
    oldparts, _, old_inputs = players.unpack(original)
    profile = players._embedded_profile(original)
    profile["coded_size"] = [WIDTH, HEIGHT]
    symbols = ball_stream.unpack(SYMBOLS.read_bytes()).ball_payload
    recipe = json.loads((crops.RUN / "clean-60/receiver.manifest").read_text())
    assert recipe.pop("carrier_sha256") == sha(original)
    assert recipe.pop("composition") == "decoded-base,farthest-to-nearest-players,optional-ball"
    cache_profile = {
        "type": "joint-initialization-v1",
        "runtime": recipe,
        "resources": profile["resources"],
    }
    cache = wire.pack(
        cache_profile, [(k, "reference", v) for k, v in oldparts.items() if k.startswith("ref")]
    )
    (out / "initial.psr").write_bytes(cache)
    restored_cache, refs = open_packet((out / "initial.psr").read_bytes())
    manifest = {
        "kind": "joint-trajectory-v1",
        "player": {k: v for k, v in profile.items() if k != "resources"},
        "cache_sha256": sha(cache),
        "ball_canvas": [960, 544],
        "display": [1920, 1080],
        "order": ["clean_base", "ball", "far_player", "near_player"],
    }

    def pack(base):
        return wire.pack(
            manifest,
            [
                ("base", "frame", base),
                ("poses", "frame", oldparts["keypoints"]),
                ("trajectory", "frame", symbols),
            ],
        )

    overhead = len(pack(b""))
    assert overhead + 32 + 12 * 240 < min(CAPS) * 600
    raw = out / "clean.yuv"
    cmd = [
        tools["ffmpeg"]["path"],
        "-v",
        "error",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pixel_format",
        "yuv420p",
        "-video_size",
        "960x544",
        "-framerate",
        "50",
        "-i",
        str(CLEAN),
        "-vf",
        "scale=160:96:flags=area",
        "-filter_threads",
        "1",
        "-frames:v",
        "240",
        "-f",
        "rawvideo",
        str(raw),
    ]
    subprocess.run(cmd, check=True, timeout=90)

    def encode(crf, n):
        directory = out / f"base-{crf}-{n}"
        directory.mkdir()
        return svtav1.code(
            raw,
            directory,
            width=WIDTH,
            height=HEIGHT,
            fps=Fraction(50),
            frames=n,
            crf=crf,
            preset=4,
            threads=1,
            full_range=False,
            encoder=tools["SvtAv1EncApp"]["path"],
            decoder=tools["dav1d"]["path"],
            timeout=150,
        )

    before = cpu()
    smoke = encode(50, 24)
    assert smoke["decoded_frames"] == smoke["temporal_units"] == 24
    forecast = (cpu() - before) * 10 * 12 * 1.5 + 180
    assert forecast < 1000
    trials = {}

    def test(q):
        if q not in trials:
            r = encode(q, 240)
            assert r["decoded_frames"] == r["temporal_units"] == 240
            trials[q] = r
        return len(pack(Path(trials[q]["stream"]).read_bytes()))

    test(36)
    test(70)
    streams = {}
    frames = {}
    for cap in CAPS:
        low, high = 36, 70
        while high - low > 1:
            mid = (low + high) // 2
            if test(mid) <= cap * 600:
                high = mid
            else:
                low = mid
        fits = [q for q in trials if test(q) <= cap * 600]
        q = min(fits) if fits else min(trials, key=test)
        base = Path(trials[q]["stream"]).read_bytes()
        file = out / f"joint-{cap}.psw"
        file.write_bytes(pack(base))
        payload = file.read_bytes()
        rcfg, parts = open_packet(payload)
        assert rcfg["cache_sha256"] == sha(cache) and restored_cache["runtime"] == recipe
        resolved_profile = {**rcfg["player"], "resources": restored_cache["resources"]}
        assert resolved_profile == profile
        condition = players.build(resolved_profile, parts["base"], parts["poses"], refs)
        _, _, recovered = players.unpack(condition)
        for unit in old_inputs:
            assert len(recovered[unit]) == len(old_inputs[unit])
            for a, b in zip(recovered[unit], old_inputs[unit]):
                assert a["seed"] == b["seed"]
                assert all(np.array_equal(a[k], b[k]) for k in ("kp", "boxes", "t", "reference"))
        independent = out / f"decoded-{cap}.yuv"
        ivf = out / f"verified-{cap}.ivf"
        ivf.write_bytes(parts["base"])
        subprocess.run(
            svtav1.decode_command(tools["dav1d"]["path"], ivf, independent, threads=1),
            check=True,
            timeout=60,
            capture_output=True,
        )
        assert crops.file_sha(independent) == crops.file_sha(Path(trials[q]["decoded"]))
        decoded_track, decoded_looks = ball.decode_frames(parts["trajectory"], 240, 960, 544)
        assert ball.encode_frames(decoded_track, decoded_looks, 960, 544) == symbols
        frames[cap] = np.memmap(independent, np.uint8, "r", shape=(240, 144, 160))
        streams[cap] = dict(
            crf=q,
            bytes=len(payload),
            cold_bytes=len(payload) + len(cache),
            kbps=len(payload) / 600,
            cold_kbps=(len(payload) + len(cache)) / 600,
            fits=len(payload) <= 600 * cap,
            wire_sha256=sha(payload),
            base_bytes=len(base),
            reference_bytes=len(cache),
            pose_bytes=len(parts["poses"]),
            ball_bytes=len(symbols),
            framing_bytes=len(payload) - len(base) - len(parts["poses"]) - len(symbols),
            decoded_yuv_sha256=crops.file_sha(independent),
        )
    for row in streams.values():
        assert row["bytes"] == sum(
            row[k] for k in ("base_bytes", "pose_bytes", "ball_bytes", "framing_bytes")
        )
    assert (
        crops.file_sha(crops.LABELS)
        == "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"
    )
    assert (
        crops.file_sha(crops.BUNDLE)
        == "5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb"
    )
    records = {r["tick"]: r for r in json.loads(crops.LABELS.read_text())["frames"]}
    assert list(records) == crops.TICKS
    finish = json.loads((crops.STAGE / "finish/finish.json").read_text())
    audit = json.loads((crops.STAGE / "finish-full-audit.json").read_text())
    assert audit["all_frames_exact"] and audit["carrier_sha256"] == finish["carrier_sha256"]
    utc = time.time()
    with ExitStack() as stack:
        truth = stack.enter_context(zipfile.ZipFile(crops.BUNDLE))
        archive = stack.enter_context(zipfile.ZipFile(out / "crops.zip", "x"))
        gens = dict(
            crops.bind_generation(
                stack,
                crops.AUDIT / f"player{i}-new-full.tar",
                crops.STAGE / f"player{i}-full.json",
                finish,
            )
            for i in (1, 2)
        )
        masks = {}
        for unit in crops.UNITS:
            path = crops.STAGE / "finish" / (unit + "-masks.npz")
            assert crops.file_sha(path) == finish["mask_files"][unit]
            z = stack.enter_context(zipfile.ZipFile(path))
            assert np.array_equal(crops.read_small(z, "t"), np.arange(240))
            masks[unit] = stack.enter_context(
                crops.array_frames(z, "masks", (240, 784, 512), np.bool_)
            )
        velocity = ball.velocities(decoded_track)
        sse = {cap: 0 for cap in CAPS}
        rgb_digests = {cap: hashlib.sha256() for cap in CAPS}
        for tick in range(240):
            gen = {u: next(gens[u][1]) for u in crops.UNITS}
            mask = {u: next(masks[u]) for u in crops.UNITS}
            images = {}
            for cap in CAPS:
                small = yuv_to_rgb(frames[cap][tick], WIDTH, HEIGHT, False)
                scene = cv2.resize(small, (960, 544), interpolation=cv2.INTER_CUBIC)
                scene = ball.render(scene, tick, decoded_track, velocity, decoded_looks)
                display = cv2.resize(scene, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                support = np.zeros((1080, 1920), bool)
                for u in crops.UNITS:
                    area, status = crops.overlay(
                        display, support, gen[u], gens[u][0][tick], mask[u]
                    )
                    assert (area, status) == (
                        finish["rows"][tick]["projected_pixels"][u],
                        finish["rows"][tick]["status"][u],
                    )
                rgb_digests[cap].update(display)
                if tick in (78, 86, 94):
                    # The three phases and rectangle were frozen by the original source-only stroke review.
                    crop_bgr = display[390:1032, 854:1324, ::-1]
                    (out / f"serve-{cap}-{tick}.png").write_bytes(crops.png(crop_bgr))
                images[str(cap)] = display
            if tick == 23:
                assert (time.time() - utc) * 10 * 1.5 + 60 < 700
            if tick in records:
                r = records[tick]
                png = truth.read(f"frame-{tick:03d}/source.png")
                assert sha(png) == r["source_png_sha256"]
                im = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
                images["source"] = im
                for cap in CAPS:
                    delta = im.astype(np.int32) - images[str(cap)]
                    sse[cap] += int(np.sum(delta * delta, dtype=np.int64))
                for region, y, x, maskcrop in crops.supports(r):
                    prefix = f"{tick:03d}/{region}/"
                    archive.writestr(prefix + "mask.png", crops.png(maskcrop))
                    for name, img in images.items():
                        archive.writestr(prefix + name + ".png", crops.png(img[y, x, ::-1]))
    for cap in CAPS:
        streams[cap]["all_240_rgb_sha256"] = rgb_digests[cap].hexdigest()
        streams[cap]["reviewed_full_rgb_psnr"] = 10 * math.log10(
            255**2 * 12 * 1920 * 1080 * 3 / sse[cap]
        )
    result = dict(
        streams=streams,
        trials={q: r["file_bytes"] for q, r in trials.items()},
        input_player_carrier_sha256=sha(original),
        source_clip_sha256="b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d",
        cache_sha256=sha(cache),
        clean_sha256=crops.file_sha(CLEAN),
        ball_source=crops.file_sha(SYMBOLS),
        all_240_conditions_exact=True,
        reviewed_ticks=crops.TICKS,
        tools=tools,
        forecast=forecast,
        cpu_seconds=cpu() - start,
        script=crops.file_sha(Path(__file__)),
        generation_audit_sha256=crops.file_sha(crops.STAGE / "finish-full-audit.json"),
        crops_sha256=crops.file_sha(out / "crops.zip"),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(streams))


if __name__ == "__main__":
    run(Path(sys.argv[1]))
