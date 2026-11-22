"""Recompose archived paid receiver caches at two sizes and recover reviewed RGB regions."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "starved-base-local-20261122"))
import crops  # noqa: E402
from independent_crop_audit import array_sha  # noqa: E402
from experiments.tennis import common_player_packet, w8_payload  # noqa: E402
from src.pointstream import ball  # noqa: E402
from src.pointstream.tracked_patch import yuv_to_rgb  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("generations", "paid", "out"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    start = time.process_time()
    cv2.setNumThreads(1)
    finish_path = args.generations / "finish.json"
    assert (
        crops.file_sha(finish_path)
        == "645d719da006b9df133849753ed45b2da00a821958b716c1b4debf5e625ce131"
    )
    finish = json.loads(finish_path.read_text())
    assert (
        crops.file_sha(crops.LABELS)
        == "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"
    )
    assert (
        crops.file_sha(crops.BUNDLE)
        == "5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb"
    )
    labels = {r["tick"]: r for r in json.loads(crops.LABELS.read_text())["frames"]}
    assert list(labels) == crops.TICKS
    headers = json.loads((HERE / "nad000-final-width.json").read_text())
    prior = json.loads((HERE.parent / "joint-trajectory-20261122/result.json").read_text())
    assert (
        crops.file_sha(args.paid / "initial.psr")
        == headers["cache_sha256"]
        == prior["cache_sha256"]
    )
    inputs = HERE.parent / "receiver-replay-20261118/independent-idle-20261118"
    states = {}
    for row in headers["rows"]:
        width, height = row["width"], row["height"]
        path = args.paid / f"w{width}"
        packet = (path / "joint.pscp").read_bytes()
        assert hashlib.sha256(packet).hexdigest() == row["wire_sha256"]
        profile = common_player_packet._embedded_profile(packet)
        parts, _ = w8_payload.unpack(packet, profile)
        assert profile["cache_sha256"] == headers["cache_sha256"]
        assert crops.file_sha(path / "independent.yuv") == row["base_decoded_sha256"]
        track, looks = ball.decode_frames(parts["trajectory"], 240, 960, 544)
        frames = np.memmap(
            path / "independent.yuv", np.uint8, "r", shape=(240, height * 3 // 2, width)
        )
        states[width] = (frames, track, ball.velocities(track), looks, height)
    crop_file = args.out / "crops.zip"
    sums = dict.fromkeys(states, 0)
    digests = {w: hashlib.sha256() for w in states}
    tick_hashes = {w: [] for w in states}
    with ExitStack() as stack:
        truth = stack.enter_context(zipfile.ZipFile(crops.BUNDLE))
        archive = stack.enter_context(zipfile.ZipFile(crop_file, "x"))
        gens, geometry, masks = {}, {}, {}
        for unit in crops.UNITS:
            i = int(unit.rsplit("_", 1)[1])
            meta = json.loads((inputs / f"player{i + 1}-full.json").read_text())
            assert (
                meta["stage"] == "full"
                and meta["exact_conditioning_and_geometry"]
                and meta["unit"] == unit
            )
            gp = args.generations / f"person{i}.npz"
            mp = args.generations / f"person{i}-masks.npz"
            assert (
                crops.file_sha(gp) == meta["new_npz_sha256"] == finish["generated"][unit]["sha256"]
            )
            assert crops.file_sha(mp) == finish["mask_files"][unit]
            gz = stack.enter_context(zipfile.ZipFile(gp))
            mz = stack.enter_context(zipfile.ZipFile(mp))
            assert all(np.array_equal(crops.read_small(z, "t"), np.arange(240)) for z in (gz, mz))
            geometry[unit] = crops.read_small(gz, "box")
            assert array_sha(geometry[unit]) == meta["box_sha256"]
            gens[unit] = stack.enter_context(
                crops.array_frames(gz, "crops", (240, 784, 512, 3), np.uint8)
            )
            masks[unit] = stack.enter_context(
                crops.array_frames(mz, "masks", (240, 784, 512), np.bool_)
            )
        for tick in range(240):
            fg = {u: (next(gens[u]), next(masks[u])) for u in crops.UNITS}
            images = {}
            for w, (base, track, velocity, looks, h) in states.items():
                small = yuv_to_rgb(base[tick], w, h, False)
                scene = cv2.resize(small, (960, 544), interpolation=cv2.INTER_CUBIC)
                scene = ball.render(scene, tick, track, velocity, looks)
                display = cv2.resize(scene, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                support = np.zeros((1080, 1920), bool)
                for u, (gen, mask) in fg.items():
                    assert crops.overlay(display, support, gen, geometry[u][tick], mask) == (
                        finish["rows"][tick]["projected_pixels"][u],
                        finish["rows"][tick]["status"][u],
                    )
                digests[w].update(display)
                tick_hashes[w].append(hashlib.sha256(display).hexdigest())
                images[str(w)] = display
            if tick in labels:
                r = labels[tick]
                png = truth.read(f"frame-{tick:03d}/source.png")
                assert hashlib.sha256(png).hexdigest() == r["source_png_sha256"]
                source = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
                for w in states:
                    delta = source.astype(np.int32) - images[str(w)]
                    sums[w] += int(np.sum(delta * delta, dtype=np.int64))
                images["source"] = source
                for region, y, x, maskcrop in crops.supports(r):
                    prefix = f"{tick:03d}/{region}/"
                    archive.writestr(prefix + "mask.png", crops.png(maskcrop))
                    for name, img in images.items():
                        archive.writestr(prefix + name + ".png", crops.png(img[y, x, ::-1]))
            if tick == 23:
                forecast = (time.process_time() - start) * 15 + 60
                assert forecast < 400, forecast
    aggregates = {k: v.hexdigest() for k, v in digests.items()}
    assert aggregates[160] == prior["streams"]["20"]["all_240_rgb_sha256"]
    result = dict(
        source_sha256=prior["source_clip_sha256"],
        native_frames=240,
        fps=[50, 1],
        rgb_sha256=aggregates,
        per_tick_rgb_sha256=tick_hashes,
        reviewed_ticks=list(labels),
        crop_zip_sha256=crops.file_sha(crop_file),
        positive_all_240_original=True,
        reviewed_full_rgb_psnr={
            w: 10 * math.log10(255**2 * len(labels) * 1920 * 1080 * 3 / v) for w, v in sums.items()
        },
        cost_forecast_cpu_seconds=forecast,
        cpu_seconds=time.process_time() - start,
        opencv=cv2.__version__,
        numpy=np.__version__,
        script_sha256=crops.file_sha(Path(__file__)),
    )
    (args.out / "display.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: result[k] for k in ("rgb_sha256", "reviewed_full_rgb_psnr", "cpu_seconds")})


if __name__ == "__main__":
    main()
