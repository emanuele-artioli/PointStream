"""Size-only search of a stronger conventional control at the matched tile budget."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import resource
import subprocess
import time
import zipfile

import cv2
import numpy as np

from experiments.tennis import common_player_ball as ball, common_player_packet as player
from experiments.tennis.w1_roi import roi_text
from src.codecs import svtav1
from src.pointstream.tracked_patch import yuv_to_rgb
from run import TICKS, crops, sha


def cpu():
    return sum(
        r.ru_utime + r.ru_stime
        for r in (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
    )


def main(experiment, root):
    cv2.setNumThreads(1)
    root.mkdir(exist_ok=False)
    start = cpu()
    begin = time.monotonic()
    previous = json.loads((experiment / "render.json").read_text())
    pair = next(p for p in previous["pairs"] if p["target"] == 63 and p["matched"])
    target = round(pair["tile_kbps"] * 600)
    tools = previous["tools"]
    assert all(sha(Path(t["path"])) == t["sha256"] for t in tools.values())
    source = crops.ROOT / "ps-native-ctl-20261118-full/source.yuv"
    assert sha(source) == "6cae4bd7a46bfa4bd2a7a7ec5c926647ea43d9e7a5366eb96e464eaffe1e603c"
    w, h = 480, 272
    raw = root / "480.yuv"
    conversion = [
        tools["ffmpeg"]["path"],
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-video_size",
        "960x544",
        "-framerate",
        "50",
        "-i",
        str(source),
        "-filter_threads",
        "1",
        "-vf",
        "scale=480:272:flags=area,format=yuv420p",
        "-f",
        "rawvideo",
        str(raw),
    ]
    subprocess.run(conversion, check=True, timeout=90)
    assert raw.stat().st_size == 240 * w * h * 3 // 2
    old = crops.ROOT / "selected-rate-support-linux-png/packets/48/players.pscp"
    assert sha(old) == "144bd0870dd37b6b586663473d12a50bdb78015626661eccaa7a427fb60c9b87"
    records = player.unpack(old.read_bytes())[2]
    arm = experiment / f"q{pair['tile_crf']:g}"
    assert sha(arm / "native.pscp") == next(
        r["carrier_sha256"] for r in previous["rows"] if r["crf"] == pair["tile_crf"]
    )
    origins, _, _ = ball._open(
        (arm / "native.pscp").read_bytes(),
        (arm / "ball.sidecar").read_bytes(),
        tools["dav1d"]["sha256"],
    )
    masks = np.zeros((240, h, w), bool)
    for t, mask in enumerate(masks):
        boxes = [runs[0]["boxes"][t] for runs in records.values()]
        if origins[t] is not None:
            boxes.append((*origins[t], 32, 32))
        for x, y, ww, hh in boxes:
            mask[
                max(0, math.floor(y * h / 1080)) : min(h, math.ceil((y + hh) * h / 1080)),
                max(0, math.floor(x * w / 1920)) : min(w, math.ceil((x + ww) * w / 1920)),
            ] = True
    roi = root / "roi.txt"
    roi.write_text(roi_text(masks, w, h, -20, outside=True))
    rows = {}

    def encode(mode, q, n):
        key = (mode, q, n)
        if key not in rows:
            dest = root / f"{mode}-{q:g}-{n}"
            dest.mkdir()
            row = svtav1.code(
                raw,
                dest,
                width=w,
                height=h,
                fps=Fraction(50),
                frames=n,
                crf=q,
                preset=4,
                threads=2,
                full_range=False,
                encoder=tools["SvtAv1EncApp"]["path"],
                decoder=tools["dav1d"]["path"],
                timeout=60,
                extra=["--roi-map-file", str(roi)] if mode == "roi" else None,
            )
            assert row["decoded_frames"] == n and row["temporal_units"] == n
            row.update(
                mode=mode,
                q=q,
                n=n,
                decoded_sha256=sha(Path(row["decoded"])),
                rate=row["file_bytes"] / (n / 50) * 8 / 1000,
            )
            rows[key] = row
        return rows[key]

    smoke = encode("plain", 60, 24)
    assert (smoke["encode_cpu_seconds"] + smoke["decode_cpu_seconds"]) * 10 * 1.5 < 60
    pilot = encode("plain", 36, 240)
    # Use a full, harder encode to forecast the remaining size-only search.
    assert (pilot["encode_cpu_seconds"] + pilot["decode_cpu_seconds"]) * 20 * 1.5 + cpu() - start < 490
    selected = {}
    for mode in ["plain", "roi"]:
        lo, hi = 36.0, 63.0
        low = encode(mode, lo, 240)
        high = encode(mode, hi, 240)
        if low["file_bytes"] >= target >= high["file_bytes"]:
            for _ in range(8):
                mid = round((lo + hi) * 2) / 4
                if mid in [lo, hi]:
                    break
                trial = encode(mode, mid, 240)
                if abs(trial["file_bytes"] / target - 1) <= 0.005:
                    break
                if trial["file_bytes"] > target:
                    lo = mid
                else:
                    hi = mid
                if cpu() - start > 400:
                    raise RuntimeError("CPU budget for size search")
        candidates = [r for (m, _, n), r in rows.items() if m == mode and n == 240]
        closest = min(candidates, key=lambda r: abs(r["file_bytes"] - target))
        if abs(closest["file_bytes"] / target - 1) <= 0.005:
            selected[mode] = closest
        else:
            for category, filtered in [
                ("under", [r for r in candidates if r["file_bytes"] <= target]),
                ("over", [r for r in candidates if r["file_bytes"] >= target]),
            ]:
                if filtered:
                    selected[mode + category] = min(
                        filtered, key=lambda r: abs(r["file_bytes"] - target)
                    )
    assert sha(experiment / "crops.zip") == previous["crops_sha256"]
    labels = crops.ROOT / "foreground-human-reviewed-20261118" / f"{crops.LABEL}.json"
    assert sha(labels) == crops.LABEL
    frames = json.loads(labels.read_text())["frames"]
    with (
        zipfile.ZipFile(root / "crops.zip", "x") as z,
        zipfile.ZipFile(experiment / "crops.zip") as original,
    ):
        for key in original.namelist():
            if Path(key).stem in ("source", "mask"):
                z.writestr(key, original.read(key))
        for arm_name, row in selected.items():
            native = np.memmap(row["decoded"], "uint8", "r", shape=(240, 408, 480))
            for t, f in zip(TICKS, frames, strict=True):
                assert t == f["tick"]
                canvas = cv2.resize(
                    yuv_to_rgb(native[t], w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
                )
                for region, y, x, _ in crops.supports(f):
                    ok, png = cv2.imencode(".png", cv2.cvtColor(canvas[y, x], cv2.COLOR_RGB2BGR))
                    assert ok
                    z.writestr(f"{t:03d}/{region}/{arm_name}.png", png.tobytes())
    assert cpu() - start < 500
    result = dict(
        previous_sha256=sha(experiment / "render.json"),
        target_bytes=target,
        source_sha256=sha(source),
        tools=tools,
        roi_map_sha256=sha(roi),
        size_conversion=conversion,
        driver_sha256=sha(Path(__file__)),
        scaled_source_sha256=sha(raw),
        crop_sha256=sha(root / "crops.zip"),
        all_encodes=list(rows.values()),
        selected=selected,
        cpu_seconds=cpu() - start,
        wall_seconds=time.monotonic() - begin,
    )
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print([(a, r["rate"]) for a, r in selected.items()])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("experiment", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    main(args.experiment, args.output)
