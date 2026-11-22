"""Deliver and independently decode a single-base, source-fitted ball stream near 20 kbps."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import zipfile

import cv2
import numpy as np
from src.pointstream import ball, ball_stream
from src.pointstream.tracked_patch import yuv_to_rgb
from src.codecs import svtav1
from experiments.tennis.w1_roi import roi_text

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "starved-base-local-20261122"))
from crops import LABELS, SOURCE, supports  # noqa: E402


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cpu():
    return sum(
        resource.getrusage(k).ru_utime + resource.getrusage(k).ru_stime
        for k in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    )


def yuv(path, w, h):
    return np.memmap(path, np.uint8, "r", shape=(240, h * 3 // 2, w))


def main(out, target_kbps=20.0):
    width, height = (160, 96) if target_kbps == 20 else (240, 136)
    target_base_bytes = round(target_kbps * 600 - 900)
    out.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    start = cpu()
    tools = {s: svtav1.tool(s) for s in ("ffmpeg", "dav1d", "SvtAv1EncApp")}
    src = SOURCE / "ps-native-ctl-20261118-full/source.yuv"
    trackfile = HERE.parent / "ball-on-development-rally-20261118/track.json"
    assert sha(src) == "6cae4bd7a46bfa4bd2a7a7ec5c926647ea43d9e7a5366eb96e464eaffe1e603c"
    assert sha(trackfile) == "8abbacb923215d2f93b54275eef43cc16ba2c63ec6f46e9c35898a094add7ca6"
    coords = json.loads(trackfile.read_text())["centers"]
    track = [
        (min(959, round(x[0] * 0.5)), min(543, round(x[1] * 544 / 1080))) if x is not None else None
        for x in coords
    ]
    assert len(track) == 240 and sum(x is not None for x in track) == 195
    roi = np.zeros((240, 136, 240), bool)
    for t, p in enumerate(coords):
        if p:
            x, y = p
            roi[
                t,
                max(0, (y - 16) * 136 // 1080) : min(136, (y + 16) * 136 // 1080 + 1),
                max(0, (x - 16) // 8) : min(240, (x + 16) // 8 + 1),
            ] = True
    roifile = out / "roi.txt"
    roifile.write_text(roi_text(roi, 240, 136, -20, outside=True))
    raw = {}
    commands = []
    for w, h in [(160, 96), (240, 136)]:
        raw[w] = out / f"source-{w}.yuv"
        command = [
            tools["ffmpeg"]["path"],
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "yuv420p",
            "-video_size",
            "960x544",
            "-framerate",
            "50",
            "-i",
            str(src),
            "-filter_threads",
            "1",
            "-vf",
            f"scale={w}:{h}:flags=area,format=yuv420p",
            "-frames:v",
            "240",
            "-f",
            "rawvideo",
            str(raw[w]),
        ]
        subprocess.run(command, check=True, timeout=40)
        commands.append(command)
        assert raw[w].stat().st_size == 240 * h * w * 3 // 2
    runs = {}

    def enc(w, mode, q, n=240):
        if (w, mode, q, n) in runs:
            return runs[w, mode, q, n]
        h = {160: 96, 240: 136}[w]
        dest = out / f"{w}-{mode}-{q:g}-{n}"
        dest.mkdir()
        r = svtav1.code(
            raw[w],
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
            extra=["--roi-map-file", str(roifile)] if mode == "roi" else None,
        )
        assert r["decoded_frames"] == r["temporal_units"] == n
        r.update(
            width=w,
            height=h,
            q=q,
            mode=mode,
            n=n,
            decoded_sha256=sha(Path(r["decoded"])),
            kbps=r["file_bytes"] * 8 / (n / 50) / 1000,
        )
        runs[w, mode, q, n] = r
        return r

    def tune(w, mode, target):
        lo, hi = (36.0 if target_kbps <= 61 else 20.0), 63.0
        for q in (lo, hi):
            enc(w, mode, q)
        for _ in range(7):
            tested = sorted(q for (ww, m, q, n) in runs if (ww, m, n) == (w, mode, 240))
            best = min(
                [runs[w, mode, q, 240] for q in tested], key=lambda r: abs(r["file_bytes"] - target)
            )
            if abs(best["file_bytes"] / target - 1) <= 0.005:
                break
            pairs = [
                (a, b)
                for a, b in zip(tested, tested[1:])
                if (runs[w, mode, a, 240]["file_bytes"] - target)
                * (runs[w, mode, b, 240]["file_bytes"] - target)
                < 0
                and b - a >= 0.5
            ]
            if not pairs:
                break
            a, b = pairs[0]
            enc(w, mode, round((a + b) * 2) / 4)
            if cpu() - start > 440:
                raise RuntimeError("CPU size-search cap")
        candidates = [r for (ww, m, q, n), r in runs.items() if (ww, m, n) == (w, mode, 240)]
        b = min(candidates, key=lambda r: abs(r["file_bytes"] - target))
        if abs(b["file_bytes"] / target - 1) <= 0.005:
            return [b]
        return [
            min(items, key=lambda r: abs(r["file_bytes"] - target))
            for items in (
                [r for r in candidates if r["file_bytes"] <= target],
                [r for r in candidates if r["file_bytes"] > target],
            )
            if items
        ]

    smoke = enc(width, "plain", 60, 24)
    assert (smoke["encode_cpu_seconds"] + smoke["decode_cpu_seconds"]) * 15 + 15 < 80
    selected = min(
        tune(width, "plain", target_base_bytes),
        key=lambda r: abs(r["file_bytes"] - target_base_bytes),
    )
    native = yuv(selected["decoded"], width, height)
    full = yuv(src, 960, 544)
    references = np.memmap(out / "fit-source.rgb", "uint8", "w+", shape=(240, 544, 960, 3))
    base = np.memmap(out / "fit-base.rgb", "uint8", "w+", shape=references.shape)
    for t in range(240):
        references[t] = yuv_to_rgb(full[t], 960, 544, False)
        base[t] = cv2.resize(
            yuv_to_rgb(native[t], width, height, False), (960, 544), interpolation=cv2.INTER_CUBIC
        )

    def fit(start, end):
        tr = track[start:end]
        group = ball.FrameCrops.build(tr, references[start:end], [base[start:end]])
        (result,) = ball.fit_frames(group, ball.grid("streak"), (544, 960))
        return result["track"], result["appearances"]

    t0 = cpu()
    a, b = fit(64, 88)
    assert len(a) == len(b) == 24 and any(p is not None for p in a)
    smoke_secs = cpu() - t0
    forecast = (
        smoke_secs * 195 / sum(p is not None for p in track[64:88]) * 1.5 + cpu() - start + 100
    )
    assert forecast < 550, forecast
    locations, apps = fit(0, 240)
    symbols = ball.encode_frames(locations, apps, 960, 544)
    packet = ball_stream.pack(
        ball_stream.Stream(
            960,
            544,
            Fraction(50),
            240,
            "svt-av1-ivf",
            f"svt4-dav1d-bt709-limited-bicubic{width}x{height}-to960x544-v1",
            Path(selected["stream"]).read_bytes(),
            symbols,
        )
    )
    (out / "stream.psbl").write_bytes(packet)
    decoded = ball_stream.unpack((out / "stream.psbl").read_bytes())
    paid, looks = ball.decode_frames(decoded.ball_payload, 240, 960, 544)
    assert paid == locations and looks == apps
    (out / "independent.ivf").write_bytes(decoded.native)
    subprocess.run(
        svtav1.decode_command(
            tools["dav1d"]["path"], out / "independent.ivf", out / "independent.yuv", threads=1
        ),
        check=True,
        capture_output=True,
        timeout=45,
    )
    assert sha(out / "independent.yuv") == selected["decoded_sha256"]
    recovered = yuv(out / "independent.yuv", width, height)
    vel = ball.velocities(paid)
    tickhash = {}
    for t in range(240):
        rgb = cv2.resize(
            yuv_to_rgb(recovered[t], width, height, False),
            (960, 544),
            interpolation=cv2.INTER_CUBIC,
        )
        shown = ball.render(rgb, t, paid, vel, looks)
        assert np.array_equal(
            shown, ball.render(base[t], t, locations, ball.velocities(locations), apps)
        )
        tickhash[t] = hashlib.sha256(shown.tobytes()).hexdigest()
    anchors = {}
    for w, m in [(160, "plain"), (240, "plain"), (240, "roi")]:
        for r in tune(w, m, len(packet)):
            anchors[f"{m}{w}Q{r['q']:g}"] = r
    labels = json.loads(LABELS.read_text())["frames"]
    assert sha(LABELS) == "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"
    frames = SOURCE / "ps-development-source-refs-20261118-native/source-only-review.zip"
    assert sha(frames) == "5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb"
    mse = {}
    errors = []
    with zipfile.ZipFile(frames) as original, zipfile.ZipFile(out / "crops.zip", "x") as z:
        for f in labels:
            t = f["tick"]
            blob = original.read(f"frame-{t:03d}/source.png")
            assert hashlib.sha256(blob).hexdigest() == f["source_png_sha256"]
            src_rgb = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
            shown = ball.render(base[t], t, paid, vel, looks)
            assert hashlib.sha256(shown.tobytes()).hexdigest() == tickhash[t]
            imgs = dict(
                source=src_rgb,
                pointstream=cv2.resize(shown, (1920, 1080), interpolation=cv2.INTER_CUBIC),
            )
            for name, r in anchors.items():
                w, h = r["width"], r["height"]
                imgs[name] = cv2.resize(
                    yuv_to_rgb(yuv(r["decoded"], w, h)[t], w, h, False),
                    (1920, 1080),
                    interpolation=cv2.INTER_CUBIC,
                )
            for name, img in imgs.items():
                if name != "source":
                    mse.setdefault(name, []).append(
                        float(np.mean((src_rgb.astype(np.float32) - img) ** 2, dtype=np.float64))
                    )
            for tag, y, x, mask in supports(f):
                for name, arr in {
                    "mask": mask,
                    **{k: v[y, x, ::-1] for k, v in imgs.items()},
                }.items():
                    ok, data = cv2.imencode(".png", arr)
                    assert ok
                    z.writestr(f"{t:03d}/{tag}/{name}.png", data.tobytes())
            gt = f["ball"]["center_xy"]
            errors.append(
                dict(
                    tick=t,
                    visible=f["ball"]["status"],
                    error_px=float(
                        np.hypot(2 * paid[t][0] - gt[0], 1080 / 544 * paid[t][1] - gt[1])
                    )
                    if paid[t]
                    else None,
                )
            )
    assert cpu() - start < 550
    result = dict(
        target_kbps=target_kbps,
        paid_bytes=len(packet),
        native_bytes=len(decoded.native),
        symbol_bytes=len(symbols),
        kbps=len(packet) / 600,
        sha256=sha(out / "stream.psbl"),
        paid_ticks=sum(p is not None for p in paid),
        sender_ticks=195,
        positions=errors,
        selected=selected,
        anchors=anchors,
        encodes=list(runs.values()),
        cpu_seconds=cpu() - start,
        smoke_cpu_seconds=smoke_secs,
        smoke_forecast_cpu=forecast,
        tools=tools,
        source=sha(src),
        sender=sha(trackfile),
        raw_source_commands=commands,
        crops_sha256=sha(out / "crops.zip"),
        rgb_psnr={k: float(10 * np.log10(255**2 / np.mean(v))) for k, v in mse.items()},
        decoded_frame_hashes=tickhash,
        script_sha256=sha(Path(__file__)),
        repo_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["kbps"], result["paid_ticks"], [(name, r["kbps"]) for name, r in anchors.items()])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out", type=Path)
    cli.add_argument("--target", type=float, choices=(20.0, 60.67, 122.855), default=20.0)
    args = cli.parse_args()
    main(args.out, args.target)
