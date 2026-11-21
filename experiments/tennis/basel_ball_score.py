"""Choose complete packet rates before opening ball-crop scores on the Basel match."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.tennis import ball_confirmation as prior
from experiments.tennis import ball_starved_scaled as tiles
from experiments.tennis import basel_ball_sender as sender
from experiments.tennis import k5d_encoder, w1_roi
from src.codecs import quality, svtav1
from src.pointstream import tracked_patch

CAP = 60.6
WIDTHS = (320, 640, 1280)


def roi_text(track, w, h):
    masks = np.zeros((len(track), h, w), bool)
    scale = 1280 // w
    radius = 32 // scale
    for t, p in enumerate(track):
        if p is not None:
            x, y = p[0] // scale, p[1] // scale
            masks[
                t, max(0, y - radius) : min(h, y + radius), max(0, x - radius) : min(w, x + radius)
            ] = True
    rows = w1_roi.roi_text(masks, w, h, -40, outside=True).splitlines()
    for t, p in enumerate(track):
        if p is None:
            rows[t] = " ".join([str(t)] + ["0"] * (((h + 63) // 64) * ((w + 63) // 64)))
    return "\n".join(rows) + "\n"


def wire(payload):
    packed = zlib.compress(payload, 6)
    return b"\x01" + packed if len(packed) < len(payload) else b"\x00" + payload


def label_rows(path, n):
    if prior.sha(path) != sender.LABELS:
        raise ValueError("Wrong official labels")
    labels = [(-1, 0, 0)] * n
    with path.open() as f:
        for row in csv.DictReader(f):
            tick, kind, x, y = (int(row[k]) for k in ("Frame", "Visibility", "X", "Y"))
            if tick < n:
                if tick < 0 or labels[tick][0] >= 0 or kind not in (0, 1):
                    raise ValueError("Invalid sparse source label")
                labels[tick] = kind, round(2 * x / 3), round(2 * y / 3)
    return labels


def run(a):
    start = time.monotonic()
    n = a.frames
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "basel-score"
    publish = work.parent / "publish"
    work.mkdir()
    publish.mkdir()
    if len(os.sched_getaffinity(0)) != 1 or n not in (16, 64):
        raise ValueError("Wrong CPU or stage")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    source = sender.source_frames(a.source, n)
    rec = json.loads(a.sender.read_text())
    if (
        rec["source_sha256"] != sender.SOURCE
        or rec["frames"] != 64
        or (n == 64 and rec["rgb_sha256"] != hashlib.sha256(source).hexdigest())
    ):
        raise ValueError("Sender/source mismatch")
    track = prior.checked_track(rec["positions"][:n], n)
    tools = {k: svtav1.tool(k) for k in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    if any(tool["sha256"] != rec["tools"][k]["sha256"] for k, tool in tools.items()):
        raise ValueError("Native codec identity changed")
    yuv = work / "source.yuv"
    k5d_encoder.convert_rgb(tools["ffmpeg"]["path"], source, yuv, sender.FPS)

    def code(path, tag, w, h, q, extra=None):
        dest = work / tag
        dest.mkdir()
        return svtav1.code(
            path,
            dest,
            width=w,
            height=h,
            fps=sender.FPS,
            frames=n,
            crf=q,
            preset=4,
            threads=1,
            full_range=False,
            encoder=tools["SvtAv1EncApp"]["path"],
            decoder=tools["dav1d"]["path"],
            timeout=350,
            extra=extra,
        )

    patches, placements = prior.patch_frames(yuv, track, n)
    patch_yuv = work / "patch.yuv"
    patch_yuv.write_bytes(patches.tobytes())
    patch = Path(code(patch_yuv, "patch", 64, 64, 20)["stream"]).read_bytes()
    origins = [[x["left"], x["top"]] if x["present"] else None for x in placements]
    meta = {
        "native": {
            "version": 1,
            "profile": "native-av1",
            "width": 320,
            "height": 180,
            "fps": [25, 1],
            "frames": n,
            "full_range": False,
            "decoder_sha256": tools["dav1d"]["sha256"],
        },
        "canvas": [1280, 720],
        "positions": origins,
    }
    trials, selected = [], []
    for w in WIDTHS:
        h = w * 9 // 16
        small = yuv
        if w != 1280:
            small = work / f"source-{w}.yuv"
            subprocess.run(
                [
                    tools["ffmpeg"]["path"],
                    "-v",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "yuv420p",
                    "-video_size",
                    "1280x720",
                    "-framerate",
                    "25",
                    "-i",
                    str(yuv),
                    "-vf",
                    f"scale={w}:{h}:flags=area,format=yuv420p",
                    "-frames:v",
                    str(n),
                    "-f",
                    "rawvideo",
                    str(small),
                ],
                check=True,
                timeout=150,
            )
        roi = work / f"roi-{w}.txt"
        roi.write_text(roi_text(track, w, h))
        native_cache = {}
        for mode in ("plain", "roi", "tile") if w == 320 else ("plain", "roi"):

            def probe(q):
                key = ("roi" if mode == "roi" else "plain", q)
                if key not in native_cache:
                    name = f"{w}-{key[0]}-{q}"
                    native_cache[key] = code(
                        small,
                        name,
                        w,
                        h,
                        q,
                        ["--roi-map-file", str(roi)] if key[0] == "roi" else None,
                    )
                run = native_cache[key]
                raw = Path(run["stream"]).read_bytes()
                if mode == "tile":
                    raw = tiles.pack_scaled(meta, raw, patch)
                packet = wire(raw)
                return dict(
                    width=w,
                    mode=mode,
                    crf=q,
                    bytes=len(packet),
                    kbps=len(packet) * 0.2 / n,
                    sha256=hashlib.sha256(packet).hexdigest(),
                    decoded=run["decoded"],
                    packet=packet,
                )

            points = {}
            for q in (30, 70):
                points[q] = probe(q)
            if points[70]["kbps"] <= CAP < points[30]["kbps"]:
                lo, hi = 30, 70
                for _ in range(3 if n == 16 else 6):
                    q = round((lo + hi) * 2) / 4
                    if q in points:
                        break
                    points[q] = probe(q)
                    if points[q]["kbps"] > CAP:
                        lo = q
                    else:
                        hi = q
            trials.extend(
                {k: v for k, v in r.items() if k not in ("packet", "decoded")}
                for r in points.values()
            )
            under = [r for r in points.values() if r["kbps"] <= CAP]
            over = [r for r in points.values() if r["kbps"] > CAP]
            if under:
                selected.append(max(under, key=lambda x: x["kbps"]))
            if over:
                selected.append(min(over, key=lambda x: x["kbps"]))

    # Decisions above use bytes only. Official labels first enter here.
    labels = label_rows(a.labels, n)
    net = quality.load_dists(a.weights, "cpu")
    self_scores = tiles.score_frame(net, source, source, labels)
    if any(abs(x["dists"]) > 1e-5 for x in self_scores):
        raise ValueError("Invalid source self score")
    rows = []
    for index, row in enumerate(selected):
        packet = row.pop("packet")
        unpack = "import sys,zlib; d=sys.stdin.buffer.read(); sys.stdout.buffer.write(zlib.decompress(d[1:]) if d[0] else d[1:])"
        payload = subprocess.run(
            [sys.executable, "-c", unpack], input=packet, capture_output=True, check=True
        ).stdout
        render = work / f"receive-{index}"
        render.mkdir()
        if row["mode"] == "tile":
            rgb = tiles.render_scaled(payload, render, tools["dav1d"]["path"])
        else:
            w = row["width"]
            h = w * 9 // 16
            out = render / "frames.yuv"
            native = dict(meta["native"], width=w, height=h)
            tracked_patch.decode_native(payload, out, native, tools["dav1d"]["path"])
            if prior.sha(out) != prior.sha(Path(row["decoded"])):
                raise ValueError("Separate receiver differs from native decoder")
            raw = np.memmap(out, np.uint8, "r", shape=(n, h * 3 // 2, w))
            rgb = np.stack(
                [
                    cv2.resize(
                        tracked_patch.yuv_to_rgb(f, w, h, False),
                        sender.SIZE,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    for f in raw
                ]
            )
        del row["decoded"]
        row["file"] = f"{index}.packet"
        (publish / row["file"]).write_bytes(packet)
        row["cropped"] = tiles.score_frame(net, source, rgb, labels)
        row["dists"] = float(np.mean([v["dists"] for v in row["cropped"]]))
        row["rgb_sha256"] = hashlib.sha256(rgb).hexdigest()
        row["whole_rgb_psnr"] = k5d_encoder.pooled_psnr(source, rgb)
        rows.append(row)
    result = dict(
        source_sha256=sender.SOURCE,
        labels_sha256=sender.LABELS,
        label_rows=labels,
        sender_sha256=prior.sha(a.sender),
        frames=n,
        cap=CAP,
        rows=rows,
        trials=trials,
        source_self=self_scores,
        tools=tools,
        metric=quality.dists_record(a.weights),
        seconds=time.monotonic() - start,
    )
    (publish / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "result.json").read_text())
    n = 16 if os.environ["PS_STAGE"] == "smoke" else 64
    checks = dict(
        source=r["source_sha256"] == sender.SOURCE,
        labels=r["labels_sha256"] == sender.LABELS,
        count=r["frames"] == len(r["label_rows"]) == n,
        visible=len(r["source_self"]) > 0,
        modes={(x["width"], x["mode"]) for x in r["rows"]}
        == {(w, m) for w in WIDTHS for m in ("plain", "roi")} | {(320, "tile")},
    )
    with tarfile.open(stage / "published.tar") as archive:
        for row in r["rows"]:
            packet = archive.extractfile("publish/" + row["file"]).read()
            checks[row["file"]] = bool(
                len(packet) == row["bytes"]
                and hashlib.sha256(packet).hexdigest() == row["sha256"]
                and abs(len(packet) * 0.2 / n - row["kbps"]) < 1e-9
                and np.isfinite(row["dists"])
                and len(row["cropped"]) == len(r["source_self"])
            )
    forecast = 1.5 * r["seconds"] * 4 * 1.6 + 85 if n == 16 else None
    checks["budget"] = forecast is None or forecast <= 1550
    decision = dict(passed=all(checks.values()), checks=checks, forecast_seconds=forecast)
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(decision) + "\n")
    if not decision["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for field in ("source", "sender", "labels", "weights"):
        p.add_argument("--" + field, type=Path)
    p.add_argument("--frames", type=int, choices=(16, 64))
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
