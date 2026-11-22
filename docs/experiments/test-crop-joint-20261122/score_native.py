"""Measure both players on every native Shanghai tick, holding original support fixed."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time

import cv2
import numpy as np
import torch

from experiments.tennis import common_player_packet as packet
from experiments.tennis.k4b_pose import player_masks
from src.codecs import quality, svtav1
from src.pointstream.tracked_patch import yuv_to_rgb
from src.segmentation.masks import ClipMasks

HERE = Path(__file__).resolve().parent
WEIGHT = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
FPS = Fraction(30000, 1001)
NT = 240
REVIEW = {9, 31, 49, 72, 95, 112, 135, 152, 175, 198, 215, 238}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def tensor(arr):
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].float() / 255


def get_support(features):
    crop = json.loads((HERE / "match102-result.json").read_text())
    assert sha(features) == crop["prior_feature_archive_sha256"]
    prefix = "publish/test__match102@120/"
    with tarfile.open(features) as archive:
        info = json.load(archive.extractfile(prefix + "features.json"))
        rle = archive.extractfile(prefix + "sam-masks.rle").read()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sam.rle"
        path.write_bytes(rle)
        clip = ClipMasks.load(path)
    tracks = crop["selected_ids"]
    timeline = []
    for tick in range(NT):
        people = player_masks(clip.frames[tick], info["aliases"], tracks)
        entries = []
        for track in tracks:
            mask = np.repeat(np.repeat(people[track], 2, axis=0), 2, axis=1)
            assert mask.shape == (1080, 1920) and mask.any()
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            x0, y0 = max(0, x - 32), max(0, y - 32)
            x1, y1 = min(1920, x + w + 32), min(1080, y + h + 32)
            roi = np.ascontiguousarray(mask[y0:y1, x0:x1] > 0)
            entries.append(((x0, y0, x1, y1), roi))
        timeline.append(entries)
    return timeline, hashlib.sha256(rle).hexdigest()


def control_catalog(a):
    bank = HERE.parent / "cohort-expansion-20261122/test-physical"
    r = json.loads((bank / "match102-rate.json").read_text())
    assert (
        r["source_sha256"] == sha(a.source)
        and sha(a.av1) == "37d21fdac95e0c6818be4f335eff1da308f51ab81344ec250ec6a048d56587d5"
    )
    rows = []
    for d in r["rows"]:
        rows.append(
            dict(
                name="av1-" + d["native_file"].split(".")[0],
                width=d["width"],
                height=d["height"],
                bytes=d["bytes"],
                kind="ivf_tar",
                archive=str(a.av1),
                member="publish/" + d["native_file"],
                sha256=d["native_sha256"],
            )
        )
    assert sha(a.neural) == "bacd5dacbe6a33959d7afe15b915ecbe28da82e3dfcd2fb1c3cc12e822be6370"
    n = json.loads((bank / "match102-neural.json").read_text())
    assert n["source_sha256"] == r["source_sha256"] and len(n["yuv_frames"]) == 240
    rows.insert(
        0,
        dict(
            name="dcvc-480-q24",
            width=480,
            height=272,
            bytes=n["native"]["bytes"],
            kind="yuv_tar",
            archive=str(a.neural),
            member="publish/decoded.yuv",
            frame_sha256=n["yuv_frames"],
            stream_sha256=n["native"]["container_sha256"],
        ),
    )
    return rows


def prep(a, arm, work):
    w, h = arm["width"], arm["height"]
    if arm["kind"] == "rgb_npy":
        path = Path(arm["path"])
        assert sha(path) == arm["sha256"]
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        assert data.shape == (240, 1080, 1920, 3) and data.dtype == np.uint8
        return data, arm["sha256"]
    if arm["kind"] == "ivf":
        path = Path(arm["path"])
        data = path.read_bytes()
    elif arm["kind"] in ("ivf_tar", "yuv_tar"):
        with tarfile.open(arm["archive"]) as z:
            data = z.extractfile(arm["member"]).read()
            if arm["kind"] == "yuv_tar":
                stream = z.extractfile("publish/stream.psdc").read()
                assert (
                    len(stream) == arm["bytes"]
                    and hashlib.sha256(stream).hexdigest() == arm["stream_sha256"]
                )
    elif arm["kind"] == "yuv":
        path = Path(arm["path"])
        assert path.stat().st_size == NT * w * h * 3 // 2
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == arm["sha256"]
    else:
        raise ValueError("unknown decoder route")
    path = work / "decoded.yuv"
    if arm["kind"] in ("ivf", "ivf_tar"):
        assert len(data) == arm["bytes"] and hashlib.sha256(data).hexdigest() == arm["sha256"]
        packet._ivf(data, w, h, (30000, 1001))
        ivf = work / "source.ivf"
        ivf.write_bytes(data)
        subprocess.run(
            svtav1.decode_command(svtav1.tool("dav1d")["path"], ivf, path, threads=1),
            check=True,
            capture_output=True,
            timeout=120,
        )
    else:
        path.write_bytes(data)
    assert path.stat().st_size == NT * w * h * 3 // 2
    native = np.memmap(path, np.uint8, "r", shape=(NT, h * 3 // 2, w))
    if "frame_sha256" in arm:
        assert [hashlib.sha256(frame.tobytes()).hexdigest() for frame in native] == arm[
            "frame_sha256"
        ]
    return native, sha(path)


def score(a, arm, model, support, support_hash):
    before = time.monotonic()
    cpu0 = time.process_time()
    w, h = arm["width"], arm["height"]
    steps = []
    sse = 0.0
    self_abs = 0.0
    with tempfile.TemporaryDirectory(dir=a.scratch) as tmp:
        display, decoded_hash = prep(a, arm, Path(tmp))
        src = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
        assert abs(src.get(cv2.CAP_PROP_FPS) - float(FPS)) < 0.001
        for t in range(NT):
            ok, picture = src.read()
            assert ok
            rgb = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
            im = (
                display[t]
                if arm["kind"] == "rgb_npy"
                else cv2.resize(
                    yuv_to_rgb(display[t], w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
                )
            )
            sse += cv2.norm(rgb, im, cv2.NORM_L2SQR)
            for role, ((x0, y0, x1, y1), m) in zip(("near", "far"), support[t], strict=True):
                ref = rgb[y0:y1, x0:x1]
                ob = im[y0:y1, x0:x1]
                active = torch.from_numpy(m.copy())[None, None]
                with torch.inference_mode():
                    tr = tensor(ref)
                    val = float(quality.masked_dists(model, tensor(ob), tr, active))
                    if not steps:
                        self_abs = abs(float(quality.masked_dists(model, tr, tr, active)))
                        assert self_abs < 1e-5
                assert math.isfinite(val)
                diff = ob[m].astype(np.float32) - ref[m].astype(np.float32)
                steps.append(
                    dict(
                        tick=t,
                        role=role,
                        body_dists=val,
                        area=int(m.sum()),
                        rgb_squared_error=float((diff * diff).sum()),
                    )
                )
            if t == 5:
                forecast = (time.process_time() - cpu0) * 240 / 6 * 1.5
                if forecast > 1700:
                    raise ValueError(f"CPU stage forecast too high: {forecast}")
        assert not src.read()[0]
        src.release()

    def group(rows):
        return dict(
            dists=float(np.mean([r["body_dists"] for r in rows])),
            rgb_psnr=10
            * math.log10(
                255**2
                * 3
                * sum(r["area"] for r in rows)
                / sum(r["rgb_squared_error"] for r in rows)
            ),
            ticks=len(rows),
        )

    out = dict(
        name=arm["name"],
        source_sha256=sha(a.source),
        support_rle_sha256=support_hash,
        support_method="Original frozen SAM31 source supports, per-player DISTS on tight bounds plus 32 pixels; 2x replication of half-resolution binary masks.",
        native_fps=str(FPS),
        frames=240,
        bytes=arm["bytes"],
        kbps=arm["bytes"] * float(FPS) / 30000,
        width=w,
        height=h,
        kind=arm["kind"],
        sha256=arm.get("sha256", arm.get("stream_sha256")),
        decoded_sha256=decoded_hash,
        pooled=group(steps),
        near=group([r for r in steps if r["role"] == "near"]),
        far=group([r for r in steps if r["role"] == "far"]),
        optional_review_ticks_on_source_model=group([r for r in steps if r["tick"] in REVIEW]),
        whole_rgb_psnr=10 * math.log10(255**2 * NT * 1080 * 1920 * 3 / sse),
        source_self_abs=self_abs,
        cpu_seconds=time.process_time() - cpu0,
        benchmark_forecast=forecast,
        elapsed_s=time.monotonic() - before,
        tool={
            "cv2": cv2.__version__,
            "torch": torch.__version__,
            "weight_sha256": WEIGHT,
            "device": "cpu",
        },
        rows=steps,
    )
    return out


def main(a):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    assert (
        sha(a.weights) == WEIGHT
        and sha(a.source)
        == json.loads((HERE / "match102-result.json").read_text())["source_sha256"]
    )
    supp, hs = get_support(a.features)
    model = quality.load_dists(a.weights, "cpu")
    rows = json.loads(a.arms.read_text()) if a.arms else control_catalog(a)
    if a.limit == "warm":
        rows = [r for r in rows if r["bytes"] <= 17794]
    elif a.limit == "cold":
        rows = [r for r in rows if 17794 < r["bytes"] <= 155274]
    a.output.mkdir(exist_ok=True, parents=True)
    assert len({r["name"] for r in rows}) == len(rows)
    for row in rows:
        dest = a.output / (row["name"] + ".json")
        if dest.exists():
            saved = json.loads(dest.read_text())
            assert (
                saved["name"] == row["name"]
                and saved["bytes"] == row["bytes"]
                and saved["support_rle_sha256"] == hs
                and len(saved["rows"]) == 480
            )
            continue
        out = score(a, row, model, supp, hs)
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, separators=(",", ":")) + "\n")
        tmp.rename(dest)
        print(
            row["name"],
            round(out["pooled"]["dists"], 5),
            round(out["whole_rgb_psnr"], 3),
            round(out["cpu_seconds"], 1),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "features", "weights", "scratch", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("arms", "av1", "neural"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--limit", choices=["warm", "cold", "all"], default="all")
    main(p.parse_args())
