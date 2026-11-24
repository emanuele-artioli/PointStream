"""Measure MIT RacketVision Badminton shuttlecock trajectory representation against SVT-AV1."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import tarfile
import time
from typing import Any

import cv2
import numpy as np
import torch

from experiments.tennis import cohort_ball_rate
from experiments.tennis.player_requalification import sha256
from src.codecs import quality, svtav1
from src.pointstream import ball, ball_stream, codec, tracked_patch

BADMINTON_TAR_SHA256 = "1442d215c8dc31787e2a60a290470f5c4072c8d5a2b594d4f853c8399da8ca30"
VGG16_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
DISPLAY_W, DISPLAY_H = 960, 544
SRC_W, SRC_H = 1920, 1080
MAX_FRAMES = 240
TARGET_KBPS = 20.4


def patch(img: np.ndarray, xy: tuple[int, int], side: int = 64) -> np.ndarray:
    radius = side // 2
    h, w = img.shape[:2]
    x, y = xy
    crop = img[max(0, y - radius) : min(h, y + radius), max(0, x - radius) : min(w, x + radius)]
    return cv2.copyMakeBorder(
        crop,
        max(0, radius - y),
        max(0, y + radius - h),
        max(0, radius - x),
        max(0, x + radius - w),
        cv2.BORDER_REPLICATE,
    )


def dists64(net: Any, a: np.ndarray, b: np.ndarray) -> float:
    ta = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float() / 255.0
    tb = torch.from_numpy(np.ascontiguousarray(b)).permute(2, 0, 1)[None].float() / 255.0
    with torch.inference_mode():
        return float(quality.masked_dists(net, tb, ta, torch.ones_like(ta[:, :1], dtype=torch.bool)).item())


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a.astype(np.float32) - b.astype(np.float32)), dtype=np.float64))


def psnr_val(val: float) -> float:
    return 100.0 if val <= 0 else min(100.0, 10.0 * math.log10(255.0**2 / val))


def extract_bytes(tf: tarfile.TarFile, member: str) -> bytes:
    stream = tf.extractfile(member)
    if stream is None:
        raise FileNotFoundError(f"missing archive member {member}")
    return stream.read()


def select_clips(tf: tarfile.TarFile, count: int) -> list[dict[str, Any]]:
    test_pairs = json.loads(extract_bytes(tf, "badminton/info/test.json"))
    eligible: list[dict[str, Any]] = []
    for match_id, rally_id in test_pairs:
        clip_id = f"{match_id}_{rally_id}"
        csv_member = f"badminton/all/{match_id}/csv/{rally_id}_ball.csv"
        mp4_member = f"badminton/videos/{clip_id}.mp4"
        rows = list(csv.DictReader(io.StringIO(extract_bytes(tf, csv_member).decode())))
        vis_240 = [r for r in rows if int(r["Frame"]) < MAX_FRAMES and r["Visibility"] == "1"]
        if len(vis_240) < 15:
            continue
        salt = hashlib.sha256(f"pointstream-badminton-v8-20261124:{clip_id}".encode()).hexdigest()
        eligible.append(
            {
                "clip_id": clip_id,
                "match_id": match_id,
                "rally_id": rally_id,
                "salt": salt,
                "csv_member": csv_member,
                "mp4_member": mp4_member,
                "rows": rows,
                "visible_240": len(vis_240),
            }
        )
    eligible.sort(key=lambda c: c["salt"])
    return eligible[:count]


def encode_search(
    width: int,
    height: int,
    raw_rgb: Path,
    fps: Fraction,
    n_frames: int,
    payload: bytes,
    target_bytes: int,
    tools: dict[str, dict[str, str]],
    work: Path,
) -> dict[str, Any]:
    tiny = work / f"src-{width}.yuv"
    subprocess.run(
        [
            tools["ffmpeg"]["path"],
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{DISPLAY_W}x{DISPLAY_H}",
            "-framerate",
            str(fps),
            "-i",
            str(raw_rgb),
            "-vf",
            f"scale={width}:{height}:flags=area:out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-f",
            "rawvideo",
            str(tiny),
        ],
        check=True,
    )
    bank: dict[int, dict[str, Any]] = {}
    paths: dict[int, Path] = {}
    res_id = f"svtav1-scaled-{width}x{height}"

    def pack(data: bytes) -> bytes:
        return ball_stream.pack(
            ball_stream.Stream(DISPLAY_W, DISPLAY_H, fps, n_frames, "svt-av1-ivf", res_id, data, payload)
        )

    def evaluate(q: int) -> dict[str, Any]:
        if q not in bank:
            path = work / f"{width}-{q}.ivf"
            svtav1.run(
                svtav1.encode_command(
                    tools["SvtAv1EncApp"]["path"],
                    tiny,
                    path,
                    width=width,
                    height=height,
                    fps=fps,
                    frames=n_frames,
                    crf=q / 4,
                    preset=4,
                    full_range=False,
                ),
                90,
            )
            data = path.read_bytes()
            codec.check_native(
                {"width": width, "height": height, "fps": [fps.numerator, fps.denominator], "frames": n_frames},
                data,
            )
            priced = pack(data)
            bank[q] = {
                "width": width,
                "height": height,
                "crf": q / 4,
                "bytes": len(data),
                "ball_bytes": len(priced),
                "native_sha256": sha256(path),
                "ball_sha256": hashlib.sha256(priced).hexdigest(),
            }
            paths[q] = path
        return bank[q]

    for field in ("bytes", "ball_bytes"):
        cohort_ball_rate.search(evaluate, field, target_bytes)

    return {"bank": bank, "paths": paths, "pack": pack}


def decode_ivf_yuv(ivf_path: Path, width: int, height: int, n_frames: int, tools: dict[str, dict[str, str]], out_yuv: Path) -> np.ndarray:
    subprocess.run(
        svtav1.decode_command(tools["dav1d"]["path"], ivf_path, out_yuv, threads=1),
        check=True,
        capture_output=True,
        timeout=90,
    )
    return np.fromfile(out_yuv, dtype=np.uint8).reshape(n_frames, height * 3 // 2, width)


def eval_clip(clip: dict[str, Any], tf: tarfile.TarFile, net: Any, tools: dict[str, dict[str, str]], scratch: Path) -> dict[str, Any]:
    clip_dir = scratch / clip["clip_id"]
    clip_dir.mkdir()
    mp4_path = clip_dir / f"{clip['clip_id']}.mp4"
    mp4_path.write_bytes(extract_bytes(tf, clip["mp4_member"]))

    probe = json.loads(
        subprocess.check_output(
            [
                tools["ffprobe"]["path"],
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,avg_frame_rate",
                "-of",
                "json",
                str(mp4_path),
            ]
        )
    )["streams"][0]
    fps = Fraction(probe["avg_frame_rate"])
    n_frames = min(MAX_FRAMES, int(probe["nb_frames"]))
    if (int(probe["width"]), int(probe["height"])) != (SRC_W, SRC_H) or n_frames < 120:
        raise ValueError(f"unexpected source geometry/frames for {clip['clip_id']}: {probe}")

    cap = cv2.VideoCapture(str(mp4_path), cv2.CAP_FFMPEG)
    raw_rgb = clip_dir / "source960.rgb"
    orig960 = np.memmap(raw_rgb, dtype=np.uint8, mode="w+", shape=(n_frames, DISPLAY_H, DISPLAY_W, 3))
    try:
        for t in range(n_frames):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"failed frame {t} of {clip['clip_id']}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            orig960[t] = cv2.resize(rgb, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
        orig960.flush()
    finally:
        cap.release()

    by_frame = {int(r["Frame"]): r for r in clip["rows"] if int(r["Frame"]) < n_frames}
    centers: list[tuple[int, int] | None] = []
    vis_ticks: list[int] = []
    for t in range(n_frames):
        row = by_frame.get(t)
        if row is not None and row["Visibility"] == "1":
            cx = max(0, min(DISPLAY_W - 1, round(float(row["X"]) * DISPLAY_W / SRC_W)))
            cy = max(0, min(DISPLAY_H - 1, round(float(row["Y"]) * DISPLAY_H / SRC_H)))
            centers.append((cx, cy))
            vis_ticks.append(t)
        else:
            centers.append(None)

    appearance = ball.Appearance((235, 235, 235), 2, 0, 3)
    looks = [appearance if c is not None else None for c in centers]
    payload = ball.encode_frames(centers, looks, DISPLAY_W, DISPLAY_H)
    dec_centers, dec_looks = ball.decode_frames(payload, n_frames, DISPLAY_W, DISPLAY_H)
    if (dec_centers, dec_looks) != (centers, looks):
        raise ValueError("PSBL roundtrip mismatch")
    velocity = ball.velocities(dec_centers)

    target_bytes = round(Fraction(int(TARGET_KBPS * 1000) * n_frames, 8) / fps)
    searches = {}
    for w, h in ((320, 180), (640, 360)):
        searches[w] = encode_search(w, h, raw_rgb, fps, n_frames, payload, target_bytes, tools, clip_dir)

    # Pick best affordable ordinary AV1 across {320, 640} and matched pair (prefer 320 same-res match, else 640).
    all_ord = [pt for res in searches.values() for pt in res["bank"].values() if pt["bytes"] <= target_bytes]
    all_ps = [pt for res in searches.values() for pt in res["bank"].values() if pt["ball_bytes"] <= target_bytes]
    if not all_ord or not all_ps:
        raise RuntimeError(f"no affordable encode under {target_bytes} bytes for {clip['clip_id']}")

    # Same-resolution matched pair first (highest resolution with valid pair, preferring <=0.5% gap)
    same_res_pairs = []
    for res in searches.values():
        pts = list(res["bank"].values())
        for ps_pt in pts:
            for ord_pt in pts:
                if ps_pt["ball_bytes"] <= target_bytes and ord_pt["bytes"] <= target_bytes:
                    same_res_pairs.append((ps_pt, ord_pt))
    ps_choice, ord_same_res = min(
        same_res_pairs, key=lambda pair: cohort_ball_rate.pair_order(pair[0], pair[1], target_bytes)
    )

    # Best affordable ordinary AV1 (largest bytes <= target_bytes, break ties by widest width)
    ord_best = max(all_ord, key=lambda pt: (pt["bytes"], pt["width"]))

    ps_path = searches[ps_choice["width"]]["paths"][int(ps_choice["crf"] * 4)]
    ord_same_path = searches[ord_same_res["width"]]["paths"][int(ord_same_res["crf"] * 4)]
    ord_best_path = searches[ord_best["width"]]["paths"][int(ord_best["crf"] * 4)]

    ps_yuv = decode_ivf_yuv(ps_path, ps_choice["width"], ps_choice["height"], n_frames, tools, clip_dir / "ps.yuv")
    ord_same_yuv = decode_ivf_yuv(ord_same_path, ord_same_res["width"], ord_same_res["height"], n_frames, tools, clip_dir / "ord_same.yuv")
    if ord_best["width"] == ord_same_res["width"] and ord_best["crf"] == ord_same_res["crf"]:
        ord_best_yuv = ord_same_yuv
    else:
        ord_best_yuv = decode_ivf_yuv(ord_best_path, ord_best["width"], ord_best["height"], n_frames, tools, clip_dir / "ord_best.yuv")

    # Serialization parity on visible frames (in source 1920x1080 pixels)
    max_quant_err_px = 0.0
    for t in vis_ticks:
        row = by_frame[t]
        sx, sy = float(row["X"]), float(row["Y"])
        dx, dy = dec_centers[t][0] * SRC_W / DISPLAY_W, dec_centers[t][1] * SRC_H / DISPLAY_H  # type: ignore[index]
        err = math.hypot(dx - sx, dy - sy)
        max_quant_err_px = max(max_quant_err_px, err)

    arm_scores: dict[str, dict[str, list[float]]] = {
        k: {"dists64": [], "mse64": [], "mse12": [], "whole_mse": []}
        for k in ("pointstream", "av1_same_res", "av1_best", "same_base")
    }
    for t in range(n_frames):
        ref = orig960[t]
        base = cv2.resize(
            tracked_patch.yuv_to_rgb(ps_yuv[t], ps_choice["width"], ps_choice["height"], False),
            (DISPLAY_W, DISPLAY_H),
            interpolation=cv2.INTER_CUBIC,
        )
        ps_img = ball.render(base, t, dec_centers, velocity, dec_looks)
        ord_same_img = cv2.resize(
            tracked_patch.yuv_to_rgb(ord_same_yuv[t], ord_same_res["width"], ord_same_res["height"], False),
            (DISPLAY_W, DISPLAY_H),
            interpolation=cv2.INTER_CUBIC,
        )
        ord_best_img = cv2.resize(
            tracked_patch.yuv_to_rgb(ord_best_yuv[t], ord_best["width"], ord_best["height"], False),
            (DISPLAY_W, DISPLAY_H),
            interpolation=cv2.INTER_CUBIC,
        )
        imgs = {
            "pointstream": ps_img,
            "same_base": base,
            "av1_same_res": ord_same_img,
            "av1_best": ord_best_img,
        }
        for k, im in imgs.items():
            arm_scores[k]["whole_mse"].append(mse(ref, im))
            c = dec_centers[t]
            if c is not None:
                p_ref = patch(ref, c, 64)
                p_im = patch(im, c, 64)
                arm_scores[k]["dists64"].append(dists64(net, p_ref, p_im))
                arm_scores[k]["mse64"].append(mse(p_ref, p_im))
                arm_scores[k]["mse12"].append(mse(p_ref[26:38, 26:38], p_im[26:38, 26:38]))

    summary_arms = {}
    for k, vals in arm_scores.items():
        summary_arms[k] = {
            "dists64": mean(vals["dists64"]),
            "psnr64_db": psnr_val(mean(vals["mse64"])),
            "psnr12_db": psnr_val(mean(vals["mse12"])),
            "whole_psnr_db": psnr_val(mean(vals["whole_mse"])),
        }

    return {
        "clip_id": clip["clip_id"],
        "fps": str(fps),
        "n_frames": n_frames,
        "visible_marks": len(vis_ticks),
        "target_bytes": target_bytes,
        "payload_bytes": len(payload),
        "max_quant_err_src_px": max_quant_err_px,
        "selected": {
            "pointstream": {
                "width": ps_choice["width"],
                "crf": ps_choice["crf"],
                "bytes": ps_choice["ball_bytes"],
                "kbps": ps_choice["ball_bytes"] * 8 * float(fps) / n_frames / 1000.0,
            },
            "av1_same_res": {
                "width": ord_same_res["width"],
                "crf": ord_same_res["crf"],
                "bytes": ord_same_res["bytes"],
                "kbps": ord_same_res["bytes"] * 8 * float(fps) / n_frames / 1000.0,
            },
            "av1_best": {
                "width": ord_best["width"],
                "crf": ord_best["crf"],
                "bytes": ord_best["bytes"],
                "kbps": ord_best["bytes"] * 8 * float(fps) / n_frames / 1000.0,
            },
        },
        "scores": summary_arms,
    }


def run_eval(args: argparse.Namespace) -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    stage = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    expected = {"smoke": 1, "full": 6}[os.environ["PS_STAGE"]]
    if args.clips != expected:
        raise ValueError(f"clips={args.clips} mismatches stage {os.environ['PS_STAGE']} ({expected})")
    if sha256(args.badminton_tar) != BADMINTON_TAR_SHA256 or sha256(args.vgg) != VGG16_SHA256:
        raise ValueError("input sha256 mismatch")

    tools = {k: svtav1.tool(k) for k in ("ffmpeg", "ffprobe", "dav1d", "SvtAv1EncApp")}
    net = quality.load_dists(args.vgg, "cpu")
    t0 = time.monotonic()
    with tarfile.open(args.badminton_tar) as tf:
        clips = select_clips(tf, args.clips)
        if len(clips) != args.clips:
            raise RuntimeError(f"only found {len(clips)} eligible clips")
        results = [eval_clip(c, tf, net, tools, scratch) for c in clips]

    wins_same = sum(
        1 for r in results if r["scores"]["pointstream"]["dists64"] < r["scores"]["av1_same_res"]["dists64"]
    )
    wins_best = sum(
        1 for r in results if r["scores"]["pointstream"]["dists64"] < r["scores"]["av1_best"]["dists64"]
    )
    out = {
        "clips_evaluated": len(results),
        "target_kbps": TARGET_KBPS,
        "badminton_tar_sha256": BADMINTON_TAR_SHA256,
        "vgg16_sha256": VGG16_SHA256,
        "tools": tools,
        "elapsed_s": time.monotonic() - t0,
        "pooled": {
            "wins_vs_av1_same_res": wins_same,
            "wins_vs_av1_best": wins_best,
            "pointstream_dists64": mean(r["scores"]["pointstream"]["dists64"] for r in results),
            "av1_same_res_dists64": mean(r["scores"]["av1_same_res"]["dists64"] for r in results),
            "av1_best_dists64": mean(r["scores"]["av1_best"]["dists64"] for r in results),
            "same_base_dists64": mean(r["scores"]["same_base"]["dists64"] for r in results),
            "pointstream_kbps": mean(r["selected"]["pointstream"]["kbps"] for r in results),
            "av1_same_res_kbps": mean(r["selected"]["av1_same_res"]["kbps"] for r in results),
            "av1_best_kbps": mean(r["selected"]["av1_best"]["kbps"] for r in results),
        },
        "clips": results,
    }
    (stage / "badminton_oracle.json").write_text(json.dumps(out, indent=2) + "\n")


def validate(args: argparse.Namespace) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    expected = {"smoke": 1, "full": 6}[os.environ["PS_STAGE"]]
    data = json.loads((stage / "badminton_oracle.json").read_text())
    assert data["clips_evaluated"] == expected
    for r in data["clips"]:
        assert 120 <= r["n_frames"] <= MAX_FRAMES and r["visible_marks"] >= 15
        assert r["max_quant_err_src_px"] < 2.0
        for arm in ("pointstream", "av1_same_res", "av1_best", "same_base"):
            assert math.isfinite(r["scores"][arm]["dists64"])
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {
                    "clips_evaluated": expected,
                    "psbl_roundtrip": True,
                    "finite_dists64": True,
                },
                "pooled": data["pooled"],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--badminton-tar", type=Path, required=True)
    p.add_argument("--vgg", type=Path, required=True)
    p.add_argument("--clips", type=int, choices=(1, 6), required=True)
    p.add_argument("--validate", action="store_true")
    if sys.argv[1:] == ["--validate"]:
        stage = Path(os.environ["PS_STAGE_DIR"])
        argv = json.loads((stage / "dispatch.json").read_text())["command"]
        validate(p.parse_args(["--validate", *argv[3:]]))
    else:
        args = p.parse_args()
        if args.validate:
            validate(args)
        else:
            run_eval(args)


if __name__ == "__main__":
    main()
