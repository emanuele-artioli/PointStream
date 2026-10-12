"""PLAN step G6, steps 2-3: the background at H5's format (1080p, native frame rate), on one window.

    python -m experiments.background.g6 format --eval-set E.JSON --archive DIR --masks DIR --video P.MP4 \\
        --item ID --arm DIR --image-ckpt P --video-ckpt P --lpips-backbone P --dists-backbone P --inception P \\
        --propainter-tree DIR --propainter-ckpt P --raft-ckpt P --flow-completion-ckpt P \\
        --methods svt-1080,scene-540 --limit-frames N
    python -m experiments.background.g6 validate

G5c's evidence is at 960×540 and 10 frames/s; B2's harness, which scores H5,
codes evaluation set v2's windows at 1080p and their native rate (50 or 59.94
frames/s). On one window this codes the background as H5 would: the window
decoded as B2 does (`experiments.visor.b2.decode_window`), the foreground of
B1b's mask set (`visor_dense_sam_fill`) dilated as G1 does at 1080p and filled
by ProPainter at 960×540 (upsampled into the holes only; the fill is coded,
never scored), then the filled frames coded by each method:

* ``svt-1080`` / ``svt-540``: SVT-AV1 (G5's preset and CRFs) at 1080p, or at
  960×540 and upsampled (bicubic) at the client;
* ``dcvc-1080``: DCVC-UF HT-L off the shelf at 1080p;
* ``scene-1080`` / ``scene-540``: the stretch's scene model (``--arm``, a
  G5c finetune archive trained at 960×540 and 10 frames/s) at 1080p as
  trained, or at 960×540 and upsampled.

Scored at 1080p on the background (not B1b's foreground, as B2's ``bg``), with
`g5c.full_score` (PSNR, LPIPS, DISTS, flicker and KID on V). Each finished
point is checkpointed. Writes ``g6.json`` in ``PS_STAGE_DIR``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background import g5, g5c
from experiments.background.camera import DILATE, TO_1080
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

METHODS = ("svt-1080", "svt-540", "dcvc-1080", "scene-1080", "scene-540")
FULL = (1920, 1080)
HALF = (g5.WIDTH, g5.HEIGHT)
MASK_SET = "visor_dense_sam_fill"


def parse_method(name: str) -> tuple[str, tuple[int, int]]:
    """(codec, coded size): ``svt``, ``dcvc`` or ``scene``, at 1080p or 960×540."""
    codec, _, size = name.partition("-")
    if name not in METHODS:
        raise ValueError(f"method {name!r}: one of {METHODS}")
    return codec, FULL if size == "1080" else HALF


def resize(frames: np.ndarray, size: tuple[int, int], *, area: bool) -> np.ndarray:
    """(n, h, w, c) uint8 or (n, h, w) bool to ``size`` (w, h): area average down, bicubic up; masks by any."""
    import cv2

    if frames.dtype == bool:
        out = np.stack([cv2.resize(f.astype(np.float32), size, interpolation=cv2.INTER_AREA) for f in frames])
        return out > 0
    flag = cv2.INTER_AREA if area else cv2.INTER_CUBIC
    return np.stack([cv2.resize(f, size, interpolation=flag) for f in frames])


def dilate_1080(mask: np.ndarray) -> np.ndarray:
    """G1's foreground dilation at 1080p (``DILATE`` analysis pixels, 16 at 1080p)."""
    import cv2

    size = 2 * round(DILATE * TO_1080) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return np.stack([cv2.dilate(m.astype(np.uint8), kernel) > 0 for m in mask])


def load_window(args: argparse.Namespace, work: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    """The item, its RGB frames at 1080p (B2's decode, in the source's range) and B1b's foreground."""
    import torch

    from experiments.visor import b2
    from src.codecs import quality

    items = {i["id"]: i for i in json.loads(Path(args.eval_set).read_text())["items"]}
    item = items[args.item]
    frames = args.limit_frames if args.limit_frames > 0 else int(item["frames"])
    record = b2.decode_window(Path(args.video), item, frames, Path(args.archive), work / "window.yuv", 8)
    if (record["width"], record["height"]) != FULL:
        raise SystemExit(f"{args.item}: {record['width']}x{record['height']}, not 1080p")
    planes = np.fromfile(work / "window.yuv", np.uint8).reshape(frames, FULL[1] * 3 // 2, FULL[0])
    rgb = np.concatenate([quality.yuv420_to_rgb(torch.from_numpy(planes[i:i + 16]), FULL[0], FULL[1],
                                                full_range=record["full_range"]).permute(0, 2, 3, 1)
                          .to(torch.uint8).numpy() for i in range(0, frames, 16)])
    (work / "window.yuv").unlink()
    clips, info = b2.load_mask_sets(item, frames, {MASK_SET: args.masks}, {}, Path(args.archive))
    fg = np.stack([clips[MASK_SET].foreground(i) for i in range(frames)])
    if fg.shape[1:] != (FULL[1], FULL[0]):
        raise SystemExit(f"{args.item}: masks {fg.shape[1:]}, not 1080p")
    return item, rgb, fg, {"window": {k: v for k, v in record.items() if k != "jpeg_gate"},
                           "jpeg_gate_holds": all(c["holds"] for c in record["jpeg_gate"]), "masks": info[MASK_SET]}


def fill(rgb: np.ndarray, holes: np.ndarray, args: argparse.Namespace, work: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """ProPainter's fill at 960×540, upsampled into the 1080p holes only."""
    small, small_holes = resize(rgb, HALF, area=True), resize(holes, HALF, area=True)
    filled, record = g5c.run_propainter(small, small_holes, [[0, len(rgb), 0, len(rgb)]], args, work / "propainter")
    out = np.where(holes[..., None], resize(filled, FULL, area=False), rgb)
    return out, record


def code_point(codec: str, size: tuple[int, int], value: int, filled: np.ndarray, clip: g5.Clip,
               args: argparse.Namespace, work: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """One rate point: the filled frames coded at ``size``, decoded and brought back to 1080p."""
    from src.codecs import svtav1

    work.mkdir(parents=True, exist_ok=True)
    source = work / "source.yuv"
    frames = filled if size == FULL else resize(filled, size, area=True)
    g5.rgb_to_yuv420(frames).tofile(source)
    n = len(frames)
    if codec == "svt":
        rec = svtav1.code(source, work, width=size[0], height=size[1], fps=clip.fps, frames=n, crf=value,
                          preset=g5.PRESET, threads=8, full_range=True, encoder="SvtAv1EncApp", decoder="dav1d")
        nbytes, extra = int(rec["payload_bytes"]), {"stream_sha256": rec["stream_sha256"]}
        decoded_path = Path(rec["decoded"])
    else:
        ckpt = args.video_ckpt if codec == "dcvc" else str(Path(args.arm) / "publish" / "finetune" / "video_ft.pth.tar")
        coder = argparse.Namespace(image_ckpt=args.image_ckpt, video_ckpt=ckpt)
        rec = g5.code_dcvc(coder, source, n, value, work / "dcvc", width=size[0], height=size[1])
        nbytes = int(rec["bytes"])
        extra = {k: rec[k] for k in ("deterministic", "decoder_matches_encoder_intra", "decode_seconds")}
        decoded_path = Path(rec["decoded"])
    raw = np.fromfile(decoded_path, np.uint8).reshape(n, size[1] * 3 // 2, size[0])
    rgb = g5.yuv420_to_rgb(raw)
    if size != FULL:
        rgb = resize(rgb, FULL, area=False)
    shutil.rmtree(work, ignore_errors=True)
    return rgb, {"bytes": nbytes, "kbps": nbytes * 8 / clip.duration / 1000, **extra}


def command_format(args: argparse.Namespace) -> int:
    methods = [parse_method(m) for m in args.methods.split(",")]
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "format"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    began = time.time()
    item, rgb, fg, source_record = load_window(args, work)
    if any(c == "scene" for c, _ in methods):
        arm = json.loads((Path(args.arm) / "publish" / "finetune" / "result.json").read_text())
        if arm["id"].split("/")[-1] != item["video"]:
            raise SystemExit(f"{args.arm}: a scene model of {arm['id']}, not of {item['video']}")
        if file_sha256(Path(args.arm) / "publish" / "finetune" / "video_ft.pth.tar") != arm["model"]["checkpoint_sha256"]:
            raise RuntimeError("the published checkpoint is not the one its result recorded")
        arm_record = {k: arm[k] for k in ("arm", "id", "fill", "steps")} | {"model_sha256": arm["model"]["checkpoint_sha256"]}
    else:
        arm_record = None
    holes = dilate_1080(fg)
    filled, propainter = fill(rgb, holes, args, work)
    n = len(rgb)
    fps = Fraction(item["fps"]).limit_denominator(1001)
    clip = g5.Clip(item["id"], "window", fps, 0, rgb, ~holes,
                   {"dataset": (list(range(n)), ~fg), "own": (list(range(n)), ~fg)},
                   {"id": item["id"], "video": item["video"], "fps": str(fps), "frames": n})
    nets = g5c.load_nets(args)
    flows = g5c.source_flows(clip)
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = g5c.restore(checkpoints)
    rows = []
    done = 0
    for name, (codec, size) in zip(args.methods.split(","), methods):
        values = g5.CRFS if codec == "svt" else g5.DCVC_QPS
        points = []
        for value in values:
            key = f"{name}-{value}"
            if key in restored:
                points.append({**restored[key], "restored_from_checkpoint": True})
            else:
                decoded, rec = code_point(codec, size, value, filled, clip, args, work / key)
                point = {"value": value, **rec, "score": g5c.full_score(clip, decoded, nets, flows)}
                g5c.save(checkpoints, key, point)
                points.append(point)
            done += 1
            progress(done)
        rows.append({"method": name, "codec": codec, "coded_size": list(size), "points": points})
    write_json(stage_dir() / "g6.json", {
        "kind": "format", "item": item["id"], "video": item["video"], "fps": str(fps), "frames": n,
        "limit_frames": args.limit_frames, "mask_set": MASK_SET, "dilate_px_1080": round(DILATE * TO_1080),
        "foreground_share": float(fg.mean()), "hole_share": float(holes.mean()), "source": source_record,
        "arm": arm_record, "propainter": propainter, "metrics": nets.record, "restored": sorted(restored),
        "seconds": round(time.time() - began, 1), **g5c.gpu_record(), "methods": rows})
    shutil.rmtree(work, ignore_errors=True)
    return 0


def validate_stage(stage: Path) -> dict[str, bool]:
    result = json.loads((stage / "g6.json").read_text())
    full = not result["limit_frames"]
    rows = result["methods"]

    def scored(p: dict[str, Any]) -> bool:
        s, k = p["score"], p["score"].get("kid", {})
        return (g5.finite(s["dataset"]["lpips_v"]) and g5.finite(s["dataset"]["psnr_v"])
                and g5.finite(s["flicker"]["flicker_v"]) and g5.finite(k.get("kid_v"))
                and k.get("patches", 0) >= (100 if full else 2))

    return {
        "intended_gpu": tuple(result.get("gpu_capability") or (0, 0)) >= (8, 6),
        "methods_present": bool(rows) and all(r["points"] for r in rows),
        "every_point_scored": all(scored(p) for r in rows for p in r["points"]),
        "dcvc_decodes": all(p["deterministic"] and p["decoder_matches_encoder_intra"]
                            for r in rows if r["codec"] != "svt" for p in r["points"]),
        "rate_follows_value": all((a["kbps"] > b["kbps"]) == (r["codec"] == "svt") for r in rows  # CRF down, QP up
                                  for a, b in zip(r["points"], r["points"][1:])),
        "foreground_plausible": 0.0 < result["foreground_share"] < 0.8,
        "window_is_the_released_one": result["source"]["jpeg_gate_holds"],
        "propainter_on_gpu": result["propainter"]["gpu"] == result["gpu"],
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(args.stage or os.environ.get("PS_STAGE_DIR") or ".")
    checks = validate_stage(stage)
    payload = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), payload)
    print(json.dumps(payload, indent=1))
    return 0 if payload["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fo = sub.add_parser("format")
    for name in ("eval-set", "archive", "masks", "video", "item", "image-ckpt", "video-ckpt", "lpips-backbone",
                 "dists-backbone", "inception", "propainter-tree", "propainter-ckpt", "raft-ckpt",
                 "flow-completion-ckpt"):
        fo.add_argument(f"--{name}", required=True)
    fo.add_argument("--arm", default=None, help="the window's stretch: a G5c finetune archive (scene methods)")
    fo.add_argument("--methods", default=",".join(METHODS))
    fo.add_argument("--limit-frames", type=int, default=0)
    fo.set_defaults(func=command_format)
    va = sub.add_parser("validate")
    va.add_argument("--stage", default=None)
    va.set_defaults(func=command_validate)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
