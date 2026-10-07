"""PLAN step B2: SVT-AV1 and DCVC-UF rate-distortion on VISOR evaluation set v2.

    python -m experiments.visor.b2 run --codec svtav1|dcvc --eval-set JSON --archive DIR \\
        --masks NAME DIR --mask-record NAME JSON --video V.MP4 ... --items all|ID,ID \\
        --frames N --points P,P,... [codec options]
    python -m experiments.visor.b2 validate --codec svtav1|dcvc
    python -m experiments.visor.b2 report --result CODEC=b2.json ... --out DIR

``run``, per item of the set:

1. decodes the window from the source video at ``first_video_index``, keeping
   the decoder's own 8-bit 4:2:0 planes (no colour conversion), and checks every
   released sparse JPEG of the item against the decoded frames, as B1 did
   (the frame the reader's rule names must be the best match within ±2 frames,
   below 3 grey levels);
2. loads each mask set (``--masks NAME DIR``: one ``masks.rle`` per item, B1's
   ``visor_dense`` now, B1b's fill later), checks it against its record, and
   counts per frame the objects a human labelled at the run's keyframes that
   the set lacks (`visor.expected_objects`);
3. for each rate point, encodes and decodes the window (SVT-AV1 with dav1d, on
   CPU; or DCVC-UF on the claimed GPU, decoded from the container bytes in its
   own process), measures the rate from the bitstream bytes and scores every
   frame (`src.codecs.quality`): PSNR over the frame and inside each mask set's
   foreground and background, weighted PSNR 0.7 fg + 0.3 bg (per frame, in dB,
   then the mean), LPIPS likewise, MS-SSIM, VMAF and plane PSNR.

Streams are published (``publish/streams/<codec>/<item>/``), so a later mask set
is added by ``run --streams DIR``, which decodes the published streams instead
of encoding and scores them with every mask set given.

Writes ``b2.json`` to ``PS_STAGE_DIR``; per-frame scores go to
``PS_SCRATCH_DIR/publish/frames/<item>/<point>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import MATCH, TIE, file_sha256, mask_digest, progress, stage_dir, write_json
from src.codecs import quality, svtav1
from src.segmentation import visor
from src.segmentation.masks import ClipMasks

#: Weight of the foreground in weighted PSNR and weighted LPIPS (docs/experiments.md).
FOREGROUND_WEIGHT = 0.7
GATE_MARGIN = 2
VIDEO_TYPES = {"hevc 1920x1080 50.000": "EK-100", "h264 1920x1080 59.940": "EK-55"}
CODECS = {"svtav1": "SVT-AV1", "dcvc": "DCVC-UF"}
#: Quality falls as SVT-AV1's CRF rises and rises with DCVC-UF's QP.
QUALITY_RISES_WITH_POINT = {"svtav1": False, "dcvc": True}
STREAM_NAMES = {"svtav1": "stream.ivf", "dcvc": "stream.psdc"}
SUMMARY_KEYS = ("wpsnr", "wpsnr_pooled", "psnr_fg", "psnr_bg", "psnr_frame", "wlpips", "lpips_fg", "lpips_bg",
                "lpips_frame", "ms_ssim", "vmaf", "psnr_y", "psnr_yuv")


# ----------------------------------------------------------------- inputs

def named(pairs: list[list[str]] | None) -> dict[str, str]:
    """``--option NAME PATH`` pairs; each is a whole argument, as fleet placeholders require."""
    return {name: path for name, path in pairs or []}


def parse_named(values: list[str] | None) -> dict[str, str]:
    out = {}
    for value in values or []:
        name, _, path = value.partition("=")
        if not name or not path:
            raise SystemExit(f"expected NAME=PATH, got {value!r}")
        out[name] = path
    return out


def select_items(eval_set: dict[str, Any], spec: str) -> list[dict[str, Any]]:
    items = eval_set["items"]
    if spec == "all":
        return list(items)
    wanted = spec.split(",")
    by_id = {item["id"]: item for item in items}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise SystemExit(f"items not in the evaluation set: {missing}")
    return [by_id[i] for i in wanted]


def parse_points(spec: str) -> list[float]:
    return [float(p) for p in spec.split(",") if p]


def point_name(value: float) -> str:
    return f"{value:g}"


def video_type(item: dict[str, Any]) -> str:
    return VIDEO_TYPES[item["video_class"]]


def mask_root(directory: Path, item_id: str) -> Path:
    for candidate in (directory / item_id, directory / "masks" / item_id, directory / "publish" / "masks" / item_id):
        if (candidate / "masks.rle").is_file():
            return candidate / "masks.rle"
    raise FileNotFoundError(f"no masks.rle for {item_id} under {directory}")


# ----------------------------------------------------------------- source window

def jpeg_targets(item: dict[str, Any], archive: Path) -> list[dict[str, Any]]:
    """Each released sparse JPEG of the item with the decoded frame the reader's rule names."""
    mapping = json.loads((archive / "frame_mapping.json").read_text())[item["video"]]
    out = []
    for member in item["sparse_jpegs"]:
        name = Path(member).name
        epic = visor.frame_number(mapping[name])
        out.append({"jpeg": member, "visor_frame": visor.frame_number(name), "epic_frame": epic,
                    "index": visor.epic_frame_to_video_index(epic, item["fps"])})
    return out


def decode_window(video: Path, item: dict[str, Any], frames: int, archive: Path, out: Path, threads: int) -> dict[str, Any]:
    """Write the window's 4:2:0 planes to ``out`` and check the item's sparse JPEGs on the decoded frames."""
    from PIL import Image

    started = time.time()
    first = int(item["first_video_index"])
    targets = jpeg_targets(item, archive)
    gate = {t["index"] + d for t in targets for d in range(-GATE_MARGIN, GATE_MARGIN + 1) if t["index"] + d >= 0}
    rgb: dict[int, np.ndarray] = {}
    digest = hashlib.sha256()
    nxt, formats, ranges, spaces = first, set(), set(), set()
    width = height = 0
    with out.open("xb") as handle:
        for index, frame in visor.decoded_frames(video, sorted(set(range(first, first + frames)) | gate), threads=threads):
            if index in gate:
                rgb[index] = frame.to_ndarray(format="rgb24")
            if not first <= index < first + frames:
                continue
            if index != nxt:
                raise RuntimeError(f"{item['id']}: decoded frame {index}, expected {nxt}")
            fmt = frame.format.name
            if fmt not in ("yuv420p", "yuvj420p"):
                raise RuntimeError(f"{item['id']}: decoded format {fmt} is not 8-bit 4:2:0")
            formats.add(fmt)
            ranges.add(str(getattr(frame, "color_range", None)))
            spaces.add(str(getattr(frame, "colorspace", None)))
            planes = np.ascontiguousarray(frame.to_ndarray())
            height, width = planes.shape[0] * 2 // 3, planes.shape[1]
            data = planes.tobytes()
            digest.update(data)
            handle.write(data)
            nxt += 1
    if nxt != first + frames:
        raise RuntimeError(f"{item['id']}: decoded {nxt - first} of {frames} frames")
    checks = []
    for target in targets:
        released = np.asarray(Image.open(archive / target["jpeg"]).convert("RGB")).astype(np.int16)
        mae = {i: float(np.abs(rgb[i].astype(np.int16) - released).mean()) for i in sorted(rgb)
               if abs(i - target["index"]) <= GATE_MARGIN}
        best = min(mae, key=lambda i: mae[i]) if mae else None
        holds = bool(mae) and target["index"] in mae and best is not None and \
            mae[target["index"]] - mae[best] <= TIE and mae[best] < MATCH
        checks.append({**target, "in_window": first <= target["index"] < first + frames,
                       "best_index": best, "rule_mae": round(mae.get(target["index"], math.nan), 3),
                       "best_mae": round(mae[best], 3) if best is not None else None, "holds": holds,
                       "window_mae": {str(i): round(v, 2) for i, v in mae.items()}})
    full_range = formats == {"yuvj420p"} or ranges == {"2"} or ranges == {"JPEG"}
    return {
        "path": str(out), "frames": frames, "first_video_index": first, "width": width, "height": height,
        "yuv_sha256": digest.hexdigest(), "pix_fmt": sorted(formats), "color_range": sorted(ranges),
        "colorspace": sorted(spaces), "full_range": full_range, "jpeg_gate": checks,
        "seconds": round(time.time() - started, 2),
    }


# ----------------------------------------------------------------- masks

def load_mask_sets(item: dict[str, Any], frames: int, masks: dict[str, str], records: dict[str, str],
                   archive: Path) -> tuple[dict[str, ClipMasks], dict[str, Any]]:
    dense = visor.load_annotations(archive / item["dense_member"])
    sparse = visor.load_annotations(archive / f"annotations/{item['video']}.json")
    expected = visor.expected_objects(dense, sparse)
    clips, info = {}, {}
    for name, directory in masks.items():
        path = mask_root(Path(directory), item["id"])
        clip = ClipMasks.load(path)
        record = None
        if name in records:
            rows = {row["id"]: row for row in json.loads(Path(records[name]).read_text())["items"]}
            record = rows.get(item["id"])
        indices = clip.meta.get("video_indices") or []
        per_frame: list[dict[str, int]] = []
        missing_labels: Counter[str] = Counter()
        for i in range(frames):
            wanted = expected.get(int(item["first_visor_frame"]) + i, set())
            present = {visor.object_key(inst.class_name, inst.label) for inst in clip.frames[i]}
            lacking = wanted - present
            missing_labels.update(lacking)
            per_frame.append({"expected": len(wanted), "missing": len(lacking),
                              "missing_hands": len(lacking & set(visor.HANDS))})
        shares = [f["missing"] / f["expected"] for f in per_frame if f["expected"]]
        info[name] = {
            "masks_rle": str(path), "masks_rle_sha256": file_sha256(path), "mask_sha256": mask_digest(clip),
            "record_masks_rle_sha256": record.get("masks_rle_sha256") if record else None,
            "record_mask_sha256": record.get("mask_sha256") if record else None,
            "frames_in_masks": len(clip), "labelled_frames": sum((clip.labelled or [False] * len(clip))[:frames]),
            "video_indices_match": indices[:frames] == list(range(item["first_video_index"], item["first_video_index"] + frames)),
            "provenance": dict(Counter(inst.provenance for f in clip.frames[:frames] for inst in f)),
            "foreground_fraction_mean": None,
            "missing_objects": {
                "rule": "objects (hand class or label) a human labelled at both keyframes of the frame's run, or at the frame itself if it is a keyframe, absent from this mask set",
                "share_per_frame": [round(f["missing"] / f["expected"], 4) if f["expected"] else None for f in per_frame],
                "mean_share": round(float(np.mean(shares)), 4) if shares else None,
                "frames_with_missing": sum(1 for f in per_frame if f["missing"]),
                "frames_with_expected": sum(1 for f in per_frame if f["expected"]),
                "missing_hand_frames": sum(1 for f in per_frame if f["missing_hands"]),
                "missing_labels": dict(sorted(missing_labels.items())),
            },
        }
        clips[name] = clip
    return clips, info


def frame_regions(clips: dict[str, ClipMasks], frames: int) -> list[dict[str, np.ndarray]]:
    out = []
    for i in range(frames):
        regions = {}
        for name, clip in clips.items():
            fg = clip.foreground(i)
            regions[f"{name}/fg"] = fg
            regions[f"{name}/bg"] = ~fg
        out.append(regions)
    return out


# ----------------------------------------------------------------- scores

def weighted(fg: float | None, bg: float | None) -> float | None:
    if fg is None or bg is None:
        return None
    return FOREGROUND_WEIGHT * fg + (1.0 - FOREGROUND_WEIGHT) * bg


def mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(kept)) if kept else None


def summarize(per_frame: list[dict[str, Any]], vmaf: list[float], mask_sets: list[str]) -> dict[str, Any]:
    """Item-level scores of one rate point: means over frames, plus PSNR from the window's pooled error."""
    def pooled(region: str) -> float | None:
        sse = sum(f[region]["sse"] for f in per_frame)
        pixels = sum(f[region]["pixels"] for f in per_frame)
        return quality.psnr(sse / (3 * pixels)) if pixels else None

    out: dict[str, Any] = {
        "frame": {
            "psnr": mean([f["frame"]["psnr"] for f in per_frame]), "psnr_pooled": pooled("frame"),
            "lpips": mean([f["frame"]["lpips"] for f in per_frame]),
            "ms_ssim": mean([f["ms_ssim"] for f in per_frame]),
            "vmaf": mean(list(vmaf)),
            **{k: mean([f["planes"][k] for f in per_frame]) for k in ("psnr_y", "psnr_u", "psnr_v", "psnr_yuv")},
        },
        "mask_sets": {},
    }
    for name in mask_sets:
        fg, bg = f"{name}/fg", f"{name}/bg"
        scored = [f for f in per_frame if f[fg]["pixels"] and f[bg]["pixels"]]
        out["mask_sets"][name] = {
            "wpsnr": mean([weighted(f[fg]["psnr"], f[bg]["psnr"]) for f in scored]),
            "wpsnr_pooled": weighted(pooled(fg), pooled(bg)),
            "psnr_fg": mean([f[fg]["psnr"] for f in per_frame]), "psnr_bg": mean([f[bg]["psnr"] for f in per_frame]),
            "wlpips": mean([weighted(f[fg]["lpips"], f[bg]["lpips"]) for f in scored]),
            "lpips_fg": mean([f[fg]["lpips"] for f in per_frame]), "lpips_bg": mean([f[bg]["lpips"] for f in per_frame]),
            "frames_scored": len(scored),
            "foreground_fraction": mean([f[fg]["pixels"] / (f[fg]["pixels"] + f[bg]["pixels"]) for f in per_frame]),
        }
    return out


def flat_summary(summary: dict[str, Any], mask_set: str) -> dict[str, float | None]:
    ms = summary["mask_sets"][mask_set]
    frame = summary["frame"]
    return {
        "wpsnr": ms["wpsnr"], "wpsnr_pooled": ms["wpsnr_pooled"], "psnr_fg": ms["psnr_fg"], "psnr_bg": ms["psnr_bg"],
        "psnr_frame": frame["psnr"], "wlpips": ms["wlpips"], "lpips_fg": ms["lpips_fg"], "lpips_bg": ms["lpips_bg"],
        "lpips_frame": frame["lpips"], "ms_ssim": frame["ms_ssim"], "vmaf": frame["vmaf"],
        "psnr_y": frame["psnr_y"], "psnr_yuv": frame["psnr_yuv"],
    }


# ----------------------------------------------------------------- variants

#: SVT-AV1's ROI map gives one quantizer offset per 64x64 block per frame.
ROI_BLOCK = 64


def variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The codec configurations one job runs; the label is empty for B2's own configuration."""
    if args.codec == "dcvc":
        return [{"label": "" if r == "full" else f"range-{r}", "range": r} for r in args.dcvc_range.split(",")]
    presets = [int(p) for p in str(args.preset).split(",")]
    offsets = [int(r) for r in str(args.roi_offset).split(",")]
    out = []
    for preset in presets:
        for offset in offsets:
            parts = [f"p{preset}"] if len(presets) > 1 or preset != 4 else []
            parts += [f"roi{offset}"] if offset else []
            parts += [f"tf{args.enable_tf}"] if args.enable_tf is not None else []
            out.append({"label": "-".join(parts), "preset": preset, "roi_offset": offset, "enable_tf": args.enable_tf})
    return out


def write_roi_map(regions: list[dict[str, np.ndarray]], mask_set: str, frames: int, width: int, height: int,
                  offset: int, path: Path) -> dict[str, Any]:
    """SVT-AV1 ROI map: per frame, its number, then one quantizer offset per 64x64 block in
    raster order. A negative ``offset`` lowers the quantizer of every block that holds any
    foreground pixel; a positive one raises it on every other block. In SVT-AV1 4.2.0's CRF
    mode only the second works: a negative offset adds bytes without raising the region's
    PSNR, a positive one lowers it (B2b probe, docs/experiments.md)."""
    rows, cols = -(-height // ROI_BLOCK), -(-width // ROI_BLOCK)
    lines, shares = [], []
    for index in range(frames):
        padded = np.zeros((rows * ROI_BLOCK, cols * ROI_BLOCK), bool)
        padded[:height, :width] = regions[index][f"{mask_set}/fg"]
        blocks = padded.reshape(rows, ROI_BLOCK, cols, ROI_BLOCK).any(axis=(1, 3))
        target = blocks if offset < 0 else ~blocks
        lines.append(f"{index} " + " ".join(str(offset if b else 0) for b in target.ravel()))
        shares.append(float(blocks.mean()))
    path.write_text("\n".join(lines) + "\n")
    return {"path": str(path), "sha256": file_sha256(path), "block": ROI_BLOCK, "grid": [rows, cols],
            "offset": offset, "mask_set": mask_set, "block_share_mean": round(float(np.mean(shares)), 4)}


def convert_range(src: Path, dst: Path, frames: int, width: int, height: int, *, to_limited: bool) -> None:
    """Full-range 8-bit 4:2:0 to limited range (Y 16-235, C 16-240) or back, rounded."""
    with dst.open("xb") as handle:
        for first in range(0, frames, 16):
            count = min(16, frames - first)
            data = quality.read_frames(src, width, height, first, count).astype(np.float32)
            luma, chroma = data[:, :height], data[:, height:]
            if to_limited:
                luma, chroma = 16.0 + luma * 219.0 / 255.0, 128.0 + (chroma - 128.0) * 224.0 / 255.0
            else:
                luma, chroma = (luma - 16.0) * 255.0 / 219.0, 128.0 + (chroma - 128.0) * 255.0 / 224.0
            out = np.concatenate([luma, chroma], axis=1)
            handle.write(np.clip(np.round(out), 0, 255).astype(np.uint8).tobytes())


# ----------------------------------------------------------------- codecs

def code_svtav1(args: argparse.Namespace, source: dict[str, Any], fps: Fraction, point: float, work: Path,
                stream: Path | None, variant: dict[str, Any], cpus: list[int] | None,
                roi: dict[str, Any] | None) -> dict[str, Any]:
    width, height, frames = source["width"], source["height"], source["frames"]
    if stream is None:
        extra = ["--roi-map-file", roi["path"]] if roi else []
        extra += ["--enable-tf", str(variant["enable_tf"])] if variant["enable_tf"] is not None else []
        record = svtav1.code(Path(source["path"]), work, width=width, height=height, fps=fps, frames=frames, crf=point,
                             preset=variant["preset"], threads=args.threads, full_range=source["full_range"],
                             encoder=args.encoder, decoder=args.decoder, cpus=cpus, extra=extra)
    else:
        decoded = work / "decoded.yuv"
        command = svtav1.decode_command(args.decoder, stream, decoded, threads=args.threads)
        timing = svtav1.run(command, 3600, cpus)
        data = stream.read_bytes()
        sizes = svtav1.ivf_frames(data)
        record = {"stream": str(stream), "decoded": str(decoded), "stream_sha256": hashlib.sha256(data).hexdigest(),
                  "file_bytes": len(data), "payload_bytes": sum(sizes), "temporal_units": len(sizes),
                  "decoded_frames": decoded.stat().st_size / quality.frame_bytes(width, height),
                  "decode_command": command, "decode_seconds": round(timing["wall"], 3), "reused_stream": True}
    record["rate_bytes"] = record["payload_bytes"]
    record["stream_frames"] = record["temporal_units"]
    record["roi"] = roi
    return record


def code_dcvc(args: argparse.Namespace, source: dict[str, Any], point: float, work: Path, stream: Path | None,
              profile: bool, variant: dict[str, Any]) -> dict[str, Any]:
    from src.codecs.dcvc_uf_worker import dcvc_command

    if int(point) != point:
        raise SystemExit(f"DCVC-UF QP must be an integer, got {point}")
    container = stream or work / "stream.psdc"
    width, height, frames = source["width"], source["height"], source["frames"]
    limited = variant["range"] == "limited"
    if limited and not source["full_range"]:
        raise SystemExit("the limited-range variant needs a full-range source")
    frames_file, out_file = Path(source["path"]), work / "decoded.yuv"
    if limited:
        # The codec sees limited range; scoring sees its output mapped back to the source's full range.
        frames_file, out_file = work / "source-limited.yuv", work / "decoded-limited.yuv"
        convert_range(Path(source["path"]), frames_file, frames, width, height, to_limited=True)
    plan = {
        "structure": args.structure, "qp": int(point), "frame_count": frames, "height": height,
        "width": width, "src_type": "yuv420", "frames_file": str(frames_file), "container": str(container),
        "out_file": str(out_file), "image_ckpt": args.image_ckpt, "image_sha256": args.image_sha256,
        "video_ckpt": args.video_ckpt, "video_sha256": args.video_sha256, "profile": profile,
    }
    plan_path = work / "plan.json"
    write_json(plan_path, plan)
    reports: dict[str, Any] = {}
    commands = {}
    for action in (("decode",) if stream else ("encode", "decode")):
        report = work / f"{action}.json"
        command, env, cwd = dcvc_command(Path(sys.prefix), action, plan_path, report)
        started = time.time()
        done = subprocess.run(command, env={**os.environ, **env}, cwd=cwd, capture_output=True, text=True, timeout=3600)
        if done.returncode:
            raise RuntimeError(f"DCVC-UF {action} failed ({done.returncode}):\n{done.stderr[-4000:]}")
        reports[action] = json.loads(report.read_text())
        reports[action]["wall_seconds"] = round(time.time() - started, 3)
        commands[action] = command
    if limited:
        convert_range(out_file, work / "decoded.yuv", frames, width, height, to_limited=False)
        os.remove(out_file)
        os.remove(frames_file)
    data = container.read_bytes()
    decode = reports["decode"]
    record = {
        "stream": str(container), "decoded": str(work / "decoded.yuv"), "stream_sha256": hashlib.sha256(data).hexdigest(),
        "file_bytes": len(data), "rate_bytes": len(data), "stream_frames": decode["frame_count"],
        "decoded_frames": (work / "decoded.yuv").stat().st_size / quality.frame_bytes(source["width"], source["height"]),
        "commands": commands, "plan": plan,
        "decode": {k: v for k, v in decode.items() if k != "passes"},
        "decode_seconds": [p["decode_seconds"] for p in decode["passes"]],
        "model_decode_seconds": [p.get("model_decode_seconds") for p in decode["passes"]], "range": variant["range"],
        "deterministic": decode["deterministic"], "reused_stream": stream is not None,
    }
    if "encode" in reports:
        encode = reports["encode"]
        record["encode"] = {k: v for k, v in encode.items() if k != "nals"}
        record["nal_count"] = len(encode["nals"])
        record["decoder_matches_encoder_intra"] = decode["passes"][0]["frame_sha256"][0] == encode["i_recon_sha256"]
        record["same_gpu"] = encode["environment"]["cuda_visible_devices"] == decode["environment"]["cuda_visible_devices"]
        record["encode_seconds"] = encode["encode_seconds"]
        record["model_encode_seconds"] = encode.get("model_encode_seconds")
    return record


# ----------------------------------------------------------------- one item

_LPIPS: dict[str, Any] = {}


def lpips_net(backbone: str, device: str) -> Any:
    key = f"{backbone}|{device}"
    if key not in _LPIPS:
        _LPIPS[key] = quality.load_lpips(Path(backbone), device)
    return _LPIPS[key]


def log(message: str) -> None:
    """One timestamped line on stderr, which the fleet keeps in command.log."""
    print(f"{time.strftime('%H:%M:%S')} [{os.getpid()}] {message}", file=sys.stderr, flush=True)


_SLOT: dict[str, list[int] | None] = {"cpus": None}


def _take_slot(slots: Any) -> None:
    """Pool initializer: this worker's own cores for its encoders, decoders and torch."""
    _SLOT["cpus"] = slots.get()


def metric_device(args: argparse.Namespace) -> str:
    return args.metric_device or ("cuda" if args.codec == "dcvc" else "cpu")


def run_item(args: argparse.Namespace, item: dict[str, Any], first_item: bool) -> dict[str, Any]:
    import torch

    torch.set_num_threads(max(1, args.threads))
    cpus = _SLOT["cpus"]
    started = time.time()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "work" / item["id"]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    publish = scratch / "publish"
    videos = {Path(v).stem: Path(v) for v in args.video}
    archive = Path(args.archive)
    frames = min(args.frames, int(item["frames"]))
    log(f"{item['id']}: decoding {frames} frames")
    # PyAV decodes in a fresh process: in one that has loaded torchvision the
    # decode stalled (jobs 20261007T091602Z-72ae5f5b, 20261007T093027Z-71cc5ec2).
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        source = pool.submit(decode_window, videos[item["video"]], item, frames, archive, work / "source.yuv",
                             args.threads).result()
    log(f"{item['id']}: loading masks")
    clips, mask_info = load_mask_sets(item, frames, named(args.masks), named(args.mask_record), archive)
    regions = frame_regions(clips, frames)
    log(f"{item['id']}: masks ready")
    fps = Fraction(item["fps"]).limit_denominator(1001)
    device = metric_device(args)
    net = lpips_net(args.lpips_backbone, device)
    log(f"{item['id']}: LPIPS on {device}, cores {cpus}")
    rows = []
    cross_device = None
    jobs = [(variant, point) for variant in variants(args) for point in parse_points(args.points)]
    for number, (variant, point) in enumerate(jobs):
        name = point_name(point)
        label = variant["label"]
        point_work = work / f"{label or 'base'}-p{name}"
        point_work.mkdir()
        stream = None
        if args.streams:
            stream = Path(args.streams) / args.codec / item["id"] / label / name / STREAM_NAMES[args.codec]
        roi = None
        if args.codec == "svtav1" and variant["roi_offset"]:
            roi = write_roi_map(regions, args.roi_mask_set, frames, source["width"], source["height"],
                                variant["roi_offset"], point_work / "roi.txt")
        t0 = time.time()
        log(f"{item['id']}: {label or 'base'} point {name}: coding")
        if args.codec == "svtav1":
            coded = code_svtav1(args, source, fps, point, point_work, stream, variant, cpus, roi)
        else:
            coded = code_dcvc(args, source, point, point_work, stream, profile=first_item and number == 0,
                              variant=variant)
        decoded = Path(coded["decoded"])
        log(f"{item['id']}: point {name}: scoring")
        t1 = time.time()
        per_frame = quality.score(Path(source["path"]), decoded, regions, width=source["width"], height=source["height"],
                                  frames=frames, full_range=source["full_range"], device=device, lpips_net=net)
        t2 = time.time()
        vmaf = quality.vmaf(args.vmaf_ffmpeg, decoded, Path(source["path"]), point_work / "vmaf.json",
                            width=source["width"], height=source["height"], fps=fps, threads=args.threads)
        t3 = time.time()
        if device == "cuda" and cross_device is None:
            cross_device = cross_device_check(Path(source["path"]), decoded, regions, source, args.lpips_backbone)
        summary = summarize(per_frame, vmaf["per_frame"], list(clips))
        seconds = frames / float(fps)
        kept = publish / "streams" / args.codec / item["id"] / label / name
        kept.mkdir(parents=True, exist_ok=True)
        if stream is None:
            shutil.copy2(coded["stream"], kept / STREAM_NAMES[args.codec])
            if roi:
                shutil.copy2(roi["path"], kept / "roi.txt")
        write_json(publish / "frames" / item["id"] / f"{args.codec}-{label or 'base'}-{name}.json", {
            "item": item["id"], "codec": args.codec, "variant": variant, "point": point, "frames": per_frame,
            "vmaf": vmaf["per_frame"],
        })
        rows.append({
            "variant": label, "variant_config": variant, "point": point, "rate_bytes": coded["rate_bytes"], "file_bytes": coded["file_bytes"],
            "kbps": coded["rate_bytes"] * 8 / seconds / 1000.0,
            "bpp": coded["rate_bytes"] * 8 / (frames * source["width"] * source["height"]),
            "stream_sha256": coded["stream_sha256"], "stream_frames": coded["stream_frames"],
            "decoded_frames": coded["decoded_frames"], "codec_record": {k: v for k, v in coded.items() if k not in ("stream", "decoded")},
            "vmaf_command": vmaf["command"], "vmaf_version": vmaf["version"], "vmaf_pooled_mean": vmaf["pooled_mean"],
            "summary": summary, "seconds": {"code": round(t1 - t0, 2), "score": round(t2 - t1, 2), "vmaf": round(t3 - t2, 2)},
        })
        shutil.rmtree(point_work)
    for name, info in mask_info.items():
        info["foreground_fraction_mean"] = rows[0]["summary"]["mask_sets"][name]["foreground_fraction"] if rows else None
    os.remove(source["path"])
    return {
        "id": item["id"], "video": item["video"], "video_type": video_type(item), "fps": item["fps"], "frames": frames,
        "first_visor_frame": item["first_visor_frame"], "video_sha256_expected": item["video_file"]["sha256"],
        "source": source, "mask_sets": mask_info, "points": rows, "cross_device": cross_device,
        "seconds": round(time.time() - started, 2),
    }


def cross_device_check(reference: Path, decoded: Path, regions: list[dict[str, np.ndarray]], source: dict[str, Any],
                       backbone: str) -> dict[str, Any]:
    """The same frames scored on CPU and on the GPU: SVT-AV1 is scored on CPU, DCVC-UF on the GPU."""
    count = min(2, source["frames"])
    out = {}
    for device in ("cpu", "cuda"):
        out[device] = quality.score(reference, decoded, regions[:count], width=source["width"], height=source["height"],
                                    frames=count, full_range=source["full_range"], device=device,
                                    lpips_net=lpips_net(backbone, device), batch=count)
    diffs = {"lpips": 0.0, "ms_ssim": 0.0, "psnr": 0.0}
    for a, b in zip(out["cpu"], out["cuda"]):
        diffs["ms_ssim"] = max(diffs["ms_ssim"], abs(a["ms_ssim"] - b["ms_ssim"]))
        for region in a:
            if isinstance(a[region], dict) and "lpips" in a[region] and a[region]["lpips"] is not None:
                diffs["lpips"] = max(diffs["lpips"], abs(a[region]["lpips"] - b[region]["lpips"]))
                diffs["psnr"] = max(diffs["psnr"], abs(a[region]["psnr"] - b[region]["psnr"]))
    return {"frames": count, "max_abs_diff": diffs}


def _run_item_job(payload: tuple[argparse.Namespace, dict[str, Any], bool]) -> dict[str, Any]:
    args, item, first = payload
    return run_item(args, item, first)


# ----------------------------------------------------------------- aggregate

def curves(items: list[dict[str, Any]], mask_sets: list[str]) -> dict[str, Any]:
    """Per video type and overall, per rate point: means over items (an item is one sample)."""
    groups = {"all": items, **{t: [i for i in items if i["video_type"] == t] for t in sorted({i["video_type"] for i in items})}}
    out: dict[str, Any] = {}
    for group, members in groups.items():
        out[group] = {}
        for mask_set in mask_sets:
            rows = []
            keys = sorted({(p.get("variant", ""), p["point"]) for i in members for p in i["points"]})
            for label, point in keys:
                matched = [(i, p) for i in members for p in i["points"] if (p.get("variant", ""), p["point"]) == (label, point)]
                values = [flat_summary(p["summary"], mask_set) for _, p in matched]
                row: dict[str, Any] = {"variant": label, "point": point, "items": len(matched)}
                for key in ("kbps", "bpp"):
                    data = [p[key] for _, p in matched]
                    row[key] = float(np.mean(data))
                    row[f"{key}_median"] = float(np.median(data))
                for key in SUMMARY_KEYS:
                    data = [v[key] for v in values if v[key] is not None]
                    row[key] = float(np.mean(data)) if data else None
                    row[f"{key}_sd"] = float(np.std(data, ddof=1)) if len(data) > 1 else None
                shares = [i["mask_sets"][mask_set]["missing_objects"]["mean_share"] for i, _ in matched]
                row["missing_object_share"] = mean(shares)
                rows.append(row)
            out[group][mask_set] = rows
    return out


def command_run(args: argparse.Namespace) -> int:
    import faulthandler

    # A stalled stage leaves every thread's stack in command.log.
    faulthandler.dump_traceback_later(240, repeat=True, file=sys.stderr)
    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    videos = {Path(v).stem for v in args.video}
    absent = sorted({i["video"] for i in items} - videos)
    if absent:
        raise SystemExit(f"no --video for {absent}")
    for attr in ("image_ckpt", "video_ckpt"):
        if getattr(args, attr, None):
            setattr(args, attr.replace("ckpt", "sha256"), file_sha256(Path(getattr(args, attr))))
    tools: dict[str, Any] = {"vmaf": quality.vmaf_record(args.vmaf_ffmpeg)}
    if args.codec == "svtav1":
        tools.update(encoder=svtav1.tool(args.encoder), decoder=svtav1.tool(args.decoder))
    else:
        tools["dcvc"] = {"tree": str(Path(sys.prefix) / "opt" / "DCVC"), "image_ckpt": args.image_ckpt,
                         "image_sha256": args.image_sha256, "video_ckpt": args.video_ckpt,
                         "video_sha256": args.video_sha256, "structure": args.structure}
        opt = Path(sys.prefix) / "opt" / "pointstream.opt.json"
        if opt.is_file():
            tools["dcvc"]["opt_record"] = json.loads(opt.read_text())
    tools["lpips"] = quality.lpips_record(Path(args.lpips_backbone))
    import torch
    import torchmetrics

    tools["torch"] = {"torch": torch.__version__, "torchmetrics": torchmetrics.__version__, "python": sys.executable}
    rows: list[dict[str, Any]] = []
    if args.codec == "svtav1":
        allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
        workers = args.workers or max(1, min(len(items), allowance // max(1, args.threads)))
        context = multiprocessing.get_context("spawn")
        manager = context.Manager()
        slots = manager.Queue()
        # Each worker confines its tools to its own cores (Linux), so encodes do not
        # spread over the host and timings mean "args.threads cores".
        allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        for k in range(workers):
            share = allowed[k * args.threads:(k + 1) * args.threads]
            slots.put(share if len(share) == args.threads else None)
        # Spawned, not forked: the parent has already started torch's thread pools.
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_take_slot,
                                 initargs=(slots,)) as pool:
            for row in pool.map(_run_item_job, [(args, item, n == 0) for n, item in enumerate(items)]):
                rows.append(row)
                progress(len(rows))
        manager.shutdown()
    else:
        workers = 1
        for n, item in enumerate(items):
            rows.append(run_item(args, item, n == 0))
            progress(len(rows))
    mask_sets = list(named(args.masks))
    write_json(stage_dir() / "b2.json", {
        "codec": args.codec, "codec_name": CODECS[args.codec],
        "config": {
            "points": parse_points(args.points), "frames": args.frames, "items": args.items,
            "preset": args.preset if args.codec == "svtav1" else None, "variants": variants(args),
            "roi_mask_set": args.roi_mask_set if args.codec == "svtav1" else None,
            "structure": args.structure if args.codec == "dcvc" else None,
            "threads": args.threads, "workers": workers, "streams": args.streams,
            "foreground_weight": FOREGROUND_WEIGHT, "metric_device": metric_device(args),
            "rgb_view": "bilinear chroma upsampling, BT.709, range as the source declares, rounded to 8 bits",
        },
        "eval_set": {"path": args.eval_set, "sha256": file_sha256(Path(args.eval_set)), "name": eval_set["name"]},
        "masks": {name: {"path": path, "record": named(args.mask_record).get(name)} for name, path in named(args.masks).items()},
        "tools": tools, "items": rows, "curves": curves(rows, mask_sets),
    })
    return 0


# ----------------------------------------------------------------- validate

def finite(value: Any, low: float, high: float) -> bool:
    return value is not None and math.isfinite(value) and low <= value <= high


def validate(stage: Path) -> dict[str, bool]:
    result = json.loads((stage / "b2.json").read_text())
    codec, rows = result["codec"], result["items"]
    eval_items = {i["id"]: i for i in json.loads(Path(result["eval_set"]["path"]).read_text())["items"]}
    staging = json.loads((stage / "staging.json").read_text()) if (stage / "staging.json").exists() else {}
    staged = {Path(s.get("source", s.get("path", ""))).stem: s.get("sha256") for s in staging.get("inputs", [])}
    points = [p for r in rows for p in r["points"]]
    gates = [g for r in rows for g in r["source"]["jpeg_gate"]]
    mask_sets = [(r, name, m) for r in rows for name, m in r["mask_sets"].items()]
    checks = {
        "items_processed": bool(rows) and all(r["points"] for r in rows),
        "every_point_per_item": all(
            len(r["points"]) == len(result["config"]["points"]) * len(result["config"].get("variants") or [None]) for r in rows),
        "windows_start_at_first_video_index": all(r["source"]["first_video_index"] == eval_items[r["id"]]["first_video_index"] for r in rows),
        "windows_are_1080p_420": all((r["source"]["width"], r["source"]["height"]) == (1920, 1080) for r in rows),
        "staged_videos_match_eval_set_sha256": all(staged.get(r["video"]) in (None, r["video_sha256_expected"]) for r in rows)
        and any(staged.get(r["video"]) == r["video_sha256_expected"] for r in rows),
        "sparse_jpegs_checked": bool(gates),
        "every_sparse_jpeg_matches_its_decoded_frame": all(g["holds"] for g in gates),
        "mask_sets_match_their_records": bool(mask_sets) and all(
            m["record_mask_sha256"] in (None, m["mask_sha256"]) and m["record_masks_rle_sha256"] in (None, m["masks_rle_sha256"])
            and (name != "visor_dense" or m["record_mask_sha256"] is not None) for _, name, m in mask_sets),
        "masks_placed_on_the_window": all(m["video_indices_match"] and m["labelled_frames"] == r["frames"] for r, _, m in mask_sets),
        "visor_dense_lacks_no_labelled_hand": all(m["missing_objects"]["missing_hand_frames"] == 0 for _, name, m in mask_sets if name == "visor_dense"),
        "missing_shares_in_unit_interval": all(s is None or 0.0 <= s <= 1.0 for _, _, m in mask_sets for s in m["missing_objects"]["share_per_frame"]),
        "decoded_every_frame": all(p["decoded_frames"] == r["frames"] and p["stream_frames"] == r["frames"] for r in rows for p in r["points"]),
        "rate_positive": all(p["rate_bytes"] > 0 for p in points),
        "psnr_plausible": all(finite(p["summary"]["frame"]["psnr"], 15, 100) and all(
            finite(m["wpsnr"], 10, 100) for m in p["summary"]["mask_sets"].values()) for p in points),
        "perceptual_in_range": all(finite(p["summary"]["frame"]["ms_ssim"], 0, 1) and finite(p["summary"]["frame"]["lpips"], 0, 1)
                                   and finite(p["summary"]["frame"]["vmaf"], 0, 100) for p in points),
    }
    rises = QUALITY_RISES_WITH_POINT[codec]
    monotone = []
    for r in rows:
        for label in sorted({p.get("variant", "") for p in r["points"]}):
            ordered = sorted((p for p in r["points"] if p.get("variant", "") == label), key=lambda p: p["point"], reverse=not rises)
            rates = [p["rate_bytes"] for p in ordered]
            quality_values = [p["summary"]["frame"]["psnr"] for p in ordered]
            monotone.append(all(a < b for a, b in zip(rates, rates[1:])) and all(a < b for a, b in zip(quality_values, quality_values[1:])))
    checks["rate_and_quality_rise_together"] = all(monotone)
    if result["config"]["metric_device"] == "cuda":
        cross = [r["cross_device"] for r in rows if r.get("cross_device")]
        checks["metrics_agree_between_cpu_and_gpu"] = bool(cross) and all(
            c["max_abs_diff"]["psnr"] < 1e-3 and c["max_abs_diff"]["lpips"] < 1e-3 and c["max_abs_diff"]["ms_ssim"] < 1e-4 for c in cross)
    if codec == "dcvc":
        dcvc = [p["codec_record"] for p in points]
        encoded = [d for d in dcvc if "encode" in d]
        checks["dcvc_decode_deterministic"] = all(d["deterministic"] for d in dcvc)
        checks["dcvc_decoder_matches_encoder_intra"] = all(d["decoder_matches_encoder_intra"] for d in encoded)
        checks["dcvc_encode_and_decode_on_one_gpu"] = all(d["same_gpu"] for d in encoded)
        envs = [d["decode"]["environment"] for d in dcvc]
        checks["dcvc_on_ada_or_a6000_with_matching_extension"] = bool(envs) and all(
            ("6000 Ada" in e["device_name"] and e["extension"]["variant"] == "sm89")
            or ("A6000" in e["device_name"] and e["extension"]["variant"] == "sm80") for e in envs)
        kernels = [d["encode"].get("kernels") for d in encoded if d["encode"].get("kernels")]
        checks["dcvc_cuda_kernels_ran"] = bool(kernels) and all(k["kernel_launches"] > 0 for k in kernels)
        checks["dcvc_codec_time_recorded"] = bool(encoded) and all(
            d.get("model_encode_seconds") and all(d.get("model_decode_seconds") or [None]) for d in encoded)
    else:
        coded = [p["codec_record"] for p in points if "encode_command" in p["codec_record"]]
        threads = result["config"]["threads"]
        checks["svtav1_single_keyframe_crf_commands"] = all(
            "--keyint" in c["encode_command"] and "--crf" in c["encode_command"] for c in coded)
        checks["svtav1_confined_to_its_cores"] = bool(coded) and all(
            c.get("cpus") and len(c["cpus"]) == threads and c.get("encode_cpu_seconds") is not None for c in coded)
        checks.update(roi_checks(rows))
    return checks


def roi_checks(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """With ROI variants, every map covers the frame grid and each nonzero offset widens the
    foreground-over-background PSNR margin of the same item and CRF without ROI, on average."""
    roi_points = [p for r in rows for p in r["points"] if (p.get("variant_config") or {}).get("roi_offset")]
    if not roi_points:
        return {}
    gains = []
    for r in rows:
        base = {p["point"]: p for p in r["points"] if not (p.get("variant_config") or {}).get("roi_offset")}
        for p in r["points"]:
            offset = (p.get("variant_config") or {}).get("roi_offset")
            if offset and p["point"] in base:
                def margin(q: dict[str, Any]) -> float:
                    m = q["summary"]["mask_sets"]["visor_dense"]
                    return float(m["psnr_fg"] - m["psnr_bg"])
                gains.append(margin(p) - margin(base[p["point"]]))
    return {
        "roi_maps_cover_the_block_grid": all(p["codec_record"]["roi"]["grid"] == [17, 30] for p in roi_points),
        "roi_raises_foreground_relative_to_background": bool(gains) and float(np.mean(gains)) > 0.0,
    }


def command_validate(args: argparse.Namespace) -> int:
    checks = validate_complexity(stage_dir()) if args.kind == "complexity" else validate(stage_dir())
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report

def bd_rate(rate_a: list[float], quality_a: list[float], rate_b: list[float], quality_b: list[float]) -> float | None:
    """Bjontegaard delta rate of B against A (%), cubic fit of log-rate on quality, over the common quality range only."""
    if min(len(rate_a), len(rate_b)) < 4:
        return None
    low, high = max(min(quality_a), min(quality_b)), min(max(quality_a), max(quality_b))
    if high <= low:
        return None
    fits = [np.polyfit(q, np.log(r), 3) for q, r in ((quality_a, rate_a), (quality_b, rate_b))]
    integrals = [np.polyval(np.polyint(f), high) - np.polyval(np.polyint(f), low) for f in fits]
    return float((math.exp((integrals[1] - integrals[0]) / (high - low)) - 1.0) * 100.0)


def merge_results(paths: list[str], variant: str = "") -> dict[str, Any]:
    """One codec's jobs as one result: items concatenated, configuration identical, only the points
    of ``variant`` (B2's own configuration when empty), curves recomputed."""
    parts = [json.loads(Path(p).read_text()) for p in paths]
    for part in parts:
        for item in part["items"]:
            item["points"] = [q for q in item["points"] if q.get("variant", "") == variant]
    first = parts[0]
    for part in parts[1:]:
        if part["codec"] != first["codec"] or part["config"]["points"] != first["config"]["points"] or \
                part["config"]["structure"] != first["config"]["structure"] or part["config"]["preset"] != first["config"]["preset"]:
            raise SystemExit(f"cannot merge results with different codec configurations: {paths}")
    items = [item for part in parts for item in part["items"]]
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise SystemExit("an item appears in more than one result")
    return {**first, "items": items, "curves": curves(items, sorted(first["masks"])), "merged_from": paths}


PLOT_LABELS = {"svtav1": "SVT-AV1 preset 4", "dcvc": "DCVC-UF"}
PLOT_METRICS = (("wpsnr", "Weighted PSNR (dB), 0.7 fg + 0.3 bg"), ("psnr_fg", "Foreground PSNR (dB)"),
                ("vmaf", "VMAF"), ("wlpips", "Weighted LPIPS (lower is better)"))


def plot_report(report: dict[str, Any], target: Path, mask_set: str = "visor_dense") -> None:
    """Rate against quality per video type: means over items at each rate point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    groups = ("EK-100", "EK-55")
    fig, axes = plt.subplots(len(groups), len(PLOT_METRICS), figsize=(16, 7.5), constrained_layout=True)
    for r, group in enumerate(groups):
        for c, (key, label) in enumerate(PLOT_METRICS):
            ax = axes[r][c]
            for codec, codec_curves in report["curves"].items():
                rows = codec_curves[group][mask_set]
                ax.plot([row["kbps"] for row in rows], [row[key] for row in rows], "o-", ms=4,
                        label=f"{PLOT_LABELS.get(codec, codec)} (n={rows[0]['items']})")
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator([300, 500, 1000, 2000, 3000]))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="both", alpha=0.3)
            ax.set_ylabel(f"{group}\n{label}" if c == 0 else label)
            if r == len(groups) - 1:
                ax.set_xlabel("Rate (kbps, mean over items)")
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"B2 on VISOR evaluation set v2 (240-frame windows), masks: {mask_set}")
    fig.savefig(target, dpi=130)
    plt.close(fig)


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None

    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        out = np.empty(len(values))
        out[order] = np.arange(len(values))
        return out

    a, b = ranks(x), ranks(y)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def complexity_correlation(bd: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    """Spearman correlation, over items, between each BD-rate and each complexity measure."""
    measures: dict[str, dict[str, float]] = {}
    for path in paths:
        for item in json.loads(Path(path).read_text())["items"]:
            measures[item["id"]] = {k: v for k, v in item["means"].items() if v is not None}
    out: dict[str, Any] = {}
    for mask_set, metrics in bd.items():
        for metric, record in metrics.items():
            ids = [i for i, v in record["per_item"].items() if v is not None and i in measures]
            keys = sorted({k for i in ids for k in measures[i]})
            out[f"{mask_set}/{metric}"] = {
                "items": len(ids),
                "spearman": {k: spearman([record["per_item"][i] for i in ids if k in measures[i]],
                                         [measures[i][k] for i in ids if k in measures[i]]) for k in keys},
            }
    return out


def command_report(args: argparse.Namespace) -> int:
    sources = parse_named(args.result)
    results = {}
    for name, spec in sources.items():
        paths, _, variant = spec.partition("@")
        results[name] = merge_results(paths.split(","), variant)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(results)
    mask_sets = sorted({m for r in results.values() for m in r["masks"]})
    report: dict[str, Any] = {"inputs": {n: sources[n] for n in names}, "curves": {}, "bd_rate": {},
                              "items": {n: len(r["items"]) for n, r in results.items()}}
    for name, result in results.items():
        report["curves"][name] = result["curves"]
    if len(names) == 2:
        a, b = names
        items_a = {i["id"]: i for i in results[a]["items"]}
        items_b = {i["id"]: i for i in results[b]["items"]}
        for mask_set in mask_sets:
            for metric in ("wpsnr", "psnr_frame", "vmaf"):
                per_item = {}
                for item_id in sorted(set(items_a) & set(items_b)):
                    rows: list[tuple[list[float], list[float]]] = []
                    for item in (items_a[item_id], items_b[item_id]):
                        pairs = sorted((p["kbps"], flat_summary(p["summary"], mask_set)[metric]) for p in item["points"])
                        kept = [(r, q) for r, q in pairs if q is not None]
                        rows.append(([r for r, _ in kept], [float(q) for _, q in kept]))
                    per_item[item_id] = bd_rate(rows[0][0], rows[0][1], rows[1][0], rows[1][1])
                types = {i["id"]: i["video_type"] for i in results[a]["items"]}
                summary = {}
                for group in ("all", "EK-100", "EK-55"):
                    values = [v for k, v in per_item.items() if v is not None and (group == "all" or types[k] == group)]
                    summary[group] = {"items_with_overlap": len(values), "mean": float(np.mean(values)) if values else None,
                                      "median": float(np.median(values)) if values else None}
                report["bd_rate"].setdefault(mask_set, {})[metric] = {
                    "of": b, "against": a, "per_item": per_item, "summary": summary,
                    "note": "positive: more rate than the anchor at equal quality; cubic fit over each item's common range",
                }
    if args.complexity:
        report["complexity_correlation"] = complexity_correlation(report["bd_rate"], args.complexity.split(","))
    write_json(out / "b2-report.json", report)
    if args.plot:
        plot_report(report, out / "b2-rd.png")
    print(json.dumps({k: v for k, v in report.items() if k != "curves"}, indent=1)[:4000])
    return 0


# ----------------------------------------------------------------- complexity (secondary study)

def vca_record(vca: str) -> dict[str, Any]:
    done = subprocess.run([vca, "--help"], capture_output=True, text=True, timeout=30, env=quality.host_tool_env(vca))
    lines = (done.stdout or done.stderr).strip().splitlines()
    real = os.path.realpath(vca)
    return {"path": vca, "real_path": real, "sha256": file_sha256(Path(real)), "version": lines[0] if lines else None}


def vca_command(vca: str, source: Path, out: Path, *, width: int, height: int, fps: float, threads: int) -> list[str]:
    return [vca, "--input", str(source), "--input-res", f"{width}x{height}", "--input-fps", f"{fps:.6f}",
            "--input-depth", "8", "--input-csp", "420", "--complexity-csv", str(out), "--threads", str(threads)]


def read_csv_columns(path: Path) -> dict[str, list[float]]:
    import csv

    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            try:
                columns.setdefault(key.strip(), []).append(float(value))
            except (TypeError, ValueError):
                continue
    return columns


def siti(ffmpeg: str, source: Path, log_path: Path, *, width: int, height: int, fps: Fraction) -> dict[str, list[float]]:
    """ITU-T P.910 spatial and temporal information per frame (ffmpeg's ``siti`` filter)."""
    command = [ffmpeg, "-hide_banner", "-nostats", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p",
               "-s", f"{width}x{height}", "-r", f"{fps.numerator}/{fps.denominator}", "-i", str(source),
               "-vf", f"siti,metadata=mode=print:file={log_path}", "-f", "null", "-"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if done.returncode:
        raise RuntimeError(f"siti failed ({done.returncode}):\n{done.stderr[-2000:]}")
    out: dict[str, list[float]] = {"si": [], "ti": []}
    for line in log_path.read_text().splitlines():
        for key in out:
            if line.startswith(f"lavfi.siti.{key}="):
                out[key].append(float(line.split("=", 1)[1]))
    return out


def region_complexity(source: Path, clip: ClipMasks, frames: int, width: int, height: int) -> dict[str, list[float | None]]:
    """Per frame on luma: mean gradient magnitude (Sobel) and mean absolute change from the
    previous frame, inside the foreground and in the background."""
    import cv2

    out: dict[str, list[float | None]] = {k: [] for k in ("spatial_fg", "spatial_bg", "temporal_fg", "temporal_bg")}
    previous = None
    for index in range(frames):
        luma = quality.read_frames(source, width, height, index, 1)[0, :height].astype(np.float32)
        fg = clip.foreground(index)
        gradient = np.hypot(cv2.Sobel(luma, cv2.CV_32F, 1, 0), cv2.Sobel(luma, cv2.CV_32F, 0, 1))
        change = np.abs(luma - previous) if previous is not None else None
        for region, mask in (("fg", fg), ("bg", ~fg)):
            out[f"spatial_{region}"].append(float(gradient[mask].mean()) if mask.any() else None)
            out[f"temporal_{region}"].append(float(change[mask].mean()) if change is not None and mask.any() else None)
        previous = luma
    return out


def complexity_item(args: argparse.Namespace, item: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "complexity" / item["id"]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    videos = {Path(v).stem: Path(v) for v in args.video}
    frames = min(args.frames, int(item["frames"]))
    archive = Path(args.archive)
    source = decode_window(videos[item["video"]], item, frames, archive, work / "source.yuv", args.threads)
    path = Path(source["path"])
    width, height = source["width"], source["height"]
    fps = Fraction(item["fps"]).limit_denominator(1001)
    vca_csv = work / "vca.csv"
    command = vca_command(args.vca, path, vca_csv, width=width, height=height, fps=item["fps"], threads=args.threads)
    done = subprocess.run(command, capture_output=True, text=True, timeout=1800, env=quality.host_tool_env(args.vca))
    if done.returncode:
        raise RuntimeError(f"VCA failed ({done.returncode}):\n{done.stderr[-2000:]}")
    clip = ClipMasks.load(mask_root(Path(dict(args.masks)[args.mask_set]), item["id"]))
    record = {
        "id": item["id"], "video_type": video_type(item), "frames": frames, "source": source,
        "vca": {"command": command, "per_frame": read_csv_columns(vca_csv)},
        "siti": siti(args.ffmpeg, path, work / "siti.txt", width=width, height=height, fps=fps),
        "regions": region_complexity(path, clip, frames, width, height), "mask_set": args.mask_set,
    }
    record["means"] = {
        **{f"vca_{k}": mean(list(v)) for k, v in record["vca"]["per_frame"].items() if k.lower() not in ("poc", "frame")},
        **{f"siti_{k}": mean(list(v)) for k, v in record["siti"].items()},
        **{k: mean(v) for k, v in record["regions"].items()},
    }
    record["seconds"] = round(time.time() - started, 2)
    shutil.rmtree(work)
    return record


def _complexity_job(payload: tuple[argparse.Namespace, dict[str, Any]]) -> dict[str, Any]:
    return complexity_item(*payload)


def command_complexity(args: argparse.Namespace) -> int:
    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    workers = max(1, min(len(items), allowance // max(1, args.threads)))
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for row in pool.map(_complexity_job, [(args, item) for item in items]):
            rows.append(row)
            progress(len(rows))
    write_json(stage_dir() / "complexity.json", {
        "eval_set": {"path": args.eval_set, "sha256": file_sha256(Path(args.eval_set))},
        "tools": {"vca": vca_record(args.vca), "ffmpeg": svtav1.tool(args.ffmpeg)},
        "frames": args.frames, "items": rows,
    })
    return 0


def validate_complexity(stage: Path) -> dict[str, bool]:
    result = json.loads((stage / "complexity.json").read_text())
    rows = result["items"]
    return {
        "items_processed": bool(rows),
        "every_sparse_jpeg_matches_its_decoded_frame": all(g["holds"] for r in rows for g in r["source"]["jpeg_gate"]),
        "vca_has_a_value_per_frame": all(
            any(len(v) == r["frames"] for v in r["vca"]["per_frame"].values()) for r in rows),
        "siti_has_a_value_per_frame": all(len(r["siti"]["si"]) == r["frames"] for r in rows),
        "regions_have_a_value_per_frame": all(len(v) == r["frames"] for r in rows for v in r["regions"].values()),
        "measures_finite": all(v is None or math.isfinite(v) for r in rows for v in r["means"].values()),
    }


# ----------------------------------------------------------------- choose (pilot decision rule)

#: Rate cap of the pilot rule: a third of the lowest pilot source bitrate (9.7 Mbps).
CAP_KBPS = 3200.0
TARGETS = 6
MIN_RANGE_DB = 3.0
MIN_POINTS = 4
STRUCTURE_GAIN = -3.0


def mean_curve(result: dict[str, Any], mask_set: str = "visor_dense", variant: str = "") -> list[tuple[float, float, float]]:
    """(point, mean kbps, mean weighted PSNR) over all items, by rate."""
    rows = [r for r in result["curves"]["all"][mask_set] if r.get("variant", "") == variant]
    return sorted(((r["point"], r["kbps"], r["wpsnr"]) for r in rows), key=lambda r: r[1])


def choose_points(curves: dict[str, list[tuple[float, float, float]]]) -> dict[str, Any]:
    """The pilot rule of docs/experiments.md (B2): rate points per codec over the common quality range."""
    def capped_high(curve: list[tuple[float, float, float]]) -> float:
        under = [q for _, kbps, q in curve if kbps < CAP_KBPS]
        return max(under) if under else min(q for _, _, q in curve)

    low = max(min(q for _, _, q in c) for c in curves.values())
    high = min(capped_high(c) for c in curves.values())
    targets = [low + (high - low) * i / (TARGETS - 1) for i in range(TARGETS)]
    chosen: dict[str, Any] = {}
    for codec, curve in curves.items():
        picks: list[float] = []
        for target in targets:
            point = min(curve, key=lambda r: abs(r[2] - target))[0]
            if point not in picks:
                picks.append(point)
        below = [r for r in curve if r[2] < low]
        extra = max(below, key=lambda r: r[2])[0] if below else None
        if extra is not None and extra not in picks:
            picks.append(extra)
        chosen[codec] = sorted(picks)
    return {
        "range_db": [round(low, 3), round(high, 3)], "targets_db": [round(t, 3) for t in targets],
        "points": chosen,
        "refine_first": high - low < MIN_RANGE_DB or any(len(p) < MIN_POINTS for p in chosen.values()),
    }


def command_choose(args: argparse.Namespace) -> int:
    results = {name: json.loads(Path(path).read_text()) for name, path in parse_named(args.result).items()}
    structures = {n: r for n, r in results.items() if r["codec"] == "dcvc"}
    anchor = next(r for r in results.values() if r["codec"] == "svtav1")
    decision: dict[str, Any] = {"inputs": parse_named(args.result)}
    chosen_structure = "hts"
    if {"hts", "htl"} <= {r["config"]["structure"] for r in structures.values()}:
        by = {r["config"]["structure"]: {i["id"]: i for i in r["items"]} for r in structures.values()}
        per_item = {}
        for item_id in sorted(set(by["hts"]) & set(by["htl"])):
            pairs = []
            for structure in ("hts", "htl"):
                pts = sorted(by[structure][item_id]["points"], key=lambda p: p["kbps"])
                pairs.append(([p["kbps"] for p in pts], [p["summary"]["mask_sets"]["visor_dense"]["wpsnr"] for p in pts]))
            per_item[item_id] = bd_rate(pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])
        values = [v for v in per_item.values() if v is not None]
        gain = float(np.mean(values)) if values else None
        chosen_structure = "htl" if gain is not None and gain < STRUCTURE_GAIN else "hts"
        decision["structure"] = {"htl_vs_hts_bd_rate_wpsnr_per_item": per_item, "mean": gain,
                                 "threshold": STRUCTURE_GAIN, "chosen": chosen_structure}
    dcvc = next(r for r in structures.values() if r["config"]["structure"] == chosen_structure)
    decision["rate_points"] = choose_points({"svtav1": mean_curve(anchor), "dcvc": mean_curve(dcvc)})
    decision["curves"] = {"svtav1": mean_curve(anchor), "dcvc": mean_curve(dcvc)}
    print(json.dumps(decision, indent=1))
    if args.out:
        write_json(Path(args.out), decision)
    return 0


# ----------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--codec", choices=sorted(CODECS), required=True)
    run.add_argument("--eval-set", required=True)
    run.add_argument("--archive", required=True, help="extracted B1 archive: frame_mapping.json, annotations/, dense/, rgb_frames/")
    run.add_argument("--masks", action="append", nargs=2, metavar=("NAME", "DIR"), required=True,
                     help="a mask set: DIR holds <item>/masks.rle")
    run.add_argument("--mask-record", action="append", nargs=2, metavar=("NAME", "JSON"),
                     help="JSON listing each item's mask_sha256 and masks_rle_sha256")
    run.add_argument("--video", action="append", required=True)
    run.add_argument("--items", required=True, help="all, or comma-separated item ids")
    run.add_argument("--frames", type=int, required=True)
    run.add_argument("--points", required=True, help="comma-separated CRF (svtav1) or QP (dcvc) values")
    run.add_argument("--threads", type=int, default=8, help="threads per item (encoder, decoder, metrics, VMAF)")
    run.add_argument("--lpips-backbone", required=True)
    run.add_argument("--vmaf-ffmpeg", default="/opt/local/bin/ffmpeg")
    run.add_argument("--streams", help="score published streams under DIR/<codec>/<item>/<point>/ instead of encoding")
    run.add_argument("--preset", default="4", help="SVT-AV1 preset, or a comma-separated list")
    run.add_argument("--roi-offset", default="0", help="SVT-AV1 ROI quantizer offset, or a list: negative lowers foreground blocks, positive raises background blocks (0: no ROI)")
    run.add_argument("--roi-mask-set", default="visor_dense")
    run.add_argument("--enable-tf", type=int, choices=(0, 1, 2), help="SVT-AV1 temporal filtering (default: the encoder's)")
    run.add_argument("--dcvc-range", default="full", help="DCVC-UF input range: full, limited, or both")
    run.add_argument("--metric-device", choices=("cpu", "cuda"), help="default: cuda for DCVC-UF, cpu for SVT-AV1")
    run.add_argument("--workers", type=int, default=0, help="SVT-AV1 items in parallel (0: allowance / threads)")
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--decoder", default="dav1d")
    run.add_argument("--structure", choices=("ld", "hts", "htl"), default="hts")
    run.add_argument("--image-ckpt")
    run.add_argument("--video-ckpt")
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--codec", choices=sorted(CODECS), help="informational; the result names its codec")
    validate_parser.add_argument("--kind", choices=("b2", "complexity"), default="b2")
    complexity = sub.add_parser("complexity", help="spatial and temporal complexity per frame and region (CPU)")
    complexity.add_argument("--eval-set", required=True)
    complexity.add_argument("--archive", required=True)
    complexity.add_argument("--masks", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    complexity.add_argument("--mask-set", default="visor_dense")
    complexity.add_argument("--video", action="append", required=True)
    complexity.add_argument("--items", required=True)
    complexity.add_argument("--frames", type=int, required=True)
    complexity.add_argument("--threads", type=int, default=4)
    complexity.add_argument("--vca", default="/opt/local/bin/vca")
    complexity.add_argument("--ffmpeg", default="ffmpeg")
    report = sub.add_parser("report")
    report.add_argument("--result", action="append", required=True, help="NAME=b2.json[,b2.json...][@VARIANT] (one codec, merged; one variant, B2's own if omitted); with two names, the second is compared against the first")
    report.add_argument("--out", required=True)
    report.add_argument("--complexity", help="complexity.json files (comma-separated) to correlate with the BD-rates")
    report.add_argument("--plot", action="store_true", help="also draw b2-rd.png (needs matplotlib)")
    choose = sub.add_parser("choose", help="apply the pilot rule: DCVC-UF structure and rate points")
    choose.add_argument("--result", action="append", required=True, help="NAME=pilot b2.json (SVT-AV1, DCVC-UF HT-S, HT-L)")
    choose.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "run" and args.codec == "dcvc" and not (args.image_ckpt and args.video_ckpt):
        parser.error("dcvc needs --image-ckpt and --video-ckpt")
    return {"run": command_run, "validate": command_validate, "report": command_report, "choose": command_choose,
            "complexity": command_complexity}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
