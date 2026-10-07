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


# ----------------------------------------------------------------- codecs

def code_svtav1(args: argparse.Namespace, source: dict[str, Any], fps: Fraction, point: float, work: Path,
                stream: Path | None) -> dict[str, Any]:
    width, height, frames = source["width"], source["height"], source["frames"]
    if stream is None:
        record = svtav1.code(Path(source["path"]), work, width=width, height=height, fps=fps, frames=frames, crf=point,
                             preset=args.preset, threads=args.threads, full_range=source["full_range"],
                             encoder=args.encoder, decoder=args.decoder)
    else:
        decoded = work / "decoded.yuv"
        command = svtav1.decode_command(args.decoder, stream, decoded, threads=args.threads)
        seconds = svtav1.run(command, 3600)
        data = stream.read_bytes()
        sizes = svtav1.ivf_frames(data)
        record = {"stream": str(stream), "decoded": str(decoded), "stream_sha256": hashlib.sha256(data).hexdigest(),
                  "file_bytes": len(data), "payload_bytes": sum(sizes), "temporal_units": len(sizes),
                  "decoded_frames": decoded.stat().st_size / quality.frame_bytes(width, height),
                  "decode_command": command, "decode_seconds": round(seconds, 3), "reused_stream": True}
    record["rate_bytes"] = record["payload_bytes"]
    record["stream_frames"] = record["temporal_units"]
    return record


def code_dcvc(args: argparse.Namespace, source: dict[str, Any], point: float, work: Path, stream: Path | None,
              profile: bool) -> dict[str, Any]:
    from src.codecs.dcvc_uf_worker import dcvc_command

    if int(point) != point:
        raise SystemExit(f"DCVC-UF QP must be an integer, got {point}")
    container = stream or work / "stream.psdc"
    plan = {
        "structure": args.structure, "qp": int(point), "frame_count": source["frames"], "height": source["height"],
        "width": source["width"], "src_type": "yuv420", "frames_file": source["path"], "container": str(container),
        "out_file": str(work / "decoded.yuv"), "image_ckpt": args.image_ckpt, "image_sha256": args.image_sha256,
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
    data = container.read_bytes()
    decode = reports["decode"]
    record = {
        "stream": str(container), "decoded": str(work / "decoded.yuv"), "stream_sha256": hashlib.sha256(data).hexdigest(),
        "file_bytes": len(data), "rate_bytes": len(data), "stream_frames": decode["frame_count"],
        "decoded_frames": (work / "decoded.yuv").stat().st_size / quality.frame_bytes(source["width"], source["height"]),
        "commands": commands, "plan": plan,
        "decode": {k: v for k, v in decode.items() if k != "passes"},
        "decode_seconds": [p["decode_seconds"] for p in decode["passes"]],
        "deterministic": decode["deterministic"], "reused_stream": stream is not None,
    }
    if "encode" in reports:
        encode = reports["encode"]
        record["encode"] = {k: v for k, v in encode.items() if k != "nals"}
        record["nal_count"] = len(encode["nals"])
        record["decoder_matches_encoder_intra"] = decode["passes"][0]["frame_sha256"][0] == encode["i_recon_sha256"]
        record["same_gpu"] = encode["environment"]["cuda_visible_devices"] == decode["environment"]["cuda_visible_devices"]
        record["encode_seconds"] = encode["encode_seconds"]
    return record


# ----------------------------------------------------------------- one item

_LPIPS: dict[str, Any] = {}


def lpips_net(backbone: str, device: str) -> Any:
    key = f"{backbone}|{device}"
    if key not in _LPIPS:
        _LPIPS[key] = quality.load_lpips(Path(backbone), device)
    return _LPIPS[key]


def run_item(args: argparse.Namespace, item: dict[str, Any], first_item: bool) -> dict[str, Any]:
    import torch

    torch.set_num_threads(max(1, args.threads))
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
    source = decode_window(videos[item["video"]], item, frames, archive, work / "source.yuv", args.threads)
    clips, mask_info = load_mask_sets(item, frames, named(args.masks), named(args.mask_record), archive)
    regions = frame_regions(clips, frames)
    fps = Fraction(item["fps"]).limit_denominator(1001)
    device = "cuda" if args.codec == "dcvc" else "cpu"
    net = lpips_net(args.lpips_backbone, device)
    rows = []
    cross_device = None
    for number, point in enumerate(parse_points(args.points)):
        name = point_name(point)
        point_work = work / f"p{name}"
        point_work.mkdir()
        stream = None
        if args.streams:
            stream = Path(args.streams) / args.codec / item["id"] / name / STREAM_NAMES[args.codec]
        t0 = time.time()
        if args.codec == "svtav1":
            coded = code_svtav1(args, source, fps, point, point_work, stream)
        else:
            coded = code_dcvc(args, source, point, point_work, stream, profile=first_item and number == 0)
        decoded = Path(coded["decoded"])
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
        kept = publish / "streams" / args.codec / item["id"] / name
        kept.mkdir(parents=True, exist_ok=True)
        if stream is None:
            shutil.copy2(coded["stream"], kept / STREAM_NAMES[args.codec])
        write_json(publish / "frames" / item["id"] / f"{args.codec}-{name}.json", {
            "item": item["id"], "codec": args.codec, "point": point, "frames": per_frame, "vmaf": vmaf["per_frame"],
        })
        rows.append({
            "point": point, "rate_bytes": coded["rate_bytes"], "file_bytes": coded["file_bytes"],
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
            points = sorted({p["point"] for i in members for p in i["points"]})
            for point in points:
                matched = [(i, p) for i in members for p in i["points"] if p["point"] == point]
                values = [flat_summary(p["summary"], mask_set) for _, p in matched]
                row: dict[str, Any] = {"point": point, "items": len(matched)}
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
        workers = max(1, min(len(items), allowance // max(1, args.threads)))
        # Spawned, not forked: the parent has already started torch's thread pools.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(_run_item_job, [(args, item, n == 0) for n, item in enumerate(items)]):
                rows.append(row)
                progress(len(rows))
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
            "preset": args.preset if args.codec == "svtav1" else None,
            "structure": args.structure if args.codec == "dcvc" else None,
            "threads": args.threads, "workers": workers, "streams": args.streams,
            "foreground_weight": FOREGROUND_WEIGHT, "metric_device": "cuda" if args.codec == "dcvc" else "cpu",
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
        "every_point_per_item": all(len(r["points"]) == len(result["config"]["points"]) for r in rows),
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
        ordered = sorted(r["points"], key=lambda p: p["point"], reverse=not rises)
        rates = [p["rate_bytes"] for p in ordered]
        quality_values = [p["summary"]["frame"]["psnr"] for p in ordered]
        monotone.append(all(a < b for a, b in zip(rates, rates[1:])) and all(a < b for a, b in zip(quality_values, quality_values[1:])))
    checks["rate_and_quality_rise_together"] = all(monotone)
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
        cross = [r["cross_device"] for r in rows if r.get("cross_device")]
        checks["metrics_agree_between_cpu_and_gpu"] = bool(cross) and all(
            c["max_abs_diff"]["psnr"] == 0.0 and c["max_abs_diff"]["lpips"] < 1e-3 and c["max_abs_diff"]["ms_ssim"] < 1e-4 for c in cross)
    else:
        checks["svtav1_single_keyframe_crf_commands"] = all(
            "--keyint" in p["codec_record"]["encode_command"] and "--crf" in p["codec_record"]["encode_command"]
            for p in points if "encode_command" in p["codec_record"])
    return checks


def command_validate(args: argparse.Namespace) -> int:
    checks = validate(stage_dir())
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


def command_report(args: argparse.Namespace) -> int:
    results = {name: json.loads(Path(path).read_text()) for name, path in parse_named(args.result).items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(results)
    mask_sets = sorted({m for r in results.values() for m in r["masks"]})
    report: dict[str, Any] = {"inputs": {n: parse_named(args.result)[n] for n in names}, "curves": {}, "bd_rate": {}}
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
    write_json(out / "b2-report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "curves"}, indent=1)[:4000])
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
    run.add_argument("--preset", type=int, default=4)
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--decoder", default="dav1d")
    run.add_argument("--structure", choices=("ld", "hts", "htl"), default="hts")
    run.add_argument("--image-ckpt")
    run.add_argument("--video-ckpt")
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--codec", choices=sorted(CODECS), required=True)
    report = sub.add_parser("report")
    report.add_argument("--result", action="append", required=True, help="NAME=b2.json; with two, the second is compared against the first")
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "run" and args.codec == "dcvc" and not (args.image_ckpt and args.video_ckpt):
        parser.error("dcvc needs --image-ckpt and --video-ckpt")
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
