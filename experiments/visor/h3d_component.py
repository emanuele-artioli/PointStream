"""PLAN step H3: H3d's component, the super-resolution network delivered per video, per person or with the decoder
(docs/experiments.md, 2026-10-12, "H3d component").

    python -m experiments.visor.h3d_component footage --target ID --arm V|K|G --video V.MP4 ... --eval-set JSON \\
        --sam-checkpoint PT [--segments N]
    python -m experiments.visor.h3d_component generic --footage TAR --video V.MP4 ... --point NAME,... --steps N \\
        --image-ckpt PTH --video-ckpt PTH --lpips-backbone PTH
    python -m experiments.visor.h3d_component score <the oracle's window arguments> --footage TAR --generic TAR ... \\
        --point NAME --steps N
    python -m experiments.visor.h3d_component validate

``footage`` places 48-frame segments content-blind (`choose_segments`), decodes them, and masks R with SAM 3.1
("hand" plus "arm", G1's analysis size); it publishes the masks and each segment's decoded-plane sha256, not the
frames, which later jobs decode again and check. ``generic`` fits G's network (`SIZES['L']`) on the other
participants' footage, one per point. ``score`` codes one item's window at one point (the oracle's code), fits V at
each size and K on their footage coded at the same point, applies G's network, and scores every output as the
oracle; the unit is ``component/<item>@<point>``. The rate and the rule are applied by ``h3d report --component``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from experiments.visor.h2 import load_units, restore_units, save_unit
from experiments.visor.h3 import rgb_memmap
from experiments.visor.h3b import scaled_size
from experiments.visor.h3c import CROP, decode_windows, downscale_planes, load_window
from experiments.visor import h3d

#: Network sizes (blocks, channels): L is G5d's (the oracle's); M and S only for V, whose bits are billed.
SIZES: dict[str, tuple[int, int]] = {"L": (8, 64), "M": (4, 48), "S": (4, 24)}
ARMS = ("V", "K", "G")
FOOTAGE: dict[str, Any] = {"segments": 24, "length": 48, "edge_s": 5.0, "guard_s": 5.0, "max_videos": 6,
                           "prompts": {"hand": "hand", "arm": "arm"}}
ANALYSIS = (960, 540)  # G1's analysis size (`experiments.background.camera.ANALYSIS_SIZE`)
VERIFIED = ("1920x1080@50", "1920x1080@59.94")  # the evaluation's classes (B1)
EXCLUDED = ("P01", "P02", "P03", "P09")  # the pilot items' participants: never in G's footage
GENERIC = "generic"  # G's footage serves every item

#: The decision rule, fixed before any fleet run (docs/experiments.md, H3d component entry).
COMPONENT: dict[str, Any] = {
    "stands": "an arm stands by the oracle's rule against the bar (h3d.DECISION): mean BD-rate on LPIPS_R <= -10% with "
              "at least 3 of 4 items below 0",
    "rate": {"V": "rate on R plus the network's fp16 bits over the video's duration", "K": "rate on R", "G": "rate on R"},
    "v_size": "V at its best size (lowest mean BD-rate)",
    "prefer": "G if it stands; K instead if it also stands and beats G by 10% (mean BD-rate of K against G); V only if it "
              "beats both standing shared arms by 10%; if no arm stands, H3d is not built",
    "beats_by": -0.10,
}


def participant(video: str) -> str:
    return video.split("_")[0]


def sha_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def choose_segments(target: str, arm: str, videos: dict[str, tuple[float, int]], count: int, length: int,
                    exclude: dict[str, tuple[int, int]] | None = None) -> list[dict[str, Any]]:
    """``count`` non-overlapping segments of ``length`` frames from ``videos`` (name -> (fps, frames)), dealt round-robin
    over the videos in hash order, each placed by sha256 of the target, the arm, the video and its number, at least
    `FOOTAGE['edge_s']` from a video's ends and outside ``exclude`` (name -> frame range, guard included); fixed before
    any run and blind to content (as `experiments.background.g5c.choose_segments`)."""
    order = sorted(videos, key=lambda v: sha_int(f"{h3d.SEED}:{target}:{arm}:{v}"))
    taken: dict[str, list[tuple[int, int]]] = {v: [] for v in order}
    blocked = {v: [r] for v, r in (exclude or {}).items()}
    out: list[dict[str, Any]] = []
    k = 0
    while len(out) < count and k < 50 * count:
        video = order[k % len(order)]
        k += 1
        fps, frames = videos[video]
        edge = int(round(FOOTAGE["edge_s"] * fps))
        room = frames - 2 * edge - length
        if room <= 0:
            continue
        for attempt in range(200):
            first = edge + sha_int(f"{h3d.SEED}:{target}:{arm}:{video}:{len(taken[video])}:{attempt}") % room
            last = first + length - 1
            if all(last < a or first > b for a, b in taken[video] + blocked.get(video, [])):
                taken[video].append((first, last))
                out.append({"video": video, "fps": fps, "first_index": first, "last_index": last})
                break
    if len(out) < count:
        raise SystemExit(f"{target} {arm}: only {len(out)} of {count} segments fit in {sorted(videos)}")
    return out


def video_class(path: Path) -> tuple[str, float, int]:
    """(class, nominal fps, frames) of a video, the class as `VERIFIED` names it."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.base_rate or stream.average_rate)
        seconds = float(stream.duration * stream.time_base) if stream.duration else float(container.duration) / 1e6
        return f"{stream.width}x{stream.height}@{round(rate, 2):g}", rate, int(seconds * rate)


def decode_segment(task: dict[str, Any]) -> dict[str, Any]:
    """A segment's 4:2:0 planes (to ``planes``, if given) and their sha256; with ``jpeg_dir``, its frames at G1's
    analysis size as SAM 3.1's JPEG chunk. Runs in a spawned process (PyAV before torch)."""
    import cv2

    from src.segmentation import visor

    cv2.setNumThreads(max(1, task["threads"]))
    first, last = task["first_index"], task["last_index"]
    digest = hashlib.sha256()
    expected, formats, ranges = first, set(), set()
    handle = open(task["planes"], "wb") if task.get("planes") else None
    jpeg = Path(task["jpeg_dir"]) / "chunk_000000" if task.get("jpeg_dir") else None
    if jpeg:
        jpeg.mkdir(parents=True, exist_ok=True)
    try:
        for index, frame in visor.decoded_frames(task["path"], range(first, last + 1), threads=task["threads"]):
            if not first <= index <= last:
                continue
            if index != expected:
                raise RuntimeError(f"{task['path']}: decoded frame {index}, expected {expected}")
            formats.add(frame.format.name)
            ranges.add(str(getattr(frame, "color_range", None)))
            planes = np.ascontiguousarray(frame.to_ndarray())
            digest.update(planes.tobytes())
            if handle:
                handle.write(planes.tobytes())
            if jpeg:
                rgb = cv2.resize(frame.to_ndarray(format="rgb24"), ANALYSIS, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(jpeg / f"{index - first:05d}.jpg"), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            expected += 1
    finally:
        if handle:
            handle.close()
    if expected != last + 1:
        raise RuntimeError(f"{task['path']}: decoded {expected - first} of {last - first + 1} frames")
    if formats - {"yuv420p", "yuvj420p"}:
        raise RuntimeError(f"{task['path']}: decoded formats {formats} are not 8-bit 4:2:0")
    return {"yuv_sha256": digest.hexdigest(), "pix_fmt": sorted(formats), "color_range": sorted(ranges),
            "full_range": formats == {"yuvj420p"} or ranges in ({"2"}, {"JPEG"})}


def spawn_pool() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))


# ----------------------------------------------------------------- footage (SAM 3.1)


def command_footage(args: argparse.Namespace) -> int:
    from experiments.background import g1

    paths = {Path(v).stem: Path(v) for v in args.video}
    eval_set = {it["id"]: it for it in json.loads(Path(args.eval_set).read_text())["items"]}
    if (args.arm == "G") != (args.target == GENERIC):
        raise SystemExit(f"G's footage, and only G's, has the target {GENERIC!r}")
    item = eval_set.get(args.target)
    if args.arm != "G" and item is None:
        raise SystemExit(f"{args.target} is not an evaluation item")
    own = str(item["video"]) if item else ""
    if args.arm == "V" and set(paths) != {own}:
        raise SystemExit(f"V's footage is the item's own video {own}, not {sorted(paths)}")
    if args.arm == "K" and (own in paths or {participant(v) for v in paths} != {participant(own)}):
        raise SystemExit(f"K's footage is {participant(own)}'s other recordings, not {sorted(paths)}")
    if args.arm == "G" and any(participant(v) in EXCLUDED for v in paths):
        raise SystemExit(f"G's footage excludes {EXCLUDED}: {sorted(paths)}")
    classes = {v: video_class(p) for v, p in paths.items()}
    if bad := {v: c for v, (c, _, _) in classes.items() if c not in VERIFIED}:
        raise SystemExit(f"not of the evaluation's classes {VERIFIED}: {bad}")
    info = {v: (fps, n) for v, (_, fps, n) in classes.items()}
    exclude = None
    if args.arm == "V" and item:
        guard = int(round(FOOTAGE["guard_s"] * info[own][0]))
        exclude = {own: (int(item["first_video_index"]) - guard, int(item["first_video_index"]) + int(item["frames"]) + guard)}
    count = int(args.segments)
    segments = choose_segments(args.target, args.arm, info, FOOTAGE["segments"], FOOTAGE["length"], exclude)[:count]
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "footage"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    publish = scratch / "publish" / "footage"
    publish.mkdir(parents=True, exist_ok=True)
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    began = time.time()
    decoded = []
    with spawn_pool() as pool:  # every segment decoded before torch loads
        for n, seg in enumerate(segments):
            decoded.append(pool.submit(decode_segment, {**seg, "path": str(paths[seg["video"]]), "threads": allowance,
                                                         "jpeg_dir": str(work / f"seg{n}")}).result())
    from src.segmentation.sam31 import Sam31SequenceSegmenter

    segmenter = Sam31SequenceSegmenter(checkpoint_path=args.sam_checkpoint)
    provenance = segmenter.provenance("offline_bidirectional")
    masks, records = [], []
    for n, (seg, dec) in enumerate(zip(segments, decoded)):
        clip = {"masks": {"prompts": FOOTAGE["prompts"]}, "analysis": {"mode": "native"}, "fps": seg["fps"],
                "indices": list(range(seg["first_index"], seg["last_index"] + 1))}
        t0 = time.time()
        found = g1.sam_text_masks(segmenter, clip, work / f"seg{n}")
        union = np.stack([g1.to_analysis(g1.union(f, (found.height, found.width))) for f in found.frames])
        masks.append(union)
        records.append({**seg, **dec, "frames": len(union), "share_r": float(union.mean()),
                        "frames_with_r": int(union.any((1, 2)).sum()), "sam_seconds": round(time.time() - t0, 1),
                        "empty_chunks": found.meta.get("empty_chunks", [])})
        shutil.rmtree(work / f"seg{n}", ignore_errors=True)
        progress(n + 1)
    stack = np.concatenate(masks)
    unit = f"{args.target}@{args.arm}"
    arrays = {"masks": np.packbits(stack, axis=-1), "segment": np.repeat(np.arange(len(masks)), [len(m) for m in masks])}
    meta = {"unit": unit, "target": args.target, "arm": args.arm, "videos": {v: {"class": c, "fps": f, "frames": n}
            for v, (c, f, n) in classes.items()}, "segments": records, "shape": list(stack.shape), "settings": FOOTAGE,
            "sam": {"checkpoint_sha256": provenance.checkpoint_sha256, "model_revision": provenance.model_revision}}
    save_unit(scratch / "publish", "footage", unit, arrays, meta)
    write_json(stage_dir() / "h3d.json", {"command": "footage", "target": args.target, "arm": args.arm, "units": [unit],
                                          "segments": len(records), "seconds": round(time.time() - began, 1), **gpu_record()})
    shutil.rmtree(work, ignore_errors=True)
    return 0


def gpu_record() -> dict[str, Any]:
    import torch

    from experiments.audit.env_smoke import device_record

    return {"device": device_record(torch.cuda.get_device_name(0)), "gpu_capability": list(torch.cuda.get_device_capability(0))}


# ----------------------------------------------------------------- footage as training pairs


def load_footage(tars: list[str]) -> dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]]:
    return {meta["unit"]: (meta, a) for meta, a in load_units(tars, "footage")}


def footage_planes(meta: dict[str, Any], paths: dict[str, Path], out: Path) -> tuple[np.ndarray, bool]:
    """The footage's 4:2:0 planes, decoded again and checked against the footage job's sha256 per segment."""
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle, spawn_pool() as pool:
        for n, seg in enumerate(meta["segments"]):
            part = out.with_suffix(f".seg{n}")
            got = pool.submit(decode_segment, {**seg, "path": str(paths[seg["video"]]), "threads": allowance,
                                               "planes": str(part)}).result()
            if got["yuv_sha256"] != seg["yuv_sha256"]:
                raise RuntimeError(f"{meta['unit']} segment {n}: decoded planes differ from the footage job's")
            with part.open("rb") as src:
                shutil.copyfileobj(src, handle)
            part.unlink()
    full_range = {bool(s["full_range"]) for s in meta["segments"]}
    if len(full_range) != 1:
        raise RuntimeError(f"{meta['unit']}: segments differ in range")
    frames = sum(s["frames"] for s in meta["segments"])
    return np.memmap(out, np.uint8, "r", shape=(frames, 1080 * 3 // 2, 1920)), full_range.pop()


def first_segments(meta: dict[str, Any], a: dict[str, np.ndarray], count: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The footage limited to its first ``count`` segments (smokes)."""
    if count >= len(meta["segments"]):
        return meta, a
    keep = a["segment"] < count
    shape = [int(keep.sum()), *meta["shape"][1:]]
    return {**meta, "segments": meta["segments"][:count], "shape": shape}, {**a, "masks": a["masks"][keep], "segment": a["segment"][keep]}


def full_masks(meta: dict[str, Any], a: dict[str, np.ndarray]) -> np.ndarray:
    """R per footage frame at G1's analysis size (bool)."""
    n, h, w = meta["shape"]
    return np.unpackbits(a["masks"], axis=-1, count=w).astype(bool).reshape(n, h, w)


def crop_box(mask: np.ndarray, size: int = CROP) -> tuple[int, int]:
    """The top-left corner (y, x) of a ``size`` crop of the 1080p frame centred on R's box (``mask`` at analysis size)."""
    ys, xs = np.nonzero(mask)
    sy, sx = 1080 / mask.shape[0], 1920 / mask.shape[1]
    cy, cx = (ys.min() + ys.max() + 1) / 2 * sy, (xs.min() + xs.max() + 1) / 2 * sx
    return int(min(max(0, round(cy - size / 2)), 1080 - size)), int(min(max(0, round(cx - size / 2)), 1920 - size))


def training_pairs(args: argparse.Namespace, meta: dict[str, Any], a: dict[str, np.ndarray], paths: dict[str, Path],
                   value: float, scale: int, work: Path, device: str) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], dict[str, Any]]:
    """The footage coded at the point (its segments as one sequence), upscaled bicubically; ``CROP`` crops centred on R
    of the upscaled decode and the source, and R, for every footage frame with R."""
    import cv2

    planes, full_range = footage_planes(meta, paths, work / "footage.yuv")
    masks = full_masks(meta, a)
    frames = len(planes)
    sw, sh = scaled_size(1920, 1080, scale)
    source = work / "footage-small.yuv"
    with source.open("wb") as handle:
        for t in range(frames):
            handle.write(downscale_planes(np.asarray(planes[t]), 1920, 1080, sw, sh).tobytes())
    rec = code_only(args, source, (sw, sh), frames, int(value), work / "code")
    source.unlink()
    small = rgb_memmap(Path(rec["decoded"]), work / "footage-rgb.npy", frames, sw, sh, full_range, device)
    truth = rgb_memmap(work / "footage.yuv", work / "footage-truth.npy", frames, 1920, 1080, full_range, device)
    keep = [t for t in range(frames) if masks[t].any()]
    xs, ts, rs = (np.empty((len(keep), CROP, CROP, 3), np.uint8), np.empty((len(keep), CROP, CROP, 3), np.uint8), [])
    for i, t in enumerate(keep):
        y, x = crop_box(masks[t])
        up = cv2.resize(np.asarray(small[t]), (1920, 1080), interpolation=cv2.INTER_CUBIC)
        xs[i] = up[y:y + CROP, x:x + CROP]
        ts[i] = np.asarray(truth[t])[y:y + CROP, x:x + CROP]
        big = cv2.resize(masks[t].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
        rs.append(np.ascontiguousarray(big[y:y + CROP, x:x + CROP]))
    del small, truth
    for leftover in ("footage.yuv", "footage-rgb.npy", "footage-truth.npy"):
        (work / leftover).unlink(missing_ok=True)
    shutil.rmtree(work / "code", ignore_errors=True)
    return xs, ts, rs, {"frames": frames, "frames_with_r": len(keep), "bytes": rec["bytes"], "stream_sha256": rec["stream_sha256"],
                        "deterministic": rec["deterministic"], "full_range": full_range}


def code_only(args: argparse.Namespace, source: Path, size: tuple[int, int], frames: int, qp: int, work: Path) -> dict[str, Any]:
    """DCVC-UF HT-L off the shelf on ``source`` (B2's worker), encode and decode only: training data needs no rate."""
    from src.codecs.dcvc_uf_worker import dcvc_command

    work.mkdir(parents=True, exist_ok=True)
    width, height = size
    plan = {"structure": "htl", "qp": qp, "frame_count": frames, "height": height, "width": width, "src_type": "yuv420",
            "frames_file": str(source), "container": str(work / "stream.psdc"), "out_file": str(work / "decoded.yuv"),
            "image_ckpt": args.image_ckpt, "image_sha256": args.image_sha256, "video_ckpt": args.video_ckpt,
            "video_sha256": args.video_sha256, "profile": False}
    write_json(work / "plan.json", plan)
    reports = {}
    for action in ("encode", "decode"):
        command, env, cwd = dcvc_command(Path(sys.prefix), action, work / "plan.json", work / f"{action}.json")
        done = subprocess.run(command, env={**os.environ, **env}, cwd=cwd, capture_output=True, text=True, timeout=3600)
        if done.returncode:
            raise RuntimeError(f"DCVC-UF {action} failed ({done.returncode}):\n{done.stderr[-4000:]}")
        reports[action] = json.loads((work / f"{action}.json").read_text())
    decode = reports["decode"]
    # As H3c's coder: a decode whose two passes differ, or that is not the encoder's, is not data (G's ½ networks,
    # 2026-10-13: QP 21's footage decoded to noise on a shared gpu5 and was fitted before this check existed).
    if not decode["deterministic"] or decode["passes"][0]["frame_sha256"][0] != reports["encode"]["i_recon_sha256"]:
        raise RuntimeError(f"DCVC-UF QP {qp} at {width}x{height}: the decode is not the encoder's (two decodes agree: "
                           f"{decode['deterministic']})")
    count = (work / "decoded.yuv").stat().st_size // (width * height * 3 // 2)
    if count < frames:
        raise RuntimeError(f"DCVC-UF QP {qp}: decoded {count} of {frames} frames")
    return {"bytes": (work / "stream.psdc").stat().st_size, "stream_sha256": file_sha256(work / "stream.psdc"),
            "decoded": str(work / "decoded.yuv"), "deterministic": True}


def parse_point(name: str) -> tuple[float, int]:
    """``dcvc@QP/SCALE`` to (QP, scale)."""
    qp, scale = name.split("@")[1].split("/")
    point = (float(qp), int(scale))
    if point not in h3d.BASES["dcvc"]:
        raise SystemExit(f"{name} is not one of the oracle's DCVC-UF points")
    return point


def common_setup(args: argparse.Namespace) -> str:
    import torch

    args.image_sha256 = file_sha256(Path(args.image_ckpt))
    args.video_sha256 = file_sha256(Path(args.video_ckpt))
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU (DCVC-UF)")
    return "cuda"


def load_lpips_loss(path: str, device: str) -> Any:
    from src.codecs.quality import load_lpips

    net = load_lpips(Path(path), device)
    for p in net.parameters():  # a loss: gradients flow through it, not into it
        p.requires_grad_(False)
    return net


# ----------------------------------------------------------------- G's networks


def command_generic(args: argparse.Namespace) -> int:
    import torch

    device = common_setup(args)
    footage = load_footage(args.footage)
    if len(footage) != 1 or next(iter(footage.values()))[0]["arm"] != "G":
        raise SystemExit("--footage: one archive of G's footage")
    meta, a = first_segments(*next(iter(footage.values())), int(args.segments))
    paths = {Path(v).stem: Path(v) for v in args.video}
    lpips_net = load_lpips_loss(args.lpips_backbone, device)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish" / "generic"
    publish.mkdir(parents=True, exist_ok=True)
    began, rows = time.time(), []
    for n, name in enumerate(args.point.split(",")):
        value, scale = parse_point(name)
        xs, ts, rs, data = training_pairs(args, meta, a, paths, value, scale, scratch / "work", device)
        net, info = h3d.fit(xs, ts, rs, int(args.steps), lpips_net, device, *SIZES["L"])
        target = publish / f"{name.replace('@', '-').replace('/', '_')}.pt"
        torch.save(net.state_dict(), target)
        rows.append({"point": name, "network": target.name, "sha256": file_sha256(target), "fit": info, "data": data})
        write_json(publish / "generic.json", {"footage": meta["unit"], "footage_npz_sha256": meta["npz_sha256"], "rows": rows})
        progress(n + 1)
    write_json(stage_dir() / "h3d.json", {"command": "generic", "rows": rows, "seconds": round(time.time() - began, 1),
                                          "image_sha256": args.image_sha256, "video_sha256": args.video_sha256, **gpu_record()})
    return 0


def generic_network(tars: list[str], name: str, device: str) -> tuple[Any, dict[str, Any]]:
    """G's network for the point ``name``, from the generic jobs' archives (sha256 checked)."""
    import io
    import tarfile

    import torch

    from experiments.background.g5d import enhancer

    for path in tars:
        with tarfile.open(path) as tar:
            members = {m.name: m for m in tar.getmembers()}
            index = members.get("publish/generic/generic.json")
            if index is None:
                continue
            handle = tar.extractfile(index)
            assert handle is not None
            for row in json.loads(handle.read())["rows"]:
                if row["point"] != name:
                    continue
                blob = tar.extractfile(members[f"publish/generic/{row['network']}"])
                assert blob is not None
                data = blob.read()
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise RuntimeError(f"G's network for {name}: sha256 differs")
                net = enhancer(*SIZES["L"]).to(device)
                net.load_state_dict(torch.load(io.BytesIO(data), map_location=device, weights_only=True))
                return net.eval(), {"sha256": row["sha256"], "fit": row["fit"], "data": row["data"]}
    raise SystemExit(f"no G network for {name} in {tars}")


# ----------------------------------------------------------------- scoring one item at one point


def command_score(args: argparse.Namespace) -> int:
    import torch

    from experiments.visor.b2 import select_items
    from experiments.visor.h3b import load_dists
    from experiments.visor.h3c import Scorer
    from src.codecs.quality import lpips_record

    device = common_setup(args)
    value, scale = parse_point(args.point)
    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    if len(items) != 1:
        raise SystemExit("score takes one item")
    item = items[0]
    footage = load_footage(args.footage)
    arms = {u.split("@")[1]: v for u, v in footage.items() if u.split("@")[0] == item["id"]}
    if set(arms) != {"V", "K"}:
        raise SystemExit(f"--footage: V's and K's footage of {item['id']}, found {sorted(footage)}")
    paths = {Path(v).stem: Path(v) for v in args.video}
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    unit = f"{item['id']}@{args.point.replace('/', '_')}"
    restored = restore_units(publish, "component")
    began = time.time()
    if unit not in restored:
        windows = decode_windows(args, [item], int(args.frames), scratch)
        lpips_net = load_lpips_loss(args.lpips_backbone, device)
        nets = {"lpips": lpips_net, "dists": load_dists(Path(args.dists_backbone), device)}
        g_net, g_info = generic_network(args.generic, args.point, device)
        for it, decoded in windows:
            frames = int(args.frames)
            width, height = decoded["width"], decoded["height"]
            fps = Fraction(it["fps"]).limit_denominator(1001)
            work = scratch / "work"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            planes, regions, mask_info = load_window(args, it, frames, decoded)
            truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, width, height, decoded["full_range"], device)
            score = Scorer(nets, truth, regions, device)
            rgb, record = h3d.code_point(args, "dcvc", value, scale, decoded, planes, regions, frames, fps, work, device)
            upscaled = h3d.upscale_all(rgb, frames, width, height)
            del rgb
            (work / "dcvc-rgb.npy").unlink(missing_ok=True)
            shutil.rmtree(work / f"dcvc-{scale}-{value:g}", ignore_errors=True)
            rows, scores, flicker, fits = ["base"], [score(upscaled, upscale=False)], [], {}
            flicker.append(h3d.flicker_flow_free(h3d.composites(upscaled, truth, regions), truth, regions))
            data: dict[str, Any] = {}

            def add(row: str, net: Any) -> None:  # each output scored and dropped: a window is 1.5 GB
                out = h3d.apply(net, upscaled, regions, device)
                rows.append(row)
                scores.append(score(out, upscale=False))
                flicker.append(h3d.flicker_flow_free(h3d.composites(out, truth, regions), truth, regions))

            for arm in ("V", "K"):
                meta, a = first_segments(*arms[arm], int(args.segments))
                xs, ts, rs, data[arm] = training_pairs(args, meta, a, paths, value, scale, work / f"footage-{arm}", device)
                for size in (SIZES if arm == "V" else {"L": SIZES["L"]}):
                    net, info = h3d.fit(xs, ts, rs, int(args.steps), lpips_net, device, *SIZES[size])
                    fits[f"{arm}-{size}"] = {**info, "footage": meta["unit"], "footage_npz_sha256": meta["npz_sha256"]}
                    add(f"{arm}-{size}", net)
                    del net
                del xs, ts, rs
            add("G-L", g_net)
            fits["G-L"] = {**g_info["fit"], "network_sha256": g_info["sha256"], "data": g_info["data"]}
            arrays = {"pixels_r": np.array([int(r.sum()) for r in regions], np.int64),
                      "lpips": np.stack([s["lpips"] for s in scores]), "sse": np.stack([s["sse"] for s in scores]),
                      "dists": np.stack([s["dists"] for s in scores]), "flicker": np.stack(flicker)}
            meta_out = {"unit": unit, "item": it["id"], "video": it["video"], "name": args.point, "h3c_name": h3d.h3c_name("dcvc", value, scale),
                        "frames": frames, "seconds_coded": frames / float(fps), "rows": rows, "point": record, "fits": fits,
                        "training_data": data, "sizes": {k: list(v) for k, v in SIZES.items()}, "mask_set": mask_info,
                        "decode": {k: v for k, v in decoded.items() if k != "path"}, "seconds": round(time.time() - began, 1)}
            save_unit(publish, "component", unit, arrays, meta_out)
            os.remove(decoded["path"])
            shutil.rmtree(work, ignore_errors=True)
        progress(1)
        record_lpips = lpips_record(Path(args.lpips_backbone))
    else:
        record_lpips = None
    write_json(stage_dir() / "h3d.json", {"command": "score", "unit": unit, "item": item["id"], "point": args.point,
                                          "steps": int(args.steps), "restored": unit in restored, "lpips": record_lpips,
                                          "seconds": round(time.time() - began, 1), "torch": torch.__version__, **gpu_record()})
    return 0


# ----------------------------------------------------------------- validation


def validate(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    tar = str(stage / "published.tar")
    checks: dict[str, bool] = {"intended_gpu": tuple(result["gpu_capability"]) >= (8, 6)}
    if result["command"] == "footage":
        units = load_units([tar], "footage")
        checks.update({
            "unit_present": len(units) == 1 and units[0][0]["npz_hash_ok"],
            "every_segment_masked": all(s["frames"] == FOOTAGE["length"] for m, _ in units for s in m["segments"]),
            # Over enough segments, hands are in a good share of egocentric footage; SAM finding almost none would be a wrong
            # prompt or input. A smoke's two content-blind segments may show none (P03_10's waited at a toaster).
            "hands_found": all(len(m["segments"]) < 8 or sum(s["frames_with_r"] for s in m["segments"]) >= 0.25 * m["shape"][0]
                               for m, _ in units),
            "verified_classes": all(v["class"] in VERIFIED for m, _ in units for v in m["videos"].values()),
        })
    elif result["command"] == "generic":
        checks.update({
            "every_point_fitted": bool(result["rows"]) and all(r["fit"]["curve"][-1]["step"] == r["fit"]["steps"] for r in result["rows"]),
            "losses_finite": all(np.isfinite([c["mse_fixed"] for c in r["fit"]["curve"]]).all() for r in result["rows"]),
            "footage_decodes_deterministic": all(r["data"]["deterministic"] for r in result["rows"]),
            "most_footage_has_r": all(r["data"]["frames_with_r"] >= 0.5 * r["data"]["frames"] for r in result["rows"]),
        })
    else:
        units = load_units([tar], "component")
        hand = [(m, a, a["pixels_r"] > 0) for m, a in units]
        checks.update({
            "unit_present": len(units) == 1 and units[0][0]["npz_hash_ok"],
            "every_row_scored": all(a["lpips"].shape[0] == len(m["rows"]) == 1 + len(SIZES) + 2 for m, a, _ in hand),
            "scores_finite_on_hand_frames": all(np.isfinite(a["lpips"][:, h]).all() and np.isfinite(a["dists"][:, h]).all()
                                                for _, a, h in hand),
            "fits_ran_every_step": all(f["curve"][-1]["step"] == f["steps"] for m, _, _ in hand for f in m["fits"].values()),
            "sizes_as_declared": all(m["fits"][f"V-{k}"]["parameters"] == size_parameters(*v) for m, _, _ in hand for k, v in SIZES.items()),
            "footage_decodes_deterministic": all(d["deterministic"] for m, _, _ in hand for d in m["training_data"].values()),
            "window_decode_deterministic": all(m["point"]["deterministic"] and m["point"]["decoder_matches_encoder_intra"] for m, _, _ in hand),
            "mask_set_matches_record": all(m["mask_set"]["masks_rle_sha256"] == m["mask_set"]["record_masks_rle_sha256"] for m, _, _ in hand),
        })
    return checks


def size_parameters(blocks: int, channels: int) -> int:
    """G5d's enhancer's parameter count: head 3→C, ``blocks`` × two C→C convolutions, tail C→3 (3×3, with biases)."""
    conv = lambda cin, cout: 9 * cin * cout + cout  # noqa: E731
    return conv(3, channels) + 2 * blocks * conv(channels, channels) + conv(channels, 3)


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    result = json.loads((stage / "h3d.json").read_text())
    checks = {k: bool(v) for k, v in validate(stage, result).items()}
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.visor.h3d_component")
    commands = parser.add_subparsers(dest="command", required=True)
    footage = commands.add_parser("footage")
    footage.add_argument("--target", required=True, help="the pilot item the footage serves")
    footage.add_argument("--arm", choices=ARMS, required=True)
    footage.add_argument("--video", action="append", required=True)
    footage.add_argument("--eval-set", required=True)
    footage.add_argument("--sam-checkpoint", required=True)
    footage.add_argument("--segments", default=str(FOOTAGE["segments"]), help="the first N of the arm's segments (smoke)")

    def coder(p: argparse.ArgumentParser) -> None:
        p.add_argument("--video", action="append", default=[])
        p.add_argument("--image-ckpt", required=True)
        p.add_argument("--video-ckpt", required=True)
        p.add_argument("--lpips-backbone", required=True)
        p.add_argument("--steps", required=True)
        p.add_argument("--footage", action="append", required=True)
        p.add_argument("--segments", default="1000", help="the footage's first N segments (smoke)")

    generic = commands.add_parser("generic")
    coder(generic)
    generic.add_argument("--point", required=True, help="comma list of the oracle's DCVC-UF point names")
    score = commands.add_parser("score")
    coder(score)
    score.add_argument("--point", required=True)
    score.add_argument("--generic", action="append", required=True)
    score.add_argument("--eval-set", required=True)
    score.add_argument("--archive", required=True)
    score.add_argument("--masks", required=True)
    score.add_argument("--mask-record", required=True)
    score.add_argument("--items", required=True)
    score.add_argument("--frames", default="240")
    score.add_argument("--dists-backbone", required=True)
    commands.add_parser("validate")
    args = parser.parse_args(argv)
    return {"footage": command_footage, "generic": command_generic, "score": command_score,
            "validate": command_validate}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
