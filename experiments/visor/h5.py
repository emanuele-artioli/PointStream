"""PLAN step H5: the first working PointStream on VISOR evaluation set v2 (docs/experiments.md, 2026-10-16).

    python -m experiments.visor.h5 foreground WINDOW --points dcvc@0/4,... --generic TAR ... --limit-frames N
    python -m experiments.visor.h5 score WINDOW --foreground DIR --background DIR ... --routes scene,enh,svt \\
        [--anchors DIR ... --b2-result JSON ...] --dists-backbone P --max-combos N --limit-frames N
    python -m experiments.visor.h5 time WINDOW --arm DIR --enhancers DIR --generic TAR ... --propainter-tree DIR \\
        --propainter-ckpt P --raft-ckpt P --flow-completion-ckpt P --threads 32
    python -m experiments.visor.h5 validate
    python -m experiments.visor.h5 report --score H5.JSON ... --out DIR [--billing BILLING.JSON]
    python -m experiments.visor.h5 model-size --arm DIR ... --video-ckpt P [--limit-tensors N]
    python -m experiments.visor.h5 mask-video --eval-set E.JSON --fill-masks DIR --item ID [--limit-frames N]

WINDOW is ``--eval-set E.JSON --archive DIR --dense-masks DIR --fill-masks DIR --video P.MP4 --item ID
--image-ckpt P --video-ckpt P --lpips-backbone P``.

``foreground`` codes the window's foreground stream at each point (`src.pointstream.composite.foreground_input`:
F and its ring kept, the rest padded by push-pull; DCVC-UF HT-L off the shelf at 1/scale), decodes it as the
client does (bicubic to 1080p, then per class an enhancer slot: H3d's G on the hands at ¼ and ½, the identity on
objects) and publishes each point's 1080p frames (FFV1, 4:2:0, BT.709 full range), its stream and a manifest.
It also measures what F and the class map would cost (granted in this run): FFV1 and lzma, lossless.

``score`` composites every (foreground point, background point) pair of the selected routes (G6's published
windows: ``format`` for the scene model, ``enhance`` for the enhancer and plain SVT-AV1), packs each pair's
container (`src.pointstream.bitstream`: its length is the rate) and scores the composite with B2's harness
(`src.codecs.quality.score`, both mask sets) plus masked DISTS per region. With ``--anchors`` it rescores B2's
published SVT-AV1 and DCVC-UF streams of the item the same way, checked against B2's recorded weighted PSNR.
Each finished row is checkpointed. Writes ``h5.json`` in ``PS_STAGE_DIR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from src.pointstream import bitstream as bs
from src.pointstream import composite as cp

DENSE, FILL = "visor_dense", "visor_dense_sam_fill"
MASK_SETS = (DENSE, FILL)
FULL = (1920, 1080)
#: H3d's coder (¼ and ½ with G's network for the point) and full resolution without a network; QP 63 at ½ and
#: full added after the first pilot score (the foreground capped the hull's quality, docs/experiments.md H5).
POINTS = ("dcvc@0/4", "dcvc@21/4", "dcvc@42/4", "dcvc@0/2", "dcvc@21/2", "dcvc@42/2", "dcvc@21/1", "dcvc@42/1",
          "dcvc@63/2", "dcvc@63/1")
G_SCALES = (2, 4)
#: The points H3d fitted G for (one network per point).
G_POINTS = tuple(f"dcvc@{qp}/{scale}" for scale in G_SCALES for qp in (0, 21, 42))
#: Enhancer ids in the container's slot list (0: the identity).
ENHANCER_IDS = {"G": 1, "G_HO": 2}
ROUTE_OF_METHOD = {"scene-540": "scene", "scene-1080": "scene", "enh-1080": "enh", "svt-1080": "svt"}
DCVC_ATTEMPTS = 3  # K5: re-code a point whose decode is not the encoder's, every attempt recorded
DISTS_BATCH = 2
DISTS_EVERY = 4  # DISTS on frames 0, 4, 8, ... of every method (LPIPS and PSNR on all frames, as B2)
CHUNK = 16  # frames composited at a time
ANCHOR_WPSNR_TOLERANCE_DB = 0.02
FOREGROUND_WEIGHT = 0.7


def parse_point(name: str) -> tuple[int, int]:
    """``dcvc@QP/SCALE`` to (QP, scale)."""
    codec, rest = name.split("@")
    qp, scale = rest.split("/")
    if codec != "dcvc" or int(scale) not in (1, 2, 4):
        raise SystemExit(f"{name}: not a DCVC-UF point at scale 1, 2 or 4")
    return int(qp), int(scale)


def point_key(name: str) -> str:
    qp, scale = parse_point(name)
    return f"dcvc-{qp}_{scale}"


def coded_size(scale: int) -> tuple[int, int]:
    from experiments.visor.h3b import scaled_size

    return scaled_size(FULL[0], FULL[1], scale)


# ----------------------------------------------------------------- the window


def decode_source(args: argparse.Namespace, item: dict[str, Any], frames: int, work: Path) -> dict[str, Any]:
    """B2's decode of the window (its JPEG gate checked) in a spawned process (PyAV beside torch stalls, B2)."""
    from experiments.visor.b2 import decode_window

    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        record = pool.submit(decode_window, Path(args.video), item, frames, Path(args.archive),
                             work / "source.yuv", 8).result()
    if (record["width"], record["height"]) != FULL:
        raise SystemExit(f"{item['id']}: {record['width']}x{record['height']}, not 1080p")
    if not record["full_range"]:
        raise SystemExit(f"{item['id']}: the composite is written in full range; the source is limited")
    return record


def load_window(args: argparse.Namespace, work: Path) -> dict[str, Any]:
    """The item, its source (planes on disk and RGB), both mask sets, the class map and B2's regions."""
    from experiments.background import g6
    from experiments.visor.b2 import frame_regions, load_mask_sets
    from src.segmentation.visor import HANDS

    item = g6.load_item(args.eval_set, args.item)
    frames = args.limit_frames if args.limit_frames > 0 else int(item["frames"])
    source = decode_source(args, item, frames, work)
    planes = np.fromfile(source["path"], np.uint8).reshape(frames, FULL[1] * 3 // 2, FULL[0])
    rgb = g6.to_rgb(planes, True)
    clips, info = load_mask_sets(item, frames, {DENSE: args.dense_masks, FILL: args.fill_masks}, {},
                                 Path(args.archive))
    fill = clips[FILL]
    classes = np.zeros((frames, FULL[1], FULL[0]), np.uint8)
    for t in range(frames):
        hands = fill.class_mask(t, HANDS[0]) | fill.class_mask(t, HANDS[1])
        classes[t][fill.foreground(t)] = cp.OBJECT
        classes[t][hands] = cp.HAND
    fps = Fraction(item["fps"]).limit_denominator(1001)
    return {"item": item, "frames": frames, "fps": fps, "source": source, "rgb": rgb, "classes": classes,
            "regions": frame_regions(clips, frames), "masks": info,
            "classes_sha256": hashlib.sha256(np.packbits(classes > 0).tobytes()
                                             + np.packbits(classes == cp.HAND).tobytes()).hexdigest()}


def mask_cost(classes: np.ndarray, fps: Fraction, work: Path) -> dict[str, Any]:
    """What sending F and the class map would cost, losslessly: FFV1 (gray, 1080p; packet bytes) and lzma over
    the frame-to-frame XOR of the packed F and hand bit planes. Granted in H5's first run, reported apart."""
    import av

    from experiments.background.g1d import FFV1_OPTIONS

    seconds = len(classes) / float(fps)
    ffv1 = 0
    with av.open(str(work / "classes.mkv"), "w", format="matroska") as out:
        stream = out.add_stream("ffv1", rate=fps)
        stream.width, stream.height, stream.pix_fmt = classes.shape[2], classes.shape[1], "gray"
        stream.options = dict(FFV1_OPTIONS)
        for frame in classes:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="gray")):
                ffv1 += packet.size
        for packet in stream.encode():
            ffv1 += packet.size
    planes = np.stack([np.packbits(classes > 0, axis=-1), np.packbits(classes == cp.HAND, axis=-1)], axis=1)
    delta = planes.copy()
    delta[1:] ^= planes[:-1]
    packed = len(lzma.compress(delta.tobytes(), preset=9 | lzma.PRESET_EXTREME))
    best = min(ffv1, packed)
    return {"ffv1_bytes": ffv1, "lzma_xor_bytes": packed, "bytes": best, "kbps": best * 8 / seconds / 1000,
            "coder": "ffv1" if ffv1 <= packed else "lzma_xor", "frames": len(classes),
            "share": float((classes > 0).mean()), "hand_share": float((classes == cp.HAND).mean())}


def coded_classes(classes: np.ndarray, scale: int, fps: Fraction) -> tuple[np.ndarray, bytes, dict[str, Any]]:
    """H5m: the client's F̂ from F's class map coded at 1/``scale`` (`src.pointstream.mask`), the MASK payload, and
    its record (rate, F̂ against F)."""
    from src.pointstream import mask as mk

    reduced = mk.reduce(classes, scale)
    payload = mk.encode(reduced, scale)
    decoded, _ = mk.decode(payload)
    shown = mk.upsample(decoded, (classes.shape[2], classes.shape[1]))
    f, g = classes > 0, shown > 0
    hands = classes == cp.HAND
    record = {"scale": scale, "bytes": len(payload), "kbps": len(payload) * 8 / (len(classes) / float(fps)) / 1000,
              "lossless_on_reduced_grid": bool(np.array_equal(decoded, reduced)),
              "iou": float((f & g).sum() / max(int((f | g).sum()), 1)),
              "missed_share": float((f & ~g).sum() / max(int(f.sum()), 1)),
              "extra_share": float((g & ~f).sum() / max(int(f.sum()), 1)),
              "hand_kept": float((shown[hands] == cp.HAND).mean()) if hands.any() else None}
    return shown, payload, record


def read_ffv1(path: Path) -> np.ndarray:
    """(n, h·3/2, w) 4:2:0 planes of an FFV1 file."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 8
        return np.stack([np.ascontiguousarray(f.to_ndarray(format="yuv420p")) for f in container.decode(stream)])


def checkpoint_dir() -> Path | None:
    return Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None


def scratch_dir() -> Path:
    return Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")


# ----------------------------------------------------------------- foreground


def code_dcvc(args: argparse.Namespace, source: Path, frames: int, qp: int, size: tuple[int, int],
              work: Path) -> dict[str, Any]:
    """G5's DCVC-UF HT-L coder, repeated until the decode is deterministic and matches the encoder's intra frame."""
    from experiments.background import g5

    coder = argparse.Namespace(image_ckpt=args.image_ckpt, video_ckpt=args.video_ckpt)
    attempts = []
    for attempt in range(DCVC_ATTEMPTS):
        shutil.rmtree(work, ignore_errors=True)
        try:
            rec = g5.code_dcvc(coder, source, frames, qp, work, width=size[0], height=size[1])
        except RuntimeError as error:  # the decoder can segfault after its own encode (G5c step 1)
            attempts.append({"error": str(error)[:300]})
            continue
        attempts.append({k: rec[k] for k in ("bytes", "deterministic", "decoder_matches_encoder_intra")})
        if rec["deterministic"] and rec["decoder_matches_encoder_intra"]:
            return {**rec, "attempts": attempts}
    raise RuntimeError(f"DCVC-UF QP {qp} at {size}: no decode matched its encoder in {DCVC_ATTEMPTS} attempts: "
                       f"{attempts}")


def hand_enhancer(net: Any, device: str) -> cp.Enhancer:
    from experiments.visor import h3d

    def run(frames: np.ndarray, where: np.ndarray) -> np.ndarray:
        import torch

        out = h3d.apply(net, frames, list(where), device)
        torch.cuda.empty_cache()
        return out

    return run


def foreground_point(args: argparse.Namespace, w: dict[str, Any], name: str, nearest: np.ndarray,
                     nets: dict[tuple[str, str], tuple[Any, dict[str, Any], str]], work: Path,
                     points: Path) -> dict[str, Any]:
    """One foreground point: coded, decoded as the client does, kept as FFV1 with its stream."""
    from experiments.background import g5, g6
    from experiments.visor import h3d

    qp, scale = parse_point(name)
    size = coded_size(scale)
    key = point_key(name)
    frames = w["frames"]
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # K from the F the client will composite with: F̂ when F is coded (H5p 4), else F
    padded = cp.foreground_input(w["rgb"], w.get("shown", w["classes"]) > 0, scale, size)
    source = work / "input.yuv"
    g5.rgb_to_yuv420(padded).tofile(source)
    del padded
    t1 = time.time()
    import torch

    torch.cuda.empty_cache()  # G's cached blocks would otherwise sit beside the DCVC-UF worker (33 GB peak)
    rec = code_dcvc(args, source, frames, qp, size, work / "dcvc")
    t2 = time.time()
    decoded = g5.yuv420_to_rgb(np.fromfile(rec["decoded"], np.uint8).reshape(frames, size[1] * 3 // 2, size[0]))
    upsampled = h3d.upscale_all(decoded, frames, FULL[0], FULL[1]) if size != FULL else decoded
    slots: dict[int, cp.Enhancer] = {}
    networks: dict[str, Any] = {"hand": None, "object": None}
    for cls, label in ((cp.HAND, "hand"), (cp.OBJECT, "object")):
        if (name, label) in nets:  # each class's slot: G on the hands (H3d), or G_HO on both (H4c)
            net, info, arm = nets[(name, label)]
            slots[cls] = hand_enhancer(net, "cuda")
            networks[label] = {"point": name, "arm": arm, "sha256": info["sha256"]}
    if (name, "hand") in nets and nets.get((name, "object")) is nets[(name, "hand")]:
        # one network in both slots: one pass over F's box, as the client would run it
        both = hand_enhancer(nets[(name, "hand")][0], "cuda")
        enhanced = both(upsampled, nearest > 0)
        out = np.where((nearest > 0)[..., None], enhanced, upsampled)
    else:
        out = cp.apply_slots(upsampled, nearest, slots)
    t3 = time.time()
    yuv = g5.rgb_to_yuv420(out)
    sse, count = 0.0, 0
    for first in range(0, frames, CHUNK):
        sl = slice(first, first + CHUNK)
        f = w["classes"][sl] > 0
        diff = g6.to_rgb(yuv[sl], True).astype(np.float32) - w["rgb"][sl].astype(np.float32)
        sse, count = sse + float((diff[f] ** 2).sum()), count + 3 * int(f.sum())
    psnr_f = float(10 * np.log10(255.0**2 / max(sse / count, 1e-12))) if count else None
    target = points / key
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    digests = g6.write_ffv1(yuv, target / "decoded.mkv", w["fps"])
    if g6.ffv1_digests(target / "decoded.mkv") != digests:
        raise RuntimeError(f"{key}: the FFV1 file does not decode to the frames written")
    shutil.copyfile(work / "dcvc" / "stream.psdc", target / "stream.psdc")
    seconds = frames / float(w["fps"])
    return {"name": name, "key": key, "qp": qp, "scale": scale, "coded_size": list(size), "bytes": rec["bytes"],
            "kbps": rec["bytes"] * 8 / seconds / 1000, "attempts": rec["attempts"],
            "deterministic": rec["deterministic"], "decoder_matches_encoder_intra": rec["decoder_matches_encoder_intra"],
            "slots": networks, "psnr_f": psnr_f,
            "changed_by_slots": float((out != upsampled).any(axis=3).mean()),
            "decoded": f"points/{key}/decoded.mkv", "decoded_ffv1_sha256": file_sha256(target / "decoded.mkv"),
            "decoded_frame_sha256": digests, "bitstream": f"points/{key}/stream.psdc",
            "bitstream_sha256": file_sha256(target / "stream.psdc"),
            "seconds": {"pad": round(t1 - t0, 2), "code": round(t2 - t1, 2), "client": round(t3 - t2, 2)}}


def command_foreground(args: argparse.Namespace) -> int:
    # G's crops differ in size every frame; without expandable segments the caching allocator keeps every size and
    # fills the GPU (48.5 GB with G_HO over F's box). Set before CUDA starts.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    from experiments.background import g5c, g6
    from experiments.visor.h3d_component import generic_network

    names = args.points.split(",")
    scratch = scratch_dir()
    work = scratch / "foreground"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    began = time.time()
    torch.backends.cudnn.benchmark = False
    w = load_window(args, work)
    video_model = None
    if args.video_arm:  # H5q 1: the window's venue model (G6 `train --fill none`, the anchor's) codes the foreground
        video_model, args.video_ckpt = g6.load_arm(args.video_arm)
        if video_model["fill"] != "none" or video_model["id"].split("/")[-1] != w["item"]["video"]:
            raise SystemExit(f"{args.video_arm}: not the whole-frame venue model of {w['item']['video']}")
    video_model = {"video_ckpt_sha256": file_sha256(Path(args.video_ckpt)), "venue": video_model}
    mask_record = None
    if args.mask_scale > 0:  # H5p 4: the stream built for F̂ (K and the slots), as the client composites it
        w["shown"], _, mask_record = coded_classes(w["classes"], args.mask_scale, w["fps"])
    nearest = np.stack([cp.alpha_and_classes(c)[1] for c in w.get("shown", w["classes"])])
    cost = mask_cost(w["classes"], w["fps"], work)
    nets: dict[tuple[str, str], tuple[Any, dict[str, Any], str]] = {}
    for n in names:
        if n not in G_POINTS:
            continue
        if args.slots == "G":  # H3d: G on the hands, the identity on objects
            net, info = generic_network(args.generic, n, "cuda")
            nets[(n, "hand")] = (net, info, "G")
        else:  # H4c (user, 2026-10-17): G_HO on hands and objects
            from experiments.visor.h4c import load_networks

            net, info = load_networks(args.object_networks, n, "cuda")["G_HO"]
            nets[(n, "hand")] = nets[(n, "object")] = (net, info, "G_HO")
    checkpoints = checkpoint_dir()
    restored = g5c.restore(checkpoints)
    points = (checkpoints or scratch) / "points"
    points.mkdir(parents=True, exist_ok=True)
    rows = []
    for done, name in enumerate(names, 1):
        key = point_key(name)
        if key in restored and (points / key / "decoded.mkv").is_file():
            rows.append({**restored[key], "restored_from_checkpoint": True})
        else:
            row = foreground_point(args, w, name, nearest, nets, work / key, points)
            shutil.rmtree(work / key, ignore_errors=True)
            g5c.save(checkpoints, key, row)
            rows.append(row)
        progress(done)
    publish = scratch / "publish"
    for row in rows:
        shutil.rmtree(publish / "points" / row["key"], ignore_errors=True)
        shutil.copytree(points / row["key"], publish / "points" / row["key"])
    item = w["item"]
    manifest = {"window": item["id"], "video": item["video"], "frames": w["frames"], "fps": str(w["fps"]),
                "yuv": g6.YUV, "classes_sha256": w["classes_sha256"], "mask_set": FILL, "mask_cost": cost,
                "ring_px": cp.RING, "keep_radius_px": {str(s): cp.keep_radius(s) for s in (1, 2, 4)},
                "mask_scale": args.mask_scale, "mask": mask_record,
                "video_model": video_model, **g6.provenance(), **g5c.gpu_record(), "points": rows}
    write_json(publish / "manifest.json", manifest)
    write_json(stage_dir() / "h5.json", {
        "kind": "foreground", "item": item["id"], "frames": w["frames"], "limit_frames": args.limit_frames,
        "source": {k: v for k, v in w["source"].items() if k != "jpeg_gate"},
        "jpeg_gate_holds": all(c["holds"] for c in w["source"]["jpeg_gate"]), "masks": w["masks"],
        "mask_cost": cost, "points": rows, "video_model": video_model, "mask_scale": args.mask_scale,
        "mask": mask_record,
        "manifest_sha256": file_sha256(publish / "manifest.json"),
        "seconds": round(time.time() - began, 1), **g5c.gpu_record(), **g6.provenance()})
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ----------------------------------------------------------------- scoring


class Judges:
    """LPIPS (B2's backbone, inside `quality.score`) and DISTS, on the claimed GPU."""

    def __init__(self, args: argparse.Namespace) -> None:
        from experiments.visor.b2 import lpips_net
        from src.codecs import quality

        self.lpips = lpips_net(args.lpips_backbone, "cuda")
        self.dists = quality.load_dists(Path(args.dists_backbone), "cuda")
        self.record = {"lpips": quality.lpips_record(Path(args.lpips_backbone)),
                       "dists": quality.dists_record(Path(args.dists_backbone))}


def dists_regions(net: Any, x: Any, y: Any, masks: list[Any]) -> list[Any]:
    """`quality.masked_dists` for several masks from one pass of the network over ``x`` and ``y``."""
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        fx, fy = net.forward_once(x), net.forward_once(y)
        total = net.alpha.sum() + net.beta.sum()
        alpha = torch.split(net.alpha / total, net.chns, dim=1)
        beta = torch.split(net.beta / total, net.chns, dim=1)
        out = []
        for mask in masks:
            structure = texture = torch.zeros(x.shape[0], 1, 1, 1, device=x.device)
            for k, (a, b) in enumerate(zip(fx, fy)):
                wt = F.adaptive_avg_pool2d(mask.float(), a.shape[-2:])
                wt = wt / wt.sum((2, 3), keepdim=True).clamp_min(1e-12)
                a_mean, b_mean = (a * wt).sum((2, 3), keepdim=True), (b * wt).sum((2, 3), keepdim=True)
                s1 = (2 * a_mean * b_mean + 1e-6) / (a_mean**2 + b_mean**2 + 1e-6)
                a_var = ((a - a_mean) ** 2 * wt).sum((2, 3), keepdim=True)
                b_var = ((b - b_mean) ** 2 * wt).sum((2, 3), keepdim=True)
                cov = (a * b * wt).sum((2, 3), keepdim=True) - a_mean * b_mean
                s2 = (2 * cov + 1e-6) / (a_var + b_var + 1e-6)
                structure = structure + (alpha[k] * s1).sum(1, keepdim=True)
                texture = texture + (beta[k] * s2).sum(1, keepdim=True)
            out.append((1 - (structure + texture).flatten()).cpu().numpy())
        return out


def dists_frames(judges: Judges, shown: np.ndarray, truth: np.ndarray, regions: list[dict[str, np.ndarray]],
                 first: int) -> list[dict[str, float | None]]:
    """Per frame, DISTS over the frame and over each mask set's fg and bg (None for an empty region)."""
    import torch

    names = ["frame"] + [f"{m}/{r}" for m in MASK_SETS for r in ("fg", "bg")]
    rows: list[dict[str, float | None]] = []
    for start in range(0, len(shown), DISTS_BATCH):
        idx = range(start, min(start + DISTS_BATCH, len(shown)))
        x = torch.from_numpy(np.ascontiguousarray(shown[idx.start:idx.stop])).cuda().permute(0, 3, 1, 2).float() / 255
        y = torch.from_numpy(np.ascontiguousarray(truth[idx.start:idx.stop])).cuda().permute(0, 3, 1, 2).float() / 255
        stacks = {n: np.stack([np.ones(FULL[::-1], bool) if n == "frame" else regions[first + i][n] for i in idx])
                  for n in names}
        values = dists_regions(judges.dists, x, y, [torch.from_numpy(stacks[n]).cuda()[:, None] for n in names])
        for j, i in enumerate(idx):
            rows.append({n: float(v[j]) if stacks[n][j].any() else None for n, v in zip(names, values)})
    return rows


def summarize(per_frame: list[dict[str, Any]], dists: list[dict[str, float | None]]) -> dict[str, Any]:
    """B2's summary (both mask sets) with weighted DISTS beside, flattened per mask set."""
    from experiments.visor.b2 import flat_summary, mean, summarize as b2_summarize, weighted

    summary = b2_summarize(per_frame, [], list(MASK_SETS))
    out: dict[str, Any] = {"frame": {"psnr": summary["frame"]["psnr"], "lpips": summary["frame"]["lpips"],
                                     "dists": mean([d["frame"] for d in dists]), "ms_ssim": summary["frame"]["ms_ssim"]}}
    for m in MASK_SETS:
        fg, bg = f"{m}/fg", f"{m}/bg"
        both = [d for d in dists if d[fg] is not None and d[bg] is not None]
        out[m] = {**{k: v for k, v in flat_summary(summary, m).items() if k not in ("vmaf",)},
                  "wdists": mean([weighted(d[fg], d[bg]) for d in both]),
                  "dists_fg": mean([d[fg] for d in dists]), "dists_bg": mean([d[bg] for d in dists])}
    return out


def score_decoded(w: dict[str, Any], judges: Judges, yuv_path: Path) -> dict[str, Any]:
    """Score a whole decoded window (1080p 4:2:0 in the source's full range) as B2 does, plus DISTS."""
    from src.codecs import quality

    frames = w["frames"]
    t0 = time.perf_counter()
    per_frame = quality.score(Path(w["source"]["path"]), yuv_path, w["regions"], width=FULL[0], height=FULL[1],
                              frames=frames, full_range=True, device="cuda", lpips_net=judges.lpips)
    t1 = time.perf_counter()
    planes = np.memmap(yuv_path, np.uint8, "r", shape=(frames, FULL[1] * 3 // 2, FULL[0]))
    judged = list(range(0, frames, DISTS_EVERY))
    dists = []
    for first in range(0, len(judged), CHUNK):
        idx = judged[first:first + CHUNK]
        shown = rgb_gpu(np.asarray(planes[idx]))
        dists += dists_frames(judges, shown, w["rgb"][idx], [w["regions"][i] for i in idx], 0)
    return {**summarize(per_frame, dists), "seconds": {"b2_harness": round(t1 - t0, 2),
                                                       "dists": round(time.perf_counter() - t1, 2)}}


def load_backgrounds(dirs: list[str], routes: set[str], item_id: str) -> list[dict[str, Any]]:
    """Every published background point of the selected routes for the item, with its directory and the venue
    model its decoder holds."""
    out = []
    for d in dirs:
        root = Path(d) / "publish" if (Path(d) / "publish" / "manifest.json").is_file() else Path(d)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["window"] != item_id:
            raise SystemExit(f"{d}: background of {manifest['window']}, not {item_id}")
        scene = (manifest.get("scene_model") or {}).get("model_sha256")
        for p in manifest["points"]:
            route = ROUTE_OF_METHOD[p["method"]]
            if route not in routes:
                continue
            model = scene if route == "scene" else (p.get("enhancer") or {}).get("weights_sha256")
            out.append({**p, "route": route, "key": f"{p['method']}-{p['value']}", "root": str(root),
                        "model_sha256": model, "manifest_job": manifest.get("job_id"),
                        "manifest_git_head": manifest.get("git_head"), "f_sha256": manifest["foreground"]})
    return sorted(out, key=lambda p: (p["route"], p["key"]))


def container_for(w: dict[str, Any], bg: dict[str, Any], fg: dict[str, Any], fg_root: Path,
                  mask_payload: bytes | None = None) -> bs.Container:
    """The window's container for one pair: the background stream, the foreground stream with its slots, and
    references to the networks the decoder holds."""
    bg_data = (Path(bg["root"]) / bg["bitstream"]).read_bytes()
    if hashlib.sha256(bg_data).hexdigest() != bg["bitstream_sha256"]:
        raise RuntimeError(f"{bg['key']}: background stream differs from its manifest")
    codec = "av1" if bg["codec"] in ("svt", "enh") else "dcvc-uf"
    stream = bs.ivf_to_obus(bg_data)[0] if codec == "av1" else bg_data
    size = bg["coded_size"]
    c = bs.Container(FULL[0], FULL[1], w["fps"], w["frames"])
    c.add_coded(bs.Coded("background", bs.ROUTES[bg["route"]], codec, size[0], size[1], stream))
    fg_data = (fg_root / fg["bitstream"]).read_bytes()
    if hashlib.sha256(fg_data).hexdigest() != fg["bitstream_sha256"]:
        raise RuntimeError(f"{fg['key']}: foreground stream differs from its manifest")
    hand, obj = fg["slots"]["hand"], fg["slots"].get("object")
    slots = tuple((bs.CLASSES[k], ENHANCER_IDS[v["arm"] if v.get("arm") else "G"] if v else 0)
                  for k, v in (("hand", hand), ("object", obj)))
    c.add_coded(bs.Coded("foreground", fg["scale"], "dcvc-uf", fg["coded_size"][0], fg["coded_size"][1], fg_data,
                         slots))
    if mask_payload is not None:  # H5m: F's class map coded at 1/s (`src.pointstream.mask`)
        c.add(bs.Tag.MASK, mask_payload)
    if bg["model_sha256"]:
        c.add_model(bs.ModelRef("background", bg["model_sha256"]))
    for sha in dict.fromkeys(v["sha256"] for v in (hand, obj) if v):
        c.add_model(bs.ModelRef("foreground", sha))
    return c


def bitstream_check(c: bs.Container, data: bytes, bg: dict[str, Any], fg: dict[str, Any], fg_root: Path) -> dict[str, bool]:
    """Rule (d): the packed container parses back to its segments and its streams are the published ones."""
    back = bs.Container.unpack(data)
    [b] = back.coded(bs.Tag.BACKGROUND)
    [f] = back.coded(bs.Tag.FOREGROUND)
    original = (Path(bg["root"]) / bg["bitstream"]).read_bytes()
    if b.codec == "av1":
        rebuilt = bs.obus_to_ivf(b.stream, b.width, b.height, c.fps)
        from src.codecs.svtav1 import ivf_frames

        same_bg = ivf_frames(rebuilt) == ivf_frames(original) and bs.ivf_to_obus(rebuilt) == bs.ivf_to_obus(original)
    else:
        same_bg = b.stream == original
    return {"segments_round_trip": back.segments == c.segments and back.pack() == data,
            "background_stream_is_published": same_bg,
            "foreground_stream_is_published": f.stream == (fg_root / fg["bitstream"]).read_bytes(),
            "rate_is_length": sum(back.breakdown().values()) == len(data)}


def rgb_gpu(planes: np.ndarray) -> np.ndarray:
    """(n, h·3/2, w) full-range 4:2:0 planes to (n, h, w, 3) uint8 RGB, converted on the GPU (`g6.to_rgb`'s
    conversion: `quality.yuv420_to_rgb`)."""
    import torch

    from src.codecs import quality

    out = np.empty((len(planes), planes.shape[1] * 2 // 3, planes.shape[2], 3), np.uint8)
    for first in range(0, len(planes), CHUNK):
        x = torch.from_numpy(np.ascontiguousarray(planes[first:first + CHUNK])).cuda()
        rgb = quality.yuv420_to_rgb(x, planes.shape[2], planes.shape[1] * 2 // 3, full_range=True)
        out[first:first + CHUNK] = rgb.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
    return out


def composite_gpu(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """`cp.composite` then `g5.rgb_to_yuv420` on the GPU, the same float32 arithmetic: (n, h·3/2, w) uint8."""
    import torch

    from src.codecs.quality import KB, KR

    a = torch.from_numpy(np.ascontiguousarray(alpha)).cuda()[..., None]
    mixed = torch.round(a * torch.from_numpy(fg).cuda().float() + (1.0 - a) * torch.from_numpy(bg).cuda().float())
    rgb = mixed.clamp(0, 255) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = KR * r + (1.0 - KR - KB) * g + KB * b
    cb = (b - y) / (2.0 * (1.0 - KB))
    cr = (r - y) / (2.0 * (1.0 - KR))
    n, h, w = y.shape

    def pool(c: Any) -> Any:
        return c.reshape(n, h // 2, 2, w // 2, 2).mean(dim=(2, 4))

    out = torch.empty((n, h * 3 // 2, w), dtype=torch.uint8, device=y.device)
    out[:, :h] = torch.round(y * 255.0).clamp(0, 255).to(torch.uint8)
    chroma = torch.stack([pool(cb), pool(cr)], dim=1) * 255.0 + 128.0
    out[:, h:] = torch.round(chroma).clamp(0, 255).to(torch.uint8).reshape(n, h // 2, w)
    return out.cpu().numpy()


def composite_window(w: dict[str, Any], fg_rgb: np.ndarray, bg_rgb: np.ndarray, alpha: np.ndarray,
                     out: Path) -> str:
    """The client's output (4:2:0, full range) written to ``out``; its sha256. Both decodes as RGB."""
    digest = hashlib.sha256()
    with out.open("wb") as handle:
        for first in range(0, w["frames"], CHUNK):
            sl = slice(first, first + CHUNK)
            data = composite_gpu(fg_rgb[sl], bg_rgb[sl], alpha[sl]).tobytes()
            digest.update(data)
            handle.write(data)
    return digest.hexdigest()


def gpu_matches_cpu(fg_rgb: np.ndarray, bg_rgb: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    """The GPU compositor against `cp.composite` and `g5.rgb_to_yuv420` on the first frames: largest difference."""
    from experiments.background import g5

    sl = slice(0, 4)
    cpu = g5.rgb_to_yuv420(cp.composite(fg_rgb[sl], bg_rgb[sl], alpha[sl])).astype(np.int16)
    gpu = composite_gpu(fg_rgb[sl], bg_rgb[sl], alpha[sl]).astype(np.int16)
    diff = np.abs(cpu - gpu)
    return {"max_abs": int(diff.max()), "share_differing": float((diff > 0).mean())}


def score_anchors(args: argparse.Namespace, w: dict[str, Any], judges: Judges, work: Path) -> list[dict[str, Any]]:
    """B2's published streams of the item, decoded and scored as PointStream's composites are."""
    from experiments.visor import b2

    item = w["item"]
    recorded = {}
    for path in args.b2_result or []:
        result = json.loads(Path(path).read_text())
        for it in result["items"]:
            for p in it["points"]:
                if not p.get("variant"):
                    recorded[(it["id"], result["codec"], float(p["point"]))] = p
    coder = argparse.Namespace(decoder="dav1d", threads=8, structure="htl", image_ckpt=args.image_ckpt,
                               image_sha256=file_sha256(Path(args.image_ckpt)), video_ckpt=args.video_ckpt,
                               video_sha256=file_sha256(Path(args.video_ckpt)))
    source = {**w["source"], "frames": w["frames"]}
    rows = []
    for directory in args.anchors or []:
        root = b2.stream_root(Path(directory))
        for codec in ("svtav1", "dcvc"):
            base = root / codec / item["id"]
            if not base.is_dir():
                continue
            for point_dir in sorted(base.iterdir(), key=lambda p: float(p.name)):
                point = float(point_dir.name)
                stream = point_dir / b2.STREAM_NAMES[codec]
                pw = work / f"{codec}-{point_dir.name}"
                shutil.rmtree(pw, ignore_errors=True)
                pw.mkdir(parents=True)
                if codec == "svtav1":
                    rec = b2.code_svtav1(coder, source, w["fps"], point, pw, stream, {}, None, None)
                else:  # a stored DCVC-UF stream never decodes in a later job (docs/fleet.md): re-encode B2's point
                    rec = b2.code_dcvc(coder, source, point, pw, None, False, {"range": "full"})
                    if not (rec["deterministic"] and rec["decoder_matches_encoder_intra"]):
                        raise RuntimeError(f"B2's DCVC-UF point {point:g} re-encoded: its decode is not the encoder's")
                if args.limit_frames > 0:  # the smoke scores the window's first frames of the whole stream
                    head = np.fromfile(rec["decoded"], np.uint8, count=w["frames"] * FULL[0] * FULL[1] * 3 // 2)
                    head.tofile(rec["decoded"])
                scores = score_decoded(w, judges, Path(rec["decoded"]))
                scores.pop("seconds")
                seconds = w["frames"] / float(w["fps"])
                b2_row = recorded.get((item["id"], codec, point))
                rows.append({"codec": codec, "point": point, "rate_bytes": rec["rate_bytes"],
                             "reencoded": codec == "dcvc", "b2_stream_sha256": file_sha256(stream),
                             "kbps": rec["rate_bytes"] * 8 / (int(item["frames"]) / float(w["fps"])) / 1000,
                             "kbps_window": rec["rate_bytes"] * 8 / seconds / 1000,
                             "stream_sha256": rec["stream_sha256"], "deterministic": rec.get("deterministic"),
                             "scores": scores,
                             "b2": None if b2_row is None else {
                                 "kbps": b2_row["kbps"], "stream_sha256": b2_row["stream_sha256"],
                                 "wpsnr": b2_row["summary"]["mask_sets"][DENSE]["wpsnr"],
                                 "wlpips": b2_row["summary"]["mask_sets"][DENSE]["wlpips"]}})
                shutil.rmtree(pw, ignore_errors=True)
    return rows


LADDER = {1080: (1920, 1080), 720: (1280, 720), 540: (960, 540), 360: (640, 360)}


def parse_ladder(spec: str | None) -> list[tuple[int, int]]:
    """``1080=41,48;540=0,21`` to [(height, value), ...]."""
    out = []
    for part in (spec or "").split(";"):
        if part.strip():
            height, _, values = part.partition("=")
            if int(height) not in LADDER:
                raise SystemExit(f"{height}: not a ladder height ({sorted(LADDER)})")
            out += [(int(height), int(v)) for v in values.split(",")]
    return out


def ladder_source(w: dict[str, Any], height: int, work: Path) -> dict[str, Any]:
    """The window at a ladder size: 1080p as decoded; lower sizes by area average in RGB (as G6's backgrounds),
    then 4:2:0 in full range."""
    from experiments.background import g5, g6

    if height == FULL[1]:
        return {**w["source"], "frames": w["frames"]}
    size = LADDER[height]
    path = work / f"source-{height}.yuv"
    if not path.is_file():
        g5.rgb_to_yuv420(g6.resize(w["rgb"], size, area=True)).tofile(path)
    return {"path": str(path), "width": size[0], "height": size[1], "frames": w["frames"], "full_range": True}


def to_full(decoded: Path, height: int, frames: int) -> Path:
    """A ladder decode at 1080p as the client shows it: bicubic in RGB, then 4:2:0 (in place)."""
    from experiments.background import g5, g6

    if height == FULL[1]:
        return decoded
    w, h = LADDER[height]
    raw = np.fromfile(decoded, np.uint8).reshape(frames, h * 3 // 2, w)
    out = np.empty((frames, FULL[1] * 3 // 2, FULL[0]), np.uint8)
    for a in range(0, frames, CHUNK):
        out[a:a + CHUNK] = g5.rgb_to_yuv420(g6.resize(g5.yuv420_to_rgb(raw[a:a + CHUNK]), FULL, area=False))
    out.tofile(decoded)
    return decoded


def command_anchors(args: argparse.Namespace) -> int:
    """H5p 3: fair anchors for the window. SVT-AV1 at the equal-time preset and DCVC-UF off the shelf, each at
    1080p and lower ladder sizes (upscaled bicubically), and G6's venue-adapted whole-frame DCVC-UF from its
    published decodes; all scored as H5's anchors. Writes ``h5.json`` after every point."""
    from experiments.visor import b2

    began = time.time()
    work = scratch_dir() / "anchors"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    w = load_window(args, work)
    judges = Judges(args)
    coder = argparse.Namespace(encoder=args.encoder, decoder=args.decoder, threads=args.threads, structure="htl",
                               image_ckpt=args.image_ckpt, image_sha256=file_sha256(Path(args.image_ckpt)),
                               video_ckpt=args.video_ckpt, video_sha256=file_sha256(Path(args.video_ckpt)))
    seconds = w["frames"] / float(w["fps"])
    planned = [("svtav1-p%d" % args.preset, h, v) for h, v in parse_ladder(args.svt)] + \
        [("dcvc-ladder", h, v) for h, v in parse_ladder(args.dcvc)]
    venue = load_backgrounds(args.venue or [], {"scene"}, w["item"]["id"])
    rows: list[dict[str, Any]] = []
    result = {"kind": "anchors", "item": w["item"]["id"], "frames": w["frames"], "limit_frames": args.limit_frames,
              "jpeg_gate_holds": all(c["holds"] for c in w["source"]["jpeg_gate"]),
              "preset": args.preset, "planned": len(planned) + len(venue), "rows": rows, "judges": judges.record,
              "encoder": b2.svtav1.tool(args.encoder) if args.svt else None,
              "decoder": b2.svtav1.tool(args.decoder) if args.svt else None,
              "venue_manifests": sorted({v["root"] for v in venue})}
    for codec, height, value in planned:
        pw = work / f"{codec}-{height}-{value}"
        shutil.rmtree(pw, ignore_errors=True)
        pw.mkdir(parents=True)
        source = ladder_source(w, height, work)
        if codec.startswith("svtav1"):
            rec = b2.code_svtav1(coder, source, w["fps"], value, pw, None, {"preset": args.preset, "enable_tf": None},
                                 None, None)
        else:  # DCVC-UF decodes non-deterministically on a contended GPU (docs/fleet.md): re-code, as K5
            attempts = []
            for _ in range(DCVC_ATTEMPTS):
                shutil.rmtree(pw, ignore_errors=True)
                pw.mkdir(parents=True)
                rec = b2.code_dcvc(coder, source, value, pw, None, False, {"range": "full"})
                attempts.append({k: rec.get(k) for k in ("stream_sha256", "deterministic", "decoder_matches_encoder_intra")})
                if rec["deterministic"] and rec["decoder_matches_encoder_intra"]:
                    break
            else:
                raise RuntimeError(f"DCVC-UF {height}p QP {value}: its decode is not the encoder's ({DCVC_ATTEMPTS} attempts)")
            rec["attempts"] = attempts
        scores = score_decoded(w, judges, to_full(Path(rec["decoded"]), height, w["frames"]))
        rows.append({"codec": codec, "point": value, "height": height, "rate_bytes": rec["rate_bytes"],
                     "kbps": rec["rate_bytes"] * 8 / seconds / 1000, "stream_sha256": rec["stream_sha256"],
                     "decoded_frames": rec["decoded_frames"], "deterministic": rec.get("deterministic"),
                     "encode_seconds": rec.get("encode_seconds"), "attempts": rec.get("attempts"), "scores": scores,
                     "reencoded": False})
        shutil.rmtree(pw, ignore_errors=True)
        progress(len(rows))
        write_json(stage_dir() / "h5.json", {**result, "seconds": round(time.time() - began, 1)})
    for v in venue:  # G6's published decode of the unfilled window (FFV1); never its DCVC-UF stream
        manifest = json.loads((Path(v["root"]) / "manifest.json").read_text())
        if not str(manifest.get("fill", "")).startswith("none"):
            raise SystemExit(f"{v['root']}: a filled background, not the venue-adapted whole frame")
        planes = read_ffv1(Path(v["root"]) / v["decoded"])[:w["frames"]]
        path = work / "venue.yuv"
        planes.tofile(path)
        scores = score_decoded(w, judges, path)
        rows.append({"codec": "dcvc-venue", "point": v["value"], "height": v["coded_size"][1], "method": v["method"],
                     "rate_bytes": v["bytes"], "kbps": v["kbps"], "stream_sha256": v["bitstream_sha256"],
                     "decoded_frames": len(planes), "deterministic": v.get("deterministic"),
                     "model_sha256": v["model_sha256"], "manifest_job": v["manifest_job"], "scores": scores,
                     "reencoded": False})
        progress(len(rows))
        write_json(stage_dir() / "h5.json", {**result, "seconds": round(time.time() - began, 1)})
    write_json(stage_dir() / "h5.json", {**result, "seconds": round(time.time() - began, 1)})
    return 0


def validate_anchors(result: dict[str, Any]) -> dict[str, bool]:
    rows = result["rows"]

    def falls(a: dict[str, Any], b: dict[str, Any]) -> bool:  # SVT-AV1: CRF up, rate down; DCVC-UF: QP up, rate up
        if (a["codec"], a["height"]) != (b["codec"], b["height"]) or a["point"] >= b["point"]:
            return True
        return a["kbps"] >= b["kbps"] if a["codec"].startswith("svtav1") else a["kbps"] <= b["kbps"]

    return {
        "window_is_the_released_one": result["jpeg_gate_holds"],
        "every_point_scored": len(rows) == result["planned"] and result["planned"] > 0,
        "every_frame_decoded": all(r["decoded_frames"] == result["frames"] for r in rows),
        "dcvc_deterministic": all(r["deterministic"] in (None, True) for r in rows),
        "scores_plausible": all(scores_plausible(r["scores"]) for r in rows),
        "rate_moves_with_the_knob": all(falls(a, b) for a in rows for b in rows),
    }


def command_score(args: argparse.Namespace) -> int:
    from experiments.background import g5c, g6

    routes = set(args.routes.split(",")) if args.routes else set()
    scratch = scratch_dir()
    work = scratch / "score"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    began = time.time()
    w = load_window(args, work)
    judges = Judges(args)
    mask_payload, mask_record = None, None
    shown_classes = w["classes"]
    if args.mask_scale > 0:  # H5m: the client's F̂ from the coded class map replaces F
        shown_classes, mask_payload, mask_record = coded_classes(w["classes"], args.mask_scale, w["fps"])
    alpha = np.stack([cp.alpha_and_classes(c)[0] for c in shown_classes])
    fg_points, manifests = [], []
    for directory in args.foreground:  # several foreground jobs' points (a grid extended later)
        root = Path(directory) / "publish" if (Path(directory) / "publish").is_dir() else Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["window"] != w["item"]["id"] or (args.limit_frames <= 0
                                                     and manifest["classes_sha256"] != w["classes_sha256"]):
            raise SystemExit("the foreground was coded for another window or another F")
        if manifest.get("mask_scale", 0) not in (0, args.mask_scale):  # a stream built for F̂ needs that F̂
            raise SystemExit(f"{directory}: foreground built for F at 1/{manifest['mask_scale']}, "
                             f"scored at 1/{args.mask_scale}")
        manifests.append(manifest)
        fg_points += [{**p, "root": str(root), "manifest_job": manifest.get("job_id"),
                       "built_for_mask_scale": manifest.get("mask_scale", 0)} for p in manifest["points"]]
    if args.foreground_points:
        fg_points = [p for p in fg_points if p["name"] in set(args.foreground_points.split(","))]
    if len({p["key"] for p in fg_points}) != len(fg_points):
        raise SystemExit("a foreground point appears in two manifests")
    fg_manifest = manifests[0]
    backgrounds = load_backgrounds(args.background, routes, w["item"]["id"]) if routes else []
    if args.background_points:
        wanted = set(args.background_points.split(","))
        backgrounds = [b for b in backgrounds if b["key"] in wanted]
    f_sha = hashlib.sha256(np.packbits(w["classes"] > 0).tobytes()).hexdigest()
    same_f = all(b["f_sha256"].get("f_packbits_sha256") == f_sha for b in backgrounds)
    if args.limit_frames <= 0 and not same_f:
        raise SystemExit("a background was filled for another F than the foreground's")
    pairs = [(b, f) for b in backgrounds for f in fg_points]
    if args.pairs:
        wanted = set(args.pairs.split(","))
        pairs = [(b, f) for b, f in pairs if f"{b['key']}+{f['key']}" in wanted]
    if args.max_combos > 0:  # smoke: a spread of pairs, every route
        pairs = [p for i, p in enumerate(pairs) if i % max(1, len(pairs) // args.max_combos) == 0][:args.max_combos]
    checkpoints = checkpoint_dir()
    restored = g5c.restore(checkpoints)
    fg_cache: dict[str, np.ndarray] = {}
    rows, checks = [], {}
    seconds = w["frames"] / float(w["fps"])
    done = 0
    for bg_key in dict.fromkeys(b["key"] for b, _ in pairs):
        bg = next(b for b, _ in pairs if b["key"] == bg_key)
        bg_rgb = None
        for _, fg in [p for p in pairs if p[0]["key"] == bg_key]:
            key = f"{bg_key}+{fg['key']}"
            c = container_for(w, bg, fg, Path(fg["root"]), mask_payload)
            data = c.pack()
            if bg["route"] not in checks:  # rule (d), once per route
                checks[bg["route"]] = {"pair": key, **bitstream_check(c, data, bg, fg, Path(fg["root"]))}
                (scratch / "publish" / "containers").mkdir(parents=True, exist_ok=True)
                (scratch / "publish" / "containers" / f"{key}.pstm").write_bytes(data)
            if key in restored:
                rows.append({**restored[key], "restored_from_checkpoint": True})
            else:
                t0 = time.perf_counter()
                if bg_rgb is None:
                    bg_rgb = rgb_gpu(read_ffv1(Path(bg["root"]) / bg["decoded"])[:w["frames"]])
                if fg["key"] not in fg_cache:
                    fg_cache[fg["key"]] = rgb_gpu(read_ffv1(Path(fg["root"]) / fg["decoded"])[:w["frames"]])
                if "gpu_compositor" not in checks:
                    checks["gpu_compositor"] = gpu_matches_cpu(fg_cache[fg["key"]], bg_rgb, alpha)
                t1 = time.perf_counter()
                out = work / "composite.yuv"
                composite_sha = composite_window(w, fg_cache[fg["key"]], bg_rgb, alpha, out)
                t2 = time.perf_counter()
                parts = c.breakdown()
                row = {"pair": key, "route": bg["route"], "background": bg_key, "foreground": fg["key"],
                       "foreground_job": fg["manifest_job"], "mask_scale": args.mask_scale, "tag": args.tag,
                       "foreground_built_for_mask_scale": fg["built_for_mask_scale"],
                       "bytes": len(data), "kbps": len(data) * 8 / seconds / 1000, "breakdown": parts,
                       "kbps_background": parts.get("background", 0) * 8 / seconds / 1000,
                       "kbps_foreground": parts.get("foreground", 0) * 8 / seconds / 1000,
                       "container_sha256": hashlib.sha256(data).hexdigest(), "composite_sha256": composite_sha,
                       "background_job": bg["manifest_job"], "scores": score_decoded(w, judges, out)}
                row["seconds"] = {"decode_files": round(t1 - t0, 2), "composite": round(t2 - t1, 2),
                                  **row["scores"].pop("seconds")}
                g5c.save(checkpoints, key, row)
                rows.append(row)
            done += 1
            progress(done)
    anchors = score_anchors(args, w, judges, work / "anchors") if args.anchors else []
    write_json(stage_dir() / "h5.json", {
        "kind": "score", "item": w["item"]["id"], "frames": w["frames"], "limit_frames": args.limit_frames,
        "routes": sorted(routes), "source": {k: v for k, v in w["source"].items() if k != "jpeg_gate"},
        "jpeg_gate_holds": all(c["holds"] for c in w["source"]["jpeg_gate"]), "masks": w["masks"],
        "classes_sha256": w["classes_sha256"], "foreground_manifest": {k: fg_manifest.get(k) for k in (
            "job_id", "git_head", "mask_cost", "classes_sha256")},
        "backgrounds": [{k: b.get(k) for k in ("key", "route", "kbps", "bytes", "bitstream_sha256", "manifest_job",
                                                "manifest_git_head", "model_sha256")} for b in backgrounds],
        "pairs_planned": len(pairs), "background_f_is_foreground_f": same_f, "bitstream_checks": checks,
        "mask": mask_record, "rows": rows, "anchors": anchors,
        "judges": judges.record, "seconds": round(time.time() - began, 1), **g5c.gpu_record(), **g6.provenance()})
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ----------------------------------------------------------------- visual check (H5p 5, CPU)

CROP_MIN = 384  # F's box is widened to at least this (px, at 1080p) so a crop shows context
CROP_PAD = 32
PANEL_FULL = (640, 360)


def parse_match(spec: str) -> dict[str, Any]:
    """``BG+FG|VENUE_KEY|HEIGHT=CRF``: a PointStream pair, the venue anchor's point and an SVT-AV1 ladder point
    near its rate."""
    pair, venue, svt = spec.split("|")
    height, crf = svt.split("=")
    if "+" not in pair or int(height) not in LADDER:
        raise SystemExit(f"{spec}: not PAIR|VENUE|HEIGHT=CRF")
    return {"pair": pair, "venue": venue, "svt": (int(height), int(crf))}


def crop_box(f: np.ndarray, size: tuple[int, int] = FULL) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): F's bounding box padded and widened to `CROP_MIN`, inside the frame; the centre if F is
    empty."""
    w, h = size
    ys, xs = np.nonzero(f)
    if len(xs):
        x0, x1, y0, y1 = xs.min() - CROP_PAD, xs.max() + CROP_PAD, ys.min() - CROP_PAD, ys.max() + CROP_PAD
    else:
        x0, x1, y0, y1 = w // 2, w // 2, h // 2, h // 2
    box = []
    for lo, hi, limit in ((x0, x1, w), (y0, y1, h)):
        span = max(hi - lo, CROP_MIN)
        centre = (lo + hi) // 2
        lo = int(np.clip(centre - span // 2, 0, max(limit - span, 0)))
        box.append((lo, min(lo + span, limit)))
    return box[0][0], box[1][0], box[0][1], box[1][1]


def labelled(image: np.ndarray, text: str) -> np.ndarray:
    import cv2

    bar = np.full((28, image.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([bar, image], axis=0)


def panel_grid(frames: dict[str, np.ndarray], box: tuple[int, int, int, int],
               crop_width: int = PANEL_FULL[0]) -> np.ndarray:
    """One row of F's crop and one of the whole frame, a column per named RGB frame (label: name)."""
    import cv2

    x0, y0, x1, y1 = box
    crop_height = max(1, round((y1 - y0) * crop_width / (x1 - x0)))
    top, bottom = [], []
    for name, rgb in frames.items():
        top.append(labelled(cv2.resize(np.ascontiguousarray(rgb[y0:y1, x0:x1]), (crop_width, crop_height),
                                       interpolation=cv2.INTER_AREA), name))
        whole = cv2.resize(rgb, PANEL_FULL, interpolation=cv2.INTER_AREA)
        sx, sy = PANEL_FULL[0] / FULL[0], PANEL_FULL[1] / FULL[1]
        cv2.rectangle(whole, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx) - 1, int(y1 * sy) - 1), (255, 0, 0), 1)
        bottom.append(whole)
    def row(panels: list[np.ndarray]) -> np.ndarray:
        gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
        return np.concatenate([x for p in panels for x in (p, gap)][:-1], axis=1)

    row1 = row(top)
    row2 = row([cv2.resize(b, (crop_width, round(PANEL_FULL[1] * crop_width / PANEL_FULL[0]))) for b in bottom])
    return np.concatenate([row1, np.full((6, row1.shape[1], 3), 255, np.uint8), row2], axis=0)


def command_crops(args: argparse.Namespace) -> int:
    """H5p 5: side-by-side crops of PointStream and the anchors at matched rates, rebuilt on the CPU from the
    published decodes (the composite as `score` makes it; the venue anchor's FFV1; SVT-AV1 at the equal-time
    preset re-encoded at the matched ladder point). Writes PNGs to ``publish/crops`` and ``h5.json``."""
    import cv2

    from experiments.background import g6
    from experiments.visor import b2

    began = time.time()
    scratch = scratch_dir()
    work = scratch / "crops"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    out_dir = scratch / "publish" / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    w = load_window(args, work)
    frames = w["frames"]
    shown_t = [int(t) for t in args.frames.split(",") if int(t) < frames]
    seconds = frames / float(w["fps"])
    shown_classes, mask_payload, mask_record = w["classes"], None, None
    if args.mask_scale > 0:
        shown_classes, mask_payload, mask_record = coded_classes(w["classes"], args.mask_scale, w["fps"])
    alpha = np.stack([cp.alpha_and_classes(shown_classes[t])[0] for t in shown_t])
    fg_points = {}
    for directory in args.foreground:
        root = Path(directory) / "publish" if (Path(directory) / "publish").is_dir() else Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["window"] != w["item"]["id"] or manifest.get("mask_scale", 0) not in (0, args.mask_scale):
            raise SystemExit(f"{directory}: another window, or a stream built for another F̂")
        for p in manifest["points"]:
            fg_points[p["key"]] = {**p, "root": str(root), "manifest_job": manifest.get("job_id")}
    backgrounds = {b["key"]: b for b in load_backgrounds(args.background, {"scene", "enh", "svt"}, w["item"]["id"])}
    venue = {v["key"]: v for v in load_backgrounds(args.venue, {"scene"}, w["item"]["id"])}
    coder = argparse.Namespace(encoder=args.encoder, decoder=args.decoder, threads=args.threads)
    rows = []
    for spec in args.match:
        m = parse_match(spec)
        bg_key, fg_key = m["pair"].split("+")
        bg, fg, ven = backgrounds[bg_key], fg_points[fg_key], venue[m["venue"]]
        c = container_for(w, bg, fg, Path(fg["root"]), mask_payload)
        ps_kbps = len(c.pack()) * 8 / seconds / 1000

        def at(planes: np.ndarray) -> np.ndarray:
            return g6.to_rgb(np.ascontiguousarray(planes[shown_t]), True)

        fg_rgb = at(read_ffv1(Path(fg["root"]) / fg["decoded"]))
        bg_rgb = at(read_ffv1(Path(bg["root"]) / bg["decoded"]))
        ps = cp.composite(fg_rgb, bg_rgb, alpha)
        venue_rgb = at(read_ffv1(Path(ven["root"]) / ven["decoded"]))
        height, crf = m["svt"]
        pw = work / f"svt-{height}-{crf}"
        shutil.rmtree(pw, ignore_errors=True)
        pw.mkdir(parents=True)
        rec = b2.code_svtav1(coder, ladder_source(w, height, work), w["fps"], crf, pw, None,
                             {"preset": args.preset, "enable_tf": None}, None, None)
        decoded = to_full(Path(rec["decoded"]), height, frames)
        svt_planes = np.fromfile(decoded, np.uint8).reshape(frames, FULL[1] * 3 // 2, FULL[0])
        svt_rgb = at(svt_planes)
        svt_kbps = rec["rate_bytes"] * 8 / seconds / 1000
        shutil.rmtree(pw, ignore_errors=True)
        pngs = []
        for i, t in enumerate(shown_t):
            panels = {"source": w["rgb"][t], f"PointStream {ps_kbps:.0f} kbps": ps[i],
                      f"DCVC-UF venue {ven['kbps']:.0f} kbps": venue_rgb[i],
                      f"SVT-AV1 p{args.preset} {height}p {svt_kbps:.0f} kbps": svt_rgb[i]}
            grid = panel_grid(panels, crop_box(w["classes"][t] > 0))
            name = f"{w['item']['id'].rsplit('_', 1)[0]}_{m['pair']}_f{t:03d}.png"
            cv2.imwrite(str(out_dir / name), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            pngs.append({"frame": t, "png": f"crops/{name}", "sha256": file_sha256(out_dir / name),
                         "shape": list(grid.shape), "std": float(grid.std())})
        rows.append({"match": spec, "pair": m["pair"], "pointstream_kbps": ps_kbps, "venue_key": m["venue"],
                     "venue_kbps": ven["kbps"], "svt": {"height": height, "crf": crf, "kbps": svt_kbps,
                                                        "stream_sha256": rec["stream_sha256"],
                                                        "decoded_frames": rec["decoded_frames"]},
                     "foreground_job": fg["manifest_job"], "background_job": bg["manifest_job"],
                     "venue_job": ven["manifest_job"], "pngs": pngs})
        progress(len(rows))
    write_json(stage_dir() / "h5.json", {
        "kind": "crops", "item": w["item"]["id"], "frames": frames, "limit_frames": args.limit_frames,
        "jpeg_gate_holds": all(c["holds"] for c in w["source"]["jpeg_gate"]), "shown_frames": shown_t,
        "mask_scale": args.mask_scale, "mask": mask_record, "planned": len(args.match), "rows": rows,
        "preset": args.preset, "encoder": b2.svtav1.tool(args.encoder), "decoder": b2.svtav1.tool(args.decoder),
        "seconds": round(time.time() - began, 1), **g6.provenance()})
    shutil.rmtree(work, ignore_errors=True)
    return 0


def validate_crops(result: dict[str, Any]) -> dict[str, bool]:
    rows = result["rows"]
    return {
        "window_is_the_released_one": result["jpeg_gate_holds"],
        "every_match_rendered": len(rows) == result["planned"] > 0,
        "every_frame_shown": all(len(r["pngs"]) == len(result["shown_frames"]) > 0 for r in rows),
        "panels_are_pictures": all(p["std"] > 5 and p["shape"][2] == 3 for r in rows for p in r["pngs"]),
        "rates_recorded": all(r["pointstream_kbps"] > 0 and r["venue_kbps"] > 0 and r["svt"]["kbps"] > 0 for r in rows),
        "svt_decoded_every_frame": all(r["svt"]["decoded_frames"] == result["frames"] for r in rows),
    }


# ----------------------------------------------------------------- timing (exclusive GPU)

#: The operating points timed per window: (background method, value, foreground point). One per route.
TIMED = (("scene-1080", 38, "dcvc@21/2"), ("enh-1080", 48, "dcvc@21/4"))
REPEATS = 3


def synchronized(fn: Any) -> tuple[Any, float]:
    """``fn()`` and its wall seconds, the GPU synchronized before and after."""
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def dcvc_worker(action: str, plan: dict[str, Any], work: Path) -> dict[str, Any]:
    """One DCVC-UF worker run; its report (``model_*_seconds``: the codec calls, GPU-synchronized, B2b's
    codec time)."""
    import subprocess

    import torch

    torch.cuda.empty_cache()  # the worker is a separate process; this process's cached blocks would crowd it

    from src.codecs.dcvc_uf_worker import dcvc_command

    work = work / f"{action}-{Path(plan['container']).stem}"  # one directory per call: the worker never overwrites
    work.mkdir(parents=True, exist_ok=False)
    plan_path, report = work / "plan.json", work / "report.json"
    write_json(plan_path, plan)
    command, env, cwd = dcvc_command(Path(sys.prefix), action, plan_path, report)
    done = subprocess.run(command, env={**os.environ, **env}, cwd=cwd, capture_output=True, text=True, timeout=3600)
    if done.returncode:
        raise RuntimeError(f"DCVC-UF {action} failed ({done.returncode}):\n{done.stderr[-4000:]}")
    return json.loads(report.read_text())


def dcvc_plan(args: argparse.Namespace, video_ckpt: str, qp: int, size: tuple[int, int], frames: int, source: Path,
              container: Path, out: Path) -> dict[str, Any]:
    return {"structure": "htl", "qp": qp, "frame_count": frames, "height": size[1], "width": size[0],
            "src_type": "yuv420", "frames_file": str(source), "container": str(container), "out_file": str(out),
            "image_ckpt": args.image_ckpt, "image_sha256": file_sha256(Path(args.image_ckpt)), "video_ckpt": video_ckpt,
            "video_sha256": file_sha256(Path(video_ckpt)), "profile": False}


def median(values: list[float]) -> float:
    return float(np.median(values))


def time_window(args: argparse.Namespace, w: dict[str, Any], method: str, value: int, fg_name: str, nets: dict[str, Any],
                scene_ckpt: str, enhancers: Path, work: Path, done: int = 0) -> dict[str, Any]:
    """PointStream's encoder and decoder at one operating point, each stage timed ``REPEATS`` times."""
    import torch

    from experiments.background import g5, g5d, g6
    from experiments.visor import h3d
    from src.codecs import svtav1

    frames, fps = w["frames"], w["fps"]
    f = w["classes"] > 0
    cpus = sorted(os.sched_getaffinity(0))[:args.threads]  # type: ignore[attr-defined,unused-ignore]
    t: dict[str, list[float]] = {}

    def note(name: str, seconds: float) -> None:
        t.setdefault(name, []).append(seconds)

    route = ROUTE_OF_METHOD[method]
    size = FULL if method.endswith("1080") else g6.HALF
    qp, scale = parse_point(fg_name)
    fg_size = coded_size(scale)
    containers = []
    for repeat in range(REPEATS):
        rw = work / f"r{repeat}"
        shutil.rmtree(rw, ignore_errors=True)
        rw.mkdir(parents=True)
        # ---- encoder (the main process's cached GPU blocks are released before each subprocess stage, untimed)
        torch.cuda.empty_cache()
        (filled, _), seconds = synchronized(lambda: g6.fill(w["rgb"], f, args, rw / "fill"))
        note("encode/fill (ProPainter, wall with its model load)", seconds)
        bg_source = rw / "bg.yuv"
        g5.rgb_to_yuv420(filled if size == FULL else g6.resize(filled, size, area=True)).tofile(bg_source)
        if route == "scene":
            rep = dcvc_worker("encode", dcvc_plan(args, scene_ckpt, value, size, frames, bg_source, rw / "bg.psdc",
                                                  rw / "bg-dec.yuv"), rw)
            note("encode/background (scene model, DCVC-UF codec calls)", rep["model_encode_seconds"])
            bg_stream = (rw / "bg.psdc").read_bytes()
            codec = "dcvc-uf"
        else:
            (rw / "svt").mkdir()
            rec = svtav1.code(bg_source, rw / "svt", width=FULL[0], height=FULL[1], fps=fps, frames=frames, crf=value,
                              preset=g5.PRESET, threads=args.threads, full_range=True, encoder="SvtAv1EncApp",
                              decoder="dav1d", cpus=cpus)
            note(f"encode/background (SVT-AV1 preset {g5.PRESET}, {args.threads} threads, wall)", rec["encode_seconds"])
            bg_stream = bs.ivf_to_obus(Path(rec["stream"]).read_bytes())[0]
            codec = "av1"
        padded, seconds = synchronized(lambda: cp.foreground_input(w["rgb"], f, scale, fg_size))
        note("encode/foreground padding (CPU)", seconds)
        g5.rgb_to_yuv420(padded).tofile(rw / "fg.yuv")
        rep = dcvc_worker("encode", dcvc_plan(args, args.video_ckpt, qp, fg_size, frames, rw / "fg.yuv", rw / "fg.psdc",
                                              rw / "fg-dec.yuv"), rw)
        note("encode/foreground (DCVC-UF codec calls)", rep["model_encode_seconds"])
        c = bs.Container(FULL[0], FULL[1], fps, frames)
        c.add_coded(bs.Coded("background", bs.ROUTES[route], codec, size[0], size[1], bg_stream))
        c.add_coded(bs.Coded("foreground", scale, "dcvc-uf", fg_size[0], fg_size[1], (rw / "fg.psdc").read_bytes(),
                             ((bs.CLASSES["hand"], 1 if fg_name in nets else 0), (bs.CLASSES["object"], 0))))
        data = c.pack()
        containers.append({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        # ---- decoder, from the container's own streams
        back = bs.Container.unpack(data)
        [bg], [fg] = back.coded(bs.Tag.BACKGROUND), back.coded(bs.Tag.FOREGROUND)
        if bg.codec == "dcvc-uf":
            (rw / "bg-in.psdc").write_bytes(bg.stream)
            rep = dcvc_worker("decode", dcvc_plan(args, scene_ckpt, value, size, frames, bg_source, rw / "bg-in.psdc",
                                                  rw / "bg-out.yuv"), rw)
            note("decode/background (scene model, DCVC-UF codec calls)", rep["passes"][0]["model_decode_seconds"])
            bg_yuv = np.fromfile(rw / "bg-out.yuv", np.uint8).reshape(frames, size[1] * 3 // 2, size[0])
            bg_rgb, seconds = synchronized(lambda: g5.yuv420_to_rgb(bg_yuv) if size == FULL else
                                           g6.resize(g5.yuv420_to_rgb(bg_yuv), FULL, area=False))
            note("decode/background to RGB (and upsampling)", seconds)
        else:
            (rw / "bg-in.ivf").write_bytes(bs.obus_to_ivf(bg.stream, bg.width, bg.height, fps))
            timing = svtav1.run(svtav1.decode_command("dav1d", rw / "bg-in.ivf", rw / "bg-out.yuv", threads=8), 3600,
                                cpus)
            note("decode/background (dav1d, 8 threads, wall)", timing["wall"])
            plain = g5.yuv420_to_rgb(np.fromfile(rw / "bg-out.yuv", np.uint8).reshape(frames, FULL[1] * 3 // 2, FULL[0]))
            state = torch.load(enhancers / f"crf{value}.pt", map_location="cuda", weights_only=True)
            net = g5d.enhancer(g5d.TRAIN["blocks"], g5d.TRAIN["channels"]).cuda()
            net.load_state_dict({k: v.float() for k, v in state.items()})
            bg_rgb, seconds = synchronized(lambda: g5d.apply(net.eval(), plain, "cuda"))
            note("decode/background enhancer (G5d)", seconds)
            del net, state
        (rw / "fg-in.psdc").write_bytes(fg.stream)
        rep = dcvc_worker("decode", dcvc_plan(args, args.video_ckpt, qp, fg_size, frames, rw / "fg.yuv", rw / "fg-in.psdc",
                                              rw / "fg-out.yuv"), rw)
        note("decode/foreground (DCVC-UF codec calls)", rep["passes"][0]["model_decode_seconds"])
        fg_yuv = np.fromfile(rw / "fg-out.yuv", np.uint8).reshape(frames, fg_size[1] * 3 // 2, fg_size[0])
        up, seconds = synchronized(lambda: h3d.upscale_all(g5.yuv420_to_rgb(fg_yuv), frames, FULL[0], FULL[1]))
        note("decode/foreground to RGB and bicubic upsampling", seconds)
        (alpha, nearest), seconds = synchronized(lambda: tuple(np.stack(x) for x in zip(
            *[cp.alpha_and_classes(cl) for cl in w["classes"]])))
        note("decode/alpha and nearest class (from the granted F)", seconds)
        slots = {cp.HAND: hand_enhancer(nets[fg_name][0], "cuda")} if fg_name in nets else {}
        fg_rgb, seconds = synchronized(lambda: cp.apply_slots(up, nearest, slots))
        note("decode/foreground slots (G on the hands)", seconds)
        _, seconds = synchronized(lambda: g5.rgb_to_yuv420(np.concatenate(
            [cp.composite(fg_rgb[i:i + CHUNK], bg_rgb[i:i + CHUNK], alpha[i:i + CHUNK]) for i in range(0, frames, CHUNK)])))
        note("decode/composite and 4:2:0 output", seconds)
        shutil.rmtree(rw, ignore_errors=True)
        progress(done + repeat + 1)  # one unit per repeat: an operating point takes about 25 minutes
    stages: dict[str, dict[str, Any]] = {k: {"median_s": median(v), "runs_s": [round(x, 3) for x in v]}
                                         for k, v in t.items()}
    encode = sum(s["median_s"] for k, s in stages.items() if k.startswith("encode/"))
    decode = sum(s["median_s"] for k, s in stages.items() if k.startswith("decode/"))
    return {"method": method, "value": value, "route": route, "foreground": fg_name, "stages": stages,
            "encode_s_per_window": encode, "decode_s_per_window": decode,
            "decode_fps": frames / decode if decode else None, "containers": containers}


def command_time(args: argparse.Namespace) -> int:
    import torch

    from experiments.background import g5c, g6
    from experiments.visor.h3d_component import generic_network

    if os.environ.get("PS_GPU_SHARING") == "shared":
        raise SystemExit("timing runs only on an exclusive GPU")
    scratch = scratch_dir()
    work = scratch / "time"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    began = time.time()
    w = load_window(args, work)
    arm = Path(args.arm) / "publish" / "finetune"
    scene = json.loads((arm / "result.json").read_text())
    if file_sha256(arm / "video_ft.pth.tar") != scene["model"]["checkpoint_sha256"]:
        raise RuntimeError("the scene model is not the one its training recorded")
    enhancers = Path(args.enhancers) / "publish" / "enhancers"
    timed = [t for t in TIMED if ROUTE_OF_METHOD[t[0]] in args.routes.split(",")]
    timed = timed[:1] if args.limit_frames > 0 else timed
    nets = {fg: generic_network(args.generic, fg, "cuda") for _, _, fg in timed if fg in G_POINTS}
    torch.backends.cudnn.benchmark = False
    rows = []
    for n, (method, value, fg) in enumerate(timed):
        rows.append(time_window(args, w, method, value, fg, nets, str(arm / "video_ft.pth.tar"), enhancers,
                                work / f"{method}-{value}", n * REPEATS))
    write_json(stage_dir() / "h5.json", {
        "kind": "time", "item": w["item"]["id"], "frames": w["frames"], "limit_frames": args.limit_frames,
        "routes": args.routes.split(","), "timed": len(timed),
        "jpeg_gate_holds": all(c["holds"] for c in w["source"]["jpeg_gate"]), "threads": args.threads,
        "scene_model_sha256": scene["model"]["checkpoint_sha256"], "rows": rows, "not_timed": [
            "F and the class map (granted: dataset masks)", "venue models' training (offline, once per venue)"],
        "gpu_sharing": os.environ.get("PS_GPU_SHARING", "exclusive"), "seconds": round(time.time() - began, 1),
        **g5c.gpu_record(), **g6.provenance()})
    shutil.rmtree(work, ignore_errors=True)
    return 0


def validate_time(result: dict[str, Any]) -> dict[str, bool]:
    rows = result["rows"]
    return {
        "window_is_the_released_one": result["jpeg_gate_holds"],
        "exclusive_gpu": result["gpu_sharing"] != "shared",
        "every_point_timed": len(rows) == result["timed"] and len(rows) > 0,
        "repeats": all(len(s["runs_s"]) == REPEATS for r in rows for s in r["stages"].values()),
        "containers_reproducible": all(len({c["sha256"] for c in r["containers"]}) == 1 for r in rows),
        "times_positive": all(s["median_s"] > 0 for r in rows for s in r["stages"].values()),
        "encoder_and_decoder": all(r["encode_s_per_window"] > 0 and r["decode_s_per_window"] > 0 for r in rows),
    }


# ----------------------------------------------------------------- validation


def finite(value: Any, low: float, high: float) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(value) and low <= value <= high


def scores_plausible(s: dict[str, Any]) -> bool:
    return all(finite(s[m]["wpsnr"], 15, 70) and finite(s[m]["wlpips"], 0, 1) and finite(s[m]["wdists"], 0, 1)
               for m in MASK_SETS) and finite(s["frame"]["psnr"], 15, 70)


def validate_foreground(result: dict[str, Any]) -> dict[str, bool]:
    rows = result["points"]
    by_scale: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_scale.setdefault(r["scale"], []).append(r)
    return {
        "window_is_the_released_one": result["jpeg_gate_holds"],
        "every_point_coded": len(rows) > 0 and all(r["deterministic"] and r["decoder_matches_encoder_intra"]
                                                   for r in rows),
        "rate_rises_with_qp": all(a["kbps"] < b["kbps"] for s in by_scale.values()  # DCVC-UF: QP up, rate up
                                  for a, b in zip(sorted(s, key=lambda r: r["qp"]), sorted(s, key=lambda r: r["qp"])[1:])),
        "foreground_decodes_to_a_picture": all(r["psnr_f"] is not None and 20 < r["psnr_f"] < 60 for r in rows),
        "g_on_its_points_only": all((r["slots"]["hand"] is not None) == (r["name"] in G_POINTS) for r in rows),
        "object_slot_as_declared": all(r["slots"].get("object") is None or (
            r["name"] in G_POINTS and r["slots"]["object"].get("arm") == "G_HO") for r in rows),
        "g_changes_hands": all(r["changed_by_slots"] > 0 for r in rows if r["slots"]["hand"]),
        "frames_published": all(len(r["decoded_frame_sha256"]) == result["frames"] for r in rows),
        "mask_cost_measured": result["mask_cost"]["bytes"] > 0 and 0 < result["mask_cost"]["share"] < 0.8,
        "full_window": result["limit_frames"] > 0 or result["frames"] == 240,
        "venue_model_is_the_windows": venue_model_ok(result),
        # H5p 4: a stream built for F̂ took F̂ from a MASK payload that decodes exactly, and F̂ is close to F
        "f_hat_as_the_client_decodes": not result.get("mask_scale") or (
            result["mask"]["lossless_on_reduced_grid"] and result["mask"]["scale"] == result["mask_scale"]
            and result["mask"]["iou"] > 0.7),
    }


def venue_model_ok(result: dict[str, Any]) -> bool:
    """H5q 1: a foreground coded with a venue model names the whole-frame model of its own video, converged."""
    venue = (result.get("video_model") or {}).get("venue")
    if venue is None:
        return True
    converged = venue["converged"]  # G6's monitor record, or a flag
    converged = converged.get("converged") if isinstance(converged, dict) else converged
    return venue["fill"] == "none" and converged is True and \
        venue["model_sha256"] == result["video_model"]["video_ckpt_sha256"] and \
        result["item"].startswith(venue["id"].split("/")[-1] + "_")


def validate_score(result: dict[str, Any]) -> dict[str, bool]:
    rows, anchors = result["rows"], result["anchors"]
    checks = {
        "window_is_the_released_one": result["jpeg_gate_holds"],
        "every_pair_scored": len(rows) == result["pairs_planned"],
        "scores_plausible": all(scores_plausible(r["scores"]) for r in rows + anchors),
        "rate_is_the_container": all(r["bytes"] == sum(r["breakdown"].values()) and r["kbps"] > 0 for r in rows),
        "bitstream_holds": bool(result["bitstream_checks"]) == bool(result["routes"])
        and all(all(v for k, v in c.items() if k != "pair") for r, c in result["bitstream_checks"].items()
                if r != "gpu_compositor"),
        "gpu_compositor_is_the_cpus": not rows or result["bitstream_checks"].get("gpu_compositor", {}).get(
            "max_abs", 99) <= 1,
        "foreground_decodes_on_f": all(r["scores"][FILL]["psnr_fg"] > 15 for r in rows),
        "background_f_is_foreground_f": result["limit_frames"] > 0 or result["background_f_is_foreground_f"],
        "mask_billed": not result.get("mask") or (
            result["mask"]["lossless_on_reduced_grid"] and all(r["breakdown"].get("mask", 0) > result["mask"]["bytes"]
                                                                for r in rows)),
    }
    if anchors:
        full = result["limit_frames"] <= 0
        checks["anchors_deterministic"] = all(a["deterministic"] in (None, True) for a in anchors)
        # SVT-AV1: B2's own stream decoded; DCVC-UF: B2's point re-encoded (its match with B2's bytes reported)
        checks["anchors_are_b2s_points"] = all(a["b2"] is not None and a["b2"]["stream_sha256"] == a["b2_stream_sha256"]
                                               and (a["reencoded"] or a["stream_sha256"] == a["b2_stream_sha256"])
                                               for a in anchors)
        checks["anchors_reproduce_b2"] = all(
            abs(a["scores"][DENSE]["wpsnr"] - a["b2"]["wpsnr"]) <= ANCHOR_WPSNR_TOLERANCE_DB
            and (a["reencoded"] or abs(a["kbps"] - a["b2"]["kbps"]) < 1e-6 * a["kbps"]) for a in anchors) if full else True
    return checks


def validate(stage: Path) -> dict[str, bool]:
    result = json.loads((stage / "h5.json").read_text())
    if result["kind"] == "time":
        return validate_time(result)
    if result["kind"] == "model-size":
        return validate_model_size(result)
    if result["kind"] == "mask-video":
        return validate_mask_video(result)
    if result["kind"] == "anchors":
        return validate_anchors(result)
    if result["kind"] == "crops":
        return validate_crops(result)
    return validate_foreground(result) if result["kind"] == "foreground" else validate_score(result)


def load_classes(args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray]:
    """The item and its class map from the fill mask set alone (no source decode)."""
    from experiments.background import g6
    from experiments.visor.b2 import mask_root
    from src.segmentation.masks import ClipMasks
    from src.segmentation.visor import HANDS

    item = g6.load_item(args.eval_set, args.item)
    frames = args.limit_frames if args.limit_frames > 0 else int(item["frames"])
    clip = ClipMasks.load(mask_root(Path(args.fill_masks), item["id"]))
    classes = np.zeros((frames, FULL[1], FULL[0]), np.uint8)
    for t in range(frames):
        classes[t][clip.foreground(t)] = cp.OBJECT
        classes[t][clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1])] = cp.HAND
    return item, classes


def command_mask_video(args: argparse.Namespace) -> int:
    """H5p 2: F coded as a video (SVT-AV1, motion compensated) against H5m's lzma coder, by rate and by how
    closely the client's F̂ matches F. CPU only."""
    from src.codecs import svtav1
    from src.pointstream import mask as mk

    began = time.time()
    item, classes = load_classes(args)
    fps = Fraction(item["fps"]).limit_denominator(1001)
    seconds = len(classes) / float(fps)
    work = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch") / "mask-video"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for scale in (int(x) for x in args.lzma_scales.split(",")):
        payload = mk.encode(mk.reduce(classes, scale), scale)
        decoded, _ = mk.decode(payload)
        rows.append({"coder": "lzma", "scale": scale, "bytes": len(payload), "kbps": len(payload) * 8 / seconds / 1000,
                     **mk.accuracy(classes, mk.upsample(decoded, FULL))})
        progress(len(rows))
    for representation in args.representations.split(","):
        for scale in (int(x) for x in args.scales.split(",")):
            pw, ph = mk.padded_size(FULL[0], FULL[1], scale)
            planes = mk.video_planes(classes, scale, representation)
            source = work / f"{representation}-{scale}.yuv"
            planes.tofile(source)
            for crf in (int(x) for x in args.crfs.split(",")):
                point = work / f"{representation}-{scale}-{crf}"
                point.mkdir()
                coded = svtav1.code(source, point, width=pw, height=ph, fps=fps, frames=len(classes), crf=crf,
                                    preset=args.preset, threads=args.threads, full_range=True,
                                    encoder=args.encoder, decoder=args.decoder)
                back = np.fromfile(coded["decoded"], np.uint8).reshape(len(classes), -1)
                rows.append({"coder": "svtav1", "representation": representation, "scale": scale, "crf": crf,
                             "coded_size": [pw, ph], "bytes": coded["payload_bytes"],
                             "kbps": coded["payload_bytes"] * 8 / seconds / 1000,
                             "stream_sha256": coded["stream_sha256"], "decoded_frames": coded["decoded_frames"],
                             "encode_seconds": coded["encode_seconds"],
                             **mk.accuracy(classes, mk.video_classes(back, scale, FULL))})
                shutil.rmtree(point)
                progress(len(rows))
            source.unlink()
    write_json(stage_dir() / "h5.json", {
        "kind": "mask-video", "item": item["id"], "frames": len(classes), "fps": str(fps),
        "limit_frames": args.limit_frames, "preset": args.preset, "share": float((classes > 0).mean()),
        "classes_sha256": hashlib.sha256(np.packbits(classes > 0).tobytes()
                                         + np.packbits(classes == cp.HAND).tobytes()).hexdigest(),
        "encoder": svtav1.tool(args.encoder), "decoder": svtav1.tool(args.decoder), "rows": rows,
        "seconds": round(time.time() - began, 1)})
    return 0


def validate_mask_video(result: dict[str, Any]) -> dict[str, bool]:
    rows = result["rows"]
    video = [r for r in rows if r["coder"] == "svtav1"]
    lossless = [r for r in rows if r["coder"] == "lzma"]
    return {
        "both_coders": bool(video) and bool(lossless),
        "every_frame_decoded": all(r["decoded_frames"] == result["frames"] for r in video),
        "accuracy_plausible": all(0 < r["iou"] <= 1 and 0 <= r["class_accuracy"] <= 1 for r in rows),
        # at one scale and representation, a higher CRF never costs more
        "rate_falls_with_crf": all(a["bytes"] >= b["bytes"] for a in video for b in video
                                   if (a["representation"], a["scale"]) == (b["representation"], b["scale"])
                                   and a["crf"] < b["crf"]),
        "lzma_finer_is_closer": all(a["iou"] >= b["iou"] for a in lossless for b in lossless if a["scale"] < b["scale"]),
    }


MODEL_VARIANTS = ("delta-4", "delta-8")  # G5c: delta-4 is recipe L's shipped size (keeps quality within +2.2%)


def command_model_size(args: argparse.Namespace) -> int:
    """H5p 1: what each scene model costs to send (G5c's coder: the change from the public HT-L weights the client
    has, quantized per output channel, LZMA per tensor). CPU only."""
    from experiments.background.g5c import compress_state, load_plain

    began = time.time()
    public_sha = file_sha256(Path(args.video_ckpt))
    public = load_plain(Path(args.video_ckpt))
    rows = []
    for i, arm in enumerate(args.arm):
        ft_dir = Path(arm) / "publish" / "finetune"
        trained = json.loads((ft_dir / "result.json").read_text())
        checkpoint = ft_dir / "video_ft.pth.tar"
        sha = file_sha256(checkpoint)
        if sha != trained["model"]["checkpoint_sha256"]:
            raise RuntimeError(f"{arm}: the published checkpoint is not the one its result recorded")
        if trained["base_checkpoints"]["video_sha256"] != public_sha:
            raise SystemExit(f"{arm}: fine-tuned from other public weights than {args.video_ckpt}")
        ft = load_plain(checkpoint)
        names = sorted(ft)[:args.limit_tensors] if args.limit_tensors > 0 else sorted(ft)
        ft, base = {k: ft[k] for k in names}, {k: public[k] for k in names}
        sizes = [compress_state(ft, base, v, workers=args.threads)[1] for v in args.variants.split(",")]
        rows.append({"item": trained["item"], "arm": trained["arm"], "train_seconds": trained.get("train_seconds"),
                     "checkpoint_sha256": sha, "checkpoint_bytes": checkpoint.stat().st_size,
                     "parameters": trained["model"]["parameters"], "tensors": len(names), "sizes": sizes})
        progress(i + 1)
    write_json(stage_dir() / "h5.json", {"kind": "model-size", "limit_tensors": args.limit_tensors,
                                         "public_sha256": public_sha, "rows": rows,
                                         "seconds": round(time.time() - began, 1)})
    return 0


def validate_model_size(result: dict[str, Any]) -> dict[str, bool]:
    rows = result["rows"]
    return {
        "every_arm_sized": bool(rows) and all(len(r["sizes"]) == len(MODEL_VARIANTS) for r in rows),
        "distinct_models": len({r["checkpoint_sha256"] for r in rows}) == len(rows),
        "fewer_bits_smaller": all(r["sizes"][0]["bytes"] < r["sizes"][1]["bytes"] < r["checkpoint_bytes"] for r in rows),
        # a change from the public weights, so smaller than the 16-bit whole model (parameters x 2 bytes)
        "change_below_whole16": all(s["bytes"] < 2 * r["parameters"] for r in rows for s in r["sizes"])
        if result["limit_tensors"] <= 0 else True,
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(os.environ.get("PS_STAGE_DIR") or ".")
    checks = validate(stage)
    passed = bool(checks) and all(checks.values())
    out = Path(os.environ.get("PS_VALIDATION_PATH") or stage / "validation.json")
    write_json(out, {"passed": passed, "checks": checks})
    print(json.dumps({"passed": passed, "checks": checks}, indent=1))
    return 0 if passed else 1


# ----------------------------------------------------------------- report

#: name: (mask set or "frame", key, sign, minimum overlap). Higher sign·value is better.
METRICS: dict[str, tuple[str, str, float, float]] = {
    "wlpips_fill": (FILL, "wlpips", -1.0, 0.005), "wdists_fill": (FILL, "wdists", -1.0, 0.005),
    "wpsnr_fill": (FILL, "wpsnr", 1.0, 0.5), "wpsnr_dense": (DENSE, "wpsnr", 1.0, 0.5),
    "wlpips_dense": (DENSE, "wlpips", -1.0, 0.005), "psnr_frame": ("frame", "psnr", 1.0, 0.5),
    "lpips_frame": ("frame", "lpips", -1.0, 0.005), "dists_frame": ("frame", "dists", -1.0, 0.005),
}
HEADLINE = "wlpips_fill"
#: Rule (a), fixed before any run (docs/experiments.md, H5 entry).
DECISION = {"metric": HEADLINE, "anchor": "pooled", "mean_bd_at_most": -0.10, "items_below_zero": 3}
#: PointStream's variants: the background routes its curve may use.
VARIANTS = {"pointstream": ("scene", "enh"), "scene": ("scene",), "enh": ("enh",), "svt-background": ("svt",)}


def curve(points: list[dict[str, Any]], metric: str, extra_kbps: float = 0.0) -> list[tuple[float, float]]:
    where, key, sign, _ = METRICS[metric]
    return [(p["kbps"] + extra_kbps, sign * p["scores"][where][key]) for p in points
            if p["scores"][where].get(key) is not None]


def compare(test: list[tuple[float, float]], anchor: list[tuple[float, float]], minimum: float) -> dict[str, Any]:
    """BD-rate of ``test``'s upper concave hull against ``anchor``'s (PCHIP of log rate over the quality range
    both cover); otherwise whether test is better or worse at every rate both reach."""
    import math

    from scipy.interpolate import PchipInterpolator

    from experiments.background.g5 import envelope

    t, a = envelope(test), envelope(anchor)
    if not t or not a:
        return {"bd": None, "where": "no points"}
    lo, hi = max(t[0][1], a[0][1]), min(t[-1][1], a[-1][1])
    if len(t) >= 2 and len(a) >= 2 and hi - lo >= minimum:
        def area(points: list[tuple[float, float]]) -> float:
            f = PchipInterpolator([q for _, q in points], [math.log(r) for r, _ in points])
            return float(f.integrate(lo, hi))

        return {"bd": math.exp((area(t) - area(a)) / (hi - lo)) - 1.0, "range": [lo, hi], "where": "overlap"}
    sides = set()
    for rate, q in t:
        if a[0][0] <= rate:
            at = a[-1][1] if rate >= a[-1][0] else float(np.interp(math.log(rate), [math.log(r) for r, _ in a],
                                                                     [x for _, x in a]))
            sides.add("better" if q > at else "worse")
    where = f"{sides.pop()} at every rate" if len(sides) == 1 else (
        "no shared rate" if not sides else "the curves cross without a shared range")
    return {"bd": None, "where": where}


def below_zero(c: dict[str, Any]) -> bool:
    return (c["bd"] is not None and c["bd"] < 0) or c["where"] == "better at every rate"


def matched_regions(points: list[dict[str, Any]], anchor: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """At each anchor point's rate, PointStream's best pair on the headline metric at or under that rate: its
    foreground and background scores beside the anchor's (rule (b))."""
    where, key, sign, _ = METRICS[HEADLINE]
    out = []
    for a in sorted(anchor, key=lambda x: x["kbps"]):
        cheaper = [p for p in points if p["kbps"] <= a["kbps"]]
        if not cheaper:
            continue
        best = max(cheaper, key=lambda p: sign * p["scores"][where][key])
        out.append({"anchor": f"{a['codec']}-{a['point']:g}", "anchor_kbps": a["kbps"], "pair": best["pair"],
                    "kbps": best["kbps"], **{f"{side}_{k}": {"pointstream": best["scores"][FILL][f"{k}_{side}"],
                                                             "anchor": a["scores"][FILL][f"{k}_{side}"]}
                                             for side in ("fg", "bg") for k in ("lpips", "psnr", "dists")}})
    return out


#: A tagged foreground swap falls back to this variant's rows on the points it did not recode (H5q 1: the venue
#: model's foreground, on G_HO's PointStream).
BASE_OF_TAG = {"venuefg": "gho"}


def variant_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """H5's variants (F granted) by background route, and H5m's: PointStream with F coded at 1/s, and the hull
    over every s."""
    granted = [r for r in rows if not r.get("mask_scale") and not r.get("tag")]
    out = {v: [r for r in granted if r["route"] in routes] for v, routes in VARIANTS.items()}
    tags = {r["tag"] for r in rows if r.get("tag") and not r.get("mask_scale")}  # coded-F rows go to mask{s} below
    for tag in sorted(tags, key=lambda t: t in BASE_OF_TAG):
        # a foreground swap: its rows, and its base variant's (H5's, or G_HO's for H5q) for the other points
        tagged = [r for r in rows if r.get("tag") == tag and not r.get("mask_scale")]
        swapped = {r["foreground"] for r in tagged}
        base = out.get(BASE_OF_TAG.get(tag, "pointstream"), out["pointstream"])
        out[tag] = tagged + [r for r in base if r["foreground"] not in swapped]
    coded = [r for r in rows if r.get("mask_scale")]
    for s in sorted({r["mask_scale"] for r in coded}):
        out[f"mask{s}"] = [r for r in coded if r["mask_scale"] == s]
    if coded:
        out["mask-hull"] = coded
    return out


def billed(rows: list[dict[str, Any]], model_bytes: dict[str, float], seconds: float) -> list[dict[str, Any]]:
    """H5p 1: each pair with its background route's model bits spread over ``seconds`` of stream."""
    return [{**r, "kbps": r["kbps"] + 8 * model_bytes.get(r["route"], 0.0) / seconds / 1000} for r in rows]


def payback(points: list[dict[str, Any]], anchors: list[dict[str, Any]], model_bytes: dict[str, float]) -> list[dict[str, Any]]:
    """At each point of ``points``' hull (headline metric) inside the anchors' quality range: the rate it saves
    against the anchors' hull at equal quality, and the seconds of stream its route's model takes to repay."""
    import math

    from experiments.background.g5 import envelope

    a = envelope(curve(anchors, HEADLINE))
    out = []
    for rate, q in envelope(curve(points, HEADLINE)):
        if not a or not a[0][1] <= q <= a[-1][1]:
            continue
        row = next(r for r in points if r["kbps"] == rate and METRICS[HEADLINE][2] * r["scores"][FILL]["wlpips"] == q)
        anchor_kbps = math.exp(float(np.interp(q, [x for _, x in a], [math.log(r) for r, _ in a])))
        saving = anchor_kbps - rate
        bits = 8 * model_bytes.get(row["route"], 0.0)
        out.append({"pair": row["pair"], "route": row["route"], "kbps": rate, "anchor_kbps": anchor_kbps,
                    "saving_kbps": saving, "payback_seconds": bits / (saving * 1000) if saving > 0 else None})
    return out


def item_report(rows: list[dict[str, Any]], anchors: list[dict[str, Any]], mask_kbps: float,
                billing: dict[str, Any] | None = None) -> dict[str, Any]:
    from experiments.background.g5 import envelope

    groups = {"svtav1": [a for a in anchors if a["codec"].startswith("svtav1")],
              "dcvc": [a for a in anchors if a["codec"] in ("dcvc", "dcvc-ladder")],
              "venue": [a for a in anchors if a["codec"] == "dcvc-venue"], "pooled": anchors}
    out: dict[str, Any] = {"comparisons": {}, "hull": {}, "mask_kbps": mask_kbps}
    for variant, points in variant_rows(rows).items():
        if not points:
            continue
        for metric, (_, _, _, minimum) in METRICS.items():
            for name, group in groups.items():
                if group:
                    out["comparisons"][f"{variant}|{name}|{metric}"] = compare(curve(points, metric), curve(group, metric),
                                                                             minimum)
            if variant == "pointstream" and groups["pooled"]:  # F and its classes billed too
                out["comparisons"][f"pointstream+mask|pooled|{metric}"] = compare(
                    curve(points, metric, mask_kbps), curve(groups["pooled"], metric), minimum)
        hull = envelope(curve(points, HEADLINE))
        on = sorted((r for r in points for k, q in hull if r["kbps"] == k and -r["scores"][FILL]["wlpips"] == q),
                    key=lambda r: r["kbps"])
        out["hull"][variant] = [{"pair": r["pair"], "kbps": r["kbps"], "wlpips": r["scores"][FILL]["wlpips"],
                                 "background_share": r["kbps_background"] / r["kbps"]} for r in on]
    if billing and groups["pooled"]:  # H5p 1: the venue models' bits paid over a video or a kitchen
        variants = variant_rows(rows)
        out["payback"] = {}
        for variant in ("pointstream", "gho"):
            if not variants.get(variant):
                continue
            out["payback"][variant] = payback(variants[variant], groups["pooled"], billing["bytes"])
            for basis, seconds in billing["seconds"].items():
                points = billed(variants[variant], billing["bytes"], seconds)
                # the venue-adapted anchor pays for its model the same way (H5p 3)
                venue = billed([{**a, "route": "venue"} for a in groups["venue"]], billing["bytes"], seconds)
                billed_groups = {"pooled": [a for a in anchors if a["codec"] != "dcvc-venue"] + venue,
                                 "venue": venue}
                for metric, (_, _, _, minimum) in METRICS.items():
                    for name, group in billed_groups.items():
                        if group and (name == "pooled" or groups["venue"]):
                            out["comparisons"][f"{variant}@{basis}|{name}|{metric}"] = compare(
                                curve(points, metric), curve(group, metric), minimum)
    pointstream = [r for r in rows if r["route"] in VARIANTS["pointstream"] and not r.get("mask_scale")]
    out["matched"] = matched_regions(pointstream, anchors) if pointstream and anchors else []
    out["matched_by_variant"] = {v: matched_regions(points, anchors) for v, points in variant_rows(rows).items()
                                 if (v in BASE_OF_TAG or v in BASE_OF_TAG.values()) and points and anchors}
    return out


def summary(items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from experiments.visor.h3 import bootstrap_mean

    import zlib

    keys = sorted({k for it in items.values() for k in it["comparisons"]})
    out: dict[str, Any] = {}
    for key in keys:
        cs = {i: it["comparisons"][key] for i, it in items.items() if key in it["comparisons"]}
        bds = np.array([c["bd"] for c in cs.values() if c["bd"] is not None], float)
        out[key] = {"per_item": {i: c["bd"] if c["bd"] is not None else c["where"] for i, c in cs.items()},
                    "items": len(cs), "items_below_zero": sum(below_zero(c) for c in cs.values()),
                    "bd": bootstrap_mean(bds, np.random.default_rng(zlib.crc32(key.encode())))  # per key: reproducible
                    if len(bds) else None, "bd_items": len(bds)}
    def verdict(variant: str) -> str:
        head = out.get(f"{variant}|{DECISION['anchor']}|{DECISION['metric']}")
        if not head or head["items"] < 4:
            return "incomplete"
        passes = head["bd"] is not None and head["bd"]["mean"] <= DECISION["mean_bd_at_most"] \
            and head["items_below_zero"] >= DECISION["items_below_zero"]
        if head["bd_items"] < head["items"]:  # an item without a number: the rule reads its side
            passes = head["items_below_zero"] >= DECISION["items_below_zero"] and (
                head["bd"] is None or head["bd"]["mean"] <= DECISION["mean_bd_at_most"])
        return "passes" if passes else "fails"

    variants = sorted({k.split("|")[0] for k in out})
    return {"comparisons": out, "decision": {**DECISION, "verdict": verdict("pointstream")},
            "verdicts": {v: verdict(v) for v in variants}}


def plot(items: dict[str, dict[str, Any]], data: dict[str, dict[str, Any]], target: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.background.g5 import envelope

    shown = ("wpsnr_fill", "wlpips_fill", "wdists_fill")
    fig, axes = plt.subplots(len(items), len(shown), figsize=(4.2 * len(shown), 3.2 * len(items)), squeeze=False)
    for i, item in enumerate(sorted(items)):
        rows, anchors = data[item]["rows"], data[item]["anchors"]
        for j, metric in enumerate(shown):
            ax = axes[i][j]
            sign = METRICS[metric][2]
            for label, pts, style, colour in (
                    ("PointStream", [r for r in rows if r["route"] in ("scene", "enh")], "-o", "tab:blue"),
                    ("PointStream, SVT-AV1 background", [r for r in rows if r["route"] == "svt"], ":.", "tab:gray"),
                    ("SVT-AV1", [a for a in anchors if a["codec"].startswith("svtav1")], "--s", "tab:orange"),
                    ("DCVC-UF HT-L", [a for a in anchors if a["codec"] in ("dcvc", "dcvc-ladder")], "--^", "tab:green"),
                    ("DCVC-UF, venue-adapted", [a for a in anchors if a["codec"] == "dcvc-venue"], "--v", "tab:red")):
                hull = envelope(curve(pts, metric))
                if hull:
                    ax.plot([r for r, _ in hull], [sign * q for _, q in hull], style, color=colour, label=label, ms=3)
            ax.set_xscale("log")
            ax.set_title(f"{item} {metric}", fontsize=8)
            ax.set_xlabel("kbps", fontsize=8)
            if j == 0:
                ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(target, dpi=110)


def command_report(args: argparse.Namespace) -> int:
    data: dict[str, dict[str, Any]] = {}
    for path in args.score:
        result = json.loads(Path(path).read_text())
        if result["kind"] not in ("score", "anchors") or result["limit_frames"] > 0:
            raise SystemExit(f"{path}: not a full score or anchors stage")
        d = data.setdefault(result["item"], {"rows": [], "anchors": [], "mask": None, "sources": []})
        if result["kind"] == "anchors":  # H5p 3: fair anchors only
            d["anchors"] += result["rows"]
            d["sources"].append({"path": path, "job_id": result.get("job_id"), "git_head": result.get("git_head")})
            continue
        seen = {(r["pair"], r.get("mask_scale", 0), r.get("tag", "")) for r in d["rows"]}  # once per scale and tag
        d["rows"] += [r for r in result["rows"] if (r["pair"], r.get("mask_scale", 0), r.get("tag", "")) not in seen]
        d["anchors"] += result["anchors"]
        d["mask"] = d["mask"] or result["foreground_manifest"]["mask_cost"]
        d["sources"].append({"path": path, "job_id": result.get("job_id"), "git_head": result.get("git_head"),
                             "gpu": result.get("gpu")})
    billing = json.loads(Path(args.billing).read_text()) if args.billing else {}
    if args.anchor_codecs:  # H5p 3: the fair set (e.g. without B2's preset-4 SVT-AV1)
        keep = set(args.anchor_codecs.split(","))
        for d in data.values():
            d["anchors"] = [a for a in d["anchors"] if a["codec"] in keep]
    items = {i: item_report(d["rows"], d["anchors"], d["mask"]["kbps"], billing.get(i)) for i, d in data.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"items": items, **summary(items), "sources": {i: d["sources"] for i, d in data.items()},
              "mask": {i: d["mask"] for i, d in data.items()}, "billing": billing}
    write_json(out / "h5-report.json", report)
    if not args.no_plot:
        plot(items, data, out / "h5-rd.png")
    head = report["comparisons"].get(f"pointstream|pooled|{HEADLINE}")
    print(json.dumps({"decision": report["decision"], "headline": head}, indent=1))
    return 0


# ----------------------------------------------------------------- command line


def add_window(p: argparse.ArgumentParser) -> None:
    for name in ("--eval-set", "--archive", "--dense-masks", "--fill-masks", "--video", "--item", "--image-ckpt",
                 "--video-ckpt", "--lpips-backbone"):
        p.add_argument(name, required=True)
    p.add_argument("--limit-frames", type=int, default=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("foreground")
    add_window(p)
    p.add_argument("--points", default=",".join(POINTS))
    p.add_argument("--generic", action="append", required=True, help="H3d's generic-network job archives")
    p.add_argument("--slots", choices=("G", "G_HO"), default="G", help="the enhancers in the class slots: H3d's G on "
                   "the hands, or H4c's G_HO on hands and objects")
    p.add_argument("--object-networks", action="append", default=[], help="H4c's networks job archives (G_HO)")
    p.add_argument("--video-arm", help="H5q 1: G6's `train --fill none` archive of the window's video (extracted); "
                   "its fine-tuned DCVC-UF HT-L replaces --video-ckpt")
    p.add_argument("--mask-scale", type=int, default=0, help="H5p 4: build the stream for the client's F̂ (F coded "
                   "at 1/s, `src.pointstream.mask`): K and the slots from F̂ (0: from F)")
    p = sub.add_parser("score")
    add_window(p)
    p.add_argument("--foreground", action="append", required=True, help="foreground jobs' archives (extracted)")
    p.add_argument("--foreground-points", help="only these foreground points (dcvc@QP/SCALE, comma separated)")
    p.add_argument("--background", action="append", default=[])
    p.add_argument("--routes", default="scene,enh")
    p.add_argument("--anchors", action="append", help="B2's job archives (extracted) with the item's streams")
    p.add_argument("--b2-result", action="append", help="those jobs' b2.json, for the reproduction check")
    p.add_argument("--dists-backbone", required=True)
    p.add_argument("--max-combos", type=int, default=0)
    p.add_argument("--background-points", help="only these background points (method-value, comma separated)")
    p.add_argument("--pairs", help="only these pairs (background+foreground keys, comma separated)")
    p.add_argument("--tag", default="", help="a variant's name for its rows (e.g. gho: G_HO in the foreground slots)")
    p.add_argument("--mask-scale", type=int, default=0, help="H5m: code F's class map at 1/s and composite with it "
                   "(0: F granted, H5)")
    p = sub.add_parser("time")
    add_window(p)
    for name in ("--arm", "--enhancers", "--propainter-tree", "--propainter-ckpt", "--raft-ckpt", "--flow-completion-ckpt"):
        p.add_argument(name, required=True)
    p.add_argument("--generic", action="append", required=True)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--routes", default="scene,enh", help="the background routes to time (one operating point each)")
    sub.add_parser("validate")
    p = sub.add_parser("report")
    p.add_argument("--score", action="append", required=True, help="full score stages' h5.json")
    p.add_argument("--out", required=True)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--billing", help="H5p 1: JSON {item: {bytes: {route: model bytes}, seconds: {basis: s}}}")
    p.add_argument("--anchor-codecs", help="only these anchor codecs (svtav1, dcvc, svtav1-p1, dcvc-ladder, "
                   "dcvc-venue)")
    p = sub.add_parser("model-size")
    p.add_argument("--arm", action="append", required=True, help="G6's train jobs' archives (extracted)")
    p.add_argument("--video-ckpt", required=True, help="the public DCVC-UF HT-L weights they were fine-tuned from")
    p.add_argument("--variants", default=",".join(MODEL_VARIANTS))
    p.add_argument("--limit-tensors", type=int, default=0, help="the first N tensors only (smoke)")
    p.add_argument("--threads", type=int, default=16)
    p = sub.add_parser("anchors")
    add_window(p)
    p.add_argument("--dists-backbone", required=True)
    p.add_argument("--svt", help="SVT-AV1 ladder: HEIGHT=CRF,...;HEIGHT=...")
    p.add_argument("--dcvc", help="DCVC-UF HT-L ladder: HEIGHT=QP,...;HEIGHT=...")
    p.add_argument("--preset", type=int, default=1, help="SVT-AV1 preset (H5's timing: 1 is the equal-time one)")
    p.add_argument("--venue", action="append", help="G6 format archives of the unfilled window (--fill none)")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--encoder", default="SvtAv1EncApp")
    p.add_argument("--decoder", default="dav1d")
    p = sub.add_parser("crops")
    add_window(p)
    p.add_argument("--foreground", action="append", required=True)
    p.add_argument("--background", action="append", required=True)
    p.add_argument("--venue", action="append", required=True, help="G6 format archives of the unfilled window")
    p.add_argument("--match", action="append", required=True, help="PAIR|VENUE_KEY|HEIGHT=CRF (see parse_match)")
    p.add_argument("--frames", default="60,180", help="frames shown")
    p.add_argument("--mask-scale", type=int, default=0)
    p.add_argument("--preset", type=int, default=1)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--encoder", default="SvtAv1EncApp")
    p.add_argument("--decoder", default="dav1d")
    p = sub.add_parser("mask-video")
    for name in ("--eval-set", "--fill-masks", "--item"):
        p.add_argument(name, required=True)
    p.add_argument("--limit-frames", type=int, default=0)
    p.add_argument("--representations", default="sdf,binary")
    p.add_argument("--scales", default="2,4,8")
    p.add_argument("--crfs", default="20,35,50")
    p.add_argument("--lzma-scales", default="4,8,16")
    p.add_argument("--preset", type=int, default=4)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--encoder", default="SvtAv1EncApp")
    p.add_argument("--decoder", default="dav1d")
    args = parser.parse_args(argv)
    if args.command == "mask-video":
        return command_mask_video(args)
    if args.command == "anchors":
        return command_anchors(args)
    if args.command == "crops":
        return command_crops(args)
    if args.command == "foreground":
        return command_foreground(args)
    if args.command == "score":
        return command_score(args)
    if args.command == "report":
        return command_report(args)
    if args.command == "time":
        return command_time(args)
    if args.command == "model-size":
        return command_model_size(args)
    return command_validate(args)


if __name__ == "__main__":
    sys.exit(main())
