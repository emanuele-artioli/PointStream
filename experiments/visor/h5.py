"""PLAN step H5: the first working PointStream on VISOR evaluation set v2 (docs/experiments.md, 2026-10-16).

    python -m experiments.visor.h5 foreground WINDOW --points dcvc@0/4,... --generic TAR ... --limit-frames N
    python -m experiments.visor.h5 score WINDOW --foreground DIR --background DIR ... --routes scene,enh,svt \\
        [--anchors DIR ... --b2-result JSON ...] --dists-backbone P --max-combos N --limit-frames N
    python -m experiments.visor.h5 time WINDOW --arm DIR --enhancers DIR --generic TAR ... --propainter-tree DIR \\
        --propainter-ckpt P --raft-ckpt P --flow-completion-ckpt P --threads 32
    python -m experiments.visor.h5 validate
    python -m experiments.visor.h5 report --score H5.JSON ... --out DIR

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
        return h3d.apply(net, frames, list(where), device)

    return run


def foreground_point(args: argparse.Namespace, w: dict[str, Any], name: str, nearest: np.ndarray,
                     nets: dict[str, Any], work: Path, points: Path) -> dict[str, Any]:
    """One foreground point: coded, decoded as the client does, kept as FFV1 with its stream."""
    from experiments.background import g5, g6
    from experiments.visor import h3d

    qp, scale = parse_point(name)
    size = coded_size(scale)
    key = point_key(name)
    frames = w["frames"]
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    padded = cp.foreground_input(w["rgb"], w["classes"] > 0, scale, size)
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
    network = None
    if name in nets:
        net, info = nets[name]
        slots[cp.HAND] = hand_enhancer(net, "cuda")
        network = {"point": name, "sha256": info["sha256"]}
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
            "slots": {"hand": network, "object": None}, "psnr_f": psnr_f,
            "changed_by_slots": float((out != upsampled).any(axis=3).mean()),
            "decoded": f"points/{key}/decoded.mkv", "decoded_ffv1_sha256": file_sha256(target / "decoded.mkv"),
            "decoded_frame_sha256": digests, "bitstream": f"points/{key}/stream.psdc",
            "bitstream_sha256": file_sha256(target / "stream.psdc"),
            "seconds": {"pad": round(t1 - t0, 2), "code": round(t2 - t1, 2), "client": round(t3 - t2, 2)}}


def command_foreground(args: argparse.Namespace) -> int:
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
    nearest = np.stack([cp.alpha_and_classes(c)[1] for c in w["classes"]])
    cost = mask_cost(w["classes"], w["fps"], work)
    nets = {n: generic_network(args.generic, n, "cuda") for n in names if n in G_POINTS}
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
                **g6.provenance(), **g5c.gpu_record(), "points": rows}
    write_json(publish / "manifest.json", manifest)
    write_json(stage_dir() / "h5.json", {
        "kind": "foreground", "item": item["id"], "frames": w["frames"], "limit_frames": args.limit_frames,
        "source": {k: v for k, v in w["source"].items() if k != "jpeg_gate"},
        "jpeg_gate_holds": all(c["holds"] for c in w["source"]["jpeg_gate"]), "masks": w["masks"],
        "mask_cost": cost, "points": rows, "manifest_sha256": file_sha256(publish / "manifest.json"),
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


def container_for(w: dict[str, Any], bg: dict[str, Any], fg: dict[str, Any], fg_root: Path) -> bs.Container:
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
    hand = fg["slots"]["hand"]
    slots = ((bs.CLASSES["hand"], 1 if hand else 0), (bs.CLASSES["object"], 0))
    c.add_coded(bs.Coded("foreground", fg["scale"], "dcvc-uf", fg["coded_size"][0], fg["coded_size"][1], fg_data,
                         slots))
    if bg["model_sha256"]:
        c.add_model(bs.ModelRef("background", bg["model_sha256"]))
    if hand:
        c.add_model(bs.ModelRef("foreground", hand["sha256"]))
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
    alpha = np.stack([cp.alpha_and_classes(c)[0] for c in w["classes"]])
    fg_points, manifests = [], []
    for directory in args.foreground:  # several foreground jobs' points (a grid extended later)
        root = Path(directory) / "publish" if (Path(directory) / "publish").is_dir() else Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["window"] != w["item"]["id"] or (args.limit_frames <= 0
                                                     and manifest["classes_sha256"] != w["classes_sha256"]):
            raise SystemExit("the foreground was coded for another window or another F")
        manifests.append(manifest)
        fg_points += [{**p, "root": str(root), "manifest_job": manifest.get("job_id")} for p in manifest["points"]]
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
            c = container_for(w, bg, fg, Path(fg["root"]))
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
                       "foreground_job": fg["manifest_job"],
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
        "pairs_planned": len(pairs), "background_f_is_foreground_f": same_f, "bitstream_checks": checks, "rows": rows, "anchors": anchors,
        "judges": judges.record, "seconds": round(time.time() - began, 1), **g5c.gpu_record(), **g6.provenance()})
    shutil.rmtree(work, ignore_errors=True)
    return 0


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

    from src.codecs.dcvc_uf_worker import dcvc_command

    plan_path, report = work / f"plan-{action}.json", work / f"{action}.json"
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
                scene_ckpt: str, enhancers: Path, work: Path) -> dict[str, Any]:
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
        # ---- encoder
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
    timed = [t for t in TIMED if args.limit_frames <= 0 or t == TIMED[0]]
    nets = {fg: generic_network(args.generic, fg, "cuda") for _, _, fg in timed if fg in G_POINTS}
    torch.backends.cudnn.benchmark = False
    rows = []
    for done, (method, value, fg) in enumerate(timed, 1):
        rows.append(time_window(args, w, method, value, fg, nets, str(arm / "video_ft.pth.tar"), enhancers,
                                work / f"{method}-{value}"))
        progress(done)
    write_json(stage_dir() / "h5.json", {
        "kind": "time", "item": w["item"]["id"], "frames": w["frames"], "limit_frames": args.limit_frames,
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
        "every_point_timed": len(rows) == (len(TIMED) if result["limit_frames"] <= 0 else 1),
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
        "g_changes_hands": all(r["changed_by_slots"] > 0 for r in rows if r["slots"]["hand"]),
        "frames_published": all(len(r["decoded_frame_sha256"]) == result["frames"] for r in rows),
        "mask_cost_measured": result["mask_cost"]["bytes"] > 0 and 0 < result["mask_cost"]["share"] < 0.8,
        "full_window": result["limit_frames"] > 0 or result["frames"] == 240,
    }


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
    return validate_foreground(result) if result["kind"] == "foreground" else validate_score(result)


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


def item_report(rows: list[dict[str, Any]], anchors: list[dict[str, Any]], mask_kbps: float) -> dict[str, Any]:
    from experiments.background.g5 import envelope

    groups = {"svtav1": [a for a in anchors if a["codec"] == "svtav1"],
              "dcvc": [a for a in anchors if a["codec"] == "dcvc"], "pooled": anchors}
    out: dict[str, Any] = {"comparisons": {}, "hull": {}, "mask_kbps": mask_kbps}
    for variant, routes in VARIANTS.items():
        points = [r for r in rows if r["route"] in routes]
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
        on = [r for r in points for k, q in hull if r["kbps"] == k and
              -r["scores"][FILL]["wlpips"] == q]
        out["hull"][variant] = [{"pair": r["pair"], "kbps": r["kbps"], "wlpips": r["scores"][FILL]["wlpips"],
                                 "background_share": r["kbps_background"] / r["kbps"]} for r in on]
    pointstream = [r for r in rows if r["route"] in VARIANTS["pointstream"]]
    out["matched"] = matched_regions(pointstream, anchors) if pointstream and anchors else []
    return out


def summary(items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from experiments.visor.h3 import bootstrap_mean

    rng = np.random.default_rng(5)
    keys = sorted({k for it in items.values() for k in it["comparisons"]})
    out: dict[str, Any] = {}
    for key in keys:
        cs = {i: it["comparisons"][key] for i, it in items.items() if key in it["comparisons"]}
        bds = np.array([c["bd"] for c in cs.values() if c["bd"] is not None], float)
        out[key] = {"per_item": {i: c["bd"] if c["bd"] is not None else c["where"] for i, c in cs.items()},
                    "items": len(cs), "items_below_zero": sum(below_zero(c) for c in cs.values()),
                    "bd": bootstrap_mean(bds, rng) if len(bds) else None, "bd_items": len(bds)}
    head = out.get(f"pointstream|{DECISION['anchor']}|{DECISION['metric']}")
    verdict = "incomplete"
    if head and head["items"] >= 4:
        passes = head["bd"] is not None and head["bd"]["mean"] <= DECISION["mean_bd_at_most"] \
            and head["items_below_zero"] >= DECISION["items_below_zero"]
        if head["bd_items"] < head["items"]:  # an item without a number: the rule reads its side
            passes = head["items_below_zero"] >= DECISION["items_below_zero"] and (
                head["bd"] is None or head["bd"]["mean"] <= DECISION["mean_bd_at_most"])
        verdict = "passes" if passes else "fails"
    return {"comparisons": out, "decision": {**DECISION, "verdict": verdict}}


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
            for label, pts, style in (("PointStream", [r for r in rows if r["route"] in ("scene", "enh")], "-o"),
                                      ("SVT-AV1 bg", [r for r in rows if r["route"] == "svt"], ":."),
                                      ("SVT-AV1", [a for a in anchors if a["codec"] == "svtav1"], "--s"),
                                      ("DCVC-UF", [a for a in anchors if a["codec"] == "dcvc"], "--^")):
                hull = envelope(curve(pts, metric))
                if hull:
                    ax.plot([r for r, _ in hull], [sign * q for _, q in hull], style, label=label, ms=3)
            ax.set_xscale("log")
            ax.set_title(f"{item} {metric}", fontsize=8)
            ax.set_xlabel("kbps", fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(target, dpi=110)


def command_report(args: argparse.Namespace) -> int:
    data: dict[str, dict[str, Any]] = {}
    for path in args.score:
        result = json.loads(Path(path).read_text())
        if result["kind"] != "score" or result["limit_frames"] > 0:
            raise SystemExit(f"{path}: not a full score stage")
        d = data.setdefault(result["item"], {"rows": [], "anchors": [], "mask": None, "sources": []})
        seen = {r["pair"] for r in d["rows"]}
        d["rows"] += [r for r in result["rows"] if r["pair"] not in seen]
        d["anchors"] += result["anchors"]
        d["mask"] = result["foreground_manifest"]["mask_cost"]
        d["sources"].append({"path": path, "job_id": result.get("job_id"), "git_head": result.get("git_head"),
                             "gpu": result.get("gpu")})
    items = {i: item_report(d["rows"], d["anchors"], d["mask"]["kbps"]) for i, d in data.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"items": items, **summary(items), "sources": {i: d["sources"] for i, d in data.items()},
              "mask": {i: d["mask"] for i, d in data.items()}}
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
    p = sub.add_parser("time")
    add_window(p)
    for name in ("--arm", "--enhancers", "--propainter-tree", "--propainter-ckpt", "--raft-ckpt", "--flow-completion-ckpt"):
        p.add_argument(name, required=True)
    p.add_argument("--generic", action="append", required=True)
    p.add_argument("--threads", type=int, default=32)
    sub.add_parser("validate")
    p = sub.add_parser("report")
    p.add_argument("--score", action="append", required=True, help="full score stages' h5.json")
    p.add_argument("--out", required=True)
    p.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "foreground":
        return command_foreground(args)
    if args.command == "score":
        return command_score(args)
    if args.command == "report":
        return command_report(args)
    if args.command == "time":
        return command_time(args)
    return command_validate(args)


if __name__ == "__main__":
    sys.exit(main())
