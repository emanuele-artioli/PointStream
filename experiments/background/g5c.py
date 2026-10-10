"""PLAN step G5c, part 1: checks on G5b's scene models without training.

    python -m experiments.background.g5c checks --prepared DIR --clips CLIPS.JSON --visor-fill DIR --select ID \\
        --svt DIR ... --dcvc DIR --arms DIR ... --image-ckpt PATH --video-ckpt PATH \\
        --lpips-backbone PATH --dists-backbone PATH --limit-frames N
    python -m experiments.background.g5c compress --prepared DIR --clips CLIPS.JSON --visor-fill DIR --select ID \\
        --arm DIR --variants whole-16,whole-8,... --image-ckpt PATH --video-ckpt PATH \\
        --lpips-backbone PATH --dists-backbone PATH --limit-frames N
    python -m experiments.background.g5c fill --prepared DIR --clips CLIPS.JSON --visor-fill DIR --select ID,ID \\
        --propainter-tree DIR --propainter-ckpt P --raft-ckpt P --flow-completion-ckpt P --lpips-backbone PATH --dists-backbone PATH --limit-frames N
    python -m experiments.background.g5c validate
    python -m experiments.background.g5c report --result g5c.json ... --out DIR

``checks`` decodes every stored stream of a G5b pilot excerpt (G5's SVT-AV1
streams, G5b's unadapted DCVC-UF streams and each fine-tuned arm's streams
with its published checkpoint), nothing re-encoded, and scores each as G5b
(`experiments.background.g5.score`) plus DISTS on V (`src.codecs.quality.masked_dists`)
and flicker on V (`flicker`). ``compress`` sends a fine-tuned model smaller:
the whole model or its change from the public HT-L weights, quantized per
output channel and LZMA-coded (`compress_state`), and re-codes the excerpt
with each variant through B2's worker. ``fill`` compares the foreground
fill every method codes: OpenCV Telea per frame (G5's ``filled``) against
ProPainter (`experiments.background.propainter_fill`, in its tree), each
coded by SVT-AV1 at G5's CRFs and scored on V. Each writes ``g5c.json`` in
``PS_STAGE_DIR``; ``report`` applies `DECISION` (docs/experiments.md, G5c).
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from experiments.background import g1, g5, g5_cond
from experiments.background.g5b import find_stored
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

#: The decision rules, fixed before any fleet run (docs/experiments.md, G5c entry).
DECISION: dict[str, Any] = {
    "gate_tier": "dataset",
    "component": "scene-lpips",  # G5b's β = 0.026 scene model
    "dists_transfer_clip": "visor-long/P06_03",  # where the component beat SVT-AV1 on LPIPS_V
    "flicker_clips": ["visor-long/P26_02", "visor-long/P06_03"],
    "keeps_quality_bd": 0.05,  # a size variant's BD-rate on LPIPS_V against the 16-bit model's curve
    "fill_wins_bd": -0.02,  # ProPainter's fill replaces Telea's if SVT-AV1 saves this much on LPIPS_V on every clip
}
VARIANTS = ["whole-16", "whole-8", "whole-4", "whole-2", "delta-8", "delta-4", "delta-2",
            "delta-8-s0.1", "delta-4-s0.1", "delta-2-s0.1"]
QUANT_MIN_VALUES = 1024  # smaller tensors (biases, norms, per-QP scales) stay at fp16
LUMA = (0.2126, 0.7152, 0.0722)


# ----------------------------------------------------------------- DISTS and flicker on V


def dists_v(clip: g5.Clip, rgb: np.ndarray, net: Any, device: str) -> dict[str, float | None]:
    """Per tier: mean over its frames of DISTS on V, the foreground pasted back from the source."""
    import torch

    from src.codecs.quality import masked_dists

    out: dict[str, float | None] = {}
    for tier, (frames, masks) in clip.scored.items():
        values = []
        for first in range(0, len(frames), 4):
            idx = frames[first:first + 4]
            mask = masks[first:first + 4]
            pasted = np.where(clip.keep[idx][..., None], rgb[idx], clip.frames[idx])
            x = torch.from_numpy(pasted).to(device).permute(0, 3, 1, 2).float() / 255.0
            y = torch.from_numpy(clip.frames[idx]).to(device).permute(0, 3, 1, 2).float() / 255.0
            m = torch.from_numpy(mask).to(device)[:, None]
            d = masked_dists(net, x, y, m).cpu().numpy()
            values += [float(v) for v, k in zip(d, mask) if k.any()]
        out[tier] = float(np.mean(values)) if values else None
    return out


def source_flows(clip: g5.Clip) -> np.ndarray:
    """DIS flow on the source from each frame to the one before (frame 0 has none)."""
    return g5_cond.oracle_flows(clip.frames, [None] + list(range(clip.n - 1)))


def flicker(clip: g5.Clip, rgb: np.ndarray, flows: np.ndarray, device: str) -> dict[str, float | None]:
    """Temporal residual on V against the source's: for t ≥ 1, (decoded t − decoded t−1 warped by the source's
    flow) − (source t − source t−1 warped the same way), on luma (8-bit units), mean absolute value over the
    pixels in V at t and at t−1 (warped) and inside the image; averaged over pairs. Also each residual's mean
    absolute value. Decoded frames have the foreground pasted back, as for scoring."""
    import torch

    luma = torch.tensor(LUMA, device=device).view(1, 3, 1, 1)
    pasted = np.where(clip.keep[..., None], rgb, clip.frames)
    per_pair: list[float] = []
    dec_energy: list[float] = []
    src_energy: list[float] = []
    for first in range(1, clip.n, 16):
        idx = list(range(first, min(first + 16, clip.n)))
        prev = [i - 1 for i in idx]
        flow = torch.from_numpy(flows[idx].astype(np.float32)).to(device)

        def tensor(a: np.ndarray) -> Any:
            return torch.from_numpy(a).to(device).permute(0, 3, 1, 2).float()

        stacked = torch.cat([tensor(pasted[prev]), tensor(clip.frames[prev]),
                             torch.from_numpy(clip.keep[prev]).to(device)[:, None].float().expand(-1, 3, -1, -1)], 1)
        warped, valid = g5_cond.warp(stacked, flow)
        dec_prev, src_prev, keep_prev = warped[:, :3], warped[:, 3:6], warped[:, 6:7]
        region = torch.from_numpy(clip.keep[idx]).to(device)[:, None] & (keep_prev > 0.999) & (valid > 0)
        r_dec = ((tensor(pasted[idx]) - dec_prev) * luma).sum(1, keepdim=True)
        r_src = ((tensor(clip.frames[idx]) - src_prev) * luma).sum(1, keepdim=True)
        count = region.flatten(1).sum(1).float()
        for name, values in (("pair", (r_dec - r_src).abs()), ("dec", r_dec.abs()), ("src", r_src.abs())):
            means = (values * region).flatten(1).sum(1) / count.clamp_min(1)
            target = {"pair": per_pair, "dec": dec_energy, "src": src_energy}[name]
            target += [float(v) for v, c in zip(means.cpu(), count.cpu()) if c > 0]
    return {"flicker_v": float(np.mean(per_pair)) if per_pair else None,
            "residual_decoded": float(np.mean(dec_energy)) if dec_energy else None,
            "residual_source": float(np.mean(src_energy)) if src_energy else None, "pairs": len(per_pair)}


def full_score(clip: g5.Clip, rgb: np.ndarray, nets: SimpleNamespace, flows: np.ndarray) -> dict[str, Any]:
    """G5's score (PSNR and LPIPS on V per tier) with DISTS on V per tier and flicker on V added."""
    scored = g5.strip_frames(g5.score(clip, rgb, nets.lpips, "cuda"))
    for tier, value in dists_v(clip, rgb, nets.dists, "cuda").items():
        scored[tier]["dists_v"] = value
    scored["flicker"] = flicker(clip, rgb, flows, "cuda")
    return scored


def load_nets(args: argparse.Namespace) -> SimpleNamespace:
    from src.codecs import quality

    return SimpleNamespace(lpips=g5.load_lpips(args.lpips_backbone, "cuda"),
                           dists=quality.load_dists(Path(args.dists_backbone), "cuda"),
                           record={"lpips": quality.lpips_record(Path(args.lpips_backbone)),
                                   "dists": quality.dists_record(Path(args.dists_backbone))})


# ----------------------------------------------------------------- decoding stored streams


def decode_dcvc(image_ckpt: str, video_ckpt: str, container: Path, work: Path) -> dict[str, Any]:
    """B2's worker decode of a stored DCVC-UF container (twice, from its bytes and the checkpoints only)."""
    from src.codecs.dcvc_uf_worker import dcvc_command

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    plan = {"structure": "htl", "container": str(container), "src_type": "yuv420", "out_file": str(work / "out.yuv"),
            "image_ckpt": image_ckpt, "image_sha256": file_sha256(Path(image_ckpt)), "video_ckpt": video_ckpt,
            "video_sha256": file_sha256(Path(video_ckpt)), "profile": False}
    write_json(work / "plan.json", plan)
    command, env, cwd = dcvc_command(Path(sys.prefix), "decode", work / "plan.json", work / "report.json")
    done = subprocess.run(command, env={**os.environ, **env, "PYTHONFAULTHANDLER": "1"}, cwd=cwd, capture_output=True,
                          text=True, timeout=1800)
    if done.returncode:
        return {"returncode": done.returncode, "stderr_tail": done.stderr[-3000:]}
    report = json.loads((work / "report.json").read_text())
    return {"returncode": 0, "decoded": str(work / "out.yuv"), "deterministic": report["deterministic"],
            "frame_count": report["frame_count"], "video_sha256": plan["video_sha256"]}


def read_decoded(path: Path, frames: int, n: int) -> np.ndarray:
    raw = np.fromfile(path, np.uint8)
    if raw.size != frames * g5.WIDTH * g5.HEIGHT * 3 // 2:
        raise RuntimeError(f"{path}: {raw.size} bytes for {frames} frames")
    return g5.yuv420_to_rgb(raw.reshape(frames, g5.HEIGHT * 3 // 2, g5.WIDTH)[:n])


def stored_methods(args: argparse.Namespace, clip_id: str) -> list[dict[str, Any]]:
    """Every stored stream of the clip: (method, its points with stream path, recorded bytes and rate)."""
    methods = []
    stored = find_stored([Path(p) for p in args.svt], clip_id)
    base = json.loads((stored / "baselines.json").read_text())
    for name in ("frame", "filled"):
        points = [{"crf": p["crf"], "kbps": p["kbps"], "stream": str(stored / "streams" / f"{p['input']}-crf{p['crf']}.ivf"),
                   "stream_sha256": p["stream_sha256"]} for p in base["points"] if p["input"] == name]
        methods.append({"arm": f"svtav1-{name}", "codec": "svtav1", "frames": base["frames"], "points": points})
    dcvc_dir = Path(args.dcvc) / "publish" / "clips" / g1.safe(clip_id)
    ref = json.loads((dcvc_dir / "dcvc.json").read_text())
    methods.append({"arm": "dcvc", "codec": "dcvc", "frames": ref["frames"], "video_ckpt": args.video_ckpt,
                    "points": [{"qp": p["qp"], "kbps": p["kbps"], "bytes": p["bytes"],
                                "stream": str(dcvc_dir / "streams" / f"filled-qp{p['qp']}.psdc")} for p in ref["points"]]})
    for root in args.arms:
        ft = Path(root) / "publish" / "finetune"
        row = json.loads((ft / "result.json").read_text())
        if row["id"] != clip_id:
            raise SystemExit(f"{root}: an arm of {row['id']}, not {clip_id}")
        methods.append({"arm": row["arm"], "codec": "dcvc", "frames": row["frames"], "fit_clip": row["fit_clip"],
                        "lpips_weight": row["lpips_weight"], "video_ckpt": str(ft / "video_ft.pth.tar"),
                        "model_sha256": row["model"]["checkpoint_sha256"],
                        "points": [{"qp": p["qp"], "kbps": p["kbps"], "bytes": p["bytes"],
                                    "stream": str(ft / "streams" / f"filled-qp{p['qp']}.psdc")} for p in row["points"]]})
    return methods


def decode_point(method: dict[str, Any], point: dict[str, Any], args: argparse.Namespace, n: int,
                 work: Path, source: Path, duration: float) -> tuple[np.ndarray, dict[str, Any]]:
    """SVT-AV1: the stored stream decoded (dav1d). DCVC-UF: the stored stream decoded alone in a fresh process
    (recorded, not required: it segfaulted in the first smoke), and the excerpt re-encoded with the stored
    checkpoint as G5b did, whose decode is scored; the new stream's bytes are compared with the stored ones."""
    from src.codecs import svtav1

    stream = Path(point["stream"])
    check: dict[str, Any] = {"stream_sha256": file_sha256(stream), "stream_bytes": stream.stat().st_size}
    if method["codec"] == "svtav1":
        if check["stream_sha256"] != point["stream_sha256"]:
            raise RuntimeError(f"{stream}: bytes differ from the recorded stream")
        out = work / "decoded.yuv"
        out.unlink(missing_ok=True)
        svtav1.run(svtav1.decode_command("dav1d", stream, out, threads=8), 600)
        rgb = read_decoded(out, method["frames"], n)
        out.unlink()
        return rgb, {**check, "matches_record": True}
    if method.get("model_sha256") and file_sha256(Path(method["video_ckpt"])) != method["model_sha256"]:
        raise RuntimeError(f"{method['video_ckpt']}: not the checkpoint its result recorded")
    alone = decode_dcvc(args.image_ckpt, method["video_ckpt"], stream, work / "dcvc")
    shutil.rmtree(work / "dcvc")
    coder = SimpleNamespace(image_ckpt=args.image_ckpt, video_ckpt=method["video_ckpt"])
    rec = g5.code_dcvc(coder, source, n, point["qp"], work / "recode")  # type: ignore[arg-type]
    rgb = read_decoded(Path(rec.pop("decoded")), n, n)
    recoded = (work / "recode" / "stream.psdc").read_bytes()
    shutil.rmtree(work / "recode")
    return rgb, {**check, "matches_record": check["stream_bytes"] == point["bytes"],
                 "stored_decode_alone": {k: v for k, v in alone.items() if k != "decoded"},
                 "recoded_same_bytes": recoded == stream.read_bytes(), "recoded_bytes": len(recoded),
                 "recoded_kbps": 8 * len(recoded) / duration / 1000, "deterministic": rec["deterministic"],
                 "decoder_matches_encoder_intra": rec["decoder_matches_encoder_intra"]}


def restore(checkpoints: Path | None) -> dict[str, dict[str, Any]]:
    if checkpoints is None or not (checkpoints / "done").is_dir():
        return {}
    return {p.stem: json.loads(p.read_text()) for p in (checkpoints / "done").glob("*.json")}


def save(checkpoints: Path | None, key: str, row: dict[str, Any]) -> None:
    if checkpoints is not None:
        (checkpoints / "done").mkdir(parents=True, exist_ok=True)
        write_json(checkpoints / "done" / f"{key}.json", row)


def gpu_record() -> dict[str, Any]:
    import torch

    return {"gpu": torch.cuda.get_device_name(0), "gpu_capability": list(torch.cuda.get_device_capability(0))}


def command_checks(args: argparse.Namespace) -> int:
    spec = json.loads(Path(args.clips).read_text())
    by_id = {c["id"]: c for c in spec["clips"]}
    clip_id = g5.only_clip(spec, args.select)
    clip = g5.load_clip(g5.clip_dirs(Path(args.prepared))[clip_id], by_id[clip_id], Path(args.visor_fill),
                        args.limit_frames)
    nets = load_nets(args)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "checks"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = restore(checkpoints)
    began = time.time()
    flows = source_flows(clip)
    flow_seconds = time.time() - began
    source = work / "filled.yuv"
    g5.rgb_to_yuv420(g5.inpaint(clip)).tofile(source)
    rows = []
    methods = stored_methods(args, clip_id)
    methods = methods[:args.max_methods] if args.max_methods > 0 else methods
    for done, method in enumerate(methods, 1):
        if method["arm"] in restored:
            rows.append({**restored[method["arm"]], "restored_from_checkpoint": True})
            progress(done)
            continue
        points = []
        for p in method["points"]:
            rgb, check = decode_point(method, p, args, clip.n, work, source, clip.duration)
            points.append({**{k: v for k, v in p.items() if k != "stream"}, "check": check,
                           "score": full_score(clip, rgb, nets, flows)})
        row = {**{k: v for k, v in method.items() if k != "points"}, "points": points}
        save(checkpoints, method["arm"], row)
        rows.append(row)
        progress(done)
    write_json(stage_dir() / "g5c.json", {
        "kind": "checks", "clip": clip.record, "limit_frames": args.limit_frames, "decision_rule": DECISION,
        "metrics": nets.record, "flow_seconds": round(flow_seconds, 1), "restored": sorted(restored),
        "seconds": round(time.time() - began, 1), **gpu_record(), "methods": rows})
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ----------------------------------------------------------------- model size


def parse_variant(name: str) -> tuple[str, int, float]:
    """``whole-B``, ``delta-B`` or ``delta-B-sF``: what is sent, its bits, the share of changes kept."""
    parts = name.split("-")
    kind, bits = parts[0], int(parts[1])
    keep = float(parts[2][1:]) if len(parts) > 2 else 1.0
    if kind not in ("whole", "delta") or bits not in (16, 8, 4, 2) or not 0 < keep <= 1 or (kind == "whole" and keep < 1):
        raise ValueError(f"unknown variant {name!r}")
    return kind, bits, keep


def quantize_channels(t: Any, bits: int) -> tuple[Any, Any, Any]:
    """Symmetric uniform quantization per output channel (dim 0): int8 levels, fp16 scales, the dequantized
    tensor (with the fp16 scales the decoder reads)."""
    import torch

    levels = 2 ** (bits - 1) - 1
    flat = t.reshape(t.shape[0], -1).float()
    scale = (flat.abs().amax(1) / levels).half().float()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(flat / safe[:, None]), -levels, levels)
    q[scale == 0] = 0
    return q.to(torch.int8), scale.half(), (q * scale[:, None]).reshape(t.shape)


def keep_largest(d: Any, keep: float) -> Any:
    """``d`` with all but the largest ``keep`` share of its values (by magnitude) set to zero."""
    import torch

    if keep >= 1:
        return d
    k = max(1, math.ceil(keep * d.numel()))
    threshold = torch.topk(d.abs().flatten(), k).values[-1]
    return torch.where(d.abs() >= threshold, d, torch.zeros_like(d))


def compress_state(ft: dict[str, Any], public: dict[str, Any], variant: str,
                   workers: int = 16) -> tuple[dict[str, Any], dict[str, Any]]:
    """The state the client rebuilds from a variant, and the variant's size. ``whole``: every tensor sent.
    ``delta``: the change from ``public``, which the client has. Tensors of `QUANT_MIN_VALUES` or more values
    are quantized per output channel to the variant's bits; smaller ones, and every tensor at 16 bits, go as
    fp16. Integer tensors go raw, and not at all in a change that leaves them equal. Each tensor's payload is
    LZMA-coded on its own; the zeroth-order entropy of the levels is the bound beside it."""
    import torch

    from experiments.background.g5_cond import entropy_bits

    kind, bits, keep = parse_variant(variant)
    if set(ft) != set(public):
        raise ValueError("fine-tuned and public states differ in keys")
    state, payloads = {}, []
    entropy_bytes, quantized, kept, total = 0.0, 0, 0, 0
    for name, t in ft.items():
        base = public[name]
        if not torch.is_floating_point(t):
            state[name] = t
            if kind == "whole" or not torch.equal(t, base):
                payloads.append(t.numpy().tobytes())
                entropy_bytes += t.numel() * t.element_size()
            continue
        v = t.float() - (base.float() if kind == "delta" else 0.0)
        total += v.numel()
        if bits == 16 or v.numel() < QUANT_MIN_VALUES:
            sent = v.half()
            payloads.append(sent.numpy().tobytes())
            entropy_bytes += 2 * sent.numel()
            rebuilt = sent.float()
            kept += int((sent != 0).sum())
        else:
            v = keep_largest(v, keep) if kind == "delta" else v
            q, scale, rebuilt = quantize_channels(v, bits)
            payloads += [q.numpy().tobytes(), scale.numpy().tobytes()]
            entropy_bytes += entropy_bits(q.numpy().ravel()) / 8 + 2 * scale.numel()
            quantized += v.numel()
            kept += int((q != 0).sum())
        state[name] = (rebuilt + (base.float() if kind == "delta" else 0.0)).to(t.dtype)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        coded = sum(pool.map(lambda b: len(lzma.compress(b, preset=6)), payloads))
    return state, {"variant": variant, "kind": kind, "bits": bits, "keep": keep, "bytes": coded,
                   "entropy_bytes": round(entropy_bytes), "raw_fp16_bytes": 2 * total,
                   "quantized_share": quantized / total, "nonzero_share": kept / total}


def load_plain(path: Path) -> dict[str, Any]:
    import torch
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = state.get("net", state)
    consume_prefix_in_state_dict_if_present(state, prefix="module.")
    return dict(state)


def command_compress(args: argparse.Namespace) -> int:
    import torch

    spec = json.loads(Path(args.clips).read_text())
    by_id = {c["id"]: c for c in spec["clips"]}
    clip_id = g5.only_clip(spec, args.select)
    clip = g5.load_clip(g5.clip_dirs(Path(args.prepared))[clip_id], by_id[clip_id], Path(args.visor_fill),
                        args.limit_frames)
    ft_dir = Path(args.arm) / "publish" / "finetune"
    arm = json.loads((ft_dir / "result.json").read_text())
    if arm["id"] != clip_id:
        raise SystemExit(f"{args.arm}: an arm of {arm['id']}, not {clip_id}")
    if file_sha256(ft_dir / "video_ft.pth.tar") != arm["model"]["checkpoint_sha256"]:
        raise RuntimeError("the published checkpoint is not the one its result recorded")
    nets = load_nets(args)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    work = scratch / "compress"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = restore(checkpoints)
    began = time.time()
    flows = source_flows(clip)
    source = work / "filled.yuv"
    g5.rgb_to_yuv420(g5.inpaint(clip)).tofile(source)
    ft, public = load_plain(ft_dir / "video_ft.pth.tar"), load_plain(Path(args.video_ckpt))
    qps = [int(q) for q in args.qps.split(",")]
    rows = []
    for done, variant in enumerate(args.variants.split(","), 1):
        if variant in restored:
            rows.append({**restored[variant], "restored_from_checkpoint": True})
            progress(done)
            continue
        t0 = time.time()
        state, size = compress_state(ft, public, variant)
        path = work / f"{variant}.pth.tar"
        torch.save({"state_dict": state}, path)
        size["seconds"] = round(time.time() - t0, 1)
        del state
        coder = SimpleNamespace(image_ckpt=args.image_ckpt, video_ckpt=str(path))
        points = []
        for qp in qps:
            rec = g5.code_dcvc(coder, source, clip.n, qp, work / f"qp{qp}")  # type: ignore[arg-type]
            rgb = read_decoded(Path(rec.pop("decoded")), clip.n, clip.n)
            stream = work / f"qp{qp}" / "stream.psdc"
            published = ft_dir / "streams" / f"filled-qp{qp}.psdc"
            same = published.is_file() and stream.read_bytes() == published.read_bytes()
            published_bytes = published.stat().st_size if published.is_file() else None
            points.append({"qp": qp, "kbps": rec["bytes"] * 8 / clip.duration / 1000, **rec,
                           "stream_sha256": file_sha256(stream), "same_as_published": same,
                           "published_bytes": published_bytes,
                           "score": full_score(clip, rgb, nets, flows)})
            shutil.rmtree(work / f"qp{qp}")
        row = {"variant": variant, "size": size, "model_sha256": file_sha256(path), "points": points}
        path.unlink()
        save(checkpoints, variant, row)
        rows.append(row)
        progress(done)
    write_json(stage_dir() / "g5c.json", {
        "kind": "compress", "clip": clip.record, "arm": arm["arm"], "arm_model_sha256": arm["model"]["checkpoint_sha256"],
        "public_sha256": file_sha256(Path(args.video_ckpt)), "qps": qps, "limit_frames": args.limit_frames,
        "decision_rule": DECISION, "metrics": nets.record, "restored": sorted(restored),
        "seconds": round(time.time() - began, 1), **gpu_record(), "variants": rows})
    shutil.rmtree(work, ignore_errors=True)
    return 0


# ----------------------------------------------------------------- the fill (Telea against ProPainter)

def propainter_fill(clip: g5.Clip, tree: Path, weights: dict[str, Path], work: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """ProPainter's fill of the clip's holes (not V of the training masks), run in its tree (the extracted
    archive's ``ProPainter`` directory); ``weights`` maps propainter, raft and flow_completion to checkpoints."""
    work.mkdir(parents=True, exist_ok=True)
    np.save(work / "frames.npy", clip.frames)
    np.save(work / "holes.npy", ~clip.keep)
    plan: dict[str, Any] = {"frames": str(work / "frames.npy"), "holes": str(work / "holes.npy"),
                            "out": str(work / "filled.npy")}
    for key, path in weights.items():
        plan[key] = str(path)
        plan[f"{key}_sha256"] = file_sha256(path)
    tree = tree / "ProPainter" if (tree / "ProPainter").is_dir() else tree
    write_json(work / "plan.json", plan)
    script = Path(__file__).resolve().parent / "propainter_fill.py"
    done = subprocess.run([sys.executable, str(script), "--plan", str(work / "plan.json"), "--report",
                           str(work / "report.json")], cwd=tree, env={**os.environ, "PYTHONPATH": str(tree),
                                                                    "PYTHONNOUSERSITE": "1"},
                          capture_output=True, text=True, timeout=2400)
    if done.returncode:
        raise RuntimeError(f"ProPainter failed ({done.returncode}):\n{done.stderr[-4000:]}")
    filled = np.load(work / "filled.npy")
    if filled.shape != clip.frames.shape:
        raise RuntimeError(f"ProPainter returned {filled.shape}, not {clip.frames.shape}")
    filled[clip.keep] = clip.frames[clip.keep]  # its dilated ring repaints V's edge; every method codes V as is
    return filled, {**json.loads((work / "report.json").read_text()), **{k: v for k, v in plan.items() if "sha256" in k}}


def fill_flicker(filled: np.ndarray, holes: np.ndarray, device: str) -> float | None:
    """Mean absolute luma change of the fill inside the holes of both frames, after warping frame t−1 onto t
    by DIS flow on the filled video itself (the fill's own motion, not the hidden actor's)."""
    import torch

    flows = g5_cond.oracle_flows(filled, [None] + list(range(len(filled) - 1)))
    luma = torch.tensor(LUMA, device=device).view(1, 3, 1, 1)
    values = []
    for t in range(1, len(filled)):
        a = torch.from_numpy(filled[[t - 1]]).to(device).permute(0, 3, 1, 2).float()
        m = torch.from_numpy(holes[[t - 1]]).to(device)[:, None].float()
        warped, valid = g5_cond.warp(torch.cat([a, m], 1), torch.from_numpy(flows[[t]].astype(np.float32)).to(device))
        region = torch.from_numpy(holes[[t]]).to(device)[:, None] & (warped[:, 3:4] > 0.999) & (valid > 0)
        if region.any():
            b = torch.from_numpy(filled[[t]]).to(device).permute(0, 3, 1, 2).float()
            values.append(float((((b - warped[:, :3]) * luma).sum(1, keepdim=True).abs() * region).sum() / region.sum()))
    return float(np.mean(values)) if values else None


def command_fill(args: argparse.Namespace) -> int:
    spec = json.loads(Path(args.clips).read_text())
    by_id = {c["id"]: c for c in spec["clips"]}
    dirs = g5.clip_dirs(Path(args.prepared))
    nets = load_nets(args)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = restore(checkpoints)
    began = time.time()
    rows = []
    for done, clip_id in enumerate((c["id"] for c in g5.select_clips(spec, args.select)), 1):
        key = g1.safe(clip_id)
        if key in restored:
            rows.append({**restored[key], "restored_from_checkpoint": True})
            progress(done)
            continue
        clip = g5.load_clip(dirs[clip_id], by_id[clip_id], Path(args.visor_fill), args.limit_frames)
        work = scratch / "fill" / key
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        flows = source_flows(clip)
        t0 = time.time()
        fills = {"telea": g5.inpaint(clip)}
        telea_seconds = time.time() - t0
        weights = {"propainter": Path(args.propainter_ckpt), "raft": Path(args.raft_ckpt),
                   "flow_completion": Path(args.flow_completion_ckpt)}
        fills["propainter"], propainter = propainter_fill(clip, Path(args.propainter_tree), weights, work / "propainter")
        methods = []
        for name, filled in fills.items():
            source = work / f"{name}.yuv"
            g5.rgb_to_yuv420(filled).tofile(source)
            points = []
            for crf in g5.CRFS:
                rec = g5.code_svtav1({"work": str(work / f"{name}-crf{crf}"), "source": str(source), "fps": str(clip.fps),
                                      "frames": clip.n, "crf": crf, "threads": 8, "cpus": None})
                rgb = read_decoded(Path(rec["decoded"]), clip.n, clip.n)
                points.append({"crf": crf, "kbps": 8 * rec["payload_bytes"] / clip.duration / 1000,
                               "stream_sha256": rec["stream_sha256"], "score": full_score(clip, rgb, nets, flows)})
                shutil.rmtree(work / f"{name}-crf{crf}")
            methods.append({"arm": f"svtav1-{name}", "codec": "svtav1", "fill": name, "points": points,
                            "fill_flicker": fill_flicker(filled, ~clip.keep, "cuda"),
                            "fill_seconds": round(telea_seconds, 1) if name == "telea" else propainter["seconds"]})
        row = {"clip": clip.record, "propainter": propainter, "methods": methods}
        save(checkpoints, key, row)
        rows.append(row)
        shutil.rmtree(work, ignore_errors=True)
        progress(done)
    write_json(stage_dir() / "g5c.json", {
        "kind": "fill", "select": args.select, "limit_frames": args.limit_frames, "decision_rule": DECISION,
        "metrics": nets.record, "restored": sorted(restored), "seconds": round(time.time() - began, 1),
        **gpu_record(), "clips": rows})
    return 0


# ----------------------------------------------------------------- decision


def quality(point: dict[str, Any], tier: str, metric: str) -> float | None:
    """Higher is better: −100 × LPIPS_V or DISTS_V, PSNR_V as is, −flicker."""
    score = point["score"]
    value = score["flicker"]["flicker_v"] if metric == "flicker_v" else score.get(tier, {}).get(metric)
    if value is None:
        return None
    return float(value) if metric == "psnr_v" else (-float(value) if metric == "flicker_v" else -100.0 * float(value))


def rate(point: dict[str, Any]) -> float:
    """The rate of the stream that was decoded and scored: a re-encoded DCVC-UF stream's own (``checks``)."""
    return float(point.get("check", {}).get("recoded_kbps", point["kbps"]))


def curve(points: list[dict[str, Any]], tier: str, metric: str) -> list[tuple[float, float]]:
    return [(rate(p), q) for p in points if (q := quality(p, tier, metric)) is not None]


METRICS = ("lpips_v", "dists_v", "psnr_v", "flicker_v")


def worse_everywhere(test: list[tuple[float, float]], anchor: list[tuple[float, float]]) -> bool:
    """No overlap and every test point below the anchor's best quality: worse, not unclear."""
    return bool(test) and bool(anchor) and max(q for _, q in test) < max(q for _, q in anchor)


def checks_decision(by_clip: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comp = DECISION["component"]
    transfer = by_clip.get(DECISION["dists_transfer_clip"], {}).get(comp, {}).get("dists_v")
    found = []
    for clip_id in DECISION["flicker_clips"]:
        v = by_clip.get(clip_id, {}).get(comp, {}).get("flicker_v")
        if v is None:
            continue
        found.append((v["bd_rate"] is not None and v["bd_rate"] > 0) or (v["bd_rate"] is None and v.get("worse")))
    return {"dists_transfers": None if transfer is None or transfer["bd_rate"] is None else transfer["bd_rate"] < 0,
            "flicker_found": None if len(found) < len(DECISION["flicker_clips"]) else any(found)}


def command_report(args: argparse.Namespace) -> int:
    tier = DECISION["gate_tier"]
    checks: dict[str, dict[str, Any]] = {}
    sizes: dict[str, dict[str, Any]] = {}
    fills: list[dict[str, Any]] = []
    runs = []
    for path in args.result:
        doc = json.loads(Path(path).read_text())
        runs.append({"path": path, "sha256": file_sha256(Path(path)), "kind": doc["kind"]})
        if doc["kind"] == "checks":
            checks[doc["clip"]["id"]] = doc
        elif doc["kind"] == "compress":
            sizes[doc["clip"]["id"]] = doc
        elif doc["kind"] == "fill":
            fills.append(doc)
    report: dict[str, Any] = {"runs": runs, "decision_rule": DECISION, "checks": {}, "size": {}}
    for clip_id, doc in sorted(checks.items()):
        svt = [p for m in doc["methods"] if m["codec"] == "svtav1" for p in m["points"]]
        arms: dict[str, Any] = {}
        for m in doc["methods"]:
            if m["codec"] == "svtav1":
                continue
            entry: dict[str, Any] = {}
            for metric in METRICS:
                test, anchor = curve(m["points"], tier, metric), curve(svt, tier, metric)
                v = g5.clip_verdict(test, anchor)
                entry[metric] = {**v, "worse": v["bd_rate"] is None and worse_everywhere(test, anchor)}
            entry["flicker"] = [{"kbps": rate(p), **p["score"]["flicker"]} for p in m["points"]]
            arms[m["arm"]] = entry
        arms["svtav1"] = {"flicker": [{"kbps": p["kbps"], **p["score"]["flicker"]} for p in svt]}
        report["checks"][clip_id] = arms
    report["checks_decision"] = checks_decision(report["checks"])
    for clip_id, doc in sorted(sizes.items()):
        rows = {r["variant"]: r for r in doc["variants"]}
        if "whole-16" not in rows:
            continue
        reference = curve(rows["whole-16"]["points"], tier, "lpips_v")
        svt = [p for m in checks.get(clip_id, {}).get("methods", []) if m["codec"] == "svtav1" for p in m["points"]]
        out = {}
        for name, r in rows.items():
            test = curve(r["points"], tier, "lpips_v")
            bd = g5.clip_verdict(test, reference)["bd_rate"]
            out[name] = {**r["size"], "vs_16bit": bd,
                         "keeps_quality": bd is not None and bd <= DECISION["keeps_quality_bd"],
                         "vs_svtav1": g5.clip_verdict(test, curve(svt, tier, "lpips_v"))["bd_rate"] if svt else None}
        keepers = [n for n, e in out.items() if e["keeps_quality"]]
        report["size"][clip_id] = {"variants": out,
                                   "smallest_keeping_quality": min(keepers, key=lambda n: out[n]["bytes"]) if keepers else None}
    fill: dict[str, Any] = {}
    for doc in fills:
        for row in doc["clips"]:
            by = {m["fill"]: m for m in row["methods"]}
            entry = {name: {"fill_flicker": m["fill_flicker"], "fill_seconds": m["fill_seconds"]}
                     for name, m in by.items()}
            for metric in METRICS:
                entry[f"propainter_vs_telea_{metric}"] = g5.clip_verdict(
                    curve(by["propainter"]["points"], tier, metric), curve(by["telea"]["points"], tier, metric))["bd_rate"]
            fill[row["clip"]["id"]] = entry
    report["fill"] = fill
    values = [e["propainter_vs_telea_lpips_v"] for e in fill.values()]
    report["fill_decision"] = {"propainter_wins": None if not values or any(v is None for v in values)
                               else all(v < DECISION["fill_wins_bd"] for v in values)}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "g5c-report.json", report)
    print(json.dumps({"checks_decision": report["checks_decision"], "fill_decision": report["fill_decision"],
                      "smallest": {c: s["smallest_keeping_quality"] for c, s in report["size"].items()}}, indent=1))
    return 0


# ----------------------------------------------------------------- validation


def validate_stage(stage: Path) -> dict[str, bool]:
    result = json.loads((stage / "g5c.json").read_text())
    full = not result["limit_frames"]
    checks: dict[str, bool] = {"intended_gpu": tuple(result.get("gpu_capability") or (0, 0)) >= (8, 6)}

    def scored(p: dict[str, Any]) -> bool:
        s = p["score"]
        d = s["dataset"].get("dists_v")
        return (g5.finite(s["dataset"]["lpips_v"]) and d is not None and 0 <= d <= 1
                and g5.finite(s["flicker"]["flicker_v"]) and s["flicker"]["pairs"] > 0)

    if result["kind"] == "checks":
        methods = result["methods"]
        checks["every_method"] = {m["arm"] for m in methods} >= {"svtav1-frame", "svtav1-filled", "dcvc"} and len(methods) >= 4
        checks["every_point_scored"] = all(m["points"] and all(scored(p) for p in m["points"]) for m in methods)
        checks["streams_match_records"] = all(p["check"]["matches_record"] for m in methods for p in m["points"])
        checks["deterministic_decode"] = all(p["check"].get("deterministic", True) for m in methods for p in m["points"])
        if full:
            checks["recoded_matches_stored"] = all(  # within 0.1%; bit-exact equality is recorded
                abs(p["check"].get("recoded_bytes", p["check"]["stream_bytes"]) - p["check"]["stream_bytes"])
                <= 1e-3 * p["check"]["stream_bytes"] for m in methods for p in m["points"])
        checks["dists_falls_with_rate"] = all(
            all(a["score"]["dataset"]["dists_v"] >= b["score"]["dataset"]["dists_v"] - 0.01 for a, b in zip(pts, pts[1:]))
            for m in methods if m["codec"] == "svtav1" for pts in [sorted(m["points"], key=lambda p: p["kbps"])])
        checks["source_moves"] = all(m["points"][0]["score"]["flicker"]["residual_source"] > 0 for m in methods)
    elif result["kind"] == "compress":
        rows = {r["variant"]: r for r in result["variants"]}
        checks["every_variant"] = bool(rows) and all(len(r["points"]) == len(result["qps"]) for r in rows.values())
        checks["every_point_scored"] = all(scored(p) for r in rows.values() for p in r["points"])
        checks["deterministic_decode"] = all(p["deterministic"] and p["decoder_matches_encoder_intra"]
                                             for r in rows.values() for p in r["points"])
        checks["fewer_bits_smaller"] = all(rows[f"{k}-{a}"]["size"]["bytes"] > rows[f"{k}-{b}"]["size"]["bytes"]
                                           for k in ("whole", "delta") for a, b in ((16, 8), (8, 4), (4, 2))
                                           if f"{k}-{a}" in rows and f"{k}-{b}" in rows)
        if full and "whole-16" in rows:
            checks["16bit_reproduces_g5b"] = all(  # bit-exact re-encoding is recorded, not required (G5c entry)
                p["published_bytes"] and abs(p["bytes"] - p["published_bytes"]) <= 1e-3 * p["published_bytes"]
                for p in rows["whole-16"]["points"])
    elif result["kind"] == "fill":
        rows = result["clips"]
        checks["clips_present"] = bool(rows)
        checks["both_fills"] = all({m["fill"] for m in r["methods"]} == {"telea", "propainter"} for r in rows)
        checks["every_point_scored"] = all(len(m["points"]) == len(g5.CRFS) and all(scored(p) for p in m["points"])
                                           for r in rows for m in r["methods"])
        checks["fill_flicker_finite"] = all(g5.finite(m["fill_flicker"]) for r in rows for m in r["methods"])
        checks["propainter_on_gpu"] = all(r["propainter"]["gpu"] == result["gpu"] for r in rows)
    return checks


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(args.stage or os.environ.get("PS_STAGE_DIR") or ".")
    checks = validate_stage(stage)
    payload = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), payload)
    print(json.dumps(payload, indent=1))
    return 0 if payload["passed"] else 1


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--prepared", required=True, help="G1d's extracted prepared archive")
        p.add_argument("--clips", required=True, help="G1's clips.json")
        p.add_argument("--visor-fill", required=True, help="B1b's fill (the stretches' dense masks)")
        p.add_argument("--select", required=True)
        p.add_argument("--limit-frames", type=int, default=0)
        p.add_argument("--lpips-backbone", required=True)
        p.add_argument("--dists-backbone", required=True)
        p.add_argument("--image-ckpt", default=None)
        p.add_argument("--video-ckpt", default=None, help="the public HT-L checkpoint")

    ch = sub.add_parser("checks")
    common(ch)
    ch.add_argument("--svt", nargs="+", required=True, help="extracted G5 baselines archives")
    ch.add_argument("--dcvc", required=True, help="G5b's extracted dcvc archive")
    ch.add_argument("--arms", nargs="+", default=[], help="G5b's extracted finetune archives of this clip")
    ch.add_argument("--max-methods", type=int, default=0, help="smoke: the first N methods (SVT-AV1, DCVC-UF, arms)")
    ch.set_defaults(func=command_checks)
    co = sub.add_parser("compress")
    common(co)
    co.add_argument("--arm", required=True, help="G5b's extracted finetune archive to compress")
    co.add_argument("--variants", default=",".join(VARIANTS))
    co.add_argument("--qps", default=",".join(map(str, g5.DCVC_QPS)))
    co.set_defaults(func=command_compress)
    fi = sub.add_parser("fill")
    common(fi)
    fi.add_argument("--propainter-tree", required=True, help="ProPainter's extracted repository")
    fi.add_argument("--propainter-ckpt", required=True)
    fi.add_argument("--raft-ckpt", required=True)
    fi.add_argument("--flow-completion-ckpt", required=True)
    fi.set_defaults(func=command_fill)
    va = sub.add_parser("validate")
    va.add_argument("--stage", default=None)
    va.set_defaults(func=command_validate)
    rp = sub.add_parser("report")
    rp.add_argument("--result", action="append", required=True)
    rp.add_argument("--out", required=True)
    rp.set_defaults(func=command_report)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
