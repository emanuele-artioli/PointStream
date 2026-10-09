"""PLAN step G1d (c): Depth Anything 3 depth of every keyframe G1d's evaluation references.

    python -m experiments.background.da3 run --prepared DIR --select all|pilot|rest|ID,ID --limit-frames N \\
        --weights FILE --config FILE --overlay DIR
    python -m experiments.background.da3 validate

Per clip, `g1d.register_clip` (CPU pool) repeats the deterministic registration
of ``g1d run`` to find the keyframes that ``direct`` frames reference; each of
them is then run through DA3 alone (monocular) on the GPU, at DA3's default
``upper_bound_resize`` to 504 px (960x540 becomes 504x280: a resize, not a crop,
so the depth maps back onto the analysis grid). Writes
``publish/da3/<id>/depth.npz`` (per keyframe index: float16 z-depth at DA3's
resolution, and DA3's intrinsics) and ``da3.json`` (runtime, kernels, weights).

DA3 (`--overlay`: the pure-Python overlay with ``depth_anything_3``, ``addict``
and ``omegaconf``, docs/resources.md) imports ``pycolmap`` and ``evo`` at module
level for exporters and pose alignment that monocular inference never calls;
`install_stubs` provides placeholders that raise if they are ever used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background import g1, g1d
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

PROCESS_RES = 504
PROCESS_METHOD = "upper_bound_resize"
DA3_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"


def install_stubs() -> None:
    """Placeholders for modules DA3 imports but monocular inference never uses."""

    class Unavailable:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("not installed in the PointStream environment (DA3 stub)")

    def module(name: str, **attrs: Any) -> None:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__dict__.update(attrs, __pointstream_stub__=True)
            sys.modules[name] = stub

    module("pycolmap")
    module("evo")
    module("evo.core")
    module("evo.core.trajectory", PosePath3D=Unavailable)


def load_model(weights: Path, config: Path, overlay: Path, scratch: Path) -> tuple[Any, dict[str, Any]]:
    """DA3 from staged files: they are linked into one directory, as ``from_pretrained`` expects."""
    import torch

    model_dir = scratch / "da3-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, source in (("model.safetensors", weights), ("config.json", config)):
        link = model_dir / name
        if not link.exists():
            link.symlink_to(source.resolve())

    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(overlay))
    install_stubs()
    from depth_anything_3.api import DepthAnything3

    began = time.time()
    model = DepthAnything3.from_pretrained(str(model_dir)).to(device="cuda").eval()
    info = {"load_seconds": round(time.time() - began, 2), "model_dir": str(model_dir),
            "weights_sha256": file_sha256(model_dir / "model.safetensors"),
            "config_sha256": file_sha256(model_dir / "config.json"),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "overlay": str(overlay), "da3_commit": DA3_COMMIT,
            "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
            "bf16": bool(torch.cuda.is_bf16_supported()),
            "sdpa": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                     "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                     "math": torch.backends.cuda.math_sdp_enabled()}}
    return model, info


def kernel_check(model: Any, rgb: np.ndarray) -> dict[str, Any]:
    """Profile one inference: which attention kernels ran, on which device."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        model.inference([rgb], process_res=PROCESS_RES, process_res_method=PROCESS_METHOD)
        torch.cuda.synchronize()
    names = [e.key for e in prof.key_averages()]
    attention = sorted({n for n in names if any(s in n.lower() for s in ("flash", "fmha", "efficient_attention", "sdpa", "attention"))})
    cuda_kernels = sum(1 for e in prof.key_averages() if getattr(e, "device_type", None) is not None and "CUDA" in str(e.device_type))
    return {"attention_ops": attention[:40], "flash_kernel": any("flash" in n.lower() for n in names),
            "math_fallback": any("_scaled_dot_product_attention_math" in n for n in names),
            "cuda_kernel_kinds": cuda_kernels}


def keyframe_rgb(clip_dir: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    out = {}
    for i, rgb in enumerate(g1d.decode_ffv1(clip_dir / "frames.mkv", 4)):
        if i in wanted:
            out[i] = rgb
        if len(out) == len(wanted):
            break
    return out


def command_run(args: argparse.Namespace) -> int:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import pickle

    import cv2
    import torch

    prepared = Path(args.prepared)
    metas = {json.loads((d / "meta.json").read_text())["id"]: d for d in (prepared / "publish" / "clips").iterdir()
             if (d / "meta.json").is_file()}
    spec = {"clips": [{"id": i, "group": i.split("/")[0], "video": i.split("/")[1]} for i in metas]}
    clips = [c["id"] for c in g1d.select_clips(spec, args.select)]
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = g1.restore_clips(checkpoints, publish / "da3_ckpt") if checkpoints else {}
    allowance = max(2, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 2))
    pool = ProcessPoolExecutor(max_workers=max(1, allowance - 2), mp_context=multiprocessing.get_context("spawn"))
    todo = [c for c in clips if c not in restored]
    futures = {c: pool.submit(g1d.register_clip, {"clip_dir": str(metas[c]), "work": str(scratch / "work" / g1.safe(c)),
                                                  "limit_frames": args.limit_frames}) for c in todo}
    model, info = load_model(Path(args.weights), Path(args.config), Path(args.overlay), scratch)
    torch.cuda.reset_peak_memory_stats()
    kernels: dict[str, Any] | None = None
    results: dict[str, dict[str, Any]] = {c: restored[c]["result"] for c in restored}
    done = 0
    for clip_id in todo:
        futures[clip_id].result()
        work = scratch / "work" / g1.safe(clip_id)
        with (work / "state.pkl").open("rb") as handle:
            state = pickle.load(handle)
        keys = sorted({p.reference for p in state["poses"] if p.status == "direct" and p.reference is not None})
        rgb = keyframe_rgb(metas[clip_id], set(keys))
        if kernels is None and keys:
            kernels = kernel_check(model, rgb[keys[0]])
        began = time.time()
        depth: dict[str, np.ndarray] = {}
        hfov = []
        jpeg: dict[str, int] = {}
        for k in keys:
            # The keyframe's colour, which every keyframe representation sends once (JPEG q90).
            jpeg[str(k)] = len(cv2.imencode(".jpg", rgb[k][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])[1])
            pred = model.inference([rgb[k]], process_res=PROCESS_RES, process_res_method=PROCESS_METHOD)
            d = np.asarray(pred.depth[0], np.float32)
            K = np.asarray(pred.intrinsics[0], np.float64)
            depth[f"depth_{k}"] = d.astype(np.float16)
            depth[f"intrinsics_{k}"] = K
            hfov.append(float(np.degrees(2 * np.arctan(d.shape[1] / 2 / K[0, 0]))))
        seconds = time.time() - began
        target = publish / "da3" / g1.safe(clip_id)
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / "depth.npz", **depth)
        result = {"id": clip_id, "keyframes": len(keys), "seconds": round(seconds, 2),
                  "seconds_per_keyframe": round(seconds / max(1, len(keys)), 4),
                  "depth_shape": list(next(iter(depth.values())).shape) if depth else None,
                  "hfov_deg_median": round(float(np.median(hfov)), 2) if hfov else None,
                  "depth_finite": bool(all(np.isfinite(v).all() for n, v in depth.items() if n.startswith("depth_"))),
                  "depth_npz_sha256": file_sha256(target / "depth.npz"), "jpeg_q90_bytes": jpeg}
        write_json(target / "result.json", result)
        results[clip_id] = result
        if checkpoints is not None:
            g1.save_clip(checkpoints, target, clip_id, result, None)
        done += 1
        progress(done)
    pool.shutdown()
    info["peak_gpu_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    write_json(stage_dir() / "da3.json", {"prepared": str(prepared), "select": args.select, "limit_frames": args.limit_frames,
                                         "process_res": PROCESS_RES, "process_res_method": PROCESS_METHOD, "runtime": info,
                                         "kernels": kernels, "restored_clips": sorted(restored),
                                         "results": [results[c] for c in clips]})
    return 0


def validate_stage(stage: Path) -> dict[str, bool]:
    import tarfile

    result = json.loads((stage / "da3.json").read_text())
    rows = result["results"]
    with tarfile.open(stage / "published.tar") as tar:
        names = set(tar.getnames())
    runtime, kernels = result["runtime"], result["kernels"] or {}
    return {
        "clips_done": bool(rows),
        "every_clip_published": all(f"publish/da3/{g1.safe(r['id'])}/depth.npz" in names for r in rows),
        "every_keyframe_has_depth": all(r["keyframes"] > 0 and r["depth_finite"] for r in rows),
        "on_ada_or_a6000": any(n in runtime["device"] for n in ("RTX 6000 Ada", "RTX A6000")),
        "bf16_autocast": bool(runtime["bf16"]),
        "fused_attention_no_math_fallback": bool(kernels.get("attention_ops")) and not kernels.get("math_fallback"),
        "pinned_weights": len(runtime["weights_sha256"]) == 64,
        "depth_at_patch_multiple": all(r["depth_shape"] and r["depth_shape"][0] % 14 == 0 and r["depth_shape"][1] % 14 == 0
                                       for r in rows),
    }


def command_validate(args: argparse.Namespace) -> int:
    checks = validate_stage(stage_dir())
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


def overlay_digest(overlay: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in overlay.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(overlay)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--prepared", required=True)
    run.add_argument("--select", required=True)
    run.add_argument("--limit-frames", type=int, default=0)
    run.add_argument("--weights", required=True, help="DA3 model.safetensors")
    run.add_argument("--config", required=True, help="DA3 config.json")
    run.add_argument("--overlay", required=True, help="extracted DA3 overlay (pip --target tree)")
    sub.add_parser("validate")
    args = parser.parse_args(argv)
    return {"run": command_run, "validate": command_validate}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
