"""G6 item 2: latency, excerpt oracle, held-out footage fit, and replay of narrower enhancers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background import g5, g5b, g5c, g5d, g6
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from src.codecs import svtav1
from src.codecs.enhancer import apply

BLOCKS = 8
WIDTHS = (32, 24, 16)


def decision(plain: float, reference: float, candidate: float, timing: dict[str, Any], fps: float) -> dict[str, Any]:
    """Retain 80% of the reference gain and meet the enhancement frame deadline."""
    gain = plain - reference
    retained = (plain - candidate) / gain if gain > 0 else None
    limit = 1000 / fps
    quality = retained is not None and retained >= 0.8 and candidate <= 0.9 * plain
    speed = timing["wall_ms_mean"] <= limit and timing["wall_ms_p95"] <= limit
    return {"retained_gain": retained, "limit_ms": limit, "quality_pass": quality,
            "speed_pass": speed, "passed": quality and speed}


def roundtrip(frames: np.ndarray) -> np.ndarray:
    return np.concatenate([g6.to_rgb(g5.rgb_to_yuv420(frames[i:i + 16]), full_range=True)
                           for i in range(0, len(frames), 16)])


def published_points(paths: list[str], item: str) -> dict[int, tuple[Path, dict[str, Any], dict[str, Any]]]:
    found = {}
    for path in paths:
        root = Path(path) / "publish"
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["window"] != item or manifest["foreground"]["dilate_px_1080"] != 0:
            raise ValueError("published window or exact mask mismatch")
        for anchor in manifest["points"]:
            if anchor["method"] == "svt-1080":
                enhanced = next(p for p in manifest["points"] if p["method"] == "enh-1080" and p["value"] == anchor["value"])
                found[anchor["value"]] = (root, manifest, {"anchor": anchor, "enhanced": enhanced})
    return found


def footage_pairs(args: argparse.Namespace, work: Path, manifest: dict[str, Any], crf: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    [(meta, yuv_path, keep_path)] = g5b.load_footage([args.footage])
    if meta["target"] != args.item or meta["set"] != "native":
        raise ValueError("expected native, own-video footage")
    if meta["yuv420_sha256"] != manifest["enhancer"]["footage"]["yuv420_sha256"]:
        raise ValueError("training footage differs from the full enhancer's")
    guard = round(meta["guard_s"] * meta["source_fps"])
    lo, hi = meta["window"][0] - guard, meta["window"][1] + guard
    heldout = all(s["last_index"] < lo or s["first_index"] > hi for s in meta["segments"])
    if not heldout or meta["guard_s"] < 5:
        raise ValueError("footage overlaps the scored window or its guard")
    ids = [i for seg in meta["segments"] for i in range(seg["offset"], seg["end"])]
    if os.environ.get("PS_STAGE") == "smoke":
        ids = ids[:48]
    yuv = np.ascontiguousarray(np.load(yuv_path, mmap_mode="r")[ids])
    keep = np.ascontiguousarray(np.load(keep_path, mmap_mode="r")[ids])
    source = work / "footage.yuv"
    yuv.tofile(source)
    target = g6.to_rgb(yuv, full_range=True)
    directory = work / f"footage-crf{crf}"
    directory.mkdir(parents=True, exist_ok=True)
    record = svtav1.code(source, directory, width=1920, height=1080, fps=Fraction(manifest["fps"]),
                         frames=len(ids), crf=crf, preset=g5.PRESET, threads=8, full_range=True,
                         encoder="SvtAv1EncApp", decoder="dav1d", timeout=600)
    decoded = np.fromfile(record["decoded"], np.uint8).reshape(len(ids), 1620, 1920)
    return g6.to_rgb(decoded, full_range=True), target, keep, {
        "frames": len(ids), "heldout": heldout, "guard_s": meta["guard_s"],
        "yuv420_sha256": meta["yuv420_sha256"], "keep_sha256": meta["keep_sha256"],
        "segments": meta["segments"], "window": meta["window"], "codec": record}


def command_run(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "narrow"
    work.mkdir(parents=True, exist_ok=True)
    publish = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"])
    if args.fit_state:
        # Preserve the real supplied training state even if smoke is interrupted
        # before its short fit. Keep it separate from smoke's fit signature.
        seed_directory = checkpoints / "seed-fits"
        seed_directory.mkdir(parents=True, exist_ok=True)
        for saved_fit in (Path(args.fit_state) / "checkpoint" / "fits").glob("*.pt"):
            target = seed_directory / saved_fit.name
            if not target.exists():
                temporary = target.with_name(f".{target.name}")
                shutil.copyfile(saved_fit, temporary)
                temporary.replace(target)
    available = published_points(args.published, args.item)
    crfs = [int(c) for c in args.crfs.split(",")]
    widths = [int(c) for c in args.channels.split(",")]
    if any(c not in WIDTHS for c in widths):
        raise ValueError("expected one of the declared channel widths")
    item, rgb, fg, source = g6.load_window(args, work)
    n = len(rgb)
    fps = Fraction(item["fps"]).limit_denominator(1001)
    clip = g5.Clip(item["id"], "window", fps, 0, rgb, ~fg,
                   {"dataset": (list(range(n)), ~fg), "own": (list(range(n)), ~fg)},
                   {"id": item["id"], "video": item["video"], "fps": str(fps), "frames": n})
    nets = g5c.load_nets(args)
    flows = None if args.mode == "latency" else g5c.source_flows(clip)
    rows = []
    completed = 0
    for crf in crfs:
        root, manifest, points = available[crf]
        if source["masks"]["mask_sha256"] != manifest["foreground"]["mask_sha256"]:
            raise ValueError("scoring mask identity mismatch")
        anchor, enhanced = points["anchor"], points["enhanced"]
        stream = root / anchor["bitstream"]
        if file_sha256(stream) != anchor["bitstream_sha256"]:
            raise ValueError("stream checksum mismatch")
        command = svtav1.decode_command("dav1d", stream, work / "decoded.yuv", threads=8)
        decode = svtav1.run(command, 120)
        yuv = np.fromfile(work / "decoded.yuv", np.uint8).reshape(manifest["frames"], 1620, 1920)
        if [hashlib.sha256(f.tobytes()).hexdigest() for f in yuv] != anchor["decoded_frame_sha256"]:
            raise ValueError("native decoded frames differ from the published anchor")
        decoded = g6.to_rgb(yuv[:n], full_range=True)
        reference = None
        plain_score = None
        if args.mode != "latency":
            assert flows is not None
            weights = root / enhanced["enhancer"]["weights"]
            if file_sha256(weights) != enhanced["enhancer"]["weights_sha256"]:
                raise ValueError("reference model checksum mismatch")
            teacher = g5d.enhancer(manifest["enhancer"]["blocks"], manifest["enhancer"]["channels"])
            teacher.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
            output, timing = apply(teacher, decoded, "fp16-channels-last")
            reference = {"timing": timing, "score": g5c.full_score(clip, roundtrip(output), nets, flows),
                         "weights_sha256": enhanced["enhancer"]["weights_sha256"]}
            del teacher, output
            plain_score = g5c.full_score(clip, decoded, nets, flows)
        if args.mode == "footage":
            pairs = footage_pairs(args, work, manifest, crf)
        else:
            pairs = decoded, rgb, ~fg, {"frames": n, "heldout": False, "fit": "scored excerpt"}
        for channels in widths:
            key = f"{args.mode}-crf{crf}-c{channels}"
            resume_checked = False
            weights_path = checkpoints / "models" / f"{key}.pt"
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            if args.mode == "replay":
                supplied = Path(args.model) / "publish" / "models" / f"footage-crf{crf}-c{channels}.pt"
                package = torch.load(supplied, map_location="cpu", weights_only=True)
                if package["blocks"] != BLOCKS or package["channels"] != channels or package["item"] != args.item or package["crf"] != crf:
                    raise ValueError("replay model does not match its declared operating point")
                net = g5d.enhancer(BLOCKS, channels)
                net.load_state_dict(package["state_dict"], strict=True)
                trained = package["train"]
                fit_record = package["footage"]
            else:
                fit_checkpoint = checkpoints / "fits" / f"{key}.pt"
                if args.fit_state and os.environ.get("PS_STAGE") != "smoke" and not fit_checkpoint.exists():
                    saved_fit = checkpoints / "seed-fits" / f"{key}.pt"
                    fit_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(saved_fit, fit_checkpoint)
                if os.environ.get("PS_STAGE") == "smoke":
                    continuous, _ = g5d.fit(pairs[0], pairs[1], pairs[2], "lpips", args.steps, nets.lpips,
                                            "cuda", blocks=BLOCKS, channels=channels)
                    if not fit_checkpoint.is_file():
                        partial, _ = g5d.fit(pairs[0], pairs[1], pairs[2], "lpips", args.steps, nets.lpips,
                                            "cuda", blocks=BLOCKS, channels=channels, checkpoint=fit_checkpoint,
                                            stop_after=max(1, args.steps // 2))
                        del partial
                    net, trained = g5d.fit(pairs[0], pairs[1], pairs[2], "lpips", args.steps, nets.lpips,
                                          "cuda", blocks=BLOCKS, channels=channels, checkpoint=fit_checkpoint)
                    resume_checked = all(torch.allclose(a, net.state_dict()[k], atol=1e-5, rtol=1e-5)
                                         for k, a in continuous.state_dict().items())
                    del continuous
                    if not resume_checked:
                        raise ValueError("checkpoint continuation differs from uninterrupted training")
                else:
                    net, trained = g5d.fit(pairs[0], pairs[1], pairs[2], "lpips", args.steps, nets.lpips,
                                          "cuda", blocks=BLOCKS, channels=channels, checkpoint=fit_checkpoint)
                fit_record = pairs[3]
            output, timing = apply(net, decoded, "fp16-channels-last")
            row: dict[str, Any] = {"crf": crf, "blocks": BLOCKS, "channels": channels,
                "frames": n, "kbps": anchor["kbps"], "stream_sha256": anchor["bitstream_sha256"],
                "decoded_hashes_match": True, "manifest_sha256": file_sha256(root / "manifest.json"),
                "train": trained, "footage": fit_record, "timing": timing,
                "decode_command": command, "dav1d_ms_per_frame": decode["wall"] * 1000 / manifest["frames"]}
            row["checkpoint_resume_checked"] = resume_checked
            if args.mode != "latency":
                assert plain_score is not None and reference is not None and flows is not None
                row["plain_score"], row["reference"] = plain_score, reference
                row["score"] = g5c.full_score(clip, roundtrip(output), nets, flows)
                row["decision"] = decision(plain_score["dataset"]["lpips_v"], reference["score"]["dataset"]["lpips_v"],
                                           row["score"]["dataset"]["lpips_v"], timing, float(fps))
            torch.save({"state_dict": {k: v.half() for k, v in net.state_dict().items()},
                        "blocks": BLOCKS, "channels": channels, "item": args.item, "crf": crf,
                        "train": trained, "footage": fit_record}, weights_path)
            row["weights_sha256"] = file_sha256(weights_path)
            row["weights_bytes"] = weights_path.stat().st_size
            model_directory = publish / "models"
            model_directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(weights_path, model_directory / weights_path.name)
            rows.append(row)
            write_json(stage_dir() / f"{key}.json", row)
            g5c.save(checkpoints, key, row)
            del net, output
            completed += 1
            progress(completed)
    record = {"stage": os.environ.get("PS_STAGE"), "mode": args.mode, "item": args.item, "frames": n, "fps": str(fps), "steps": args.steps,
              "crfs": crfs, "channels": widths, "blocks": BLOCKS, "source": source,
              "metrics": nets.record, "dav1d": svtav1.tool("dav1d"),
              "encoder": svtav1.tool("SvtAv1EncApp") if args.mode == "footage" else None,
              "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
              **g6.provenance(), **g5c.gpu_record(), "points": rows}
    write_json(stage_dir() / "g6-narrow.json", record)
    write_json(publish / "g6-narrow.json", record)
    return 0


def validate(stage: Path) -> dict[str, bool]:
    r = json.loads((stage / "g6-narrow.json").read_text())
    rows = r["points"]
    expected = {(c, w) for c in r["crfs"] for w in r["channels"]}
    checks = {"intended_gpu": tuple(r["gpu_capability"]) == (8, 9),
              "checkpoint_continuation": r["stage"] != "smoke" or r["mode"] == "replay"
                                         or all(p["checkpoint_resume_checked"] for p in rows),
              "every_point": {(p["crf"], p["channels"]) for p in rows} == expected,
              "native_decode_hashes": all(p["decoded_hashes_match"] for p in rows),
              "timed_every_frame": all(len(p["timing"]["wall_ms_per_frame"]) == r["frames"] for p in rows),
              "cuda_kernels": all(p["timing"]["cuda_kernels"] and "float16" in p["timing"]["dtype"] for p in rows),
              "training_complete": all(p["train"]["curve"][-1]["step"] == p["train"]["steps"]
                                       and (r["mode"] == "replay" or p["train"]["steps"] == r["steps"]) for p in rows),
              "finite_training": all(all(np.isfinite(c["mse_fixed"]) and np.isfinite(c["lpips_fixed"])
                                          for c in p["train"]["curve"]) for p in rows),
              "heldout_footage": r["mode"] not in ("footage", "replay") or all(p["footage"]["heldout"] for p in rows)}
    if r["mode"] != "latency":
        def finite(s: dict[str, Any]) -> bool:
            values = [s["dataset"][m] for m in ("lpips_v", "dists_v", "psnr_v")]
            values += [s["flicker"]["flicker_v"], s["kid"]["kid_v"]]
            return all(v is not None and np.isfinite(v) for v in values)
        checks["finite_scores"] = all(finite(p["score"]) and finite(p["plain_score"])
                                       and finite(p["reference"]["score"]) for p in rows)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("eval-set", "archive", "masks", "video", "item", "lpips-backbone", "dists-backbone", "inception"):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--published", nargs="+", required=True)
    run.add_argument("--mode", choices=("latency", "oracle", "footage", "replay"), required=True)
    run.add_argument("--channels", default="32,24,16")
    run.add_argument("--steps", type=int, required=True)
    run.add_argument("--limit-frames", type=int, required=True)
    run.add_argument("--crfs", default="48")
    run.add_argument("--footage", default=None)
    run.add_argument("--model", default=None)
    run.add_argument("--fit-state", default=None,
                     help="SHA256-staged checkpoint archive to continue an interrupted full-stage fit")
    val = sub.add_parser("validate")
    val.add_argument("--stage", default=None)
    args = parser.parse_args()
    if args.command == "run":
        return command_run(args)
    checks = validate(Path(args.stage) if args.stage else stage_dir())
    result = {"passed": all(checks.values()), "checks": checks}
    write_json(Path(os.environ["PS_VALIDATION_PATH"]), result)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
