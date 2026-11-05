"""K5d single-clip six-point pilot. Requires fleet admission before invocation.

The receiver is a fresh process fed PSBL and an installed resource map. This
establishes packet-path parity, not filesystem denial. Outputs are new scratch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.tennis import k5, k5b1b, k5b_trackers as trk, k5c_independent_ball as old
from experiments.tennis import k5d_source_oracle as oracle, windows as tw
from experiments.tennis.k5d_analysis import score_frames
from src.pointstream import ball, ball_stream

POINTS = old.POINTS


def pooled_psnr(source: np.ndarray, images: np.ndarray) -> float:
    """Pool RGB squared error without allocating full-clip float64 arrays."""
    if source.shape != images.shape or len(source) == 0:
        raise ValueError("image shape mismatch")
    total = 0.0
    count = 0
    for a, b in zip(source, images):
        diff = a.astype(np.float64) - b.astype(np.float64)
        total += float(np.sum(diff * diff))
        count += diff.size
    return k5.mse_to_psnr(total / count)


def installed_decoder_resources(image: Path, video: Path, decoder: Path) -> dict[str, Any]:
    """Generic installed resources are identical for native and PSBL arms."""
    def record(path: Path) -> dict[str, Any]:
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": oracle.sha(path)}
    image_record, video_record, native_decoder = record(image), record(video), record(decoder)
    return {"scope": "generic installed decoder resources; same for native and BALL arms; excluded from per-clip wire bytes",
            "dcvc": {"image_checkpoint": image_record, "video_checkpoint": video_record,
                     "checkpoint_total_bytes": image_record["bytes"] + video_record["bytes"]},
            "svt": {"decoder": native_decoder}}


def crop64(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """Fixed 64x64 window, shifted inside the image at boundaries."""
    h, w = image.shape[:2]
    if h < 64 or w < 64 or not (0 <= x < w and 0 <= y < h):
        raise ValueError("invalid crop centre or image")
    left = min(max(int(round(x)) - 32, 0), w - 64)
    top = min(max(int(round(y)) - 32, 0), h - 64)
    return image[top:top + 64, left:left + 64]


def metrics(source: np.ndarray, images: np.ndarray, labels: list[tuple[int, float, float]], nets: tuple[Any, Any] | None) -> list[dict[str, Any]]:
    if source.shape != images.shape or len(source) != len(labels):
        raise ValueError("metric frame count mismatch")
    from experiments.visor.h3b import dists_crops, lpips_reduce
    result = []
    for i, (a, b, (kind, x, y)) in enumerate(zip(source, images, labels)):
        diff = a.astype(np.int32) - b.astype(np.int32)
        sse, count = int(np.sum(diff.astype(np.int64) ** 2)), int(diff.size)
        row: dict[str, Any] = {"frame": i, "sse": sse, "pixels": count,
                               "frame_psnr": k5.mse_to_psnr(sse / count), "full_lpips": None,
                               "full_dists": None, "crop_lpips": None, "crop_dists": None}
        if nets is not None:
            lp, di = nets
            mask = np.ones(a.shape[:2], dtype=bool)
            row["full_lpips"] = lpips_reduce(lp, [b], a, [[mask]], "cuda")[0][0]
            row["full_dists"] = dists_crops(di, [b], a, mask, "cuda")[0]
            if kind == 1:
                ac, bc = crop64(a, x, y), crop64(b, x, y)
                cm = np.ones((64, 64), dtype=bool)
                row["crop_lpips"] = lpips_reduce(lp, [bc], ac, [[cm]], "cuda")[0][0]
                row["crop_dists"] = dists_crops(di, [bc], ac, cm, "cuda")[0]
        result.append(row)
    return result


def score_arms(source: np.ndarray, decoded: np.ndarray, composite: np.ndarray,
               labels: list[tuple[int, float, float]], nets: tuple[Any, Any] | None,
               judges: Callable[[np.ndarray], dict[str, Any]], guard: bool, timer: Any = None, prefix: str = ""
               ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], str | None]:
    if guard and not np.array_equal(composite, decoded):
        raise ValueError("guard-off composite differs from baseline pixels")
    def measure(name: str, fn: Any) -> Any:
        return timer.measure(name, fn) if timer is not None else fn()
    baseline_metrics = measure(f"metrics.{prefix}.baseline", lambda: metrics(source, decoded, labels, nets))
    baseline_judges = measure(f"judges.{prefix}.baseline", lambda: judges(decoded))
    if guard:
        return (baseline_metrics, baseline_judges, baseline_metrics, baseline_judges,
                "guard disabled BALL; composite verified byte-identical to native decode")
    return (baseline_metrics, baseline_judges,
            measure(f"metrics.{prefix}.streak", lambda: metrics(source, composite, labels, nets)),
            measure(f"judges.{prefix}.streak", lambda: judges(composite)), None)


def prediction_rows(labels: list[tuple[int, float, float]], predictions: list[trk.Prediction]) -> dict[str, Any]:
    if len(predictions) != len(labels):
        raise ValueError("judge prediction count mismatch")
    rows = [{"detected": bool(d), "x": float(x), "y": float(y)} for d, x, y in predictions]
    return {"rows": rows, "scores": score_frames(labels, predictions)}


def convert_rgb(ffmpeg: str, source: np.ndarray, destination: Path, fps: Fraction) -> list[str]:
    n, h, w, channels = source.shape
    if channels != 3 or source.dtype != np.uint8 or w % 2 or h % 2:
        raise ValueError("invalid source RGB geometry")
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "pipe:0", "-frames:v", str(n), "-vf",
           "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p", "-f", "rawvideo", str(destination)]
    subprocess.run(cmd, input=source.tobytes(), check=True, capture_output=True, timeout=1800)
    if destination.stat().st_size != n * w * h * 3 // 2:
        raise ValueError("converted frame count mismatch")
    return cmd


def fidelity_gate(path: Path, digest: str, archive_sha: str, *, purpose: str = "full-retrospective-benchmark",
                  clip: str | None = None, frames: int | None = None) -> dict[str, Any]:
    if oracle.sha(path) != digest:
        raise ValueError("fidelity gate receipt hash mismatch")
    receipt = json.loads(path.read_text())
    required = {"source_transform", "native_and_wire", "receiver_parity", "perceptual_metrics", "full_cost"}
    evidence = receipt.get("evidence_sha256")
    if (not isinstance(evidence, str) or len(evidence) != 64 or any(c not in "0123456789abcdef" for c in evidence)
            or receipt.get("schema_version") != 1 or receipt.get("passed") is not True
            or receipt.get("scope") != "fidelity-only" or receipt.get("archive_sha256") != archive_sha
            or not receipt.get("calibration_job_id") or not receipt.get("evidence_sha256")
            or not required.issubset(set(receipt.get("checks", [])))):
        raise ValueError("fidelity gate lacks independent calibration and cost checks")
    if receipt.get("purpose") != purpose:
        raise ValueError("fidelity gate purpose mismatch")
    if purpose == "48-frame-cost-pilot":
        if (receipt.get("clip") != "tracknet/game10/Clip3" or clip != receipt["clip"]
                or receipt.get("max_frames") != 48 or frames not in (16, 48)
                or receipt.get("calibration_job_id") != "20261104T223611Z-ed28cf8e"
                or receipt.get("accepted_smoke_sha256") != "dc1bc14133cf210589f2452785fbaf6d4905e91aae05fdc6d7ac0f25da8ec066"):
            raise ValueError("cost-pilot fidelity gate clip or frame limit mismatch")
    elif purpose == "188-frame-confirmation":
        if (receipt.get("clip") != "tracknet/game10/Clip3" or clip != receipt["clip"]
                or receipt.get("max_frames") != 188 or frames not in (16, 188)):
            raise ValueError("confirmation fidelity gate clip or frame limit mismatch")
    elif purpose != "full-retrospective-benchmark":
        raise ValueError("unrecognized fidelity gate purpose")
    return receipt


def validate(out: Path) -> dict[str, Any]:
    report = json.loads((out / "pilot.json").read_text())
    qualification = json.loads((out / "source-qualification.json").read_text())
    oracle.validate_report(qualification)
    if report["frames"] != qualification["frames"] or report["clip"] != qualification["clip"]:
        raise ValueError("source qualification frame count or clip mismatch")
    gate = report.get("fidelity_gate")
    if gate:
        fidelity_gate(out / "fidelity-gate.json", gate["sha256"], qualification["identities"]["archive"]["sha256"],
                      purpose=gate.get("purpose", "full-retrospective-benchmark"), clip=report["clip"], frames=report["frames"])
    elif not qualification["qualified"]:
        raise ValueError("source qualification failed without independent fidelity gate")

    if "source_task_qualified" in report and report["source_task_qualified"] != qualification["qualified"]:
        raise ValueError("source task qualification was relabeled")
    labels = [(r["label_kind"], r["label_x"], r["label_y"]) for r in qualification["rows"]]
    n = len(labels)
    width, height = report["size"]
    fps = Fraction(report["fps"])
    if (width <= 0 or height <= 0 or width % 2 or height % 2 or fps <= 0
            or report["duration_seconds"] != n / float(fps)):
        raise ValueError("invalid source geometry or duration")
    if [(p["codec"], p["point"]) for p in report["points"]] != list(POINTS):
        raise ValueError("six point coverage incomplete")
    if oracle.sha(out / "resources.json") != report["resource_map_sha256"]:
        raise ValueError("resource map mismatch")
    for p in report["points"]:
        directory = out / f'{p["codec"]}-{p["point"]}'
        wire_path, native = directory / "stream.psbl", directory / "native.bin"
        stream = ball_stream.unpack(wire_path.read_bytes())
        if (stream.frames != n or (stream.width, stream.height, stream.fps) != (width, height, fps) or stream.codec != ("dcvc-uf" if p["codec"] == "dcvc" else "svt-av1-ivf")
                or stream.native != native.read_bytes() or oracle.sha(wire_path) != p["wire_sha256"]
                or oracle.sha(native) != p["native_sha256"] or wire_path.stat().st_size != p["wire_bytes"]
                or len(stream.native) != p["native_bytes"] or p["wire_bytes"] <= p["native_bytes"]):
            raise ValueError("wire/native accounting mismatch")
        receipt = p["receiver"]
        if (receipt["wire_sha256"] != p["wire_sha256"] or receipt["frames"] != n
                or receipt["native_sha256"] != p["native_sha256"] or receipt["resource_map_sha256"] != report["resource_map_sha256"]
                or receipt["decoded_yuv_sha256"] != p["encoder_decoded_sha256"]
                or receipt["rendered_rgb_sha256"] != p["encoder_rendered_sha256"]
                or not p["decode_parity"] or not p["render_parity"]):
            raise ValueError("receiver parity failed")
        expected_rate = p["wire_bytes"] * 8 / (n / float(Fraction(report["fps"]))) / 1000
        if not math.isclose(p["wire_kbps"], expected_rate, rel_tol=1e-12):
            raise ValueError("wire rate mismatch")
        native_rate = p["native_bytes"] * 8 / (n / float(fps)) / 1000
        baseline = p["baseline"]
        zero_path = directory / "baseline.psbl"
        zero = ball_stream.unpack(zero_path.read_bytes())
        empty = ball.encode_frames([None] * n, [None] * n, width, height)
        if (p["native_record"]["stream_sha256"] != p["native_sha256"]
                or baseline["native_bytes"] != p["native_bytes"]
                or baseline["native_sha256"] != p["native_sha256"]
                or not math.isclose(baseline["native_kbps"], native_rate, rel_tol=1e-12)
                or zero.native != stream.native or zero.ball_payload != empty
                or (zero.width, zero.height, zero.fps, zero.frames, zero.codec, zero.resource_id)
                != (stream.width, stream.height, stream.fps, stream.frames, stream.codec, stream.resource_id)
                or baseline["zero_ball_wire_bytes"] != zero_path.stat().st_size
                or baseline["zero_ball_wire_sha256"] != oracle.sha(zero_path)
                or not math.isclose(baseline["zero_ball_wire_kbps"], zero_path.stat().st_size * 8 / (n / float(fps)) / 1000, rel_tol=1e-12)):
            raise ValueError("native or zero-ball wire accounting mismatch")
        for arm in ("baseline", "streak"):
            item = p[arm]
            if len(item["metrics"]) != n or [r["frame"] for r in item["metrics"]] != list(range(n)):
                raise ValueError("metric frame coverage mismatch")
            for i, row in enumerate(item["metrics"]):
                if "sse" in row and (type(row["sse"]) is not int or not 0 <= row["sse"] <= width * height * 3 * 255**2
                                     or row.get("pixels") != width * height * 3):
                    raise ValueError("invalid raw squared error")
                for key in ("frame_psnr", "full_lpips", "full_dists", "crop_lpips", "crop_dists"):
                    value = row[key]
                    required = key == "frame_psnr" or (report["full_metrics"] and (not key.startswith("crop") or labels[i][0] == 1))
                    if required and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                        raise ValueError("missing or nonfinite metric")
                    if not required and value is not None:
                        raise ValueError("unexpected metric frame")
            for name in ("wasb", "balltrack"):
                judge = item["judges"][name]
                preds = [(x["detected"], x["x"], x["y"]) for x in judge["rows"]]
                if len(preds) != n or score_frames(labels, preds) != judge["scores"]:
                    raise ValueError("judge frame coverage or counts mismatch")
        if p["streak"]["guard_off"] and p["baseline"]["metrics"] != p["streak"]["metrics"]:
            raise ValueError("disabled guard did not reproduce baseline")
    return {"frames": n, "points": len(report["points"]), "full_metrics": report["full_metrics"],
            "scope": "packet replay; filesystem denial not established"}


def run(args: argparse.Namespace) -> None:
    import torch
    from src.codecs import quality, svtav1

    started = time.monotonic()
    cpu_started = time.process_time()
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    inputs = [Path(getattr(args, key)) for key in ("manifest", "archive", "balltrack_code", "balltrack_weight", "wasb_code", "wasb_weights", "wasb_config", "exposure", "image_ckpt", "video_ckpt", "lpips_backbone", "dists_backbone")]
    oracle.validate_output(args.output, scratch, inputs)
    if len(args.code_revision) != 40 or any(c not in "0123456789abcdef" for c in args.code_revision):
        raise ValueError("code revision must be a full lowercase Git SHA1")
    local_root = os.environ.get("PS_LOCAL_ROOT")
    if local_root and not Path(args.archive).resolve().is_relative_to(Path(local_root).resolve()):
        raise ValueError("archive is outside PS_LOCAL_ROOT")
    benchmark = bool(getattr(args, "benchmark", False))
    cost_pilot = bool(getattr(args, "cost_pilot", False))
    checkpoint_prepare = bool(getattr(args, "checkpoint_prepare", False))
    if cost_pilot and (benchmark or args.clip != "tracknet/game10/Clip3" or args.frames not in (16, 48)):
        raise ValueError("cost pilot is restricted to game10/Clip3 at 16 or 48 frames")
    if args.start_frame != 0 or (not benchmark and args.clip not in ("tracknet/game9/Clip5", "tracknet/game10/Clip3")):
        raise ValueError("only registered clips starting at zero")
    if len(os.sched_getaffinity(0)) != 1 or os.environ.get("PS_CPU_ALLOWANCE") != "1" or torch.cuda.device_count() != 1:
        raise ValueError("requires one CPU and one CUDA device")
    for name in ("ffmpeg", "encoder", "decoder"):
        if not Path(getattr(args, name)).is_absolute():
            raise ValueError("native tools must be absolute paths")
    selection = oracle.select(json.loads(Path(args.manifest).read_text()), args.clip, 0, args.frames)
    if ((benchmark and args.frames != selection["label_rows"] and not (os.environ.get("PS_STAGE") == "smoke" and args.frames == 16))
            or (not benchmark and not cost_pilot and args.frames not in (16, selection["label_rows"]))):
        raise ValueError("benchmark requires full clip; calibration requires smoke or full clip")
    args.output.mkdir(parents=True, exist_ok=False)
    if cost_pilot:
        if getattr(args, "timer", None) is None:
            from experiments.tennis.k5d_timing import Timer
            args.timer = Timer(args.output / "timing.json")
        else:
            args.timer.relocate(args.output / "timing.json")
    timer = getattr(args, "timer", None)
    def measure(name: str, fn: Any) -> Any:
        return timer.measure(name, fn) if timer is not None else fn()
    for name in ("image_ckpt", "video_ckpt", "lpips_backbone", "dists_backbone"):
        if measure(f"preflight.hash.{name}", lambda: oracle.sha(Path(getattr(args, name)))) != getattr(args, name + "_sha256"):
            raise ValueError(f"{name} hash mismatch")
    try:
        report = measure("source.oracle_total", lambda: oracle.run(args))
    except Exception as exc:
        (args.output / "source-qualification-error.json").write_text(json.dumps(
            {"qualified": False, "error": f"{type(exc).__name__}: {exc}",
             "elapsed_seconds": time.monotonic() - started}) + "\n")
        raise
    (args.output / "source-qualification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    gate_record = None
    gate_purpose = getattr(args, "gate_purpose", None)
    if gate_purpose not in (None, "188-frame-confirmation") or (gate_purpose and (benchmark or cost_pilot)):
        raise ValueError("invalid or conflicting explicit gate purpose")
    if benchmark or gate_purpose or (cost_pilot and not report["qualified"]):
        gate_path = Path(args.fidelity_gate)
        purpose = "48-frame-cost-pilot" if cost_pilot else gate_purpose or "full-retrospective-benchmark"
        fidelity_gate(gate_path, args.fidelity_gate_sha256, args.archive_sha256, purpose=purpose, clip=args.clip, frames=args.frames)
        shutil.copyfile(gate_path, args.output / "fidelity-gate.json")
        gate_record = {"sha256": args.fidelity_gate_sha256, "scope": "fidelity-only", "purpose": purpose}
    elif not report["qualified"]:
        raise ValueError("source qualification failed; retained before coding")
    args.image_sha256 = args.image_ckpt_sha256
    args.video_sha256 = args.video_ckpt_sha256
    clip = next(c for c in trk.tracknet_clips(Path(args.archive), [int(args.clip.split('/')[1][4:])]) if c["id"] == args.clip)
    clip = {**clip, "rows": clip["rows"][:args.frames]}
    source = measure("source.encoder_rgb_decode", lambda: trk.tracknet_frames(Path(args.archive), clip))
    labels = trk.tracknet_labels(clip)
    n, h, w, _ = source.shape
    fps = Fraction(30, 1)
    work = scratch / (args.output.name + "-work")
    work.mkdir(exist_ok=False)
    yuv = work / "source.yuv"
    conversion = measure("source.rgb_to_yuv", lambda: convert_rgb(args.ffmpeg, source, yuv, fps))
    converted = measure("source.yuv_to_rgb_floor", lambda: tw.rgb_frames(yuv, w, h, n, work / "converted.npy"))
    tracker = measure("model.balltrack_init", lambda: trk.BallTrack(Path(args.balltrack_code), Path(args.balltrack_weight)))
    zoo = measure("model.wasb_init", lambda: trk.Zoo(Path(args.wasb_code), Path(args.wasb_weights), names=("wasb-step1",)))
    if next(tracker.net.parameters()).device.type != "cuda" or next(zoo.settings["wasb-step1"]["net"].parameters()).device.type != "cuda":
        raise ValueError("judge device mismatch")
    def judges(images: np.ndarray) -> dict[str, Any]:
        return {"wasb": measure("judge.wasb", lambda: prediction_rows(labels, zoo.run("wasb-step1", images))),
                "balltrack": measure("judge.balltrack", lambda: prediction_rows(labels, tracker.run(images)))}
    nets = None if args.feasibility_no_metrics else (
        measure("model.lpips_init", lambda: quality.load_lpips(Path(args.lpips_backbone), "cuda")),
        measure("model.dists_init", lambda: quality.load_dists(Path(args.dists_backbone), "cuda")))
    start = measure("encoder.source_track", lambda: k5b1b.track_of(tracker.run(source), (h, w)))
    window = {"size": [w, h], "fps": float(fps)}
    coded = []
    for kind, point in POINTS:
        path = work / f"{kind}-{point}"
        rec = measure(f"native.{kind}.{point}.total", lambda: k5.dcvc_attempt(args, yuv, path, window, n, point) if kind == "dcvc" else k5.code_svt(args, yuv, path, window, n, point, sorted(os.sched_getaffinity(0))))
        if kind == "dcvc" and not (rec["deterministic"] and rec["decoder_matches_encoder_intra"] and rec["luma_psnr"] >= k5.SANE_FRAME_PSNR):
            raise ValueError("DCVC native validation failed")
        native = path / ("stream.psdc" if kind == "dcvc" else "stream.ivf")
        decoded = measure(f"native.{kind}.{point}.yuv_to_rgb", lambda: tw.rgb_frames(Path(rec["decoded"]), w, h, n, path / "decoded.npy"))
        directory = args.output / f"{kind}-{point}"
        directory.mkdir()
        shutil.copyfile(native, directory / "native.bin")
        if kind == "dcvc":
            for name in ("plan.json", "encode.json", "decode.json"):
                shutil.copyfile(path / name, directory / name)
        (directory / "native-checkpoint.json").write_text(json.dumps(rec, indent=2, allow_nan=False) + "\n")
        coded.append((kind, point, native, rec, decoded))
    fits = measure("encoder.fit_streak", lambda: old.fit_streak(start, source, [x[4] for x in coded], (h, w)))
    resource_map = {"resources": {"svt": {"codec": "svt-av1-ivf", "decoder": args.decoder, "decoder_sha256": oracle.sha(Path(args.decoder))},
                      "dcvc": {"codec": "dcvc-uf", "image_ckpt": args.image_ckpt, "image_ckpt_sha256": args.image_ckpt_sha256,
                               "video_ckpt": args.video_ckpt, "video_ckpt_sha256": args.video_ckpt_sha256}}}
    map_path = work / "resources.json"
    map_path.write_text(json.dumps(resource_map))
    shutil.copyfile(map_path, args.output / "resources.json")
    results = []
    for (kind, point, native, rec, decoded), fit in zip(coded, fits):
        track, apps = fit["track"], fit["appearances"]
        guard = measure(f"encoder.{kind}.{point}.guard", lambda: pooled_psnr(source, old.render_frames(decoded, track, apps)) < pooled_psnr(source, decoded))
        if guard:
            track, apps = [None] * n, [None] * n
        payload = measure(f"encoder.{kind}.{point}.ball_encode", lambda: ball.encode_frames(track, apps, w, h))
        stream = ball_stream.Stream(w, h, fps, n, "dcvc-uf" if kind == "dcvc" else "svt-av1-ivf", kind, native.read_bytes(), payload)
        directory = args.output / f"{kind}-{point}"
        if kind == "dcvc":
            from src.codecs.dcvc_uf_worker import dcvc_command
            rec["commands"] = {action: dcvc_command(Path(sys.prefix), action, native.parent / "plan.json",
                              native.parent / f"{action}.json")[0] for action in ("encode", "decode")}
        published = directory / "native.bin"
        wire = directory / "stream.psbl"
        wire.write_bytes(ball_stream.pack(stream))
        baseline_wire = directory / "baseline.psbl"
        baseline_stream = ball_stream.Stream(w, h, fps, n, stream.codec, stream.resource_id, stream.native,
                                             ball.encode_frames([None] * n, [None] * n, w, h))
        baseline_wire.write_bytes(ball_stream.pack(baseline_stream))
        receive = work / f"receiver-{kind}-{point}"
        cmd = [sys.executable, "-m", "experiments.tennis.k5d_worker", "decode", "--wire", str(wire), "--resources", str(map_path), "--work-dir", str(receive)]
        measure(f"receiver.{kind}.{point}", lambda: subprocess.run(cmd, check=True, capture_output=True, timeout=3600))
        receipt = json.loads((receive / "receiver.json").read_text())
        if cost_pilot and kind == "dcvc":
            shutil.copyfile(receive / "native-decoder-report.json", directory / "receiver-native-decoder-report.json")
        parity = oracle.sha(Path(rec["decoded"])) == receipt["decoded_yuv_sha256"]
        composite = old.render_frames(decoded, track, apps)
        render_parity = __import__("hashlib").sha256(composite.tobytes()).hexdigest() == receipt["rendered_rgb_sha256"]
        if not parity or not render_parity:
            raise ValueError("receiver parity failed")
        encoder_decoded_sha256 = oracle.sha(Path(rec.pop("decoded")))
        if checkpoint_prepare:
            baseline_metrics: list[dict[str, Any]] = []
            baseline_judges: dict[str, Any] = {}
            streak_metrics: list[dict[str, Any]] = []
            streak_judges: dict[str, Any] = {}
            reuse: str | None = None
        else:
            baseline_metrics, baseline_judges, streak_metrics, streak_judges, reuse = score_arms(
                source, decoded, composite, labels, nets, judges, guard, timer, f"{kind}.{point}")
        result = {"codec": kind, "point": point, "native_record": rec, "native_bytes": published.stat().st_size,
                  "native_sha256": oracle.sha(published), "wire_bytes": wire.stat().st_size, "wire_sha256": oracle.sha(wire),
                  "receiver": receipt, "decode_parity": parity, "render_parity": render_parity,
                  "encoder_decoded_sha256": encoder_decoded_sha256,
                  "encoder_rendered_sha256": __import__("hashlib").sha256(composite.tobytes()).hexdigest(),
                  "wire_kbps": wire.stat().st_size * 8 / (n / float(fps)) / 1000,
                  "baseline": {"native_bytes": published.stat().st_size, "native_sha256": oracle.sha(published),
                               "native_kbps": published.stat().st_size * 8 / (n / float(fps)) / 1000,
                               "zero_ball_wire_bytes": baseline_wire.stat().st_size, "zero_ball_wire_sha256": oracle.sha(baseline_wire),
                               "zero_ball_wire_kbps": baseline_wire.stat().st_size * 8 / (n / float(fps)) / 1000,
                               "metrics": baseline_metrics, "judges": baseline_judges},
                  "streak": {"guard_off": guard, "metrics": streak_metrics, "judges": streak_judges, "reuse": reuse}}
        results.append(result)
        (directory / "point.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    output = {"frames": n, "clip": args.clip, "size": [w, h], "fps": str(fps), "duration_seconds": n / float(fps),
              "full_metrics": nets is not None,
              "scope": ("16/48-frame bounded cost pilot; filesystem denial not established" if cost_pilot else
                        "feasibility only" if nets is None else "six-point calibration; filesystem denial not established"),
              "conversion": {"command": conversion, "yuv_sha256": measure("source.hash.yuv", lambda: oracle.sha(yuv)),
                             "floor_psnr": measure("source.conversion_floor", lambda: pooled_psnr(source, converted))},
              "resource_map_sha256": oracle.sha(map_path), "fidelity_gate": gate_record,
              "installed_decoder_resources": measure("resources.hash_installed", lambda: installed_decoder_resources(Path(args.image_ckpt), Path(args.video_ckpt), Path(args.decoder))),
              "source_task_qualified": bool(report["qualified"]),
              "cost_pilot": cost_pilot, "metric_geometry": {"width": w, "height": h, "frames": n,
                    "visible_crop_frames": sum(label[0] == 1 for label in labels), "crop_size": 64, "arms_per_point": 2},
              "metric_models": None if nets is None else {"lpips": quality.lpips_record(Path(args.lpips_backbone)),
                                                             "dists": quality.dists_record(Path(args.dists_backbone))},
              "identities": {name: {"path": getattr(args, name), "sha256": getattr(args, name + "_sha256")} for name in ("image_ckpt", "video_ckpt", "lpips_backbone", "dists_backbone")},
              "tools": {name: svtav1.tool(name) for name in (args.ffmpeg, args.encoder, args.decoder)},
              "elapsed_seconds": time.monotonic() - started, "parent_cpu_seconds": time.process_time() - cpu_started,
              "cpu_accounting": "parent process only; fleet must meter whole process tree", "python": sys.version, "torch": torch.__version__,
              "revision": args.code_revision, "snapshot_identity": getattr(args, "snapshot_identity", None),
              "encoder_source_sha256": oracle.sha(Path(__file__)), "points": results}
    if checkpoint_prepare:
        output["checkpoint_prepare"] = True
    (args.output / "pilot.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    validate(args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=Path)
    for name in ("manifest", "archive", "balltrack-code", "balltrack-weight", "wasb-code", "wasb-weights", "wasb-config", "exposure", "image-ckpt", "video-ckpt", "lpips-backbone", "dists-backbone"):
        parser.add_argument("--" + name)
        parser.add_argument("--" + name + "-sha256")
    for name in ("clip", "ffmpeg", "encoder", "decoder", "code-revision"):
        parser.add_argument("--" + name)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--feasibility-no-metrics", action="store_true")
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate(args.validate)))
    else:
        required = ("manifest", "archive", "balltrack_code", "balltrack_weight", "wasb_code", "wasb_weights", "wasb_config", "exposure", "image_ckpt", "video_ckpt", "lpips_backbone", "dists_backbone")
        if any(not getattr(args, n) or not getattr(args, n + "_sha256") for n in required) or not all((args.clip, args.ffmpeg, args.encoder, args.decoder, args.code_revision, args.frames, args.output)):
            parser.error("all inputs, their SHA256s, clip, tools, frames and output are required")
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
