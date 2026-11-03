"""Sequential, bounded-window component pilot; not an isolated complete-codec receiver.

This entrypoint is preparation only. A fleet card, aggregate process-tree CPU
metering, training-manifest audit and user authorization are required before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.tennis import k5, k5b1b, k5b_trackers as trk
from experiments.tennis import windows as tw
from src.pointstream import ball
from src.pointstream.bitstream import Container, Tag

POINTS = (("dcvc", 0), ("dcvc", 21), ("dcvc", 42), ("svt", 48), ("svt", 55), ("svt", 62))
WINDOW = "tracknet/game8/Clip6"



WASB_SHA = "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"
CONFIG_SHA = "6dbcad60e207bf684d0094f59aa922b107be1c1ce83998dc5b167ed59b0d1e0d"
BALLTRACK_SHA = "b052479e6229d840a1a7f42dbc83a6b752a53ef8bc5841a4b957acd519563fec"


def load_window(path: Path, frames: int) -> dict[str, Any]:
    data = json.loads(path.read_text())
    # K4 emits an object with a windows array; accept a list as an explicit
    # alternate export, while rejecting missing, duplicate or malformed rows.
    windows = data.get("windows") if isinstance(data, dict) else data
    if not isinstance(windows, list):
        raise ValueError("window manifest must contain a windows list")
    selected = [row for row in windows if isinstance(row, dict) and row.get("id") == WINDOW]
    if len(selected) != 1:
        raise ValueError("expected exactly one selected window")
    window = selected[0]
    if window.get("dataset") != "tracknet" or int(window.get("frames", 0)) < frames:
        raise ValueError("wrong or too-short window")
    rows = window.get("labels")
    if not isinstance(rows, list) or len(rows) < frames:
        raise ValueError("missing labels")
    for row in rows[:frames]:
        if not isinstance(row, (list, tuple)) or len(row) != 3 or row[0] not in (0, 1, 2, 3):
            raise ValueError("invalid label")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in row):
            raise ValueError("nonfinite label")
    return window


def admit(receipt: Path, config: Path, weights: Path, manifest: Path) -> dict[str, Any]:
    """Bind author-reported split to exact checkpoint and selected window manifest."""
    r = json.loads(receipt.read_text())
    expected = {"scope": "author-reported-split-not-training-log", "window": WINDOW,
                "wasb_sha256": WASB_SHA, "config_sha256": CONFIG_SHA,
                "window_manifest_sha256": sha(manifest), "train_games": list(range(1, 8)),
                "test_games": [8, 9, 10]}
    if any(r.get(key) != value for key, value in expected.items()):
        raise ValueError("exposure receipt missing or mismatched")
    if sha(config) != CONFIG_SHA or sha(weights) != WASB_SHA:
        raise ValueError("WASB input hashes differ from documented model and config")
    return r


def fit_streak(start: ball.Track, source: np.ndarray, decodes: list[np.ndarray],
               shape: tuple[int, int]) -> list[dict[str, Any]]:
    if not any(p is not None for p in start):
        return [{"track": [None] * len(start), "appearances": [None] * len(start), "delta_se": 0.0}
                for _ in decodes]
    return ball.fit_frames(ball.FrameCrops.build(start, source, decodes), ball.grid("streak"), shape)


def one_cpu(affinity: set[int], allowance: str | None) -> list[int]:
    """Reject unverified multi-core execution instead of constructing empty sets."""
    if allowance != "1" or len(affinity) != 1:
        raise ValueError("pilot requires PS_CPU_ALLOWANCE=1 and exactly one permitted CPU")
    return sorted(affinity)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger(native: Path, packet: Path, seconds: float) -> dict[str, Any]:
    if seconds <= 0:
        raise ValueError("duration must be positive")
    n, b = native.stat().st_size, packet.stat().st_size
    return {"native_bytes": n, "ball_container_bytes": b, "total_bytes": n + b,
            "kbps": (n + b) * 8 / seconds / 1000, "native_sha256": sha(native), "ball_sha256": sha(packet)}


def frame_results(preds: list[trk.Prediction], labels: list[tuple[int, float, float]]) -> dict[str, Any]:
    if len(preds) != len(labels):
        raise ValueError("prediction/label length mismatch")
    rows = []
    for index, ((detected, x, y), (kind, lx, ly)) in enumerate(zip(preds, labels)):
        if kind not in (-1, 0, 1):
            raise ValueError("unknown label kind")
        distance = math.hypot(x - lx, y - ly) if detected and kind == 1 else None
        rows.append({"frame": index, "label_kind": kind, "label_x": lx, "label_y": ly,
                     "detected": detected, "x": x if detected else None, "y": y if detected else None,
                     "distance": distance})
    counts = trk.counts(preds, labels)
    return {"frames": rows, "counts": counts, "rates": trk.rates(counts)}


def judge_images(images: np.ndarray, labels: list[tuple[int, float, float]],
                 wasb: Callable[[np.ndarray], list[trk.Prediction]],
                 balltrack: Callable[[np.ndarray], list[trk.Prediction]]) -> dict[str, Any]:
    """Both callables receive pixels only; coordinate payloads are not inputs."""
    if len(images) != len(labels):
        raise ValueError("image/label length mismatch")
    return {"wasb": frame_results(wasb(images), labels), "balltrack": frame_results(balltrack(images), labels)}


def packet_write(path: Path, track: ball.Track, apps: list[ball.Appearance | None],
                 width: int, height: int, fps: Fraction) -> None:
    packed = Container(width, height, fps, len(track))
    packed.add(Tag.BALL, ball.encode_frames(track, apps, width, height))
    path.write_bytes(packed.pack())


def packet_read(path: Path, width: int, height: int, fps: Fraction, frames: int
                ) -> tuple[ball.Track, list[ball.Appearance | None]]:
    packed = Container.unpack(path.read_bytes())
    if (packed.width, packed.height, packed.fps, packed.frames) != (width, height, fps, frames):
        raise ValueError("BALL packet geometry differs")
    if len(packed.segments) != 1 or packed.segments[0][0] != Tag.BALL:
        raise ValueError("expected exactly one BALL segment")
    return ball.decode_frames(packed.segments[0][1], frames, width, height)


def render_frames(decoded: np.ndarray, track: ball.Track, apps: list[ball.Appearance | None]) -> np.ndarray:
    if len(decoded) != len(track) or len(track) != len(apps):
        raise ValueError("frame count mismatch")
    velocity = ball.velocities(track)
    return np.stack([ball.render(frame, i, track, velocity, apps) for i, frame in enumerate(decoded)])


def psnr(source: np.ndarray, images: np.ndarray) -> float:
    if source.shape != images.shape:
        raise ValueError("image shape mismatch")
    diff = source.astype(np.float64) - images.astype(np.float64)
    return k5.mse_to_psnr(float(np.mean(diff * diff)))


def _native_decode(kind: str, point_dir: Path, native: Path, destination: Path, decoder: str) -> None:
    """Decode again from persisted native stream; no source isolation is claimed."""
    if kind == "svt":
        from src.codecs.svtav1 import decode_command
        cmd = decode_command(decoder, native, destination, threads=1)
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
    else:
        from src.codecs.dcvc_uf_worker import dcvc_command
        plan = json.loads((point_dir / "plan.json").read_text())
        plan["container"] = str(native)
        plan["out_file"] = str(destination)
        receiver_plan = point_dir / "receiver-plan.json"
        receiver_plan.write_text(json.dumps(plan))
        cmd, env, cwd = dcvc_command(Path(sys.prefix), "decode", receiver_plan, point_dir / "receiver-report.json")
        subprocess.run(cmd, env={**k5.STAGE_ENV, **env}, cwd=cwd, check=True, capture_output=True, timeout=3600)


def run(args: argparse.Namespace) -> None:
    # This does not enforce aggregate CPU/GPU budgets; the fleet supervisor must.
    cpus = one_cpu(set(os.sched_getaffinity(0)), os.environ.get("PS_CPU_ALLOWANCE"))
    if not 1 <= args.frames <= 48:
        raise ValueError("pilot permits 1–48 frames")
    stage = os.environ.get("PS_STAGE")
    if stage in ("smoke", "full") and args.frames != {"smoke": 16, "full": 48}[stage]:
        raise ValueError("fleet smoke requires 16 frames and full requires 48")
    window = load_window(Path(args.windows), args.frames)
    exposure = admit(Path(args.exposure_receipt), Path(args.wasb_config),
                     Path(args.wasb_weights) / "wasb_tennis_best.pth.tar", Path(args.windows))
    if sha(Path(args.balltrack)) != BALLTRACK_SHA:
        raise ValueError("BallTrack checkpoint hash differs")
    for path, expected in ((Path(args.image_ckpt), args.image_sha256), (Path(args.video_ckpt), args.video_sha256)):
        if sha(path) != expected:
            raise ValueError("native checkpoint hash differs")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    from src.codecs.svtav1 import tool
    identities = {"window_manifest_sha256": sha(Path(args.windows)), "exposure_receipt_sha256": sha(Path(args.exposure_receipt)),
                  "wasb_sha256": WASB_SHA, "balltrack_sha256": BALLTRACK_SHA,
                  "image_sha256": args.image_sha256, "video_sha256": args.video_sha256,
                  "tools": {name: tool(name) for name in (args.ffmpeg, args.encoder, args.decoder)}}
    if window["source"]["kind"] == "jpeg_dir":
        identities["source_frames"] = [sha(Path(args.source) / (window["source"]["pattern"] % i))
                                       for i in range(args.frames)]
    else:
        identities["source_sha256"] = sha(Path(args.source))
    scratch = os.environ.get("PS_SCRATCH_DIR")
    if not scratch and (args.output is None or args.work_dir is None):
        raise ValueError("specify output and work-dir, or set PS_SCRATCH_DIR")
    out = Path(args.output) if args.output else Path(scratch or "") / "publish"
    work = Path(args.work_dir) if args.work_dir else Path(scratch or "") / "k5c-work"
    if out.resolve() == work.resolve() or out.resolve() in work.resolve().parents or work.resolve() in out.resolve().parents:
        raise ValueError("publish and work directories must be separate")
    out.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.windows, out / "windows.json")
    shutil.copyfile(args.exposure_receipt, out / "exposure-receipt.json")
    w, h = window["size"]
    fps = Fraction(window["fps"]).limit_denominator(1001)
    labels = k5b1b.label_kinds(tw.labels(window, args.frames))
    if len(labels) != args.frames:
        raise ValueError("missing labels")
    yuv = work / "source.yuv"
    conversion = tw.convert(args.ffmpeg, window, Path(args.source), yuv, args.frames)
    source = tw.rgb_frames(yuv, w, h, args.frames, work / "source.npy")
    tracker = trk.BallTrack(Path(args.balltrack_code), Path(args.balltrack))
    zoo = trk.Zoo(Path(args.wasb_code), Path(args.wasb_weights), names=("wasb-step1",))
    on_cuda = {"balltrack": str(next(tracker.net.parameters()).device),
               "wasb": str(next(zoo.settings["wasb-step1"]["net"].parameters()).device)}
    if not torch.cuda.is_available() or not all(x.startswith("cuda") for x in on_cuda.values()):
        raise ValueError("both judges must execute on the claimed CUDA device")
    predict_ball = tracker.run
    def predict_wasb(pixels: np.ndarray) -> list[trk.Prediction]:
        return zoo.run("wasb-step1", pixels)
    source_judges = judge_images(source, labels, predict_wasb, predict_ball)
    source_recall = source_judges["wasb"]["rates"]["detection_rate"]
    (out / "source-qualification.json").write_text(json.dumps({
        "window": WINDOW, "frames": args.frames, "identities": identities,
        "on_cuda": on_cuda, "reference_labels": labels, "source_judges": source_judges,
        "passed": source_recall is not None and source_recall >= 0.8,
        "scope": "source-image judge qualification only; no native or rendered outcome",
    }, indent=2, default=str) + "\n")
    if source_recall is None or source_recall < 0.8:
        raise ValueError("WASB source-frame qualification below 0.8 or undefined")

    coded: list[tuple[str, int, Path, dict[str, Any], np.ndarray]] = []
    for kind, point in POINTS:
        directory = work / f"{kind}-{point}"
        published = out / f"{kind}-{point}"
        published.mkdir()
        if directory.exists():
            raise ValueError("refusing to overwrite a point directory")
        record = (k5.dcvc_attempt(args, yuv, directory, window, args.frames, point) if kind == "dcvc"
                  else k5.code_svt(args, yuv, directory, window, args.frames, point, cpus))
        if kind == "dcvc" and not (record["deterministic"] and record["decoder_matches_encoder_intra"]
                                   and record["luma_psnr"] >= k5.SANE_FRAME_PSNR):
            raise ValueError("native DCVC decode validation failed")
        # The original coding helper validates its own decode; receive again from the saved stream.
        received = directory / "receiver.yuv"
        native = published / ("stream.psdc" if kind == "dcvc" else "stream.ivf")
        shutil.copyfile(directory / native.name, native)
        _native_decode(kind, directory, native, received, args.decoder)
        encoded_decode_hash = sha(Path(record["decoded"]))
        receiver_decode_hash = sha(received)
        if receiver_decode_hash != encoded_decode_hash:
            raise ValueError("saved-stream receiver decode differs from encoder-side decode")
        decoded = tw.rgb_frames(received, w, h, args.frames, directory / "receiver.npy")
        record["encoder_decoded_sha256"] = encoded_decode_hash
        record["receiver_decoded_sha256"] = receiver_decode_hash
        record.pop("decoded", None)
        record.pop("stream", None)
        coded.append((kind, point, published, record, decoded))
    start = k5b1b.track_of(predict_ball(source), (h, w))
    fits = fit_streak(start, source, [x[4] for x in coded], (h, w))
    if len(fits) != len(coded):
        raise ValueError("fit count mismatch")
    results: list[dict[str, Any]] = []
    for (kind, point, directory, record, decoded), fit in zip(coded, fits):
        track, apps = fit["track"], fit["appearances"]
        rendered = render_frames(decoded, track, apps)
        guard_off = psnr(source, rendered) < psnr(source, decoded)
        if guard_off:
            track, apps = [None] * args.frames, [None] * args.frames
        packet = directory / "ball.psb"
        packet_write(packet, track, apps, w, h, fps)
        received_track, received_apps = packet_read(packet, w, h, fps, args.frames)
        if (track, apps) != (received_track, received_apps):
            raise ValueError("BALL roundtrip mismatch")
        composite = render_frames(decoded, received_track, received_apps)
        native = directory / ("stream.psdc" if kind == "dcvc" else "stream.ivf")
        entry = {"codec": kind, "point": point, "native_record": record,
                 "baseline": {"bytes": native.stat().st_size, "sha256": sha(native),
                              "frame_psnr": psnr(source, decoded),
                              "judges": judge_images(decoded, labels, predict_wasb, predict_ball)},
                 "streak": {"guard_off": guard_off, "ball_on": any(p is not None for p in received_track),
                            "roundtrip": True, "ledger": ledger(native, packet, args.frames / float(fps)),
                            "frame_psnr": psnr(source, composite),
                            "judges": judge_images(composite, labels, predict_wasb, predict_ball)}}
        results.append(entry)
    output = {"window": WINDOW, "frames": args.frames, "fps": str(fps), "conversion": conversion,
              "exposure": {"development": True, "basis": exposure, "training_logs_independently_verified": False},
              "on_cuda": on_cuda, "identities": identities, "size": [w, h], "quality_space": "RGB PSNR",
              "receiver": "native stream and BALL packet reread; source isolation not established",
              "cpu_budget": "single affinity only; aggregate process-tree enforcement not established",
              "reference_labels": labels, "source_judges": source_judges, "points": results}
    (out / "pilot.json").write_text(json.dumps(output, indent=2, default=str) + "\n")


def validate(output: Path) -> dict[str, Any]:
    """Audit saved point identities, byte ledgers and all per-frame judge records."""
    report = json.loads((output / "pilot.json").read_text())
    frames = report["frames"]
    if report["window"] != WINDOW or not 1 <= frames <= 48:
        raise ValueError("invalid pilot scope")
    if report["quality_space"] != "RGB PSNR" or not all(
        str(report["on_cuda"][name]).startswith("cuda") for name in ("wasb", "balltrack")
    ):
        raise ValueError("invalid quality space or device")
    if [(x["codec"], x["point"]) for x in report["points"]] != list(POINTS):
        raise ValueError("missing, repeated or reordered points")
    def finite(value: Any) -> bool:
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, list):
            return all(finite(x) for x in value)
        if isinstance(value, dict):
            return all(finite(x) for x in value.values())
        return True
    if not finite(report):
        raise ValueError("nonfinite report")
    if sha(output / "windows.json") != report["identities"]["window_manifest_sha256"]:
        raise ValueError("selected manifest hash mismatch")
    reference = k5b1b.label_kinds(tw.labels(load_window(output / "windows.json", frames), frames))
    if [list(x) for x in reference] != report["reference_labels"]:
        raise ValueError("reference labels differ from selected manifest")
    def check_judges(judges: dict[str, Any]) -> None:
        for name in ("wasb", "balltrack"):
            item = judges[name]
            rows = item["frames"]
            if [row["frame"] for row in rows] != list(range(frames)):
                raise ValueError("invalid per-frame judge output")
            labels = [(row["label_kind"], row["label_x"], row["label_y"]) for row in rows]
            if labels != reference:
                raise ValueError("judge labels differ from manifest")
            preds = [(row["detected"], row["x"] if row["detected"] else 0.0,
                      row["y"] if row["detected"] else 0.0) for row in rows]
            recalculated = frame_results(preds, labels)
            if recalculated["counts"] != item["counts"] or recalculated["rates"] != item["rates"]:
                raise ValueError("judge counts differ from rows")
    check_judges(report["source_judges"])
    source_recall = report["source_judges"]["wasb"]["rates"]["detection_rate"]
    if source_recall is None or source_recall < 0.8:
        raise ValueError("WASB source qualification missing")
    fps = Fraction(report["fps"])
    width, height = report["size"]
    for point in report["points"]:
        kind, q = point["codec"], point["point"]
        directory = output / f"{kind}-{q}"
        native = directory / ("stream.psdc" if kind == "dcvc" else "stream.ivf")
        packet = directory / "ball.psb"
        if ledger(native, packet, frames / float(fps)) != point["streak"]["ledger"]:
            raise ValueError("persisted ledger mismatch")
        if sha(native) != point["baseline"]["sha256"] or native.stat().st_size != point["baseline"]["bytes"]:
            raise ValueError("baseline native identity mismatch")
        track, _ = packet_read(packet, width, height, fps, frames)
        if any(p is not None for p in track) != point["streak"]["ball_on"] or not point["streak"]["roundtrip"]:
            raise ValueError("BALL status mismatch")
        if point["native_record"]["receiver_decoded_sha256"] != point["native_record"]["encoder_decoded_sha256"]:
            raise ValueError("native receiver mismatch")
        baseline_psnr = point["baseline"]["frame_psnr"]
        streak_psnr = point["streak"]["frame_psnr"]
        if not isinstance(baseline_psnr, (float, int)) or not isinstance(streak_psnr, (float, int)) or streak_psnr < baseline_psnr:
            raise ValueError("frame-PSNR guard mismatch")
        if point["streak"]["guard_off"] and point["streak"]["ball_on"]:
            raise ValueError("guard-off packet contains a ball")
        for variant in ("baseline", "streak"):
            check_judges(point[variant]["judges"])
    return {"passed": True, "checks": {"persisted_component_records": True},
            "points": len(POINTS), "frames": frames,
            "scope": "persisted component records; not independently replayed model inference"}



def cost_projection(stage_dir: Path, now: float) -> dict[str, float]:
    """Conservative 48-frame promotion gate, using the fleet's monotonic clock."""
    execution = json.loads((stage_dir / "execution.json").read_text())
    dispatch = json.loads((stage_dir / "dispatch.json").read_text())
    seconds = execution["seconds"]
    clock = dispatch["campaign_clock"]
    started, phase_started, end = clock["started"], clock["phase_started"], clock["budget_end"]
    values = (seconds, started, phase_started, end, now)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("missing or invalid campaign timing")
    if execution.get("exit_code") != 0 or seconds <= 0 or not (0 < started <= phase_started <= now < end):
        raise ValueError("inconsistent campaign timing")
    estimated_work = seconds * 3 * 1.25
    # Leave a 60-second lifecycle reserve within the 420-second bound, and a
    # further 60 seconds of the remaining campaign for validation/publication.
    projected = estimated_work + 60.0
    remaining = end - now
    if projected > 420.0 or projected > remaining - 60.0:
        raise ValueError("48-frame cost projection exceeds full-stage or campaign ceiling")
    return {"smoke_command_seconds": seconds, "estimated_work_seconds": estimated_work,
            "projected_with_reserve_seconds": projected, "remaining_campaign_seconds": remaining,
            "campaign_reserve_seconds": 60.0}


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--validate-only":
        target = Path(os.environ["PS_VALIDATION_PATH"])
        try:
            # The fleet publishes and releases local scratch before invoking the validator.
            archive = Path(os.environ["PS_STAGE_DIR"]) / "published.tar"
            with tempfile.TemporaryDirectory(prefix="k5c-validator-", dir=os.environ["TMPDIR"]) as folder:
                with tarfile.open(archive, "r:") as bundle:
                    bundle.extractall(folder, filter="data")
                result = validate(Path(folder) / "publish")
            if os.environ.get("PS_STAGE") == "smoke":
                if result["frames"] != 16:
                    raise ValueError("smoke output must contain 16 frames")
                result["cost_projection"] = cost_projection(Path(os.environ["PS_STAGE_DIR"]), time.monotonic())
                result["checks"]["full_stage_cost_projection"] = True
            elif os.environ.get("PS_STAGE") == "full" and result["frames"] != 48:
                raise ValueError("full pilot output must contain 48 frames")
        except Exception as exc:
            target.write_text(json.dumps({"passed": False, "error": str(exc)}) + "\n")
            raise
        target.write_text(json.dumps(result) + "\n")
        return
    if len(sys.argv) == 3 and sys.argv[1] == "validate":
        print(json.dumps(validate(Path(sys.argv[2])), indent=2))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("windows", "source", "image-ckpt", "image-sha256", "video-ckpt", "video-sha256",
                 "balltrack", "balltrack-code", "wasb-code", "wasb-weights", "wasb-config", "exposure-receipt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output")
    parser.add_argument("--work-dir")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--encoder", default="SvtAv1EncApp")
    parser.add_argument("--decoder", default="dav1d")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
