from __future__ import annotations
from pathlib import Path
from typing import Any

import json
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


def _safe_file_size(path_like: str | Path | None) -> int | None:
    if path_like is None:
        return None
    candidate = Path(str(path_like))
    if not candidate.exists() or not candidate.is_file():
        if candidate.exists() and candidate.is_dir():
            return int(sum(path.stat().st_size for path in candidate.rglob("*") if path.is_file()))
        return None
    return int(candidate.stat().st_size)


def _load_frame_sequence(path: Path, max_frames: int | None = None) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    if path.is_dir():
        frame_paths = sorted(
            [candidate for candidate in path.iterdir() if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg"}],
            key=lambda candidate: candidate.name,
        )
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                continue
            frames.append(frame)
        return frames

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return []
    try:
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _stream_frame_pairs(
    reference_video: Path,
    predicted_video: Path,
    max_frames: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    cap_ref = cv2.VideoCapture(str(reference_video))
    cap_pred = cv2.VideoCapture(str(predicted_video))
    if not cap_ref.isOpened() or not cap_pred.isOpened():
        cap_ref.release()
        cap_pred.release()
        return []

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        while True:
            if max_frames is not None and len(pairs) >= max_frames:
                break
            ok_ref, ref_frame = cap_ref.read()
            ok_pred, pred_frame = cap_pred.read()
            if not ok_ref or not ok_pred:
                break
            pairs.append((ref_frame, pred_frame))
    finally:
        cap_ref.release()
        cap_pred.release()
    return pairs


def _read_frames_with_timestamps(
    path: Path,
    max_frames: int | None = None,
) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return []

    frames: list[tuple[float, np.ndarray]] = []
    try:
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            frames.append((timestamp_ms, frame))
    finally:
        cap.release()
    return frames


def _match_nearest_timestamp_pairs(
    reference_frames: list[tuple[float, np.ndarray]],
    predicted_frames: list[tuple[float, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not reference_frames or not predicted_frames:
        return []

    pred_times = [ts for ts, _ in predicted_frames]
    pred_images = [frame for _, frame in predicted_frames]
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for ref_time, ref_frame in reference_frames:
        nearest_idx = min(range(len(pred_times)), key=lambda idx: abs(pred_times[idx] - ref_time))
        pairs.append((ref_frame, pred_images[nearest_idx]))
    return pairs


def _compute_psnr(reference_video: Path, predicted_video: Path, max_frames: int | None = None) -> dict[str, Any]:
    if not reference_video.exists() or not reference_video.is_file():
        return {"psnr_mean": None, "psnr_std": None, "psnr_num_frames": 0, "note": "missing reference video"}
    if not predicted_video.exists():
        return {"psnr_mean": None, "psnr_std": None, "psnr_num_frames": 0, "note": "missing predicted artifact"}

    pairs = _stream_frame_pairs(
        reference_video=reference_video,
        predicted_video=predicted_video,
        max_frames=max_frames,
    )
    if not pairs:
        reference_frames = _read_frames_with_timestamps(reference_video, max_frames=max_frames)
        predicted_frames = _read_frames_with_timestamps(predicted_video, max_frames=max_frames)
        pairs = _match_nearest_timestamp_pairs(reference_frames, predicted_frames)
    if not pairs:
        return {"psnr_mean": None, "psnr_std": None, "psnr_num_frames": 0, "note": "no valid frame pairs"}
    psnr_values: list[float] = []
    for (reference_frame, predicted_frame) in pairs:
        if reference_frame is None or predicted_frame is None:
            continue
        if reference_frame.shape[:2] != predicted_frame.shape[:2]:
            predicted_frame = cv2.resize(
                predicted_frame,
                (reference_frame.shape[1], reference_frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        psnr = cv2.PSNR(reference_frame, predicted_frame)
        if np.isfinite(psnr) or np.isinf(psnr):
            psnr_values.append(float(psnr))

    if not psnr_values:
        return {"psnr_mean": None, "psnr_std": None, "psnr_num_frames": 0, "psnr_infinite_frames": 0, "note": "no valid frame pairs"}

    arr = np.array(psnr_values, dtype=float)
    infinite_count = int(np.isinf(arr).sum())
    finite_vals = arr[np.isfinite(arr)]

    if finite_vals.size > 0:
        mean_val = float(np.mean(finite_vals))
        std_val = float(np.std(finite_vals))
    else:
        mean_val = None
        std_val = None

    return {
        "psnr_mean": mean_val,
        "psnr_std": std_val,
        "psnr_num_frames": int(len(psnr_values)),
        "psnr_infinite_frames": infinite_count,
        "note": None,
    }


def _resolve_binary_path(env_var: str, binary_name: str) -> str:
    explicit = os.environ.get(env_var)
    if explicit:
        return explicit

    resolved = shutil.which(binary_name)
    if resolved:
        return resolved

    raise FileNotFoundError(
        f"Required binary '{binary_name}' was not found in PATH. "
        f"Install FFmpeg tools or set {env_var} to the executable path."
    )


def _compute_ssim_ffmpeg(reference_video: Path, predicted_video: Path, max_frames: int | None = None) -> dict[str, Any]:
    if not reference_video.exists() or not reference_video.is_file():
        return {"ssim_mean": None, "ssim_std": None, "ssim_num_frames": 0, "note": "missing reference video"}
    if not predicted_video.exists():
        return {"ssim_mean": None, "ssim_std": None, "ssim_num_frames": 0, "note": "missing predicted artifact"}

    ffmpeg_bin = _resolve_binary_path("FFMPEG_BIN", "ffmpeg")
    stats_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    stats_path = Path(stats_file.name)
    stats_file.close()

    filter_complex = f"[0:v][1:v]ssim=stats_file={stats_path}"
    ffmpeg_cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(reference_video),
        "-i",
        str(predicted_video),
    ]
    if max_frames is not None:
        ffmpeg_cmd.extend(["-frames:v", str(int(max_frames))])
    ffmpeg_cmd.extend(["-filter_complex", filter_complex, "-f", "null", "-"])

    try:
        process = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if process.returncode != 0:
            stderr_text = (process.stderr or "").strip()
            return {
                "ssim_mean": None,
                "ssim_std": None,
                "ssim_num_frames": 0,
                "note": f"ffmpeg ssim failed: {stderr_text or 'unknown error'}",
            }

        if not stats_path.exists():
            return {
                "ssim_mean": None,
                "ssim_std": None,
                "ssim_num_frames": 0,
                "note": "ffmpeg ssim did not emit stats",
            }

        values: list[float] = []
        for line in stats_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "All:" not in line:
                continue
            for token in line.split():
                if token.startswith("All:"):
                    try:
                        values.append(float(token.split(":", maxsplit=1)[1]))
                    except ValueError:
                        pass
                    break

        if not values:
            return {
                "ssim_mean": None,
                "ssim_std": None,
                "ssim_num_frames": 0,
                "note": "ffmpeg ssim returned no values",
            }

        arr = np.array(values, dtype=float)
        return {
            "ssim_mean": float(np.mean(arr)),
            "ssim_std": float(np.std(arr)),
            "ssim_num_frames": int(arr.size),
            "note": None,
        }
    finally:
        if stats_path.exists():
            stats_path.unlink(missing_ok=True)


def _compute_vmaf_ffmpeg(reference_video: Path, predicted_video: Path, max_frames: int | None = None) -> dict[str, Any]:
    if not reference_video.exists() or not reference_video.is_file():
        return {"vmaf_mean": None, "vmaf_num_frames": 0, "note": "missing reference video"}
    if not predicted_video.exists():
        return {"vmaf_mean": None, "vmaf_num_frames": 0, "note": "missing predicted artifact"}

    ffmpeg_bin = _resolve_binary_path("FFMPEG_BIN", "ffmpeg")
    log_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    log_path = Path(log_file.name)
    log_file.close()

    filter_complex = f"[0:v][1:v]libvmaf=log_path={log_path}:log_fmt=json"
    ffmpeg_cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(reference_video),
        "-i",
        str(predicted_video),
    ]
    if max_frames is not None:
        ffmpeg_cmd.extend(["-frames:v", str(int(max_frames))])
    ffmpeg_cmd.extend(["-filter_complex", filter_complex, "-f", "null", "-"])

    try:
        process = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if process.returncode != 0:
            stderr_text = (process.stderr or "").strip()
            return {
                "vmaf_mean": None,
                "vmaf_num_frames": 0,
                "note": f"ffmpeg vmaf failed: {stderr_text or 'unknown error'}",
            }

        if not log_path.exists():
            return {
                "vmaf_mean": None,
                "vmaf_num_frames": 0,
                "note": "ffmpeg vmaf did not emit log",
            }

        payload = json.loads(log_path.read_text(encoding="utf-8", errors="replace"))
        vmaf_mean = None
        pooled = payload.get("pooled_metrics", {})
        if isinstance(pooled, dict):
            vmaf = pooled.get("vmaf", {})
            if isinstance(vmaf, dict):
                if "mean" in vmaf:
                    vmaf_mean = vmaf.get("mean")
                elif "value" in vmaf:
                    vmaf_mean = vmaf.get("value")
        if vmaf_mean is None and isinstance(payload.get("aggregate"), dict):
            aggregate = payload.get("aggregate", {})
            vmaf_mean = aggregate.get("VMAF_score")

        frames = payload.get("frames")
        num_frames = int(len(frames)) if isinstance(frames, list) else 0

        return {
            "vmaf_mean": float(vmaf_mean) if vmaf_mean is not None else None,
            "vmaf_num_frames": num_frames,
            "note": None if vmaf_mean is not None else "ffmpeg vmaf returned no score",
        }
    finally:
        if log_path.exists():
            log_path.unlink(missing_ok=True)


def _normalize_evaluation_metrics(metrics: Any | None) -> list[str]:
    if metrics is None:
        return ["psnr"]
    if isinstance(metrics, str):
        normalized = metrics.strip().lower()
        if not normalized or normalized == "none":
            return []
        items = [item.strip().lower() for item in normalized.split(",")]
    elif isinstance(metrics, (list, tuple, set)):
        items = [str(item).strip().lower() for item in metrics]
    else:
        raise ValueError("evaluation metrics must be a string or list")

    items = [item for item in items if item]
    if not items:
        return []
    if "none" in items:
        if len(items) > 1:
            raise ValueError("evaluation metrics cannot include 'none' with other values")
        return []

    allowed = {"psnr", "ssim", "vmaf"}
    invalid = [item for item in items if item not in allowed]
    if invalid:
        raise ValueError(f"unsupported evaluation metrics: {', '.join(sorted(set(invalid)))}")

    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def evaluate_run_summary(
    summary: dict[str, Any],
    experiment_dir: str | Path,
    max_frames: int | None = None,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Compute evaluation metrics (PSNR, sizes) for a completed pipeline run.
    
    Note: Only includes metrics that are computed by this function. Fields that are
    already present in the run_summary (pipeline_total_sec, encode_chunk_sec, etc.)
    are NOT duplicated here to avoid redundancy in the JSON output.
    """
    source_uri = summary.get("source_uri")
    decoded_uri = summary.get("decoded_uri")
    transport_total_size_bytes = summary.get("transport_total_size_bytes")
    source_size_bytes = summary.get("source_size_bytes")

    source_path = Path(str(source_uri)).expanduser() if source_uri is not None else None
    decoded_path = Path(str(decoded_uri)).expanduser() if decoded_uri is not None else None

    normalized_metrics = _normalize_evaluation_metrics(metrics)
    reference_video = source_path if source_path is not None else Path(experiment_dir) / "missing_source.mp4"
    predicted_video = decoded_path if decoded_path is not None else Path(experiment_dir) / "missing_decoded.mp4"

    # Only include metrics computed by evaluation; omit copies of run_summary fields
    transport_savings_percent: float | None = None
    if isinstance(source_size_bytes, int) and source_size_bytes > 0 and isinstance(transport_total_size_bytes, int):
        transport_savings_percent = (1.0 - float(transport_total_size_bytes) / float(source_size_bytes)) * 100.0

    evaluation = {
        "decoded_video_size_bytes": _safe_file_size(decoded_path) if decoded_path is not None else None,
        "transport_savings_percent": transport_savings_percent,
    }

    if "psnr" in normalized_metrics:
        evaluation.update(
            _compute_psnr(
                reference_video=reference_video,
                predicted_video=predicted_video,
                max_frames=max_frames,
            )
        )
    if "ssim" in normalized_metrics:
        evaluation.update(
            _compute_ssim_ffmpeg(
                reference_video=reference_video,
                predicted_video=predicted_video,
                max_frames=max_frames,
            )
        )
    if "vmaf" in normalized_metrics:
        evaluation.update(
            _compute_vmaf_ffmpeg(
                reference_video=reference_video,
                predicted_video=predicted_video,
                max_frames=max_frames,
            )
        )
    return evaluation
