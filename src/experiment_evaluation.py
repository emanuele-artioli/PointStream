from __future__ import annotations
from pathlib import Path
from typing import Any

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


def evaluate_run_summary(summary: dict[str, Any], experiment_dir: str | Path, max_frames: int | None = None) -> dict[str, Any]:
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

    psnr = _compute_psnr(
        reference_video=source_path if source_path is not None else Path(experiment_dir) / "missing_source.mp4",
        predicted_video=decoded_path if decoded_path is not None else Path(experiment_dir) / "missing_decoded.mp4",
        max_frames=max_frames,
    )

    # Only include metrics computed by evaluation; omit copies of run_summary fields
    transport_savings_percent: float | None = None
    if isinstance(source_size_bytes, int) and source_size_bytes > 0 and isinstance(transport_total_size_bytes, int):
        transport_savings_percent = (1.0 - float(transport_total_size_bytes) / float(source_size_bytes)) * 100.0

    evaluation = {
        "decoded_video_size_bytes": _safe_file_size(decoded_path) if decoded_path is not None else None,
        "transport_savings_percent": transport_savings_percent,
    }

    evaluation.update(psnr)
    return evaluation