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
        return None
    return int(candidate.stat().st_size)


def _compute_psnr(reference_video: Path, predicted_video: Path, max_frames: int | None = None) -> dict[str, Any]:
    if not reference_video.exists() or not reference_video.is_file():
        return {"psnr_mean": None, "psnr_std": None, "psnr_num_frames": 0, "note": "missing reference video"}
    if not predicted_video.exists() or not predicted_video.is_file():
        return {"psnr_mean": None, "psnr_std": None, "psnr_num_frames": 0, "note": "missing predicted video"}

    # Unified frame pairing helper: index-first streaming, then timestamp-nearest fallback.
    def _get_frame_pairs(ref_path: Path, pred_path: Path, max_frames: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        # Try index-wise streaming pairing first (minimal memory).
        ref_cap = cv2.VideoCapture(str(ref_path))
        pred_cap = cv2.VideoCapture(str(pred_path))
        if not ref_cap.isOpened() or not pred_cap.isOpened():
            ref_cap.release()
            pred_cap.release()
            return []

        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        idx = 0
        try:
            while True:
                if max_frames is not None and idx >= max_frames:
                    break
                ref_ok, ref_frame = ref_cap.read()
                pred_ok, pred_frame = pred_cap.read()
                if not ref_ok or not pred_ok:
                    break

                pairs.append((ref_frame, pred_frame))
                idx += 1
        finally:
            ref_cap.release()
            pred_cap.release()

        if pairs:
            return pairs

        # Streaming pairing produced no pairs; fall back to timestamp-based nearest matching.
        ref_cap = cv2.VideoCapture(str(ref_path))
        pred_cap = cv2.VideoCapture(str(pred_path))
        if not ref_cap.isOpened() or not pred_cap.isOpened():
            ref_cap.release()
            pred_cap.release()
            return []

        ref_frames: list[np.ndarray] = []
        ref_times: list[float] = []
        pred_frames: list[np.ndarray] = []
        pred_times: list[float] = []
        try:
            while True:
                ok, frame = ref_cap.read()
                if not ok:
                    break
                t_ms = float(ref_cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                ref_frames.append(frame)
                ref_times.append(t_ms)

            while True:
                ok, frame = pred_cap.read()
                if not ok:
                    break
                t_ms = float(pred_cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                pred_frames.append(frame)
                pred_times.append(t_ms)
        finally:
            ref_cap.release()
            pred_cap.release()

        if not ref_frames or not pred_frames:
            return []

        # Build nearest-neighbor mapping from reference times to prediction frames.
        used_pred: set[int] = set()
        out_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for i, rt in enumerate(ref_times):
            diffs = [abs(rt - pt) for pt in pred_times]
            j = int(np.argmin(np.array(diffs, dtype=float)))
            if j in used_pred:
                order = np.argsort(np.array(diffs, dtype=float))
                found = False
                for cand in order:
                    if int(cand) not in used_pred:
                        j = int(cand)
                        found = True
                        break
                if not found:
                    j = int(order[0])
            used_pred.add(j)
            out_pairs.append((ref_frames[i], pred_frames[j]))
            if max_frames is not None and len(out_pairs) >= max_frames:
                break

        return out_pairs

    pairs = _get_frame_pairs(reference_video, predicted_video, max_frames=max_frames)
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
    resolved_experiment_dir = Path(experiment_dir).expanduser()
    source_uri = summary.get("source_uri")
    decoded_uri = summary.get("decoded_uri")
    transport_total_size_bytes = summary.get("transport_total_size_bytes")
    source_size_bytes = summary.get("source_size_bytes")

    source_path = Path(str(source_uri)).expanduser() if source_uri is not None else None
    decoded_path = Path(str(decoded_uri)).expanduser() if decoded_uri is not None else None

    psnr = _compute_psnr(
        reference_video=source_path if source_path is not None else resolved_experiment_dir / "missing_source.mp4",
        predicted_video=decoded_path if decoded_path is not None else resolved_experiment_dir / "missing_decoded.mp4",
        max_frames=max_frames,
    )

    evaluation = {
        "experiment_dir": str(resolved_experiment_dir),
        "source_uri": str(source_path) if source_path is not None else None,
        "decoded_uri": str(decoded_path) if decoded_path is not None else None,
        "reference_video_size_bytes": source_size_bytes,
        "decoded_video_size_bytes": _safe_file_size(decoded_path) if decoded_path is not None else None,
        "transport_total_size_bytes": transport_total_size_bytes,
        "pipeline_total_sec": summary.get("pipeline_total_sec"),
        "encode_chunk_sec": summary.get("encode_chunk_sec"),
        "transport_send_sec": summary.get("transport_send_sec"),
        "transport_receive_sec": summary.get("transport_receive_sec"),
        "decode_sec": summary.get("decode_sec"),
        "transport_savings_percent": None,
        "decoded_vs_reference_percent": None,
    }

    if isinstance(source_size_bytes, int) and source_size_bytes > 0:
        if isinstance(transport_total_size_bytes, int):
            evaluation["transport_savings_percent"] = (1.0 - float(transport_total_size_bytes) / float(source_size_bytes)) * 100.0
        decoded_size_bytes = evaluation["decoded_video_size_bytes"]
        if isinstance(decoded_size_bytes, int):
            evaluation["decoded_vs_reference_percent"] = (1.0 - float(decoded_size_bytes) / float(source_size_bytes)) * 100.0

    evaluation.update(psnr)
    return evaluation