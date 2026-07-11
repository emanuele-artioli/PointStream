from __future__ import annotations
from pathlib import Path
from typing import Any

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

from src.encoder.video_io import decode_video_to_tensor
from src.shared.fvd import compute_fvd_from_frames


def _safe_file_size(path_like: str | Path | None) -> int | None:
    if path_like is None:
        return None
    candidate = Path(str(path_like))
    if not candidate.exists() or not candidate.is_file():
        if candidate.exists() and candidate.is_dir():
            return int(sum(path.stat().st_size for path in candidate.rglob("*") if path.is_file()))
        return None
    return int(candidate.stat().st_size)


def _compute_psnr(reference_video: Path, predicted_video: Path, max_frames: int | None = None) -> dict[str, Any]:
    """Compute PSNR via the system ffmpeg's `psnr` filter.

    cv2.VideoCapture uses opencv-python's bundled ffmpeg libs, which are built
    without an AV1 decoder — decoded chunks encoded with libsvtav1 open
    (isOpened() True) but yield zero frames from read(), so any cv2-based
    pairing silently finds "no valid frame pairs". SSIM/VMAF already sidestep
    this by shelling out to the system ffmpeg binary; PSNR now does the same.
    """
    if not reference_video.exists() or not reference_video.is_file():
        return {"psnr_mean": None, "psnr_std": None, "psnr_num_frames": 0, "note": "missing reference video"}
    if not predicted_video.exists():
        return {"psnr_mean": None, "psnr_std": None, "psnr_num_frames": 0, "note": "missing predicted artifact"}

    ffmpeg_bin = _resolve_binary_path("FFMPEG_BIN", "ffmpeg")
    stats_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    stats_path = Path(stats_file.name)
    stats_file.close()

    reference_dims = _get_video_dimensions(reference_video)
    predicted_dims = _get_video_dimensions(predicted_video)
    if reference_dims is not None and predicted_dims is not None and reference_dims != predicted_dims:
        width, height = reference_dims
        filter_complex = (
            f"[1:v]scale={width}:{height}[psnr_pred_scaled];"
            f"[0:v][psnr_pred_scaled]psnr=stats_file=" + str(stats_path)
        )
    else:
        filter_complex = "[0:v][1:v]psnr=stats_file=" + str(stats_path)

    ffmpeg_cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
    ]

    fps = "30"
    if not reference_video.is_dir():
        fps = _get_video_framerate(reference_video)
    elif not predicted_video.is_dir():
        fps = _get_video_framerate(predicted_video)

    if reference_video.is_dir():
        ffmpeg_cmd.extend(["-framerate", fps, "-i", str(reference_video / "frame_%06d.png")])
    else:
        ffmpeg_cmd.extend(["-i", str(reference_video)])

    if predicted_video.is_dir():
        ffmpeg_cmd.extend(["-framerate", fps, "-i", str(predicted_video / "frame_%06d.png")])
    else:
        ffmpeg_cmd.extend(["-i", str(predicted_video)])
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
                "psnr_mean": None,
                "psnr_std": None,
                "psnr_num_frames": 0,
                "note": f"ffmpeg psnr failed: {stderr_text or 'unknown error'}",
            }

        if not stats_path.exists():
            return {
                "psnr_mean": None,
                "psnr_std": None,
                "psnr_num_frames": 0,
                "note": "ffmpeg psnr did not emit stats",
            }

        values: list[float] = []
        for line in stats_path.read_text(encoding="utf-8", errors="replace").splitlines():
            for token in line.split():
                if token.startswith("psnr_avg:"):
                    raw = token.split(":", maxsplit=1)[1]
                    try:
                        values.append(float("inf") if raw.lower() == "inf" else float(raw))
                    except ValueError:
                        pass
                    break

        if not values:
            return {
                "psnr_mean": None,
                "psnr_std": None,
                "psnr_num_frames": 0,
                "psnr_infinite_frames": 0,
                "note": "no valid frame pairs",
            }

        arr = np.array(values, dtype=float)
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
            "psnr_num_frames": int(arr.size),
            "psnr_infinite_frames": infinite_count,
            "note": None,
        }
    finally:
        if stats_path.exists():
            stats_path.unlink(missing_ok=True)


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


def _get_video_framerate(video_path: Path) -> str:
    if video_path.is_dir():
        return "30"
    try:
        ffprobe_bin = _resolve_binary_path("FFPROBE_BIN", "ffprobe")
    except FileNotFoundError:
        return "30"
    
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        fps = res.stdout.strip()
        if not fps or fps == "0/0":
            return "30"
        return fps
    except Exception:
        return "30"


def _get_video_dimensions(video_path: Path) -> tuple[int, int] | None:
    probe_target = video_path
    if video_path.is_dir():
        frame_paths = sorted(
            candidate
            for candidate in video_path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not frame_paths:
            return None
        probe_target = frame_paths[0]

    try:
        ffprobe_bin = _resolve_binary_path("FFPROBE_BIN", "ffprobe")
    except FileNotFoundError:
        return None

    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(probe_target),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        width_str, _, height_str = res.stdout.strip().partition(",")
        return int(width_str), int(height_str)
    except Exception:
        return None


def _compute_ssim_ffmpeg(reference_video: Path, predicted_video: Path, max_frames: int | None = None) -> dict[str, Any]:
    if not reference_video.exists() or not reference_video.is_file():
        return {"ssim_mean": None, "ssim_std": None, "ssim_num_frames": 0, "note": "missing reference video"}
    if not predicted_video.exists():
        return {"ssim_mean": None, "ssim_std": None, "ssim_num_frames": 0, "note": "missing predicted artifact"}

    ffmpeg_bin = _resolve_binary_path("FFMPEG_BIN", "ffmpeg")
    stats_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    stats_path = Path(stats_file.name)
    stats_file.close()

    reference_dims = _get_video_dimensions(reference_video)
    predicted_dims = _get_video_dimensions(predicted_video)
    if reference_dims is not None and predicted_dims is not None and reference_dims != predicted_dims:
        width, height = reference_dims
        filter_complex = (
            f"[1:v]scale={width}:{height}[ssim_pred_scaled];"
            f"[0:v][ssim_pred_scaled]ssim=stats_file=" + str(stats_path)
        )
    else:
        filter_complex = "[0:v][1:v]ssim=stats_file=" + str(stats_path)
    
    ffmpeg_cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning"
    ]
    
    fps = "30"
    if not reference_video.is_dir():
        fps = _get_video_framerate(reference_video)
    elif not predicted_video.is_dir():
        fps = _get_video_framerate(predicted_video)
        
    if reference_video.is_dir():
        ffmpeg_cmd.extend(["-framerate", fps, "-i", str(reference_video / "frame_%06d.png")])
    else:
        ffmpeg_cmd.extend(["-i", str(reference_video)])
        
    if predicted_video.is_dir():
        ffmpeg_cmd.extend(["-framerate", fps, "-i", str(predicted_video / "frame_%06d.png")])
    else:
        ffmpeg_cmd.extend(["-i", str(predicted_video)])
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

    reference_dims = _get_video_dimensions(reference_video)
    predicted_dims = _get_video_dimensions(predicted_video)
    if reference_dims is not None and predicted_dims is not None and reference_dims != predicted_dims:
        width, height = reference_dims
        filter_complex = (
            f"[1:v]scale={width}:{height}[vmaf_pred_scaled];"
            f"[0:v][vmaf_pred_scaled]libvmaf=log_path=" + str(log_path) + ":log_fmt=json"
        )
    else:
        filter_complex = "[0:v][1:v]libvmaf=log_path=" + str(log_path) + ":log_fmt=json"
    
    ffmpeg_cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning"
    ]
    
    fps = "30"
    if not reference_video.is_dir():
        fps = _get_video_framerate(reference_video)
    elif not predicted_video.is_dir():
        fps = _get_video_framerate(predicted_video)

    if reference_video.is_dir():
        ffmpeg_cmd.extend(["-framerate", fps, "-i", str(reference_video / "frame_%06d.png")])
    else:
        ffmpeg_cmd.extend(["-i", str(reference_video)])
        
    if predicted_video.is_dir():
        ffmpeg_cmd.extend(["-framerate", fps, "-i", str(predicted_video / "frame_%06d.png")])
    else:
        ffmpeg_cmd.extend(["-i", str(predicted_video)])
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


def _compute_fvd(reference_video: Path, predicted_video: Path, max_frames: int | None = None) -> dict[str, Any]:
    """Compute Frechet Video Distance via a pretrained I3D (Kinetics-400) backbone.

    Unlike PSNR/SSIM/VMAF this does not shell out to ffmpeg filters — it
    decodes both videos to frame tensors (reusing `video_io.decode_video_to_tensor`,
    which already sidesteps the AV1-decode gap in opencv-python's bundled
    ffmpeg libs), builds I3D clips, and computes the Frechet distance between
    per-video clip-feature distributions (see `src.shared.fvd`).
    """
    if not reference_video.exists() or not reference_video.is_file():
        return {"fvd": None, "fvd_num_clips_reference": 0, "fvd_num_clips_predicted": 0, "note": "missing reference video"}
    if not predicted_video.exists():
        return {"fvd": None, "fvd_num_clips_reference": 0, "fvd_num_clips_predicted": 0, "note": "missing predicted artifact"}

    try:
        reference_decoded = decode_video_to_tensor(reference_video)
        predicted_decoded = decode_video_to_tensor(predicted_video)

        reference_frames = reference_decoded.tensor
        predicted_frames = predicted_decoded.tensor
        if max_frames is not None:
            reference_frames = reference_frames[: int(max_frames)]
            predicted_frames = predicted_frames[: int(max_frames)]

        return compute_fvd_from_frames(reference_frames, predicted_frames)
    except Exception as exc:  # noqa: BLE001 - surface any failure as a metric note, matching PSNR/SSIM/VMAF
        return {
            "fvd": None,
            "fvd_num_clips_reference": 0,
            "fvd_num_clips_predicted": 0,
            "note": f"fvd computation failed: {exc}",
        }


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

    allowed = {"psnr", "ssim", "vmaf", "fvd"}
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
    evaluation_block = summary.get("evaluation", {})
    sizes_block = evaluation_block.get("sizes_bytes", {})
    
    transport_total_size_bytes = sizes_block.get("transport_total")
    source_size_bytes = sizes_block.get("source")

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
        "sizes_bytes": {
            "decoded_video": _safe_file_size(decoded_path) if decoded_path is not None else None,
            "transport_savings_percent": transport_savings_percent,
        },
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
    if "fvd" in normalized_metrics:
        evaluation.update(
            _compute_fvd(
                reference_video=reference_video,
                predicted_video=predicted_video,
                max_frames=max_frames,
            )
        )

    return evaluation
