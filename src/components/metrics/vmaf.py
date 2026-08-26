"""VMAF. The network is injectable so tests never invent a score."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.components.metrics.frames import paired, to_clip
from src.contracts.metrics import VMAF

VmafModel = Callable[[np.ndarray, np.ndarray], float]


class VmafMetric:
    """Headline video-quality score in ``[0, 100]``.

    In-memory frames are scored by an injected ``model`` (the test path) or by
    writing the clip and calling ffmpeg's ``libvmaf``. There is no arithmetic
    stand-in: a missing model and a missing libvmaf raise rather than return a
    PSNR-shaped number labelled VMAF.

    **Usable range on this host's tier content (3840×2160, measured BP23).**
    Quote a score beside these anchors from ``outputs/bp23-tier/metric-calibration.json``:

    | pair | VMAF |
    |---|---|
    | identical | **97.54** (ceiling — not 100) |
    | mild blur | 84.96 |
    | severe blur | **0.00** (floor) |
    | unrelated clip | **0.00** (floor) |

    The ceiling is below 100 even on identical frames. Severe blur and an unrelated
    broadcast clip both hit the floor: two arms near 0 are not ranked. Use PSNR or
    LPIPS to separate anything in that region.
    """

    name = VMAF.name

    def __init__(self, model: VmafModel | None = None) -> None:
        self._model = model

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        ref, pred = paired(reference, predicted)
        if self._model is not None:
            return float(self._model(ref, pred))
        return _libvmaf_on_clips(ref, pred)


def _libvmaf_on_clips(reference: np.ndarray, predicted: np.ndarray) -> float:
    ffmpeg = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "VMAF needs a model callable or ffmpeg with libvmaf. "
            "Neither is available (set FFMPEG_BIN or pass model=...)."
        )
    with tempfile.TemporaryDirectory(prefix="pointstream-vmaf-") as tmp:
        root = Path(tmp)
        ref_dir = root / "ref"
        pred_dir = root / "pred"
        log_path = root / "vmaf.json"
        _write_png_clip(ref_dir, reference)
        _write_png_clip(pred_dir, predicted)
        # ffmpeg's libvmaf takes [distorted][reference], in that order. Input 0
        # here is the reference and input 1 the prediction, so the labels are
        # crossed deliberately. Passing them straight through scored a blurred
        # clip at 100.0 against 97.4 for an identical one — see the calibration
        # test in tests/components/test_metrics_integration.py.
        filter_complex = (
            f"[1:v]format=yuv420p[dist];[0:v]format=yuv420p[ref];"
            f"[dist][ref]libvmaf=log_path={log_path}:log_fmt=json"
        )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "30",
            "-i",
            str(ref_dir / "frame_%06d.png"),
            "-framerate",
            "30",
            "-i",
            str(pred_dir / "frame_%06d.png"),
            "-filter_complex",
            filter_complex,
            "-f",
            "null",
            "-",
        ]
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode != 0 or not log_path.is_file():
            detail = (process.stderr or process.stdout or "libvmaf produced no log").strip()
            raise RuntimeError(f"ffmpeg libvmaf failed: {detail}")
        return _read_vmaf_mean(log_path)


def _write_png_clip(directory: Path, clip: np.ndarray) -> None:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    uint8 = np.clip(np.rint(to_clip(clip)), 0, 255).astype(np.uint8)
    for index, frame in enumerate(uint8):
        path = directory / f"frame_{index:06d}.png"
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"failed to write VMAF frame {path}")


def _read_vmaf_mean(log_path: Path) -> float:
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    pooled = payload.get("pooled_metrics", {})
    if isinstance(pooled, dict):
        vmaf = pooled.get("vmaf", {})
        if isinstance(vmaf, dict):
            value = vmaf.get("mean", vmaf.get("value"))
            if value is not None:
                return float(value)
    aggregate = payload.get("aggregate", {})
    if isinstance(aggregate, dict) and aggregate.get("VMAF_score") is not None:
        return float(aggregate["VMAF_score"])
    raise RuntimeError("ffmpeg libvmaf returned no score")
