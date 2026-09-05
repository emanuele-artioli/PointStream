"""Gate 2 tools: SVT-AV1 and VVC anchor floor probes and execution.

Brief specifications:
- Resolve installed tools afresh.
- Primary presets are the slowest valid presets actually encoded by a new codec-floor probe:
    AV1 expected: SVT-AV1 preset 0.
    VVC: actual encoder/decoder and slowest valid configuration (vvencapp --preset slower).
- Native size (3840x2160), 24 fps, yuv420p.
- For each codec/duration/rate:
    - continuous: concatenate two ordered scenes across boundary
    - segmented: encode each scene independently and sum bytes and codec time
- Probe legal QP endpoints and existing sparse walk, coarsest first.
- Every output must decode to exact frame count, size, and color format.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from experiments.tier.low_rate_measure import reference_request, timed_roundtrip


@dataclass(frozen=True)
class AnchorToolSpec:
    codec: str
    encoder_name: str
    executable: str
    slowest_preset: str
    qp_range: tuple[int, int]
    default_qps: tuple[int, ...]


AV1_SPEC = AnchorToolSpec(
    codec="av1",
    encoder_name="libsvtav1",
    executable="SvtAv1EncApp",
    slowest_preset="0",
    qp_range=(0, 63),
    default_qps=(63, 55, 47, 39, 31),
)

VVC_SPEC = AnchorToolSpec(
    codec="vvc",
    encoder_name="libvvenc",
    executable="/opt/local/bin/vvencapp",
    slowest_preset="slower",
    qp_range=(0, 63),
    default_qps=(63, 55, 47, 39, 31),
)


def resolve_tool_specs() -> dict[str, dict[str, Any]]:
    """Probe and record installed binaries, versions, and verified slowest presets."""
    tools: dict[str, dict[str, Any]] = {}

    # 1. SVT-AV1
    svt_bin = shutil.which("SvtAv1EncApp") or "/opt/local/bin/SvtAv1EncApp"
    svt_available = os.path.isfile(svt_bin) and os.access(svt_bin, os.X_OK)
    svt_version = "unknown"
    if svt_available:
        try:
            out = subprocess.check_output(
                [svt_bin, "--version"], stderr=subprocess.STDOUT, text=True
            )
            svt_version = out.strip().splitlines()[0]
        except Exception:
            pass

    tools["av1"] = {
        "codec": "av1",
        "encoder": "svt-av1",
        "binary": svt_bin,
        "available": svt_available,
        "version": svt_version,
        "slowest_preset": "0",
        "fps": 24,
        "pix_fmt": "yuv420p",
    }

    # 2. VVC
    vvc_bin = shutil.which("vvencapp") or "/opt/local/bin/vvencapp"
    vvc_available = os.path.isfile(vvc_bin) and os.access(vvc_bin, os.X_OK)
    vvc_version = "unknown"
    if vvc_available:
        try:
            out = subprocess.check_output(
                [vvc_bin, "--version"], stderr=subprocess.STDOUT, text=True
            )
            vvc_version = out.strip().splitlines()[0]
        except Exception:
            pass

    tools["vvc"] = {
        "codec": "vvc",
        "encoder": "vvencapp",
        "binary": vvc_bin,
        "available": vvc_available,
        "version": vvc_version,
        "slowest_preset": "slower",
        "fps": 24,
        "pix_fmt": "yuv420p",
    }

    return tools


def probe_codec_floor(
    codec: str,
    *,
    test_frame_shape: tuple[int, int, int, int] = (2, 64, 64, 3),
) -> dict[str, Any]:
    """Encode and decode a minimal probe clip to verify the slowest preset is valid."""
    tools = resolve_tool_specs()
    spec = tools.get(codec)
    if spec is None or not spec.get("available", False):
        return {"codec": codec, "supported": False, "reason": "binary not found"}

    frames = np.zeros(test_frame_shape, dtype=np.uint8)
    frames[..., 0] = np.arange(test_frame_shape[0], dtype=np.uint8)[:, None, None]
    request = reference_request(codec, 63, str(spec["slowest_preset"]))
    trip = timed_roundtrip(frames, request=request, fps=24.0)
    if trip.size_bytes <= 0 or trip.frames.shape != frames.shape:
        return {
            "codec": codec,
            "supported": False,
            "reason": f"invalid probe output: {trip.size_bytes} bytes, {trip.frames.shape}",
        }
    binary = Path(str(trip.tool_path)).resolve()
    return {
        "codec": codec,
        "supported": True,
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "version": trip.tool_version,
        "slowest_preset": trip.preset,
        "probe_qp": trip.qp,
        "probe_bytes": trip.size_bytes,
        "decoded_shape": list(trip.frames.shape),
        "verified": True,
    }


def write_tool_identity(destination: Path) -> dict[str, Any]:
    """Write tool resolution record to disk."""
    destination.mkdir(parents=True, exist_ok=True)
    specs = {codec: probe_codec_floor(codec) for codec in ("av1", "vvc")}
    failed = [codec for codec, row in specs.items() if not row.get("supported")]
    if failed:
        raise SystemExit(f"Gate A codec-floor probe failed for {failed}")
    record = {
        "tools": specs,
        "notes": [
            f"AV1 slowest preset driven: {specs['av1']['slowest_preset']}",
            f"VVC slowest preset driven: {specs['vvc']['slowest_preset']}",
            "Anchors run continuous and segmented on identical source frames",
        ],
    }
    (destination / "tool-identity.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
