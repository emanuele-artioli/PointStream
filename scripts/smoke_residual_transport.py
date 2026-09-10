"""Smoke test script demonstrating native residual transport fidelity end-to-end.

Produces /tmp/pointstream-worker-a/artifacts/residual_smoke.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import time
import sys

# Ensure worktree root is first on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Host rule: import sqlite3 before import torch
import sqlite3  # noqa: F401

import numpy as np

from src.components.codec import tools
from src.contracts.codecs import RateControl
from src.contracts.config import PointstreamConfig, ResidualConfig
from src.contracts.lattice import WHOLE_FRAME_RESIDUAL
from src.pipeline.reconstruction import bit_identical
from src.runner import lattice_config_from, run
from src.runner.client import reconstruct_serialized_client


def main() -> None:
    print("Running residual transport smoke test...")
    ffmpeg = tools.resolve_ffmpeg()
    print(f"Using ffmpeg: {ffmpeg.path} (version: {ffmpeg.version})")

    # 4 frames, 64x64, 3 channels
    t, h, w = 4, 64, 64
    x = np.linspace(30, 220, w, dtype=np.uint8)
    plane = np.tile(x, (h, 1))
    source = np.stack([np.stack([plane, plane, plane], axis=-1) for _ in range(t)])

    # Residual config: AVC CRF 23, gating and downscale disabled
    residual_cfg = ResidualConfig(
        codec="avc",
        rate_control=RateControl.CRF,
        rate=23,
        preset="ultrafast",
        block_size=1,
        block_threshold=0.0,
        background_downscale=1,
    )
    config = PointstreamConfig(
        lattice=lattice_config_from(WHOLE_FRAME_RESIDUAL),
        residual=residual_cfg,
    )

    t0 = time.perf_counter()
    result = run(config, [source])
    elapsed_total = time.perf_counter() - t0

    chunk = result.chunks[0]
    wire_request = chunk.bag.get("wire_request")
    assert isinstance(wire_request, bytes), "wire_request must be present in chunk bag as bytes"

    transmitted = chunk.bag.get("transmitted_residual")
    from src.pipeline.residual.codec import TransmittedResidual
    assert isinstance(transmitted, TransmittedResidual), "transmitted_residual must be present"

    # Fresh standalone client reconstruction from wire bytes
    client_frames = reconstruct_serialized_client(wire_request)
    assert isinstance(client_frames, np.ndarray), "client_frames must be ndarray"

    # Verification checks
    client_delivered_identical = bool(bit_identical(client_frames, result.frames))
    ledger_reconciled = bool(
        chunk.sizes.transport_total == len(wire_request)
        and chunk.sizes.residual == len(transmitted.bitstream)
    )
    psnr_delivered = float(result.delivered_quality.whole_frame())
    psnr_unaided = float(result.quality.whole_frame())

    smoke_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "host": {
            "node": platform.node(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "codec": {
            "name": transmitted.codec_name,
            "tool_path": ffmpeg.path,
            "tool_version": ffmpeg.version,
            "preset": transmitted.preset,
            "rate_control": "crf",
            "qp_rate": transmitted.qp,
            "mode": transmitted.mode,
        },
        "stream_accounting": {
            "frame_shape": list(source.shape),
            "bitstream_bytes": len(transmitted.bitstream),
            "wire_request_bytes": len(wire_request),
            "ledger_residual_bytes": chunk.sizes.residual,
            "ledger_reconciled": ledger_reconciled,
            "sizes_is_rate": chunk.sizes.is_rate,
        },
        "timing_seconds": {
            "total_wall": elapsed_total,
            "encoder_seconds": chunk.encoder_seconds,
            "client_seconds": chunk.client_seconds,
            "evaluation_seconds": chunk.evaluation_seconds,
            "codec_encode_seconds": transmitted.encode_seconds,
            "codec_decode_seconds": transmitted.decode_seconds,
        },
        "fidelity": {
            "client_identical_to_delivered": client_delivered_identical,
            "client_frames_shape": list(client_frames.shape),
            "psnr_unaided_db": psnr_unaided,
            "psnr_residual_delivered_db": psnr_delivered,
            "quality_gain_db": psnr_delivered - psnr_unaided,
        },
        "limitations": [
            "Native codec invocation uses ffmpeg CLI subprocess rather than in-process C-API bindings.",
            "Subsampled chroma (yuv420p) introduces high-frequency colour quantization relative to full RGB.",
            "Full-range mapping provides <= 1 grey level precision across [-255, 255], while clipped mode bounds to [-128, 127].",
        ],
        "quality_ladder_sweep_command": (
            "python -m src.runner.run --config config/tier_fast.yaml "
            "--set residual.codec=avc --set residual.block_size=1 --set residual.block_threshold=0.0 "
            "--set residual.background_downscale=1 --sweep residual.rate=18,23,28,33,38"
        ),
    }

    out_path = Path("/tmp/pointstream-worker-a/artifacts/residual_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(smoke_data, indent=2))
    print(f"Artifact successfully saved to {out_path}")
    print(f"Delivered PSNR: {psnr_delivered:.2f} dB (gain: {psnr_delivered - psnr_unaided:.2f} dB)")
    print(f"Bitstream bytes: {len(transmitted.bitstream)}, ledger matches: {ledger_reconciled}")
    print(f"Client identical: {client_delivered_identical}")


if __name__ == "__main__":
    main()
