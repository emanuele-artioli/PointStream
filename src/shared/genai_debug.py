"""Debug instrumentation for GenAI compositor divergence analysis."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def export_compositor_artifacts(
    debug_dir: str | Path | None,
    stage: str,
    frame_idx: int,
    actor_id: str,
    artifacts: dict[str, np.ndarray],
) -> None:
    """Export compositor artifacts provided by strategy and compositor."""
    if not debug_dir or not Path(debug_dir).exists():
        return

    output_dir = Path(debug_dir) / stage / f"frame_{frame_idx:04d}" / f"actor_{actor_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, img_np in artifacts.items():
        if img_np is None:
            continue
        cv2.imwrite(str(output_dir / filename), img_np)


def debug_export_frame_pair(
    frame_idx: int,
    encoder_tensor: torch.Tensor,
    decoder_tensor: torch.Tensor,
    debug_dir: str | Path | None = None,
) -> None:
    """Export encoder/decoder frame pairs for direct comparison."""
    if not debug_dir or not Path(debug_dir).exists():
        return

    comparison_dir = Path(debug_dir) / "comparison" / f"frame_{frame_idx:04d}"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Export encoder version
    enc_np = encoder_tensor.cpu().numpy()
    if len(enc_np.shape) == 3 and enc_np.shape[0] == 3:
        enc_np = np.transpose(enc_np, (1, 2, 0))
    cv2.imwrite(str(comparison_dir / "00_encoder.png"), enc_np)

    # Export decoder version
    dec_np = decoder_tensor.cpu().numpy()
    if len(dec_np.shape) == 3 and dec_np.shape[0] == 3:
        dec_np = np.transpose(dec_np, (1, 2, 0))
    cv2.imwrite(str(comparison_dir / "01_decoder.png"), dec_np)

    # Export difference
    if encoder_tensor.shape == decoder_tensor.shape:
        diff = torch.abs(encoder_tensor.float() - decoder_tensor.float())
        diff_np = (diff.cpu().numpy() / diff.max().item() * 255.0).astype(np.uint8) if diff.max().item() > 0 else np.zeros_like(enc_np)
        if len(diff_np.shape) == 3 and diff_np.shape[0] == 3:
            diff_np = np.transpose(diff_np, (1, 2, 0))
        cv2.imwrite(str(comparison_dir / "02_difference.png"), diff_np)

        # Compute and log difference statistics
        diff_cpu = diff.cpu().numpy()
        if len(diff_cpu.shape) == 3:
            diff_cpu = diff_cpu.mean(axis=0)
        diff_mean = float(diff_cpu.mean())
        diff_max = float(diff_cpu.max())
        diff_pct_nonzero = float(100.0 * np.count_nonzero(diff_cpu > 0) / diff_cpu.size)

        with open(comparison_dir / "stats.txt", "w") as f:
            f.write(f"mean_diff={diff_mean:.2f}\n")
            f.write(f"max_diff={diff_max:.2f}\n")
            f.write(f"pct_nonzero={diff_pct_nonzero:.1f}%\n")

        logger.info(f"Frame {frame_idx}: encoder vs decoder diff = {diff_mean:.2f} (max {diff_max:.2f})")


def create_debug_report(debug_root: str | Path) -> str:
    """Generate a comparison report from debug exports."""
    root = Path(debug_root)
    if not root.exists():
        return "Debug directory does not exist."

    report_lines = ["GenAI Compositor Divergence Report\n" + "=" * 60]

    # Check encoder frames
    encoder_dir = root / "encoder"
    decoder_dir = root / "decoder"
    comparison_dir = root / "comparison"

    if encoder_dir.exists():
        frame_dirs = sorted([d for d in encoder_dir.iterdir() if d.is_dir()])
        report_lines.append(f"\nEncoder frames: {len(frame_dirs)}")
        for fd in frame_dirs[:5]:  # Show first 5
            actor_dirs = sorted([d for d in fd.iterdir() if d.is_dir()])
            report_lines.append(f"  {fd.name}: {len(actor_dirs)} actors")

    if decoder_dir.exists():
        frame_dirs = sorted([d for d in decoder_dir.iterdir() if d.is_dir()])
        report_lines.append(f"\nDecoder frames: {len(frame_dirs)}")
        for fd in frame_dirs[:5]:
            actor_dirs = sorted([d for d in fd.iterdir() if d.is_dir()])
            report_lines.append(f"  {fd.name}: {len(actor_dirs)} actors")

    if comparison_dir.exists():
        frame_dirs = sorted([d for d in comparison_dir.iterdir() if d.is_dir()])
        report_lines.append(f"\nComparison frames: {len(frame_dirs)}")
        
        total_mean_diff = 0.0
        total_max_diff = 0.0
        count = 0
        for fd in frame_dirs:
            stats_file = fd / "stats.txt"
            if stats_file.exists():
                with open(stats_file) as f:
                    lines = f.read().strip().split("\n")
                    for line in lines:
                        if "mean_diff" in line:
                            mean_val = float(line.split("=")[1])
                            total_mean_diff += mean_val
                        if "max_diff" in line:
                            max_val = float(line.split("=")[1])
                            total_max_diff = max(total_max_diff, max_val)
                count += 1

        if count > 0:
            report_lines.append(f"\nAverage mean_diff: {total_mean_diff / count:.2f}")
            report_lines.append(f"Maximum max_diff: {total_max_diff:.2f}")

    return "\n".join(report_lines)
