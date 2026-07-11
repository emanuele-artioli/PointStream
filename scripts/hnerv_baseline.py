"""HNeRV learned-codec baseline: per-video implicit-network overfit + compression sweep.

Answers the reviewer-critical "learned codec baseline" gap (R2, R5,
reports/6_action_matrix.md; reports/9_codec_baselines_report.md's TL;DR:
"One learned codec (HNeRV or DCVC) baseline remains entirely open"). Unlike
`scripts/codec_baseline_sweep.py` (a CLI wrapper around AV1/HEVC), HNeRV
"encodes" a video by training a small implicit network (src/shared/hnerv_arch.py)
to overfit that one clip:

    train (per-video)  ->  serialize decoder+embeddings to disk (bytes)
                        ->  decode = one forward pass through the trained
                            decoder, reloaded from that same file
                        ->  score via src.experiment_evaluation.evaluate_run_summary
                            (identical PSNR/SSIM/VMAF code path as the AV1/HEVC
                            sweep, so numbers land on the same axes)

Usage:
    # Tiny smoke test (mock-first: proves the full loop before a real run):
    python -m scripts.hnerv_baseline --max-frames 4 --height 32 --width 32 \\
        --embed-height 4 --embed-width 4 --strides 2,2,2 --channels 16,8,4 \\
        --epochs 20 --output-root outputs/hnerv_smoke

    # Real run on assets/real_tennis.mp4 (defaults chosen for that clip):
    python -m scripts.hnerv_baseline --epochs 4000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.encoder.video_io import iter_video_frames_ffmpeg, probe_video_metadata  # noqa: E402
from src.experiment_evaluation import evaluate_run_summary  # noqa: E402
from src.shared.hnerv_arch import (  # noqa: E402
    HNeRVConfig,
    HNeRVModel,
    count_decoder_parameters,
    load_hnerv_checkpoint,
    save_hnerv_checkpoint,
)

# Defaults chosen for assets/real_tennis.mp4 (3840x2160, 12fps, 60 frames):
# training/decoding at a reduced resolution keeps per-video overfit training
# tractable; evaluate_run_summary's PSNR/SSIM/VMAF ffmpeg filters auto-scale
# the decoded (lower-res) frames back up to the source resolution before
# scoring (same mechanism the AV1/HEVC sweep would hit if its outputs were a
# different resolution), so the comparison stays apples-to-apples on bytes
# and quality even though HNeRV trains at a smaller frame size.
DEFAULT_HEIGHT = 360
DEFAULT_WIDTH = 640
DEFAULT_EMBED_HEIGHT = 9
DEFAULT_EMBED_WIDTH = 16
DEFAULT_EMBED_CHANNELS = 64
DEFAULT_STRIDES = (5, 4, 2)
DEFAULT_CHANNELS = (128, 64, 32)
DEFAULT_EPOCHS = 4000
DEFAULT_LR = 1e-3


def _find_default_input() -> Path | None:
    candidate = PROJECT_ROOT / "assets" / "real_tennis.mp4"
    return candidate if candidate.is_file() else None


def _load_training_frames(input_path: Path, height: int, width: int, max_frames: int | None) -> torch.Tensor:
    """Decode a video at native resolution, resize each frame, return an RGB float tensor.

    Returns Shape: [NumFrames, 3, Height, Width], values in [0, 1], channel order RGB.
    """
    metadata = probe_video_metadata(input_path)
    frames_bgr = []
    for index, frame in enumerate(iter_video_frames_ffmpeg(input_path, width=metadata.width, height=metadata.height)):
        if max_frames is not None and index >= max_frames:
            break
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames_bgr.append(resized)
    if not frames_bgr:
        raise ValueError(f"No frames decoded from {input_path}")

    stacked = np.stack(frames_bgr, axis=0)  # Shape: [NumFrames, Height, Width, 3] BGR uint8
    stacked_rgb = stacked[..., ::-1]  # BGR -> RGB
    tensor = torch.from_numpy(np.ascontiguousarray(stacked_rgb)).permute(0, 3, 1, 2).contiguous()
    tensor = tensor.to(torch.float32) / 255.0  # Shape: [NumFrames, 3, Height, Width]
    return tensor


@dataclass
class TrainResult:
    decoder_state_embeddings: torch.Tensor  # Shape: [NumFrames, EmbedChannels, EmbedHeight, EmbedWidth]
    history: list[dict[str, Any]]
    train_seconds: float


def train_hnerv(
    frames: torch.Tensor,
    config: HNeRVConfig,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    log_every: int = 100,
    checkpoint_every: int | None = None,
    checkpoint_path: Path | None = None,
    seed: int = 0,
    progress_path: Path | None = None,
) -> tuple[HNeRVModel, TrainResult]:
    """Overfit an HNeRVModel to `frames` via full-batch gradient descent.

    Checkpoints the decoder+embeddings to `checkpoint_path` every
    `checkpoint_every` epochs (if both given) so a long real run leaves
    recoverable partial progress if interrupted.
    """
    torch.manual_seed(seed)
    model = HNeRVModel(config).to(device)
    frames_dev = frames.to(device)  # Shape: [NumFrames, 3, Height, Width]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        reconstruction, _ = model(frames_dev)
        loss = F.l1_loss(reconstruction, frames_dev)
        loss.backward()
        optimizer.step()
        scheduler.step()

        is_last = epoch == epochs - 1
        if epoch % log_every == 0 or is_last:
            with torch.no_grad():
                mse = F.mse_loss(reconstruction, frames_dev).item()
                psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))
            entry = {"epoch": epoch, "loss": float(loss.item()), "mse": mse, "psnr": psnr}
            history.append(entry)
            print(f"[hnerv] epoch {epoch}/{epochs} loss={loss.item():.5f} psnr={psnr:.2f}dB", flush=True)
            if progress_path is not None:
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")

        if checkpoint_every and checkpoint_path is not None and (epoch % checkpoint_every == 0 or is_last):
            model.eval()
            with torch.no_grad():
                _, embedding = model(frames_dev)
            save_hnerv_checkpoint(checkpoint_path, model.decoder, embedding.cpu())

    train_seconds = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        _, final_embedding = model(frames_dev)
    return model, TrainResult(
        decoder_state_embeddings=final_embedding.cpu(), history=history, train_seconds=train_seconds
    )


def decode_hnerv(checkpoint_path: Path, device: torch.device) -> torch.Tensor:
    """Reload the saved checkpoint and run the decoder-only forward pass ("decoding")."""
    decoder, embeddings = load_hnerv_checkpoint(checkpoint_path, device=device)
    decoder = decoder.to(device).eval()
    with torch.no_grad():
        reconstruction = decoder(embeddings.to(device))  # Shape: [NumFrames, 3, Height, Width]
    return reconstruction.cpu()


def write_frames_png(frames_rgb01: torch.Tensor, output_dir: Path) -> None:
    """Write an RGB float [0,1] tensor as lossless frame_%06d.png files (BGR for cv2)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_uint8 = (frames_rgb01.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
    for index in range(frames_uint8.shape[0]):
        frame_rgb = frames_uint8[index].transpose(1, 2, 0)  # HWC RGB
        frame_bgr = frame_rgb[..., ::-1]
        cv2.imwrite(str(output_dir / f"frame_{index:06d}.png"), frame_bgr)


def _fmt_ratio(output_bytes: int, source_bytes: int) -> str:
    return f"{output_bytes / source_bytes:.4f}" if source_bytes else "—"


def _fmt_quality(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "null"


def build_report(
    *,
    input_path: Path,
    config: HNeRVConfig,
    epochs: int,
    decoder_params: int,
    output_bytes: int,
    source_bytes: int,
    evaluation: dict[str, Any],
    train_seconds: float,
    decode_seconds: float,
) -> tuple[str, dict[str, Any]]:
    lines = [
        "# HNeRV baseline (learned implicit-representation codec, single-video overfit)",
        "",
        f"Source: `{input_path}`",
        "",
        f"Trained/decoded resolution: {config.width}x{config.height} "
        f"(embedding grid {config.embed_width}x{config.embed_height}x{config.embed_channels}, "
        f"decoder strides {list(config.strides)}, decoder params {decoder_params:,}, epochs {epochs}).",
        "evaluate_run_summary auto-scales the decoded frames to the source resolution "
        "(ffmpeg scale filter) before scoring, the same mechanism the AV1/HEVC sweep uses "
        "for any resolution mismatch, so bytes/psnr/ssim/vmaf below are directly comparable "
        "to `outputs/codec_baselines/<ts>/report.md`.",
        "",
        "| codec | bytes | ratio-to-source | psnr | ssim | vmaf | train (s) | decode (s) |",
        "|---|---|---|---|---|---|---|---|",
        f"| HNeRV | {output_bytes:,} | {_fmt_ratio(output_bytes, source_bytes)} "
        f"| {_fmt_quality(evaluation.get('psnr_mean'))} | {_fmt_quality(evaluation.get('ssim_mean'))} "
        f"| {_fmt_quality(evaluation.get('vmaf_mean'))} | {train_seconds:.0f} | {decode_seconds:.1f} |",
    ]
    markdown = "\n".join(lines) + "\n"
    payload: dict[str, Any] = {
        "source": str(input_path),
        "config": config.as_dict(),
        "epochs": epochs,
        "decoder_params": decoder_params,
        "output_bytes": output_bytes,
        "source_bytes": source_bytes,
        "evaluation": evaluation,
        "train_seconds": train_seconds,
        "decode_seconds": decode_seconds,
    }
    return markdown, payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="Input video. Defaults to assets/real_tennis.mp4.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit to first N frames (smoke testing).")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--embed-height", type=int, default=DEFAULT_EMBED_HEIGHT)
    parser.add_argument("--embed-width", type=int, default=DEFAULT_EMBED_WIDTH)
    parser.add_argument("--embed-channels", type=int, default=DEFAULT_EMBED_CHANNELS)
    parser.add_argument(
        "--strides", default=",".join(str(s) for s in DEFAULT_STRIDES), help="Comma-separated decoder upscale factors."
    )
    parser.add_argument(
        "--channels", default=",".join(str(c) for c in DEFAULT_CHANNELS), help="Comma-separated decoder stage channels."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=250, help="0 disables periodic checkpointing.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Defaults to cuda:0 if available, else cpu.")
    parser.add_argument("--output-root", default="outputs/hnerv_baseline")
    parser.add_argument(
        "--metrics", default="psnr,ssim,vmaf", help="Comma-separated evaluate_run_summary metrics (or 'none')."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    input_video = args.input or (_find_default_input() and str(_find_default_input()))
    if input_video is None:
        raise FileNotFoundError("No --input given and assets/real_tennis.mp4 not found.")
    input_path = Path(input_video).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    strides = tuple(int(s.strip()) for s in str(args.strides).split(",") if s.strip())
    channels = tuple(int(c.strip()) for c in str(args.channels).split(",") if c.strip())
    config = HNeRVConfig(
        height=args.height,
        width=args.width,
        embed_height=args.embed_height,
        embed_width=args.embed_width,
        embed_channels=args.embed_channels,
        strides=strides,
        channels=channels,
    )

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(str(args.output_root)).expanduser().resolve() / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Config: {config}")
    print(f"Device: {device}")
    print(f"Output root: {output_root}")

    frames = _load_training_frames(input_path, config.height, config.width, args.max_frames)
    print(f"Loaded {frames.shape[0]} frames at {config.width}x{config.height}", flush=True)

    checkpoint_path = output_root / "hnerv_checkpoint.pt.gz"
    progress_path = output_root / "progress.jsonl"

    model, train_result = train_hnerv(
        frames,
        config,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        log_every=args.log_every,
        checkpoint_every=(args.checkpoint_every or None),
        checkpoint_path=checkpoint_path,
        seed=args.seed,
        progress_path=progress_path,
    )
    decoder_params = count_decoder_parameters(model.decoder)

    output_bytes = save_hnerv_checkpoint(checkpoint_path, model.decoder, train_result.decoder_state_embeddings)
    print(f"Saved checkpoint: {checkpoint_path} ({output_bytes:,} bytes)", flush=True)

    decode_started = time.perf_counter()
    reconstruction = decode_hnerv(checkpoint_path, device)
    decode_seconds = time.perf_counter() - decode_started

    decoded_dir = output_root / "decoded_frames"
    write_frames_png(reconstruction, decoded_dir)
    print(f"Wrote decoded frames to {decoded_dir}", flush=True)

    source_bytes = input_path.stat().st_size
    summary = {
        "source_uri": str(input_path),
        "decoded_uri": str(decoded_dir),
        "evaluation": {"sizes_bytes": {}},
    }
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    evaluation = evaluate_run_summary(summary, experiment_dir=output_root, max_frames=args.max_frames, metrics=metrics)

    markdown, payload = build_report(
        input_path=input_path,
        config=config,
        epochs=args.epochs,
        decoder_params=decoder_params,
        output_bytes=output_bytes,
        source_bytes=source_bytes,
        evaluation=evaluation,
        train_seconds=train_result.train_seconds,
        decode_seconds=decode_seconds,
    )
    print()
    print(markdown)

    (output_root / "report.md").write_text(markdown, encoding="utf-8")
    (output_root / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_root / "history.json").write_text(json.dumps(train_result.history, indent=2) + "\n", encoding="utf-8")

    print(f"Report written to {output_root / 'report.md'} and {output_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
